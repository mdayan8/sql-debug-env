"""
OpenEnv GRPO training (production-oriented, simple stack).

Produces real training artifacts (trainer log_history, metrics JSON, reward plots) and
optional Hub push of LoRA weights. Every execution reward calls your live Space (or
local server) at OPENENV_BASE_URL — not a mock.

Environment (control cost vs quality on HF Jobs / local GPU):
  OPENENV_BASE_URL          — OpenEnv HTTP root (default: Space URL from openenv.yaml)
  OPENENV_TASK_IDS          — Comma list; if unset, uses GET /tasks from the server
  ROWS_PER_TASK             — GRPO rows per task_id (default: 48)
  OPENENV_REQUEST_TIMEOUT_SEC — HTTP timeout for reset/step (default: 120)
  TRAIN_MAX_STEPS           — GRPO steps (default 200)
  TRL_REPORT_TO             — none | wandb | tensorboard (auto: wandb if key else none)
  BOOTSTRAP_*_VERSION       — pin transformers / accelerate / trl (defaults satisfy trl>=4.50)
  Artifacts: artifacts/train_log_history.jsonl, metrics, plots
  HF_HUB_REPO_ID            — push target (default md896/sota-sql-agent-7b)
  SKIP_HUB_PUSH=1           — do not push after train
  SKIP_PRETRAIN_EVAL=1    — skip baseline/per-task/hard eval before GRPO (faster to step logs; weaker metrics)
  ATTN_IMPLEMENTATION       — default sdpa on CUDA (else eager); set eager if you hit attention errors
  HF_TOKEN / HUGGING_FACE_HUB_TOKEN — Hub auth for push

Designed for Hugging Face Jobs / Spaces where:
- system Python may be externally managed (PEP-668) → uses --break-system-packages
- preinstalled CUDA/PyTorch stacks can conflict with optional vision packages

Key stability choices:
- Avoid importing torchvision in text-only runs (it can break when torch/torchvision
  versions are mismatched by dependency resolution).
- Produce plots and metrics from the *actual* GRPO run (no hard-coded scores).
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

def _run(cmd: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check)


def _pip(args: List[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return _run([sys.executable, "-m", "pip", *args], check=check)


def bootstrap_deps() -> None:
    """
    Best-effort dependency bootstrap for ephemeral HF containers.

    Set SKIP_BOOTSTRAP=1 to disable.
    Pins: BOOTSTRAP_TRANSFORMERS_VERSION, BOOTSTRAP_ACCELERATE_VERSION, BOOTSTRAP_TRL_VERSION.
    """
    if os.environ.get("SKIP_BOOTSTRAP") == "1":
        return

    # Ensure text-only transformers runs never hard-import torchvision even if it
    # is present in the base image.
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    # Ubuntu 24.04+ images may mark system Python as "externally managed"
    # (PEP-668). Prefer an explicit opt-out for all pip ops in ephemeral jobs.
    os.environ.setdefault("PIP_BREAK_SYSTEM_PACKAGES", "1")

    print("Bootstrapping dependencies...")

    # Text-only run: torchvision/torchaudio are not required and are a common source
    # of crashes when torch versions shift in container images.
    _pip(["uninstall", "--break-system-packages", "-y", "torchvision", "torchaudio"], check=False)
    _pip(["uninstall", "-y", "torchao"], check=False)

    # trl 0.18.x needs transformers>=4.50. datasets 4.x pulls huggingface-hub 1.x which breaks 4.5x.
    _tf = os.environ.get("BOOTSTRAP_TRANSFORMERS_VERSION", "4.51.3")
    _acc = os.environ.get("BOOTSTRAP_ACCELERATE_VERSION", "0.34.2")
    _trl = os.environ.get("BOOTSTRAP_TRL_VERSION", "0.18.2")
    _pip(
        [
            "install",
            "--break-system-packages",
            "httpx>=0.27.0",
            "datasets>=3.2.0,<4.0.0",
            "matplotlib",
            "tensorboard",
            f"transformers=={_tf}",
            f"accelerate=={_acc}",
            f"trl=={_trl}",
        ]
    )

    if os.environ.get("WANDB_API_KEY"):
        _pip(["install", "--break-system-packages", "wandb"], check=False)

    _pip(
        [
            "install",
            "--break-system-packages",
            "--force-reinstall",
            "--no-deps",
            f"transformers=={_tf}",
            f"accelerate=={_acc}",
        ]
    )
    _pip(["install", "--break-system-packages", "--no-deps", f"trl=={_trl}"])

    _pip(["uninstall", "-y", "torchao"], check=False)
    _pip(["uninstall", "--break-system-packages", "-y", "torchvision", "torchaudio"], check=False)

    # Keep bootstrap import-free; training imports happen below.


bootstrap_deps()

import httpx
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# --- 1. CONFIGURATION (env-first; defaults match openenv.yaml) ---
_DEFAULT_OPENENV_BASE = "https://md896-sql-debug-env.hf.space"
BYPASS_HEADERS: Dict[str, str] = {}

MODEL_NAME = os.environ.get("TRAIN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")


def get_bridge_url() -> str:
    return os.environ.get("OPENENV_BASE_URL", _DEFAULT_OPENENV_BASE).rstrip("/")


def get_request_timeout() -> float:
    return float(os.environ.get("OPENENV_REQUEST_TIMEOUT_SEC", "120"))


def _fetch_task_ids(client: httpx.Client) -> List[str]:
    raw = os.environ.get("OPENENV_TASK_IDS", "").strip()
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    r = client.get("/tasks", timeout=get_request_timeout())
    r.raise_for_status()
    body = r.json()
    tasks = body.get("tasks") or []
    ids = [t["task_id"] for t in tasks if t.get("task_id")]
    if not ids:
        raise RuntimeError("/tasks returned no task_id entries")
    return ids


def make_real_dataset() -> Dataset:
    """Plain prompts + live /tasks (same spirit as colab_real_world.py, HF Space instead of loca.lt)."""
    bridge = get_bridge_url()
    timeout = get_request_timeout()
    rows_per_task = max(1, int(os.environ.get("ROWS_PER_TASK", "48")))
    marker = os.environ.get("COMPLETION_SQL_MARKER", "Fixed SQL:")

    print(f"Connecting to OpenEnv at {bridge} (timeout={timeout}s)...")
    rows: List[Dict[str, Any]] = []

    with httpx.Client(base_url=bridge, headers=BYPASS_HEADERS, timeout=timeout) as client:
        h = client.get("/health", timeout=min(30.0, timeout))
        h.raise_for_status()
        print(f"OpenEnv health: {h.json()}")

        task_ids = _fetch_task_ids(client)
        print(f"Training task_ids ({len(task_ids)}): {task_ids}")

        for t_id in task_ids:
            resp = client.post("/reset", json={"task_id": t_id})
            resp.raise_for_status()
            obs = resp.json()["observation"]

            prompt = (
                "Fix the following SQL query and provide only the fixed SQL.\n"
                f"Task: {obs['task_description']}\n"
                f"Broken Query: {obs['original_query']}\n"
                f"{marker}"
            )
            for _ in range(rows_per_task):
                rows.append({"prompt": prompt, "task_id": t_id})

    if not rows:
        raise RuntimeError("Failed to build dataset (no rows).")
    print(f"Dataset: {len(rows)} prompts ({rows_per_task} per task).")
    return Dataset.from_list(rows)


def make_task_dataset(task_id: str, rows_per_task: int) -> Dataset:
    bridge = get_bridge_url()
    timeout = get_request_timeout()
    marker = os.environ.get("COMPLETION_SQL_MARKER", "Fixed SQL:")
    with httpx.Client(base_url=bridge, headers=BYPASS_HEADERS, timeout=timeout) as client:
        resp = client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        obs = resp.json()["observation"]
    prompt = (
        "Fix the following SQL query and provide only the fixed SQL.\n"
        f"Task: {obs['task_description']}\n"
        f"Broken Query: {obs['original_query']}\n"
        f"{marker}"
    )
    rows = [{"prompt": prompt, "task_id": task_id} for _ in range(max(1, rows_per_task))]
    return Dataset.from_list(rows)


# --- 3. One live OpenEnv reward (colab_real_world style) ---


def openenv_sql_reward_func(completions, task_id, **kwargs):
    """Score completions by executing extracted SQL against the real OpenEnv HTTP API."""
    base = get_bridge_url()
    timeout = get_request_timeout()
    marker = os.environ.get("COMPLETION_SQL_MARKER", "Fixed SQL:")
    rewards: List[float] = []
    with httpx.Client(base_url=base, headers=BYPASS_HEADERS, timeout=timeout) as client:
        for completion, t_id in zip(completions, task_id):
            if marker in completion:
                sql = completion.split(marker, 1)[-1].strip()
            else:
                sql = completion.strip()
            if not sql:
                rewards.append(0.0)
                continue
            hdr = {"X-Session-Id": str(uuid.uuid4())}
            try:
                client.post("/reset", json={"task_id": t_id}, headers=hdr).raise_for_status()
                resp = client.post(
                    "/step",
                    json={"action": {"action_type": "submit_query", "query": sql}},
                    headers=hdr,
                )
                resp.raise_for_status()
                r = float(resp.json().get("reward", 0.0))
            except Exception:
                r = 0.0
            r += random.uniform(-1e-6, 1e-6)
            rewards.append(r)
    return rewards


def eval_model_reward(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    *,
    max_items: int,
) -> float:
    subset = dataset.select(range(min(max_items, len(dataset))))
    prompts = subset["prompt"]
    task_ids = subset["task_id"]
    completions: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.7")),
                top_p=float(os.environ.get("EVAL_TOP_P", "0.9")),
                renormalize_logits=True,
                remove_invalid_values=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions.append(tokenizer.decode(out[0], skip_special_tokens=True))
    rewards = openenv_sql_reward_func(completions, task_ids)
    return float(sum(rewards) / max(len(rewards), 1))


def eval_model_reward_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    *,
    max_items: int,
) -> List[float]:
    subset = dataset.select(range(min(max_items, len(dataset))))
    prompts = subset["prompt"]
    task_ids = subset["task_id"]
    completions: List[str] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=float(os.environ.get("EVAL_TEMPERATURE", "0.7")),
                top_p=float(os.environ.get("EVAL_TOP_P", "0.9")),
                renormalize_logits=True,
                remove_invalid_values=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        completions.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return openenv_sql_reward_func(completions, task_ids)


# --- 3b. ARTIFACTS / PLOTS (REAL, FROM LOGS) ---

@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def logs_jsonl(self) -> Path:
        return self.root / "train_log_history.jsonl"

    @property
    def metrics_json(self) -> Path:
        return self.root / "train_metrics.json"

    @property
    def reward_curve_png(self) -> Path:
        return self.root / "reward_curve.png"

    @property
    def reward_curve_moving_avg_png(self) -> Path:
        return self.root / "reward_curve_moving_avg.png"

    @property
    def training_diagnostics_png(self) -> Path:
        return self.root / "training_diagnostics.png"

    @property
    def task_delta_png(self) -> Path:
        return self.root / "task_delta.png"

    @property
    def train_series_json(self) -> Path:
        return self.root / "train_series.json"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_log_history(log_history: List[Dict[str, Any]], paths: ArtifactPaths) -> None:
    _ensure_dir(paths.root)
    with paths.logs_jsonl.open("w", encoding="utf-8") as f:
        for row in log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_reward_series(log_history: List[Dict[str, Any]]) -> List[tuple[float, float]]:
    """
    Returns [(step, reward_like_value)] extracted from trainer log_history.
    TRL log keys vary; this is resilient and will pick the most relevant.
    """
    candidates = [
        "reward",
        "rewards/mean",
        "rewards",
        "train/reward",
        "train/rewards",
        "objective/mean_reward",
        "mean_reward",
    ]

    series: List[tuple[float, float]] = []
    for row in log_history:
        step = row.get("step") or row.get("global_step") or row.get("epoch")
        if step is None:
            continue
        value = None
        for key in candidates:
            if key in row and isinstance(row[key], (int, float)):
                value = float(row[key])
                break
        if value is None:
            # fallback: pick any numeric key containing "reward"
            for k, v in row.items():
                if "reward" in str(k).lower() and isinstance(v, (int, float)):
                    value = float(v)
                    break
        if value is None:
            continue
        series.append((float(step), value))

    # de-dup by step while preserving order
    seen = set()
    deduped: List[tuple[float, float]] = []
    for s, v in series:
        if s in seen:
            continue
        seen.add(s)
        deduped.append((s, v))
    return deduped


def write_metrics(log_history: List[Dict[str, Any]], reward_series: List[tuple[float, float]], paths: ArtifactPaths) -> None:
    metrics = {
        "generated_at_epoch_s": time.time(),
        "log_rows": len(log_history),
        "reward_points": len(reward_series),
        "reward_first": reward_series[0][1] if reward_series else None,
        "reward_last": reward_series[-1][1] if reward_series else None,
        "reward_max": max((v for _, v in reward_series), default=None),
    }
    _ensure_dir(paths.root)
    paths.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def plot_reward_curve(reward_series: List[tuple[float, float]], paths: ArtifactPaths) -> None:
    if not reward_series:
        print("⚠️ No reward series found in log history; skipping plot.")
        return
    import matplotlib.pyplot as plt

    xs = [s for s, _ in reward_series]
    ys = [v for _, v in reward_series]
    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys, linewidth=2)
    plt.title("GRPO Reward Over Time (from run logs)")
    plt.xlabel("step")
    plt.ylabel("reward (extracted)")
    plt.grid(True, linestyle="--", alpha=0.4)
    _ensure_dir(paths.root)
    plt.tight_layout()
    plt.savefig(paths.reward_curve_png, dpi=200)
    print(f"Saved {paths.reward_curve_png}")


def extract_numeric_series(log_history: List[Dict[str, Any]], key_contains: str) -> List[tuple[float, float]]:
    series: List[tuple[float, float]] = []
    needle = key_contains.lower()
    for row in log_history:
        step = row.get("step") or row.get("global_step") or row.get("epoch")
        if step is None:
            continue
        for k, v in row.items():
            if needle in str(k).lower() and isinstance(v, (int, float)):
                series.append((float(step), float(v)))
                break
    # de-dup by step preserving order
    seen = set()
    out: List[tuple[float, float]] = []
    for s, v in series:
        if s in seen:
            continue
        seen.add(s)
        out.append((s, v))
    return out


def _moving_average(values: List[float], window: int) -> List[float]:
    if not values:
        return []
    w = max(1, window)
    out: List[float] = []
    run = 0.0
    for i, val in enumerate(values):
        run += val
        if i >= w:
            run -= values[i - w]
        out.append(run / min(i + 1, w))
    return out


def plot_reward_moving_average(reward_series: List[tuple[float, float]], paths: ArtifactPaths) -> None:
    if not reward_series:
        return
    import matplotlib.pyplot as plt

    xs = [s for s, _ in reward_series]
    ys = [v for _, v in reward_series]
    ys_ma = _moving_average(ys, window=max(5, len(ys) // 20 or 5))
    plt.figure(figsize=(9, 4))
    plt.plot(xs, ys, alpha=0.25, linewidth=1, label="raw reward")
    plt.plot(xs, ys_ma, linewidth=2, label="moving average")
    plt.title("Reward Curve (raw + moving average)")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.reward_curve_moving_avg_png, dpi=200)
    print(f"Saved {paths.reward_curve_moving_avg_png}")


def plot_training_diagnostics(
    reward_series: List[tuple[float, float]],
    loss_series: List[tuple[float, float]],
    paths: ArtifactPaths,
) -> None:
    if not reward_series and not loss_series:
        return
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    if reward_series:
        rx = [s for s, _ in reward_series]
        ry = [v for _, v in reward_series]
        ax1.plot(rx, ry, color="#2563eb", label="reward")
        ax1.set_ylabel("reward", color="#2563eb")
    ax1.set_xlabel("step")
    ax1.grid(True, linestyle="--", alpha=0.3)

    if loss_series:
        ax2 = ax1.twinx()
        lx = [s for s, _ in loss_series]
        ly = [v for _, v in loss_series]
        ax2.plot(lx, ly, color="#ef4444", label="loss")
        ax2.set_ylabel("loss", color="#ef4444")

    fig.suptitle("Training Diagnostics: Reward and Loss")
    fig.tight_layout()
    fig.savefig(paths.training_diagnostics_png, dpi=200)
    print(f"Saved {paths.training_diagnostics_png}")


def plot_performance_comparison(
    before_by_task: Dict[str, float],
    after_by_task: Dict[str, float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    task_ids = sorted(set(before_by_task) | set(after_by_task))
    if not task_ids:
        return

    base_vals = [before_by_task.get(t, 0.0) for t in task_ids]
    tuned_vals = [after_by_task.get(t, 0.0) for t in task_ids]
    overall_base = sum(base_vals) / max(1, len(base_vals))
    overall_tuned = sum(tuned_vals) / max(1, len(tuned_vals))

    labels = task_ids + ["overall"]
    base_plot = base_vals + [overall_base]
    tuned_plot = tuned_vals + [overall_tuned]

    x = list(range(len(labels)))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar([xi - width / 2 for xi in x], base_plot, width=width, label="base model", color="#94a3b8")
    plt.bar([xi + width / 2 for xi in x], tuned_plot, width=width, label="trained (GRPO)", color="#2563eb")
    plt.ylim(0.0, 1.0)
    plt.xticks(x, labels, rotation=15, ha="right")
    plt.ylabel("avg reward")
    plt.title("Performance Comparison by Task (OpenEnv reward)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_task_delta(before_by_task: Dict[str, float], after_by_task: Dict[str, float], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    task_ids = sorted(set(before_by_task) | set(after_by_task))
    if not task_ids:
        return
    deltas = [after_by_task.get(t, 0.0) - before_by_task.get(t, 0.0) for t in task_ids]
    colors = ["#16a34a" if d >= 0 else "#dc2626" for d in deltas]
    plt.figure(figsize=(9, 4.5))
    plt.bar(task_ids, deltas, color=colors)
    plt.axhline(0.0, color="#111827", linewidth=1)
    plt.xticks(rotation=15, ha="right")
    plt.ylabel("reward delta (post - base)")
    plt.title("Per-task Improvement Delta")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def plot_reward_distribution_shift(before: List[float], after: List[float], out_path: Path) -> None:
    if not before or not after:
        return
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    bins = 20
    plt.hist(before, bins=bins, alpha=0.5, label="START", color="#f87171")
    plt.hist(after, bins=bins, alpha=0.5, label="END", color="#22c55e")
    plt.xlim(0.0, 1.0)
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.title("Reward Distribution Shift")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def _resolve_report_to() -> str:
    raw = os.environ.get("TRL_REPORT_TO", "").strip().lower()
    if raw in ("", "auto"):
        return "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if raw in ("false", "no", "off", "none"):
        return "none"
    return raw


# --- 4. Simple GRPO training loop (live OpenEnv rewards) ---
def run_sota_train():
    # Helps with CUDA memory fragmentation in long generation/training loops.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    max_steps = int(os.environ.get("TRAIN_MAX_STEPS", "240"))
    out_dir = os.environ.get("OUTPUT_DIR", "./sota_results")

    print(f"Starting GRPO on {MODEL_NAME}...")
    print(
        f"OpenEnv={get_bridge_url()} | max_steps={max_steps} | "
        f"rows_per_task={os.environ.get('ROWS_PER_TASK', '48')} | "
        f"report_to={_resolve_report_to()}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    use_cuda = torch.cuda.is_available()
    # L4/A10/A100 are typically more numerically stable with bf16 than fp16 for RL-style sampling.
    torch_dtype = torch.bfloat16 if use_cuda else torch.float32
    # Default sdpa on CUDA: avoids Qwen sliding-window + "eager" mismatch warning; override with ATTN_IMPLEMENTATION=eager if needed.
    _attn = os.environ.get("ATTN_IMPLEMENTATION", "sdpa" if use_cuda else "eager")
    print(f"Loading model weights (attn={_attn}) — Hub download only on cold cache / fresh job disk...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map="auto",
        attn_implementation=_attn,
    )
    print("=== Model load finished (checkpoint shards in VRAM). Not training yet. ===", flush=True)
    # Runtime generation safety defaults (used by both eval and GRPO generate path).
    model.generation_config.remove_invalid_values = True
    model.generation_config.renormalize_logits = True
    model.generation_config.top_p = float(os.environ.get("GRPO_TOP_P", "0.9"))
    model.generation_config.temperature = float(os.environ.get("GRPO_TEMPERATURE", "0.7"))

    train_dataset = make_real_dataset()

    eval_samples = max(4, int(os.environ.get("TASK_EVAL_SAMPLES", "12")))
    skip_pre = os.environ.get("SKIP_PRETRAIN_EVAL", "").strip().lower() in ("1", "true", "yes")
    if skip_pre:
        print(
            "SKIP_PRETRAIN_EVAL=1 — skipping baseline / per-task / hard pre-eval (faster to GRPO steps; "
            "artifacts will miss before_* metrics).",
            flush=True,
        )
        baseline_avg_reward = None  # type: ignore[assignment]
        eval_task_ids = []
        before_by_task = {}
        before_samples_all = []
        hard_eval_n = int(os.environ.get("HARD_EVAL_SAMPLES", "16"))
        hard_dataset = make_task_dataset("hard_finance_explosion", rows_per_task=hard_eval_n)
        base_hard_reward = None  # type: ignore[assignment]
    else:
        print(
            "=== Pre-training evaluation (HTTP + generate per sample; can take many minutes). "
            "GRPO step logs start only after this block. ===",
            flush=True,
        )
        print("Quick baseline eval (pre-train)...", flush=True)
        baseline_avg_reward = eval_model_reward(model, tokenizer, train_dataset, max_items=8)

        with httpx.Client(base_url=get_bridge_url(), headers=BYPASS_HEADERS, timeout=get_request_timeout()) as client:
            eval_task_ids = _fetch_task_ids(client)
        print(
            f"Per-task pre-eval: {len(eval_task_ids)} tasks × up to {eval_samples} samples each "
            f"(sequential; watch for slow OpenEnv).",
            flush=True,
        )
        before_by_task: Dict[str, float] = {}
        before_samples_all: List[float] = []
        for t_id in eval_task_ids:
            print(f"  pre-eval task_id={t_id!r} ...", flush=True)
            ds = make_task_dataset(t_id, rows_per_task=eval_samples)
            samples = eval_model_reward_samples(model, tokenizer, ds, max_items=eval_samples)
            before_samples_all.extend(samples)
            before_by_task[t_id] = float(sum(samples) / max(1, len(samples)))

        hard_eval_n = int(os.environ.get("HARD_EVAL_SAMPLES", "16"))
        hard_dataset = make_task_dataset("hard_finance_explosion", rows_per_task=hard_eval_n)
        print(f"Hard-set pre-eval ({hard_eval_n} samples)...", flush=True)
        base_hard_reward = eval_model_reward(model, tokenizer, hard_dataset, max_items=hard_eval_n)
        print("=== Pre-training evaluation done. Building GRPOTrainer next. ===", flush=True)

    report_to = _resolve_report_to()
    tb_dir = Path(out_dir) / "tensorboard"
    if report_to == "tensorboard":
        _ensure_dir(tb_dir)

    per_device_bs = int(os.environ.get("PER_DEVICE_TRAIN_BS", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "2"))
    requested_num_gen = int(os.environ.get("GRPO_NUM_GENERATIONS", "8"))
    effective_bs = max(1, per_device_bs * grad_accum)
    if effective_bs % requested_num_gen != 0:
        valid = [d for d in range(2, effective_bs + 1) if effective_bs % d == 0]
        num_gen = valid[-1] if valid else 2
        print(
            f"Adjusting GRPO_NUM_GENERATIONS from {requested_num_gen} to {num_gen} "
            f"for effective batch size {effective_bs}."
        )
    else:
        num_gen = requested_num_gen

    _cfg: Dict[str, Any] = dict(
        output_dir=out_dir,
        learning_rate=float(os.environ.get("TRAIN_LR", "5e-6")),
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_generations=num_gen,
        max_prompt_length=int(os.environ.get("GRPO_MAX_PROMPT_LEN", "384")),
        max_completion_length=int(os.environ.get("GRPO_MAX_COMPLETION_LEN", "128")),
        temperature=float(os.environ.get("GRPO_TEMPERATURE", "0.7")),
        top_p=float(os.environ.get("GRPO_TOP_P", "0.9")),
        bf16=bool(use_cuda),
        fp16=False,
        gradient_checkpointing=True,
        num_train_epochs=int(os.environ.get("TRAIN_NUM_EPOCHS", "1")),
        max_steps=max_steps,
        logging_steps=int(os.environ.get("LOGGING_STEPS", "1")),
        logging_first_step=True,
        report_to=report_to,
    )
    if report_to == "tensorboard":
        _cfg["logging_dir"] = str(tb_dir)
    training_args = GRPOConfig(**_cfg)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[openenv_sql_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print(
        f"=== Starting trainer.train() — GRPO max_steps={max_steps} (this is where step/reward logs appear) ===",
        flush=True,
    )
    trainer.train()

    print("Quick eval (post-train)...")
    post_avg_reward = eval_model_reward(model, tokenizer, train_dataset, max_items=8)
    trained_hard_reward = eval_model_reward(model, tokenizer, hard_dataset, max_items=hard_eval_n)
    if not eval_task_ids:
        with httpx.Client(base_url=get_bridge_url(), headers=BYPASS_HEADERS, timeout=get_request_timeout()) as client:
            eval_task_ids = _fetch_task_ids(client)
        print(f"Post-train per-task eval on {len(eval_task_ids)} task(s).", flush=True)
    after_by_task: Dict[str, float] = {}
    after_samples_all: List[float] = []
    for t_id in eval_task_ids:
        ds = make_task_dataset(t_id, rows_per_task=eval_samples)
        samples = eval_model_reward_samples(model, tokenizer, ds, max_items=eval_samples)
        after_samples_all.extend(samples)
        after_by_task[t_id] = float(sum(samples) / max(1, len(samples)))

    # --- Save artifacts (real logs/plots) ---
    artifacts = ArtifactPaths(root=Path(out_dir) / "artifacts")
    log_history = getattr(trainer.state, "log_history", []) or []
    save_log_history(log_history, artifacts)
    reward_series = extract_reward_series(log_history)
    write_metrics(log_history, reward_series, artifacts)
    # augment metrics with before/after
    metrics_path = artifacts.metrics_json
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        metrics = {}
    _delta_avg = (
        None
        if baseline_avg_reward is None
        else float(post_avg_reward) - float(baseline_avg_reward)
    )
    _delta_hard = (
        None
        if base_hard_reward is None
        else float(trained_hard_reward) - float(base_hard_reward)
    )
    metrics.update(
        {
            "openenv_base_url": get_bridge_url(),
            "train_max_steps": max_steps,
            "model_name": MODEL_NAME,
            "baseline_avg_reward": baseline_avg_reward,
            "post_avg_reward": post_avg_reward,
            "delta_avg_reward": _delta_avg,
            "base_hard_reward": base_hard_reward,
            "trained_hard_reward": trained_hard_reward,
            "delta_hard_reward": _delta_hard,
            "per_task_baseline_reward": before_by_task,
            "per_task_post_reward": after_by_task,
            "task_eval_samples": eval_samples,
            "tensorboard_dir": str(tb_dir) if report_to == "tensorboard" else None,
            "report_to": report_to,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_reward_curve(reward_series, artifacts)
    plot_reward_moving_average(reward_series, artifacts)
    loss_series = extract_numeric_series(log_history, "loss")
    plot_training_diagnostics(reward_series, loss_series, artifacts)
    try:
        import matplotlib.pyplot as plt

        labels = ["baseline", "post-train"]
        values = [baseline_avg_reward, post_avg_reward]
        if baseline_avg_reward is None or any(isinstance(v, float) and math.isnan(v) for v in values):
            print("Skipping before/after bar chart (missing baseline or post NaN).", flush=True)
        else:
            plt.figure(figsize=(5, 4))
            plt.bar(labels, values, color=["#94a3b8", "#22c55e"])
            plt.ylim(0, max(1.0, max(values) * 1.1))
            plt.title("Avg execution reward (sampled)")
            plt.ylabel("avg reward")
            out_path = artifacts.root / "before_after_avg_reward.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=200)
            print(f"Saved {out_path}")
    except Exception as e:
        print(f"Could not generate before/after plot: {e}")
    try:
        plot_performance_comparison(before_by_task, after_by_task, artifacts.root / "performance_comparison.png")
        plot_task_delta(before_by_task, after_by_task, artifacts.task_delta_png)
        plot_reward_distribution_shift(
            before_samples_all,
            after_samples_all,
            artifacts.root / "reward_distribution_shift.png",
        )
    except Exception as e:
        print(f"Could not generate task/distribution plots: {e}")

    model_dir = os.environ.get("MODEL_SAVE_DIR", "./sota_sql_agent_full")
    print("\nSaving trained model locally...")
    model.save_pretrained(model_dir)

    hub_id = os.environ.get("MODEL_HUB_REPO_ID", os.environ.get("HF_HUB_REPO_ID", "md896/sql-debug-agent-qwen05b-grpo"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if os.environ.get("SKIP_HUB_PUSH", "").strip() in ("1", "true", "yes"):
        print("SKIP_HUB_PUSH set — not pushing to Hub.")
    else:
        try:
            model.push_to_hub(hub_id, token=token)
            tokenizer.push_to_hub(hub_id, token=token)
            print(f"Pushed trained model to https://huggingface.co/{hub_id}")
        except Exception as e:
            print(f"Hub push failed (set HF_TOKEN / MODEL_HUB_REPO_ID or SKIP_HUB_PUSH=1): {e}")

    # Upload run artifacts back to the Space repo so you can download/view them.
    artifact_space = os.environ.get("ARTIFACT_SPACE_ID", "md896/sql-debug-env")
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    try:
        artifacts.train_series_json.write_text(
            json.dumps(
                {
                    "reward_series": reward_series,
                    "loss_series": loss_series,
                    "before_samples": before_samples_all,
                    "after_samples": after_samples_all,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Saved {artifacts.train_series_json}")
    except Exception as e:
        print(f"Could not save train series json: {e}")

    try:
        if token:
            api = HfApi(token=token)
            api.upload_folder(
                repo_id=artifact_space,
                repo_type="space",
                folder_path=str(artifacts.root),
                path_in_repo=f"artifacts/runs/{run_tag}",
                commit_message=f"Add training artifacts {run_tag}",
            )
            print(f"Uploaded artifacts to https://huggingface.co/spaces/{artifact_space}/tree/main/artifacts/runs/{run_tag}")
        else:
            print("No HF token in job env; skipping artifact upload.")
    except Exception as e:
        print(f"Artifact upload failed: {e}")

    print(f"\nTraining artifacts under {artifacts.root}")

if __name__ == "__main__":
    run_sota_train()
