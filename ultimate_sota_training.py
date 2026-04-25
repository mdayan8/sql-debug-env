"""
Unsloth + OpenEnv GRPO training (production-oriented).

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

import importlib.metadata as importlib_metadata
import json
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
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        ]
    )

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

    # Do not import transformers/trl here. Unsloth must be imported first later.


bootstrap_deps()

import httpx
import torch
from datasets import Dataset

# --- CRITICAL FIXES FOR HF JOBS ---
# 0. Unsloth checks importlib.metadata.version("vllm") at import time.
# In text-only GRPO runs we don't install vllm, so return a dummy version instead
# of crashing with PackageNotFoundError.
_real_pkg_version = importlib_metadata.version


def _safe_pkg_version(dist_name: str) -> str:
    if dist_name == "vllm":
        return "0.0.0"
    return _real_pkg_version(dist_name)


importlib_metadata.version = _safe_pkg_version

# 1. Mock vllm: TRL's GRPOTrainer (v0.18+) has a buggy import path that hard-fails if vllm is missing.
# We must provide a mock that satisfies both 'import' and 'importlib.util.find_spec'.
import sys
import types
import importlib.machinery

def mock_vllm_hierarchy():
    pkg_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.device_communicators",
        "vllm.model_executor",
        "vllm.model_executor.parallel_utils",
    ]
    leaf_names = [
        "vllm.distributed.device_communicators.pynccl",
    ]

    # Create proper package-like modules with submodule_search_locations so
    # unsloth's import fixes that inspect package paths don't crash.
    for m_name in pkg_names:
        mod = types.ModuleType(m_name)
        mod.__package__ = m_name
        mod.__path__ = [f"/tmp/mock_{m_name.replace('.', '_')}"]
        spec = importlib.machinery.ModuleSpec(m_name, loader=None, is_package=True)
        spec.submodule_search_locations = mod.__path__
        mod.__spec__ = spec
        sys.modules[m_name] = mod

    for m_name in leaf_names:
        mod = types.ModuleType(m_name)
        mod.__package__ = m_name.rsplit(".", 1)[0]
        mod.__spec__ = importlib.machinery.ModuleSpec(m_name, loader=None, is_package=False)
        sys.modules[m_name] = mod

mock_vllm_hierarchy()

# Import Unsloth before transformers / trl for its patching path.
from unsloth import FastLanguageModel

# 2. Mock llm_blender: Fix for TRANSFORMERS_CACHE removal in transformers 4.40+.
import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = "/tmp"

from trl import GRPOConfig, GRPOTrainer

# --- 1. CONFIGURATION (env-first; defaults match openenv.yaml) ---
_DEFAULT_OPENENV_BASE = "https://md896-sql-debug-env.hf.space"
BYPASS_HEADERS: Dict[str, str] = {}

MODEL_NAME = os.environ.get("TRAIN_MODEL_NAME", "unsloth/Qwen2.5-Coder-7B-Instruct")


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


def _resolve_report_to() -> str:
    raw = os.environ.get("TRL_REPORT_TO", "").strip().lower()
    if raw in ("", "auto"):
        return "wandb" if os.environ.get("WANDB_API_KEY") else "none"
    if raw in ("false", "no", "off", "none"):
        return "none"
    return raw


# --- 4. Unsloth GRPO training loop (live OpenEnv rewards) ---
def run_sota_train():
    max_steps = int(os.environ.get("TRAIN_MAX_STEPS", "200"))
    out_dir = os.environ.get("OUTPUT_DIR", "./sota_results")

    print(f"Starting Unsloth GRPO on {MODEL_NAME}...")
    print(
        f"OpenEnv={get_bridge_url()} | max_steps={max_steps} | "
        f"rows_per_task={os.environ.get('ROWS_PER_TASK', '48')} | "
        f"report_to={_resolve_report_to()}"
    )

    max_seq = int(os.environ.get("MAX_SEQ_LENGTH", "1024"))
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=max_seq,
        load_in_4bit=True,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # APPLY UNSLOTH LORA ADAPTERS
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    train_dataset = make_real_dataset()

    def quick_exec_eval(max_items: int = 8) -> float:
        """Sample prompts, generate completions, score with the same OpenEnv SQL reward."""
        subset = train_dataset.select(range(min(max_items, len(train_dataset))))
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
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completions.append(tokenizer.decode(out[0], skip_special_tokens=True))

        rewards = openenv_sql_reward_func(completions, task_ids)
        return float(sum(rewards) / max(len(rewards), 1))

    print("Quick baseline eval (pre-train)...")
    baseline_avg_reward = quick_exec_eval()

    report_to = _resolve_report_to()
    tb_dir = Path(out_dir) / "tensorboard"
    if report_to == "tensorboard":
        _ensure_dir(tb_dir)

    _cfg: Dict[str, Any] = dict(
        output_dir=out_dir,
        learning_rate=float(os.environ.get("TRAIN_LR", "5e-6")),
        per_device_train_batch_size=int(os.environ.get("PER_DEVICE_TRAIN_BS", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "2")),
        num_generations=int(os.environ.get("GRPO_NUM_GENERATIONS", "8")),
        max_completion_length=int(os.environ.get("GRPO_MAX_COMPLETION_LEN", "256")),
        temperature=float(os.environ.get("GRPO_TEMPERATURE", "0.9")),
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

    print("Training with live execution rewards against OpenEnv...")
    trainer.train()

    print("Quick eval (post-train)...")
    post_avg_reward = quick_exec_eval()

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
    metrics.update(
        {
            "openenv_base_url": get_bridge_url(),
            "train_max_steps": max_steps,
            "model_name": MODEL_NAME,
            "baseline_avg_reward": baseline_avg_reward,
            "post_avg_reward": post_avg_reward,
            "delta_avg_reward": post_avg_reward - baseline_avg_reward,
            "tensorboard_dir": str(tb_dir) if report_to == "tensorboard" else None,
            "report_to": report_to,
        }
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_reward_curve(reward_series, artifacts)
    try:
        import matplotlib.pyplot as plt

        labels = ["baseline", "post-train"]
        values = [baseline_avg_reward, post_avg_reward]
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

    lora_dir = os.environ.get("LORA_SAVE_DIR", "./sota_sql_agent_unsloth")
    print("\nSaving LoRA weights locally...")
    model.save_pretrained(lora_dir)

    hub_id = os.environ.get("HF_HUB_REPO_ID", "md896/sota-sql-agent-7b")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if os.environ.get("SKIP_HUB_PUSH", "").strip() in ("1", "true", "yes"):
        print("SKIP_HUB_PUSH set — not pushing to Hub.")
    else:
        try:
            model.push_to_hub(hub_id, token=token)
            print(f"Pushed LoRA to https://huggingface.co/{hub_id}")
        except Exception as e:
            print(f"Hub push failed (set HF_TOKEN / HF_HUB_REPO_ID or SKIP_HUB_PUSH=1): {e}")

    print(f"\nTraining artifacts under {artifacts.root}")

if __name__ == "__main__":
    run_sota_train()
