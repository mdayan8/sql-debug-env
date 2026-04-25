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
  REASONING_XML_TAG         — XML tag name for chain-of-thought (default: think)
  TRAIN_MAX_STEPS           — GRPO optimizer steps (default: 200; was 30 for smoke)
  TRAIN_NUM_EPOCHS, TRAIN_LR, GRPO_NUM_GENERATIONS, GRPO_MAX_COMPLETION_LEN
  PER_DEVICE_TRAIN_BS, GRAD_ACCUM
  TRL_REPORT_TO             — none | wandb | tensorboard (auto: wandb if key else tensorboard)
  BOOTSTRAP_*_VERSION       — pin transformers / accelerate / trl for HF Jobs (see bootstrap_deps)
  Artifacts: artifacts/reward_components.jsonl, artifacts/trainer_on_log.jsonl, tensorboard/
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

import contextvars
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set by TrainerCallback so reward funcs can tag JSONL rows with the real global_step.
CURRENT_GRPO_STEP: contextvars.ContextVar[int] = contextvars.ContextVar("CURRENT_GRPO_STEP", default=-1)


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

    _pip(
        [
            "install",
            "--break-system-packages",
            "httpx>=0.27.0",
            "datasets>=3.4.1,<4.4.0",
            "matplotlib",
            "tensorboard",
            "wandb",
        ]
    )

    _tf = os.environ.get("BOOTSTRAP_TRANSFORMERS_VERSION", "4.48.3")
    _acc = os.environ.get("BOOTSTRAP_ACCELERATE_VERSION", "0.34.2")
    _trl = os.environ.get("BOOTSTRAP_TRL_VERSION", "0.18.2")
    _pip(
        [
            "install",
            "--break-system-packages",
            f"transformers=={_tf}",
            f"accelerate=={_acc}",
            f"trl=={_trl}",
        ]
    )

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

    try:
        import accelerate  # noqa: F401
        import transformers  # noqa: F401
        from trl import GRPOConfig as _BootstrapGRPOConfig  # noqa: F401

        _ = _BootstrapGRPOConfig
    except Exception as e:
        raise RuntimeError(
            "Post-bootstrap import check failed. Adjust BOOTSTRAP_*_VERSION or SKIP_BOOTSTRAP=1."
        ) from e


bootstrap_deps()

import httpx
import torch
from datasets import Dataset

# --- CRITICAL FIXES FOR HF JOBS ---
# 1. Mock vllm: TRL's GRPOTrainer (v0.18+) has a buggy import path that hard-fails if vllm is missing.
# We must provide a mock that satisfies both 'import' and 'importlib.util.find_spec'.
import sys
import types
import importlib.machinery
from unittest.mock import MagicMock

def mock_vllm_hierarchy():
    for m_name in [
        "vllm", 
        "vllm.distributed", 
        "vllm.distributed.device_communicators", 
        "vllm.distributed.device_communicators.pynccl",
        "vllm.model_executor",
        "vllm.model_executor.parallel_utils",
    ]:
        mock_m = MagicMock(spec=types.ModuleType)
        mock_m.__name__ = m_name
        mock_m.__spec__ = importlib.machinery.ModuleSpec(m_name, None)
        sys.modules[m_name] = mock_m

mock_vllm_hierarchy()

# 2. Mock llm_blender: Fix for TRANSFORMERS_CACHE removal in transformers 4.40+.
import transformers.utils.hub
if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
    transformers.utils.hub.TRANSFORMERS_CACHE = "/tmp"

from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# --- 1. CONFIGURATION (env-first; defaults match openenv.yaml) ---
_DEFAULT_OPENENV_BASE = "https://md896-sql-debug-env.hf.space"
BYPASS_HEADERS: Dict[str, str] = {}

MODEL_NAME = os.environ.get("TRAIN_MODEL_NAME", "unsloth/Qwen2.5-Coder-7B-Instruct")


def get_bridge_url() -> str:
    return os.environ.get("OPENENV_BASE_URL", _DEFAULT_OPENENV_BASE).rstrip("/")


def get_request_timeout() -> float:
    return float(os.environ.get("OPENENV_REQUEST_TIMEOUT_SEC", "120"))


def build_system_prompt() -> str:
    """Single prompt template for every task (easy → expert); tag name is configurable."""
    tag = os.environ.get("REASONING_XML_TAG", "think")
    return f"""You are an elite SQL engineer. You fix broken SQLite analytics queries using the task description and the broken query.
You MUST output your reasoning process inside <{tag}> tags.
After you have finished thinking, you MUST output the exact fixed SQL query inside <sql> tags.
Do not output any markdown blocks like ```sql.

Example:
<{tag}>
I will check joins, filters, and aggregation, then write a corrected SELECT or WITH query.
</{tag}>
<sql>
WITH OrderTotals AS (SELECT order_id, SUM(amount) AS total FROM line_items GROUP BY order_id)
SELECT o.id, ot.total FROM orders o JOIN OrderTotals ot ON o.id = ot.order_id;
</sql>"""


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
    bridge = get_bridge_url()
    timeout = get_request_timeout()
    rows_per_task = max(1, int(os.environ.get("ROWS_PER_TASK", "48")))
    system = build_system_prompt()

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
                f"{system}\n\n"
                f"Task: {obs['task_description']}\n"
                f"Broken Query: {obs['original_query']}\n\n"
                f"Provide your <{os.environ.get('REASONING_XML_TAG', 'think')}> and <sql> output:"
            )
            for _ in range(rows_per_task):
                rows.append({"prompt": prompt, "task_id": t_id})

    if not rows:
        raise RuntimeError("Failed to build dataset (no rows).")
    print(f"Dataset: {len(rows)} prompts ({rows_per_task} per task).")
    return Dataset.from_list(rows)

# --- 3. MULTI-REWARD SHAPING + JSONL logging (per-component batch stats) ---

_REWARD_COMPONENTS_JSONL: Optional[Path] = None


def extract_xml_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _reward_batch_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
    return {"mean": mean, "std": math.sqrt(var), "min": min(values), "max": max(values)}


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _log_reward_component(name: str, values: List[float]) -> None:
    if _REWARD_COMPONENTS_JSONL is None:
        return
    _append_jsonl(
        _REWARD_COMPONENTS_JSONL,
        {
            "time_epoch_s": time.time(),
            "global_step": CURRENT_GRPO_STEP.get(),
            "reward_component": name,
            "n": len(values),
            **_reward_batch_stats(values),
        },
    )


def format_reward_func(completions, **kwargs):
    """Reward 1: CoT + sql XML tags (+0.1). Tag name follows REASONING_XML_TAG."""
    tag = os.environ.get("REASONING_XML_TAG", "think")
    rewards = []
    for comp in completions:
        has_think = extract_xml_tag(comp, tag) is not None
        has_sql = extract_xml_tag(comp, "sql") is not None
        rewards.append(0.1 if (has_think and has_sql) else 0.0)
    _log_reward_component("format_xml", rewards)
    return rewards


def syntax_reward_func(completions, **kwargs):
    """Reward 2: Does the SQL look like valid code? (+0.2)"""
    rewards = []
    for comp in completions:
        sql = extract_xml_tag(comp, "sql")
        if sql and (sql.upper().startswith("SELECT") or sql.upper().startswith("WITH")):
            rewards.append(0.2)
        else:
            rewards.append(0.0)
    _log_reward_component("syntax_select_with", rewards)
    return rewards


def execution_reward_func(completions, task_id, **kwargs):
    """Reward 3: live OpenEnv submit_query against the real Space/API (not a stub)."""
    rewards: List[float] = []
    base = get_bridge_url()
    timeout = get_request_timeout()
    with httpx.Client(base_url=base, headers=BYPASS_HEADERS, timeout=timeout) as client:
        for query, t_id in zip(completions, task_id):
            sql = extract_xml_tag(query, "sql")
            if not sql:
                rewards.append(0.0)
                continue

            session_headers = {"X-Session-Id": str(uuid.uuid4())}
            try:
                r0 = client.post("/reset", json={"task_id": t_id}, headers=session_headers)
                r0.raise_for_status()
                resp = client.post(
                    "/step",
                    json={"action": {"action_type": "submit_query", "query": sql}},
                    headers=session_headers,
                )
                resp.raise_for_status()
                reward = float(resp.json().get("reward", 0.0))
            except Exception:
                reward = 0.0

            reward += random.uniform(-1e-6, 1e-6)
            rewards.append(reward)
    _log_reward_component("openenv_execution", rewards)
    return rewards


def length_shape_reward_func(completions, **kwargs):
    """Reward 4: soft preference for shorter completions (bounded; does not replace execution reward)."""
    cap = float(os.environ.get("COMPLETION_SOFT_CHAR_CAP", "3500"))
    bonus_max = float(os.environ.get("LENGTH_BONUS_MAX", "0.05"))
    rewards: List[float] = []
    for comp in completions:
        L = len(comp) if comp else 0
        if L <= 0:
            rewards.append(0.0)
        else:
            rewards.append(bonus_max * max(0.0, 1.0 - min(L, cap) / cap))
    _log_reward_component("length_shape", rewards)
    return rewards


class GrpoStepContextCallback(TrainerCallback):
    """Expose true global_step to reward funcs for JSONL alignment."""

    def on_step_begin(self, args, state, control, **kwargs):
        CURRENT_GRPO_STEP.set(int(state.global_step))


class JsonlOnLogCallback(TrainerCallback):
    """Mirror every trainer `logs` dict to JSONL (loss, learning_rate, reward keys, etc.)."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open("w", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        row: Dict[str, Any] = {"global_step": int(state.global_step), **dict(logs)}
        self._fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fp.flush()

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._fp.close()
        except Exception:
            pass

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
        if os.environ.get("WANDB_API_KEY"):
            return "wandb"
        return "tensorboard"
    if raw in ("false", "no", "off", "none"):
        return "none"
    return raw


# --- 4. Unsloth GRPO training loop (live OpenEnv rewards) ---
def run_sota_train():
    global _REWARD_COMPONENTS_JSONL

    max_steps = int(os.environ.get("TRAIN_MAX_STEPS", "200"))
    out_dir = os.environ.get("OUTPUT_DIR", "./sota_results")
    artifacts_early = Path(out_dir) / "artifacts"
    _ensure_dir(artifacts_early)
    _REWARD_COMPONENTS_JSONL = artifacts_early / "reward_components.jsonl"
    _REWARD_COMPONENTS_JSONL.write_text("", encoding="utf-8")

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
        """
        Quick before/after check: sample prompts, generate CoT + sql, score via live OpenEnv.
        """
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

        rewards = execution_reward_func(completions, task_ids)
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
        max_completion_length=int(os.environ.get("GRPO_MAX_COMPLETION_LEN", "512")),
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

    trainer_logs_path = artifacts_early / "trainer_on_log.jsonl"
    trainer_logs_path.write_text("", encoding="utf-8")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[
            format_reward_func,
            syntax_reward_func,
            execution_reward_func,
            length_shape_reward_func,
        ],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[
            GrpoStepContextCallback(),
            JsonlOnLogCallback(trainer_logs_path),
        ],
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
            "reward_components_jsonl": str(artifacts_early / "reward_components.jsonl"),
            "trainer_on_log_jsonl": str(artifacts_early / "trainer_on_log.jsonl"),
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
