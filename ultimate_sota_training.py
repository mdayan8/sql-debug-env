"""
🏆 Unsloth + OpenEnv GRPO training script

Goal: produce *real* training evidence (reward curves + logs) and optionally push LoRA
weights to the Hub.

This script is designed to run inside Hugging Face Jobs/Spaces containers where:
- system Python may be externally managed (PEP-668) → uses --break-system-packages
- preinstalled CUDA/PyTorch stacks can conflict with optional vision packages

Key stability choices:
- Avoid importing torchvision in text-only runs (it can break when torch/torchvision
  versions are mismatched by dependency resolution).
- Produce plots and metrics from the *actual* GRPO run (no hard-coded scores).
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
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
    """
    if os.environ.get("SKIP_BOOTSTRAP") == "1":
        return

    print("📦 Bootstrapping dependencies...")

    # Text-only run: torchvision/torchaudio are not required and are a common source
    # of crashes when torch versions shift in container images.
    _pip(["uninstall", "-y", "torchvision", "torchaudio"], check=False)

    # Keep these scoped; avoid blanket -U to reduce resolver churn.
    _pip(
        [
            "install",
            "--break-system-packages",
            "httpx>=0.27.0",
            "datasets>=3.4.1,<4.4.0",
            "trl>=0.18.2,<=0.24.0",
            "wandb",
            "matplotlib",
        ]
    )

    # Unsloth (and its dependency set) can be fast-moving; install from git.
    # Build isolation/resolution can sometimes change torch; removing torchvision
    # above keeps transformers imports stable for text-only workloads.
    _pip(
        [
            "install",
            "--break-system-packages",
            "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        ]
    )


bootstrap_deps()

import httpx
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

# --- 1. CONFIGURATION ---
# Using your permanent Hugging Face Space!
BRIDGE_URL = "https://md896-sql-debug-env.hf.space" 
BYPASS_HEADERS = {} # No longer needed for HF Spaces!

# Using the massive 7B Coder model, but squeezing it into memory using Unsloth 4-bit!
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct" 

# --- 2. THE XML FORMATTING PROMPT ---
SYSTEM_PROMPT = """You are an elite SQL Database Administrator fixing a critical fan trap (Cartesian Explosion).
You MUST output your reasoning process inside <think> tags.
After you have finished thinking, you MUST output the exact fixed SQL query inside <sql> tags.
Do not output any markdown blocks like ```sql.

Example:
<think>
I need to aggregate the totals first using a CTE to avoid a Cartesian explosion.
</think>
<sql>
WITH OrderTotals AS ( ... ) SELECT ...
</sql>"""

def make_real_dataset():
    print(f"🔗 Connecting to Environment at {BRIDGE_URL}...")
    tasks = ["hard_finance_explosion"] 
    rows = []
    
    with httpx.Client(base_url=BRIDGE_URL, headers=BYPASS_HEADERS, timeout=30.0) as client:
        for t_id in tasks:
            resp = client.post("/reset", json={"task_id": t_id})
            obs = resp.json()["observation"]
            
            prompt = (
                f"{SYSTEM_PROMPT}\n\n"
                f"Task: {obs['task_description']}\n"
                f"Broken Query: {obs['original_query']}\n\n"
                "Provide your <think> and <sql> output:"
            )
            # Generate 40 identical starting states for the model to explore
            for _ in range(40): 
                rows.append({"prompt": prompt, "task_id": t_id})
                
    if not rows:
        raise RuntimeError("Failed to connect to environment!")
    return Dataset.from_list(rows)

# --- 3. MULTI-REWARD SHAPING (The Secret Weapon) ---

def extract_xml_tag(text, tag):
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None

def format_reward_func(completions, **kwargs):
    """Reward 1: Did the model use <think> and <sql> tags? (+0.1)"""
    rewards = []
    for comp in completions:
        has_think = extract_xml_tag(comp, "think") is not None
        has_sql = extract_xml_tag(comp, "sql") is not None
        rewards.append(0.1 if (has_think and has_sql) else 0.0)
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
    return rewards

def execution_reward_func(completions, task_id, **kwargs):
    """Reward 3: The Ultimate Sandbox Test (+1.0)"""
    rewards = []
    with httpx.Client(base_url=BRIDGE_URL, headers=BYPASS_HEADERS, timeout=30.0) as client:
        for query, t_id in zip(completions, task_id):
            sql = extract_xml_tag(query, "sql")
            if not sql:
                rewards.append(0.0) 
                continue
                
            try:
                client.post("/reset", json={"task_id": t_id})
                resp = client.post("/step", json={"action": {"action_type": "submit_query", "query": sql}})
                reward = resp.json().get("reward", 0.0)
            except Exception:
                reward = 0.0
                
            reward += random.uniform(-1e-6, 1e-6) 
            rewards.append(reward)
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
    print(f"✅ Saved {paths.reward_curve_png}")


# --- 4. THE UNSLOTH + DEEPSEEK-R1 TRAINING LOOP ---
def run_sota_train():
    print(f"🚀 Starting Unsloth GRPO on {MODEL_NAME}...")
    
    # LOAD WITH UNSLOTH 4-BIT QUANTIZATION (2X FASTER, 70% LESS MEMORY)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=1024,
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
        Quick before/after check:
        - sample a few prompts
        - generate <think>/<sql>
        - score via live execution reward
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

    print("📏 Quick baseline eval (pre-train)...")
    baseline_avg_reward = quick_exec_eval()

    training_args = GRPOConfig(
        output_dir="./sota_results",
        learning_rate=5e-6, 
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=8, 
        max_completion_length=400, # Lots of room for <think> and <sql> CTEs
        temperature=0.9, # Forces creative exploration
        num_train_epochs=1,
        max_steps=30, 
        logging_steps=1,
        report_to="none" 
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, syntax_reward_func, execution_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("🧠 SOTA Sandbox Active. Let the RL begin...")
    trainer.train()

    print("📏 Quick eval (post-train)...")
    post_avg_reward = quick_exec_eval()

    # --- Save artifacts (real logs/plots) ---
    artifacts = ArtifactPaths(root=Path("./sota_results/artifacts"))
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
            "baseline_avg_reward": baseline_avg_reward,
            "post_avg_reward": post_avg_reward,
            "delta_avg_reward": post_avg_reward - baseline_avg_reward,
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
        print(f"✅ Saved {out_path}")
    except Exception as e:
        print(f"⚠️ Could not generate before/after plot: {e}")

    print("\n💾 Saving and (optionally) pushing LoRA weights...")
    model.save_pretrained("./sota_sql_agent_unsloth")
    
    # CRITICAL: Since you are running on HF Jobs, the server deletes everything when it finishes.
    # We MUST push the weights to your account so you can actually use them!
    try:
        model.push_to_hub("md896/sota-sql-agent-7b", token=os.environ.get("HF_TOKEN"))
        print("✅ Successfully pushed to https://huggingface.co/md896/sota-sql-agent-7b")
    except Exception as e:
        print(f"⚠️ Could not push to hub. Make sure HF_TOKEN is set. Error: {e}")

    print("\n📊 Training artifacts saved under ./sota_results/artifacts")

if __name__ == "__main__":
    run_sota_train()
