import os
import json
import httpx
import torch
import random
from typing import List, Dict, Any
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ────────────────────────────────────────────────────────────
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
# We use a tiny model for local testing. In the hackathon, upgrade this to 1.5B or 7B.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-Coder-0.5B-Instruct")
OUTPUT_DIR = "./grpo_sql_debug_results"

# ── 1. Dataset Generation ────────────────────────────────────────────────────
def make_dataset():
    """
    Creates a training dataset by pulling observations from the live environment.
    """
    print(f"[GRPO] Connecting to {ENV_URL} to build dataset...")
    tasks = ["easy_syntax_fix", "medium_logic_fix", "hard_multi_bug"]
    rows = []
    
    with httpx.Client(base_url=ENV_URL, timeout=10.0) as client:
        for task_id in tasks:
            try:
                resp = client.post("/reset", json={"task_id": task_id})
                resp.raise_for_status()
                obs = resp.json()["observation"]
                
                prompt = (
                    "Fix the following SQL query and provide only the fixed SQL.\n"
                    f"Task: {obs['task_description']}\n"
                    f"Broken Query: {obs['original_query']}\n"
                    "Fixed SQL:"
                )
                
                # Each task is repeated to create a batch for the trainer
                for _ in range(20):
                    rows.append({
                        "prompt": prompt,
                        "task_id": task_id
                    })
            except Exception as e:
                print(f"[GRPO] Failed to pull task {task_id}: {e}")
    
    if not rows:
        raise RuntimeError("Could not build dataset. Is the environment server running?")
        
    return Dataset.from_list(rows)

# ── 2. Reward Function ───────────────────────────────────────────────────────
def sql_reward_func(completions: List[str], task_id: List[str], **kwargs) -> List[float]:
    """
    The heart of the Self-Improving Agent. 
    It submits the model's generated query to the environment and returns the reward.
    """
    rewards = []
    
    with httpx.Client(base_url=ENV_URL, timeout=5.0) as client:
        # completions and task_id are lists of the same length
        for query, t_id in zip(completions, task_id):
            try:
                # Use a unique session ID for each generation in the GRPO group
                session_id = f"grpo-eval-{os.urandom(4).hex()}"
                
                # 1. Reset to the specific task
                client.post("/reset", json={"task_id": t_id}, headers={"x-session-id": session_id})
                
                # 2. Submit the generated query
                sql_part = query.split("Fixed SQL:")[-1].strip() if "Fixed SQL:" in query else query.strip()
                
                resp = client.post(
                    "/step", 
                    json={"action": {"action_type": "submit_query", "query": sql_part}},
                    headers={"x-session-id": session_id}
                )
                
                if resp.status_code == 200:
                    reward = float(resp.json().get("reward", 0.0))
                else:
                    reward = 0.0
            except Exception:
                reward = 0.0
            
            # ADD MICROSCOPIC NOISE: Prevents Zero-Variance crash
            reward += random.uniform(-1e-6, 1e-6)
            
            print(f"  [REWARD] Task: {t_id:18} | Score: {reward:.4f} | Query: {query[:50].strip()}...", flush=True)
            rewards.append(reward)
            
    return rewards

# ── 3. Training Loop ─────────────────────────────────────────────────────────
def train():
    print(f"[GRPO] Loading model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device = "cpu" # Forcing CPU for 100% stability on Mac
    print(f"[GRPO] Using device: {device} (Safe Mode)")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32, 
    ).to(device)

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4, 
        max_completion_length=32, # Short and sweet
        num_train_epochs=1,
        max_steps=5, 
        logging_steps=1,
        max_grad_norm=0.1, # Tightest possible clip
        beta=0.01,         # Low KL pressure
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sql_reward_func],
        args=training_args,
        train_dataset=make_dataset(),
        processing_class=tokenizer,
    )

    print("[GRPO] Starting training...")
    trainer.train()
    
    print(f"[GRPO] Training complete. Saving to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")

if __name__ == "__main__":
    # Check if server is running
    try:
        httpx.get(f"{ENV_URL}/health")
        train()
    except Exception as e:
        print(f"ERROR during training execution.")
        print(f"Details: {e}")
