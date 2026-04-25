# 🏆 SQL Debug Env: FINAL REAL-WORLD BRIDGE
# (This script automatically installs its own dependencies)

# 1. AUTO-INSTALL LIBRARIES
import os
print("📦 Checking libraries...")
os.system("pip install trl accelerate wandb -U")

import httpx
import torch
import random
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- 2. BRIDGE CONFIGURATION ---
# Put your Localtunnel URL here
BRIDGE_URL = "https://metal-bushes-lie.loca.lt"
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Headers to bypass the Localtunnel landing page
BYPASS_HEADERS = {"Bypass-Tunnel-Reminder": "true"}

# --- 3. REAL DATASET GENERATION ---
def make_real_dataset():
    print(f"🔗 Connecting to your Mac at {BRIDGE_URL}...")
    tasks = ["easy_syntax_fix", "medium_logic_fix", "hard_multi_bug"]
    rows = []
    
    with httpx.Client(base_url=BRIDGE_URL, headers=BYPASS_HEADERS, timeout=30.0) as client:
        for t_id in tasks:
            try:
                resp = client.post("/reset", json={"task_id": t_id})
                obs = resp.json()["observation"]
                prompt = (
                    "Fix the following SQL query and provide only the fixed SQL.\n"
                    f"Task: {obs['task_description']}\n"
                    f"Broken Query: {obs['original_query']}\n"
                    "Fixed SQL:"
                )
                for _ in range(10): 
                    rows.append({"prompt": prompt, "task_id": t_id})
            except Exception as e:
                print(f"⚠️ Error fetching task {t_id}: {e}")
                
    if not rows:
        raise RuntimeError("Dataset is empty. Is your local server and tunnel running?")
    return Dataset.from_list(rows)

# --- 4. REAL REWARD FUNCTION ---
def sql_reward_func(completions, task_id, **kwargs):
    rewards = []
    with httpx.Client(base_url=BRIDGE_URL, headers=BYPASS_HEADERS, timeout=30.0) as client:
        for query, t_id in zip(completions, task_id):
            try:
                client.post("/reset", json={"task_id": t_id})
                sql_part = query.split("Fixed SQL:")[-1].strip() if "Fixed SQL:" in query else query.strip()
                resp = client.post("/step", json={"action": {"action_type": "submit_query", "query": sql_part}})
                reward = resp.json()["reward"]
            except Exception as e:
                print(f"❌ Connection Error for {t_id}: {e}")
                reward = 0.0
            
            reward += random.uniform(-1e-6, 1e-6)
            rewards.append(reward)
    return rewards

# --- 5. TRAINING LOOP ---
def run_real_world_train():
    print(f"🚀 Starting Real-World GRPO on Cloud GPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32, 
        device_map="auto"
    )

    training_args = GRPOConfig(
        output_dir="./real_results",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=64,
        num_train_epochs=1,
        max_steps=20, 
        logging_steps=1,
        fp16=False,
        report_to="wandb",
        push_to_hub=True, # <--- NEW: Pushes logs and model to HF
        hub_model_id="sql-debug-agent-7b", # <--- NEW: Your HF Model Repo Name
        hub_strategy="every_save"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[sql_reward_func],
        args=training_args,
        train_dataset=make_real_dataset(),
        processing_class=tokenizer,
    )

    print("🧠 Cloud Brain connected. Starting Real-World training...")
    trainer.train()

if __name__ == "__main__":
    run_real_world_train()
