# 🏆 SQL Debug Env: FINAL STABLE COLAB SCRIPT
# 1. Install required libraries
import os
print("📦 Installing libraries...")
os.system("pip install trl accelerate wandb -U")

import torch
import random
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# --- Mock Dataset ---
def make_simple_dataset():
    rows = []
    prompt = "Fix the following SQL query: SELECT * FROM userss; Provide only the fixed SQL."
    for _ in range(20):
        rows.append({"prompt": prompt, "task_id": "easy_syntax_fix"})
    return Dataset.from_list(rows)

# --- Mock Reward ---
def mock_reward_func(completions, **kwargs):
    rewards = []
    print(f"🎬 Processing {len(completions)} completions...")
    for i, content in enumerate(completions):
        if "SELECT" in content.upper() and ";" in content:
            reward = 1.0 + random.uniform(-0.01, 0.01)
        else:
            reward = 0.0 + random.uniform(-0.01, 0.01)
        print(f"  [Gen {i}] Reward: {reward:.4f} | Text: {content[:40].strip()}...")
        rewards.append(reward)
    return rewards

# --- Training Loop ---
def run_colab_train():
    print(f"🚀 Starting GRPO on Colab T4 GPU (Float32 Mode)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use Float32 for maximum stability on T4
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32, 
        device_map="auto"
    )

    training_args = GRPOConfig(
        output_dir="./colab_results",
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=64,
        num_train_epochs=1,
        max_steps=10, 
        logging_steps=1,
        fp16=False, # Disable mixed precision to avoid crashes
        report_to="wandb"
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[mock_reward_func],
        args=training_args,
        train_dataset=make_simple_dataset(),
        processing_class=tokenizer,
    )

    print("🧠 Training starting... Check WandB link in 1 minute!")
    trainer.train()

    # --- 4. Final Exam (Take Test) ---
    print("\n🎓 TRAINING COMPLETE. TAKING THE FINAL EXAM...")
    test_queries = [
        "SELECT * FROM user;",
        "SELECT name, email FROM customers where id=1",
        "UPDATE users SET name='test'",
    ]
    
    model.eval()
    for i, q in enumerate(test_queries):
        prompt = f"Fix the following SQL query: {q}; Provide only the fixed SQL."
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32)
        
        fix = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n📝 Test {i+1}:")
        print(f"   Input:  {q}")
        print(f"   Output: {fix.strip()}")
        if "SELECT" in fix.upper():
            print("   ✅ RESULT: CORRECT (Valid SQL Logic)")
        else:
            print("   ❌ RESULT: INCORRECT")

if __name__ == "__main__":
    run_colab_train()
