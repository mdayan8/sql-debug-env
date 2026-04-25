import os
import torch
import httpx
from transformers import AutoTokenizer, AutoModelForCausalLM

ENV_URL = "http://localhost:7860"
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

def test_logic():
    print(f"🚀 Starting Logic Smoke Test...")
    
    # 1. Check if server is up
    try:
        httpx.get(f"{ENV_URL}/health")
        print("✅ Environment server is alive.")
    except:
        print("❌ Error: Server not found. Run 'python3 -m uvicorn server.main:app --port 7860' first.")
        return

    # 2. Load model (CPU only to save disk/temp space)
    print(f"📦 Loading model {MODEL_NAME} on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")
    
    # 3. Get a task
    resp = httpx.post(f"{ENV_URL}/reset", json={"task_id": "easy_syntax_fix"})
    obs = resp.json()["observation"]
    print(f"📝 Task Loaded: {obs['task_description'][:100]}...")

    # 4. Ask Model for a fix
    prompt = f"Fix this SQL query:\n{obs['original_query']}\nProvide ONLY the fixed SQL query, no other text."
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("🤖 AI is thinking...")
    outputs = model.generate(
        inputs.input_ids, 
        max_new_tokens=100, 
        pad_token_id=tokenizer.eos_token_id
    )
    # Decode only the NEW tokens
    fix = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    if not fix:
        fix = "SELECT * FROM users;" # Fallback for test if AI is silent
        print("⚠️ AI was silent, using fallback query for connection test.")
    else:
        print(f"✨ AI Proposed Fix: {fix}")

    # 5. Get Reward
    print("🎯 Sending to environment for grading...")
    step_resp = httpx.post(
        f"{ENV_URL}/step", 
        json={"action": {"action_type": "submit_query", "query": fix}}
    )
    
    if step_resp.status_code != 200:
        print(f"❌ Server Error {step_resp.status_code}: {step_resp.text}")
        return

    result = step_resp.json()
    
    print(f"🏆 TEST RESULT:")
    print(f"   - Reward Score: {result.get('reward', 'MISSING')}")
    print(f"   - Done: {result.get('done', 'MISSING')}")
    
    if result.get('reward') and result['reward'] >= 0.5:
        print("   - Status: Success! System is fully operational.")
    else:
        print("   - Status: Connection test passed (Reward received).")

if __name__ == "__main__":
    test_logic()
