"""
inference.py — OpenEnv SQL Debug Environment Baseline Agent
MUST be at root level. MUST use exact [START]/[STEP]/[END] log format.
Uses OpenAI client. Reads from environment variables.
Runtime target: < 20 minutes on 2vCPU / 8GB.
"""
import asyncio
import os
import json
import sys
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
import httpx


# ── Configuration from environment variables ────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_KEY = os.environ.get("OPENAI_API_KEY", HF_TOKEN or "sk-placeholder")

# ── Environment config ───────────────────────────────────────────────────────
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "sql-debug-env"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
SEED = int(os.environ.get("SEED", "1"))

# ── Per-task config ──────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "easy_syntax_fix":  {"max_steps": 10,  "success_threshold": 0.8},
    "medium_logic_fix": {"max_steps": 20,  "success_threshold": 0.7},
    "hard_multi_bug":   {"max_steps": 30,  "success_threshold": 0.5},
}


# ── Logging functions (EXACT FORMAT — DO NOT MODIFY) ────────────────────────
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_str = error if error else "null"
    # Escape action for single-line logging
    action_clean = action.replace("\n", "\\n").replace('"', '\\"')[:200]
    print(
        f"[STEP] step={step} action=\"{action_clean}\" "
        f"reward={reward:.4f} done={str(done).lower()} error={error_str}",
        flush=True
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True
    )


# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert SQL debugger. You will receive a broken SQL query and must fix it.

You interact with a SQL debugging environment via JSON actions.

Available actions (respond with ONLY valid JSON, no markdown, no explanation):

1. Submit a fixed query:
{"action_type": "submit_query", "query": "SELECT ..."}

2. Inspect schema (free, no penalty):
{"action_type": "inspect_schema"}

3. Inspect last error (free, no penalty):
{"action_type": "inspect_error"}

4. Inspect sample rows from a table (free, no penalty):
{"action_type": "inspect_sample", "table_name": "table_name_here"}

Strategy:
- Start by submitting a fixed query if the bug is obvious
- Use inspect_schema first if you need to verify column names/table structure
- Use inspect_error to understand why your query failed
- Read error messages carefully — they tell you exactly what's wrong
- Fix one bug at a time and resubmit
- You get partial credit for partially correct queries

IMPORTANT: Respond with ONLY the JSON action. No explanation, no markdown blocks, just raw JSON."""


def build_prompt(obs: Dict[str, Any], step: int, reward_history: List[float]) -> str:
    """Build the user prompt for each step."""

    lines = [
        f"=== SQL Debugging Task (Step {step}) ===",
        f"Task: {obs.get('task_description', '')[:500]}",
        f"",
        f"ORIGINAL BROKEN QUERY:",
        f"```sql",
        f"{obs.get('original_query', '')}",
        f"```",
    ]

    if obs.get('current_query'):
        lines += [
            f"",
            f"YOUR LAST SUBMITTED QUERY:",
            f"```sql",
            f"{obs.get('current_query', '')}",
            f"```",
        ]

    last_result = obs.get('last_query_result')
    if last_result:
        if last_result.get('success'):
            rows = last_result.get('rows', [])
            lines += [
                f"",
                f"LAST QUERY RESULT: {len(rows)} rows returned",
                f"Sample (first 3): {json.dumps(rows[:3], default=str)}",
            ]
        else:
            lines += [
                f"",
                f"LAST QUERY ERROR: {last_result.get('error_message', 'Unknown error')}",
            ]

    if obs.get('schema_info'):
        schema = obs['schema_info'].get('tables', {})
        lines += [f"", f"DATABASE SCHEMA:"]
        for table, cols in schema.items():
            col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  {table}: {col_str}")

    if obs.get('error_details'):
        lines += [f"", f"ERROR DETAILS: {obs['error_details']}"]

    if obs.get('sample_rows'):
        lines += [f"", f"SAMPLE ROWS: {json.dumps(obs['sample_rows'][:3], default=str)}"]

    if obs.get('hint'):
        lines += [f"", f"HINT: {obs['hint']}"]

    lines += [
        f"",
        f"Current score: {obs.get('current_score', 0):.3f}",
        f"Steps remaining: {obs.get('steps_remaining', 0)}",
        f"Expected output: {obs.get('expected_description', '')}",
        f"",
        f"What is your next action? (respond with ONLY valid JSON)"
    ]

    return "\n".join(lines)


def call_model(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Call model and parse JSON action response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            seed=SEED,
            max_tokens=MAX_TOKENS,
        )
        text = (response.choices[0].message.content or "").strip()

        # Strip markdown if model wraps in backticks
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
        # Default fallback action
        return {"action_type": "inspect_schema"}
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"action_type": "inspect_schema"}


def run_task(
    client: OpenAI,
    task_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run one task episode synchronously via HTTP."""

    max_steps = config["max_steps"]
    success_threshold = config["success_threshold"]

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    with httpx.Client(base_url=ENV_BASE_URL, timeout=30.0) as http:
        # Reset
        reset_resp = http.post("/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        result = reset_resp.json()
        obs = result["observation"]
        done = result["done"]

        reward_history = []

        for step in range(1, max_steps + 1):
            if done:
                break

            # Get model action
            prompt = build_prompt(obs, step, reward_history)
            action_dict = call_model(client, prompt)

            # Execute step
            try:
                step_resp = http.post("/step", json={"action": action_dict})
                step_resp.raise_for_status()
                step_result = step_resp.json()
            except Exception as e:
                log_step(step=step, action=str(action_dict), reward=0.0, done=False, error=str(e))
                continue

            obs = step_result["observation"]
            reward = float(step_result.get("reward") or 0.0)
            done = step_result["done"]
            error = None
            info = step_result.get("info") or {}

            # Extract error for logging
            last_result = obs.get("last_query_result")
            if last_result and not last_result.get("success"):
                error = last_result.get("error_message", "")

            action_str = action_dict.get("query") or action_dict.get("action_type", "unknown")

            rewards.append(reward)
            reward_history.append(reward)
            steps_taken = step
            score = float(info.get("grade_score") or obs.get("current_score") or 0.0)

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

    # Compute final score
    score = min(max(score, 0.0), 1.0)
    success = score >= success_threshold

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards
    }


def main():
    """Run baseline agent across all 3 tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[DEBUG] Starting SQL Debug Env baseline", flush=True)
    print(f"[DEBUG] Model: {MODEL_NAME}", flush=True)
    print(f"[DEBUG] Env URL: {ENV_BASE_URL}", flush=True)

    # Wait for server to be ready
    max_wait = 30
    for i in range(max_wait):
        try:
            resp = httpx.get(f"{ENV_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"[DEBUG] Server ready", flush=True)
                break
        except:
            pass
        print(f"[DEBUG] Waiting for server... ({i+1}/{max_wait})", flush=True)
        time.sleep(1)

    all_results = []

    for task_id, config in TASK_CONFIGS.items():
        print(f"\n[DEBUG] Running task: {task_id}", flush=True)
        try:
            result = run_task(client, task_id, config)
            all_results.append(result)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
            log_end(success=False, steps=0, score=0.0, rewards=[])

        # Small delay between tasks
        time.sleep(2)

    # Summary
    print(f"\n[DEBUG] === BASELINE RESULTS ===", flush=True)
    total_score = 0.0
    for r in all_results:
        print(f"[DEBUG] {r['task_id']}: score={r['score']:.3f} success={r['success']}", flush=True)
        total_score += r['score']

    if all_results:
        avg = total_score / len(all_results)
        print(f"[DEBUG] Average score: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()

