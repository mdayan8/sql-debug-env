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
HF_TOKEN = os.environ.get("HF_TOKEN")
# Optional: used only when running environments via from_docker_image() flows.
LOCAL_IMAGE_NAME = os.environ.get("LOCAL_IMAGE_NAME")

try:
    if not HF_TOKEN:
        print("[DEBUG] WARNING: HF_TOKEN not found in environment. Model calls will fail.", flush=True)
except Exception:
    pass

# ── Environment config ───────────────────────────────────────────────────────
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "sql-debug-env"
TEMPERATURE = 0.0
MAX_TOKENS = 1024
SEED = int(os.environ.get("SEED", "1"))

# ── Per-task config ──────────────────────────────────────────────────────────
TASK_CONFIGS = {
    "easy_syntax_fix": {"max_steps": 10, "success_threshold": 0.8},
    "medium_logic_fix": {"max_steps": 20, "success_threshold": 0.7},
    "hard_multi_bug": {"max_steps": 30, "success_threshold": 0.5},
}
MIN_STRICT_SCORE = 0.001
MAX_STRICT_SCORE = 0.999


def strict_score(value: float) -> float:
    return min(MAX_STRICT_SCORE, max(MIN_STRICT_SCORE, value))


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
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
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
        "",
        "ORIGINAL BROKEN QUERY:",
        "```sql",
        f"{obs.get('original_query', '')}",
        "```",
    ]

    if obs.get("current_query"):
        lines += [
            "",
            "YOUR LAST SUBMITTED QUERY:",
            "```sql",
            f"{obs.get('current_query', '')}",
            "```",
        ]

    last_result = obs.get("last_query_result")
    if last_result:
        if last_result.get("success"):
            rows = last_result.get("rows", [])
            lines += [
                "",
                f"LAST QUERY RESULT: {len(rows)} rows returned",
                f"Sample (first 3): {json.dumps(rows[:3], default=str)}",
            ]
        else:
            lines += [
                "",
                f"LAST QUERY ERROR: {last_result.get('error_message', 'Unknown error')}",
            ]

    if obs.get("schema_info"):
        schema = obs["schema_info"].get("tables", {})
        lines += ["", "DATABASE SCHEMA:"]
        for table, cols in schema.items():
            col_str = ", ".join(f"{c['name']} ({c['type']})" for c in cols)
            lines.append(f"  {table}: {col_str}")

    if obs.get("error_details"):
        lines += ["", f"ERROR DETAILS: {obs['error_details']}"]

    if obs.get("sample_rows"):
        lines += ["", f"SAMPLE ROWS: {json.dumps(obs['sample_rows'][:3], default=str)}"]

    if obs.get("hint"):
        lines += ["", f"HINT: {obs['hint']}"]

    lines += [
        "",
        f"Current score: {obs.get('current_score', 0):.3f}",
        f"Steps remaining: {obs.get('steps_remaining', 0)}",
        f"Expected output: {obs.get('expected_description', '')}",
        "",
        "What is your next action? (respond with ONLY valid JSON)",
    ]

    return "\n".join(lines)


def call_model(client: OpenAI, prompt: str) -> Dict[str, Any]:
    """Call model and parse JSON action response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return {"action_type": "inspect_error"}
    except Exception:
        return {"action_type": "inspect_error"}


async def run_task(task_id: str) -> None:
    cfg = TASK_CONFIGS.get(task_id, {"max_steps": 20, "success_threshold": 0.5})
    max_steps = int(cfg["max_steps"])
    success_threshold = float(cfg["success_threshold"])

    log_start(task_id, BENCHMARK, MODEL_NAME)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    rewards: List[float] = []
    score = strict_score(0.0)
    done = False
    step_i = 0

    # Reset env
    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=30.0) as env:
        r = await env.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        payload = r.json()
        obs = payload["observation"]

        while (not done) and step_i < max_steps:
            step_i += 1
            prompt = build_prompt(obs, step_i, rewards)
            action = call_model(client, prompt)

            # Step env
            try:
                step_resp = await env.post("/step", json={"action": action})
                step_resp.raise_for_status()
                step_payload = step_resp.json()
                obs = step_payload["observation"]
                reward = float(step_payload.get("reward") or 0.0)
                done = bool(step_payload.get("done") or False)
                score = strict_score(float(obs.get("current_score") or 0.0))
                rewards.append(reward)
                log_step(step_i, json.dumps(action), reward, done, None)
            except Exception as e:
                rewards.append(0.0)
                log_step(step_i, json.dumps(action), 0.0, False, str(e))
                # try to recover by inspecting error
                try:
                    step_resp = await env.post("/step", json={"action": {"action_type": "inspect_error"}})
                    if step_resp.status_code == 200:
                        obs = step_resp.json()["observation"]
                except Exception:
                    pass

    success = score >= success_threshold
    log_end(success, step_i, score, rewards)


async def main() -> None:
    task = os.environ.get("TASK_ID", "easy_syntax_fix")
    await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())

