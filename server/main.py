"""
FastAPI server exposing the OpenEnv HTTP API.
Endpoints: POST /reset, POST /step, GET /state
Also includes: GET /tasks (list available tasks), GET /health
"""
import asyncio
import time
import statistics
from typing import Dict, Optional, List, Any
from contextlib import asynccontextmanager
import sqlite3

from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models import SQLDebugAction, SQLDebugObservation, EpisodeState
from .env import SQLDebugEnv, TASKS


# Session management: one env instance per session
# For HF Space: allow up to 64 concurrent sessions
MAX_SESSIONS = 64
_sessions: Dict[str, SQLDebugEnv] = {}
_session_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup all sessions on shutdown
    for env in _sessions.values():
        env.close()


app = FastAPI(
    title="SQL Debug Environment",
    description="OpenEnv-compliant SQL query debugging environment for RL agent training.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "name": "sql-debug-env",
        "status": "ok",
        "message": "Use /health, /tasks, /reset, /step, /state, /benchmark",
    }


@app.get("/favicon.ico", status_code=204)
async def favicon():
    return None


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy_syntax_fix"


class StepRequest(BaseModel):
    action: SQLDebugAction


async def get_or_create_session(session_id: str, task_id: str = "easy_syntax_fix") -> SQLDebugEnv:
    async with _session_lock:
        if session_id not in _sessions:
            if len(_sessions) >= MAX_SESSIONS:
                # Evict oldest session
                oldest = next(iter(_sessions))
                _sessions[oldest].close()
                del _sessions[oldest]
            _sessions[session_id] = SQLDebugEnv(task_id=task_id)
        return _sessions[session_id]


@app.get("/health")
async def health():
    return {"status": "ok", "sessions_active": len(_sessions)}


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with metadata."""
    return {
        "tasks": [task.to_dict() for task in TASKS.values()]
    }


def _stats(values: list[float]) -> Dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)
    p95_idx = max(0, int(n * 0.95) - 1)
    return {
        "avg_ms": round(statistics.mean(ordered), 3),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[p95_idx], 3),
        "n": n,
    }


@app.get("/benchmark")
async def benchmark(runs: int = 20):
    """
    Real-time benchmark endpoint (fresh measurements on every call).
    Safe to call from dashboards/web pages for live verification.
    """
    runs = max(1, min(runs, 100))

    health_times: list[float] = []
    tasks_times: list[float] = []
    reset_times: list[float] = []
    step_times: list[float] = []

    bench_env = SQLDebugEnv(task_id="easy_syntax_fix")
    try:
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = {"status": "ok", "sessions_active": len(_sessions)}
            health_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            _ = [task.to_dict() for task in TASKS.values()]
            tasks_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            await bench_env.reset()
            reset_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            await bench_env.step(SQLDebugAction(action_type="inspect_schema"))
            step_times.append((time.perf_counter() - t0) * 1000)
    finally:
        bench_env.close()

    return {
        "benchmark": {
            "runs": runs,
            "task_id": "easy_syntax_fix",
            "timestamp_epoch_ms": int(time.time() * 1000),
            "results": {
                "health": _stats(health_times),
                "tasks": _stats(tasks_times),
                "reset": _stats(reset_times),
                "step_inspect_schema": _stats(step_times),
            },
        }
    }


@app.post("/reset")
async def reset(
    request: ResetRequest = ResetRequest(),
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Reset the environment for a new episode.

    Returns initial observation with task description and broken query.
    """
    session_id = x_session_id or "default"
    task_id = request.task_id or "easy_syntax_fix"

    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}")

    # Always create fresh env on reset
    async with _session_lock:
        if session_id in _sessions:
            _sessions[session_id].close()
        _sessions[session_id] = SQLDebugEnv(task_id=task_id)

    env = _sessions[session_id]
    observation, info = await env.reset()

    return {
        "observation": observation.model_dump(),
        "info": info,
        "reward": None,
        "done": False
    }


@app.post("/step")
async def step(
    request: StepRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Execute one action in the environment.

    Action types:
    - submit_query: Submit SQL for evaluation (requires 'query' field)
    - inspect_schema: Get table schema (free action)
    - inspect_error: Get last error message (free action)
    - inspect_sample: Get sample rows from table (requires 'table_name')
    - reset_query: Reset to original broken query (small penalty)
    """
    session_id = x_session_id or "default"

    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /reset first.")

    env = _sessions[session_id]

    try:
        observation, reward, done, info = await env.step(request.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.post("/step_with_review")
async def step_with_review(
    request: StepRequest,
    x_session_id: Optional[str] = Header(default=None)
):
    """
    Execute a step with a Reviewer Agent layer.
    If the action is a query submission, the Reviewer validates it first.
    """
    session_id = x_session_id or "default"
    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="Session not found. Call /reset first.")
    
    env = _sessions[session_id]
    action = request.action

    if action.action_type == "submit_query" and action.query:
        # Reviewer checks the query before execution
        state = env.get_state()
        review = reviewer_check(action.query, state.db_schema or {})
        
        if not review["approved"]:
            # Reviewer rejected — return feedback without executing
            # Keep reward in strict (0, 1) range for OpenEnv compatibility
            reward = 0.001
            obs = env.to_observation(
                last_action_type="review_rejected",
                error_details=f"REVIEWER REJECTION: {review['reason']}",
            )
            
            return {
                "observation": obs.model_dump(),
                "reward": reward,
                "done": False,
                "info": {"review_rejected": True, "reason": review["reason"]}
            }

    # If approved or not a query, proceed to normal step
    try:
        observation, reward, done, info = await env.step(action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": observation.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }


def reviewer_check(query: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple rule-based Reviewer Agent.
    Checks:
    1. Table existence
    2. Read-only (SELECT/WITH)
    3. Basic SQLite syntax (EXPLAIN)
    """
    query_upper = query.upper().strip()
    
    # Check 1: Is it a read query?
    if not (query_upper.startswith("SELECT") or query_upper.startswith("WITH")):
        return {"approved": False, "reason": "Only SELECT queries or CTEs (WITH) are allowed."}

    # Check 2: Does it reference valid tables?
    tables = list(schema.keys())
    referenced = [t for t in tables if t.upper() in query_upper]
    if not referenced and tables:
        return {"approved": False, "reason": f"Query does not reference any valid tables. Available: {tables}"}

    # Check 3: Syntax check via EXPLAIN on a lightweight schema stub.
    # Build minimal CREATE TABLE statements from the provided schema so EXPLAIN
    # doesn't fail with "no such table" for otherwise-valid queries.
    try:
        conn = sqlite3.connect(":memory:")
        for table_name, columns in (schema or {}).items():
            if not columns:
                continue
            col_defs = []
            for col in columns:
                name = col.get("name", "col")
                col_type = col.get("type", "TEXT")
                nullable = col.get("nullable")
                not_null = " NOT NULL" if str(nullable).upper() == "NO" else ""
                col_defs.append(f"{name} {col_type}{not_null}")
            cols_sql = ", ".join(col_defs) if col_defs else "id INTEGER"
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_sql})")

        # We don't have the actual data here, but EXPLAIN is sufficient for
        # catching syntax errors and many semantic issues.
        conn.execute(f"EXPLAIN {query}")
        conn.close()
    except sqlite3.OperationalError as e:
        return {"approved": False, "reason": f"Syntax error caught by Reviewer: {e}"}
    except Exception as e:
        return {"approved": False, "reason": f"Reviewer error: {e}"}

    return {"approved": True, "reason": "Query approved"}


@app.get("/state")
async def state(x_session_id: Optional[str] = Header(default=None)):
    """Return current full episode state."""
    session_id = x_session_id or "default"

    if session_id not in _sessions:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    env = _sessions[session_id]
    try:
        current_state = env.get_state()
        return current_state.model_dump()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
