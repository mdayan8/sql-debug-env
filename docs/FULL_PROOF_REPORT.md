# SQL Debug Env — Full Proof Verification Report

Date: 2026-04-23  
Workspace: `/Users/mdayan/Desktop/sql-debug-env`  
Branch/commit: `main` @ `9b71d1b`

## Executive Summary

**Working (verified):**
- Core environment logic (`server/env.py`, `server/database.py`, task graders, reward shaping)
- Unit tests (10/10) passing via `unittest`
- FastAPI server endpoints respond correctly when exercised via `curl`
- `openenv validate --verbose` passes (environment is “Ready for multi-mode deployment”)
- Docker image build succeeds and the container serves `/health`, `/tasks`, `/reset` correctly

**Not fully verified from this Codex sandbox (blocked by runtime constraints):**
- Python HTTP client scripts (`scripts/benchmark_local.py`, `inference.py`) cannot connect to `localhost` here due to sandbox socket restrictions (`PermissionError: [Errno 1] Operation not permitted`)

**Potential “works-on-my-machine” risks (not failures in unit tests):**
- Local installed package versions do **not** match `requirements.txt` pins (server still works in these checks, but reproducibility depends on using the pinned environment, e.g. Docker).
- `inference.py` uses `openai` Chat Completions style and hard-fails at import-time if `HF_TOKEN` is missing; compatibility depends on the installed `openai` package major version and env vars.

## What’s Implemented (“What’s Done”)

This repo implements a deterministic SQL debugging RL environment with:
- **Typed action/observation/reward** models (`server/models.py`)
- **In-memory SQLite episode DB** per reset (`server/database.py`)
- **3 deterministic tasks** (easy/medium/hard) with schema + seed + expected output + graders (`server/tasks/`)
- **Dense reward shaping** with strict clamping into `(0, 1)` for validator compatibility (`server/reward.py`)
- **OpenEnv-compatible HTTP API** (`server/main.py`) with:
  - `POST /reset`, `POST /step`, `GET /state`
  - `GET /tasks`, `GET /health`, `GET /benchmark`
- **OpenEnv entrypoint** wrapper (`server/app.py`)
- **Baseline agent runner** that calls an OpenAI model + steps the env (`inference.py`)

## How the Approach Works (and Why)

### Design intent
The environment is designed to be **deterministic** and **gradeable**:
- Deterministic SQLite schema + seed data → same query always yields same result.
- Deterministic expected outputs + graders → consistent scoring across runs/models.
- Strict score clamping into `(0, 1)` → aligns with OpenEnv validator expectations.

### Runtime flow
1. `POST /reset` creates a fresh `SQLDebugEnv`, which creates a new in-memory `EpisodeDatabase` and an `EpisodeState`.
2. Each `POST /step` executes one action:
   - `submit_query` executes a **SELECT-only** SQL query, then grades rows.
   - `inspect_schema` / `inspect_error` / `inspect_sample` returns info without grading changes.
   - `reset_query` resets `current_query` and applies a penalty.
3. `compute_reward(...)` returns a dense reward combining correctness/efficiency/progress/schema bonus minus penalties.

## Verification Environment

### Python/runtime
- Python: `3.14.2`

### Installed library versions (observed in this environment)
- `fastapi 0.128.0`
- `uvicorn 0.40.0`
- `pydantic 2.12.5`
- `openai 2.30.0`
- `httpx 0.28.1`
- `openenv-core 0.2.3`

Note: `requirements.txt` pins older versions (e.g. `fastapi==0.115.0`, `uvicorn==0.30.6`, `pydantic==2.9.2`).

## Tests / Checks Run (with Results)

### 1) Unit tests
Command:
```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```
Result:
- `Ran 10 tests in 0.003s` → `OK`

### 2) Bytecode compilation (syntax sanity)
Command:
```bash
python3 -m compileall -q .
```
Result:
- No errors

### 3) Dependency sanity
Command:
```bash
python3 -m pip check
```
Result:
- `No broken requirements found.`

### 4) OpenEnv structural validation
Command:
```bash
openenv validate --verbose
```
Result:
- `[OK] sql-debug-env: Ready for multi-mode deployment`

### 5) Docker build + container smoke test
Commands:
```bash
# start daemon (example: Colima)
colima start

docker build -t sql-debug-env:localtest .
docker run --rm -p 17860:7860 sql-debug-env:localtest
```
Result (verified here):
- `docker build` completed successfully.
- Container responded with:
  - `GET /health` → `200 OK`
  - `GET /tasks` → 3 tasks
  - `POST /reset` (tested with `medium_logic_fix`) → `200 OK`

## API Smoke Test (Local)

Server started (foreground) with:
```bash
uvicorn server.main:app --host 127.0.0.1 --port 7860
```

### Verified endpoints (via `curl`)
- `GET /health` → `200 OK` with `{"status":"ok","sessions_active":0}`
- `GET /tasks` → `200 OK` with 3 tasks: `easy_syntax_fix`, `medium_logic_fix`, `hard_multi_bug`
- `POST /reset` (`x-session-id: smoke`) → `200 OK` and observation includes `task_id` and `steps_taken=0`
- `POST /step` with:
  - `inspect_schema` → returns schema tables and small positive reward
  - `submit_query` (invalid table) → returns `success=false`, error recorded, not done
  - `inspect_error` → returns last error message
  - `inspect_sample` → returns 3 sample rows for a table
  - `reset_query` → resets query and returns min clamped reward
- `GET /state` → returns episode state (task id, steps, best score)

## What’s Broken / Blocked (Observed Here)

### A) Python HTTP clients cannot connect to localhost in this Codex sandbox
Observed failures:
- `python3 scripts/benchmark_local.py` → `httpx.ConnectError: [Errno 1] Operation not permitted`
- `urllib.request.urlopen("http://127.0.0.1:7860/health")` → `PermissionError: [Errno 1] Operation not permitted`

Implication:
- Any verification path that depends on Python making TCP connections (including `inference.py`) cannot be “fully proved” from this sandbox session.
- The server itself works (verified via `curl`), so this appears to be a sandbox constraint, not necessarily a repo bug.

## Recommended Next Proof Steps (If You Want CI-Grade Confidence)

- Add an integration test using FastAPI’s `TestClient` (no real sockets needed) to cover `/reset`, `/step`, `/state`.
- Add a Docker build + container smoke test in CI to ensure pinned deps and entrypoints stay healthy.
- Decide whether to:
  - Pin `openai<2` (to match `chat.completions` usage), or
  - Update `inference.py` to the current OpenAI client style and avoid import-time hard failure when env vars are missing.
