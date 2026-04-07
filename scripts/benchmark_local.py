"""
Lightweight local benchmark for sql-debug-env.

Runs deterministic endpoint checks and prints simple latency metrics.
No LLM key required.
"""
from __future__ import annotations

import statistics
import time
from typing import Dict, List

import httpx


BASE_URL = "http://localhost:7860"


def timed_call(client: httpx.Client, method: str, path: str, json_body: Dict | None = None) -> float:
    start = time.perf_counter()
    if method == "GET":
        r = client.get(path)
    else:
        r = client.post(path, json=json_body)
    r.raise_for_status()
    return (time.perf_counter() - start) * 1000


def summarize(samples: List[float]) -> str:
    p50 = statistics.median(samples)
    p95 = sorted(samples)[int(len(samples) * 0.95) - 1]
    avg = statistics.mean(samples)
    return f"avg={avg:.2f}ms p50={p50:.2f}ms p95={p95:.2f}ms n={len(samples)}"


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        # Warmup + health check
        client.get("/health").raise_for_status()

        health_times = [timed_call(client, "GET", "/health") for _ in range(25)]
        tasks_times = [timed_call(client, "GET", "/tasks") for _ in range(25)]

        reset_times: List[float] = []
        step_times: List[float] = []
        for _ in range(25):
            reset_times.append(
                timed_call(client, "POST", "/reset", {"task_id": "easy_syntax_fix"})
            )
            step_times.append(
                timed_call(client, "POST", "/step", {"action": {"action_type": "inspect_schema"}})
            )

    print("Benchmark results (local)")
    print(f"GET /health: {summarize(health_times)}")
    print(f"GET /tasks: {summarize(tasks_times)}")
    print(f"POST /reset: {summarize(reset_times)}")
    print(f"POST /step (inspect_schema): {summarize(step_times)}")


if __name__ == "__main__":
    main()

