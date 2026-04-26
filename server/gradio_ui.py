"""
Single-page Gradio UI for the Hugging Face Space (same process as the OpenEnv FastAPI API).

Playground uses POST /reset and POST /step via loopback HTTP with X-Session-Id.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple

import httpx

COLAB_FIRST_TRAINING = (
    "https://colab.research.google.com/drive/1H6SLfCBhHzRJtnymLgevjfyytWUximF5"
    "#scrollTo=j-9MptXvmPk8"
)
COLAB_TRAINING_ROOT = (
    "https://colab.research.google.com/drive/1H6SLfCBhHzRJtnymLgevjfyytWUximF5"
    "#scrollTo=x5YuvatGyyu_"
)
HF_SPACE = "https://huggingface.co/spaces/md896/sql-debug-env"
HF_SAMPLE_REWARDS = (
    "https://huggingface.co/spaces/md896/sql-debug-env/tree/main/"
    "artifacts/runs/20260426-064318-sample-rewards-32eval"
)
HF_EVAL_32 = (
    "https://huggingface.co/spaces/md896/sql-debug-env/tree/main/"
    "artifacts/runs/20260426-060502-final-pass-32eval"
)
HF_MODEL = "https://huggingface.co/md896/sql-debug-agent-qwen25-05b-grpo-wandb-continue-v2"
GITHUB_REPO = "https://github.com/mdayan8/sql-debug-env"
WANDB_TRAINING_RUN = "https://wandb.ai/mdayanbag-pesitm/sql-debug-grpo-best-budget/workspace?nw=nwusermdayanbag"
GCLOUD_TEXT2SQL_BLOG = "https://cloud.google.com/blog/products/databases/techniques-for-improving-text-to-sql"
OURBENCH_PAPER = "https://arxiv.org/abs/2601.18119"

PREDEFINED_QUERIES: dict[str, list[tuple[str, str]]] = {
    "easy_syntax_fix": [
        ("Broken baseline: typo table", "SELECT * FROM userss;"),
        ("Simple lookup", "SELECT id, name FROM users ORDER BY id LIMIT 10;"),
        ("Potential invalid write", "UPDATE users SET name='test';"),
    ],
    "medium_logic_fix": [
        ("Broken: missing GROUP BY", "SELECT department, COUNT(*) FROM employees;"),
        ("Revenue by month", "SELECT strftime('%Y-%m', order_date) AS ym, SUM(amount) FROM orders GROUP BY ym ORDER BY ym;"),
        ("Top entities", "SELECT customer_id, SUM(total) AS spend FROM invoices GROUP BY customer_id ORDER BY spend DESC LIMIT 5;"),
    ],
    "hard_multi_bug": [
        ("Broken join alias", "SELECT u.name, o.total FROM users u JOIN orders o ON user.id = o.user_id;"),
        ("Join + aggregate", "SELECT p.category, AVG(p.price) AS avg_price FROM products p GROUP BY p.category ORDER BY avg_price DESC;"),
        ("Nested query", "SELECT name FROM customers WHERE id IN (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 2);"),
    ],
    "hard_finance_explosion": [
        ("Broken finance calc", "SELECT account_id, SUM(amount) / COUNT(*) AS risk FROM txn GROUP BY account;"),
        ("PnL-style aggregate", "SELECT symbol, SUM(CASE WHEN side='BUY' THEN -notional ELSE notional END) AS pnl FROM trades GROUP BY symbol ORDER BY pnl DESC;"),
        ("Daily exposure", "SELECT date(trade_ts) AS d, SUM(abs(notional)) AS exposure FROM trades GROUP BY d ORDER BY d;"),
    ],
}

GRADIO_CSS = """
:root {
  --sde-ink: #0f172a;
  --sde-muted: #64748b;
  --sde-line: #e2e8f0;
  --sde-card: #ffffff;
  --sde-glow:
    radial-gradient(120% 140% at 0% 0%, rgba(45, 212, 191, 0.22) 0%, rgba(45, 212, 191, 0) 52%),
    radial-gradient(120% 140% at 100% 0%, rgba(147, 197, 253, 0.22) 0%, rgba(147, 197, 253, 0) 55%),
    linear-gradient(132deg, #0f172a 0%, #1e293b 45%, #0f766e 100%);
  /* HF Space embed is often dark; keep prose readable (theme alone used near-black on black). */
  --sde-body-text: #e2e8f0;
  --sde-heading-text: #f8fafc;
}
.gradio-container {
  max-width: 1180px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  color: #e2e8f0 !important;
}
/* Gradio 5 + HF: markdown / prose defaults can match a dark shell and disappear */
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose td,
.gradio-container .prose th {
  color: var(--sde-body-text) !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose strong {
  color: var(--sde-heading-text) !important;
}
.gradio-container .prose a {
  color: #7dd3fc !important;
}
.gradio-container .prose code {
  color: #fef3c7 !important;
  background: rgba(15, 23, 42, 0.55) !important;
}
.sde-hero-wrap {
  background: var(--sde-glow);
  color: #f8fafc;
  border-radius: 20px;
  padding: 1.75rem 1.5rem 1.5rem;
  margin-bottom: 1.25rem;
  border: 1px solid rgba(148, 163, 184, 0.24);
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.20), inset 0 1px 0 rgba(255, 255, 255, 0.12);
}
.sde-hero-wrap .sde-hero-title {
  margin: 0 0 0.35rem 0;
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  line-height: 1.2;
  color: #f8fafc !important;
}
.sde-hero-wrap .sde-hero-lede {
  margin: 0 0 0.5rem 0;
  color: #e2e8f0 !important;
  font-size: 0.95rem;
  line-height: 1.55;
}
.sde-hero-subnav { margin-bottom: 0.75rem; font-size: 0.88rem; }
.sde-hero-wrap .sde-hero-subnav a {
  color: #a5f3fc !important;
  font-weight: 600;
}
.sde-pill-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.sde-pill {
  display: inline-block;
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  background: rgba(15, 23, 42, 0.26);
  border: 1px solid rgba(226, 232, 240, 0.34);
  color: #f8fafc;
}
.sde-section-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--sde-heading-text) !important;
  margin: 1.5rem 0 0.75rem 0;
  letter-spacing: -0.02em;
}
.sde-muted-caption {
  color: #94a3b8 !important;
  font-size: 0.9rem;
}
.sde-link-row a {
  color: #7dd3fc !important;
  font-weight: 600;
  margin-right: 1rem;
}
.sde-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 0.5rem 0 1rem;
}
.sde-kpi {
  background: #ffffff;
  border: 1px solid #dbe3f0;
  border-radius: 14px;
  padding: 0.85rem 0.95rem;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}
.sde-kpi .v {
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: #0f172a;
}
.sde-kpi .k {
  margin-top: 0.15rem;
  font-size: 0.73rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #64748b;
}
.sde-callout {
  border-left: 4px solid #2563eb;
  background: #eff6ff;
  color: #1e3a8a;
  padding: 0.7rem 0.8rem;
  border-radius: 8px;
  margin: 0.5rem 0 0.75rem;
  font-size: 0.86rem;
}
@media (max-width: 900px) {
  .sde-kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


def _api_base() -> str:
    return os.environ.get(
        "INTERNAL_API_BASE",
        f"http://127.0.0.1:{os.environ.get('PORT', '7860')}",
    ).rstrip("/")


def _blog_url() -> str:
    return (os.environ.get("BLOG_URL") or "").strip()


def _http() -> httpx.Client:
    return httpx.Client(timeout=120.0)


def _img_path(static_dir: Path, *names: str) -> Optional[str]:
    for n in names:
        p = static_dir / n
        if p.is_file():
            return str(p.resolve())
    return None


def _preset_options(task_id: str) -> list[str]:
    return [name for name, _ in PREDEFINED_QUERIES.get(task_id, [])]


def _preset_query(task_id: str, preset_name: str) -> str:
    for name, query in PREDEFINED_QUERIES.get(task_id, []):
        if name == preset_name:
            return query
    return ""


def _safe_reward(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_blocks(static_dir: Path) -> Any:
    import gradio as gr

    wf = _img_path(static_dir, "diagram-end-to-end-workflow.png", "environment-workflow.png")
    chart_leap = _img_path(static_dir, "chart-performance-leap.png", "hero_performance_leap.png")
    chart_dual = _img_path(static_dir, "chart-comparison-shift.png", "hero_dual_benchmark.png")
    chart_spider = _img_path(static_dir, "chart-spider-benchmark.png", "hero_spider_sota.png")
    proof_combo = _img_path(static_dir, "proof-combo.png")
    proof_dist = _img_path(static_dir, "proof-distribution-shift.png")
    final_gallery_paths = [
        "training_reward_curve_final.png",
        "training_diagnostics_dual_axis_final.png",
        "baseline_vs_trained_by_task_final.png",
        "task_delta_post_minus_base_final.png",
        "reward_distribution_shift_red_green_final.png",
        "presentation_combo_final.png",
        "benchmark_style_summary_final.png",
        "checkpoint_leaderboard_step_vs_reward_final.png",
        "cost_vs_performance_final.png",
    ]
    final_gallery: list[tuple[str, str]] = []
    for filename in final_gallery_paths:
        path = _img_path(static_dir, filename)
        if path:
            title = filename.replace("_final.png", "").replace("_", " ").title()
            final_gallery.append((path, title))

    blog = _blog_url()
    blog_md = (
        f"### Blog\n[Read the write-up]({blog})"
        if blog
        else "### Blog\nAdd a **Space secret** named `BLOG_URL` with your post URL (e.g. Medium, personal site, or Hugging Face blog)."
    )

    task_choices = [
        "easy_syntax_fix",
        "medium_logic_fix",
        "hard_multi_bug",
        "hard_finance_explosion",
    ]

    def reset_fn(
        task_id: str, session_id: Optional[str]
    ) -> Tuple[str, str, str, str]:
        sid = session_id or str(uuid.uuid4())
        try:
            with _http() as client:
                r = client.post(
                    f"{_api_base()}/reset",
                    json={"task_id": task_id},
                    headers={"X-Session-Id": sid},
                )
                r.raise_for_status()
                data = r.json()
        except Exception as e:
            err = {"error": str(e), "hint": "Is the server listening on PORT?"}
            return json.dumps(err, indent=2), "", sid, f"Session: `{sid}` · **error**"
        obs = json.dumps(data, indent=2)
        q = (data.get("observation") or {}).get("original_query") or ""
        return obs, q, sid, f"Session: `{sid}`"

    def submit_fn(
        query: str, session_id: Optional[str]
    ) -> Tuple[str, str]:
        if not session_id:
            return (
                json.dumps({"error": "Click “Reset task” first to create a session."}, indent=2),
                "",
            )
        payload = {"action": {"action_type": "submit_query", "query": query or ""}}
        try:
            with _http() as client:
                r = client.post(
                    f"{_api_base()}/step",
                    json=payload,
                    headers={"X-Session-Id": session_id},
                )
                r.raise_for_status()
                data = r.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            return json.dumps({"error": str(e), "detail": detail}, indent=2), ""
        except Exception as e:
            return json.dumps({"error": str(e)}, indent=2), ""
        out = json.dumps(data, indent=2)
        reward = data.get("reward")
        done = data.get("done")
        return out, f"**reward** `{reward}` · **done** `{done}`"

    def run_preset_suite(
        task_id: str, session_id: Optional[str]
    ) -> Tuple[str, str, str, str]:
        sid = session_id or str(uuid.uuid4())
        presets = PREDEFINED_QUERIES.get(task_id, [])
        if not presets:
            return "No presets for selected task.", "{}", sid, f"Session: `{sid}`"

        rows: list[str] = []
        rewards: list[float] = []
        done_count = 0
        error_count = 0

        with _http() as client:
            for idx, (name, query) in enumerate(presets, start=1):
                try:
                    client.post(
                        f"{_api_base()}/reset",
                        json={"task_id": task_id},
                        headers={"X-Session-Id": sid},
                    ).raise_for_status()
                    step_resp = client.post(
                        f"{_api_base()}/step",
                        json={"action": {"action_type": "submit_query", "query": query}},
                        headers={"X-Session-Id": sid},
                    )
                    step_resp.raise_for_status()
                    data = step_resp.json()
                    reward = _safe_reward(data.get("reward"))
                    done = bool(data.get("done"))
                    info = data.get("info") or {}
                    label = "pass" if reward >= 0.5 else "check"
                    rewards.append(reward)
                    done_count += int(done)
                    note = "review_rejected" if info.get("review_rejected") else ""
                    rows.append(
                        f"| {idx} | {name} | `{reward:.3f}` | `{done}` | {label} {note} |"
                    )
                except Exception as e:
                    error_count += 1
                    rows.append(
                        f"| {idx} | {name} | `0.000` | `False` | error: {str(e)[:120]} |"
                    )

        avg_reward = (sum(rewards) / len(rewards)) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        min_reward = min(rewards) if rewards else 0.0
        suite_md = (
            "#### Preset suite report\n"
            "| # | Preset | Reward | Done | Note |\n"
            "|---|---|---:|:---:|---|\n"
            + "\n".join(rows)
            + "\n\n"
            + f"**Summary:** avg reward `{avg_reward:.3f}` · min `{min_reward:.3f}` · max `{max_reward:.3f}` · "
              f"done count `{done_count}` · errors `{error_count}`"
        )
        suite_json = json.dumps(
            {
                "task_id": task_id,
                "session_id": sid,
                "n_presets": len(presets),
                "avg_reward": round(avg_reward, 4),
                "min_reward": round(min_reward, 4),
                "max_reward": round(max_reward, 4),
                "done_count": done_count,
                "error_count": error_count,
            },
            indent=2,
        )
        return suite_md, suite_json, sid, f"Session: `{sid}`"

    font = gr.themes.GoogleFont("Plus Jakarta Sans")
    mono = gr.themes.GoogleFont("JetBrains Mono")
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="slate",
        neutral_hue="slate",
        font=(font, "ui-sans-serif", "system-ui"),
        font_mono=(mono, "ui-monospace", "monospace"),
    )

    with gr.Blocks(
        title="SQL Debug Environment",
        analytics_enabled=False,
        theme=theme,
        css=GRADIO_CSS,
    ) as demo:
        gr.HTML(
            """
<div class="sde-hero-wrap">
  <div class="sde-hero-subnav">
    <a href="/demo">HTML demo (main Space page)</a> · <a href="/docs">OpenEnv API</a>
  </div>
  <div class="sde-hero-title">SQL Debug Environment</div>
  <p class="sde-hero-lede">OpenEnv-compliant SQL repair · live SQLite rewards · TRL / GRPO training on this same Space.
     One page: benchmarks, artifacts, architecture, and a live playground.</p>
  <div class="sde-pill-row">
    <span class="sde-pill">OpenEnv</span>
    <span class="sde-pill">FastAPI</span>
    <span class="sde-pill">Gradio</span>
    <span class="sde-pill">TRL · GRPO</span>
  </div>
</div>
            """.strip(),
            elem_id="sde-hero",
        )

        gr.Markdown(
            "### First context: training proof first\n"
            f"- **Source code:** [GitHub — mdayan8/sql-debug-env]({GITHUB_REPO})\n"
            f"- **First training notebook (auto-install cell):** [Open in Colab]({COLAB_FIRST_TRAINING})\n"
            f"- **Full training Colab (root anchor):** [Open in Colab]({COLAB_TRAINING_ROOT})\n"
            f"- **Weights & Biases (example run):** [Dashboard]({WANDB_TRAINING_RUN})\n"
            f"- **Sample-reward eval artifacts (32-run JSON on Hub):** [Browse files]({HF_SAMPLE_REWARDS})\n"
            f"- **Earlier 32-eval pass folder:** [Browse files]({HF_EVAL_32})\n"
            f"- **Trained model card:** [md896/sql-debug-agent…]({HF_MODEL})\n"
            f"- **This Space:** [{HF_SPACE}]({HF_SPACE})"
        )

        gr.HTML(
            """
<div class="sde-kpi-grid">
  <div class="sde-kpi"><div class="v">0.5B → 7B</div><div class="k">Model progression</div></div>
  <div class="sde-kpi"><div class="v">32-run eval</div><div class="k">Final artifact pass</div></div>
  <div class="sde-kpi"><div class="v">78.5%</div><div class="k">Spider-style headline</div></div>
  <div class="sde-kpi"><div class="v">Execution reward</div><div class="k">Primary training signal</div></div>
</div>
            """.strip()
        )

        gr.HTML(
            '<div class="sde-callout"><strong>Notebook vibe:</strong> this page is intentionally written as field notes + reproducible cells, not a static deck. Every number should map to an artifact.</div>'
        )

        gr.Code(
            label="First training context cell (from your Colab)",
            language="python",
            interactive=False,
            value=(
                "# 🏆 SQL Debug Env: FINAL REAL-WORLD BRIDGE\n"
                "import os\n"
                "print('📦 Checking libraries...')\n"
                "os.system('pip install trl accelerate wandb -U')\n\n"
                "import httpx\n"
                "import torch\n"
            ),
            lines=8,
        )

        gr.Markdown(
            "### Lab notebook stats (TL;DR)\n"
            "- First training pass started with **Qwen/Qwen2.5-Coder-0.5B-Instruct** for environment wiring and fast iteration.\n"
            "- Main training/eval track used **Qwen/Qwen2.5-Coder-7B-Instruct** with execution-grounded reward loops.\n"
            "- Final reporting is tied to run artifacts and static charts committed under `server/static/`."
        )

        gr.Markdown(
            "| Track | Model | Role | Evidence |\n"
            "|---|---|---|---|\n"
            "| First bridge run | Qwen/Qwen2.5-Coder-0.5B-Instruct | Fast validation of API/reward loop and notebook flow | First training context + W&B run |\n"
            "| Base reference | Qwen/Qwen2.5-Coder-7B-Instruct | Baseline behavior before RL updates | Spider/comparison charts |\n"
            "| Current agent | RL-updated checkpoint on 7B track | Improved execution-grounded SQL fixing | HF model + eval artifacts + sample rewards |\n"
        )

        gr.Markdown(
            "### Run timeline (quick history)\n"
            "| Stage | What happened | Why it matters |\n"
            "|---|---|---|\n"
            "| Bridge run | Fast setup with **Qwen2.5-Coder-0.5B** | Validated API + reward wiring quickly |\n"
            "| Main baseline | Moved to **Qwen2.5-Coder-7B-Instruct** | Better capacity for SQL structure + joins |\n"
            "| RL iterations | Session-consistent reset/step reward loop | Converted text quality into runtime behavior |\n"
            "| Hard-test reporting | `presentation_graphs_out_final` committed under `server/static/` | Keeps evaluation auditable on-page |\n"
        )

        gr.Code(
            label="Notebook cell: baseline evaluator sketch (7B track)",
            language="python",
            interactive=False,
            value=(
                "from transformers import AutoTokenizer, AutoModelForCausalLM\n"
                "MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'\n"
                "tokenizer = AutoTokenizer.from_pretrained(MODEL)\n"
                "model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto')\n"
                "# generate SQL -> POST /reset -> POST /step -> score by execution reward\n"
            ),
            lines=7,
        )

        gr.Code(
            label="Notebook cell: regenerate presentation plots from real artifacts",
            language="shell",
            interactive=False,
            value=(
                "python presentation_graphs.py \\\n"
                "  --sample-rewards-json artifacts/runs/20260426-064318-sample-rewards-32eval/sample_rewards_final.json \\\n"
                "  --output-dir presentation_graphs_out_final\n"
                "cp presentation_graphs_out_final/*.png server/static/\n"
            ),
            lines=5,
        )

        gr.Code(
            label="Notebook cell: live reward loop (execution-grounded)",
            language="python",
            interactive=False,
            value=(
                "with httpx.Client(base_url=ENV_URL, timeout=30.0) as client:\n"
                "    client.post('/reset', json={'task_id': task}, headers={'X-Session-Id': sid})\n"
                "    resp = client.post('/step', json={'action': {'action_type': 'submit_query', 'query': sql}},\n"
                "                       headers={'X-Session-Id': sid})\n"
                "    reward = resp.json().get('reward', 0.0)\n"
                "    # reward drives policy updates and eval comparisons\n"
            ),
            lines=7,
        )

        gr.Markdown(
            "### Failure taxonomy (from runtime debugging)\n"
            "| Failure type | Typical symptom | Why execution feedback helps |\n"
            "|---|---|---|\n"
            "| Schema mismatch | unknown table/column | reward drops immediately and error details guide correction |\n"
            "| Join logic bug | duplicated or missing rows | execution reveals semantic mismatch not visible in text quality |\n"
            "| Aggregation bug | incorrect GROUP BY totals | deterministic graders expose numerical drift |\n"
            "| Risky query behavior | unsafe or invalid action | reviewer path blocks while preserving learning signal |\n"
        )

        gr.Markdown('<p class="sde-section-title">Benchmark visuals</p>')
        gr.Markdown(
            "| Metric snapshot | Value |\n"
            "|---|---|\n"
            "| Spider chart: Industry baseline | **48.2%** |\n"
            "| Spider chart: Qwen-7B base | **52.4%** |\n"
            "| Spider chart: RL agent | **78.5%** |\n"
            "| Performance leap chart | **0.0% -> 25.0%** (base to RL in that run view) |\n"
        )
        with gr.Row(equal_height=True):
            if chart_leap:
                gr.Image(value=chart_leap, label="Performance leap (Spider-style)", type="filepath", scale=1)
            if chart_dual:
                gr.Image(value=chart_dual, label="Comparison + reward shift", type="filepath", scale=2)
            if chart_spider:
                gr.Image(value=chart_spider, label="Spider-style headline chart", type="filepath", scale=1)

        gr.Markdown(
            '<p class="sde-section-title">Training run charts (repo static)</p>'
            '<span class="sde-muted-caption">Training plots from real runs. Regenerate with `presentation_graphs.py`; commit PNGs under `server/static/`.</span>'
        )
        with gr.Row():
            if proof_combo:
                gr.Image(value=proof_combo, label="Presentation combo", type="filepath", scale=1)
            if proof_dist:
                gr.Image(value=proof_dist, label="Reward distribution shift", type="filepath", scale=1)

        if final_gallery:
            gr.Markdown(
                '<p class="sde-section-title">Hard-testing proof set (presentation_graphs_out_final)</p>'
                "<span style='color:#64748b;font-size:0.9rem'>All generated graphs from the final evaluation set.</span>"
            )
            gr.Gallery(
                value=final_gallery,
                label="Final hard-testing charts",
                preview=True,
                columns=3,
                height="auto",
                object_fit="contain",
            )

        gr.Markdown('<p class="sde-section-title">Environment architecture</p>')
        if wf:
            gr.Image(value=wf, label="End-to-end workflow", type="filepath", show_label=True)
        else:
            gr.Markdown("*Add `server/static/diagram-end-to-end-workflow.png`*")

        gr.Markdown(
            '<p class="sde-section-title">OpenEnv HTTP API</p>'
            f"`GET /health` · `GET /tasks` · `POST /reset` · `POST /step` · `POST /step_with_review` · `GET /state` · `GET /benchmark` · "
            f"loopback base `{_api_base()}` (override with **INTERNAL_API_BASE**)."
        )

        gr.Markdown('<p class="sde-section-title">Live playground</p>')
        session = gr.State(None)
        session_md = gr.Markdown("Session: *click “Reset task”*")
        with gr.Row():
            task = gr.Dropdown(
                choices=task_choices,
                value="easy_syntax_fix",
                label="Task",
                scale=1,
            )
            btn_reset = gr.Button("Reset task", variant="primary", scale=0, min_width=140)
            btn_submit = gr.Button("Submit query", variant="secondary", scale=0, min_width=140)
            btn_run_suite = gr.Button("Run preset suite", variant="secondary", scale=0, min_width=160)
        preset_name = gr.Dropdown(
            choices=_preset_options("easy_syntax_fix"),
            value=_preset_options("easy_syntax_fix")[0],
            label="Predefined test query",
        )
        btn_load_preset = gr.Button("Load predefined query", variant="secondary")
        sql = gr.Code(label="Candidate SQL", language="sql", lines=12)
        result_hint = gr.Markdown("")
        with gr.Row():
            obs_json = gr.Code(
                language="json",
                label="Observation (/reset)",
                lines=12,
                interactive=False,
                scale=1,
            )
            step_json = gr.Code(
                language="json",
                label="Step (/step)",
                lines=12,
                interactive=False,
                scale=1,
            )
        suite_md = gr.Markdown("")
        suite_json = gr.Code(
            label="Preset suite summary",
            language="json",
            lines=10,
            interactive=False,
        )

        btn_reset.click(
            reset_fn,
            inputs=[task, session],
            outputs=[obs_json, sql, session, session_md],
        )
        btn_submit.click(
            submit_fn,
            inputs=[sql, session],
            outputs=[step_json, result_hint],
        )
        task.change(
            lambda t: gr.Dropdown(
                choices=_preset_options(t),
                value=_preset_options(t)[0] if _preset_options(t) else None,
            ),
            inputs=[task],
            outputs=[preset_name],
        )
        btn_load_preset.click(
            lambda t, p: _preset_query(t, p or ""),
            inputs=[task, preset_name],
            outputs=[sql],
        )
        btn_run_suite.click(
            run_preset_suite,
            inputs=[task, session],
            outputs=[suite_md, suite_json, session, session_md],
        )

        gr.Markdown('<p class="sde-section-title">Blog</p>')
        gr.Markdown(blog_md)
        gr.Markdown(
            "### Why I picked SQL debugging and why this architecture exists\n"
            "“The goal is not to generate beautiful SQL text. The goal is to produce SQL fixes that survive execution, repeatedly, under changing runtime conditions.”\n\n"
            "### The cost of “almost right” SQL\n"
            "Industry time-use reporting commonly puts **roughly a quarter to a third** of analytics and data-engineering work into fixing queries and pipelines—"
            "**not** shipping net-new insights, **not** launching features, but **debugging SQL that already looked reasonable** in a notebook or PR.\n\n"
            "### Benchmarks vs production\n"
            "On Spider-style leaderboards, headline numbers often sit in the **high 80s to low 90s (%)**. In messy enterprise warehouses—drifting schemas, implicit business rules, "
            "join explosions, permissioned views—teams routinely describe effective success rates closer to the **10–30%** band unless the system closes the loop with "
            "**execution-grounded feedback** (run the SQL, read the error or result, attribute reward to what changed).\n\n"
            "SQL debugging is one of the few tasks where *language quality* and *system quality* diverge sharply: a query can be neat, plausible, and still fail in production. "
            "This project forces the agent to optimize for **behavior under execution**, not only fluency under prompting."
        )
        gr.HTML(
            """
<div class="sde-kpi-grid">
  <div class="sde-kpi"><div class="v">0.5B -> 7B</div><div class="k">Model track from first bridge run to main baseline.</div></div>
  <div class="sde-kpi"><div class="v">32-run eval</div><div class="k">Final artifact path with sample rewards and run logs.</div></div>
  <div class="sde-kpi"><div class="v">Execution-first</div><div class="k">Reward is computed from runtime outcomes, not prompt resemblance.</div></div>
  <div class="sde-kpi"><div class="v">Traceable claims</div><div class="k">Metrics should map back to run files and checkpoints.</div></div>
</div>
            """.strip()
        )
        gr.Markdown(
            "#### What leaderboards hide\n"
            "Canonical text-to-SQL suites are valuable scientific instruments: they keep comparisons honest. They are also cleaner than most corporate warehouses. "
            "That is why two statements can both be true: models can score **very high** on Spider-style tasks while practitioners still report **low tens to low thirties** "
            "effective reliability in production unless they invest in harnesses, guardrails, and iterative repair grounded in execution.\n\n"
            "- **Latency of truth**: prose feedback is fast; execution feedback is slower—and decisive.\n"
            "- **Credit assignment**: without runtime signal you reward plausible text; with it you reward joins, aggregates, and safe rewrites that actually run.\n"
            "- **Drift**: schemas evolve; the training surface must stay repeatable even when the world is messy.\n\n"
            "#### OpenEnv framing (why this is not just a demo UI)\n"
            "The environment follows an OpenEnv-style interface: `reset -> observation`, `step(action) -> observation, reward, done, info`. "
            "This is important because it gives the training loop a stable contract. Every algorithmic change can be tested against the same API semantics, which improves reproducibility.\n\n"
            "#### Reward math (what is actually optimized)\n"
            "At a high level, each step reward is composed from executed outcomes:\n\n"
            "\\[\n"
            "R_t = w_c C_t + w_e E_t + w_p P_t + w_s S_t - \\lambda \\cdot \\text{Penalty}_t\n"
            "\\]\n\n"
            "- \\(C_t\\): correctness signal (did query satisfy the task objective)\n"
            "- \\(E_t\\): execution quality (valid execution / error handling)\n"
            "- \\(P_t\\): progress toward a valid fix\n"
            "- \\(S_t\\): schema-aware behavior bonus\n"
            "- Penalty: unsafe / invalid / degenerate behavior\n\n"
            "Episode objective:\n\n"
            "\\[\n"
            "J(\\pi) = \\mathbb{E}_{\\tau \\sim \\pi}\\left[\\sum_{t=0}^{T} \\gamma^t R_t\\right]\n"
            "\\]\n\n"
            "This makes the optimization target explicit: not token similarity, but expected runtime return.\n\n"
            "#### Architecture decisions that matter technically\n"
            "1. **Session-isolated database state**: each episode gets a clean in-memory SQLite environment.\n"
            "2. **Deterministic tasks/graders**: stable reward surfaces for comparison across runs.\n"
            "3. **Reviewer-guard path**: risk control without collapsing the learning signal.\n"
            "4. **Typed observations + action history**: easier debugging and post-hoc analysis.\n\n"
            "#### Data and reporting stats on this page\n"
            "| Metric | Value | Source |\n"
            "|---|---:|---|\n"
            "| Spider-style industry baseline | 48.2% | chart-spider-benchmark |\n"
            "| Qwen-7B base | 52.4% | chart-spider-benchmark |\n"
            "| RL agent headline | 78.5% | chart-spider-benchmark |\n"
            "| Performance leap view | 0.0% -> 25.0% | chart-performance-leap |\n"
            "| Eval artifact pass | 32-run | HF run folder + sample rewards |\n\n"
            "#### Why start with 0.5B then move to 7B\n"
            "The first bridge run on **Qwen2.5-Coder-0.5B** is intentionally about speed of iteration: verify environment wiring, reward path, and notebook workflow quickly. "
            "The **7B track** is then used for stronger SQL reasoning capacity and better convergence under execution-grounded rewards.\n\n"
            "#### How to read this Space\n"
            "- **Diagram** — client → API → env core → data/reward → training and artifacts.\n"
            "- **Playground** — same `POST /reset` and `POST /step` loop as training, with explicit `X-Session-Id`.\n"
            "- **Charts + static PNGs** — committed under `server/static/` so claims stay diffable and auditable.\n\n"
            "#### Motivation recap\n"
            "I did not build this to prove that a model can emit valid-looking SQL. I built it to make SQL repair measurable as an engineering problem under runtime constraints. "
            "The evidence-first layout (first context, live loop, artifact chain) is deliberate: each reported number should be traceable to run data, not presentation-only visuals.\n\n"
            "*Note: percentage ranges summarize common practitioner reporting and public benchmark narratives; your organization’s numbers will differ—treat them as motivation to measure, not as universal constants.*"
        )
        gr.Markdown(
            f"- [Google Cloud: techniques for improving text-to-SQL]({GCLOUD_TEXT2SQL_BLOG})\n"
            f"- [OurBench / Squirrel: enterprise SQL debugging benchmark]({OURBENCH_PAPER})\n"
            f"- [GitHub repository]({GITHUB_REPO})"
        )

    return demo


def mount_gradio(app: Any, static_dir: Path) -> Any:
    """Mount single-page Gradio at `/` (Space home) while API routes stay on the same app."""
    import gradio as gr

    blocks = build_blocks(static_dir)
    return gr.mount_gradio_app(
        app,
        blocks,
        path="/gradio",
        allowed_paths=[str(static_dir.resolve())],
    )
