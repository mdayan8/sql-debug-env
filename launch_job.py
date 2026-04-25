"""
Submit ultimate_sota_training.py to Hugging Face GPU Jobs (HfApi.run_job).

The Job command must be a single robust shell line (semicolon-separated). Hugging Face
has been observed to flatten multiline `bash -lc` payloads, which breaks `set` and can
leave the job stuck or failing silently.

Requires: huggingface_hub, `huggingface-cli login`.

Secrets: if SKIP_HUB_PUSH is not 1, the job requests Hub secret name HF_TOKEN mapped into
the container as env HF_TOKEN (Settings → Access Tokens / Job secrets).

Environment (optional):
  HF_JOB_NAMESPACE     default: whoami
  HF_JOB_FLAVOR        default: h200 for 7B GRPO (VRAM headroom); l4x1 for smaller models; override anytime
  HF_JOB_IMAGE         default: pytorch CUDA 12.4 devel
  HF_JOB_TIMEOUT       default: 8h
  TRAIN_REPO_GIT_URL, OPENENV_BASE_URL
  TRAIN_MAX_STEPS      default: 80 (faster run; raise for stronger fit)
  ROWS_PER_TASK        default: 32
  GRPO_NUM_GENERATIONS default: 2
  SKIP_HUB_PUSH        default: 0
"""
from __future__ import annotations

import os
import shlex

from huggingface_hub import HfApi
from huggingface_hub.utils import get_token

_DEFAULT_REPO = "https://huggingface.co/spaces/md896/sql-debug-env"
_REPO_URL = os.environ.get("TRAIN_REPO_GIT_URL", _DEFAULT_REPO)
_OPENENV = os.environ.get("OPENENV_BASE_URL", "https://md896-sql-debug-env.hf.space")
_MAX_STEPS = os.environ.get("TRAIN_MAX_STEPS", "900")
_ROWS = os.environ.get("ROWS_PER_TASK", "128")
_NUM_GEN = os.environ.get("GRPO_NUM_GENERATIONS", "1")
_SKIP_PUSH = os.environ.get("SKIP_HUB_PUSH", "0")
_TIMEOUT = os.environ.get("HF_JOB_TIMEOUT", "12h")
_TRAIN_MODEL_NAME = os.environ.get("TRAIN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
# 7B + GRPO needs headroom (reference model, generations). H200 queues but runs reliably.
_DEFAULT_FLAVOR = "h200" if "7b" in _TRAIN_MODEL_NAME.lower() else "l4x1"
_FLAVOR = os.environ.get("HF_JOB_FLAVOR", _DEFAULT_FLAVOR)
_IMAGE = os.environ.get(
    "HF_JOB_IMAGE",
    "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
)
_NAMESPACE = os.environ.get("HF_JOB_NAMESPACE", "md896")

_SECRETS = None
_local_hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or get_token()
if _SKIP_PUSH.strip().lower() not in ("1", "true", "yes"):
    if _local_hf_token:
        _SECRETS = {"HF_TOKEN": _local_hf_token}
    else:
        # Job can still train; push/upload steps in script will gracefully skip/fail with clear logs.
        _SECRETS = None

# One line only — survives UI/API newline flattening.
_bash = (
    "set -euxo pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    "apt-get update -qq && apt-get install -y -qq git ca-certificates; "
    "export PIP_BREAK_SYSTEM_PACKAGES=1; "
    f"rm -rf train-repo; git clone {shlex.quote(_REPO_URL)} train-repo; "
    "cd train-repo; "
    "python -u ultimate_sota_training.py"
)

_job_env = {
    "OPENENV_BASE_URL": _OPENENV,
    "TRAIN_MAX_STEPS": _MAX_STEPS,
    "ROWS_PER_TASK": _ROWS,
    "GRPO_NUM_GENERATIONS": _NUM_GEN,
    "SKIP_HUB_PUSH": _SKIP_PUSH,
    "TRAIN_MODEL_NAME": _TRAIN_MODEL_NAME,
    "TRAIN_LR": os.environ.get("TRAIN_LR", "3e-6"),
    "GRPO_MAX_PROMPT_LEN": os.environ.get("GRPO_MAX_PROMPT_LEN", "384"),
    "GRPO_MAX_COMPLETION_LEN": os.environ.get("GRPO_MAX_COMPLETION_LEN", "128"),
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    "TASK_EVAL_SAMPLES": os.environ.get("TASK_EVAL_SAMPLES", "16"),
    "ARTIFACT_SPACE_ID": os.environ.get("ARTIFACT_SPACE_ID", "md896/sql-debug-env"),
    "MODEL_HUB_REPO_ID": os.environ.get(
        "MODEL_HUB_REPO_ID",
        "md896/sql-debug-agent-qwen25-7b-grpo-long" if "7b" in _TRAIN_MODEL_NAME.lower() else "md896/sql-debug-agent-qwen05b-grpo",
    ),
    "HARD_EVAL_SAMPLES": os.environ.get("HARD_EVAL_SAMPLES", "16"),
}

if __name__ == "__main__":
    api = HfApi()
    ns = _NAMESPACE
    job = api.run_job(
        image=_IMAGE,
        command=["bash", "-lc", _bash],
        flavor=_FLAVOR,
        namespace=ns,
        timeout=_TIMEOUT,
        secrets=_SECRETS,
        env=_job_env,
    )
    print("JOB_ID:", job.id)
    print("JOB_URL:", job.url)
    print("FLAVOR:", _FLAVOR, "| TRAIN_MAX_STEPS:", _MAX_STEPS, "| ROWS_PER_TASK:", _ROWS, "| TRAIN_MODEL_NAME:", _TRAIN_MODEL_NAME)
    print(
        "Note: SCHEDULING = HF queue until a GPU is assigned; 'No logs' is normal until RUNNING. "
        "For 7B GRPO default flavor is h200 (top VRAM). Override HF_JOB_FLAVOR if queue is too long."
    )
