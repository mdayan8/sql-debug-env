from huggingface_hub import HfApi
api = HfApi()
try:
    job = api.create_compute_job(
        namespace="md896",
        flavor="a10g-small",
        image="pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel",
        command=["bash", "-c", "set -euxo pipefail; apt-get update; apt-get install -y git; git clone https://huggingface.co/spaces/md896/sql-debug-env; cd sql-debug-env; python -u ultimate_sota_training.py"],
        secrets=["HF_TOKEN"]
    )
    print("JOB_ID:", job.job_id)
except Exception as e:
    print("FAILED:", str(e))
