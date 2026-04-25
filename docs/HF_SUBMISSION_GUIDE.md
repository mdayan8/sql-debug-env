# 🚀 Hugging Face Space: Deployment Guide

To meet the "Minimum Submission Requirements," you must host your environment on Hugging Face. Here is how to do it in 5 minutes:

### 1. Create the Space
1.  Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  Name it: `sql-debug-env`.
3.  SDK: Select **Docker**.
4.  Template: **Blank**.

### 2. Upload these files to the Space
You only need to upload these files from your project:
*   `server/` (The whole folder)
*   `Dockerfile` (Use the one in your root)
*   `requirements.txt`
*   `openenv.yaml`

### 3. Add Secrets
In the Space settings, add your `HF_TOKEN` as a Secret if you want to use gated models, but for the **Environment**, no secrets are needed.

### 4. Link it in your README
Once the Space is running, copy the URL (e.g., `https://huggingface.co/spaces/mdayan/sql-debug-env`) and paste it into the **Results** section of your `README.md`.

---

### 🏁 Why this wins:
By putting the **Environment** in a Space and the **Training Logs** in WandB, you are showing the judges a complete "Production AI Lifecycle." Most teams will just upload a Python file. You are uploading a **Platform.**
