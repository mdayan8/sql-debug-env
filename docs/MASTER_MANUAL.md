# 🏆 SQL Debug Env: The Ultimate Master Manual
> **Comprehensive Wiki & Technical Bible for the Meta PyTorch × OpenEnv Hackathon**

---

## 📖 Table of Contents
1. [The "Simple" Concept](#1-the-simple-concept)
2. [Architecture: How the Machine Works](#2-architecture-how-the-machine-works)
3. [The Industry Benchmark: Spider vs. BIRD vs. YOU](#3-the-industry-benchmark-spider-vs-bird-vs-you)
4. [Deep-Dive: The Codebase Map](#4-deep-dive-the-codebase-map)
5. [The Science: GRPO & Reinforcement Learning](#5-the-science-grpo--reinforcement-learning)
6. [The "Day in the Life" of a SQL Query](#6-the-day-in-the-life-of-a-sql-query)
7. [Current Project Status & Roadmap](#7-current-project-status--roadmap)
8. [Live Spider Evaluation (The "Ultimate Proof")](#8-live-spider-evaluation-the-ultimate-proof)
9. [Winning the Q&A (The Cheat Sheet)](#9-winning-the-qa-the-cheat-sheet)

---

## 1. The "Simple" Concept
Imagine you are a teacher. You have a student (the **AI**) who is good at English but bad at Math (the **SQL**).
Instead of just giving the student a textbook, you put them in a room with a calculator (the **Database**).
The student tries a problem, uses the calculator, sees the answer is wrong, and tries again.
**You have built the Room, the Calculator, and the Reward System (the "Stars") that makes the student smarter.**

---

## 2. Architecture: How the Machine Works
The project is split into two main "Brains":

### A. The Environment (The Body / server/)
This is the "physical world" where the SQL lives.
- **FastAPI:** The "telephone" that lets the AI talk to the database.
- **SQLite:** The "sandbox" where queries are actually run.
- **Graders:** The "judge" that compares the result of the AI's query to the "truth."

### B. The Agent (The Brain / grpo_train.py)
This is the intelligence that is trying to learn.
- **Model (Qwen2.5-Coder):** The actual neural network.
- **GRPO Logic:** The mathematical formula that tells the model: *"Fix #3 was better than Fix #1, change your weights to be more like #3."*

---

## 3. The Industry Benchmark: Spider vs. BIRD vs. YOU
**Judge Question:** *"Why should we use your environment instead of existing datasets like Spider?"*

| Feature | Spider / BIRD (Standard) | **SQL Debug Env (YOU)** |
| :--- | :--- | :--- |
| **Task Type** | One-Shot Generation | **Iterative Debugging** |
| **Feedback** | None (Static) | **Live Database Feedback** |
| **Difficulty** | High-level Text-to-SQL | **Low-level Logic/Syntax Fixes** |
| **Evaluation** | Fuzzy (String matching) | **Deterministic (Row matching)** |

**The Reference:** Your project is inspired by the **DeepSeek R1** and **OpenAI o1** reasoning models. You are applying their "Reinforcement Learning from Feedback" (RLHF) philosophy to the niche world of SQL engineering.

---

## 4. Deep-Dive: The Codebase Map

| File | What is it? | Why is it here? |
| :--- | :--- | :--- |
| **`server/main.py`** | The Heart | Acts as the API server. It handles `/reset` (new game) and `/step` (make a move). |
| **`server/env.py`** | The World | Manages the session state. It knows if the user is in Task 1 or Task 3. |
| **`server/database.py`** | The Sandbox | Creates temporary SQLite databases in memory so the AI can't break anything. |
| **`server/reward.py`** | The Scorekeeper | Calculates the "Reward" (0.0 to 1.0). It checks syntax, efficiency, and correctness. |
| **`grpo_train.py`** | The Trainer | The script that actually "upgrades" the AI's brain using RL. |
| **`inference.py`** | The Test | A simple script to see how smart the AI is *right now* before training. |
| **`openenv.yaml`** | The ID Card | Tells the hackathon platform how to connect to your project. |

---

## 5. The Science: GRPO & Reinforcement Learning
If a judge asks: *"How does it learn?"*

### The Old Way: SFT (Supervised Fine-Tuning)
- You show the AI 1,000 "Correct" answers.
- **Problem:** The AI just memorizes. It doesn't learn how to "debug" when it sees a new error.

### Your Way: GRPO (Group Relative Policy Optimization)
- **Step 1:** The AI looks at a broken query.
- **Step 2:** It generates **4 different ways** to fix it (a "Group").
- **Step 3:** We run all 4 in the database and get 4 scores.
- **Step 4:** We compare them. We tell the AI: *"Compared to your other 3 tries, your 2nd try was the best. Do more of that."*
- **Innovation:** This is **"Self-Generated Reasoning."** The AI is its own teacher.

---

## 6. The "Day in the Life" of a SQL Query
Follow a query from start to finish:
1. **The Prompt:** "Fix this query: SELECT * FROM userss (typo)."
2. **The Reviewer:** Your `reviewer_check` in `main.py` looks at it. If it sees `DROP TABLE`, it rejects it immediately.
3. **The Sandbox:** The query is run in a private SQLite memory space.
4. **The Comparison:** The system runs the "Correct" query in the background. It compares the rows. 
5. **The Reward:** If the rows match, the AI gets `+1.0`. If they don't, but the syntax is valid, it gets `+0.2`.
6. **The Memory:** The AI updates its "Weights" (its digital brain) to remember this success.

---

## 7. Current Project Status & Roadmap
**Project Completion: 95%**

### ✅ Completed:
- Core FastAPI Server & SQLite Sandbox.
- 3 Realistic SQL Debugging Tasks (Easy, Medium, Hard).
- Multi-Agent Reviewer Layer.
- GRPO Training Script verified on Apple Silicon (M2).
- Smoke Test verified (Handshake is 100% working).

### ⏳ Remaining (For Hackathon Site):
- Scale to **Qwen 7B/14B** on A100 GPUs.
- Connect **Weights & Biases (WandB)** for the live presentation curve.

---

## 8. Live Spider Evaluation (The "Ultimate Proof")
**How to show the judges your agent can handle real-world academic benchmarks:**

1.  **Launch the Spider Task:**
    Run `/reset` with the `spider_cross_eval` task ID (handled by `server/tasks/task_spider.py`).
2.  **The "Blind Test":**
    Ask a judge to pick a random SQL query from the **Spider dev set**.
3.  **Introduce a Bug:**
    Delete a semicolon, misspell a JOIN, or remove a WHERE clause.
4.  **The Demonstration:**
    Run `inference.py` on that broken Spider query.
    **The Result:** The agent will use its trained GRPO weights to analyze the error, inspect the Spider schema, and return the fix.

**Why this wins:** You are showing that your environment isn't a "closed loop." It can ingest and solve the industry's hardest academic benchmark in real-time.

---

## 9. Winning the Q&A (The Cheat Sheet)

**Q: "Why SQLite?"**
> *"Because it's the world's most used DB. If the agent can reason in SQLite, it can reason in PostgreSQL. I built a 'Simulator' that is DB-agnostic."*

**Q: "What makes this 'Multi-Agent'?"**
> *"I have two roles: The **Fixer** (the LLM) and the **Reviewer** (the guardrail logic). They interact to ensure every query is safe and syntactically sound before execution."*

---
**This manual is your secret weapon. Read it, understand it, and you will own the stage.** 🚀
