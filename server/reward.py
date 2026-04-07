"""
Reward function for the SQL Debug Environment.

Reward is computed at every step (not just end of episode).
This provides dense, meaningful signal for RL training.

Reward components:
- correctness:      0.0–0.6  (row-level match vs expected)
- efficiency:       0.0–0.2  (bonus for solving quickly)  
- syntax_progress:  0.0–0.1  (valid SQL even if wrong content)
- schema_bonus:     0.0–0.1  (correct tables/columns referenced)
- penalty:          0.0 to 0.2  (deduction for bad actions)

Total range: 0.0 to 1.0 (clamped to [0.0, 1.0])
"""
from typing import Optional, List, Dict, Any
from .models import SQLDebugReward


def compute_reward(
    action_type: str,
    query_result: Optional[Dict[str, Any]],
    grade_score: float,
    steps_taken: int,
    max_steps: int,
    previous_best_score: float,
    schema_tables: List[str],
    submitted_query: Optional[str] = None,
) -> SQLDebugReward:
    """
    Compute the full reward for a step.

    Args:
    action_type: The action taken this step
    query_result: Result dict from EpisodeDatabase.execute_query()
    grade_score: 0.0-1.0 score from task grader
    steps_taken: How many steps have been used (1-indexed)
    max_steps: Maximum steps for this task
    previous_best_score: Best grade score seen so far
    schema_tables: List of valid table names in this task's DB
    submitted_query: The SQL query string (if action was submit_query)
    """

    correctness = 0.0
    efficiency = 0.0
    syntax_progress = 0.0
    schema_bonus = 0.0
    penalty = 0.0  # deduction magnitude (non-negative)

    if action_type == "submit_query":
        # Correctness: primary signal
        correctness = min(0.6, grade_score * 0.6)

        # Syntax progress: reward for at least getting a valid query
        if query_result and query_result.get("success"):
            syntax_progress = 0.1
        elif query_result and not query_result.get("success"):
            # Partially reward if it's getting closer (fewer errors)
            error = query_result.get("error_message", "")
            if "no such column" in error.lower():
                syntax_progress = 0.03  # Structure is right but wrong column
            elif "no such table" in error.lower():
                syntax_progress = 0.01
            else:
                syntax_progress = 0.0

        # Schema bonus: correct table references
        if submitted_query and schema_tables:
            query_upper = submitted_query.upper()
            tables_referenced = sum(
                1 for t in schema_tables if t.upper() in query_upper
            )
            schema_bonus = min(0.1, (tables_referenced / len(schema_tables)) * 0.1)

        # Efficiency bonus: reward solving with fewer steps
        if grade_score >= 0.95:  # Near-perfect solution
            steps_fraction = steps_taken / max_steps
            if steps_fraction <= 0.3:
                efficiency = 0.2
            elif steps_fraction <= 0.5:
                efficiency = 0.15
            elif steps_fraction <= 0.7:
                efficiency = 0.1
            else:
                efficiency = 0.05

        # Penalty: if score went DOWN from previous best (regressed)
        if grade_score < previous_best_score - 0.1:
            penalty = 0.05

    elif action_type == "reset_query":
        # Penalize resetting — agent should be making progress
        penalty = 0.05

    elif action_type in ("inspect_schema", "inspect_error", "inspect_sample"):
        # Free information actions — small positive for using schema info
        # (encourages agents to explore rather than blindly guess)
        syntax_progress = 0.01

    # Penalty: approaching step limit (urgency signal)
    steps_remaining = max_steps - steps_taken
    if steps_remaining <= 2 and grade_score < 0.5:
        penalty += 0.03

    total_raw = correctness + efficiency + syntax_progress + schema_bonus - penalty
    total = round(max(0.0, min(1.0, total_raw)), 4)

    breakdown = (
        f"correctness={correctness:.3f} + "
        f"efficiency={efficiency:.3f} + "
        f"syntax={syntax_progress:.3f} + "
        f"schema={schema_bonus:.3f} + "
        f"penalty={penalty:.3f} = {total:.4f}"
    )

    return SQLDebugReward(
        value=total,
        correctness=correctness,
        efficiency=efficiency,
        syntax_progress=syntax_progress,
        schema_bonus=schema_bonus,
        penalty=penalty,
        breakdown=breakdown
    )

