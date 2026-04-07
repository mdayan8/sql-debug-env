"""
Core SQL Debug Environment.
Manages episode state, delegates to tasks and reward function.
"""
import uuid
import asyncio
from typing import Optional, Dict, Any, List
from .models import (
    SQLDebugAction, SQLDebugObservation, SQLDebugReward,
    EpisodeState, ActionType, QueryResult, SchemaInfo
)
from .database import EpisodeDatabase
from .reward import compute_reward
from .tasks.task_easy import EasyTask
from .tasks.task_medium import MediumTask, MediumTaskGrader
from .tasks.task_hard import HardTask


TASKS = {
    "easy_syntax_fix": EasyTask(),
    "medium_logic_fix": MediumTask(),
    "hard_multi_bug": HardTask(),
}


class SQLDebugEnv:
    """
    The SQL Debug Environment.
    Manages one active episode at a time per session.
    Thread-safe for concurrent sessions via instance-per-session pattern.
    """

    def __init__(self, task_id: str = "easy_syntax_fix"):
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._db: Optional[EpisodeDatabase] = None
        self._state: Optional[EpisodeState] = None
        self._lock = asyncio.Lock()

    async def reset(self) -> tuple[SQLDebugObservation, Dict]:
        """Reset environment to initial state. Returns (observation, info)."""
        async with self._lock:
            # Close previous DB if exists
            if self._db:
                self._db.close()

            # Fresh DB
            self._db = EpisodeDatabase(
                task_id=self.task.task_id,
                schema_sql=self.task.schema_sql,
                seed_data_sql=self.task.seed_data_sql
            )

            # Fresh state
            self._state = EpisodeState(
                task_id=self.task.task_id,
                task_difficulty=self.task.difficulty,
                original_query=self.task.broken_query,
                current_query=None,
                best_score_so_far=0.0,
                steps_taken=0,
                max_steps=self.task.max_steps,
                action_history=[],
                reward_history=[],
                is_done=False,
                success=False,
                db_schema=self._db.get_schema()
            )

            obs = SQLDebugObservation(
                task_id=self.task.task_id,
                task_description=self.task.description,
                original_query=self.task.broken_query,
                current_query=None,
                expected_description=self.task.expected_output_description,
                last_action_type="reset",
                last_query_result=None,
                steps_taken=0,
                steps_remaining=self.task.max_steps,
                current_score=0.0,
                schema_info=SchemaInfo(tables=self._db.get_schema()),
                is_done=False,
                success=False
            )

            return obs, {"task": self.task.to_dict()}

    async def step(self, action: SQLDebugAction) -> tuple[SQLDebugObservation, float, bool, Dict]:
        """
        Execute one action.
        Returns (observation, reward_value, done, info)
        """
        async with self._lock:
            if self._state is None:
                raise RuntimeError("Call reset() before step()")

            if self._state.is_done:
                raise RuntimeError("Episode is done. Call reset() to start new episode.")

            self._state.steps_taken += 1
            steps_taken = self._state.steps_taken

            query_result_raw = None
            prev_best_score = self._state.best_score_so_far
            grade_score = self._state.best_score_so_far
            schema_info = None
            error_details = None
            sample_rows = None
            hint = None

            # --- Execute action ---
            if action.action_type == ActionType.SUBMIT_QUERY:
                if not action.query:
                    raise ValueError("query is required for submit_query action")

                self._state.current_query = action.query
                query_result_raw = self._db.execute_query(action.query)

                # Grade the result
                actual_rows = query_result_raw.get("rows") if query_result_raw.get("success") else None

                # Use custom grader for medium task
                if self.task.task_id == "medium_logic_fix":
                    grade_score = MediumTaskGrader.grade(actual_rows or [])
                else:
                    grade_score = self.task.grade(actual_rows)

                if grade_score > self._state.best_score_so_far:
                    self._state.best_score_so_far = grade_score

            elif action.action_type == ActionType.INSPECT_SCHEMA:
                schema = self._db.get_schema()
                schema_info = SchemaInfo(tables=schema)
                grade_score = self._state.best_score_so_far

            elif action.action_type == ActionType.INSPECT_ERROR:
                # Return last error if available
                if self._state.action_history:
                    last = self._state.action_history[-1]
                    error_details = last.get("error_message", "No error recorded from last query.")
                else:
                    error_details = "No query has been submitted yet."
                grade_score = self._state.best_score_so_far

            elif action.action_type == ActionType.INSPECT_SAMPLE:
                if not action.table_name:
                    raise ValueError("table_name required for inspect_sample")
                sample_rows = self._db.get_sample_rows(action.table_name)
                grade_score = self._state.best_score_so_far

            elif action.action_type == ActionType.RESET_QUERY:
                self._state.current_query = self.task.broken_query
                grade_score = self._state.best_score_so_far

            # --- Compute reward ---
            schema_tables = list(self._db.get_schema().keys())
            reward_obj = compute_reward(
                action_type=action.action_type.value,
                query_result=query_result_raw,
                grade_score=grade_score,
                steps_taken=steps_taken,
                max_steps=self.task.max_steps,
                previous_best_score=prev_best_score,
                schema_tables=schema_tables,
                submitted_query=action.query if action.action_type == ActionType.SUBMIT_QUERY else None
            )

            # --- Check done conditions ---
            is_done = False
            success = False

            if grade_score >= 0.95:
                is_done = True
                success = True
            elif steps_taken >= self.task.max_steps:
                is_done = True
                success = self._state.best_score_so_far >= 0.5

            self._state.is_done = is_done
            self._state.success = success

            # --- Hint logic ---
            hint_threshold = 3 if self.task.difficulty == "easy" else 5
            if steps_taken >= hint_threshold:
                hint = self.task.hint

            # --- Record history ---
            self._state.action_history.append({
                "step": steps_taken,
                "action_type": action.action_type.value,
                "query": action.query,
                "grade_score": grade_score,
                "reward": reward_obj.value,
                "error_message": query_result_raw.get("error_message") if query_result_raw else None
            })
            self._state.reward_history.append(reward_obj.value)

            # --- Build observation ---
            qr = QueryResult(**query_result_raw) if query_result_raw else None

            obs = SQLDebugObservation(
                task_id=self.task.task_id,
                task_description=self.task.description,
                original_query=self.task.broken_query,
                current_query=self._state.current_query,
                expected_description=self.task.expected_output_description,
                last_action_type=action.action_type.value,
                last_query_result=qr,
                steps_taken=steps_taken,
                steps_remaining=max(0, self.task.max_steps - steps_taken),
                current_score=self._state.best_score_so_far,
                schema_info=schema_info,
                error_details=error_details,
                sample_rows=sample_rows,
                hint=hint,
                is_done=is_done,
                success=success
            )

            return obs, reward_obj.value, is_done, {
                "grade_score": grade_score,
                "reward_breakdown": reward_obj.breakdown,
                "success": success,
                "steps_taken": steps_taken
            }

    def get_state(self) -> EpisodeState:
        if self._state is None:
            raise RuntimeError("Call reset() first")
        return self._state

    def close(self):
        if self._db:
            self._db.close()
            self._db = None

