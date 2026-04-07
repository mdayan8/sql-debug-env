"""Base class for all SQL Debug tasks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class BaseTask(ABC):
    _MIN_STRICT_SCORE = 0.001
    _MAX_STRICT_SCORE = 0.999

    def _strict_score(self, score: float) -> float:
        """Keep task score strictly inside (0, 1) for validator compatibility."""
        return round(
            min(self._MAX_STRICT_SCORE, max(self._MIN_STRICT_SCORE, score)),
            3,
        )

    """
    Abstract base for all tasks.

    Each task defines:
    - A broken SQL query (the one the agent must fix)
    - A database schema (SQLite CREATE TABLE statements)
    - Seed data (INSERT statements, deterministic)
    - Expected output (what the correct query should return)
    - A grader (compares agent output vs expected)
    - Metadata (id, name, difficulty, description, hint)
    """

    @property
    @abstractmethod
    def task_id(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def difficulty(self) -> str:
        pass  # "easy", "medium", "hard"

    @property
    @abstractmethod
    def description(self) -> str:
        """Natural language description given to the agent."""
        pass

    @property
    @abstractmethod
    def expected_output_description(self) -> str:
        """Describes what the correct output looks like."""
        pass

    @property
    @abstractmethod
    def broken_query(self) -> str:
        """The SQL query with bugs that the agent must fix."""
        pass

    @property
    @abstractmethod
    def schema_sql(self) -> str:
        """SQLite CREATE TABLE statements."""
        pass

    @property
    @abstractmethod
    def seed_data_sql(self) -> str:
        """INSERT statements for deterministic test data."""
        pass

    @property
    @abstractmethod
    def expected_output(self) -> List[Dict[str, Any]]:
        """
        The exact rows the correct query should return.
        Used by the grader to score the agent's output.
        Must be deterministic and match seed_data_sql exactly.
        """
        pass

    @property
    def hint(self) -> str:
        """Optional hint shown after N steps. Override in subclass."""
        return ""

    @property
    def max_steps(self) -> int:
        """Maximum steps for this task."""
        return {"easy": 10, "medium": 20, "hard": 30}.get(self.difficulty, 20)

    def grade(self, actual_rows: Optional[List[Dict[str, Any]]]) -> float:
        """
        Grade the agent's query output vs expected output.
        Returns a score 0.0-1.0.

        Scoring:
        - 1.0: exact match (correct rows, correct order if ORDER BY expected)
        - 0.5-0.9: partial match (subset of correct rows, or wrong order)
        - 0.1-0.4: syntactically valid but wrong content
        - 0.0: null result, syntax error, or empty when non-empty expected
        """
        if not actual_rows:
            return self._strict_score(0.0)

        expected = self.expected_output

        if not expected:
            # Expected empty result
            return self._strict_score(1.0 if len(actual_rows) == 0 else 0.0)

        # Exact row count match
        if len(actual_rows) != len(expected):
            # Partial credit for getting some rows right
            overlap = self._count_matching_rows(actual_rows, expected)
            return self._strict_score(min(0.5, overlap / max(len(expected), 1) * 0.5))

        # Check row-by-row match (order-sensitive if task requires it)
        matching = self._count_matching_rows(actual_rows, expected)
        score = matching / len(expected)

        # Check column names match
        if actual_rows and expected:
            actual_cols = set(actual_rows[0].keys())
            expected_cols = set(expected[0].keys())
            if actual_cols != expected_cols:
                score *= 0.7  # Penalty for wrong columns

        return self._strict_score(score)

    def _count_matching_rows(
        self,
        actual: List[Dict[str, Any]],
        expected: List[Dict[str, Any]]
    ) -> int:
        """Count how many actual rows match expected rows (normalized comparison)."""
        matches = 0
        expected_normalized = [self._normalize_row(r) for r in expected]

        for i, actual_row in enumerate(actual):
            actual_norm = self._normalize_row(actual_row)
            if i < len(expected_normalized):
                # Positional match (respects ORDER BY)
                if actual_norm == expected_normalized[i]:
                    matches += 1
            else:
                # Extra rows don't count
                break

        return matches

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a row for comparison: lowercase keys, string-normalize values."""
        normalized = {}
        for k, v in row.items():
            key = k.lower().strip()
            if isinstance(v, float):
                val = round(v, 2)
            elif isinstance(v, str):
                val = v.strip()
            else:
                val = v
            normalized[key] = val
        return normalized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "difficulty": self.difficulty,
            "description": self.description,
            "expected_output_description": self.expected_output_description,
            "broken_query": self.broken_query,
            "max_steps": self.max_steps,
            "hint": self.hint
        }

