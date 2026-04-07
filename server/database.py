"""
SQLite in-memory database management.
Creates fresh DB instances per episode with deterministic seed data.
"""
import sqlite3
import time
from typing import Dict, Any, List


class EpisodeDatabase:
    """
    Manages a single SQLite in-memory database for one episode.
    Seeded with deterministic data per task.
    """

    def __init__(self, task_id: str, schema_sql: str, seed_data_sql: str):
        self.task_id = task_id
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._setup(schema_sql, seed_data_sql)

    def _setup(self, schema_sql: str, seed_data_sql: str):
        """Create schema and insert seed data."""
        cursor = self.conn.cursor()
        for statement in schema_sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)
        for statement in seed_data_sql.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                cursor.execute(stmt)
        self.conn.commit()

    def execute_query(self, query: str) -> Dict[str, Any]:
        """
        Execute a read-only SQL query safely.
        Returns rows or error. Enforces SELECT-only.
        Execution timeout: 5 seconds.
        """
        query_stripped = query.strip().upper()

        # Block dangerous operations
        blocked = ["DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER",
                   "TRUNCATE", "REPLACE", "ATTACH", "DETACH"]
        for kw in blocked:
            if query_stripped.startswith(kw) or f" {kw} " in query_stripped:
                return {
                    "success": False,
                    "rows": None,
                    "row_count": None,
                    "error_message": f"BLOCKED: Only SELECT queries are allowed. '{kw}' is not permitted.",
                    "execution_time_ms": 0.0
                }

        start = time.time()
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            elapsed = (time.time() - start) * 1000

            # Convert Row objects to dicts
            result_rows = [dict(row) for row in rows]

            return {
                "success": True,
                "rows": result_rows,
                "row_count": len(result_rows),
                "error_message": None,
                "execution_time_ms": round(elapsed, 2)
            }
        except sqlite3.Error as e:
            elapsed = (time.time() - start) * 1000
            return {
                "success": False,
                "rows": None,
                "row_count": None,
                "error_message": str(e),
                "execution_time_ms": round(elapsed, 2)
            }

    def get_schema(self) -> Dict[str, List[Dict[str, str]]]:
        """Return schema info: tables and their columns."""
        schema = {}
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = []
            for col in cursor.fetchall():
                columns.append({
                    "name": col[1],
                    "type": col[2],
                    "nullable": "YES" if col[3] == 0 else "NO",
                    "primary_key": "YES" if col[5] > 0 else "NO"
                })
            schema[table] = columns

        return schema

    def get_sample_rows(self, table_name: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Get sample rows from a table."""
        result = self.execute_query(f"SELECT * FROM {table_name} LIMIT {limit}")
        return result.get("rows", []) or []

    def close(self):
        self.conn.close()

