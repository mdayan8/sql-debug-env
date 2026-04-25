import unittest

from fastapi.testclient import TestClient

from server.main import app


class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.session_id = "test-session"

    def test_health_and_tasks(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

        r = self.client.get("/tasks")
        self.assertEqual(r.status_code, 200)
        tasks = r.json()["tasks"]
        task_ids = {t["task_id"] for t in tasks}
        self.assertIn("easy_syntax_fix", task_ids)
        self.assertIn("medium_logic_fix", task_ids)
        self.assertIn("hard_multi_bug", task_ids)
        self.assertIn("hard_finance_explosion", task_ids)

    def test_reset_step_state_roundtrip(self) -> None:
        r = self.client.post(
            "/reset",
            headers={"x-session-id": self.session_id},
            json={"task_id": "easy_syntax_fix"},
        )
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertEqual(payload["observation"]["task_id"], "easy_syntax_fix")
        self.assertEqual(payload["observation"]["steps_taken"], 0)

        r = self.client.post(
            "/step",
            headers={"x-session-id": self.session_id},
            json={"action": {"action_type": "inspect_schema"}},
        )
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertEqual(payload["observation"]["steps_taken"], 1)
        self.assertEqual(payload["observation"]["last_action_type"], "inspect_schema")
        self.assertIsInstance(payload["reward"], float)

        r = self.client.get("/state", headers={"x-session-id": self.session_id})
        self.assertEqual(r.status_code, 200)
        state = r.json()
        self.assertEqual(state["task_id"], "easy_syntax_fix")
        self.assertEqual(state["steps_taken"], 1)

    def test_step_with_review_rejects_non_select(self) -> None:
        self.client.post(
            "/reset",
            headers={"x-session-id": self.session_id},
            json={"task_id": "easy_syntax_fix"},
        )

        r = self.client.post(
            "/step_with_review",
            headers={"x-session-id": self.session_id},
            json={"action": {"action_type": "submit_query", "query": "DELETE FROM customers;"}},
        )
        self.assertEqual(r.status_code, 200)
        payload = r.json()
        self.assertEqual(payload["info"]["review_rejected"], True)
        self.assertEqual(payload["reward"], 0.001)
        self.assertEqual(payload["observation"]["last_action_type"], "review_rejected")


if __name__ == "__main__":
    unittest.main()

