import unittest

from server.reward import compute_reward


class TestReward(unittest.TestCase):
    def test_submit_query_perfect_reward(self):
        reward = compute_reward(
            action_type="submit_query",
            query_result={"success": True},
            grade_score=0.999,
            steps_taken=1,
            max_steps=10,
            previous_best_score=0.001,
            schema_tables=["t1", "t2"],
            submitted_query="SELECT * FROM t1 JOIN t2",
        )
        self.assertAlmostEqual(reward.value, 0.999, places=4)

    def test_reset_query_penalty(self):
        reward = compute_reward(
            action_type="reset_query",
            query_result=None,
            grade_score=0.001,
            steps_taken=1,
            max_steps=10,
            previous_best_score=0.001,
            schema_tables=[],
            submitted_query=None,
        )
        self.assertAlmostEqual(reward.value, 0.001, places=4)

    def test_inspect_schema_urgency_penalty(self):
        # Make steps_remaining <= 2 and grade_score < 0.5 to trigger urgency penalty.
        reward = compute_reward(
            action_type="inspect_schema",
            query_result=None,
            grade_score=0.001,
            steps_taken=8,
            max_steps=9,
            previous_best_score=0.001,
            schema_tables=[],
            submitted_query=None,
        )
        # syntax_progress=0.01, penalty=0.03 => total_raw=-0.02, clamped to strict min
        self.assertAlmostEqual(reward.value, 0.001, places=4)


if __name__ == "__main__":
    unittest.main()

