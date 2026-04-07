import unittest

from server.tasks.task_easy import EasyTask
from server.tasks.task_medium import MediumTask, MediumTaskGrader
from server.tasks.task_hard import HardTask


class TestGraders(unittest.TestCase):
    def test_easy_grade_perfect(self):
        task = EasyTask()
        score = task.grade(task.expected_output)
        self.assertAlmostEqual(score, 0.999, places=3)

    def test_hard_grade_perfect(self):
        task = HardTask()
        score = task.grade(task.expected_output)
        self.assertAlmostEqual(score, 0.999, places=3)

    def test_easy_grade_empty(self):
        task = EasyTask()
        score = task.grade(None)
        self.assertAlmostEqual(score, 0.001, places=3)

    def test_medium_grader_perfect(self):
        task = MediumTask()
        score = MediumTaskGrader.grade(task.expected_output)
        self.assertAlmostEqual(score, 0.999, places=3)

    def test_medium_grader_partial(self):
        # Flip one row's avg_salary so it no longer matches within tolerance.
        task = MediumTask()
        actual = [dict(r) for r in task.expected_output]

        # Expected avg_salary is None for "Legal". Any non-None/non-zero value should fail.
        for r in actual:
            if r["department_name"] == "Legal":
                r["avg_salary"] = 12345.0

        score = MediumTaskGrader.grade(actual)
        self.assertLess(score, 0.999)
        self.assertAlmostEqual(score, 0.75, places=3)


if __name__ == "__main__":
    unittest.main()

