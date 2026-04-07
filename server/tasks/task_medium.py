"""
TASK 2 — MEDIUM: Logic Error Fix
Difficulty: Medium  
Bug types: Wrong JOIN type causing missing rows, incorrect aggregation logic,
           missing HAVING clause, wrong date filter
Max steps: 20
Expected baseline model score: 0.3-0.6
"""
from typing import List, Dict, Any
from .base import BaseTask


class MediumTask(BaseTask):
    """
    Scenario: HR analytics team wants monthly headcount and average salary 
    by department for the current year, including departments with zero employees 
    (i.e., departments that exist but no one joined this year).
    
    Bugs:
    1. Uses INNER JOIN instead of LEFT JOIN — excludes empty departments
    2. Uses AVG(salary) over all employees instead of only those who joined this year
    3. Missing: the date filter for 'this year' is applied in WHERE, breaking the LEFT JOIN
       (should be in ON clause or use CASE)
    4. GROUP BY missing department_id (ambiguous grouping)
    """

    @property
    def task_id(self) -> str:
        return "medium_logic_fix"

    @property
    def name(self) -> str:
        return "Department Headcount Report — Logic Error Fix"

    @property
    def difficulty(self) -> str:
        return "medium"

    @property
    def description(self) -> str:
        return """You are debugging a HR analytics SQL query.

The query should produce a monthly department headcount report showing:
- department_name
- headcount: number of employees who joined IN 2023
- avg_salary: average salary of employees who joined IN 2023
- All departments must appear, even those with 0 new hires in 2023

The current query has 3 logic bugs:
1. It uses the wrong JOIN type, which silently drops departments with no 2023 hires
2. The WHERE clause on hire_date breaks the outer join semantics
3. The AVG calculation includes employees from all years, not just 2023

Fix these logic errors. The result should be ordered by department_name ascending."""

    @property
    def expected_output_description(self) -> str:
        return "4 rows (all departments), headcount=0 for 'Legal', correct avg_salary only from 2023 hires."

    @property
    def broken_query(self) -> str:
        return """SELECT 
    d.name AS department_name,
    COUNT(e.id) AS headcount,
    ROUND(AVG(e.salary), 2) AS avg_salary
FROM departments d
INNER JOIN employees e ON d.id = e.department_id
WHERE strftime('%Y', e.hire_date) = '2023'
GROUP BY d.name
ORDER BY department_name ASC"""

    @property
    def schema_sql(self) -> str:
        return """
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    budget REAL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL,
    salary REAL NOT NULL,
    hire_date TEXT NOT NULL,
    FOREIGN KEY (department_id) REFERENCES departments(id)
)"""

    @property
    def seed_data_sql(self) -> str:
        return """
INSERT INTO departments VALUES (1,'Engineering',500000);
INSERT INTO departments VALUES (2,'Marketing',200000);
INSERT INTO departments VALUES (3,'Sales',300000);
INSERT INTO departments VALUES (4,'Legal',150000);

INSERT INTO employees VALUES (1,'Ana Lima',1,95000,'2023-03-15');
INSERT INTO employees VALUES (2,'Ben Sharma',1,102000,'2023-06-01');
INSERT INTO employees VALUES (3,'Chris Wang',1,88000,'2022-01-10');
INSERT INTO employees VALUES (4,'Diana Patel',2,72000,'2023-04-20');
INSERT INTO employees VALUES (5,'Erik Johnson',2,68000,'2022-11-05');
INSERT INTO employees VALUES (6,'Fatima Al-Hassan',3,55000,'2023-01-08');
INSERT INTO employees VALUES (7,'George Okafor',3,61000,'2023-07-22');
INSERT INTO employees VALUES (8,'Hannah Kim',3,58000,'2022-05-30');
INSERT INTO employees VALUES (9,'Ivan Petrov',1,91000,'2022-08-14')"""

    @property
    def expected_output(self) -> List[Dict[str, Any]]:
        # Engineering 2023 hires: Ana 95000, Ben 102000 → count=2, avg=98500
        # Marketing 2023 hires: Diana 72000 → count=1, avg=72000
        # Sales 2023 hires: Fatima 55000, George 61000 → count=2, avg=58000
        # Legal 2023 hires: none → count=0, avg=NULL
        return [
            {"department_name": "Engineering", "headcount": 2, "avg_salary": 98500.00},
            {"department_name": "Legal", "headcount": 0, "avg_salary": None},
            {"department_name": "Marketing", "headcount": 1, "avg_salary": 72000.00},
            {"department_name": "Sales", "headcount": 2, "avg_salary": 58000.00},
        ]

    @property
    def hint(self) -> str:
        return "Hint: When you want ALL rows from the left table even when there's no match on the right, think about which JOIN type preserves those rows. Also, WHERE on a nullable column after a join changes join semantics — consider moving that condition."


class MediumTaskGrader:
    """
    Custom grader for medium task — handles NULL comparison.
    """
    @staticmethod
    def grade(actual: List[Dict]) -> float:
        if not actual or len(actual) != 4:
            return 0.0

        # Sort both by dept name for comparison
        actual_sorted = sorted(actual, key=lambda r: r.get("department_name", ""))
        expected = [
            {"department_name": "Engineering", "headcount": 2, "avg_salary": 98500.00},
            {"department_name": "Legal", "headcount": 0, "avg_salary": None},
            {"department_name": "Marketing", "headcount": 1, "avg_salary": 72000.00},
            {"department_name": "Sales", "headcount": 2, "avg_salary": 58000.00},
        ]

        matches = 0
        for a, e in zip(actual_sorted, expected):
            dept_ok = str(a.get("department_name","")).lower() == str(e["department_name"]).lower()
            count_ok = int(a.get("headcount", -1)) == e["headcount"]

            e_salary = e["avg_salary"]
            a_salary = a.get("avg_salary")
            if e_salary is None:
                salary_ok = a_salary is None or a_salary == 0
            else:
                try:
                    salary_ok = abs(float(a_salary) - float(e_salary)) < 1.0
                except (TypeError, ValueError):
                    salary_ok = False

            if dept_ok and count_ok and salary_ok:
                matches += 1

        return round(matches / 4, 3)

