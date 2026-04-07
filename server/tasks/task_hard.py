"""
TASK 3 — HARD: Multi-bug + Optimization
Difficulty: Hard
Bug types: 
  1. Correlated subquery returns wrong scope
  2. Window function partition incorrect
  3. CTE has circular logic bug
  4. Off-by-one in date range
  5. Missing DISTINCT causing row duplication
Max steps: 30
Expected baseline model score: 0.0-0.3 (frontier models barely pass)
"""
from typing import List, Dict, Any
from .base import BaseTask


class HardTask(BaseTask):
    """
    Scenario: SaaS product analytics — find users who:
    1. Signed up in Q1 2023 (Jan 1 – Mar 31)
    2. Made at least 2 purchases in their first 30 days
    3. Return their: user_id, username, signup_date, 
                     first_purchase_date, days_to_first_purchase,
                     purchases_in_first_30_days, total_lifetime_value
    
    Bugs:
    1. Date range is '>= 2023-01-01 AND < 2023-04-01' but query uses '<= 2023-03-31' 
       (off by 1 for timestamps — in SQLite string comparison this is actually fine, 
        but the REAL bug is the upper bound uses wrong column: filters on purchase_date 
        instead of signup_date in the CTE)
    2. The window function for running total uses PARTITION BY user_id but 
       ORDER BY is missing — gives wrong cumulative values
    3. HAVING clause uses COUNT(*) but should use COUNT(DISTINCT purchase_id) 
       due to JOIN multiplication
    4. The subquery for first_purchase_date is not correlated properly 
       (missing WHERE p.user_id = u.id)
    5. days_to_first_purchase calculation uses wrong date subtraction direction
    """

    @property
    def task_id(self) -> str:
        return "hard_multi_bug"

    @property
    def name(self) -> str:
        return "SaaS Cohort Activation Report — Multi-Bug Fix"

    @property
    def difficulty(self) -> str:
        return "hard"

    @property
    def description(self) -> str:
        return """You are debugging a SaaS product analytics query.

The query should identify "activated users": users who signed up in Q1 2023 
AND made at least 2 purchases within their first 30 days of signup.

For each activated user, return:
- user_id (INTEGER)
- username (TEXT)
- signup_date (TEXT, YYYY-MM-DD)
- first_purchase_date (TEXT, YYYY-MM-DD)
- days_to_first_purchase (INTEGER, how many days after signup they first purchased)
- purchases_in_first_30_days (INTEGER)
- total_lifetime_value (REAL, sum of all their purchases ever, rounded to 2 dp)

Results ordered by total_lifetime_value DESC.

The query has FIVE bugs — some are logic errors, one is a missing correlation 
in a subquery, one is an incorrect window function, one causes row duplication.
You must find and fix all of them to get the correct result.

Q1 2023 = signup_date >= '2023-01-01' AND signup_date <= '2023-03-31'"""

    @property
    def expected_output_description(self) -> str:
        return "2 rows: users who made 2+ purchases in first 30 days. Maya Torres first (higher LTV), then James Osei."

    @property
    def broken_query(self) -> str:
        return """WITH q1_users AS (
    SELECT DISTINCT u.id, u.username, u.signup_date
    FROM users u
    JOIN purchases p ON u.id = p.user_id
    WHERE u.signup_date >= '2023-01-01' 
      AND u.signup_date <= '2023-03-31'
      AND p.purchase_date <= '2023-03-31'
),
user_purchase_stats AS (
    SELECT 
        q.id AS user_id,
        q.username,
        q.signup_date,
        (SELECT MIN(purchase_date) FROM purchases WHERE amount > 0) AS first_purchase_date,
        COUNT(*) AS purchases_in_first_30_days,
        SUM(SUM(p.amount)) OVER (PARTITION BY q.id) AS total_lifetime_value
    FROM q1_users q
    JOIN purchases p ON q.id = p.user_id
    WHERE julianday(p.purchase_date) - julianday(q.signup_date) <= 30
    GROUP BY q.id, q.username, q.signup_date
)
SELECT 
    user_id,
    username,
    signup_date,
    first_purchase_date,
    CAST(julianday(q1_users.signup_date) - julianday(first_purchase_date) AS INTEGER) AS days_to_first_purchase,
    purchases_in_first_30_days,
    ROUND(total_lifetime_value, 2) AS total_lifetime_value
FROM user_purchase_stats
WHERE purchases_in_first_30_days >= 2
ORDER BY total_lifetime_value DESC"""

    @property
    def schema_sql(self) -> str:
        return """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL,
    email TEXT UNIQUE,
    signup_date TEXT NOT NULL,
    plan TEXT DEFAULT 'free'
);

CREATE TABLE purchases (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    amount REAL NOT NULL,
    purchase_date TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
)"""

    @property
    def seed_data_sql(self) -> str:
        return """
INSERT INTO users VALUES (1,'maya_torres','maya@ex.com','2023-01-15','pro');
INSERT INTO users VALUES (2,'james_osei','james@ex.com','2023-02-10','pro');
INSERT INTO users VALUES (3,'sophie_liang','sophie@ex.com','2023-03-05','free');
INSERT INTO users VALUES (4,'raj_mehta','raj@ex.com','2023-06-01','free');
INSERT INTO users VALUES (5,'anna_kovacs','anna@ex.com','2022-12-20','pro');

-- Maya: 2 purchases in first 30 days (days 5 and 18), more later
INSERT INTO purchases VALUES (1,1,'Pro Plan',99.00,'2023-01-20');
INSERT INTO purchases VALUES (2,1,'Add-on Pack',29.00,'2023-02-02');
INSERT INTO purchases VALUES (3,1,'Pro Renewal',99.00,'2023-04-15');
INSERT INTO purchases VALUES (4,1,'Consulting',150.00,'2023-07-01');

-- James: 2 purchases in first 30 days (days 3 and 25)
INSERT INTO purchases VALUES (5,2,'Starter Plan',49.00,'2023-02-13');
INSERT INTO purchases VALUES (6,2,'Storage Add-on',19.00,'2023-03-07');
INSERT INTO purchases VALUES (7,2,'Starter Renewal',49.00,'2023-05-10');

-- Sophie: only 1 purchase in first 30 days (should NOT qualify)
INSERT INTO purchases VALUES (8,3,'Free Trial Upgrade',9.00,'2023-03-10');
INSERT INTO purchases VALUES (9,3,'Pro Plan',99.00,'2023-04-20');

-- Raj: signed up Q2, not Q1 (should NOT qualify)
INSERT INTO purchases VALUES (10,4,'Starter Plan',49.00,'2023-06-05');
INSERT INTO purchases VALUES (11,4,'Add-on',19.00,'2023-06-10');

-- Anna: signed up Q4 2022, not Q1 2023 (should NOT qualify)
INSERT INTO purchases VALUES (12,5,'Pro Plan',99.00,'2023-01-01');
INSERT INTO purchases VALUES (13,5,'Consulting',150.00,'2023-03-15')"""

    @property
    def expected_output(self) -> List[Dict[str, Any]]:
        # Maya: signup 2023-01-15, first purchase 2023-01-20 (day 5)
        #   purchases in 30 days: Jan-20 (day5), Feb-02 (day18) = 2 ✓
        #   total LTV: 99+29+99+150 = 377
        # James: signup 2023-02-10, first purchase 2023-02-13 (day 3)
        #   purchases in 30 days: Feb-13 (day3), Mar-07 (day25) = 2 ✓  
        #   total LTV: 49+19+49 = 117
        return [
            {
                "user_id": 1,
                "username": "maya_torres",
                "signup_date": "2023-01-15",
                "first_purchase_date": "2023-01-20",
                "days_to_first_purchase": 5,
                "purchases_in_first_30_days": 2,
                "total_lifetime_value": 377.00
            },
            {
                "user_id": 2,
                "username": "james_osei",
                "signup_date": "2023-02-10",
                "first_purchase_date": "2023-02-13",
                "days_to_first_purchase": 3,
                "purchases_in_first_30_days": 2,
                "total_lifetime_value": 117.00
            }
        ]

    @property
    def hint(self) -> str:
        return "Hint: There are 5 bugs total. Check: (1) the subquery for first_purchase_date needs a WHERE correlation, (2) the date subtraction direction for days_to_first_purchase, (3) COUNT(*) vs COUNT(DISTINCT) when JOINs can multiply rows, (4) window functions need ORDER BY for meaningful results, (5) the q1_users CTE may be filtering on the wrong table's date column."

