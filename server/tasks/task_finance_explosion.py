from typing import Optional, List, Dict, Any
from .base import BaseTask

class FinanceExplosionTask(BaseTask):
    @property
    def task_id(self) -> str:
        return "hard_finance_explosion"

    @property
    def name(self) -> str:
        return "Financial Cartesian Explosion Fix"

    @property
    def expected_output(self) -> List[Dict[str, Any]]:
        return [
            {"name": "Alice", "total_orders": 300.0, "total_payments": 300.0},
            {"name": "Bob", "total_orders": 50.0, "total_payments": 50.0}
        ]

    @property
    def difficulty(self) -> str:
        return "expert"

    @property
    def description(self) -> str:
        return (
            "A financial dashboard is reporting massive revenue discrepancies. "
            "The query calculates the total order amount and total payment amount for each user. "
            "However, due to a 'Cartesian Explosion' (Fan Trap) in the JOINs, users with multiple orders "
            "and payments are having their totals multiplied exponentially. "
            "Rewrite the query using Common Table Expressions (CTEs) or Subqueries to aggregate "
            "orders and payments separately *before* joining them to the users table."
        )

    @property
    def expected_output_description(self) -> str:
        return "A table with 'name', 'total_orders', and 'total_payments'. The totals must accurately reflect the sum of orders and payments without multiplication from joins."

    @property
    def schema_sql(self) -> str:
        return """
        CREATE TABLE users (
            user_id INTEGER PRIMARY KEY,
            name TEXT
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            order_amount DECIMAL(10,2)
        );
        CREATE TABLE payments (
            payment_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            payment_amount DECIMAL(10,2)
        );
        """

    @property
    def seed_data_sql(self) -> str:
        return """
        INSERT INTO users (user_id, name) VALUES (1, 'Alice');
        INSERT INTO users (user_id, name) VALUES (2, 'Bob');
        
        -- Alice has 3 orders (Total: 300)
        INSERT INTO orders (order_id, user_id, order_amount) VALUES (101, 1, 100.00);
        INSERT INTO orders (order_id, user_id, order_amount) VALUES (102, 1, 100.00);
        INSERT INTO orders (order_id, user_id, order_amount) VALUES (103, 1, 100.00);
        
        -- Alice has 3 payments (Total: 300)
        INSERT INTO payments (payment_id, user_id, payment_amount) VALUES (201, 1, 100.00);
        INSERT INTO payments (payment_id, user_id, payment_amount) VALUES (202, 1, 100.00);
        INSERT INTO payments (payment_id, user_id, payment_amount) VALUES (203, 1, 100.00);

        -- Bob has 1 order and 1 payment
        INSERT INTO orders (order_id, user_id, order_amount) VALUES (104, 2, 50.00);
        INSERT INTO payments (payment_id, user_id, payment_amount) VALUES (204, 2, 50.00);
        """

    @property
    def broken_query(self) -> str:
        return """
        SELECT 
            u.name, 
            SUM(o.order_amount) as total_orders, 
            SUM(p.payment_amount) as total_payments
        FROM users u
        LEFT JOIN orders o ON u.user_id = o.user_id
        LEFT JOIN payments p ON u.user_id = p.user_id
        GROUP BY u.name
        ORDER BY u.name;
        """

    @property
    def max_steps(self) -> int:
        return 12

    @property
    def hint(self) -> str:
        return "Aggregate the 'orders' table by user_id in one CTE, and the 'payments' table in another CTE. Then join those aggregated CTEs to the users table."

    def grade(self, rows: Optional[List[Dict[str, Any]]]) -> float:
        if not rows:
            return 0.0

        try:
            # Expected exact answers based on seed data
            expected = {
                "Alice": {"total_orders": 300.0, "total_payments": 300.0},
                "Bob": {"total_orders": 50.0, "total_payments": 50.0}
            }
            
            if len(rows) != 2:
                return 0.1

            score = 0.5
            correct_users = 0

            for row in rows:
                name = row.get("name")
                if name in expected:
                    o_amt = float(row.get("total_orders", 0) or 0)
                    p_amt = float(row.get("total_payments", 0) or 0)
                    
                    if o_amt == expected[name]["total_orders"] and p_amt == expected[name]["total_payments"]:
                        correct_users += 1

            if correct_users == 2:
                return 1.0  # Perfect fix!
            elif correct_users == 1:
                return 0.7  # Partial logic fix

            return score

        except Exception:
            return 0.0
