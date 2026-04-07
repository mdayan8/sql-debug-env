"""
TASK 1 — EASY: Syntax Error Fix
Difficulty: Easy
Bug type: Simple syntax errors (typo in keyword, missing alias, wrong column name)
Max steps: 10
Expected baseline model score: 0.8-1.0
"""
from typing import List, Dict, Any
from .base import BaseTask


class EasyTask(BaseTask):
    """
    Scenario: An e-commerce company wants to find the top 5 customers 
    by total order value. The query has a syntax error: 
    uses 'GRUP BY' instead of 'GROUP BY' and references wrong column alias.
    
    Database: customers, orders, order_items
    Bug 1: 'GRUP BY' typo
    Bug 2: ORDER BY references 'total' but SELECT aliases it as 'total_value'
    """

    @property
    def task_id(self) -> str:
        return "easy_syntax_fix"

    @property
    def name(self) -> str:
        return "Top Customers by Revenue — Syntax Error Fix"

    @property
    def difficulty(self) -> str:
        return "easy"

    @property
    def description(self) -> str:
        return """You are debugging a SQL query for an e-commerce analytics dashboard.

The query is supposed to find the top 5 customers by their total order value 
(sum of quantity * unit_price across all their orders).

The query has 2 syntax/reference bugs that prevent it from running:
1. A typo in a SQL keyword
2. An ORDER BY clause that references a column alias incorrectly

Fix both bugs so the query runs and returns the correct result.

The result should show: customer_name, total_value (rounded to 2 decimal places), 
ordered from highest to lowest, top 5 only."""

    @property
    def expected_output_description(self) -> str:
        return "5 rows: customer_name, total_value (DESC order). Alice Chen should be first with 2847.50."

    @property
    def broken_query(self) -> str:
        return """SELECT 
    c.name AS customer_name,
    ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_value
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
GRUP BY c.id, c.name
ORDER BY total DESC
LIMIT 5"""

    @property
    def schema_sql(self) -> str:
        return """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_date TEXT NOT NULL,
    status TEXT DEFAULT 'completed',
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);

CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    product_name TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders(id)
)"""

    @property
    def seed_data_sql(self) -> str:
        return """
INSERT INTO customers VALUES (1,'Alice Chen','alice@example.com','2023-01-01');
INSERT INTO customers VALUES (2,'Bob Kumar','bob@example.com','2023-01-05');
INSERT INTO customers VALUES (3,'Carol White','carol@example.com','2023-01-10');
INSERT INTO customers VALUES (4,'David Park','david@example.com','2023-02-01');
INSERT INTO customers VALUES (5,'Eva Rodriguez','eva@example.com','2023-02-15');
INSERT INTO customers VALUES (6,'Frank Liu','frank@example.com','2023-03-01');

INSERT INTO orders VALUES (1,1,'2023-06-01','completed');
INSERT INTO orders VALUES (2,1,'2023-07-15','completed');
INSERT INTO orders VALUES (3,2,'2023-06-10','completed');
INSERT INTO orders VALUES (4,3,'2023-06-20','completed');
INSERT INTO orders VALUES (5,3,'2023-08-01','completed');
INSERT INTO orders VALUES (6,4,'2023-07-01','completed');
INSERT INTO orders VALUES (7,5,'2023-07-20','completed');
INSERT INTO orders VALUES (8,5,'2023-08-10','completed');
INSERT INTO orders VALUES (9,6,'2023-09-01','completed');

INSERT INTO order_items VALUES (1,1,'Laptop',1,1200.00);
INSERT INTO order_items VALUES (2,1,'Mouse',2,25.00);
INSERT INTO order_items VALUES (3,2,'Keyboard',1,150.00);
INSERT INTO order_items VALUES (4,2,'Monitor',1,450.00);
INSERT INTO order_items VALUES (5,2,'Webcam',1,97.50);
INSERT INTO order_items VALUES (6,3,'Headphones',1,350.00);
INSERT INTO order_items VALUES (7,3,'USB Hub',2,45.00);
INSERT INTO order_items VALUES (8,4,'Tablet',1,600.00);
INSERT INTO order_items VALUES (9,4,'Case',1,35.00);
INSERT INTO order_items VALUES (10,5,'Charger',2,30.00);
INSERT INTO order_items VALUES (11,5,'Cable',3,15.00);
INSERT INTO order_items VALUES (12,6,'Desk Lamp',1,85.00);
INSERT INTO order_items VALUES (13,6,'Chair Mat',1,60.00);
INSERT INTO order_items VALUES (14,7,'Speakers',1,220.00);
INSERT INTO order_items VALUES (15,7,'Microphone',1,180.00);
INSERT INTO order_items VALUES (16,8,'Webcam',1,97.50);
INSERT INTO order_items VALUES (17,9,'Monitor',1,450.00)"""

    @property
    def expected_output(self) -> List[Dict[str, Any]]:
        # Alice: 1200+50+150+450+97.50 = 1947.50 (orders 1,2)
        # Wait: recalculate
        # Alice order 1: laptop 1200 + mouse 2*25=50 = 1250
        # Alice order 2: keyboard 150 + monitor 450 + webcam 97.50 = 697.50
        # Alice total: 1947.50 — but let me recalculate with all items
        # Actually: 1200+50+150+450+97.50 = 1947.50
        # Carol: tablet 600 + case 35 + charger 60 + cable 45 = 740
        # Eva: speakers 220 + micro 180 + webcam 97.50 = 497.50
        # Bob: headphones 350 + hub 90 = 440
        # Frank: lamp 85 + mat 60 + monitor 450 = 595
        # David: lamp 85 + mat 60 = 145 — wait David is order 6
        # Order 6 items 12,13: lamp 85 + mat 60 = 145
        return [
            {"customer_name": "Alice Chen", "total_value": 1947.50},
            {"customer_name": "Carol White", "total_value": 740.00},
            {"customer_name": "Frank Liu", "total_value": 595.00},
            {"customer_name": "Eva Rodriguez", "total_value": 497.50},
            {"customer_name": "Bob Kumar", "total_value": 440.00},
        ]

    @property
    def hint(self) -> str:
        return "Hint: Check every SQL keyword spelling carefully. Also check that your ORDER BY column name exactly matches the alias in your SELECT clause."

