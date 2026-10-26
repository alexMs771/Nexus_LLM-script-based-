"""Run automated benchmark suite on the hardcoded prototype.

Usage:
  python hardcoded_backup/benchmark.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hardcoded_backup.demo import run_hardcoded_pipeline


TEST_SUITE = [
    ("Single Table with PK & types (PostgreSQL)", "Create an employee table with id bigint, name varchar(100), and salary decimal(10,2) in PostgreSQL", "postgresql"),
    ("Unique constraint & Timestamp (MySQL)", "Create a customers table with id bigint, email varchar(255) unique, and created_at timestamp in MySQL", "mysql"),
    ("Text Primary Key (SQLite)", "Create a products table with sku text primary key, title text, and price real in SQLite", "sqlite"),
    ("DBML Specification (DBML)", "Create a users table with id int, username varchar(50), and role varchar(20) in DBML", "dbml"),
    ("Alter Table Add Column (PostgreSQL)", "Add phone varchar(32) to table customers", "postgresql"),
    ("Index Creation (PostgreSQL)", "Create an index on customers (email)", "postgresql"),
    ("Multi-column composite key (PostgreSQL)", "Create a stock table with warehouse_id bigint, sku varchar(64) in postgresql", "postgresql"),
    ("Unsupported SELECT (Scope Gate)", "Select * from employees where salary > 50000", "postgresql"),
    ("Destructive DROP (Scope Gate)", "Drop all tables in production", "postgresql"),
    ("Out of Scope Question (Scope Gate)", "What is machine learning?", "postgresql"),
]


def main() -> None:
    print("\n" + "=" * 90)
    print("  HARDCODED PROTOTYPE BENCHMARK REPORT")
    print("=" * 90)
    print(f"{'#':<3} | {'Test Scenario':<38} | {'Dialect':<10} | {'Status':<12} | {'Syntax':<10} | {'Latency'}")
    print("-" * 90)

    total_time = 0.0
    passed = 0

    for idx, (scenario, query, dialect) in enumerate(TEST_SUITE, start=1):
        res = run_hardcoded_pipeline(query, dialect)
        total_time += res["latency_ms"]
        syntax_ok = res["validation"] in ("valid", "dbml_spec_valid", "not_applicable")
        if syntax_ok:
            passed += 1

        print(
            f"{idx:<3} | {scenario[:38]:<38} | {res['dialect'].upper():<10} | "
            f"{res['status'].upper():<12} | {res['validation']:<10} | {res['latency_ms']} ms"
        )

    avg_time = round(total_time / len(TEST_SUITE), 2)
    print("-" * 90)
    print(f"Summary: {passed}/{len(TEST_SUITE)} Passed Validation | Avg Latency: {avg_time} ms | Mode: Deterministic Rules")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
