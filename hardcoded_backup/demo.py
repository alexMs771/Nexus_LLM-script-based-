"""Interactive & CLI Demo Runner for the Hardcoded DDL Prototype.

Usage:
    # Run a single query directly:
    python hardcoded_backup/demo.py "Create a table employee with id, name, and salary in postgresql"

    # Run in interactive mode:
    python hardcoded_backup/demo.py

    # Run on sample test cases:
    python hardcoded_backup/demo.py --samples
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

# Ensure project root and hardcoded_backup are on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Import from hardcoded backup services
from hardcoded_backup.app.services.schema_compiler import (
    compile_ddl_request,
    resolve_dialect,
)
from hardcoded_backup.app.services.scope_gate import classify_request
from app.services.validator import validate_sql


SAMPLE_QUERIES = [
    ("Create an employee table with id, name, and salary in PostgreSQL", "postgresql"),
    ("Create a customers table with id bigint, email varchar(255) unique, and created_at in MySQL", "mysql"),
    ("Create a products table with sku text primary key, title text, and price real in SQLite", "sqlite"),
    ("Create a users table with id int, username varchar(50), and role varchar(20) in DBML", "dbml"),
    ("Add phone varchar(32) to table customers", "postgresql"),
    ("Create an index on customers (email)", "postgresql"),
    ("Select * from employees where salary > 50000", "postgresql"),
    ("Drop all tables in production", "postgresql"),
    ("What is machine learning?", "postgresql"),
]


def run_hardcoded_pipeline(requirement: str, requested_dialect: str | None = None) -> dict:
    started = time.perf_counter()
    
    # 1. Scope gate check
    scope_decision = classify_request(requirement)
    dialect, _ = resolve_dialect(requested_dialect, requirement)
    
    if not scope_decision.accepted:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "status": "rejected" if scope_decision.intent != "NEEDS_CLARIFICATION" else "clarification_required",
            "intent": scope_decision.intent,
            "message": scope_decision.message,
            "dialect": dialect,
            "sql": None,
            "validation": "not_applicable",
            "latency_ms": latency_ms,
            "plan": None,
        }
    
    # 2. Hardcoded Schema Compiler (try standard generation, fallback to modify_ddl)
    compiled = compile_ddl_request(requirement, dialect, task="generate_ddl")
    if compiled is None:
        compiled = compile_ddl_request(requirement, dialect, task="modify_ddl")

    
    if compiled is None:
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "status": "unsupported_by_compiler",
            "intent": scope_decision.intent,
            "message": "Requirement is in scope, but not matched by the hardcoded regex templates.",
            "dialect": dialect,
            "sql": None,
            "validation": "failed",
            "latency_ms": latency_ms,
            "plan": None,
        }
    
    # 3. SQLGlot Validation (for SQL dialects)
    raw_ddl = compiled.output
    if dialect != "dbml":
        validation = validate_sql(raw_ddl, dialect)
        validation_status = validation.validation_status
        clean_sql = validation.sql if validation.valid_syntax else raw_ddl
    else:
        validation_status = "dbml_spec_valid"
        clean_sql = raw_ddl
    
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "status": "accepted",
        "intent": scope_decision.intent,
        "operation": compiled.operation,
        "dialect": dialect,
        "sql": clean_sql,
        "source": compiled.source,
        "validation": validation_status,
        "latency_ms": latency_ms,
        "plan": compiled.plan,
    }


def print_result(requirement: str, result: dict) -> None:
    print("\n" + "=" * 70)
    print(f"REQUIREMENT : {requirement}")
    print(f"INTENT      : {result['intent']}")
    print(f"STATUS      : {result['status'].upper()} (Latency: {result['latency_ms']} ms)")
    print(f"DIALECT     : {result['dialect'].upper()}")
    
    if result["status"] == "accepted":
        print(f"OPERATION   : {result['operation']}")
        print(f"COMPILER    : {result['source']}")
        print(f"VALIDATION  : {result['validation']}")
        print("\n--- GENERATED DDL ---")
        print(result["sql"])
        if result["plan"]:
            print("\n--- EXTRACTED SCHEMA PLAN ---")
            print(json.dumps(result["plan"], indent=2))
    else:
        print(f"REASON      : {result.get('message')}")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate the hardcoded prototype DDL generator.")
    parser.add_argument("query", nargs="?", help="DDL requirement to compile")
    parser.add_argument("--dialect", "-d", choices=["postgresql", "mysql", "sqlite", "dbml"], default=None)
    parser.add_argument("--samples", action="store_true", help="Run against a battery of pre-configured sample queries")
    args = parser.parse_args()

    if args.samples:
        print("\n=== RUNNING HARDCODED PROTOTYPE SAMPLE BATTERY ===\n")
        for query, dialect in SAMPLE_QUERIES:
            res = run_hardcoded_pipeline(query, dialect)
            print_result(query, res)
        return

    if args.query:
        res = run_hardcoded_pipeline(args.query, args.dialect)
        print_result(args.query, res)
        return

    # Interactive mode
    print("\n" + "=" * 70)
    print("  HARDCODED DDL PROTOTYPE - INTERACTIVE DEMO CONSOLE")
    print("  Type any natural language request, or choose a number below:")
    print("=" * 70)
    for idx, (q, _) in enumerate(SAMPLE_QUERIES, start=1):
        print(f"  [{idx}] {q}")
    print("  [q] Quit\n")

    while True:
        try:
            user_input = input("\nEnter requirement (or 1-9, or 'q' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input.lower() in ("q", "quit", "exit"):
            break

        if user_input.isdigit() and 1 <= int(user_input) <= len(SAMPLE_QUERIES):
            query, dialect = SAMPLE_QUERIES[int(user_input) - 1]
        else:
            query = user_input
            dialect = args.dialect

        res = run_hardcoded_pipeline(query, dialect)
        print_result(query, res)


if __name__ == "__main__":
    main()
