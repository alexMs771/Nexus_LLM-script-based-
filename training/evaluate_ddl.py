"""Evaluate the learned DDL model on the held-out DDL split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import sqlglot

from app.models.scratch_model import ScratchSQLModel
from app.services.validator import validate_sql
from training.sql_format import build_ddl_prompt


READ_DIALECT = {"postgresql": "postgres", "mysql": "mysql", "sqlite": "sqlite"}


def _normal(value: str, dialect: str) -> str:
    cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", value.strip(), flags=re.I | re.S).rstrip(";").strip()
    try:
        statements = sqlglot.parse(cleaned, read=READ_DIALECT.get(dialect, "postgres"))
        return "; ".join(statement.sql().lower() for statement in statements if statement is not None)
    except Exception:
        return re.sub(r"\s+", " ", cleaned).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate own-ddl on held-out DDL examples.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/ddl/test.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    model = ScratchSQLModel.from_dir(args.model_dir)
    total = syntax = exact = 0
    by_operation: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "syntax_valid": 0, "exact_match": 0})
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    with args.test_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dialect = str(row.get("dialect") or "postgresql")
            generated = model.generate(
                build_ddl_prompt(row.get("question", ""), dialect, row.get("operation", "generate_ddl"), row.get("schema_context", "")),
                max_new_tokens=args.max_new_tokens,
            )
            result = validate_sql(generated, dialect, allow_destructive=False, schema_context=row.get("schema_context", ""))
            operation = str(row.get("operation", "unknown"))
            is_syntax = bool(result.valid_syntax and result.policy_allowed)
            is_exact = _normal(generated, dialect) == _normal(str(row.get("target_sql", "")), dialect)
            total += 1
            syntax += int(is_syntax)
            exact += int(is_exact)
            by_operation[operation]["total"] += 1
            by_operation[operation]["syntax_valid"] += int(is_syntax)
            by_operation[operation]["exact_match"] += int(is_exact)
            if len(failures) < 20 and (not is_syntax or not is_exact):
                failures.append({"question": row.get("question", ""), "expected_sql": row.get("target_sql", ""), "generated_sql": generated, "warnings": result.warnings})
            if args.limit and total >= args.limit:
                break
    report = {
        "model_dir": str(args.model_dir),
        "model_version": model.version,
        "total": total,
        "syntax_accuracy": syntax / max(total, 1),
        "exact_match_accuracy": exact / max(total, 1),
        "execution_accuracy": None,
        "execution_accuracy_note": "Not claimed until generated DDL is executed in disposable databases.",
        "latency_ms_per_example": (time.perf_counter() - started) * 1000 / max(total, 1),
        "by_operation": dict(by_operation),
        "failure_samples": failures,
    }
    output = args.out or args.model_dir / "evaluation_ddl.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
