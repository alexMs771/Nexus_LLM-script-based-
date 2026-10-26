"""Evaluate a trained own model without inventing execution results."""

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
from training.sql_format import build_sql_prompt


READ_DIALECT = {"postgresql": "postgres", "mysql": "mysql", "sqlite": "sqlite"}


def _clean_sql(value: str) -> str:
    value = re.sub(r"^```(?:sql)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE | re.DOTALL)
    return value.strip().rstrip(";").strip()


def _normalized(value: str, dialect: str) -> str:
    try:
        statements = sqlglot.parse(_clean_sql(value), read=READ_DIALECT.get(dialect, "postgres"))
        return "; ".join(statement.sql() for statement in statements if statement is not None).lower()
    except Exception:
        return re.sub(r"\s+", " ", _clean_sql(value)).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate own-sql on held-out NL-to-SQL records.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/nl2sql/test.jsonl"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    model = ScratchSQLModel.from_dir(args.model_dir)
    if not args.test_path.exists():
        raise SystemExit(f"Test data not found: {args.test_path}")

    total = syntax_ok = exact_ok = schema_rows = 0
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "syntax_ok": 0, "exact_match": 0})
    samples: list[dict[str, object]] = []
    started = time.perf_counter()
    with args.test_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            dialect = str(row.get("dialect") or "postgresql")
            validation_dialect = dialect if dialect in READ_DIALECT else "postgresql"
            prompt = build_sql_prompt(
                row.get("question", ""), row.get("schema_context", ""), dialect, row.get("evidence", "")
            )
            generated = model.generate(prompt, max_new_tokens=args.max_new_tokens)
            validation = validate_sql(generated, validation_dialect, schema_context=row.get("schema_context", ""))
            expected = row.get("target_sql", "")
            is_syntax = bool(validation.valid_syntax)
            is_exact = _normalized(generated, validation_dialect) == _normalized(expected, validation_dialect)
            source = str(row.get("source_id", "unknown"))
            total += 1
            syntax_ok += int(is_syntax)
            exact_ok += int(is_exact)
            schema_rows += int(bool(row.get("schema_available")))
            by_source[source]["total"] += 1
            by_source[source]["syntax_ok"] += int(is_syntax)
            by_source[source]["exact_match"] += int(is_exact)
            if len(samples) < 20 and (not is_syntax or not is_exact):
                samples.append({"source_id": source, "question": row.get("question", ""), "expected_sql": expected, "generated_sql": generated, "warnings": validation.warnings})
            if args.limit and total >= args.limit:
                break

    report = {
        "model_dir": str(args.model_dir),
        "model_version": model.version,
        "total": total,
        "syntax_accuracy": syntax_ok / max(total, 1),
        "exact_match_accuracy": exact_ok / max(total, 1),
        "schema_coverage": schema_rows / max(total, 1),
        "execution_accuracy": None,
        "execution_accuracy_note": "Not claimed: this evaluator does not execute generated SQL against disposable databases.",
        "latency_ms_per_example": (time.perf_counter() - started) * 1000 / max(total, 1),
        "by_source": dict(by_source),
        "failure_samples": samples,
    }
    output = args.out or args.model_dir / "evaluation_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
