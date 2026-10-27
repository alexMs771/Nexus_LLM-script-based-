"""Evaluate the learned DDL model on the held-out DDL split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time

import torch
import sqlglot

from app.models.scratch_model import ScratchSQLModel
from app.services.validator import validate_sql
from training.sql_format import build_ddl_prompt


READ_DIALECT = {"postgresql": "postgres", "mysql": "mysql", "sqlite": "sqlite"}


def _compact_fields(row: dict) -> tuple[str, str, str, str]:
    """Return prompt, expected output, dialect, and operation for both formats."""
    compact_prompt = str(row.get("prompt", "")).strip()
    compact_answer = str(row.get("answer", "")).strip()
    if compact_prompt and compact_answer:
        dialect_match = re.search(r"<\|dialect\|>\s*\n([^\n]+)", compact_prompt, flags=re.I)
        task_match = re.search(r"<\|task\|>\s*\n([^\n]+)", compact_prompt, flags=re.I)
        return (
            compact_prompt,
            compact_answer,
            (dialect_match.group(1).strip() if dialect_match else "postgresql"),
            (task_match.group(1).strip() if task_match else "unknown"),
        )
    dialect = str(row.get("dialect") or "postgresql")
    prompt = build_ddl_prompt(row.get("question", ""), dialect, row.get("operation", "generate_ddl"), row.get("schema_context", ""))
    return prompt, str(row.get("target_sql", "")), dialect, str(row.get("operation", "unknown"))


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
    parser.add_argument("--candidates", type=int, default=1, help="Generate multiple candidates and select the first valid DDL.")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--repetition-penalty", type=float, default=1.04)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=0)
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
            prompt, expected_sql, dialect, operation = _compact_fields(row)
            candidates: list[tuple[str, object]] = []
            for candidate_index in range(max(1, args.candidates)):
                # A deterministic seed makes repeated evaluation comparable,
                # while allowing optional stochastic candidates to explore
                # alternatives from the learned distribution.
                torch.manual_seed(1009 + total * 31 + candidate_index)
                generated_candidate = model.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0 if candidate_index == 0 else args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    top_k=args.top_k,
                )
                candidate_result = validate_sql(
                    generated_candidate,
                    dialect,
                    allow_destructive=False,
                    schema_context=row.get("schema_context", ""),
                )
                candidates.append((generated_candidate, candidate_result))
            generated, result = next(
                ((candidate, candidate_result) for candidate, candidate_result in candidates if candidate_result.valid_syntax and candidate_result.policy_allowed),
                candidates[0],
            )
            is_syntax = bool(result.valid_syntax and result.policy_allowed)
            is_exact = _normal(generated, dialect) == _normal(expected_sql, dialect)
            total += 1
            syntax += int(is_syntax)
            exact += int(is_exact)
            by_operation[operation]["total"] += 1
            by_operation[operation]["syntax_valid"] += int(is_syntax)
            by_operation[operation]["exact_match"] += int(is_exact)
            if len(failures) < 20 and (not is_syntax or not is_exact):
                failures.append({"question": row.get("question", row.get("prompt", "")), "expected_sql": expected_sql, "generated_sql": generated, "warnings": result.warnings, "candidate_count": len(candidates)})
            if args.limit and total >= args.limit:
                break
    report = {
        "model_dir": str(args.model_dir),
        "model_version": model.version,
        "total": total,
        "syntax_accuracy": syntax / max(total, 1),
        "exact_match_accuracy": exact / max(total, 1),
        "candidate_count": max(1, args.candidates),
        "decoding": {
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "top_k": args.top_k,
        },
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
