"""Evaluate an encoder-decoder DDL artifact on compact held-out JSONL."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import sqlglot

from app.models.encoder_decoder_model import EncoderDecoderDDLModel


def normalize_sql(value: str) -> str:
    cleaned = re.sub(r"^```(?:sql)?\s*|\s*```$", "", value.strip(), flags=re.I | re.S).rstrip(";").strip()
    try:
        return "; ".join(statement.sql().lower() for statement in sqlglot.parse(cleaned, read="postgres") if statement is not None)
    except Exception:
        return re.sub(r"\s+", " ", cleaned).lower()


def task_from_prompt(prompt: str) -> str:
    match = re.search(r"<\|task\|>\s*\n([^\n]+)", prompt, flags=re.I)
    if match:
        return match.group(1).strip()
    match = re.search(r"Operation:\s*([^\n]+)", prompt, flags=re.I)
    return match.group(1).strip() if match else "unknown"


def dialect_from_prompt(prompt: str) -> str:
    match = re.search(r"<\|dialect\|>\s*\n([^\n]+)", prompt, flags=re.I)
    return match.group(1).strip() if match else "postgresql"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the own encoder-decoder DDL SLM.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.25)
    args = parser.parse_args()
    model = EncoderDecoderDDLModel.from_dir(args.model_dir)
    total = syntax = exact = 0
    by_operation: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "syntax_valid": 0, "exact_match": 0})
    failures: list[dict[str, object]] = []
    started = time.perf_counter()
    with args.test_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            expected = str(row.get("answer", "")).strip()
            if not prompt or not expected:
                continue
            dialect = dialect_from_prompt(prompt)
            operation = task_from_prompt(prompt)
            candidates: list[tuple[str, bool]] = []
            for candidate_index in range(max(1, args.candidates)):
                generated = model.generate(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0 if candidate_index == 0 else args.temperature,
                )
                try:
                    sqlglot.parse(generated, read="postgres" if dialect == "postgresql" else dialect)
                    valid = bool(re.search(r"\b(create|alter|drop)\b", generated, re.I))
                except Exception:
                    valid = False
                candidates.append((generated, valid))
            generated, valid = next((item for item in candidates if item[1]), candidates[0])
            is_exact = normalize_sql(generated) == normalize_sql(expected)
            total += 1
            syntax += int(valid)
            exact += int(is_exact)
            by_operation[operation]["total"] += 1
            by_operation[operation]["syntax_valid"] += int(valid)
            by_operation[operation]["exact_match"] += int(is_exact)
            if len(failures) < 20 and (not valid or not is_exact):
                failures.append({"prompt": prompt, "expected": expected, "generated": generated, "candidate_count": len(candidates)})
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
        "candidate_count": max(1, args.candidates),
        "latency_ms_per_example": (time.perf_counter() - started) * 1000 / max(total, 1),
        "by_operation": dict(by_operation),
        "failure_samples": failures,
    }
    output = args.out or args.model_dir / "evaluation_seq2seq.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
