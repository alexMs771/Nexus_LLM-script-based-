from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from app.config import settings
from app.prompts import build_prompt
from app.services.scope_gate import classify_request
from app.services.validator import validate_sql
from training.dataset import read_jsonl
from training.intent_classifier import INTENT_LABELS, IntentClassifier
from training.prepare_data import PROCESSED_DIR, main as prepare_data


def _intent_metrics(rows: list[dict], predictor=classify_request) -> dict:
    labels = list(INTENT_LABELS)
    matrix = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for row in rows:
        actual = row["intent"]
        predicted = predictor(row["request"]).intent
        if actual in matrix and predicted in matrix[actual]:
            matrix[actual][predicted] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][predicted] for predicted in labels if predicted != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
    total = sum(sum(row.values()) for row in matrix.values())
    correct = sum(matrix[label][label] for label in labels)
    out_scope = [row for row in rows if row["intent"] == "OUT_OF_SCOPE"]
    false_accepts = sum(predictor(row["request"]).intent == "IN_SCOPE" for row in out_scope)
    return {
        "accuracy": correct / max(total, 1), "per_class": per_class, "confusion_matrix": matrix,
        "false_acceptance_rate_out_of_scope": false_accepts / max(len(out_scope), 1),
        "false_rejection_rate_in_scope": sum(predictor(row["request"]).intent != "IN_SCOPE" for row in rows if row["intent"] == "IN_SCOPE") / max(sum(row["intent"] == "IN_SCOPE" for row in rows), 1),
    }


def _classifier_metrics(rows: list[dict], classifier: IntentClassifier) -> dict:
    return _intent_metrics(rows, lambda text: type("Prediction", (), {"intent": classifier.predict(text)[0]})())


def evaluate_generation(name: str, generate_fn, rows: list[dict]) -> dict:
    started = time.perf_counter()
    generation_rows = [row for row in rows if row["intent"] == "IN_SCOPE" and row.get("expected_ddl")]
    syntax_ok = semantic_exact = 0
    details = []
    for row in generation_rows:
        prompt = build_prompt(row["request"], row.get("dialect", "postgresql"), row.get("task", "generate_ddl"), intent=row.get("intent", "IN_SCOPE"))
        output = generate_fn(prompt)
        result = validate_sql(output, row.get("dialect", "postgresql"))
        expected = validate_sql(row["expected_ddl"], row.get("dialect", "postgresql"))
        predicted_norm = result.sql.upper().replace(" ", "") if result.valid_syntax else ""
        expected_norm = expected.sql.upper().replace(" ", "")
        syntax_ok += int(result.valid_syntax)
        semantic_exact += int(result.valid_syntax and predicted_norm == expected_norm)
        details.append({"id": row["id"], "syntax_valid": result.valid_syntax, "exact_normalized": bool(result.valid_syntax and predicted_norm == expected_norm), "warnings": result.warnings})
    elapsed = time.perf_counter() - started
    total = max(len(generation_rows), 1)
    return {
        "model": name, "cases": len(generation_rows), "syntax_validity": syntax_ok / total,
        "normalized_exact_match": semantic_exact / total, "latency_ms_mean": (elapsed * 1000) / total,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scope gate and available model on one held-out benchmark.")
    parser.add_argument("--test", type=Path, default=PROCESSED_DIR / "test.jsonl")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "evaluation.json")
    parser.add_argument("--model-dir", type=Path, default=settings.scratch_dir)
    parser.add_argument("--intent-test", type=Path, default=PROCESSED_DIR / "intent_test.jsonl")
    args = parser.parse_args()
    if not args.test.exists():
        prepare_data()
    rows = read_jsonl(args.test)
    intent_rows = read_jsonl(args.intent_test) if args.intent_test.exists() else rows
    train_rows = read_jsonl(PROCESSED_DIR / "train.jsonl") if (PROCESSED_DIR / "train.jsonl").exists() else rows
    independent_classifier = IntentClassifier.train(train_rows)
    report = {
        "dataset": str(args.test), "cases": len(rows),
        "intent_benchmark": str(args.intent_test), "intent_benchmark_cases": len(intent_rows),
        "intent": {"scope_gate": _intent_metrics(intent_rows), "independent_classifier": _classifier_metrics(intent_rows, independent_classifier)},
        "models": [],
    }
    if (args.model_dir / "best.pt").exists() or (args.model_dir / "model.pt").exists():
        from app.models.scratch_model import ScratchSQLModel
        report["models"].append(evaluate_generation("scratch", ScratchSQLModel.from_dir(args.model_dir).generate, rows))
    else:
        report["models"].append({"model": "scratch", "status": "not_run", "reason": "checkpoint not found"})
    report["quality_gates"] = {
        "syntax_target": 0.95,
        "semantic_target": 0.90,
        "intent_f1_target": 0.95,
        "execution_enabled": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
