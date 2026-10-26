from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from app.config import ROOT_DIR
from app.prompts import build_prompt
from app.services.validator import validate_sql
from app.services.scope_gate import classify_request
from training.dataset import dataset_report, read_jsonl, validate_record, write_jsonl
from training.synthetic_data import build_held_out_intent_records, build_synthetic_records

EXAMPLES_PATH = ROOT_DIR / "data" / "examples" / "ddl_examples.jsonl"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.json"
TEST_GROUPS = {"held_out_library", "held_out_clinic", "held_out_generated", "scope_boundaries"}
VAL_GROUPS = {"schema_ops", "explanations", "generated_operations"}


def _normal_key(row: dict) -> str:
    return " ".join(str(row.get("request", row.get("instruction", ""))).lower().split())


def normalise_row(row: dict, index: int) -> dict:
    request = row.get("request", row.get("instruction", "")).strip()
    output = row.get("expected_ddl", row.get("output"))
    if row.get("intent") is None:
        task = row.get("task", "generate_ddl")
        request_lower = request.lower()
        output_text = str(output or "")
        if task == "explain_ddl":
            intent = "IN_SCOPE"
        elif task in {"generate_ddl", "modify_ddl"} and any(token in request_lower for token in ("drop table", "drop schema", "truncate", "delete all", "delete production")):
            intent = "RESTRICTED_OR_DESTRUCTIVE"
        elif task in {"generate_ddl", "modify_ddl"} and output_text:
            # Seed examples already declare the intended DDL task. Their SQL is
            # validated below; this keeps valid DDL from being relabelled by a
            # classifier trained on an older split during corpus preparation.
            intent = "IN_SCOPE"
        else:
            intent = classify_request(request).intent
    else:
        intent = row["intent"]
    intent = {"RESTRICTED_OR_UNSUPPORTED": "RESTRICTED_OR_DESTRUCTIVE"}.get(intent, intent)
    return {
        "id": row.get("id", f"legacy_{index:04d}"),
        "intent": intent,
        "dialect": row.get("dialect", "postgresql"),
        "operation": row.get("operation", row.get("task", "generate_ddl")).upper(),
        "request": request,
        "expected_ddl": output if intent == "IN_SCOPE" and row.get("task") != "explain_ddl" else None,
        "expected_status": row.get("expected_status", "accepted" if intent == "IN_SCOPE" else "rejected"),
        "task": row.get("task", "generate_ddl"),
        "schema_group": row.get("schema_group", "legacy"),
        "template_id": row.get("template_id", "legacy"),
        "source": row.get("source", "hand_reviewed_seed"),
        "license": row.get("license", "project-authored; verify source before redistribution"),
        "dataset_version": "ddl-v2",
    }


def split_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["schema_group"]].append(row)
    train, val, test = [], [], []
    for group, group_rows in sorted(grouped.items()):
        if group in TEST_GROUPS or group.startswith("held_out"):
            test.extend(group_rows)
        elif group in VAL_GROUPS:
            val.extend(group_rows)
        else:
            train.extend(group_rows)
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, deduplicate, and split the DDL corpus.")
    parser.add_argument("--examples", type=Path, default=EXAMPLES_PATH)
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    source_rows = read_jsonl(args.examples) if args.examples.exists() else []
    negative_path = args.examples.parent / "invalid_ddl.jsonl"
    if negative_path.exists() and negative_path != args.examples:
        source_rows.extend(read_jsonl(negative_path))
    rows = [normalise_row(row, index) for index, row in enumerate(source_rows, start=1)] + build_synthetic_records()
    # Prefer explicit held-out boundary records when a seed example is an exact duplicate;
    # this prevents a restricted case from silently living only in the training partition.
    rows.sort(key=lambda row: (0 if row["schema_group"] in TEST_GROUPS else 1, row["id"]))
    deduped: list[dict] = []
    seen: set[str] = set()
    rejected: list[dict] = []
    for row in rows:
        key = _normal_key(row)
        errors = validate_record(row)
        if key in seen or errors:
            rejected.append({"id": row["id"], "reason": "duplicate" if key in seen else "; ".join(errors)})
            continue
        if row["expected_ddl"] and row["task"] != "explain_ddl":
            result = validate_sql(row["expected_ddl"], row["dialect"])
            if not result.valid_syntax:
                rejected.append({"id": row["id"], "reason": "invalid expected SQL", "warnings": result.warnings})
                continue
        seen.add(key)
        deduped.append(row)
    train, val, test = split_rows(deduped)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in (("train", train), ("val", val), ("test", test)):
        write_jsonl(args.out_dir / f"{name}.jsonl", split)
    write_jsonl(args.out_dir / "rejected.jsonl", rejected)
    benchmark = build_held_out_intent_records()
    write_jsonl(args.out_dir / "intent_test.jsonl", benchmark)
    counts = {"train": len(train), "val": len(val), "test": len(test), "rejected": len(rejected)}
    manifest = {
        "dataset_version": "ddl-v2",
        "source_file": str(args.examples),
        "sha256_source": hashlib.sha256(args.examples.read_bytes()).hexdigest() if args.examples.exists() else None,
        "counts": counts,
        "intent_benchmark_count": len(benchmark),
        "report": dataset_report(deduped, counts),
        "split_policy": "schema_group/template held-out split; no random row split",
        "license_note": "Synthetic rows are project-authored. External sources require license review before import.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
