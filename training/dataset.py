from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from collections import Counter


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


REQUIRED_FIELDS = {"id", "intent", "request", "expected_ddl", "expected_status", "dataset_version"}


def validate_record(row: dict[str, Any], line_number: int = 0) -> list[str]:
    missing = sorted(REQUIRED_FIELDS - row.keys())
    errors = [f"missing fields: {', '.join(missing)}"] if missing else []
    if row.get("intent") not in {"IN_SCOPE", "NEEDS_CLARIFICATION", "UNSUPPORTED_SQL_OPERATION", "OUT_OF_SCOPE", "RESTRICTED_OR_DESTRUCTIVE"}:
        errors.append("invalid intent")
    if row.get("intent") == "IN_SCOPE" and not row.get("expected_ddl") and row.get("task") != "explain_ddl":
        errors.append("in-scope generation record must contain expected_ddl")
    if row.get("intent") != "IN_SCOPE" and row.get("expected_ddl") not in (None, ""):
        errors.append("non in-scope record cannot contain executable expected_ddl")
    return [f"line {line_number}: {error}" for error in errors] if line_number else errors


def dataset_report(rows: list[dict[str, Any]], split_counts: dict[str, int] | None = None) -> dict[str, Any]:
    return {
        "records": len(rows),
        "intent_distribution": dict(Counter(row.get("intent", "UNKNOWN") for row in rows)),
        "dialect_distribution": dict(Counter(row.get("dialect", "unspecified") for row in rows)),
        "operation_distribution": dict(Counter(row.get("operation", "unspecified") for row in rows)),
        "unique_schema_groups": len({row.get("schema_group", "") for row in rows}),
        "unique_templates": len({row.get("template_id", row.get("id", "")) for row in rows}),
        "source_distribution": dict(Counter(row.get("source", "unknown") for row in rows)),
        "split_counts": split_counts or {},
    }
