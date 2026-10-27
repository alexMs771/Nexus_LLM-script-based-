"""Build a compact, validated DDL instruction corpus.

This cleaner is deliberately non-destructive. It reads existing processed DDL
JSONL files and writes a new versioned corpus. The model-facing files contain
only ``prompt`` and ``answer``. Provenance and rejection reasons are retained
in separate audit files so the training format stays small without losing
traceability.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable

from app.config import ROOT_DIR
from app.services.validator import validate_sql


SOURCE_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = SOURCE_DIR / "ddl_clean_v1"
SPLITS = ("train", "val", "test")
SUPPORTED_DIALECTS = {"postgres", "postgresql", "mysql", "sqlite", "dbml"}
GENERIC_GROUPS = {"", "legacy", "unknown", "none"}
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n").replace("\r", "\n").strip()


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _key(value: str) -> str:
    value = _one_line(value).lower()
    value = re.sub(r"\s*([(),;])\s*", r"\1", value)
    return value


def _strip_fence(value: str) -> str:
    return re.sub(r"^```(?:sql|postgresql|postgres|mysql|sqlite|dbml)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE | re.DOTALL).strip()


def _dialect(value: Any) -> str:
    value = _one_line(value).lower()
    return {"postgres": "postgresql", "pg": "postgresql"}.get(value, value or "postgresql")


def _pick(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = _text(row.get(name))
        if value:
            return value
    return ""


def _source_files(source_dir: Path, output_dir: Path) -> list[Path]:
    """Find DDL splits without scanning raw sources or NL-to-SQL SELECT data."""

    paths: set[Path] = set()
    for split in SPLITS:
        root_file = source_dir / f"{split}.jsonl"
        if root_file.exists():
            paths.add(root_file)

    for directory in source_dir.iterdir():
        if not directory.is_dir() or not directory.name.startswith("ddl"):
            continue
        # Never re-ingest a previous compact/cleaned output as a source.
        if "clean" in directory.name.lower():
            continue
        if directory.resolve() == output_dir.resolve():
            continue
        for split in SPLITS:
            path = directory / f"{split}.jsonl"
            if path.exists():
                paths.add(path)
    return sorted(paths, key=lambda path: str(path).lower())


def _prompt(question: str, dialect: str, schema_context: str) -> str:
    parts = [f"Dialect: {dialect}"]
    if schema_context:
        parts.append(f"Schema context:\n{schema_context}")
    parts.append(f"User request: {question}")
    return "\n\n".join(parts)


def _read_rows(path: Path) -> Iterable[tuple[int, dict[str, Any], str | None]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, {}, f"invalid_json: {exc.msg}"
                continue
            if not isinstance(row, dict):
                yield line_number, {}, "record_is_not_an_object"
                continue
            yield line_number, row, None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _candidate(path: Path, line_number: int, row: dict[str, Any], split: str) -> tuple[dict[str, Any] | None, str | None]:
    intent = _one_line(row.get("intent") or "IN_SCOPE").upper()
    task = _one_line(row.get("task") or row.get("task_type")).lower()
    status = _one_line(row.get("expected_status") or "accepted").lower()
    if intent != "IN_SCOPE":
        return None, "not_in_scope"
    if task in {"explain_ddl", "classify_intent", "intent_classification", "text_to_sql"}:
        return None, "not_ddl_generation"
    if status not in {"", "accepted", "generated", "valid"}:
        return None, "non_accepted_status"

    question = _pick(row, "question", "request", "instruction", "prompt")
    answer = _strip_fence(_pick(row, "target_sql", "expected_ddl", "answer", "response"))
    if not question:
        return None, "missing_question"
    if not answer:
        return None, "missing_answer"

    dialect = _dialect(row.get("dialect"))
    if dialect not in SUPPORTED_DIALECTS:
        return None, "unsupported_dialect"
    schema_context = _pick(row, "schema_context", "schema", "evidence")
    validation = validate_sql(answer, dialect, allow_destructive=False, schema_context=schema_context)
    if not validation.valid_syntax:
        return None, "sql_parse_failed"
    if not validation.policy_allowed:
        return None, "destructive_sql"
    if any("outside the DDL service scope" in warning for warning in validation.warnings):
        return None, "unsupported_statement_type"

    # Keep the target readable and deterministic without rewriting dialect-specific SQL.
    answer = validation.sql.strip()
    if answer and not answer.endswith(";"):
        answer += ";"
    prompt = _prompt(_one_line(question), dialect, _one_line(schema_context))
    group = _one_line(row.get("split_group") or row.get("schema_group") or "")
    source = _one_line(row.get("source") or "unknown")
    record_id = _one_line(row.get("record_id") or row.get("id") or "")
    content_key = hashlib.sha256(f"{_key(prompt)}\n{_key(answer)}".encode("utf-8")).hexdigest()
    prompt_key = hashlib.sha256(f"{_key(prompt)}".encode("utf-8")).hexdigest()
    return {
        "prompt": prompt,
        "answer": answer,
        "_content_key": content_key,
        "_prompt_key": prompt_key,
        "_split": split,
        "_source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "_source_line": line_number,
        "_record_id": record_id,
        "_source": source,
        "_group": group,
        "_dialect": dialect,
    }, None


def _quality_rank(item: dict[str, Any]) -> tuple[int, int, str, int]:
    source = str(item.get("_source", "")).lower()
    if "hand_reviewed" in source or "project_authored" in source:
        source_rank = 0
    elif "curated" in source:
        source_rank = 1
    elif "controlled_synthetic" in source:
        source_rank = 2
    elif "derived" in source:
        source_rank = 3
    else:
        source_rank = 4
    return (
        SPLIT_PRIORITY[str(item["_split"])],
        source_rank,
        str(item["_source_file"]),
        int(item["_source_line"]),
    )


def clean(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    files = _source_files(source_dir, output_dir)
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for path in files:
        split = path.stem if path.stem in SPLITS else "train"
        for line_number, row, parse_error in _read_rows(path):
            counters["input_records"] += 1
            if parse_error:
                reason = parse_error
                rejected.append({"source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"), "source_line": line_number, "reason": reason})
                counters[reason.split(":", 1)[0]] += 1
                continue
            item, reason = _candidate(path, line_number, row, split)
            if item is None:
                rejected.append({
                    "source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
                    "source_line": line_number,
                    "record_id": _one_line(row.get("record_id") or row.get("id")),
                    "question": _pick(row, "question", "request", "instruction", "prompt"),
                    "reason": reason or "rejected",
                })
                counters[reason or "rejected"] += 1
                continue
            candidates.append(item)
            counters["validated_candidates"] += 1

    by_prompt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_prompt[str(item["_prompt_key"])].append(item)
    conflicting_prompts = {key for key, items in by_prompt.items() if len({_key(str(item["answer"])) for item in items}) > 1}
    if conflicting_prompts:
        kept_candidates: list[dict[str, Any]] = []
        for item in candidates:
            if item["_prompt_key"] in conflicting_prompts:
                rejected.append({
                    "source_file": item["_source_file"],
                    "source_line": item["_source_line"],
                    "record_id": item["_record_id"],
                    "question": item["prompt"],
                    "reason": "conflicting_answer_for_same_prompt",
                })
                counters["conflicting_answer_for_same_prompt"] += 1
            else:
                kept_candidates.append(item)
        candidates = kept_candidates

    by_content: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_content[str(item["_content_key"])].append(item)
    selected: list[dict[str, Any]] = []
    for items in by_content.values():
        winner = sorted(items, key=_quality_rank)[0]
        selected.append(winner)
        counters["duplicates_removed"] += max(0, len(items) - 1)
        for duplicate in items:
            if duplicate is winner:
                continue
            rejected.append({
                "source_file": duplicate["_source_file"],
                "source_line": duplicate["_source_line"],
                "record_id": duplicate["_record_id"],
                "question": duplicate["prompt"],
                "reason": "duplicate_content",
                "kept_source_file": winner["_source_file"],
                "kept_source_line": winner["_source_line"],
            })

    selected.sort(key=lambda item: (SPLIT_PRIORITY[str(item["_split"])], str(item["_source_file"]), int(item["_source_line"])))
    compact: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    metadata: list[dict[str, Any]] = []
    for item in selected:
        split = str(item["_split"])
        compact[split].append({"prompt": item["prompt"], "answer": item["answer"]})
        metadata.append({
            "content_key": item["_content_key"],
            "split": split,
            "source_file": item["_source_file"],
            "source_line": item["_source_line"],
            "record_id": item["_record_id"],
            "source": item["_source"],
            "split_group": item["_group"],
            "dialect": item["_dialect"],
        })

    output_counts = {split: _write_jsonl(output_dir / f"{split}.jsonl", compact[split]) for split in SPLITS}
    _write_jsonl(audit_dir / "accepted_metadata.jsonl", metadata)
    _write_jsonl(audit_dir / "rejected.jsonl", rejected)

    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for item in selected:
        group = str(item["_group"]).lower()
        if group not in GENERIC_GROUPS:
            group_splits[group].add(str(item["_split"]))
    cross_split_groups = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}

    manifest = {
        "dataset_version": "ddl-clean-v1",
        "input_scope": "processed root train/val/test and processed ddl*/train/val/test JSONL files",
        "output_format": {"fields": ["prompt", "answer"], "prompt_includes": ["dialect", "schema_context when available", "user request"]},
        "source_files": [str(path.relative_to(ROOT_DIR)).replace("\\", "/") for path in files],
        "counts": {**dict(counters), "output_records": sum(output_counts.values()), "output_by_split": output_counts, "rejected_records": len(rejected)},
        "duplicate_policy": "Exact normalized prompt+answer duplicates are reduced globally; test is preferred over val, and val over train.",
        "conflict_policy": "Records with the same normalized prompt but different answers are rejected for manual review.",
        "cross_split_groups_for_review": cross_split_groups,
        "raw_data_untouched": True,
        "audit_files": ["audit/accepted_metadata.jsonl", "audit/rejected.jsonl"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    manifest = clean(args.source_dir, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
