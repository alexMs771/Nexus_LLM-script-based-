"""Compact every processed training task into prompt/answer records.

The output is task-separated because DDL generation, NL-to-SQL generation,
and intent classification have different answer spaces. Existing raw and
processed files are never overwritten.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Iterable

from app.config import ROOT_DIR
from training.clean_ddl import clean as clean_ddl


SOURCE_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = SOURCE_DIR / "compact_clean_v2"
SPLITS = ("train", "val", "test")
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
VALID_INTENTS = {
    "IN_SCOPE",
    "NEEDS_CLARIFICATION",
    "UNSUPPORTED_SQL_OPERATION",
    "OUT_OF_SCOPE",
    "RESTRICTED_OR_DESTRUCTIVE",
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).replace("\r\n", "\n").replace("\r", "\n").strip()


def _one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _key(value: str) -> str:
    value = _one_line(value).lower()
    return re.sub(r"\s*([(),;])\s*", r"\1", value)


def _strip_fence(value: str) -> str:
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", value.strip(), flags=re.IGNORECASE | re.DOTALL).strip()


def _looks_like_query(value: str) -> bool:
    """Fast, dialect-neutral NL-to-SQL sanity check.

    The corpus includes PostgreSQL/MySQL/SQLite/vendor-specific SQL. A full
    parser pass over every row is both slow and overly strict for mixed
    dialects, so dialect-specific parse/execution validation remains a later
    evaluation stage.
    """

    without_comments = re.sub(r"/\*.*?\*/|--[^\n]*", " ", value, flags=re.DOTALL).lstrip()
    return bool(re.match(r"(?is)^(?:select|with|explain|show|describe|pragma)\b", without_comments))


def _balanced_parentheses(value: str) -> bool:
    depth = 0
    quote: str | None = None
    escaped = False
    for char in value:
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"', "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0 and quote is None


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any], str | None]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {}, f"invalid_json: {exc.msg}"
                continue
            if not isinstance(row, dict):
                yield line_number, {}, "record_is_not_an_object"
                continue
            yield line_number, row, None


def _source_info(path: Path, line_number: int, row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        "split": split,
        "source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
        "source_line": line_number,
        "record_id": _one_line(row.get("record_id") or row.get("id")),
        "source": _one_line(row.get("source") or row.get("source_id")),
        "split_group": _one_line(row.get("split_group") or row.get("schema_group")),
    }


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


def _dedupe(
    candidates: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    counters: Counter[str],
) -> list[dict[str, Any]]:
    by_prompt: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_prompt[str(item["_prompt_key"])].append(item)
    conflicts = {key for key, items in by_prompt.items() if len({_key(str(item["answer"])) for item in items}) > 1}
    if conflicts:
        retained = []
        for item in candidates:
            if item["_prompt_key"] in conflicts:
                rejected.append({**_audit(item), "reason": "conflicting_answer_for_same_prompt"})
                counters["conflicting_answer_for_same_prompt"] += 1
            else:
                retained.append(item)
        candidates = retained

    by_content: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        by_content[str(item["_content_key"])].append(item)
    selected: list[dict[str, Any]] = []
    for items in by_content.values():
        winner = sorted(items, key=_quality_rank)[0]
        selected.append(winner)
        counters["duplicates_removed"] += max(0, len(items) - 1)
        for item in items:
            if item is not winner:
                rejected.append({**_audit(item), "reason": "duplicate_content", "kept_source_file": winner["_source_file"], "kept_source_line": winner["_source_line"]})
    selected.sort(key=lambda item: (SPLIT_PRIORITY[str(item["_split"])], str(item["_source_file"]), int(item["_source_line"])))
    return selected


def _audit(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_file": item["_source_file"],
        "source_line": item["_source_line"],
        "record_id": item.get("_record_id", ""),
        "reason": "rejected",
    }


def _write_task(
    task_dir: Path,
    task_name: str,
    selected: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    counters: Counter[str],
    source_files: list[Path],
) -> dict[str, Any]:
    task_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = task_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    compact = {split: [] for split in SPLITS}
    metadata = []
    for item in selected:
        split = str(item["_split"])
        compact[split].append({"prompt": item["prompt"], "answer": item["answer"]})
        metadata.append({
            "content_key": item["_content_key"],
            "split": split,
            "source_file": item["_source_file"],
            "source_line": item["_source_line"],
            "record_id": item.get("_record_id", ""),
            "source": item.get("_source", ""),
            "split_group": item.get("_group", ""),
        })
    output_counts = {split: _write_jsonl(task_dir / f"{split}.jsonl", compact[split]) for split in SPLITS}
    _write_jsonl(audit_dir / "accepted_metadata.jsonl", metadata)
    _write_jsonl(audit_dir / "rejected.jsonl", rejected)
    manifest = {
        "task": task_name,
        "output_format": {"fields": ["prompt", "answer"]},
        "source_files": [str(path.relative_to(ROOT_DIR)).replace("\\", "/") for path in source_files],
        "counts": {**dict(counters), "output_records": sum(output_counts.values()), "output_by_split": output_counts, "rejected_records": len(rejected)},
        "audit_files": ["audit/accepted_metadata.jsonl", "audit/rejected.jsonl"],
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _make_candidate(path: Path, line_number: int, row: dict[str, Any], split: str, task: str) -> tuple[dict[str, Any] | None, str | None]:
    question = _one_line(row.get("question") or row.get("request") or row.get("instruction") or row.get("prompt"))
    schema = _one_line(row.get("schema_context") or row.get("schema"))
    evidence = _one_line(row.get("evidence"))
    dialect = _one_line(row.get("dialect") or "unknown").lower()
    database_id = _one_line(row.get("database_id"))
    if not question:
        return None, "missing_question"
    answer = _strip_fence(_one_line(row.get("target_sql") or row.get("answer")))
    if not answer:
        return None, "missing_answer"
    context = []
    if schema:
        context.append(f"Schema context:\n{schema}")
    if evidence:
        context.append(f"Evidence: {evidence}")
    if database_id:
        context.append(f"Database: {database_id}")
    prompt = "\n\n".join([f"Task: {task}", f"Dialect: {dialect}", *context, f"User request: {question}"])
    if not _looks_like_query(answer):
        return None, "non_query_sql"
    if not _balanced_parentheses(answer):
        return None, "unbalanced_sql"
    if not answer.endswith(";"):
        answer += ";"
    content_key = hashlib.sha256(f"{_key(prompt)}\n{_key(answer)}".encode("utf-8")).hexdigest()
    prompt_key = hashlib.sha256(_key(prompt).encode("utf-8")).hexdigest()
    info = _source_info(path, line_number, row, split)
    return {
        "prompt": prompt,
        "answer": answer,
        "_content_key": content_key,
        "_prompt_key": prompt_key,
        "_split": split,
        "_source_file": info["source_file"],
        "_source_line": line_number,
        "_record_id": info["record_id"],
        "_source": info["source"],
        "_group": info["split_group"],
    }, None


def _clean_nl2sql(source_dir: Path, task_dir: Path) -> dict[str, Any]:
    source_files = [source_dir / "nl2sql" / f"{split}.jsonl" for split in SPLITS]
    source_files = [path for path in source_files if path.exists()]
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in source_files:
        split = path.stem
        for line_number, row, parse_error in _read_jsonl(path):
            counters["input_records"] += 1
            if parse_error:
                rejected.append({"source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"), "source_line": line_number, "reason": parse_error})
                counters["invalid_json"] += 1
                continue
            item, reason = _make_candidate(path, line_number, row, split, "nl2sql")
            if item is None:
                rejected.append({"source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"), "source_line": line_number, "record_id": _one_line(row.get("record_id") or row.get("id")), "reason": reason or "rejected"})
                counters[reason or "rejected"] += 1
            else:
                candidates.append(item)
                counters["validated_candidates"] += 1
    selected = _dedupe(candidates, rejected, counters)
    return _write_task(task_dir, "nl2sql", selected, rejected, counters, source_files)


def _stream_clean_nl2sql(source_dir: Path, task_dir: Path) -> dict[str, Any]:
    """Compact the large NL-to-SQL corpus without holding it in memory."""

    task_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = task_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    source_files = [source_dir / "nl2sql" / f"{split}.jsonl" for split in SPLITS]
    source_files = [path for path in source_files if path.exists()]
    counters: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    source_paths = {path: str(path.relative_to(ROOT_DIR)).replace("\\", "/") for path in source_files}
    rejected_path = audit_dir / "rejected.jsonl"
    metadata_path = audit_dir / "accepted_metadata.jsonl"
    db_path = audit_dir / "dedupe.sqlite3"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE seen (content_key TEXT PRIMARY KEY)")
        connection.commit()
        with rejected_path.open("w", encoding="utf-8") as rejected_handle, metadata_path.open("w", encoding="utf-8") as metadata_handle:
            # Process test first so an exact duplicate is retained for evaluation
            # rather than being consumed by a training row.
            for split in ("test", "val", "train"):
                path = source_dir / "nl2sql" / f"{split}.jsonl"
                if not path.exists():
                    continue
                output_path = task_dir / f"{split}.jsonl"
                with path.open(encoding="utf-8") as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
                    for line_number, line in enumerate(input_handle, start=1):
                        if not line.strip():
                            continue
                        counters["input_records"] += 1
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "reason": f"invalid_json: {exc.msg}"}, separators=(",", ":")) + "\n")
                            counters["invalid_json"] += 1
                            continue
                        if not isinstance(row, dict):
                            rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "reason": "record_is_not_an_object"}, separators=(",", ":")) + "\n")
                            counters["record_is_not_an_object"] += 1
                            continue
                        item, reason = _make_candidate(path, line_number, row, split, "nl2sql")
                        if item is None:
                            rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "record_id": _one_line(row.get("record_id") or row.get("id")), "reason": reason or "rejected"}, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counters[reason or "rejected"] += 1
                            continue
                        inserted = connection.execute("INSERT OR IGNORE INTO seen(content_key) VALUES (?)", (item["_content_key"],)).rowcount
                        if not inserted:
                            rejected_handle.write(json.dumps({**_audit(item), "reason": "duplicate_content"}, ensure_ascii=False, separators=(",", ":")) + "\n")
                            counters["duplicates_removed"] += 1
                            continue
                        output_handle.write(json.dumps({"prompt": item["prompt"], "answer": item["answer"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
                        metadata_handle.write(json.dumps({
                            "content_key": item["_content_key"],
                            "split": split,
                            "source_file": item["_source_file"],
                            "source_line": item["_source_line"],
                            "record_id": item["_record_id"],
                            "source": item["_source"],
                            "split_group": item["_group"],
                        }, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counters["validated_candidates"] += 1
                        output_counts[split] += 1
                connection.commit()
    finally:
        connection.close()
    manifest = {
        "task": "nl2sql",
        "output_format": {"fields": ["prompt", "answer"]},
        "source_files": [source_paths[path] for path in source_files],
        "counts": {**dict(counters), "output_records": sum(output_counts.values()), "output_by_split": dict(output_counts), "rejected_records": counters["input_records"] - counters["validated_candidates"]},
        "validation_policy": "Fast dialect-neutral query check: SELECT/WITH/EXPLAIN/SHOW/DESCRIBE/PRAGMA prefix and balanced parentheses. Run dialect-specific parser and disposable-database execution tests before deployment.",
        "audit_files": ["audit/accepted_metadata.jsonl", "audit/rejected.jsonl", "audit/dedupe.sqlite3"],
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _stream_copy_nl2sql(source_dir: Path, task_dir: Path) -> dict[str, Any]:
    """Compact prepared NL-to-SQL files in one pass.

    ``training.prepare_nl2sql`` already performs global duplicate detection
    before writing these split files. Repeating a disk lookup for every row
    would make a multi-gigabyte corpus unnecessarily slow, so this pass keeps
    the source split and writes only compact prompt/answer records.
    """

    task_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = task_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    old_db = audit_dir / "dedupe.sqlite3"
    if old_db.exists():
        old_db.unlink()
    source_files = [source_dir / "nl2sql" / f"{split}.jsonl" for split in SPLITS]
    source_files = [path for path in source_files if path.exists()]
    counters: Counter[str] = Counter()
    output_counts: Counter[str] = Counter()
    source_paths = {path: str(path.relative_to(ROOT_DIR)).replace("\\", "/") for path in source_files}
    rejected_path = audit_dir / "rejected.jsonl"
    metadata_path = audit_dir / "accepted_metadata.jsonl"
    with rejected_path.open("w", encoding="utf-8") as rejected_handle, metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for split in SPLITS:
            path = source_dir / "nl2sql" / f"{split}.jsonl"
            if not path.exists():
                continue
            output_path = task_dir / f"{split}.jsonl"
            with path.open(encoding="utf-8") as input_handle, output_path.open("w", encoding="utf-8") as output_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    if not line.strip():
                        continue
                    counters["input_records"] += 1
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "reason": f"invalid_json: {exc.msg}"}, separators=(",", ":")) + "\n")
                        counters["invalid_json"] += 1
                        continue
                    if not isinstance(row, dict):
                        rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "reason": "record_is_not_an_object"}, separators=(",", ":")) + "\n")
                        counters["record_is_not_an_object"] += 1
                        continue
                    item, reason = _make_candidate(path, line_number, row, split, "nl2sql")
                    if item is None:
                        rejected_handle.write(json.dumps({"source_file": source_paths[path], "source_line": line_number, "record_id": _one_line(row.get("record_id") or row.get("id")), "reason": reason or "rejected"}, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counters[reason or "rejected"] += 1
                        continue
                    output_handle.write(json.dumps({"prompt": item["prompt"], "answer": item["answer"]}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    metadata_handle.write(json.dumps({
                        "content_key": item["_content_key"],
                        "split": split,
                        "source_file": item["_source_file"],
                        "source_line": item["_source_line"],
                        "record_id": item["_record_id"],
                        "source": item["_source"],
                        "split_group": item["_group"],
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counters["validated_candidates"] += 1
                    output_counts[split] += 1
    manifest = {
        "task": "nl2sql",
        "output_format": {"fields": ["prompt", "answer"]},
        "source_files": [source_paths[path] for path in source_files],
        "counts": {**dict(counters), "output_records": sum(output_counts.values()), "output_by_split": dict(output_counts), "rejected_records": counters["input_records"] - counters["validated_candidates"]},
        "validation_policy": "Fast dialect-neutral query check: SELECT/WITH/EXPLAIN/SHOW/DESCRIBE/PRAGMA prefix and balanced parentheses. Source preparation already performed global duplicate detection. Run dialect-specific parser and disposable-database execution tests before deployment.",
        "audit_files": ["audit/accepted_metadata.jsonl", "audit/rejected.jsonl"],
    }
    (task_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def _intent_candidate(path: Path, line_number: int, row: dict[str, Any], split: str) -> tuple[dict[str, Any] | None, str | None]:
    task = _one_line(row.get("task") or "").lower()
    if task and task not in {"classify_intent", "intent_classification"} and path.name != "intent_test.jsonl":
        return None, "not_intent_record"
    question = _one_line(row.get("request") or row.get("question") or row.get("prompt"))
    intent = _one_line(row.get("intent")).upper()
    if not question:
        return None, "missing_question"
    if intent not in VALID_INTENTS:
        return None, "invalid_intent_label"
    dialect = _one_line(row.get("dialect") or "postgresql").lower()
    prompt = f"Task: intent classification\nDialect: {dialect}\nUser request: {question}"
    content_key = hashlib.sha256(f"{_key(prompt)}\n{intent}".encode("utf-8")).hexdigest()
    prompt_key = hashlib.sha256(_key(prompt).encode("utf-8")).hexdigest()
    info = _source_info(path, line_number, row, split)
    return {
        "prompt": prompt,
        "answer": intent,
        "_content_key": content_key,
        "_prompt_key": prompt_key,
        "_split": split,
        "_source_file": info["source_file"],
        "_source_line": line_number,
        "_record_id": info["record_id"],
        "_source": info["source"],
        "_group": info["split_group"],
    }, None


def _clean_intent(source_dir: Path, task_dir: Path) -> dict[str, Any]:
    source_files = [source_dir / f"{split}.jsonl" for split in SPLITS]
    holdout = source_dir / "intent_test.jsonl"
    if holdout.exists():
        source_files.append(holdout)
    source_files = [path for path in source_files if path.exists()]
    counters: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for path in source_files:
        split = "test" if path.name == "intent_test.jsonl" else path.stem
        for line_number, row, parse_error in _read_jsonl(path):
            counters["input_records"] += 1
            if parse_error:
                rejected.append({"source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"), "source_line": line_number, "reason": parse_error})
                counters["invalid_json"] += 1
                continue
            item, reason = _intent_candidate(path, line_number, row, split)
            if item is None:
                rejected.append({"source_file": str(path.relative_to(ROOT_DIR)).replace("\\", "/"), "source_line": line_number, "record_id": _one_line(row.get("record_id") or row.get("id")), "reason": reason or "rejected"})
                counters[reason or "rejected"] += 1
            else:
                candidates.append(item)
                counters["validated_candidates"] += 1
    selected = _dedupe(candidates, rejected, counters)
    return _write_task(task_dir, "intent", selected, rejected, counters, source_files)


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    ddl_manifest = clean_ddl(SOURCE_DIR, output_dir / "ddl")
    nl2sql_manifest = _stream_copy_nl2sql(SOURCE_DIR, output_dir / "nl2sql")
    intent_manifest = _clean_intent(SOURCE_DIR, output_dir / "intent")
    combined = {
        "dataset_version": "compact-clean-v1",
        "source_dir": str(SOURCE_DIR.relative_to(ROOT_DIR)).replace("\\", "/"),
        "raw_data_untouched": True,
        "tasks": {"ddl": ddl_manifest, "nl2sql": nl2sql_manifest, "intent": intent_manifest},
        "excluded_from_training_conversion": [
            "raw datasets outside data/processed",
            "rejected.jsonl audit files",
            "manifest.json files",
            "model ZIP/checkpoint artifacts",
            "previous clean/compact output directories",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
