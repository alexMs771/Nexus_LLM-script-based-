"""Prepare real natural-language-to-SQL records from the downloaded corpus.

This pipeline never invents a question, schema, or SQL answer.  It only emits
an NL-to-SQL record when the source contains the corresponding fields.  Raw
files are read-only; outputs go to ``data/intermediate``, ``data/processed``,
``data/manifests``, and ``data/reports``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq

from app.config import ROOT_DIR


RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed" / "nl2sql"
INTERMEDIATE_DIR = ROOT_DIR / "data" / "intermediate"
MANIFEST_DIR = ROOT_DIR / "data" / "manifests"
REPORT_DIR = ROOT_DIR / "data" / "reports"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _jsonl_write(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _iter_parquet_rows(path: Path, columns: list[str], batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema.names)
    selected = [column for column in columns if column in available]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=selected):
        yield from batch.to_pylist()


def _iter_dataset_rows(path: Path, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(str(path))[split]
    for row in dataset:
        yield row


class Nl2SqlWriter:
    def __init__(self, limit_per_source: int = 0) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.handles = {
            name: (PROCESSED_DIR / f"{name}.jsonl").open("w", encoding="utf-8")
            for name in ("train", "val", "test", "rejected")
        }
        self.limit_per_source = limit_per_source
        self.seen: set[str] = set()
        self.source_counts: Counter[str] = Counter()
        self.split_counts: Counter[str] = Counter()
        self.rejected_counts: Counter[str] = Counter()
        self.duplicate_count = 0
        self.group_to_split: dict[str, str] = {}

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()

    def _split_for_group(self, group_key: str) -> str:
        if group_key not in self.group_to_split:
            digest = int(hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:8], 16) % 100
            self.group_to_split[group_key] = "test" if digest >= 90 else ("val" if digest >= 80 else "train")
        return self.group_to_split[group_key]

    def add(
        self,
        *,
        source_id: str,
        source_split: str,
        index: int,
        question: Any,
        schema_context: Any,
        target_sql: Any,
        dialect: str,
        group_key: Any,
        database_id: Any = "",
        evidence: Any = "",
    ) -> None:
        question_text = _clean_text(question)
        target_text = _text(target_sql)
        schema_text = _text(schema_context)
        if not question_text or not target_text:
            self.rejected_counts[source_id] += 1
            _jsonl_write(self.handles["rejected"], {"source_id": source_id, "index": index, "reason": "missing_question_or_sql"})
            return
        duplicate_key = hashlib.sha256(
            "\x1f".join((question_text.lower(), schema_text, target_text, dialect)).encode("utf-8")
        ).hexdigest()
        if duplicate_key in self.seen:
            self.duplicate_count += 1
            self.rejected_counts[source_id] += 1
            _jsonl_write(self.handles["rejected"], {"source_id": source_id, "index": index, "reason": "duplicate", "duplicate_key": duplicate_key})
            return
        self.seen.add(duplicate_key)
        group = _clean_text(group_key) or f"{source_id}:{index}"
        split = self._split_for_group(f"{source_id}:{group}")
        record = {
            "record_id": hashlib.sha1(f"{source_id}:{source_split}:{index}".encode("utf-8")).hexdigest(),
            "source_id": source_id,
            "source_split": source_split,
            "task_type": "text_to_sql",
            "dialect": dialect,
            "database_id": _clean_text(database_id),
            "question": question_text,
            "evidence": _clean_text(evidence),
            "schema_context": schema_text,
            "schema_available": bool(schema_text),
            "target_sql": target_text,
            "split_group": group,
            "duplicate_key": duplicate_key,
        }
        _jsonl_write(self.handles[split], record)
        self.source_counts[source_id] += 1
        self.split_counts[split] += 1


def _prepare_sql_create_context(writer: Nl2SqlWriter) -> None:
    path = RAW_DIR / "sql_create_context"
    for index, row in enumerate(_iter_dataset_rows(path, "train")):
        if writer.limit_per_source and index >= writer.limit_per_source:
            break
        writer.add(
            source_id="sql_create_context",
            source_split="train",
            index=index,
            question=row.get("question"),
            schema_context=row.get("context"),
            target_sql=row.get("answer"),
            dialect="unknown",
            group_key=row.get("context"),
        )


def _prepare_bird(writer: Nl2SqlWriter) -> None:
    path = RAW_DIR / "bird_mini_dev"
    dialects = {"mini_dev_mysql": "mysql", "mini_dev_pg": "postgresql", "mini_dev_sqlite": "sqlite"}
    for split, dialect in dialects.items():
        for index, row in enumerate(_iter_dataset_rows(path, split)):
            if writer.limit_per_source and index >= writer.limit_per_source:
                break
            writer.add(
                source_id="bird_mini_dev",
                source_split=split,
                index=index,
                question=row.get("question"),
                schema_context="",
                target_sql=row.get("SQL"),
                dialect=dialect,
                database_id=row.get("db_id"),
                evidence=row.get("evidence"),
                group_key=row.get("db_id"),
            )


def _prepare_spider(writer: Nl2SqlWriter) -> None:
    path = RAW_DIR / "spider"
    for source_split in ("train", "validation"):
        for index, row in enumerate(_iter_dataset_rows(path, source_split)):
            if writer.limit_per_source and index >= writer.limit_per_source:
                break
            writer.add(
                source_id="spider",
                source_split=source_split,
                index=index,
                question=row.get("question"),
                schema_context="",
                target_sql=row.get("query"),
                dialect="unknown",
                database_id=row.get("db_id"),
                group_key=row.get("db_id"),
            )


def _prepare_sqale(writer: Nl2SqlWriter) -> None:
    files = sorted((RAW_DIR / "sqale_full").glob("*.parquet"))
    for path in files:
        for index, row in enumerate(_iter_parquet_rows(path, ["schema", "question", "query"])):
            if writer.limit_per_source and index >= writer.limit_per_source:
                break
            schema = row.get("schema", "")
            writer.add(
                source_id="sqale_full",
                source_split=path.stem,
                index=index,
                question=row.get("question"),
                schema_context=schema,
                target_sql=row.get("query"),
                dialect="unknown",
                group_key=hashlib.sha1(_text(schema).encode("utf-8")).hexdigest(),
            )


def _write_general_record(handle, source_id: str, record_id: str, text: str, task_type: str) -> int:
    clean = _text(text).strip()
    if not clean:
        return 0
    _jsonl_write(handle, {"record_id": record_id, "source_id": source_id, "task_type": task_type, "text": clean})
    return 1


def _prepare_general_text(max_records: int = 0) -> dict[str, int]:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERMEDIATE_DIR / "general_text.jsonl"
    counts: Counter[str] = Counter()
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        docs = RAW_DIR / "database_docs"
        for path in sorted(docs.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".dbml"}:
                continue
            if "dbml_repository" in path.parts and path.suffix.lower() != ".dbml":
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if _write_general_record(handle, "database_docs", str(path.relative_to(docs)), text, "documentation"):
                counts["database_docs"] += 1
                written += 1
            if max_records and written >= max_records:
                return dict(counts)

        for path in sorted((RAW_DIR / "wikitext103_raw").glob("*.parquet")):
            for index, row in enumerate(_iter_parquet_rows(path, ["text"])):
                if max_records and written >= max_records:
                    return dict(counts)
                if _write_general_record(handle, "wikitext103_raw", f"{path.stem}:{index}", row.get("text", ""), "general_text"):
                    counts["wikitext103_raw"] += 1
                    written += 1

        for root_name in ("postgres-sample-dbs", "mysql-employees-db", "mysql", "sqlite"):
            root = RAW_DIR / "sample_databases" / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if max_records and written >= max_records:
                    return dict(counts)
                if not path.is_file() or path.suffix.lower() not in {".sql", ".dump", ".txt", ".md"}:
                    continue
                text = path.read_text(encoding="utf-8", errors="replace")
                if _write_general_record(handle, f"sample_databases:{root_name}", str(path.relative_to(root)), text, "sql_corpus"):
                    counts[f"sample_databases:{root_name}"] += 1
                    written += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare real NL-to-SQL and general-text corpora from data/raw.")
    parser.add_argument("--limit-per-source", type=int, default=0, help="Limit each NL-to-SQL source for a smoke run; 0 means all rows.")
    parser.add_argument("--max-general-records", type=int, default=0, help="Limit general-text records; 0 means all supported records.")
    parser.add_argument("--skip-general-text", action="store_true")
    parser.add_argument("--only-general-text", action="store_true", help="Prepare only the causal-pretraining corpus and preserve the NL-to-SQL outputs.")
    args = parser.parse_args()

    if args.only_general_text:
        general_counts = _prepare_general_text(args.max_general_records)
        manifest = {
            "dataset_version": "general-text-v1",
            "raw_directory": str(RAW_DIR),
            "records": general_counts,
            "limits": {"max_general_records": args.max_general_records},
            "policy": "Only source text is copied; no SQL question or answer is invented.",
        }
        MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (MANIFEST_DIR / "general_text_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (REPORT_DIR / "general_text_quality.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return

    writer = Nl2SqlWriter(limit_per_source=args.limit_per_source)
    try:
        _prepare_sql_create_context(writer)
        _prepare_bird(writer)
        _prepare_spider(writer)
        _prepare_sqale(writer)
    finally:
        writer.close()

    general_counts = {} if args.skip_general_text else _prepare_general_text(args.max_general_records)
    manifest = {
        "dataset_version": "nl2sql-v1",
        "raw_directory": str(RAW_DIR),
        "sources": ["sql_create_context", "bird_mini_dev", "spider", "sqale_full", "database_docs", "wikitext103_raw", "sample_databases"],
        "records": dict(writer.source_counts),
        "splits": dict(writer.split_counts),
        "rejected": dict(writer.rejected_counts),
        "duplicates_removed": writer.duplicate_count,
        "general_text_records": general_counts,
        "schema_policy": "schema_context is empty when the source does not provide one; no schema is fabricated",
        "split_policy": "deterministic split by source database/schema group; duplicate removal precedes splitting",
        "limits": {"limit_per_source": args.limit_per_source, "max_general_records": args.max_general_records},
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFEST_DIR / "nl2sql_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (REPORT_DIR / "nl2sql_quality.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
