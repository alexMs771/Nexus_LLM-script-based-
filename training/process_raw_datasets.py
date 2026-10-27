"""Process every downloaded raw source into source-aware compact corpora.

This module deliberately does not merge unrelated datasets.  Every source gets
its own output directory and validation policy under ``data/processed/
compact_clean_v2/external``.  The output is suitable for later training
mixtures, while source-specific manifests and rejection audits preserve
provenance.

The processor is offline: it reads the already-downloaded Arrow/Parquet/text
files and never mutates ``data/raw`` or the existing compact datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import unicodedata
from typing import Any, Iterable, Iterator

import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, load_from_disk
import sqlglot
from sqlglot import exp

from app.config import ROOT_DIR


RAW = ROOT_DIR / "data" / "raw"
V2 = ROOT_DIR / "data" / "processed" / "compact_clean_v2"
EXTERNAL = V2 / "external"
SPLITS = ("train", "val", "test")


def configure_output(output_root: str | Path) -> None:
    """Route processing output to a caller-selected dataset directory."""
    global V2, EXTERNAL
    candidate = Path(output_root)
    if not candidate.is_absolute():
        candidate = ROOT_DIR / candidate
    V2 = candidate.resolve()
    EXTERNAL = V2 / "external"


def text(value: Any) -> str:
    if value is None:
        return ""
    value = unicodedata.normalize("NFKC", str(value))
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def one_line(value: Any) -> str:
    return re.sub(r"\s+", " ", text(value)).strip()


def key(value: str) -> str:
    return re.sub(r"\s+", " ", text(value)).strip().lower()


def strip_fence(value: str) -> str:
    return re.sub(r"^```(?:sql)?\s*|\s*```$", "", value.strip(), flags=re.I | re.S).strip()


def balanced(value: str) -> bool:
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


def sql_candidate(value: Any) -> str:
    value = strip_fence(text(value))
    value = re.sub(r"^\s*sql\s*:\s*", "", value, flags=re.I)
    return value.strip()


def query_valid(sql: str) -> tuple[bool, str]:
    """Validate one query without requiring a source-specific dialect."""
    sql = sql_candidate(sql)
    if not sql or not balanced(sql):
        return False, "empty_or_unbalanced_sql"
    first = re.sub(r"(?is)^(?:/\*.*?\*/|--[^\n]*\n|\s)+", "", sql).lstrip()
    if not re.match(r"(?is)^(?:select|with|explain|show|describe|pragma)\b", first):
        return False, "not_read_query"
    try:
        statements = sqlglot.parse(sql, error_level="RAISE")
    except Exception:
        return False, "sqlglot_parse_failed"
    if not statements or any(statement is None for statement in statements):
        return False, "sqlglot_empty"
    return True, ""


def query_shape_valid(sql: str) -> tuple[bool, str]:
    """Fast validation for very large sources.

    Full SQLGlot parsing every row of a multi-gigabyte corpus is needlessly
    expensive and can reject valid vendor-specific SQL.  Large-source passes
    use this deterministic structural gate and perform a SQLGlot audit sample.
    """
    sql = sql_candidate(sql)
    if not sql or not balanced(sql):
        return False, "empty_or_unbalanced_sql"
    first = re.sub(r"(?is)^(?:/\*.*?\*/|--[^\n]*\n|\s)+", "", sql).lstrip()
    if not re.match(r"(?is)^(?:select|with|explain|show|describe|pragma)\b", first):
        return False, "not_read_query"
    return True, ""


def ddl_statements(value: str, dialect: str | None = None) -> str:
    """Extract CREATE/ALTER schema statements from a database SQL dump."""
    raw = text(value)
    if not raw:
        return ""
    # Remove common dump commands and data statements before parsing.  This
    # keeps the schema corpus small and prevents INSERT values becoming labels.
    raw = re.sub(r"(?im)^\s*(?:GO|DELIMITER\s+\S+|SET\s+[^;]+;|USE\s+[^;]+;|PRAGMA\s+[^;]+;)\s*$", "", raw)
    try:
        parsed = sqlglot.parse(raw, read=dialect or None, error_level="IGNORE")
    except Exception:
        parsed = []
    kept: list[str] = []
    for statement in parsed:
        if isinstance(statement, (exp.Create, exp.Alter)):
            kind = str(statement.args.get("kind") or "").upper()
            if kind in {"TABLE", "VIEW", "INDEX", "SCHEMA", "TYPE", "SEQUENCE"} or isinstance(statement, exp.Alter):
                rendered = statement.sql(dialect=dialect or None).strip()
                if rendered:
                    kept.append(rendered.rstrip(";") + ";")
    if kept:
        return "\n".join(kept)
    # Some vendor dumps contain syntax SQLGlot cannot parse.  A conservative
    # fallback keeps only complete CREATE statements and never keeps DML.
    chunks = re.split(r";\s*(?:\n|$)", raw)
    fallback = []
    for chunk in chunks:
        candidate = chunk.strip()
        if re.match(r"(?is)^create\s+(?:table|view|index|schema|type|sequence)\b", candidate):
            fallback.append(candidate.rstrip(";") + ";")
    return "\n".join(fallback)


def split_for_group(group: str, *, train_ratio: int = 90, val_ratio: int = 5) -> str:
    digest = int(hashlib.sha256(key(group).encode("utf-8")).hexdigest()[:8], 16) % 100
    if digest < train_ratio:
        return "train"
    if digest < train_ratio + val_ratio:
        return "val"
    return "test"


class Writer:
    """Deduplicating JSONL writer with a compact provenance audit."""

    def __init__(self, out_dir: Path, fields: tuple[str, ...] = ("prompt", "answer"), append: bool = False) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audit = out_dir / "audit"
        self.audit.mkdir(parents=True, exist_ok=True)
        self.fields = fields
        self.append = append
        self.handles: dict[str, Any] = {}
        self.counts: Counter[str] = Counter()
        self.seen: set[str] = set()
        mode = "a" if append else "w"
        self.reject_handle = (self.audit / "rejected.jsonl").open(mode, encoding="utf-8")
        self.accept_handle = (self.audit / "accepted_metadata.jsonl").open(mode, encoding="utf-8")
        if append:
            # Rehydrate only compact hashes and split counts.  This permits a
            # large source to resume after a timeout without loading prompts or
            # schema text into memory again.
            metadata_path = self.audit / "accepted_metadata.jsonl"
            if metadata_path.exists():
                with metadata_path.open(encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            row = json.loads(line)
                            digest = row.get("content_key")
                            if digest:
                                self.seen.add(digest)
            for split in SPLITS:
                path = self.out_dir / f"{split}.jsonl"
                if path.exists():
                    with path.open(encoding="utf-8") as handle:
                        count = sum(1 for line in handle if line.strip())
                    self.counts[f"output_{split}"] = count
            rejected_path = self.audit / "rejected.jsonl"
            if rejected_path.exists():
                with rejected_path.open(encoding="utf-8") as handle:
                    rejected_count = sum(1 for line in handle if line.strip())
                self.counts["rejected_records_existing"] = rejected_count
                self.counts["rejected_records"] = rejected_count
            self.counts["input_records_existing"] = sum(self.counts[f"output_{split}"] for split in SPLITS) + self.counts["rejected_records_existing"]

    def _handle(self, split: str):
        if split not in self.handles:
            self.handles[split] = (self.out_dir / f"{split}.jsonl").open("a" if self.append else "w", encoding="utf-8")
        return self.handles[split]

    def reject(self, source: str, row: int, reason: str) -> None:
        self.counts[reason] += 1
        self.counts["rejected_records"] += 1
        self.reject_handle.write(json.dumps({"source": source, "source_row": row, "reason": reason}, ensure_ascii=False, separators=(",", ":")) + "\n")

    def write(self, split: str, record: dict[str, Any], source: str, row: int, group: str = "") -> bool:
        content = "\n".join(key(record.get(field, "")) for field in self.fields)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in self.seen:
            self.reject(source, row, "duplicate_content")
            return False
        self.seen.add(digest)
        output = {field: record[field] for field in self.fields}
        self._handle(split).write(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.accept_handle.write(json.dumps({"content_key": digest, "split": split, "source": source, "source_row": row, "group": group}, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.counts[f"output_{split}"] += 1
        return True

    def close(self, source_files: list[str], policy: str) -> dict[str, Any]:
        for handle in self.handles.values():
            handle.close()
        self.reject_handle.close()
        self.accept_handle.close()
        output = {split: self.counts[f"output_{split}"] for split in SPLITS}
        manifest = {
            "output_format": {"fields": list(self.fields)},
            "source_files": source_files,
            "counts": {**dict(self.counts), "input_records_total": self.counts.get("input_records_existing", 0) + self.counts.get("input_records", 0), "output_records": sum(output.values()), "output_by_split": output},
            "validation_policy": policy,
            "audit_files": ["audit/accepted_metadata.jsonl", "audit/rejected.jsonl"],
        }
        (self.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


def load_dataset_dict(path: Path) -> DatasetDict | Dataset:
    return load_from_disk(str(path))


def dataset_splits(dataset: DatasetDict | Dataset) -> Iterator[tuple[str, Dataset]]:
    if isinstance(dataset, DatasetDict):
        yield from dataset.items()
    else:
        yield "train", dataset


def process_spider() -> dict[str, Any]:
    source_dir = RAW / "spider"
    out = EXTERNAL / "spider"
    writer = Writer(out)
    ds = load_dataset_dict(source_dir)
    for source_split, rows in dataset_splits(ds):
        split = {"train": "train", "validation": "val"}.get(source_split, "test")
        for index, row in enumerate(rows, 1):
            question = one_line(row.get("question"))
            answer = sql_candidate(row.get("query"))
            if not question or not answer:
                writer.reject(f"raw/spider/{source_split}", index, "missing_question_or_query")
                continue
            valid, reason = query_valid(answer)
            if not valid:
                writer.reject(f"raw/spider/{source_split}", index, reason)
                continue
            prompt = f"Task: nl2sql\nDialect: sqlite\nDatabase: {one_line(row.get('db_id'))}\nUser request: {question}"
            writer.counts["input_records"] += 1
            writer.write(split, {"prompt": prompt, "answer": answer.rstrip(";") + ";"}, f"raw/spider/{source_split}", index, one_line(row.get("db_id")))
    return writer.close(["data/raw/spider"], "Official Spider train/validation split; SQLGlot validation; exact prompt+answer deduplication.")


def process_sql_create_context() -> dict[str, Any]:
    source_dir = RAW / "sql_create_context"
    out = EXTERNAL / "sql_create_context"
    writer = Writer(out)
    ds = load_dataset_dict(source_dir)
    for source_split, rows in dataset_splits(ds):
        for index, row in enumerate(rows, 1):
            question = one_line(row.get("question"))
            schema = one_line(row.get("context"))
            answer = sql_candidate(row.get("answer"))
            writer.counts["input_records"] += 1
            if not question or not schema or not answer:
                writer.reject(f"raw/sql_create_context/{source_split}", index, "missing_question_context_or_query")
                continue
            valid, reason = query_valid(answer)
            if not valid:
                writer.reject(f"raw/sql_create_context/{source_split}", index, reason)
                continue
            prompt = f"Task: nl2sql\nDialect: sqlite\nSchema context:\n{schema}\n\nUser request: {question}"
            writer.write(split_for_group(schema), {"prompt": prompt, "answer": answer.rstrip(";") + ";"}, f"raw/sql_create_context/{source_split}", index, schema)
    return writer.close(["data/raw/sql_create_context"], "SQL Create Context source has no official validation split; deterministic schema-group train/validation/test split; schema required; SQLGlot validation; prompt+answer deduplication.")


def process_spider_like(source_name: str, dialect: str = "unknown") -> dict[str, Any]:
    source_dir = RAW / source_name
    out = EXTERNAL / source_name
    writer = Writer(out)
    ds = load_dataset_dict(source_dir)
    for source_split, rows in dataset_splits(ds):
        for index, row in enumerate(rows, 1):
            question = one_line(row.get("question"))
            schema = text(row.get("schema"))
            answer = sql_candidate(row.get("query"))
            writer.counts["input_records"] += 1
            if not question or not answer:
                writer.reject(f"raw/{source_name}/{source_split}", index, "missing_question_or_query")
                continue
            valid, reason = query_valid(answer)
            if not valid:
                writer.reject(f"raw/{source_name}/{source_split}", index, reason)
                continue
            group = schema or f"{source_name}:{index}"
            prompt_parts = [f"Task: nl2sql\nDialect: {dialect}"]
            if schema:
                prompt_parts.append(f"Schema context:\n{schema}")
            prompt_parts.append(f"User request: {question}")
            writer.write(split_for_group(group), {"prompt": "\n\n".join(prompt_parts), "answer": answer.rstrip(";") + ";"}, f"raw/{source_name}/{source_split}", index, group[:120])
    return writer.close([f"data/raw/{source_name}"], "Schema-aware NL-to-SQL validation; query prefix, balanced parentheses, SQLGlot parse, group-based split, and deduplication.")


def process_bird() -> dict[str, Any]:
    source_dir = RAW / "bird_mini_dev"
    out = EXTERNAL / "bird_mini_dev"
    writer = Writer(out)
    ds = load_dataset_dict(source_dir)
    for source_split, rows in dataset_splits(ds):
        dialect = source_split.removeprefix("mini_dev_")
        for index, row in enumerate(rows, 1):
            question = one_line(row.get("question"))
            evidence = one_line(row.get("evidence"))
            answer = sql_candidate(row.get("SQL"))
            writer.counts["input_records"] += 1
            if not question or not answer:
                writer.reject(f"raw/bird_mini_dev/{source_split}", index, "missing_question_or_query")
                continue
            valid, reason = query_valid(answer)
            if not valid:
                writer.reject(f"raw/bird_mini_dev/{source_split}", index, reason)
                continue
            prompt = f"Task: nl2sql\nDialect: {dialect}\nDatabase: {one_line(row.get('db_id'))}"
            if evidence:
                prompt += f"\nEvidence: {evidence}"
            prompt += f"\nUser request: {question}"
            writer.write("test", {"prompt": prompt, "answer": answer.rstrip(";") + ";"}, f"raw/bird_mini_dev/{source_split}", index, one_line(row.get("db_id")))
    return writer.close(["data/raw/bird_mini_dev"], "BIRD mini-dev benchmark; dialect preserved by split; SQLGlot validation; test-only output; deduplication.")


def process_clinc() -> dict[str, Any]:
    source_dir = RAW / "clinc_oos"
    out = EXTERNAL / "clinc_oos"
    writer = Writer(out)
    ds = load_dataset_dict(source_dir)
    for source_split, rows in dataset_splits(ds):
        split = {"train": "train", "validation": "val", "test": "test"}.get(source_split, "test")
        label_names = rows.features["intent"].names if hasattr(rows.features["intent"], "names") else []
        for index, row in enumerate(rows, 1):
            question = one_line(row.get("text"))
            label_value = row.get("intent")
            label = label_names[int(label_value)] if label_names and isinstance(label_value, int) else one_line(label_value)
            writer.counts["input_records"] += 1
            if not question or not label:
                writer.reject(f"raw/clinc_oos/{source_split}", index, "missing_text_or_label")
                continue
            prompt = f"Task: intent classification\nUser request: {question}"
            writer.write(split, {"prompt": prompt, "answer": label}, f"raw/clinc_oos/{source_split}", index, label)
    return writer.close(["data/raw/clinc_oos"], "Official CLINC train/validation/test split; empty text removal; label normalization; prompt+label deduplication. General-domain intent data, kept separate from SQL tasks.")


def process_wikitext() -> dict[str, Any]:
    source_dir = RAW / "wikitext103_raw"
    out = EXTERNAL / "wikitext103_raw"
    writer = Writer(out, fields=("text",))
    for file_name, split in (("train-00000-of-00002.parquet", "train"), ("train-00001-of-00002.parquet", "train"), ("validation.parquet", "val"), ("test.parquet", "test")):
        path = source_dir / file_name
        if not path.exists():
            continue
        table = pq.read_table(path, columns=["text"])
        for index, value in enumerate(table.column("text").to_pylist(), 1):
            content = text(value)
            writer.counts["input_records"] += 1
            if len(content) < 20:
                writer.reject(f"raw/wikitext103_raw/{file_name}", index, "too_short_or_empty")
                continue
            writer.write(split, {"text": content}, f"raw/wikitext103_raw/{file_name}", index)
    return writer.close(["data/raw/wikitext103_raw"], "WikiText-103 official split; Unicode normalization; blank/short line removal; exact text deduplication. Language pretraining corpus, not SQL supervision.")


def process_docs() -> dict[str, Any]:
    source_dir = RAW / "database_docs"
    out = EXTERNAL / "database_docs"
    out.mkdir(parents=True, exist_ok=True)
    audit = out / "audit"
    audit.mkdir(exist_ok=True)
    output = (out / "documents.jsonl").open("w", encoding="utf-8")
    rejected = (audit / "rejected.jsonl").open("w", encoding="utf-8")
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    include_names = {"core_spec.md", "syntax_docs.txt", "cli_docs.txt", "github_readme.md", "README.md", "llms.txt"}
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"} or path.name.lower() in {"license", "license.md", "changelog.md", "contributing.md", "code_of_conduct.md"}:
            continue
        relative = path.relative_to(source_dir).as_posix()
        if path.name not in include_names and "/docs/" not in f"/{relative.lower()}/" and "/syntax/" not in f"/{relative.lower()}/":
            continue
        content = text(path.read_text(encoding="utf-8", errors="ignore"))
        counts["input_files"] += 1
        if not content:
            rejected.write(json.dumps({"source": f"raw/database_docs/{relative}", "reason": "empty"}) + "\n")
            counts["empty_files"] += 1
            continue
        paragraphs = re.split(r"\n\s*\n", content)
        chunk: list[str] = []
        size = 0
        chunk_index = 0
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            if size + len(paragraph) > 2400 and chunk:
                value = "\n\n".join(chunk).strip()
                digest = hashlib.sha256(key(value).encode()).hexdigest()
                if digest not in seen:
                    seen.add(digest)
                    output.write(json.dumps({"text": value, "source": relative, "chunk_id": chunk_index}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts["output_chunks"] += 1
                chunk_index += 1
                chunk, size = [], 0
            chunk.append(paragraph.strip())
            size += len(paragraph)
        if chunk:
            value = "\n\n".join(chunk).strip()
            digest = hashlib.sha256(key(value).encode()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                output.write(json.dumps({"text": value, "source": relative, "chunk_id": chunk_index}, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts["output_chunks"] += 1
    output.close()
    rejected.close()
    manifest = {"output_format": {"fields": ["text", "source", "chunk_id"]}, "source_files": ["data/raw/database_docs"], "counts": dict(counts), "validation_policy": "Allowlisted DBML documentation only; excludes licenses, repository history, contribution files, and empty documents; paragraph chunking and deduplication.", "audit_files": ["audit/rejected.jsonl"]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def process_sample_databases() -> dict[str, Any]:
    source_dir = RAW / "sample_databases"
    out = EXTERNAL / "sample_databases"
    writer = Writer(out)
    dialect_words = {"postgres": "postgres", "mysql": "mysql", "sqlite": "sqlite", "oracle": "oracle", "sqlserver": "tsql", "sql-server": "tsql", "db2": "db2"}
    files: list[Path] = []
    for path in source_dir.rglob("*.sql"):
        lower = path.name.lower()
        if any(skip in lower for skip in ("insert", "data", "delete", "drop", "update", "report", "test_")):
            continue
        if any(token in lower for token in ("schema", "create", "chinook_", "employees", "objects", "ddl")):
            files.append(path)
    for index, path in enumerate(sorted(files), 1):
        relative = path.relative_to(source_dir).as_posix()
        content = path.read_text(encoding="utf-8", errors="ignore")
        dialect = "unknown"
        lower = relative.lower()
        for word, value in dialect_words.items():
            if word in lower:
                dialect = value
                break
        answer = ddl_statements(content, dialect if dialect != "unknown" else None)
        writer.counts["input_files"] += 1
        if not answer:
            writer.reject(f"raw/sample_databases/{relative}", index, "no_schema_ddl_extracted")
            continue
        prompt = f"Task: database schema DDL\nDialect: {dialect}\nSource schema file: {relative}\nReturn the complete schema DDL."
        writer.write("train", {"prompt": prompt, "answer": answer}, f"raw/sample_databases/{relative}", index, relative)
    return writer.close(["data/raw/sample_databases"], "Schema-focused SQL dump processing; excludes data/DML/drop/test dumps; CREATE/ALTER extraction with SQLGlot and conservative fallback; file-level deduplication.")


def process_sqale(start_shard: int = 0, end_shard: int | None = None, append: bool = False) -> dict[str, Any]:
    source_dir = RAW / "sqale_full"
    out = EXTERNAL / "sqale_full"
    writer = Writer(out, append=append)
    all_paths = sorted(source_dir.glob("*.parquet"))
    paths = all_paths[start_shard:end_shard]
    for path in paths:
        parquet = pq.ParquetFile(path)
        row_number = 0
        for batch in parquet.iter_batches(columns=["schema", "question", "query"], batch_size=2048):
            rows = batch.to_pylist()
            for row in rows:
                row_number += 1
                writer.counts["input_records"] += 1
                question = one_line(row.get("question"))
                schema = text(row.get("schema"))
                answer = sql_candidate(row.get("query"))
                if not question or not schema or not answer:
                    writer.reject(f"raw/sqale_full/{path.name}", row_number, "missing_question_schema_or_query")
                    continue
                valid, reason = query_shape_valid(answer)
                if not valid:
                    writer.reject(f"raw/sqale_full/{path.name}", row_number, reason)
                    continue
                if writer.counts["input_records"] % 1000 == 0:
                    parsed, _ = query_valid(answer)
                    if not parsed:
                        writer.counts["sqlglot_sample_parse_failed"] += 1
                prompt = f"Task: nl2sql\nDialect: unknown\nSchema context:\n{schema}\n\nUser request: {question}"
                writer.write(split_for_group(schema), {"prompt": prompt, "answer": answer.rstrip(";") + ";"}, f"raw/sqale_full/{path.name}", row_number, schema[:160])
        print(f"processed {path.name}: {row_number} rows", flush=True)
    manifest = writer.close(["data/raw/sqale_full"], "Streaming Parquet processing; schema-group split; fast query prefix/balance validation for every row plus SQLGlot audit sampling every 1,000th input; prompt+answer deduplication.")
    manifest["shards_processed_this_run"] = [path.name for path in paths]
    manifest["shard_range"] = {"start": start_shard, "end": end_shard if end_shard is not None else len(all_paths), "total_shards": len(all_paths)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def update_root_manifest(results: dict[str, Any]) -> None:
    manifest_path = V2 / "manifest.json"
    existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing["dataset_version"] = "compact-clean-v2"
    existing["raw_data_untouched"] = True
    existing["external_sources"] = results
    existing["external_output_root"] = "data/processed/compact_clean_v2/external"
    manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


PROCESSORS = {
    "spider": process_spider,
    "sql_create_context": process_sql_create_context,
    "sqale_sample": lambda: process_spider_like("sqale_sample"),
    "sqale_full": process_sqale,
    "bird_mini_dev": process_bird,
    "clinc_oos": process_clinc,
    "wikitext103_raw": process_wikitext,
    "database_docs": process_docs,
    "sample_databases": process_sample_databases,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Process all raw datasets into source-aware compact_clean_v2 outputs.")
    parser.add_argument("--source", choices=["all", *PROCESSORS], default="all")
    parser.add_argument("--sqale-start", type=int, default=0, help="First SQaLe Parquet shard index, inclusive.")
    parser.add_argument("--sqale-end", type=int, default=None, help="SQaLe Parquet shard index, exclusive.")
    parser.add_argument("--append", action="store_true", help="Append/resume an existing source output.")
    parser.add_argument("--output-root", default="data/processed/compact_clean_v2", help="Dataset output directory; raw data is never changed.")
    args = parser.parse_args()
    configure_output(args.output_root)
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    names = list(PROCESSORS) if args.source == "all" else [args.source]
    for name in names:
        print(f"starting {name}", flush=True)
        if name == "sqale_full":
            results[name] = process_sqale(args.sqale_start, args.sqale_end, args.append)
        else:
            results[name] = PROCESSORS[name]()
        print(json.dumps({"source": name, "counts": results[name].get("counts", {})}, ensure_ascii=False), flush=True)
    if args.source == "all":
        update_root_manifest(results)
    else:
        manifest_path = V2 / "manifest.json"
        existing = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        previous = existing.get("external_sources", {})
        previous.update(results)
        existing["dataset_version"] = "compact-clean-v2"
        existing["raw_data_untouched"] = True
        existing["external_sources"] = previous
        existing["external_output_root"] = "data/processed/compact_clean_v2/external"
        manifest_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"processed": names, "output_root": str(EXTERNAL.relative_to(ROOT_DIR)).replace("\\", "/")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
