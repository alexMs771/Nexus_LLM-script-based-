"""Build a high-quality compact DDL/DBML training package.

This is source-aware curation, not random balancing.  Every valid record from
the dedicated DDL corpus is retained, including rare operations.  Large
general SQL corpora remain preserved in the legacy archive and are not copied
into the compact DDL package.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
from typing import Any, Iterable

from training.sql_format import build_ddl_prompt


ROOT = Path(__file__).resolve().parents[1]
ACTIVE = ROOT / "data" / "processed" / "colab_ready_v1"
ARCHIVE = ROOT / "data" / "old_data" / "processed_legacy_2026-10-26.zip"
TARGET = ROOT / "data" / "processed" / "high_quality_ddl_v1"
SPLITS = ("train", "val", "test")


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def key(prompt: str, answer: str) -> str:
    return hashlib.sha256(f"{prompt.strip().lower()}\n{answer.strip().lower()}".encode()).hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def assign_ddl_split(source_split: str, row: dict[str, Any]) -> str:
    """Assign a deterministic evaluation split without random balancing.

    Existing source validation/test rows are preserved so rare operations such
    as ALTER, DROP, INDEX, VIEW, and SCHEMA remain explicit evaluation cases.
    Training rows are split by schema/request group, preventing near-duplicate
    records for one schema from leaking across train and evaluation files.
    """
    if source_split in ("val", "test"):
        return source_split
    # Include the request with the schema context.  A coarse source label such
    # as ``schema_natural_request`` can contain thousands of records and must
    # not become one giant evaluation bucket.
    context = norm(row.get("schema_context"))
    question = norm(row.get("question"))
    group = f"{context}\n{question}" if context else (question or norm(row.get("split_group")) or norm(row.get("record_id")))
    digest = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    if digest < 90:
        return "train"
    if digest < 95:
        return "val"
    return "test"


def split_top_level(value: str) -> list[str]:
    result: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(value):
        if quote:
            if char == quote and (index == 0 or value[index - 1] != "\\"):
                quote = None
            continue
        if char in "'\"`":
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(value[start:index].strip())
            start = index + 1
    result.append(value[start:].strip())
    return [item for item in result if item]


def sql_to_dbml(sql: str) -> str | None:
    """Convert simple CREATE TABLE DDL to DBML for supervised DBML targets."""
    statements = [part.strip() for part in re.split(r";\s*", sql.strip()) if part.strip()]
    if not statements or any(not re.match(r"(?is)^create\s+table\b", part) for part in statements):
        return None
    tables: list[str] = []
    refs: list[str] = []
    for statement in statements:
        match = re.match(r"(?is)^create\s+table\s+(?:if\s+not\s+exists\s+)?([\w\"`\.]+)\s*\((.*)\)$", statement, flags=re.S)
        if not match:
            return None
        table = match.group(1).strip('"`')
        body = match.group(2)
        columns: list[str] = []
        primary: list[str] = []
        for item in split_top_level(body):
            upper = item.upper()
            primary_match = re.search(r"PRIMARY\s+KEY\s*\(([^)]+)\)", item, flags=re.I)
            if primary_match:
                primary.extend(part.strip(' "`') for part in primary_match.group(1).split(","))
                continue
            foreign_match = re.search(r"FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+([\w\"`\.]+)\s*\(([^)]+)\)", item, flags=re.I)
            if foreign_match:
                left = foreign_match.group(1).strip(' "`')
                target_table = foreign_match.group(2).strip('"`')
                target_column = foreign_match.group(3).strip(' "`')
                refs.append(f"Ref: {table}.{left} > {target_table}.{target_column}")
                continue
            if upper.startswith(("CONSTRAINT ", "PRIMARY KEY", "UNIQUE ", "CHECK ")):
                continue
            column_match = re.match(r"[\"`]?([\w]+)[\"`]?\s+([A-Za-z]+(?:\s*\([^)]*\))?)((?:\s+.*)?)$", item, flags=re.S)
            if not column_match:
                return None
            name, data_type, suffix = column_match.groups()
            suffix_upper = suffix.upper()
            attrs: list[str] = []
            if "PRIMARY KEY" in suffix_upper or name in primary:
                attrs.append("pk")
            if "NOT NULL" in suffix_upper:
                attrs.append("not null")
            if "UNIQUE" in suffix_upper:
                attrs.append("unique")
            attr_text = f" [{', '.join(attrs)}]" if attrs else ""
            columns.append(f"  {name} {data_type.lower()}{attr_text}")
        if not columns:
            return None
        tables.append(f"Table {table} {{\n" + "\n".join(columns) + "\n}")
    return "\n\n".join(tables + refs)


def build_dbml_prompt(requirement: str, dialect: str, task: str, schema: str) -> str:
    """Use a separate prompt contract for declarative DBML generation."""
    return (
        "<|system|>\n"
        "Output only valid DBML for the requested database schema. No prose or Markdown fences.\n"
        "<|dialect|>\n"
        f"{dialect or 'generic'}\n"
        "<|task|>\n"
        f"{task or 'generate_schema'}\n"
        "<|schema|>\n"
        f"{schema or '(none supplied)'}\n"
        "<|request|>\n"
        f"{requirement}\n"
        "<|dbml|>\n"
    )


def main() -> None:
    if TARGET.exists():
        raise RuntimeError(f"Refusing to overwrite existing target: {TARGET}")
    for required in (ACTIVE, ARCHIVE):
        if not required.exists():
            raise FileNotFoundError(required)
    (TARGET / "supervised" / "ddl_sql").mkdir(parents=True)
    (TARGET / "supervised" / "dbml").mkdir(parents=True)
    (TARGET / "out_of_scope").mkdir(parents=True)
    (TARGET / "sql_support").mkdir(parents=True)
    (TARGET / "dbml").mkdir(parents=True)
    (TARGET / "english").mkdir(parents=True)

    counts: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    seen_sql: set[str] = set()
    seen_dbml: set[str] = set()
    sql_rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    dbml_rows_by_split: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}

    with zipfile.ZipFile(ARCHIVE) as archive:
        for split in SPLITS:
            source_name = f"ddl_expanded_v4/{split}.jsonl"
            if source_name not in archive.namelist():
                continue
            with archive.open(source_name) as source:
                for raw in source:
                    row = json.loads(raw)
                    question = norm(row.get("question"))
                    answer = str(row.get("target_sql") or "").strip()
                    if not question or not answer:
                        counts["ddl_rejected_missing"] += 1
                        continue
                    operation = norm(row.get("operation")) or "DDL"
                    operations[operation] += 1
                    dialect = norm(row.get("dialect")) or "postgresql"
                    schema = norm(row.get("schema_context"))
                    prompt = build_ddl_prompt(
                        requirement=question,
                        dialect=dialect,
                        task=operation,
                        schema=schema,
                    )
                    digest = key(prompt, answer)
                    output_split = assign_ddl_split(split, row)
                    if digest not in seen_sql:
                        seen_sql.add(digest)
                        sql_rows_by_split[output_split].append({"prompt": prompt, "answer": answer})
                        counts[f"ddl_sql_{output_split}"] += 1
                    dbml = sql_to_dbml(answer)
                    if dbml:
                        dbml_prompt = build_dbml_prompt(question, dialect, operation, schema)
                        dbml_key = key(dbml_prompt, dbml)
                        if dbml_key not in seen_dbml:
                            seen_dbml.add(dbml_key)
                            dbml_rows_by_split[output_split].append({"prompt": dbml_prompt, "answer": dbml})
                            counts[f"dbml_{output_split}"] += 1

        for split in SPLITS:
            write_jsonl(TARGET / "supervised" / "ddl_sql" / f"{split}.jsonl", sql_rows_by_split[split])
            write_jsonl(TARGET / "supervised" / "dbml" / f"{split}.jsonl", dbml_rows_by_split[split])

        # Keep all cleaned OOS examples; they are small and directly useful.
        for split in SPLITS:
            source_name = f"compact_clean_v2/external/clinc_oos/{split}.jsonl"
            rows: list[dict[str, str]] = []
            with archive.open(source_name) as source:
                for raw in source:
                    row = json.loads(raw)
                    prompt, answer = norm(row.get("prompt")), norm(row.get("answer"))
                    if prompt and answer:
                        rows.append({"prompt": prompt, "answer": answer})
            counts[f"oos_{split}"] = write_jsonl(TARGET / "out_of_scope" / f"{split}.jsonl", rows)

    # Relevant cleaned SQL/English sources; no full SQaLe duplication here.
    for source_name in ("spider", "sql_create_context", "sqale_sample", "sample_databases"):
        source_dir = ACTIVE / "external" / source_name
        for split in SPLITS:
            source_file = source_dir / f"{split}.jsonl"
            if not source_file.exists():
                continue
            target = TARGET / "sql_support" / f"{source_name}_{split}.jsonl"
            with source_file.open(encoding="utf-8") as source, target.open("w", encoding="utf-8") as destination:
                count = 0
                for line in source:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    prompt, answer = norm(row.get("prompt")), str(row.get("answer") or "").strip()
                    if prompt and answer:
                        destination.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False, separators=(",", ":")) + "\n")
                        count += 1
                counts[f"sql_support_{source_name}_{split}"] = count

    # Keep the complete cleaned English and DBML documentation corpora.
    docs = ACTIVE / "external" / "database_docs" / "documents.jsonl"
    if docs.exists():
        with docs.open(encoding="utf-8") as source, (TARGET / "dbml" / "documents.jsonl").open("w", encoding="utf-8") as destination:
            for line in source:
                row = json.loads(line)
                text = str(row.get("text") or "").strip()
                if text:
                    destination.write(json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts["dbml_documents"] += 1
    for split in SPLITS:
        source_file = ACTIVE / "external" / "wikitext103_raw" / f"{split}.jsonl"
        if source_file.exists():
            target_file = TARGET / "english" / f"{split}.jsonl"
            with source_file.open(encoding="utf-8") as source, target_file.open("w", encoding="utf-8") as destination:
                for line in source:
                    row = json.loads(line)
                    text = str(row.get("text") or "").strip()
                    if text:
                        destination.write(json.dumps({"text": text}, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[f"english_{split}"] += 1

    manifest = {
        "dataset_version": "high-quality-ddl-v1",
        "purpose": "Compact high-quality English-to-DDL/DBML training package with explicit out-of-scope detection.",
        "raw_data_untouched": True,
        "not_randomly_balanced": True,
        "active_training": ["supervised/ddl_sql", "supervised/dbml", "out_of_scope"],
        "supporting_training": ["sql_support", "dbml", "english"],
        "operations_retained": dict(operations),
        "counts": dict(counts),
        "excluded_large_sources": ["full SQaLe copy; retained in raw and legacy archive"],
        "legacy_archive": "data/old_data/processed_legacy_2026-10-26.zip",
        "default_output_format": "dbml",
        "split_policy": "Original val/test rows preserved; original train rows deterministically assigned 90/5/5 by schema-context plus request hash (or request/group fallback). No random balancing.",
        "sql_output_policy": "Use SQL output for operational ALTER/DROP/INDEX/VIEW/SCHEMA commands; use DBML for declarative schema generation.",
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
