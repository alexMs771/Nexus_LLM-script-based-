"""Expand the accepted DDL training split with supervised schema examples.

This changes only the supervised training split.  It does not add a serving
rule engine.  The extra records are derived from real schema contexts already
present in the prepared corpora and are used to teach the Transformer how a
natural request maps to a concrete DDL target.
"""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

import sqlglot
from sqlglot import exp

from app.config import ROOT_DIR
from training.dataset import read_jsonl


SOURCE_DIR = ROOT_DIR / "data" / "processed" / "ddl"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "ddl_augmented"

_CREATE_TABLE = re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)?)", re.I)


def _id(question: str, target: str, variant: str) -> str:
    return hashlib.sha1(f"ddl-aug-v1:{variant}:{question}:{target}".encode()).hexdigest()


def _variants(question: str, dialect: str, operation: str) -> list[tuple[str, str]]:
    text = " ".join(question.split()).strip().rstrip(".")
    variants: list[tuple[str, str]] = [(text, "original")]
    if operation.upper().startswith("CREATE") or operation.upper() in {"GENERATE_DDL", "DOMAIN_DDL"}:
        variants.extend([
            (f"Please {text[0].lower() + text[1:]}", "please"),
            (f"I need DDL to {text[0].lower() + text[1:]}", "need_ddl"),
            (f"Can you {text[0].lower() + text[1:]}?", "can_you"),
            (f"Build the database structure described here: {text}", "build_structure"),
            (f"Generate SQL for this database structure: {text}", "generate_sql"),
            (f"Set up the requested database objects. Requirement: {text}", "set_up"),
        ])
    elif operation.upper().startswith("MODIFY"):
        variants.extend([
            (f"Please {text[0].lower() + text[1:]}", "please"),
            (f"I need this schema change: {text}", "schema_change"),
            (f"Apply the following database structure change: {text}", "apply_change"),
        ])
    return variants


def _copy_record(row: dict, question: str, variant: str) -> dict:
    item = dict(row)
    item["record_id"] = _id(question, str(row["target_sql"]), variant)
    item["question"] = question
    item["source_id"] = f"{row.get('source_id', 'project_ddl_v2')}:augmented"
    item["source"] = "deterministic_paraphrase_of_project_target"
    item["augmentation"] = variant
    return item


def _default_table_rows(row: dict) -> list[dict]:
    target = str(row.get("target_sql") or "")
    if not re.search(r"\bCREATE\s+TABLE\b", target, re.I) or "\n" in target:
        return []
    match = _CREATE_TABLE.search(target)
    if not match:
        return []
    name = match.group(1)
    if "." in name:
        return []
    default_target = f"CREATE TABLE {name} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY);"
    prompts = [
        f"Create a table named {name}.",
        f"Create the {name} table with a default primary key.",
        f"I need a new {name} table; use a generated id primary key.",
    ]
    rows = []
    for index, prompt in enumerate(prompts):
        item = dict(row)
        item.update(
            record_id=_id(prompt, default_target, f"default_table_{index}"),
            question=prompt,
            target_sql=default_target,
            operation="CREATE_TABLE",
            source_id="project_ddl_v2:default_convention",
            source="derived_from_project_table_name",
            augmentation="default_primary_key_convention",
        )
        rows.append(item)
    return rows


def _schema_table_name(schema: str) -> str:
    match = _CREATE_TABLE.search(schema)
    return match.group(1) if match else "the requested table"


def _schema_natural_description(schema: str) -> tuple[str, str] | None:
    """Return a natural table/column description for a simple real schema."""
    try:
        statements = sqlglot.parse(schema, read="postgres")
    except Exception:
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Create):
        return None
    table = statements[0].this
    if not isinstance(table, exp.Schema) or not isinstance(table.this, exp.Table):
        return None
    table_name = table.this.name
    columns: list[str] = []
    for column in table.find_all(exp.ColumnDef):
        column_type = column.args.get("kind")
        type_text = column_type.sql() if column_type is not None else ""
        if column.this.name and type_text:
            columns.append(f"{column.this.name} {type_text}")
    if not table_name or not columns:
        return None
    return table_name, ", ".join(columns)


def _schema_copy_rows(
    limit: int,
    variants_per_schema: int,
    default_limit: int,
    natural_limit: int,
) -> tuple[list[dict], Counter[str]]:
    """Create learned schema-to-DDL examples from the prepared NL2SQL corpus.

    These records teach the model to copy and preserve real table definitions
    when a caller supplies schema context.  The SQL comes from the prepared
    corpus; no response is generated by the serving code.
    """
    source = ROOT_DIR / "data" / "processed" / "nl2sql" / "train.jsonl"
    if not source.exists() or limit <= 0:
        return [], Counter()
    rows: list[dict] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    default_names: set[str] = set()
    schema_count = 0
    natural_count = 0
    schema_prompts = [
        "Create the table defined by the supplied schema context.",
        "Create this table from the supplied schema definition.",
        "Generate DDL for the table described by the schema below.",
        "Implement the supplied table definition in SQL.",
        "Use the provided schema context to create the requested table.",
    ]
    with source.open("r", encoding="utf-8") as handle:
        source_rows = handle
        for line in source_rows:
            if not line.strip():
                continue
            source_row = json.loads(line)
            schema = " ".join(str(source_row.get("schema_context") or "").split()).strip()
            if not schema or not re.match(r"^CREATE\s+TABLE\b", schema, re.I):
                continue
            key = schema.lower()
            if key in seen:
                continue
            seen.add(key)
            schema_count += 1
            target = schema.rstrip(";") + ";"
            table_name = _schema_table_name(schema)
            for variant_index, template in enumerate(schema_prompts[: max(1, min(5, variants_per_schema))]):
                question = template.replace("the requested table", table_name)
                rows.append({
                    "record_id": _id(question, target, f"schema_copy_{variant_index}"),
                    "source_id": "prepared_schema_context:schema_copy",
                    "source_split": "train",
                    "task_type": "ddl_generation",
                    "dialect": "postgresql",
                    "operation": "CREATE_TABLE_FROM_SCHEMA",
                    "intent": "IN_SCOPE",
                    "question": question,
                    "evidence": "",
                    "schema_context": schema,
                    "target_sql": target,
                    "split_group": "schema_copy",
                    "source": "derived_from_prepared_schema_context",
                    "augmentation": f"schema_copy_{variant_index}",
                })
                counts[f"schema_copy_{variant_index}"] += 1

            natural_description = _schema_natural_description(schema)
            if natural_description is not None and natural_count < natural_limit:
                described_name, columns = natural_description
                natural_prompts = (
                    f"Create table {described_name} with columns: {columns}.",
                    f"Define a {described_name} table with these fields: {columns}.",
                )
                for variant_index, question in enumerate(natural_prompts):
                    rows.append({
                        "record_id": _id(question, target, f"schema_natural_{variant_index}"),
                        "source_id": "prepared_schema_context:natural_request",
                        "source_split": "train",
                        "task_type": "ddl_generation",
                        "dialect": "postgresql",
                        "operation": "CREATE_TABLE_FROM_SCHEMA",
                        "intent": "IN_SCOPE",
                        "question": question,
                        "evidence": "",
                        "schema_context": "",
                        "target_sql": target,
                        "split_group": "schema_natural_request",
                        "source": "derived_from_prepared_schema_context",
                        "augmentation": f"schema_natural_{variant_index}",
                    })
                    counts[f"schema_natural_{variant_index}"] += 1
                natural_count += 1

            if len(default_names) < default_limit and table_name != "the requested table" and table_name not in default_names:
                default_names.add(table_name)
                default_target = f"CREATE TABLE {table_name} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY);"
                for variant_index, question in enumerate((
                    f"Create a table named {table_name}.",
                    f"Create the {table_name} table with a generated id primary key.",
                )):
                    rows.append({
                        "record_id": _id(question, default_target, f"schema_default_{variant_index}"),
                        "source_id": "prepared_schema_context:default_convention",
                        "source_split": "train",
                        "task_type": "ddl_generation",
                        "dialect": "postgresql",
                        "operation": "CREATE_TABLE",
                        "intent": "IN_SCOPE",
                        "question": question,
                        "evidence": "",
                        "schema_context": "",
                        "target_sql": default_target,
                        "split_group": "schema_default_convention",
                        "source": "derived_from_prepared_schema_table_name",
                        "augmentation": f"schema_default_{variant_index}",
                    })
                    counts[f"schema_default_{variant_index}"] += 1
            if schema_count >= limit:
                break
    return rows, counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare augmented DDL training data.")
    parser.add_argument("--schema-examples", type=int, default=3000, help="Unique prepared schema contexts to add to train.")
    parser.add_argument("--schema-variants", type=int, default=4, help="Natural request phrasings per schema context (1-5).")
    parser.add_argument("--schema-default-examples", type=int, default=0, help="Distinct schema-derived table names to use for default-id examples.")
    parser.add_argument("--schema-natural-examples", type=int, default=0, help="Distinct simple schemas to turn into explicit natural table/column requests.")
    parser.add_argument("--core-repeat", type=int, default=1, help="Repeat non-schema DDL instruction rows for balanced supervised training.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory; the raw and held-out source files are never changed.")
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT_DIR / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    for row in read_jsonl(SOURCE_DIR / "train.jsonl"):
        question = str(row.get("question") or "").strip()
        target = str(row.get("target_sql") or "").strip()
        if not question or not target:
            continue
        dialect = str(row.get("dialect") or "postgresql")
        operation = str(row.get("operation") or "generate_ddl")
        for variant_question, variant in _variants(question, dialect, operation):
            key = (variant_question.lower(), target.lower(), dialect.lower())
            if key not in seen:
                seen.add(key)
                train_rows.append(_copy_record(row, variant_question, variant))
                counts[variant] += 1
        for item in _default_table_rows(row):
            key = (item["question"].lower(), item["target_sql"].lower(), dialect.lower())
            if key not in seen:
                seen.add(key)
                train_rows.append(item)
                counts["default_primary_key_convention"] += 1

    schema_rows, schema_counts = _schema_copy_rows(
        args.schema_examples,
        args.schema_variants,
        args.schema_default_examples,
        args.schema_natural_examples,
    )
    for item in schema_rows:
        key = (item["question"].lower(), item["target_sql"].lower(), item["dialect"].lower())
        if key not in seen:
            seen.add(key)
            train_rows.append(item)
            counts[item.get("augmentation", "schema_copy")] += 1

    # The schema-copy corpus is intentionally broad, but the core DDL
    # instruction set is the behavior we need for requests without schema
    # context. Repeat those labeled rows explicitly so they are not drowned
    # out by thousands of copy examples.
    repeat_count = max(1, args.core_repeat)
    if repeat_count > 1:
        expanded: list[dict] = []
        for item in train_rows:
            expanded.append(item)
            if str(item.get("source_id", "")).startswith("prepared_schema_context:"):
                continue
            for repeat_index in range(1, repeat_count):
                repeated = dict(item)
                repeated["record_id"] = _id(str(item["question"]), str(item["target_sql"]), f"repeat_{repeat_index}")
                repeated["augmentation"] = f"{item.get('augmentation', 'core')}:repeat_{repeat_index}"
                expanded.append(repeated)
        train_rows = expanded
        counts["core_repeated_records"] = sum(
            1 for item in train_rows
            if not str(item.get("source_id", "")).startswith("prepared_schema_context:")
        )

    # Mix derived schema records and project-labeled instruction records while
    # keeping generation deterministic.  This prevents a bounded streaming
    # shuffle in the trainer from seeing one source family first every epoch.
    random.Random(42).shuffle(train_rows)
    train_path = output_dir / "train.jsonl"
    train_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in train_rows), encoding="utf-8")
    for split in ("val", "test"):
        source = SOURCE_DIR / f"{split}.jsonl"
        destination = output_dir / f"{split}.jsonl"
        rows = list(read_jsonl(source))
        destination.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "dataset_version": "ddl-instruction-augmented-v1",
        "source_manifest": str(SOURCE_DIR / "manifest.json"),
        "train_records": len(train_rows),
        "validation_records": sum(1 for _ in read_jsonl(output_dir / "val.jsonl")),
        "test_records": sum(1 for _ in read_jsonl(output_dir / "test.jsonl")),
        "augmentation_counts": dict(counts),
        "schema_examples_requested": args.schema_examples,
        "schema_variants": max(1, min(5, args.schema_variants)),
        "schema_default_examples": args.schema_default_examples,
        "schema_natural_examples": args.schema_natural_examples,
        "core_repeat": repeat_count,
        "schema_source_counts": dict(schema_counts),
        "output_dir": str(output_dir),
        "policy": "Targets remain project-labeled DDL or exact schema definitions copied from prepared source contexts; derived prompts are marked in source/augmentation fields.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
