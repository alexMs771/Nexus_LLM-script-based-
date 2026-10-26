"""Build a balanced, domain-specific NL-to-DDL curriculum.

The repository's large schema-copy corpus is useful for learning SQL syntax,
but it overwhelms the small set of direct natural-language DDL labels.  This
builder keeps that corpus as a supporting source and adds deterministic,
reviewable examples for table design, constraints, relationships, indexes,
views, and dialect wording.  It creates training data only; the API does not
use these templates to answer requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

from app.config import ROOT_DIR


OUT_DIR = ROOT_DIR / "data" / "processed" / "ddl_balanced_v1"
EVAL_DIR = ROOT_DIR / "data" / "processed" / "ddl_eval_v2"


SIMPLE_TRAIN = [
    ("accounts", [("account_name", "string", "account name"), ("email", "string", "unique email address")]),
    ("invoices", [("invoice_number", "string", "unique invoice number"), ("issued_on", "date", "issue date"), ("total", "money", "invoice total")]),
    ("vehicles", [("registration_number", "string", "unique registration number"), ("make", "string", "vehicle make"), ("model", "string", "vehicle model")]),
    ("warehouses", [("warehouse_name", "string", "warehouse name"), ("city", "string", "warehouse city")]),
    ("shipments", [("tracking_code", "string", "unique tracking code"), ("shipped_on", "date", "shipping date"), ("delivered", "bool", "delivery flag")]),
    ("subscriptions", [("customer_email", "string", "customer email"), ("plan_name", "string", "subscription plan"), ("starts_at", "timestamp", "start time")]),
    ("workshops", [("title", "string", "workshop title"), ("capacity", "number", "maximum capacity")]),
    ("recipes", [("recipe_name", "string", "recipe name"), ("prep_minutes", "number", "preparation time in minutes")]),
    ("airports", [("airport_code", "string", "unique airport code"), ("airport_name", "string", "airport name"), ("country", "string", "country name")]),
    ("devices", [("serial_number", "string", "unique serial number"), ("device_type", "string", "device type"), ("active", "bool", "active flag")]),
    ("ship_logs", [("vessel_name", "string", "vessel name"), ("logged_at", "timestamp", "log timestamp"), ("latitude", "number", "latitude")]),
    ("alerts", [("alert_type", "string", "alert type"), ("message", "string", "alert message"), ("created_at", "timestamp", "creation time")]),
    ("employees", [("full_name", "string", "full name"), ("work_email", "string", "unique work email"), ("hired_on", "date", "hire date")]),
    ("departments", [("department_name", "string", "unique department name"), ("cost_center", "string", "cost center")]),
    ("projects", [("project_name", "string", "project name"), ("status", "string", "project status"), ("budget", "money", "project budget")]),
    ("tickets", [("subject", "string", "ticket subject"), ("priority", "string", "ticket priority"), ("opened_at", "timestamp", "opening time")]),
    ("deliveries", [("recipient_name", "string", "recipient name"), ("address", "string", "delivery address"), ("delivered_on", "date", "delivery date")]),
    ("courses", [("course_code", "string", "unique course code"), ("title", "string", "course title")]),
    ("lessons", [("lesson_title", "string", "lesson title"), ("duration_minutes", "number", "lesson duration")]),
    ("playlists", [("playlist_name", "string", "playlist name"), ("is_public", "bool", "public flag")]),
    ("songs", [("song_title", "string", "song title"), ("artist_name", "string", "artist name"), ("released_on", "date", "release date")]),
    ("exhibits", [("exhibit_name", "string", "exhibit name"), ("opened_on", "date", "opening date")]),
    ("research_samples", [("sample_code", "string", "unique sample code"), ("collected_at", "timestamp", "collection time"), ("weight", "number", "sample weight")]),
    ("garden_plots", [("plot_name", "string", "plot name"), ("soil_type", "string", "soil type"), ("irrigated", "bool", "irrigation flag")]),
    ("bookings", [("reference_code", "string", "unique booking reference"), ("booked_for", "date", "booking date"), ("guest_count", "number", "number of guests")]),
    ("payments", [("payment_reference", "string", "unique payment reference"), ("amount", "money", "payment amount"), ("paid_at", "timestamp", "payment time")]),
    ("packages", [("package_code", "string", "unique package code"), ("weight", "number", "package weight"), ("fragile", "bool", "fragile flag")]),
    ("reviews", [("reviewer_name", "string", "reviewer name"), ("rating", "number", "rating value"), ("comment", "string", "review comment")]),
    ("sensors", [("sensor_code", "string", "unique sensor code"), ("reading", "number", "sensor reading"), ("recorded_at", "timestamp", "reading time")]),
    ("events", [("event_name", "string", "event name"), ("starts_at", "timestamp", "start time"), ("location", "string", "event location")]),
    ("members", [("member_number", "string", "unique member number"), ("full_name", "string", "full name"), ("joined_on", "date", "membership date")]),
    ("campaigns", [("campaign_name", "string", "campaign name"), ("starts_on", "date", "campaign start date"), ("active", "bool", "active flag")]),
    ("documents", [("document_title", "string", "document title"), ("file_name", "string", "file name"), ("uploaded_at", "timestamp", "upload time")]),
]

SIMPLE_EVAL = [
    ("clinics", [("clinic_name", "string", "clinic name"), ("phone_number", "string", "contact phone number")]),
    ("authors", [("pen_name", "string", "author pen name"), ("born_on", "date", "birth date")]),
    ("libraries", [("library_name", "string", "library name"), ("postal_code", "string", "postal code")]),
    ("courses_catalog", [("course_title", "string", "course title"), ("version", "string", "course version")]),
    ("farms", [("farm_name", "string", "farm name"), ("acreage", "number", "farm acreage"), ("organic", "bool", "organic flag")]),
    ("music_albums", [("album_title", "string", "album title"), ("artist", "string", "artist name"), ("released_at", "date", "release date")]),
    ("orders_archive", [("order_number", "string", "unique order number"), ("customer_name", "string", "customer name"), ("total", "money", "order total")]),
    ("weather_readings", [("station_code", "string", "station code"), ("temperature", "number", "temperature"), ("observed_at", "timestamp", "observation time")]),
    ("laboratories", [("lab_name", "string", "laboratory name"), ("license_code", "string", "unique license code")]),
    ("courses_archive", [("course_name", "string", "course name"), ("release_version", "string", "release version")]),
    ("parks", [("park_name", "string", "park name"), ("opened_on", "date", "opening date"), ("accessible", "bool", "accessibility flag")]),
    ("repair_jobs", [("job_number", "string", "unique job number"), ("description", "string", "repair description"), ("cost", "money", "repair cost")]),
]

REL_TRAIN = [
    ("clinics_train", "visits_train", "clinic_id", "visit_id", "visit_date"),
    ("publishers", "titles", "publisher_id", "title_id", "published_on"),
    ("schools", "classes", "school_id", "class_id", "starts_on"),
    ("stores", "receipts", "store_id", "receipt_id", "purchased_at"),
    ("teams", "matches", "team_id", "match_id", "played_on"),
    ("farms_train", "harvests", "farm_id", "harvest_id", "harvested_on"),
]

REL_EVAL = [
    ("galleries", "paintings", "gallery_id", "painting_id", "displayed_on"),
    ("theaters", "performances_eval", "theater_id", "performance_id", "starts_at"),
    ("laboratories_eval", "experiments", "laboratory_id", "experiment_id", "started_at"),
    ("airlines", "flights_eval", "airline_id", "flight_id", "departure_at"),
]

JUNCTION_TRAIN = [
    ("articles", "labels", "article_labels", "article_id", "label_id"),
    ("students_train", "clubs", "student_clubs", "student_id", "club_id"),
    ("movies", "genres", "movie_genres", "movie_id", "genre_id"),
    ("recipes_train", "ingredients", "recipe_ingredients", "recipe_id", "ingredient_id"),
]

JUNCTION_EVAL = [
    ("albums_eval", "genres_eval", "album_genres", "album_id", "genre_id"),
    ("doctors", "specialties", "doctor_specialties", "doctor_id", "specialty_id"),
]


def _id(*parts: object) -> str:
    value = "ddl-balanced-v1:" + ":".join(map(str, parts))
    return hashlib.sha1(value.encode()).hexdigest()


def _type(kind: str, dialect: str) -> str:
    if dialect == "postgresql":
        return {"string": "VARCHAR(160)", "date": "DATE", "number": "NUMERIC(12,2)", "money": "NUMERIC(12,2)", "bool": "BOOLEAN", "timestamp": "TIMESTAMPTZ"}[kind]
    if dialect == "mysql":
        return {"string": "VARCHAR(160)", "date": "DATE", "number": "DECIMAL(12,2)", "money": "DECIMAL(12,2)", "bool": "BOOLEAN", "timestamp": "DATETIME"}[kind]
    return {"string": "TEXT", "date": "TEXT", "number": "REAL", "money": "REAL", "bool": "INTEGER", "timestamp": "TEXT"}[kind]


def _column_sql(name: str, kind: str, desc: str, dialect: str) -> str:
    suffix = ""
    if "unique" in desc:
        suffix = " UNIQUE"
    return f"{name} {_type(kind, dialect)} NOT NULL{suffix}"


def _simple_target(name: str, fields: list[tuple[str, str, str]], dialect: str) -> str:
    identity = {
        "postgresql": "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
        "mysql": "id BIGINT AUTO_INCREMENT PRIMARY KEY",
        "sqlite": "id INTEGER PRIMARY KEY",
    }[dialect]
    columns = [identity] + [_column_sql(field, kind, desc, dialect) for field, kind, desc in fields]
    return f"CREATE TABLE {name} ({', '.join(columns)});"


def _simple_prompts(name: str, fields: list[tuple[str, str, str]]) -> list[str]:
    descriptions = ", ".join(f"{field} ({desc})" for field, _, desc in fields)
    return [
        f"Create a {name} table with an auto-generated id and these fields: {descriptions}.",
        f"Please set up a database table named {name}; include {descriptions}.",
        f"I need a table for {name} that stores {descriptions}.",
        f"Can you create {name} with an id plus {descriptions}?",
        f"Build the {name} table. Its columns should cover {descriptions}.",
        f"We need to persist {descriptions} in a table called {name}.",
        f"Add a new {name} table with a generated primary key and {descriptions}.",
        f"Define {name} for the application, using columns {descriptions}.",
    ]


def _row(question: str, target: str, dialect: str, operation: str = "CREATE_TABLE", group: str = "balanced") -> dict:
    return {
        "record_id": _id(question, target, dialect),
        "source_id": "balanced_ddl_curriculum_v1",
        "source_split": "train",
        "task_type": "ddl_generation",
        "dialect": dialect,
        "operation": operation,
        "intent": "IN_SCOPE",
        "question": question,
        "evidence": "",
        "schema_context": "",
        "target_sql": target,
        "split_group": group,
        "source": "curated_domain_supervision",
    }


def _simple_rows(items: Iterable[tuple[str, list[tuple[str, str, str]]]], split_group: str) -> list[dict]:
    rows: list[dict] = []
    for name, fields in items:
        for dialect in ("postgresql", "mysql", "sqlite"):
            target = _simple_target(name, fields, dialect)
            for question in _simple_prompts(name, fields):
                rows.append(_row(question, target, dialect, group=split_group))
    return rows


def _relationship_rows(items: Iterable[tuple[str, str, str, str, str]], split_group: str) -> list[dict]:
    rows: list[dict] = []
    for parent, child, parent_id, child_id, date_col in items:
        target = (
            f"CREATE TABLE {parent} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, name VARCHAR(160) NOT NULL);\n"
            f"CREATE TABLE {child} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, {parent_id} BIGINT NOT NULL REFERENCES {parent} (id), {date_col} TIMESTAMPTZ NOT NULL);"
        )
        prompts = [
            f"Create {parent} and {child} tables. Each {child[:-1] if child.endswith('s') else child} belongs to one {parent[:-1] if parent.endswith('s') else parent}.",
            f"Set up a one-to-many relationship from {parent} to {child}, with names and a date on the child records.",
            f"I need {parent} and {child}; add a foreign key from {child} to {parent}.",
            f"Build the {parent} table and its related {child} table with a required reference and timestamp.",
        ]
        for prompt in prompts:
            rows.append(_row(prompt, target, "postgresql", "CREATE_TABLE_RELATIONSHIP", split_group))
    return rows


def _junction_rows(items: Iterable[tuple[str, str, str, str, str]], split_group: str) -> list[dict]:
    rows: list[dict] = []
    for left, right, junction, left_id, right_id in items:
        target = (
            f"CREATE TABLE {left} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, name VARCHAR(160) NOT NULL);\n"
            f"CREATE TABLE {right} (id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, name VARCHAR(160) NOT NULL);\n"
            f"CREATE TABLE {junction} ({left_id} BIGINT NOT NULL REFERENCES {left} (id), {right_id} BIGINT NOT NULL REFERENCES {right} (id), PRIMARY KEY ({left_id}, {right_id}));"
        )
        prompts = [
            f"Create a many-to-many relationship between {left} and {right} using {junction}.",
            f"Set up {left} and {right} so each can have many of the other; include the {junction} table.",
            f"I need {junction} as a junction table linking {left} and {right} with foreign keys and a composite primary key.",
        ]
        for prompt in prompts:
            rows.append(_row(prompt, target, "postgresql", "CREATE_TABLE_MANY_TO_MANY", split_group))
    return rows


def _operation_rows(split_group: str) -> list[dict]:
    specs = [
        ("Create an index on invoices by invoice number.", "CREATE INDEX idx_invoices_invoice_number ON invoices (invoice_number);", "CREATE_INDEX"),
        ("Add a status column to the accounts table.", "ALTER TABLE accounts ADD COLUMN status VARCHAR(160) NOT NULL;", "ALTER_TABLE"),
        ("Create a schema named analytics.", "CREATE SCHEMA analytics;", "CREATE_SCHEMA"),
        ("Create a view of active subscriptions showing id and plan_name.", "CREATE VIEW active_subscriptions AS SELECT id, plan_name FROM subscriptions WHERE active = TRUE;", "CREATE_VIEW"),
        ("Add a unique index for the device serial number.", "CREATE UNIQUE INDEX idx_devices_serial_number ON devices (serial_number);", "CREATE_INDEX"),
        ("Add a notes column to the tickets table.", "ALTER TABLE tickets ADD COLUMN notes VARCHAR(160);", "ALTER_TABLE"),
    ]
    return [_row(question, target, "postgresql", operation, split_group) for question, target, operation in specs]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _read_rows(path: Path, limit: int = 0) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--eval-dir", type=Path, default=EVAL_DIR)
    parser.add_argument("--schema-copy-limit", type=int, default=2500)
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT_DIR / args.out_dir
    eval_dir = args.eval_dir if args.eval_dir.is_absolute() else ROOT_DIR / args.eval_dir

    rows = _simple_rows(SIMPLE_TRAIN, "curated_simple")
    rows += _relationship_rows(REL_TRAIN, "curated_relationship")
    rows += _junction_rows(JUNCTION_TRAIN, "curated_many_to_many")
    rows += _operation_rows("curated_operations")

    # Retain the original reviewed examples with broader English variants.
    original = _read_rows(ROOT_DIR / "data" / "processed" / "ddl" / "train.jsonl")
    for item in original:
        question = str(item.get("question") or "").strip().rstrip(".")
        target = str(item.get("target_sql") or "").strip()
        if not question or not target:
            continue
        for variant in (
            question,
            f"Please create the requested database structure: {question}",
            f"I need DDL for this requirement: {question}",
            f"Can you translate this request into valid {item.get('dialect', 'postgresql')} DDL? {question}",
            f"Set up the tables described by this request: {question}",
        ):
            rows.append(_row(variant, target, str(item.get("dialect") or "postgresql"), str(item.get("operation") or "GENERATE_DDL"), "reviewed_project_examples"))

    # Add schema-copy supervision, but cap it so direct NL-to-DDL labels remain
    # a substantial part of the curriculum.
    schema_rows = _read_rows(ROOT_DIR / "data" / "processed" / "ddl_expanded_v4" / "train.jsonl")
    schema_rows = [r for r in schema_rows if str(r.get("operation")) == "CREATE_TABLE_FROM_SCHEMA"][: max(0, args.schema_copy_limit)]
    for item in schema_rows:
        target = str(item.get("target_sql") or "").strip()
        schema = str(item.get("schema_context") or "").strip()
        if not target:
            continue
        copied = dict(item)
        copied["source_id"] = "balanced_schema_copy_v1"
        copied["source"] = "derived_from_prepared_schema_context"
        rows.append(copied)

    # De-duplicate by actual learning pair and shuffle reproducibly.
    unique: dict[tuple[str, str, str], dict] = {}
    for row in rows:
        key = (str(row.get("question", "")).lower(), str(row.get("target_sql", "")).lower(), str(row.get("dialect", "")))
        unique.setdefault(key, row)
    rows = list(unique.values())
    random.Random(42).shuffle(rows)
    _write(out_dir / "train.jsonl", rows)
    # Keep a small validation split from the original reviewed validation set;
    # the larger unseen quality set is written separately below.
    _write(out_dir / "val.jsonl", _read_rows(ROOT_DIR / "data" / "processed" / "ddl" / "val.jsonl"))

    eval_rows = _simple_rows(SIMPLE_EVAL, "held_out_simple")
    eval_rows += _relationship_rows(REL_EVAL, "held_out_relationship")
    eval_rows += _junction_rows(JUNCTION_EVAL, "held_out_many_to_many")
    eval_rows += _operation_rows("held_out_operations")
    # Do not use the exact same operation names as training for operation tests
    # by keeping the quality gate focused on the held-out natural table/relational
    # cases; the operation rows remain a smoke check.
    random.Random(43).shuffle(eval_rows)
    _write(eval_dir / "test.jsonl", eval_rows)
    _write(eval_dir / "val.jsonl", eval_rows[: max(1, len(eval_rows) // 5)])
    manifest = {
        "dataset_version": "ddl-balanced-v1",
        "train_records": len(rows),
        "validation_records": len(_read_rows(out_dir / "val.jsonl")),
        "quality_test_records": len(eval_rows),
        "curated_simple_train_specs": len(SIMPLE_TRAIN),
        "curated_simple_eval_specs": len(SIMPLE_EVAL),
        "schema_copy_records": len(schema_rows),
        "train_source_policy": "curated NL-to-DDL supervision plus capped schema-copy supervision; raw datasets unchanged",
        "output_dir": str(out_dir),
        "quality_eval_dir": str(eval_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (eval_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
