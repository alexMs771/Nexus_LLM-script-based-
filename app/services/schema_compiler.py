"""Small, local schema planner for reliable common DDL requests.

The scratch language model is intentionally tiny and is not yet strong enough
to be the only source of executable SQL.  This module turns common natural
language table requests into a typed schema plan and renders that plan for the
supported output formats.  It does not execute SQL and it never handles
destructive requests; those are rejected by the scope gate before this module
is called.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


SUPPORTED_DIALECTS = ("postgresql", "mysql", "sqlite", "dbml")

_DIALECT_WORDS = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "pgsql": "postgresql",
    "my sql": "mysql",
    "mysql": "mysql",
    "mariadb": "mysql",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "dbml": "dbml",
    "database markup language": "dbml",
}

_TYPE_WORDS = (
    "bigint",
    "smallint",
    "integer",
    "int",
    "serial",
    "bigserial",
    "uuid",
    "varchar(?:\\(\\d+\\))?",
    "char(?:\\(\\d+\\))?",
    "text",
    "boolean",
    "bool",
    "date",
    "datetime",
    "timestamp(?:\\s+with\\s+time\\s+zone)?",
    "decimal(?:\\(\\d+\\s*,\\s*\\d+\\))?",
    "numeric(?:\\(\\d+\\s*,\\s*\\d+\\))?",
    "real",
    "float",
    "double",
    "json",
    "jsonb",
    "blob",
)
_TYPE_RE = re.compile(
    r"\b(?:bigint|smallint|integer|int|serial|bigserial|uuid|varchar|char|text|boolean|bool|date|datetime|"
    r"timestamp(?:\s+with\s+time\s+zone)?|decimal|numeric|real|float|double|json|jsonb|blob)"
    r"(?:\(\d+(?:\s*,\s*\d+)?\))?",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(primary\s+key|unique|not\s+null|null|nullable|references?\s+[\w.()]+)\b",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class ColumnSpec:
    name: str
    data_type: str = "TEXT"
    primary_key: bool = False
    unique: bool = False
    not_null: bool = False
    references: str | None = None
    increment: bool = False


@dataclass
class TableSpec:
    name: str
    columns: list[ColumnSpec] = field(default_factory=list)


@dataclass
class CompilationResult:
    output: str
    dialect: str
    table_names: list[str]
    warnings: list[str] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "local_schema_compiler"


def normalize_dialect(value: str | None) -> str:
    if not value:
        return "postgresql"
    normalized = value.strip().lower().replace("_", " ")
    return _DIALECT_WORDS.get(normalized, normalized)


def detect_dialect(requirement: str) -> str | None:
    lowered = requirement.lower()
    # Prefer the longest phrase so "database markup language" wins first.
    for phrase in sorted(_DIALECT_WORDS, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", lowered):
            return _DIALECT_WORDS[phrase]
    return None


def resolve_dialect(requested: str | None, requirement: str) -> tuple[str, str]:
    if requested:
        normalized = normalize_dialect(requested)
        return normalized, "request"
    detected = detect_dialect(requirement)
    return detected or "postgresql", "requirement" if detected else "default"


def _clean_identifier(value: str) -> str:
    value = value.strip().strip("`\"'").lower()
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        return "value"
    if value[0].isdigit():
        value = "_" + value
    return value


def _quote(value: str, dialect: str) -> str:
    if dialect == "mysql":
        return "`" + value.replace("`", "``") + "`"
    if dialect in {"postgresql", "sqlite"}:
        return '"' + value.replace('"', '""') + '"'
    return value


def _table_name(requirement: str) -> str | None:
    text = " ".join(requirement.strip().split())
    patterns = (
        r"\btable\s+(?:named|called)\s+([A-Za-z_][\w-]*)",
        r"\b(?:create|make|build|generate|define|set\s+up)\s+(?:a|an|the|new)\s+(?:\w+\s+)?table\s+(?:for\s+)?([A-Za-z_][\w-]*)",
        r"\b([A-Za-z_][\w-]*)\s+table\b",
        r"\btable\s+([A-Za-z_][\w-]*)\b",
        r"\b(?:customer|customers|employee|employees|user|users|product|products|order|orders|invoice|invoices|student|students|ticket|tickets)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if candidate.lower() in {"with", "for", "in", "using", "postgres", "postgresql", "mysql", "sqlite", "dbml"}:
                continue
            return _clean_identifier(candidate)
    return None


def _default_columns(table_name: str) -> list[ColumnSpec]:
    singular = table_name.rstrip("s")
    templates: dict[str, list[ColumnSpec]] = {
        "customer": [
            ColumnSpec("customer_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("full_name", "VARCHAR(200)", not_null=True),
            ColumnSpec("email", "VARCHAR(255)", unique=True, not_null=True),
            ColumnSpec("created_at", "TIMESTAMP", not_null=True),
        ],
        "employee": [
            ColumnSpec("employee_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("full_name", "VARCHAR(200)", not_null=True),
            ColumnSpec("email", "VARCHAR(255)", unique=True, not_null=True),
            ColumnSpec("department", "VARCHAR(120)"),
            ColumnSpec("created_at", "TIMESTAMP", not_null=True),
        ],
        "user": [
            ColumnSpec("user_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("username", "VARCHAR(100)", unique=True, not_null=True),
            ColumnSpec("email", "VARCHAR(255)", unique=True, not_null=True),
            ColumnSpec("created_at", "TIMESTAMP", not_null=True),
        ],
        "product": [
            ColumnSpec("product_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("name", "VARCHAR(200)", not_null=True),
            ColumnSpec("price", "DECIMAL(12,2)", not_null=True),
            ColumnSpec("created_at", "TIMESTAMP", not_null=True),
        ],
        "order": [
            ColumnSpec("order_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("customer_id", "BIGINT", not_null=True),
            ColumnSpec("status", "VARCHAR(40)", not_null=True),
            ColumnSpec("total_amount", "DECIMAL(12,2)", not_null=True),
            ColumnSpec("created_at", "TIMESTAMP", not_null=True),
        ],
    }
    if singular in templates:
        return [ColumnSpec(**vars(column)) for column in templates[singular]]
    return [ColumnSpec("id", "BIGINT", primary_key=True, increment=True)]


def _column_region(requirement: str) -> str:
    match = re.search(
        r"\b(?:with|including|containing|having|columns?|fields?|attributes?)\b\s*(.*)",
        requirement,
        re.IGNORECASE,
    )
    if not match:
        return ""
    region = match.group(1)
    region = re.split(
        r"\b(?:in|using|for)\s+(?:postgres(?:ql)?|mysql|mariadb|sqlite(?:3)?|dbml|database markup language)\b",
        region,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return region.strip(" .")


def _split_column_items(region: str) -> list[str]:
    if not region:
        return []
    # Treat the final natural-language conjunction as a separator, while
    # keeping expressions such as "first and last name" together when there
    # is no comma/type/constraint signal.
    region = re.sub(r"\s*,\s*", "|", region)
    region = re.sub(r"\s+and\s+(?=[A-Za-z_][\w -]*(?:\s+(?:as|is|type|integer|text|email|timestamp|date|boolean|unique|primary|not)\b|$))", "|", region, flags=re.IGNORECASE)
    items = [item.strip(" |.") for item in region.split("|") if item.strip(" |.")]
    return [re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE) for item in items]


def _infer_type(name: str, phrase: str) -> str:
    lowered = name.lower()
    if "email" in lowered:
        return "VARCHAR(255)"
    if lowered.endswith("_id") or lowered in {"id", "count", "quantity", "age", "rank", "year"}:
        return "BIGINT"
    if any(token in lowered for token in ("price", "amount", "cost", "balance", "salary", "rate")):
        return "DECIMAL(12,2)"
    if any(token in lowered for token in ("created", "updated", "deleted", "date", "time", "at")):
        return "TIMESTAMP"
    if lowered.startswith("is_") or lowered.startswith("has_") or lowered in {"active", "enabled", "verified"}:
        return "BOOLEAN"
    return "TEXT"


def _parse_column(item: str) -> ColumnSpec | None:
    phrase = re.sub(r"^(?:a|an|the)\s+", "", item.strip(), flags=re.IGNORECASE)
    if not phrase or re.fullmatch(r"(?:and|or|etc)", phrase, re.IGNORECASE):
        return None
    type_match = _TYPE_RE.search(phrase)
    explicit_type = type_match.group(0).upper() if type_match else ""
    before = phrase[: type_match.start()] if type_match else phrase
    before = re.sub(r"\b(?:as|of|with|type|data type|datatype)\b", " ", before, flags=re.IGNORECASE)
    before = _CONSTRAINT_RE.sub(" ", before)
    name_match = re.search(r"[A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z_][A-Za-z0-9_]*){0,4}", before.strip())
    if not name_match:
        return None
    name = _clean_identifier(name_match.group(0))
    if name in {"table", "tables", "column", "columns", "field", "fields", "and", "or"}:
        return None
    remainder = phrase[name_match.end() :]
    primary = bool(re.search(r"\bprimary\s+key\b|\bprimary\b|\bpk\b", phrase, re.IGNORECASE))
    unique = bool(re.search(r"\bunique\b", phrase, re.IGNORECASE))
    not_null = bool(re.search(r"\bnot\s+null\b|required|mandatory", phrase, re.IGNORECASE))
    reference_match = re.search(r"\breferences?\s+([A-Za-z_][\w-]*(?:\s*\([^)]*\))?)", phrase, re.IGNORECASE)
    reference = _clean_identifier(reference_match.group(1).split("(", 1)[0]) if reference_match else None
    if explicit_type in {"INT", "INTEGER"}:
        explicit_type = "INTEGER"
    if explicit_type == "SERIAL":
        explicit_type, primary = "BIGINT", True
    return ColumnSpec(
        name=name,
        data_type=explicit_type or _infer_type(name, remainder),
        primary_key=primary,
        unique=unique,
        not_null=not_null,
        references=reference,
        increment=primary and name.endswith("id"),
    )


def _parse_columns(requirement: str, table_name: str) -> list[ColumnSpec]:
    items = _split_column_items(_column_region(requirement))
    columns = [column for item in items if (column := _parse_column(item)) is not None]
    if not columns:
        return _default_columns(table_name)
    if not any(column.primary_key for column in columns):
        id_column = next((column for column in columns if column.name == "id" or column.name.endswith("_id")), None)
        if id_column:
            id_column.primary_key = True
            id_column.increment = True
        else:
            columns.insert(0, ColumnSpec("id", "BIGINT", primary_key=True, increment=True))
    return columns


def _render_type(data_type: str, dialect: str) -> str:
    value = data_type.upper()
    if dialect == "sqlite":
        replacements = {"BIGINT": "INTEGER", "VARCHAR(255)": "TEXT", "VARCHAR(200)": "TEXT", "VARCHAR(120)": "TEXT", "VARCHAR(100)": "TEXT", "DECIMAL(12,2)": "NUMERIC", "TIMESTAMP": "TEXT", "BOOLEAN": "INTEGER"}
        return replacements.get(value, value)
    if dialect == "mysql" and value == "TIMESTAMP WITH TIME ZONE":
        return "TIMESTAMP"
    return value


def _render_sql(table: TableSpec, dialect: str) -> str:
    lines: list[str] = []
    for column in table.columns:
        column_type = _render_type(column.data_type, dialect)
        parts = [_quote(column.name, dialect), column_type]
        if column.increment:
            if dialect == "postgresql":
                parts = [_quote(column.name, dialect), "BIGINT GENERATED ALWAYS AS IDENTITY"]
            elif dialect == "mysql":
                parts.append("AUTO_INCREMENT")
            elif dialect == "sqlite":
                parts = [_quote(column.name, dialect), "INTEGER", "PRIMARY KEY", "AUTOINCREMENT"]
        if column.primary_key and "PRIMARY KEY" not in parts:
            parts.append("PRIMARY KEY")
        if column.unique:
            parts.append("UNIQUE")
        if column.not_null and "PRIMARY KEY" not in parts:
            parts.append("NOT NULL")
        if column.references:
            parts.extend(["REFERENCES", _quote(column.references, dialect) + " (" + _quote("id", dialect) + ")"])
        lines.append("  " + " ".join(parts))
    return "CREATE TABLE " + _quote(table.name, dialect) + " (\n" + ",\n".join(lines) + "\n);"


def _dbml_type(data_type: str) -> str:
    value = data_type.lower()
    return {"bigint": "bigint", "integer": "int", "boolean": "boolean", "timestamp": "timestamp", "decimal(12,2)": "decimal"}.get(value, value)


def _render_dbml(table: TableSpec) -> str:
    lines = [f"Table {table.name} {{"]
    for column in table.columns:
        attrs: list[str] = []
        if column.primary_key:
            attrs.append("pk")
        if column.increment:
            attrs.append("increment")
        if column.unique:
            attrs.append("unique")
        if column.not_null:
            attrs.append("not null")
        suffix = " [" + ", ".join(attrs) + "]" if attrs else ""
        lines.append(f"  {column.name} {_dbml_type(column.data_type)}{suffix}")
    lines.append("}")
    return "\n".join(lines)


def compile_table_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a table-oriented request, or return None for unsupported DDL."""
    if not re.search(r"\b(create|make|build|generate|set\s+up|define|design)\b", requirement, re.IGNORECASE):
        return None
    if not re.search(r"\btable\b", requirement, re.IGNORECASE):
        return None
    table_name = _table_name(requirement)
    if not table_name or not _IDENTIFIER_RE.match(table_name):
        return None
    normalized = normalize_dialect(dialect)
    if normalized not in SUPPORTED_DIALECTS:
        return None
    table = TableSpec(table_name, _parse_columns(requirement, table_name))
    output = _render_dbml(table) if normalized == "dbml" else _render_sql(table, normalized)
    explicit_columns = bool(_column_region(requirement))
    confidence = 0.98 if explicit_columns else 0.90
    warnings = [] if explicit_columns else ["No columns were specified; a safe table template was selected."]
    return CompilationResult(output, normalized, [table.name], warnings, confidence)
