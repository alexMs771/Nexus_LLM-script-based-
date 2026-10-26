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
    r"(?:\(\d+(?:\s*,\s*\d+)?\))?(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(primary\s+key|unique|not\s+null|null|nullable|references?\s+[\w.()]+)\b",
    re.IGNORECASE,
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENTIFIER_TOKEN = r"[A-Za-z_][A-Za-z0-9_-]*"
_TABLE_ACTION_RE = re.compile(
    r"\b(?:create|make|build|generate|define|design|set\s+up|need|want|require)\b"
    r"|\bgive\s+me\b",
    re.IGNORECASE,
)
_RESERVED_NAME_WORDS = {
    "a", "an", "the", "new", "with", "for", "in", "using", "on", "as",
    "and", "or", "to", "inside", "it", "this", "that",
    "postgres", "postgresql", "pgsql", "mysql", "mariadb", "sqlite", "sqlite3", "dbml",
    "database", "table", "tables", "column", "columns", "field", "fields",
}


@dataclass
class ColumnSpec:
    name: str
    data_type: str = "TEXT"
    primary_key: bool = False
    unique: bool = False
    not_null: bool = False
    references: str | None = None
    reference_column: str = "id"
    default: str | None = None
    check: str | None = None
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
    operation: str = "create_table"
    plan: dict[str, object] = field(default_factory=dict)


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
    if "." in value:
        return ".".join(_quote(part, dialect) for part in value.split("."))
    if dialect == "mysql":
        return "`" + value.replace("`", "``") + "`"
    if dialect in {"postgresql", "sqlite"}:
        return '"' + value.replace('"', '""') + '"'
    return value


def _table_name(requirement: str) -> str | None:
    text = " ".join(requirement.strip().split())
    patterns = (
        rf"\b(?:create|make|build|generate|define|design|set\s+up|need|want|require)\s+"
        rf"(?:a|an|the|new)?\s*table\s+(?:named|called)\s+({_IDENTIFIER_TOKEN})",
        rf"\btable\s+(?:named|called)\s+({_IDENTIFIER_TOKEN})",
        rf"\b(?:create|make|build|generate|define|design|set\s+up|need|want|require)\s+"
        rf"(?:a|an|the|new)?\s*table\s+(?!with\b|for\b|in\b|using\b|on\b)({_IDENTIFIER_TOKEN})",
        rf"\btable\s+for\s+({_IDENTIFIER_TOKEN})",
        rf"\b(?:create|make|build|generate|define|design|set\s+up|need|want|require)\s+"
        rf"(?:a|an|the|new)?\s*({_IDENTIFIER_TOKEN})\s+table\b",
        rf"\btable\s+(?!with\b|for\b|in\b|using\b|named\b|called\b)({_IDENTIFIER_TOKEN})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if candidate.lower() in _RESERVED_NAME_WORDS:
                continue
            return _clean_identifier(candidate)
    return None


def _default_columns(table_name: str) -> list[ColumnSpec]:
    singular = _singular_name(table_name)
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
        "department": [
            ColumnSpec("department_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("department_name", "VARCHAR(120)", unique=True, not_null=True),
        ],
        "course": [
            ColumnSpec("course_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("course_code", "VARCHAR(20)", unique=True, not_null=True),
            ColumnSpec("course_name", "VARCHAR(200)", not_null=True),
        ],
        "enrollment": [
            ColumnSpec("student_id", "BIGINT", not_null=True),
            ColumnSpec("course_id", "BIGINT", not_null=True),
            ColumnSpec("enrolled_at", "DATE", not_null=True, default="CURRENT_DATE"),
        ],
        "warehouse": [
            ColumnSpec("warehouse_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("warehouse_name", "VARCHAR(120)", unique=True, not_null=True),
        ],
        "stock_item": [
            ColumnSpec("warehouse_id", "BIGINT", not_null=True),
            ColumnSpec("sku", "VARCHAR(64)", not_null=True),
            ColumnSpec("quantity", "INTEGER", not_null=True, check=">= 0"),
        ],
        "student": [
            ColumnSpec("student_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("full_name", "VARCHAR(150)", not_null=True),
        ],
        "author": [
            ColumnSpec("author_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("full_name", "VARCHAR(150)", not_null=True),
        ],
        "book": [
            ColumnSpec("book_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("title", "VARCHAR(200)", not_null=True),
            ColumnSpec("author_id", "BIGINT", not_null=True),
        ],
        "patient": [
            ColumnSpec("patient_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("full_name", "VARCHAR(150)", not_null=True),
        ],
        "appointment": [
            ColumnSpec("appointment_id", "BIGINT", primary_key=True, increment=True),
            ColumnSpec("patient_id", "BIGINT", not_null=True),
            ColumnSpec("scheduled_at", "TIMESTAMP", not_null=True),
        ],
    }
    if singular in templates:
        return [ColumnSpec(**vars(column)) for column in templates[singular]]
    return [ColumnSpec("id", "BIGINT", primary_key=True, increment=True)]


def _singular_name(value: str) -> str:
    """Return a conservative singular form for relationship inference."""
    normalized = _clean_identifier(value).lower()
    if normalized.endswith("ies") and len(normalized) > 3:
        return normalized[:-3] + "y"
    if normalized.endswith("sses"):
        return normalized[:-2]
    if normalized.endswith("s") and not normalized.endswith("ss") and len(normalized) > 1:
        return normalized[:-1]
    return normalized


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
    region = re.sub(r"^\s*(?:columns?|fields?|attributes?)\s*[:\-]?\s*", "", region, flags=re.IGNORECASE)
    region = re.sub(r"\s+(?:columns?|fields?|attributes?)\s*$", "", region, flags=re.IGNORECASE)
    return region.strip(" .:-")


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
    if "date" in lowered:
        return "DATE"
    if any(token in lowered for token in ("created", "updated", "deleted", "time", "timestamp", "_at")):
        return "TIMESTAMP"
    if lowered.startswith("is_") or lowered.startswith("has_") or lowered in {"active", "enabled", "verified"}:
        return "BOOLEAN"
    return "TEXT"


def _default_expression(phrase: str) -> str | None:
    lowered = phrase.lower()
    if re.search(r"\b(?:current\s+date|today)\b", lowered):
        return "CURRENT_DATE"
    if re.search(r"\b(?:current\s+timestamp|current\s+time)\b", lowered) or re.search(
        r"\bdefaults?(?:\s+to)?\s+now\b", lowered
    ):
        return "CURRENT_TIMESTAMP"
    match = re.search(
        r"\bdefaults?(?:\s+to)?\s+((?:'[^']*')|(?:\"[^\"]*\")|(?:[-+]?\d+(?:\.\d+)?)|(?:true|false|null))(?=\s|[.,;]|$)",
        phrase,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _apply_column_semantics(column: ColumnSpec, phrase: str) -> ColumnSpec:
    lowered = phrase.lower()
    default = _default_expression(phrase)
    if default:
        column.default = default
        if "current date" in lowered or "today" in lowered:
            column.data_type = "DATE"
        elif default == "CURRENT_TIMESTAMP":
            column.data_type = "TIMESTAMP"
    if re.search(r"\b(?:non[- ]negative|cannot be negative|must not be negative)\b", lowered):
        column.check = ">= 0"
    elif re.search(r"\b(?:positive|greater than zero|above zero)\b", lowered):
        column.check = "> 0"
    return column


def _parse_column(item: str) -> ColumnSpec | None:
    phrase = re.sub(r"^(?:a|an|the)\s+", "", item.strip(), flags=re.IGNORECASE)
    phrase = re.sub(r"\b(?:columns?|fields?|attributes?)\b", " ", phrase, flags=re.IGNORECASE)
    phrase = re.sub(r"\b(?:named|called)\b", " ", phrase, flags=re.IGNORECASE)
    phrase = " ".join(phrase.split()).strip(" .:-")
    if not phrase or re.fullmatch(r"(?:and|or|etc)", phrase, re.IGNORECASE):
        return None
    semantic_phrase = phrase
    type_match = _TYPE_RE.search(phrase)
    # In natural language, "date of birth" is a column name, not a DATE type.
    if type_match and type_match.start() == 0 and re.match(r"\s+of\b", phrase[type_match.end():], re.IGNORECASE):
        type_match = None
    explicit_type = type_match.group(0).upper() if type_match else ""
    before = phrase[: type_match.start()] if type_match else phrase
    before = re.sub(r"\b(?:as|with|type|data type|datatype)\b", " ", before, flags=re.IGNORECASE)
    before = re.sub(r"\bof\s+(?=(?:type|data\s+type|datatype)\b)", " ", before, flags=re.IGNORECASE)
    before = re.sub(r"\b(?:that|which|must|should|be|is|has|have)\b", " ", before, flags=re.IGNORECASE)
    before = re.sub(r"\b(?:non[- ]negative|positive)\b", " ", before, flags=re.IGNORECASE)
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
    reference_match = re.search(
        r"\breferences?\s+([A-Za-z_][\w-]*)(?:\s*\(\s*([A-Za-z_][\w-]*)\s*\))?",
        phrase,
        re.IGNORECASE,
    )
    reference = _clean_identifier(reference_match.group(1)) if reference_match else None
    reference_column = _clean_identifier(reference_match.group(2)) if reference_match and reference_match.group(2) else "id"
    if explicit_type in {"INT", "INTEGER"}:
        explicit_type = "INTEGER"
    if explicit_type == "SERIAL":
        explicit_type, primary = "BIGINT", True
    identity_type = explicit_type or _infer_type(name, remainder)
    increment = primary and identity_type.upper().split("(", 1)[0] in {"SMALLINT", "INTEGER", "INT", "BIGINT"}
    return _apply_column_semantics(ColumnSpec(
        name=name,
        data_type=identity_type,
        primary_key=primary,
        unique=unique,
        not_null=not_null,
        references=reference,
        reference_column=reference_column,
        increment=increment,
    ), semantic_phrase)


def _parse_columns(requirement: str, table_name: str) -> list[ColumnSpec]:
    items = _split_column_items(_column_region(requirement))
    columns = [column for item in items if (column := _parse_column(item)) is not None]
    if not columns:
        return _default_columns(table_name)
    if not any(column.primary_key for column in columns):
        # Only a literal `id` is safe to promote automatically.  Names such as
        # `customer_id` and `employee_id` are commonly foreign keys and must
        # not silently become primary keys or auto-incrementing columns.
        id_column = next((column for column in columns if column.name == "id"), None)
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
                parts = [_quote(column.name, dialect), f"{column_type} GENERATED ALWAYS AS IDENTITY"]
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
        if column.default is not None:
            parts.extend(["DEFAULT", column.default])
        if column.check:
            parts.extend(["CHECK", f"({_quote(column.name, dialect)} {column.check})"])
        if column.references:
            parts.extend(["REFERENCES", _quote(column.references, dialect) + " (" + _quote(column.reference_column, dialect) + ")"])
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
        if column.default is not None:
            attrs.append(f"default: {column.default}")
        suffix = " [" + ", ".join(attrs) + "]" if attrs else ""
        lines.append(f"  {column.name} {_dbml_type(column.data_type)}{suffix}")
    lines.append("}")
    for column in table.columns:
        if column.references:
            lines.append(f"Ref: {table.name}.{column.name} > {column.references}.{column.reference_column}")
    return "\n".join(lines)


def _column_plan(column: ColumnSpec) -> dict[str, object]:
    return {
        "name": column.name,
        "data_type": column.data_type,
        "primary_key": column.primary_key,
        "unique": column.unique,
        "not_null": column.not_null,
        "default": column.default,
        "check": column.check,
        "references": (
            {"table": column.references, "column": column.reference_column}
            if column.references
            else None
        ),
    }


def _table_plan(table: TableSpec, operation: str = "create_table") -> dict[str, object]:
    relationships = [
        {"from_column": column.name, "to_table": column.references, "to_column": column.reference_column}
        for column in table.columns
        if column.references
    ]
    return {
        "operation": operation,
        "object_type": "table",
        "name": table.name,
        "columns": [_column_plan(column) for column in table.columns],
        "relationships": relationships,
    }


def compile_table_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a table-oriented request, or return None for unsupported DDL."""
    if not _TABLE_ACTION_RE.search(requirement):
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
    return CompilationResult(
        output=output,
        dialect=normalized,
        table_names=[table.name],
        warnings=warnings,
        confidence=confidence,
        operation="create_table",
        plan=_table_plan(table),
    )


def compile_schema_table_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a simple `create schema ... and table ...` request."""
    match = re.search(
        rf"\bschema\s+(?:named|called)\s+({_IDENTIFIER_TOKEN}).*?"
        rf"(?:and\s+)?(?:a|an|the|new)?\s*"
        rf"(?:table\s+(?:named|called)\s+)?({_IDENTIFIER_TOKEN})\s+table\s+(?:inside|in)\b",
        requirement,
        re.IGNORECASE,
    )
    if not match:
        return None
    normalized = normalize_dialect(dialect)
    if normalized == "sqlite" or normalized not in SUPPORTED_DIALECTS:
        return None
    schema_name = _clean_identifier(match.group(1))
    table_name = _clean_identifier(match.group(2))
    qualified_name = f"{schema_name}.{table_name}"
    table = TableSpec(qualified_name, _default_columns(table_name))
    if normalized == "dbml":
        output = _render_dbml(table)
    else:
        output = f"CREATE SCHEMA {_quote(schema_name, normalized)};\n{_render_sql(table, normalized)}"
    plan = _table_plan(table, "create_schema_and_table")
    plan["schema"] = schema_name
    return CompilationResult(
        output=output,
        dialect=normalized,
        table_names=[qualified_name],
        confidence=0.94,
        operation="create_schema_and_table",
        plan=plan,
    )


def compile_index_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a common `index on table.column` request."""
    if not re.search(r"\bindex\b", requirement, re.IGNORECASE):
        return None
    normalized = normalize_dialect(dialect)
    if normalized not in {"postgresql", "mysql", "sqlite"}:
        return None
    match = re.search(
        rf"\bon\s+(?:the\s+)?(?:table\s+)?({_IDENTIFIER_TOKEN}(?:\.{_IDENTIFIER_TOKEN})?)",
        requirement,
        re.IGNORECASE,
    )
    if not match:
        return None
    target = match.group(1)
    target_parts = target.split(".")
    table_name = target_parts[0]
    columns = [target_parts[1]] if len(target_parts) == 2 else []
    column_match = re.search(r"\(([^)]*)\)", requirement)
    if column_match:
        columns = [_clean_identifier(value) for value in column_match.group(1).split(",") if value.strip()]
    if not columns:
        return None
    unique = bool(re.search(r"\bunique\s+index\b|\bindex\b.{0,20}\bunique\b", requirement, re.IGNORECASE))
    explicit_name = re.search(r"\bindex\s+(?:named|called)\s+([A-Za-z_][\w-]*)", requirement, re.IGNORECASE)
    index_name = _clean_identifier(explicit_name.group(1)) if explicit_name else _clean_identifier(f"idx_{table_name}_{'_'.join(columns)}")
    prefix = "CREATE UNIQUE INDEX" if unique else "CREATE INDEX"
    output = f"{prefix} {_quote(index_name, normalized)} ON {_quote(table_name, normalized)} ({', '.join(_quote(column, normalized) for column in columns)});"
    plan = {
        "operation": "create_index",
        "object_type": "index",
        "name": index_name,
        "table": table_name,
        "columns": columns,
        "unique": unique,
        "relationships": [],
    }
    return CompilationResult(
        output=output,
        dialect=normalized,
        table_names=[table_name],
        confidence=0.96,
        operation="create_index",
        plan=plan,
    )


def compile_alter_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a safe `ALTER TABLE ... ADD COLUMN ...` request."""
    normalized = normalize_dialect(dialect)
    if normalized not in {"postgresql", "mysql", "sqlite"}:
        return None
    direct = re.search(
        rf"\balter\s+table\s+({_IDENTIFIER_TOKEN}(?:\.{_IDENTIFIER_TOKEN})?)\s+"
        rf"add\s+(?:column\s+)?(?P<body>.+)$",
        requirement,
        re.IGNORECASE,
    )
    natural = re.search(
        rf"\badd\s+(?P<body>.+?)\s+(?:to|on|in)\s+(?:the\s+)?(?:table\s+)?({_IDENTIFIER_TOKEN})\b",
        requirement,
        re.IGNORECASE,
    )
    if direct:
        table_name, body = direct.group(1), direct.group("body")
    elif natural:
        body, table_name = natural.group("body"), natural.group(2)
    else:
        return None
    items = _split_column_items(body)
    columns = []
    for item in items:
        column = _parse_column(item)
        if column is not None:
            columns.append(_apply_column_semantics(column, f"{requirement} {item}"))
    if not columns:
        return None

    def render_column(column: ColumnSpec) -> str:
        parts = [_quote(column.name, normalized), _render_type(column.data_type, normalized)]
        if column.not_null:
            parts.append("NOT NULL")
        if column.default is not None:
            parts.extend(["DEFAULT", column.default])
        if column.check:
            parts.extend(["CHECK", f"({_quote(column.name, normalized)} {column.check})"])
        return " ".join(parts)

    table_name = _clean_identifier(table_name)
    clauses = ["ADD COLUMN " + render_column(column) for column in columns]
    if normalized == "sqlite":
        output = "\n".join(f"ALTER TABLE {_quote(table_name, normalized)} {clause};" for clause in clauses)
    else:
        output = f"ALTER TABLE {_quote(table_name, normalized)} " + ", ".join(clauses) + ";"
    operation = "alter_add_column" if len(columns) == 1 else "alter_add_columns"
    plan = {
        "operation": operation,
        "object_type": "columns" if len(columns) > 1 else "column",
        "name": columns[0].name if len(columns) == 1 else None,
        "table": table_name,
        "columns": [_column_plan(column) for column in columns],
        "relationships": [],
    }
    return CompilationResult(
        output=output,
        dialect=normalized,
        table_names=[table_name],
        confidence=0.94,
        operation=operation,
        plan=plan,
    )


def _multi_table_names(requirement: str) -> list[str] | None:
    """Extract two or more table names from an explicit multi-table request."""
    action = r"(?:create|make|build|generate|define|design|set\s+up|need|want|require)"
    token = rf"({_IDENTIFIER_TOKEN})"
    patterns = (
        rf"\b{action}\s+(?:the\s+|new\s+)?(?P<names>{token}(?:\s*(?:,|and)\s*{token})+)\s+tables?\b",
        rf"\b{action}\s+(?:the\s+|new\s+)?tables?\s+(?P<names>{token}(?:\s*(?:,|and)\s*{token})+)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, requirement, re.IGNORECASE)
        if not match:
            continue
        names = re.findall(_IDENTIFIER_TOKEN, match.group("names"), re.IGNORECASE)
        cleaned = [_clean_identifier(name) for name in names if name.lower() not in _RESERVED_NAME_WORDS]
        if len(cleaned) >= 2 and len(cleaned) <= 6:
            return list(dict.fromkeys(cleaned))
    return None


def _append_relationship(
    relationships: list[dict[str, str]],
    table: TableSpec,
    column: ColumnSpec,
) -> None:
    relationship = {
        "from_table": table.name,
        "from_column": column.name,
        "to_table": column.references or "",
        "to_column": column.reference_column,
    }
    if relationship not in relationships:
        relationships.append(relationship)


def _apply_explicit_relationships(requirement: str, tables: list[TableSpec]) -> list[dict[str, str]]:
    """Apply `column references table(column)` phrases before inference."""
    by_name = {_clean_identifier(table.name): table for table in tables}
    by_singular = {_singular_name(table.name): table for table in tables}
    relationships: list[dict[str, str]] = []
    pattern = re.compile(
        rf"\b((?:{_IDENTIFIER_TOKEN}_id|{_IDENTIFIER_TOKEN}_key|id))\b\s+"
        rf"(?:\w+(?:\([^)]*\))?\s+)*references?\s+"
        rf"({_IDENTIFIER_TOKEN})(?:\s*\(\s*({_IDENTIFIER_TOKEN})\s*\))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(requirement):
        column_name = _clean_identifier(match.group(1))
        parent_name = _clean_identifier(match.group(2))
        parent = by_name.get(parent_name) or by_singular.get(_singular_name(parent_name))
        if parent is None:
            continue
        parent_column = _clean_identifier(match.group(3)) if match.group(3) else None
        if parent_column is None:
            primary = next((item for item in parent.columns if item.primary_key), None)
            parent_column = primary.name if primary else "id"
        for table in tables:
            child = next((item for item in table.columns if item.name == column_name), None)
            if child is None or table is parent:
                continue
            child.references = parent.name
            child.reference_column = parent_column
            _append_relationship(relationships, table, child)
    return relationships


def _link_inferred_relationships(tables: list[TableSpec], relationships: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
    """Link conventional `<parent>_id` columns when both tables are named."""
    by_singular = {_singular_name(table.name): table for table in tables}
    relationships = relationships or []
    for table in tables:
        for column in table.columns:
            if column.references or not column.name.endswith("_id"):
                continue
            parent_key = column.name[:-3]
            parent = by_singular.get(parent_key)
            if parent is None or parent is table:
                continue
            parent_pk = next((item for item in parent.columns if item.primary_key), None)
            if parent_pk is None:
                continue
            column.references = parent.name
            column.reference_column = parent_pk.name
            _append_relationship(relationships, table, column)
    return relationships


def _dependency_order(tables: list[TableSpec]) -> list[TableSpec]:
    """Place referenced tables before dependent tables when possible."""
    remaining = list(tables)
    ordered: list[TableSpec] = []
    while remaining:
        ready = next(
            (
                table for table in remaining
                if all(
                    column.references is None
                    or column.references in {item.name for item in ordered}
                    or column.references not in {item.name for item in remaining}
                    for column in table.columns
                )
            ),
            None,
        )
        if ready is None:
            # Preserve the request order for cycles or unresolved references.
            ordered.extend(remaining)
            break
        ordered.append(ready)
        remaining.remove(ready)
    return ordered


def compile_multi_table_request(requirement: str, dialect: str | None = None) -> CompilationResult | None:
    """Compile a small explicit group of conventionally related tables."""
    names = _multi_table_names(requirement)
    if not names:
        return None
    normalized = normalize_dialect(dialect)
    if normalized not in SUPPORTED_DIALECTS:
        return None
    tables = [TableSpec(name, _default_columns(name)) for name in names]
    relationships = _apply_explicit_relationships(requirement, tables)
    relationships = _link_inferred_relationships(tables, relationships)
    tables = _dependency_order(tables)
    if normalized == "dbml":
        output = "\n\n".join(_render_dbml(table) for table in tables)
    else:
        output = "\n\n".join(_render_sql(table, normalized) for table in tables)
    plan = {
        "operation": "create_tables",
        "object_type": "tables",
        "name": None,
        "tables": [_table_plan(table) for table in tables],
        "columns": [],
        "relationships": relationships,
    }
    return CompilationResult(
        output=output,
        dialect=normalized,
        table_names=[table.name for table in tables],
        confidence=0.91,
        operation="create_tables",
        plan=plan,
    )


def compile_ddl_request(
    requirement: str,
    dialect: str | None = None,
    task: str = "generate_ddl",
) -> CompilationResult | None:
    """Compile supported common DDL requests using the local typed planner."""
    if task == "explain_ddl":
        return None
    if task == "modify_ddl":
        return compile_index_request(requirement, dialect) or compile_alter_request(requirement, dialect)
    return (
        compile_schema_table_request(requirement, dialect)
        or compile_index_request(requirement, dialect)
        or compile_multi_table_request(requirement, dialect)
        or compile_table_request(requirement, dialect)
    )
