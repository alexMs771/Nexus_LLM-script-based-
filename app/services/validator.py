from __future__ import annotations

from dataclasses import dataclass, field
import re

import sqlglot
from sqlglot import exp


DIALECT_MAP = {"postgresql": "postgres", "postgres": "postgres", "sqlite": "sqlite", "mysql": "mysql"}
DESTRUCTIVE_KEYS = {"drop", "truncate", "delete", "update", "insert", "merge"}
SUPPORTED_TOP_LEVEL = {"create", "alter", "comment", "set", "use"}


@dataclass
class ValidationResult:
    valid_syntax: bool
    warnings: list[str] = field(default_factory=list)
    sql: str = ""
    explanation: str = ""
    dialect: str = "postgresql"
    statement_types: list[str] = field(default_factory=list)
    referenced_objects: list[str] = field(default_factory=list)
    destructive: bool = False
    semantic_valid: bool | None = None
    policy_allowed: bool = True

    @property
    def validation_status(self) -> str:
        return "valid" if self.valid_syntax and self.policy_allowed else "invalid"


def _dialect(dialect: str) -> str:
    try:
        return DIALECT_MAP[dialect.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dialect: {dialect}") from exc


def split_sql_and_explanation(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:sql)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE | re.DOTALL).strip()
    if not stripped:
        return "", ""
    markers = ("CREATE ", "ALTER ", "DROP ", "TRUNCATE ", "COMMENT ", "WITH ", "TABLE ")
    upper = stripped.upper()
    if upper.startswith(markers):
        return stripped, ""
    starts = [upper.find(marker) for marker in markers if upper.find(marker) >= 0]
    if not starts:
        return "", stripped
    index = min(starts)
    return stripped[index:].strip(), stripped[:index].strip()


def _object_name(statement: exp.Expression) -> str | None:
    for attr in ("this", "expression"):
        value = statement.args.get(attr)
        if isinstance(value, exp.Table):
            return value.sql() or value.name
        if isinstance(value, exp.Schema):
            table = value.this
            return table.sql() if isinstance(table, exp.Table) else None
        if isinstance(value, exp.TableAlias):
            return value.name
    table = statement.find(exp.Table)
    return table.sql() if table is not None else None


def _statement_type(statement: exp.Expression) -> str:
    if isinstance(statement, exp.Create):
        target = statement.args.get("kind") or "object"
        if isinstance(statement.this, exp.Schema):
            return "CREATE TABLE"
        return f"CREATE {str(target).upper()}" if target != "object" else "CREATE"
    return str(statement.key or statement.__class__.__name__).replace("_", " ").upper()


def validate_sql(
    text: str,
    dialect: str = "postgresql",
    *,
    allow_destructive: bool = False,
    schema_context: str = "",
) -> ValidationResult:
    sql, explanation = split_sql_and_explanation(text)
    warnings: list[str] = []
    normalized_dialect = dialect.lower()
    if normalized_dialect == "dbml":
        return validate_dbml(text)
    if normalized_dialect not in DIALECT_MAP:
        return ValidationResult(False, [f"Unsupported SQL dialect: {dialect}"], sql, explanation, normalized_dialect)
    if not sql:
        return ValidationResult(False, ["No SQL statement was found in the model output."], "", explanation, normalized_dialect)
    try:
        statements = [statement for statement in sqlglot.parse(sql, read=_dialect(normalized_dialect)) if statement is not None]
    except (sqlglot.errors.ParseError, ValueError) as exc:
        return ValidationResult(False, [f"SQL parser rejected the output: {exc}"], sql, explanation, normalized_dialect)
    if not statements:
        return ValidationResult(False, ["Parser returned no statements."], sql, explanation, normalized_dialect)

    types: list[str] = []
    objects: list[str] = []
    destructive = False
    for statement in statements:
        key = str(statement.key or "").lower()
        types.append(_statement_type(statement))
        name = _object_name(statement)
        if name and name not in objects:
            objects.append(name)
        if key in DESTRUCTIVE_KEYS or isinstance(statement, (exp.Drop, exp.TruncateTable, exp.Delete, exp.Update)):
            destructive = True
            if not allow_destructive:
                warnings.append("Destructive or restricted SQL detected; execution is disabled by policy.")
        if key not in SUPPORTED_TOP_LEVEL and key not in DESTRUCTIVE_KEYS:
            warnings.append(f"Statement type `{key or 'unknown'}` is outside the DDL service scope.")

    # SQLGlot's AST is a syntax check. The additional checks below are deliberately
    # conservative: warnings never turn a parseable statement into a false success.
    for statement in statements:
        if isinstance(statement, exp.Create) and isinstance(statement.this, exp.Schema):
            table = statement.this.this
            if isinstance(table, exp.Table):
                has_primary_key = any(
                    isinstance(node, (exp.PrimaryKey, exp.PrimaryKeyColumnConstraint))
                    for node in statement.walk()
                )
                has_columns = bool(statement.this.expressions)
                if has_columns and not has_primary_key:
                    warnings.append(f"Table `{table.sql()}` has no primary key; verify that this is intentional.")
        for reference in statement.find_all(exp.Reference):
            table = reference.this
            if isinstance(table, exp.Table) and table.name and schema_context:
                context_lower = schema_context.lower()
                if table.name.lower() not in context_lower and table.name.lower() not in {o.lower() for o in objects}:
                    warnings.append(f"Foreign-key target `{table.name}` is not present in the supplied schema context.")

    warnings.append("Parsing does not prove execution correctness; use a disposable database for execution tests.")
    policy_allowed = not destructive or allow_destructive
    return ValidationResult(
        True,
        warnings,
        sql,
        explanation,
        normalized_dialect,
        types,
        objects,
        destructive,
        True,
        policy_allowed,
    )


def validate_dbml(text: str) -> ValidationResult:
    """Validate the useful table subset of DBML without requiring a DBML parser."""
    cleaned = re.sub(r"^```(?:dbml)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    table_matches = list(re.finditer(r"\bTable\s+([A-Za-z_][\w]*)\s*\{(.*?)\}", cleaned, flags=re.IGNORECASE | re.DOTALL))
    if not table_matches:
        return ValidationResult(False, ["No valid DBML Table block was found."], cleaned, dialect="dbml")
    warnings: list[str] = ["DBML was structurally validated; use dbdiagram.io or a database-specific transpiler for execution."]
    objects: list[str] = []
    for match in table_matches:
        name = match.group(1)
        objects.append(name)
        body = match.group(2)
        column_lines = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("//")]
        if not column_lines:
            return ValidationResult(False, [f"DBML table `{name}` has no columns."], cleaned, dialect="dbml")
        for line in column_lines:
            if line.lower().startswith(("index", "note", "ref", "primary key")):
                continue
            if not re.match(r"^[A-Za-z_][\w]*\s+[A-Za-z][\w]*(?:\([^)]*\))?(?:\s*\[[^]]*\])?$", line):
                return ValidationResult(False, [f"Invalid DBML column definition in table `{name}`: `{line}`"], cleaned, dialect="dbml")
    return ValidationResult(
        True,
        warnings,
        cleaned,
        "",
        "dbml",
        ["TABLE"] * len(table_matches),
        objects,
        False,
        True,
        True,
    )


def convert_sql(sql: str, source_dialect: str, target_dialect: str) -> str:
    if source_dialect.lower() == "dbml" or target_dialect.lower() == "dbml":
        raise ValueError("DBML conversion is not available through SQLGlot; generate DBML directly from a requirement.")
    source = _dialect(source_dialect)
    target = _dialect(target_dialect)
    statements = sqlglot.transpile(sql, read=source, write=target, pretty=True)
    return "\n".join(statement.rstrip(";") + ";" for statement in statements if statement.strip())
