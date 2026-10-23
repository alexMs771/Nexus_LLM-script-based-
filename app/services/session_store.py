"""Persistent, lightweight conversation memory for DDL follow-up requests.

The store keeps request history and the latest database objects for an explicit
session id.  It does not infer memory from a user's identity or cookies: the
client must send the same ``session_id`` on each follow-up request.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from app.config import settings


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,99}$")
_REFERENCE_RE = re.compile(
    r"\b(?:it|that\s+table|this\s+table|the\s+table|the\s+existing\s+table|the\s+previous\s+table|the\s+same\s+table)\b",
    re.IGNORECASE,
)
_TABLE_TARGET_RE = re.compile(
    r"\b(?:alter\s+table|to|on|in)\s+(?:the\s+)?(?:table\s+)?[A-Za-z_][A-Za-z0-9_-]*\b",
    re.IGNORECASE,
)
_FOLLOWUP_MODIFY_RE = re.compile(r"\b(?:add|alter|modify|change|rename)\b", re.IGNORECASE)


@dataclass
class SessionContext:
    session_id: str | None = None
    turn_count: int = 0
    memory_used: bool = False
    previous_requirement: str = ""
    resolved_requirement: str = ""
    active_table: str | None = None
    table_names: list[str] = field(default_factory=list)
    schema_context: str = ""
    previous_dialect: str | None = None
    previous_model: str | None = None


def _db_path() -> Path:
    path = Path(settings.session_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(_db_path()), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection():
    """Commit/rollback and always close a SQLite connection on Windows too."""
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _initialize() -> None:
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS ddl_sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                turn_count INTEGER NOT NULL DEFAULT 0,
                active_table TEXT,
                table_names_json TEXT NOT NULL DEFAULT '[]',
                plan_json TEXT NOT NULL DEFAULT '{}',
                last_requirement TEXT NOT NULL DEFAULT '',
                last_resolved_requirement TEXT NOT NULL DEFAULT '',
                last_response_json TEXT NOT NULL DEFAULT '{}',
                last_dialect TEXT,
                last_model TEXT
            );

            CREATE TABLE IF NOT EXISTS ddl_session_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES ddl_sessions(session_id) ON DELETE CASCADE,
                request_id TEXT NOT NULL,
                requirement TEXT NOT NULL,
                resolved_requirement TEXT NOT NULL,
                response_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_ddl_session_turns_session
                ON ddl_session_turns(session_id, id);
            """
        )
        existing_columns = {row["name"] for row in connection.execute("PRAGMA table_info(ddl_sessions)")}
        for name, definition in (("last_dialect", "TEXT"), ("last_model", "TEXT")):
            if name not in existing_columns:
                connection.execute(f"ALTER TABLE ddl_sessions ADD COLUMN {name} {definition}")


def _validate_session_id(session_id: str) -> str:
    value = session_id.strip()
    if not _SESSION_ID_RE.fullmatch(value):
        raise ValueError("session_id must contain 1-100 letters, numbers, '.', '_' , ':' or '-'.")
    return value


def _json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _schema_summary(table_names: list[str], plan: dict[str, Any]) -> str:
    if not table_names:
        return ""
    summary = "Existing database objects: " + ", ".join(table_names) + "."
    tables = plan.get("tables") if isinstance(plan, dict) else None
    if isinstance(tables, list):
        details: list[str] = []
        for table in tables:
            if not isinstance(table, dict):
                continue
            name = table.get("name")
            columns = table.get("columns")
            if name and isinstance(columns, list):
                column_names = [str(item.get("name")) for item in columns if isinstance(item, dict) and item.get("name")]
                details.append(f"{name}({', '.join(column_names)})")
        if details:
            summary += " Schema details: " + "; ".join(details) + "."
    elif isinstance(plan, dict) and plan.get("name") and isinstance(plan.get("columns"), list):
        column_names = [str(item.get("name")) for item in plan["columns"] if isinstance(item, dict) and item.get("name")]
        if column_names:
            summary += f" Schema details: {plan['name']}({', '.join(column_names)})."
    return summary


def _load_row(session_id: str) -> sqlite3.Row | None:
    _initialize()
    with _connection() as connection:
        return connection.execute(
            "SELECT * FROM ddl_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()


def resolve_followup(session_id: str | None, requirement: str, task: str = "generate_ddl") -> SessionContext:
    """Resolve references such as ``it`` using the latest session object."""
    if not session_id:
        return SessionContext(resolved_requirement=requirement)
    session_id = _validate_session_id(session_id)
    row = _load_row(session_id)
    if row is None:
        return SessionContext(session_id=session_id, resolved_requirement=requirement)

    table_names = _json_loads(row["table_names_json"], [])
    if not isinstance(table_names, list):
        table_names = []
    active_table = row["active_table"] or (table_names[-1] if table_names else None)
    resolved = requirement.strip()
    memory_used = False
    if active_table:
        replaced = _REFERENCE_RE.sub(active_table, resolved)
        memory_used = replaced != resolved
        resolved = replaced
        # A follow-up DDL request commonly omits the target after the first turn:
        # "now add phone" -> "now add phone to customers".
        if _FOLLOWUP_MODIFY_RE.search(resolved) and not _TABLE_TARGET_RE.search(resolved):
            resolved = f"{resolved.rstrip(' .;')} to {active_table}"
            memory_used = True

    schema_context = _schema_summary(table_names, _json_loads(row["plan_json"], {}))
    return SessionContext(
        session_id=session_id,
        turn_count=int(row["turn_count"]),
        memory_used=memory_used,
        previous_requirement=row["last_requirement"],
        resolved_requirement=resolved,
        active_table=active_table,
        table_names=[str(value) for value in table_names],
        schema_context=schema_context,
        previous_dialect=row["last_dialect"],
        previous_model=row["last_model"],
    )


def record_turn(
    *,
    session_id: str | None,
    request_id: str,
    requirement: str,
    resolved_requirement: str,
    response: dict[str, Any],
    table_names: list[str] | None = None,
    plan: dict[str, Any] | None = None,
    dialect: str | None = None,
    model: str | None = None,
) -> int:
    """Persist a turn and return its 1-based turn number."""
    if not session_id:
        return 0
    session_id = _validate_session_id(session_id)
    _initialize()
    table_names = table_names or []
    plan = plan or {}
    response_json = json.dumps(response, ensure_ascii=False, default=str)
    plan_json = json.dumps(plan, ensure_ascii=False, default=str)
    with _connection() as connection:
        row = connection.execute(
            "SELECT turn_count, table_names_json FROM ddl_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        previous_names = _json_loads(row["table_names_json"], []) if row else []
        if not isinstance(previous_names, list):
            previous_names = []
        merged_names = list(dict.fromkeys([str(value) for value in previous_names] + [str(value) for value in table_names]))
        turn_count = int(row["turn_count"]) + 1 if row else 1
        active_table = table_names[-1] if table_names else (merged_names[-1] if merged_names else None)
        connection.execute(
            """
            INSERT INTO ddl_sessions (
                session_id, turn_count, active_table, table_names_json, plan_json,
                last_requirement, last_resolved_requirement, last_response_json,
                last_dialect, last_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP,
                turn_count = excluded.turn_count,
                active_table = excluded.active_table,
                table_names_json = excluded.table_names_json,
                plan_json = excluded.plan_json,
                last_requirement = excluded.last_requirement,
                last_resolved_requirement = excluded.last_resolved_requirement,
                last_response_json = excluded.last_response_json,
                last_dialect = excluded.last_dialect,
                last_model = excluded.last_model
            """,
            (
                session_id,
                turn_count,
                active_table,
                json.dumps(merged_names),
                plan_json,
                requirement,
                resolved_requirement,
                response_json,
                dialect,
                model,
            ),
        )
        connection.execute(
            """
            INSERT INTO ddl_session_turns
                (session_id, request_id, requirement, resolved_requirement, response_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, request_id, requirement, resolved_requirement, response_json),
        )
    return turn_count


def get_session(session_id: str) -> dict[str, Any] | None:
    session_id = _validate_session_id(session_id)
    row = _load_row(session_id)
    if row is None:
        return None
    return {
        "session_id": session_id,
        "turn_count": int(row["turn_count"]),
        "active_table": row["active_table"],
        "table_names": _json_loads(row["table_names_json"], []),
        "last_requirement": row["last_requirement"],
        "last_resolved_requirement": row["last_resolved_requirement"],
        "last_dialect": row["last_dialect"],
        "last_model": row["last_model"],
        "updated_at": row["updated_at"],
    }


def delete_session(session_id: str) -> bool:
    session_id = _validate_session_id(session_id)
    _initialize()
    with _connection() as connection:
        cursor = connection.execute("DELETE FROM ddl_sessions WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0
