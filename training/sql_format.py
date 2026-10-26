"""The single prompt contract shared by training, evaluation, and serving."""

from __future__ import annotations


SYSTEM_TEXT = (
    "Translate the user's database question into one executable SQL statement. "
    "Use only tables and columns present in the supplied schema. "
    "Return SQL only; do not explain it or wrap it in Markdown."
)


def build_sql_prompt(question: str, schema: str = "", dialect: str = "postgresql", evidence: str = "") -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM_TEXT}\n"
        "<|dialect|>\n"
        f"{dialect.strip() or 'postgresql'}\n"
        "<|schema|>\n"
        f"{schema.strip() or '(schema not supplied)'}\n"
        "<|evidence|>\n"
        f"{evidence.strip() or '(none)'}\n"
        "<|question|>\n"
        f"{question.strip()}\n"
        "<|sql|>\n"
    )


def build_ddl_prompt(
    requirement: str,
    dialect: str = "postgresql",
    task: str = "generate_ddl",
    schema: str = "",
    memory_summary: str = "",
) -> str:
    """Serialize the DDL task without embedding a table/column rule engine."""

    return (
        "<|system|>\n"
        "Output only valid database DDL for the requested dialect. No prose, Markdown, SELECT, INSERT, UPDATE, or DELETE.\n"
        "<|dialect|>\n"
        f"{dialect.strip() or 'postgresql'}\n"
        "<|task|>\n"
        f"{task.strip() or 'generate_ddl'}\n"
        "<|schema|>\n"
        f"{schema.strip() or '(none supplied)'}\n"
        "<|memory|>\n"
        f"{memory_summary.strip() or '(none)'}\n"
        "<|request|>\n"
        f"{requirement.strip()}\n"
        "<|ddl|>\n"
    )
