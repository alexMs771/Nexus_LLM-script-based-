SYSTEM_PROMPT = (
    "You are a database-structure assistant. Handle only supported DDL tasks. "
    "Return PostgreSQL, SQLite, MySQL, or DBML structure when requested. Do not execute SQL. "
    "If required details are missing, ask one focused clarification question. "
    "Never answer unrelated questions or produce destructive SQL by default."
)


def build_prompt(
    requirement: str,
    dialect: str,
    task: str = "generate_ddl",
    schema_context: str = "",
    policy_variant: str = "common",
    intent: str = "IN_SCOPE",
) -> str:
    """Stable serialization shared by training, evaluation, and serving."""
    context = schema_context.strip() or "(none supplied)"
    return (
        "<|system|>\n"
        f"{SYSTEM_PROMPT}\n"
        f"Policy: {policy_variant}\n"
        "<|request|>\n"
        f"{requirement.strip()}\n"
        "<|dialect|>\n"
        f"{dialect}\n"
        "<|task|>\n"
        f"{task}\n"
        "<|intent|>\n"
        f"{intent}\n"
        "<|schema_context|>\n"
        f"{context}\n"
        "<|response|>\n"
        "### Response\n"
    )
