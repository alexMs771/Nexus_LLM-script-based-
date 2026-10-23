from app.prompts import build_prompt
from app.services.validator import split_sql_and_explanation


def test_prompt_includes_requirement_dialect_and_task():
    prompt = build_prompt("Create a table for students with an ID and name.", "postgresql", "generate_ddl")
    assert "Create a table for students" in prompt
    assert "postgresql" in prompt
    assert "generate_ddl" in prompt
    assert prompt.endswith("### Response\n")


def test_split_keeps_explanation_separate_from_sql():
    text = "Each order belongs to one customer.\nCREATE TABLE orders (order_id BIGINT PRIMARY KEY);"
    sql, explanation = split_sql_and_explanation(text)
    assert sql.startswith("CREATE TABLE")
    assert "belongs" in explanation
