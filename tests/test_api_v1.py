from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


client = TestClient(app)


def test_v1_scope_rejection_contains_no_sql():
    response = client.post("/api/v1/ddl/generate", json={"requirement": "What is Python?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "OUT_OF_SCOPE"
    assert payload["generated_ddl"] is None


def test_v1_validation_reports_statement_type():
    response = client.post(
        "/api/v1/ddl/validate",
        json={"dialect": "postgresql", "sql": "CREATE TABLE customers (id BIGINT PRIMARY KEY);"},
    )
    assert response.status_code == 200
    assert response.json()["statement_types"] == ["CREATE TABLE"]


def test_v1_generate_returns_the_requested_table_name_without_a_checkpoint(monkeypatch):
    # The common table compiler is local and deterministic; it must not require
    # an optional scratch checkpoint merely to answer this request.
    monkeypatch.setattr("app.services.request_engine.select_model", lambda _: object())

    response = client.post(
        "/api/v1/ddl/generate",
        json={"requirement": "create table employee", "model": "scratch"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["validation_status"] == "valid"
    assert 'CREATE TABLE "employee"' in payload["generated_ddl"]
    assert 'CREATE TABLE "create"' not in payload["generated_ddl"]
    assert payload["metadata"]["generation_method"] == "local_schema_compiler"
    assert payload["result"]["operation"] == "create_table"
    assert payload["result"]["object_type"] == "table"
    assert payload["result"]["name"] == "employee"
    assert payload["result"]["ddl"] == payload["generated_ddl"]
    assert payload["result"]["validation"]["status"] == "valid"


def test_v1_generate_returns_compact_multi_table_result():
    response = client.post(
        "/api/v1/ddl/generate",
        json={
            "requirement": "Create customers and orders tables with customer_id references customers(id)",
            "model": "scratch",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted"
    assert payload["result"]["operation"] == "create_tables"
    assert payload["result"]["object_type"] == "tables"
    assert len(payload["result"]["details"]["tables"]) == 2
    assert payload["result"]["relationships"] == [
        {"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "id"}
    ]


def test_v1_modify_request_uses_local_planner_for_index_and_alter():
    index_response = client.post(
        "/api/v1/ddl/generate",
        json={
            "task": "modify_ddl",
            "requirement": "Add an index on orders.customer_id",
            "model": "scratch",
        },
    )
    alter_response = client.post(
        "/api/v1/ddl/generate",
        json={
            "task": "modify_ddl",
            "requirement": "Add a hire_date column to employees that defaults to current date",
            "model": "scratch",
        },
    )

    assert index_response.status_code == 200
    assert alter_response.status_code == 200
    assert index_response.json()["result"]["operation"] == "create_index"
    assert "CREATE INDEX" in index_response.json()["generated_ddl"]
    assert alter_response.json()["result"]["operation"] == "alter_add_column"
    assert "DEFAULT CURRENT_DATE" in alter_response.json()["generated_ddl"]


def test_v1_session_remembers_previous_table_for_follow_up(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "session_db_path", tmp_path / "sessions.sqlite3")
    session_id = "customer-follow-up-001"

    first = client.post(
        "/api/v1/ddl/generate",
        json={
            "session_id": session_id,
            "model": "scratch",
            "dialect": "mysql",
            "requirement": "Create a customer table with name and email",
        },
    )
    follow_up = client.post(
        "/api/v1/ddl/generate",
        json={
            "session_id": session_id,
            "model": "scratch",
            "requirement": "Now add phone and date of birth columns to it",
        },
    )

    assert first.status_code == 200
    assert follow_up.status_code == 200
    payload = follow_up.json()
    assert payload["conversation_turn"] == 2
    assert payload["metadata"]["memory_used"] is True
    assert payload["metadata"]["resolved_requirement"].endswith("to customer")
    assert payload["metadata"]["effective_task"] == "modify_ddl"
    assert payload["dialect"] == "mysql"
    assert payload["model_id"] == "scratch"
    assert "ALTER TABLE `customer`" in payload["generated_ddl"]
    assert "ADD COLUMN `phone` TEXT" in payload["generated_ddl"]
    assert "ADD COLUMN `date_of_birth` DATE" in payload["generated_ddl"]

    state = client.get(f"/api/v1/sessions/{session_id}")
    assert state.status_code == 200
    assert state.json()["turn_count"] == 2
    assert state.json()["active_table"] == "customer"

    deleted = client.delete(f"/api/v1/sessions/{session_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/v1/sessions/{session_id}").status_code == 404


def test_v1_unsupported_sql_has_distinct_code_and_no_sql():
    response = client.post("/api/v1/ddl/generate", json={"requirement": "Select employees with salary greater than 1000."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "UNSUPPORTED_SQL_OPERATION"
    assert payload["code"] == "UNSUPPORTED_OPERATION"
    assert payload["generated_ddl"] is None
