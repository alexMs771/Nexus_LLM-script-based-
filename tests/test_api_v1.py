from fastapi.testclient import TestClient

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


def test_v1_unsupported_sql_has_distinct_code_and_no_sql():
    response = client.post("/api/v1/ddl/generate", json={"requirement": "Select employees with salary greater than 1000."})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "UNSUPPORTED_SQL_OPERATION"
    assert payload["code"] == "UNSUPPORTED_OPERATION"
    assert payload["generated_ddl"] is None
