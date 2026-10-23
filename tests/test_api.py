from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint_reports_not_loaded_without_artifacts():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model"] == "not_loaded"
    assert payload["loaded_models"] == []


def test_advanced_generate_returns_503_when_scratch_model_is_missing():
    response = client.post(
        "/generate",
        json={
            "requirement": "Alter the students table and add a graduation_year column.",
            "dialect": "postgresql",
            "task": "generate_ddl",
            "model": "scratch",
        },
    )
    assert response.status_code == 503
    assert "not loaded" in response.json()["detail"].lower() or "Train it" in response.json()["detail"]


def test_generate_rejects_short_requirement():
    response = client.post(
        "/generate",
        json={"requirement": "hi", "model": "scratch"},
    )
    assert response.status_code == 422
