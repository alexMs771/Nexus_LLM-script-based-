from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.learned_sql import clear_own_sql_model_cache


def test_learned_endpoint_does_not_fallback_to_hardcoded_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "own_model_dir", tmp_path / "missing-checkpoint")
    clear_own_sql_model_cache()
    client = TestClient(app)
    response = client.post(
        "/generate-sql",
        json={"question": "Create a customer table", "schema": "", "dialect": "postgresql"},
    )
    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"]
    clear_own_sql_model_cache()
