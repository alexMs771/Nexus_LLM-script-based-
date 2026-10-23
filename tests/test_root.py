from fastapi.testclient import TestClient

from app.main import app


def test_root_lists_api_links():
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"
