from fastapi.testclient import TestClient
from service.main import app

client = TestClient(app)

def test_chat_endpoint_requires_auth():
    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 401

def test_chat_endpoint_with_valid_key():
    response = client.post(
        "/api/chat",
        json={"message": "how did CVX:DVN perform?"},
        headers={"X-API-Key": "kairos-secret"},
    )
    assert response.status_code == 200
    assert "CVX:DVN" in response.json()["answer"]