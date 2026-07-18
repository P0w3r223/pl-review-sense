"""Tests for the FastAPI service (model mocked — no trained artifact needed)."""

from fastapi.testclient import TestClient

from api.main import _state, app

client = TestClient(app)


class _FakeModel:
    def predict_proba(self, texts):
        return [[0.1, 0.2, 0.7] for _ in texts]  # -> positive


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_label_and_probabilities(monkeypatch):
    monkeypatch.setitem(_state, "model", _FakeModel())
    response = client.post("/predict", json={"text": "Świetny produkt, polecam!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["label_index"] == 2
    assert set(body["probabilities"]) == {"negative", "neutral", "positive"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_503_without_model(monkeypatch):
    monkeypatch.setitem(_state, "model", None)
    assert client.post("/predict", json={"text": "cokolwiek"}).status_code == 503


def test_predict_rejects_empty_text(monkeypatch):
    monkeypatch.setitem(_state, "model", _FakeModel())
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_requires_text_field():
    assert client.post("/predict", json={}).status_code == 422
