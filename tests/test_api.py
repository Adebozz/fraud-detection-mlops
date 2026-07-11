import pytest
from fastapi.testclient import TestClient

from fraud_detection.data import FEATURE_COLUMNS


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from fraud_detection.api.main import app

    with TestClient(app) as c:
        yield c


def _payload(value: float = 0.0) -> dict:
    return {name: value for name in FEATURE_COLUMNS}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["features"] == FEATURE_COLUMNS


def test_predict_returns_valid_response(client):
    r = client.post("/predict", json=_payload())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)
    assert body["model_run_id"] == "test-run"


def test_predict_rejects_missing_feature(client):
    payload = _payload()
    payload.pop("V1")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_fraud_like_input_scores_higher_than_legit(client):
    # Synthetic fraud rows have shifted feature means (~1.5); legit ~0.
    legit = client.post("/predict", json=_payload(0.0)).json()["fraud_probability"]
    fraud = client.post("/predict", json=_payload(1.5)).json()["fraud_probability"]
    assert fraud > legit
