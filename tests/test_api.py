"""
API tests using FastAPI's TestClient. These assume a model has already been
trained and registered (run `python -m src.train` first — CI does this
before pytest, see .github/workflows/ci.yml).
"""
import pytest
from fastapi.testclient import TestClient

from src.registry import get_production_version

pytestmark = pytest.mark.skipif(
    get_production_version() is None,
    reason="No trained model in registry; run `python -m src.train` first.",
)

from api.main import app  # noqa: E402  (imported after skip check)


@pytest.fixture(scope="module")
def client():
    # Context manager form is required so FastAPI's lifespan (startup/shutdown)
    # events actually run and the model gets loaded before requests hit it.
    with TestClient(app) as c:
        yield c

GOOD_APPLICATION = {
    "income": 65000,
    "age": 32,
    "loan_amount": 25000,
    "employment_years": 6,
    "credit_score": 720,
    "existing_debt": 8000,
    "num_credit_lines": 5,
    "num_delinquencies_2yr": 0,
    "loan_term_months": 60,
    "interest_rate": 9.5,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "home_improvement",
    "employment_type": "salaried",
}


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_version"]


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    assert "roc_auc" in body["metrics"]


def test_predict_returns_valid_response(client):
    resp = client.post("/predict", json=GOOD_APPLICATION)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["prediction"] in {"low_risk", "medium_risk", "high_risk"}
    assert body["model_version"]


def test_predict_rejects_invalid_credit_score(client):
    bad = {**GOOD_APPLICATION, "credit_score": 9999}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD_APPLICATION.items() if k != "income"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_invalid_category(client):
    bad = {**GOOD_APPLICATION, "home_ownership": "SPACESHIP"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_high_risk_applicant_scores_higher_than_low_risk(client):
    low_risk = {**GOOD_APPLICATION, "credit_score": 800, "existing_debt": 1000,
                "num_delinquencies_2yr": 0, "loan_amount": 5000}
    high_risk = {**GOOD_APPLICATION, "credit_score": 500, "existing_debt": 40000,
                 "num_delinquencies_2yr": 4, "loan_amount": 45000,
                 "employment_type": "unemployed"}
    p_low = client.post("/predict", json=low_risk).json()["default_probability"]
    p_high = client.post("/predict", json=high_risk).json()["default_probability"]
    assert p_high > p_low


def test_batch_predict(client):
    resp = client.post("/predict/batch", json={"applications": [GOOD_APPLICATION, GOOD_APPLICATION]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2


def test_admin_reload(client):
    resp = client.post("/admin/reload")
    assert resp.status_code == 200
    assert resp.json()["status"] == "reloaded"
