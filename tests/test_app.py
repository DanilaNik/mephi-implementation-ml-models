from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.api import FEATURE_COLS, app, assign_ab_bucket, registry


SAMPLE_PAYLOAD = {
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 24,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 50000,
    "BILL_AMT2": 50000,
    "BILL_AMT3": 50000,
    "BILL_AMT4": 50000,
    "BILL_AMT5": 50000,
    "BILL_AMT6": 50000,
    "PAY_AMT1": 0,
    "PAY_AMT2": 0,
    "PAY_AMT3": 0,
    "PAY_AMT4": 0,
    "PAY_AMT5": 0,
    "PAY_AMT6": 0,
}


@pytest.fixture(scope="module")
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_root_lists_endpoints(client):
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["service"] == "credit-default-svc"
    assert "/predict" in body["endpoints"]


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["status"] == "ok"
    assert "v1" in body["loaded_models"]


def test_models_endpoint(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    versions = resp.get_json()["available_versions"]
    assert {"v1", "v2"}.issubset(set(versions))


def test_feature_list_matches_payload():
    assert set(FEATURE_COLS) == set(SAMPLE_PAYLOAD.keys())


def test_predict_default_version(client):
    resp = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["served_by"] == "v1"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["default_predicted"] in (0, 1)


def test_predict_explicit_v2(client):
    payload = dict(SAMPLE_PAYLOAD, model_version="v2")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    assert resp.get_json()["served_by"] == "v2"


def test_predict_unknown_version_returns_400(client):
    payload = dict(SAMPLE_PAYLOAD, model_version="v99")
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "unknown_model_version"


def test_predict_missing_features_returns_400(client):
    resp = client.post("/predict", json={"LIMIT_BAL": 10000})
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "missing_features"
    assert len(body["missing"]) > 0


def test_predict_non_json_returns_400(client):
    resp = client.post("/predict", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_predict_ab_requires_user_id(client):
    resp = client.post("/predict/ab", json=SAMPLE_PAYLOAD)
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "user_id_required_for_ab"


def test_predict_ab_uses_user_id(client):
    payload = dict(SAMPLE_PAYLOAD, user_id="client-42")
    resp = client.post("/predict/ab", json=payload)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["served_by"] in {"v1", "v2"}
    assert body["ab_group"] in {"control", "treatment"}


def test_ab_bucket_is_stable():
    assert assign_ab_bucket("user-1") == assign_ab_bucket("user-1")
    assert assign_ab_bucket("user-2") == assign_ab_bucket("user-2")


def test_ab_bucket_distribution_is_balanced():
    counts = {"v1": 0, "v2": 0}
    for i in range(2000):
        counts[assign_ab_bucket(f"client-{i}")] += 1
    # сплит 50/50
    assert abs(counts["v1"] - counts["v2"]) < 200


def test_predict_get_method_not_allowed(client):
    resp = client.get("/predict")
    assert resp.status_code == 405


def test_registry_has_expected_models():
    assert "v1" in registry.versions
    assert "v2" in registry.versions


def test_request_id_header_propagates(client):
    resp = client.post(
        "/predict",
        json=SAMPLE_PAYLOAD,
        headers={"X-Request-Id": "rid-test-001"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-Id") == "rid-test-001"
