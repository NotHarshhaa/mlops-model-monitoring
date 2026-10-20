"""Integration tests for the FastAPI inference service."""

from __future__ import annotations

import math

from fastapi.testclient import TestClient

from mlops_monitoring.api.main import create_app
from mlops_monitoring.observability.prediction_log import read_predictions


def test_health_and_readiness(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["model_loaded"] is True

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["ready"] is True


def test_model_info_exposes_schema(client, bundle):
    response = client.get("/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_name"] == bundle.model_name
    assert len(payload["feature_specs"]) == len(bundle.feature_specs)
    assert payload["metrics"]["f1"] > 0.7


def test_prediction_is_logged_with_request_id(client, settings, sample_features):
    response = client.post(
        "/predict",
        json={"features": sample_features},
        headers={"X-Request-ID": "test-request-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"
    payload = response.json()
    assert payload["prediction"] in {0, 1}
    assert payload["model_name"]
    assert payload["model_version"]
    assert payload["latency_ms"] >= 0
    assert payload["prediction_id"]
    assert math.isclose(sum(payload["probabilities"].values()), 1.0, rel_tol=1e-5)

    records = read_predictions(settings.prediction_log_path)
    assert len(records) == 1
    assert records[0]["request_id"] == "test-request-id"
    assert records[0]["prediction_id"] == payload["prediction_id"]
    assert records[0]["features"] == sample_features
    assert "probabilities" in records[0]


def test_prediction_rejects_missing_feature(client, sample_features):
    features = dict(sample_features)
    features.pop(next(iter(features)))

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "Missing required feature" in " ".join(detail["errors"])


def test_prediction_rejects_unknown_feature(client, sample_features):
    features = {**sample_features, "unexpected_feature": 1.0}

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert "Unexpected feature" in " ".join(response.json()["detail"]["errors"])


def test_prediction_rejects_non_numeric_value(client, sample_features, bundle):
    numeric_name = next(spec.name for spec in bundle.feature_specs if spec.kind == "numeric")
    features = {**sample_features, numeric_name: "not-a-number"}

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert "must be a number" in " ".join(response.json()["detail"]["errors"])


def test_prediction_rejects_empty_payload(client):
    response = client.post("/predict", json={"features": {}})

    assert response.status_code == 422


def test_batch_prediction(client, sample_features):
    response = client.post(
        "/predict/batch",
        json={"items": [{"features": sample_features}, {"features": sample_features}]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    assert payload["total_latency_ms"] >= 0


def test_batch_limit_is_enforced(trained_run, tmp_path, sample_features):
    settings = trained_run["settings"].model_copy(
        update={
            "model_path": trained_run["settings"].model_path,
            "prediction_log_path": tmp_path / "predictions.jsonl",
            "max_batch_size": 1,
        }
    )
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"items": [{"features": sample_features}, {"features": sample_features}]},
        )

    assert response.status_code == 422
    assert "maximum is 1" in response.json()["detail"]


def test_metrics_endpoint_exposes_service_metrics(client, sample_features):
    client.post("/predict", json={"features": sample_features})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "mlops_predictions_total" in response.text
    assert "mlops_api_requests_total" in response.text
    assert "mlops_model_loaded" in response.text


def test_service_without_model_is_not_ready(settings_factory, tmp_path):
    settings = settings_factory(model_path=tmp_path / "missing-model", model_name="missing-model")
    app = create_app(settings)
    with TestClient(app) as client:
        health = client.get("/health")
        ready = client.get("/ready")
        prediction = client.post(
            "/predict",
            json={"features": {"mean_radius": 12.0}},
        )
        info = client.get("/model/info")

    assert health.status_code == 200
    assert health.json()["model_loaded"] is False
    assert ready.status_code == 503
    assert prediction.status_code == 503
    assert info.status_code == 503
