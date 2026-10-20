"""Prometheus metrics for the inference service."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUEST_COUNT = Counter(
    "mlops_api_requests_total",
    "HTTP requests handled by the inference API.",
    ["method", "path", "status"],
)

REQUEST_LATENCY = Histogram(
    "mlops_api_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_COUNT = Counter(
    "mlops_predictions_total",
    "Predictions returned by the model.",
    ["model_name", "model_version", "prediction"],
)

PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Model inference latency in seconds.",
    ["model_name", "model_version"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

PREDICTION_CONFIDENCE = Histogram(
    "mlops_prediction_confidence",
    "Confidence (maximum predicted probability) of returned predictions.",
    ["model_name", "model_version"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)

PREDICTION_ERRORS = Counter(
    "mlops_prediction_errors_total",
    "Failed prediction requests by reason.",
    ["reason"],
)

MODEL_LOADED = Gauge(
    "mlops_model_loaded",
    "1 when a model artifact is loaded, otherwise 0.",
)

MODEL_INFO = Gauge(
    "mlops_model_info",
    "Static information about the loaded model.",
    ["model_name", "model_version", "target_names"],
)

# Drift and performance gauges are produced by the monitoring job and exposed
# through Prometheus' node-exporter textfile collector, not by the API process.
# Keeping them out of this registry prevents duplicate series with conflicting
# values for the same metric name.


def set_model_info(model_name: str, model_version: str, target_names: list[str]) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(
        model_name=model_name,
        model_version=model_version,
        target_names="|".join(target_names),
    ).set(1)


def record_prediction_error(reason: str) -> None:
    PREDICTION_ERRORS.labels(reason=reason).inc()


def render_metrics() -> tuple[bytes, str]:
    """Return the Prometheus exposition payload and content type."""
    return generate_latest(), CONTENT_TYPE_LATEST
