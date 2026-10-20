"""Tests for the inference service Prometheus registry."""

from __future__ import annotations

from mlops_monitoring.observability import metrics

MONITORING_METRICS = (
    "mlops_drift_share",
    "mlops_dataset_drifted",
    "mlops_feature_drift",
    "mlops_model_f1",
    "mlops_baseline_f1",
    "mlops_performance_drop",
    "mlops_monitoring_last_run_timestamp_seconds",
    "mlops_monitoring_rows",
)


def test_api_metrics_exclude_monitoring_textfile_gauges() -> None:
    payload, content_type = metrics.render_metrics()
    text = payload.decode("utf-8")

    assert content_type.startswith("text/plain")
    assert "mlops_api_requests_total" in text
    assert "mlops_model_loaded" in text

    # Monitoring gauges are published through the node-exporter textfile
    # collector. Exposing them here as well would create duplicate series with
    # conflicting values under the same metric name.
    for name in MONITORING_METRICS:
        assert name not in text
