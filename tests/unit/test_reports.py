"""Unit tests for monitoring HTML, JSON, alert, and metric output."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from mlops_monitoring.data.validation import data_quality_summary
from mlops_monitoring.monitoring.drift import detect_data_drift, detect_prediction_drift
from mlops_monitoring.monitoring.reports import (
    build_monitoring_report,
    evaluate_alerts,
    write_html_report,
    write_json_report,
    write_prometheus_metrics,
)


def _report():
    rng = np.random.default_rng(11)
    reference = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    current = reference.copy()
    current["x"] += 4.0

    drift = detect_data_drift(reference, current)
    prediction_drift = detect_prediction_drift([1] * 100 + [0] * 50, [0] * 100 + [1] * 50)
    performance = {
        "status": "ok",
        "matched_rows": 150,
        "threshold": 0.1,
        "metrics": {"accuracy": 0.7, "precision": 0.7, "recall": 0.6, "f1": 0.65, "roc_auc": 0.7},
        "baseline": {"accuracy": 0.9, "precision": 0.9, "recall": 0.9, "f1": 0.9, "roc_auc": 0.9},
        "relative_drops": {"f1": 0.2778, "recall": 0.3333},
        "max_relative_drop": 0.3333,
        "degraded": True,
        "degraded_metrics": ["f1", "recall"],
    }
    current_quality = data_quality_summary(current)
    alerts = evaluate_alerts(
        drift=drift,
        prediction_drift=prediction_drift,
        performance=performance,
        current_quality=current_quality,
    )
    report = build_monitoring_report(
        reference=reference,
        current=current,
        drift=drift,
        prediction_drift=prediction_drift,
        performance=performance,
        alerts=alerts,
        model_name="test-model",
        model_version="7",
        window={"last_n": 150},
    )
    return report


def test_alerts_include_drift_prediction_and_performance():
    report = _report()
    codes = {alert["code"] for alert in report["alerts"]}

    assert report["summary"]["status"] == "alert"
    assert "data_drift" in codes
    assert "prediction_drift" in codes
    assert "performance_degradation" in codes
    critical = [alert for alert in report["alerts"] if alert["severity"] == "critical"]
    assert len(critical) >= 1


def test_html_report_contains_monitoring_sections(tmp_path):
    report = _report()
    path = write_html_report(report, tmp_path / "drift.html")

    content = path.read_text(encoding="utf-8")
    assert "Model Monitoring Report" in content
    assert "Data drift by feature" in content
    assert "Production performance" in content
    assert "test-model" in content
    assert "x" in content


def test_json_report_roundtrip(tmp_path):
    report = _report()
    path = write_json_report(report, tmp_path / "monitoring.json")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["summary"]["dataset_drifted"]
    assert loaded["model"]["version"] == "7"
    assert loaded["current_quality"]["rows"] == 200


def test_prometheus_textfile_contains_gauges(tmp_path):
    report = _report()
    path = write_prometheus_metrics(report, tmp_path / "monitoring.prom")

    content = path.read_text(encoding="utf-8")
    assert "mlops_drift_share{" in content
    assert "mlops_feature_drift{" in content
    assert 'feature="x"' in content
    assert "mlops_model_f1{" in content
    assert "mlops_monitoring_alerts{" in content
    assert content.count("mlops_drift_share") >= 2
