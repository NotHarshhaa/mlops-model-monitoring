"""End-to-end monitoring tests using logged predictions and ground truth."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from mlops_monitoring.data.validation import data_quality_summary
from mlops_monitoring.inference import PredictionService
from mlops_monitoring.models.loading import ModelLoader
from mlops_monitoring.monitoring.drift import detect_data_drift
from mlops_monitoring.monitoring.performance import evaluate_production_performance, load_labels
from mlops_monitoring.monitoring.reports import (
    build_monitoring_report,
    evaluate_alerts,
    write_html_report,
    write_json_report,
    write_prometheus_metrics,
)
from mlops_monitoring.observability.prediction_log import read_predictions


@pytest.mark.monitoring
def test_monitoring_pipeline_with_simulated_drift(
    trained_run,
    demo_frame,
    settings,
    features_for_row,
    tmp_path,
):
    bundle = trained_run["bundle"]
    loader = ModelLoader(trained_run["settings"].model_path)
    loader.load()
    service = PredictionService(loader, settings)

    test_frame = demo_frame.sample(n=150, random_state=21).reset_index(drop=True)
    numeric_features = bundle.numeric_features
    reference_std = {
        name: float(pd.to_numeric(demo_frame[name], errors="coerce").std(ddof=0))
        for name in numeric_features
    }

    labels_path = tmp_path / "labels.jsonl"
    for _, row in test_frame.iterrows():
        features = features_for_row(row)
        for name in numeric_features:
            features[name] = float(features[name]) + 2.0 * reference_std[name]
        result = service.predict(features)
        with labels_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "prediction_id": result.prediction_id,
                        "label": int(row["target"]),
                    }
                )
                + "\n"
            )

    predictions = read_predictions(settings.prediction_log_path)
    assert len(predictions) == 150

    reference = pd.read_csv(trained_run["settings"].reference_data_path)
    reference_features = reference[bundle.feature_names]
    current_features = pd.DataFrame(
        [record["features"] for record in predictions], columns=bundle.feature_names
    )

    drift = detect_data_drift(
        reference_features,
        current_features,
        feature_specs=bundle.feature_specs,
        psi_threshold=0.2,
        share_threshold=0.3,
    )
    assert drift["drift_share"] > 0.5

    labels = load_labels(labels_path)
    performance = evaluate_production_performance(
        predictions,
        labels,
        baseline_metrics=bundle.metrics,
        drop_threshold=0.1,
    )
    assert performance["status"] == "ok"
    assert performance["matched_rows"] == 150

    alerts = evaluate_alerts(
        drift=drift,
        prediction_drift=None,
        performance=performance,
        current_quality=data_quality_summary(current_features),
    )
    report = build_monitoring_report(
        reference=reference_features,
        current=current_features,
        drift=drift,
        prediction_drift=None,
        performance=performance,
        alerts=alerts,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
        window={"prediction_rows": 150, "labelled_rows": 150},
    )

    html_path = write_html_report(report, settings.drift_report_path)
    json_path = write_json_report(report, settings.monitoring_report_path)
    metrics_path = write_prometheus_metrics(report, settings.monitoring_metrics_path)

    assert html_path.exists()
    assert json_path.exists()
    assert metrics_path.exists()
    assert "mlops_drift_share" in metrics_path.read_text(encoding="utf-8")
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    assert persisted["current_quality"]["rows"] == 150
    assert persisted["data_drift"]["drift_share"] > 0.5
