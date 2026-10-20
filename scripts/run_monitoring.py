"""Run data drift, prediction drift, and production performance monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from mlops_monitoring.config import get_settings
from mlops_monitoring.data.validation import data_quality_summary
from mlops_monitoring.models.loading import ModelArtifactError, ModelLoader
from mlops_monitoring.monitoring.drift import detect_data_drift, detect_prediction_drift
from mlops_monitoring.monitoring.performance import (
    evaluate_production_performance,
    load_labels,
)
from mlops_monitoring.monitoring.reports import (
    build_monitoring_report,
    evaluate_alerts,
    generate_evidently_report,
    write_html_report,
    write_json_report,
    write_prometheus_metrics,
)
from mlops_monitoring.observability.logging import configure_logging
from mlops_monitoring.observability.prediction_log import read_predictions

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", help="Reference CSV used during training")
    parser.add_argument("--predictions", help="Prediction log (JSONL)")
    parser.add_argument("--labels", help="JSONL or CSV ground-truth labels")
    parser.add_argument(
        "--last-n", type=int, default=5000, help="Most recent predictions to monitor"
    )
    parser.add_argument(
        "--min-samples", type=int, default=20, help="Minimum current rows for drift flags"
    )
    parser.add_argument("--output-json", help="Monitoring report JSON path")
    parser.add_argument("--output-html", help="Monitoring report HTML path")
    parser.add_argument("--metrics-file", help="Prometheus textfile output path")
    parser.add_argument(
        "--no-evidently",
        action="store_true",
        help="Use the built-in HTML report instead of Evidently",
    )
    parser.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Exit with status 1 when any monitoring alert is active",
    )
    return parser.parse_args(argv)


def _path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    reference_path = _path(args.reference, settings.reference_data_path)
    predictions_path = _path(args.predictions, settings.prediction_log_path)
    labels_path = _path(args.labels, settings.labels_path)
    report_path = _path(args.output_json, settings.monitoring_report_path)
    html_path = _path(args.output_html, settings.drift_report_path)
    metrics_path = _path(args.metrics_file, settings.monitoring_metrics_path)

    if not reference_path.exists():
        logger.error("Reference data not found", extra={"path": str(reference_path)})
        return 2

    reference = pd.read_csv(reference_path)
    predictions = read_predictions(predictions_path, limit=args.last_n)
    if not predictions:
        logger.error(
            "No predictions found; generate inference traffic before monitoring",
            extra={"path": str(predictions_path)},
        )
        return 2

    model_loader = ModelLoader(
        settings.model_path,
        default_model_name=settings.model_name,
        default_model_version=settings.model_version,
    )
    bundle = None
    try:
        bundle = model_loader.load()
    except ModelArtifactError as exc:
        logger.warning(
            "Model artifact unavailable; monitoring features without reference predictions",
            extra={"error": str(exc)},
        )

    if bundle is not None:
        feature_names = bundle.feature_names
        feature_specs = bundle.feature_specs
        missing_reference = [name for name in feature_names if name not in reference.columns]
        if missing_reference:
            logger.error(
                "Reference data is missing model features",
                extra={"features": missing_reference},
            )
            return 2
        current_features = _current_feature_frame(predictions, feature_names)
        reference_features = reference[feature_names]
        reference_predictions = bundle.model.predict(reference_features).tolist()
        baseline_metrics = bundle.metrics
        model_name = bundle.model_name
        model_version = bundle.model_version
    else:
        feature_names = [column for column in reference.columns if column != settings.target_column]
        feature_specs = None
        current_features = _current_feature_frame(predictions, feature_names)
        reference_features = reference[feature_names]
        reference_predictions = None
        baseline_metrics = {}
        model_name = settings.model_name
        model_version = settings.model_version or "unknown"

    if current_features.empty:
        logger.error("Prediction records contain no usable feature values")
        return 2

    try:
        drift = detect_data_drift(
            reference_features,
            current_features,
            feature_specs=feature_specs,
            psi_threshold=settings.drift_threshold,
            share_threshold=settings.drift_share_threshold,
            min_samples=args.min_samples,
        )
    except ValueError as exc:
        logger.error("Drift calculation failed", extra={"error": str(exc)})
        return 2

    prediction_drift = None
    if reference_predictions is not None:
        prediction_drift = detect_prediction_drift(
            reference_predictions,
            [record.get("prediction") for record in predictions],
            threshold=settings.drift_threshold,
            min_samples=args.min_samples,
        )

    labels = load_labels(labels_path)
    performance = evaluate_production_performance(
        predictions,
        labels,
        baseline_metrics=baseline_metrics,
        drop_threshold=settings.performance_drop_threshold,
    )

    current_quality = data_quality_summary(current_features)
    alerts = evaluate_alerts(
        drift=drift,
        prediction_drift=prediction_drift,
        performance=performance,
        current_quality=current_quality,
        drift_share_threshold=settings.drift_share_threshold,
        performance_drop_threshold=settings.performance_drop_threshold,
    )
    report = build_monitoring_report(
        reference=reference_features,
        current=current_features,
        drift=drift,
        prediction_drift=prediction_drift,
        performance=performance,
        alerts=alerts,
        model_name=model_name,
        model_version=model_version,
        window={
            "last_n": args.last_n,
            "predictions_path": str(predictions_path),
            "labels_path": str(labels_path) if labels_path.exists() else None,
            "prediction_rows": len(predictions),
            "labelled_rows": performance.get("matched_rows", 0),
        },
    )

    use_evidently = settings.monitoring_use_evidently and not args.no_evidently
    evidently_used = False
    evidently_error = None
    if use_evidently:
        evidently_used, evidently_error = generate_evidently_report(
            reference_features,
            current_features,
            html_path,
        )
    if not evidently_used:
        write_html_report(report, html_path)

    report["tools"] = {
        "evidently": {
            "requested": use_evidently,
            "used": evidently_used,
            "error": evidently_error,
        },
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    write_json_report(report, report_path)
    write_prometheus_metrics(report, metrics_path)

    print(
        json.dumps(
            {
                "status": report["summary"]["status"],
                "drift_share": report["summary"]["drift_share"],
                "dataset_drifted": report["summary"]["dataset_drifted"],
                "prediction_drifted": report["summary"]["prediction_drifted"],
                "performance_status": report["summary"]["performance_status"],
                "performance_degraded": report["summary"]["performance_degraded"],
                "alerts": report["alerts"],
                "html_report": str(html_path),
                "json_report": str(report_path),
                "metrics_file": str(metrics_path),
            },
            indent=2,
            default=str,
        )
    )

    if alerts:
        logger.warning(
            "Monitoring alerts active",
            extra={"alerts": [alert.code for alert in alerts]},
        )
        return 1 if args.fail_on_alert else 0
    return 0


def _current_feature_frame(
    predictions: list[dict[str, object]],
    feature_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    missing_counts: dict[str, int] = {name: 0 for name in feature_names}
    for record in predictions:
        features = record.get("features")
        if not isinstance(features, dict):
            continue
        row: dict[str, object] = {}
        for name in feature_names:
            if name in features and features[name] is not None:
                row[name] = features[name]
            else:
                missing_counts[name] += 1
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=feature_names)

    frame = pd.DataFrame(rows, columns=feature_names)
    dropped = sum(missing_counts.values())
    if dropped:
        logger.warning(
            "Some prediction records were missing feature values",
            extra={"missing_values": dropped, "missing_by_feature": missing_counts},
        )
    return frame


if __name__ == "__main__":
    sys.exit(main())
