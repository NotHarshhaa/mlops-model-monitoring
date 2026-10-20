"""Monitoring report generation, alert evaluation, and metric export."""

from __future__ import annotations

import html
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlops_monitoring.data.validation import data_quality_summary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Alert:
    code: str
    severity: str
    message: str
    metric: str
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "metric": self.metric,
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.threshold is not None:
            payload["threshold"] = self.threshold
        return payload


def evaluate_alerts(
    *,
    drift: dict[str, Any],
    prediction_drift: dict[str, Any] | None,
    performance: dict[str, Any],
    current_quality: dict[str, Any],
    drift_share_threshold: float = 0.3,
    performance_drop_threshold: float = 0.1,
) -> list[Alert]:
    """Turn monitoring measurements into a small set of actionable alerts."""
    alerts: list[Alert] = []

    if drift.get("dataset_drifted"):
        features = ", ".join(drift.get("drifted_features", []))
        alerts.append(
            Alert(
                code="data_drift",
                severity="warning",
                message=(
                    f"Data drift detected in {len(drift.get('drifted_features', []))} "
                    f"feature(s): {features}"
                ),
                metric="drift_share",
                value=drift.get("drift_share"),
                threshold=drift_share_threshold,
            )
        )

    if prediction_drift and prediction_drift.get("drifted"):
        alerts.append(
            Alert(
                code="prediction_drift",
                severity="warning",
                message="The distribution of predicted classes has shifted.",
                metric="prediction_drift_psi",
                value=prediction_drift.get("psi"),
                threshold=prediction_drift.get("threshold"),
            )
        )

    if performance.get("degraded"):
        metrics = ", ".join(performance.get("degraded_metrics", []))
        alerts.append(
            Alert(
                code="performance_degradation",
                severity="critical",
                message=f"Production performance dropped beyond the threshold for: {metrics}",
                metric="performance_drop",
                value=performance.get("max_relative_drop"),
                threshold=performance_drop_threshold,
            )
        )

    missing_fraction = float(current_quality.get("missing_fraction", 0.0))
    if missing_fraction > 0.1:
        alerts.append(
            Alert(
                code="data_quality_missing_values",
                severity="warning",
                message=f"{missing_fraction:.1%} of incoming feature cells are missing.",
                metric="missing_fraction",
                value=round(missing_fraction, 6),
                threshold=0.1,
            )
        )

    if current_quality.get("infinite_values", 0) > 0:
        alerts.append(
            Alert(
                code="data_quality_infinite_values",
                severity="critical",
                message=f"{current_quality['infinite_values']} infinite value(s) observed.",
                metric="infinite_values",
                value=float(current_quality["infinite_values"]),
                threshold=0.0,
            )
        )

    return alerts


def build_monitoring_report(
    *,
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift: dict[str, Any],
    prediction_drift: dict[str, Any] | None,
    performance: dict[str, Any],
    alerts: list[Alert],
    model_name: str = "",
    model_version: str = "",
    window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the machine-readable monitoring report."""
    current_quality = data_quality_summary(current)
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model": {"name": model_name, "version": model_version},
        "window": window or {},
        "summary": {
            "status": "alert" if alerts else "ok",
            "alert_count": len(alerts),
            "dataset_drifted": bool(drift.get("dataset_drifted")),
            "drift_share": drift.get("drift_share"),
            "prediction_drifted": bool(prediction_drift and prediction_drift.get("drifted")),
            "performance_degraded": bool(performance.get("degraded")),
            "performance_status": performance.get("status"),
        },
        "reference_quality": data_quality_summary(reference),
        "current_quality": current_quality,
        "data_drift": drift,
        "prediction_drift": prediction_drift,
        "performance": performance,
        "alerts": [alert.to_dict() for alert in alerts],
    }


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "—"
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def _badge(text: str, kind: str) -> str:
    return f'<span class="badge {html.escape(kind)}">{html.escape(text)}</span>'


def render_html_report(report: dict[str, Any]) -> str:
    """Render a dependency-free HTML report from the monitoring document."""
    summary = report.get("summary", {})
    drift = report.get("data_drift", {})
    prediction = report.get("prediction_drift") or {}
    performance = report.get("performance", {})
    current_quality = report.get("current_quality", {})
    alerts = report.get("alerts", [])

    status_kind = "critical" if summary.get("status") == "alert" else "ok"
    status_text = "ALERT" if summary.get("status") == "alert" else "OK"

    feature_rows = []
    for feature in drift.get("features", []):
        drifted = feature.get("drifted", False)
        feature_rows.append(
            "<tr>"
            f"<td>{html.escape(str(feature.get('feature')))}</td>"
            f"<td>{html.escape(str(feature.get('kind')))}</td>"
            f"<td>{_format_value(feature.get('psi'))}</td>"
            f"<td>{_format_value(feature.get('p_value'))}</td>"
            f"<td>{_badge('drifted' if drifted else 'stable', 'critical' if drifted else 'ok')}</td>"
            "</tr>"
        )

    alert_rows = []
    for alert in alerts:
        alert_rows.append(
            "<tr>"
            f"<td>{html.escape(str(alert.get('code')))}</td>"
            f"<td>{_badge(str(alert.get('severity')), str(alert.get('severity')))}</td>"
            f"<td>{html.escape(str(alert.get('message')))}</td>"
            f"<td>{_format_value(alert.get('value'))}</td>"
            f"<td>{_format_value(alert.get('threshold'))}</td>"
            "</tr>"
        )

    metric_rows = []
    baseline = performance.get("baseline", {})
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        current_value = performance.get("metrics", {}).get(name)
        baseline_value = baseline.get(name)
        drop = performance.get("relative_drops", {}).get(name)
        metric_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{_format_value(current_value)}</td>"
            f"<td>{_format_value(baseline_value)}</td>"
            f"<td>{_format_value(drop)}</td>"
            "</tr>"
        )

    prediction_rows = []
    if prediction.get("status") == "ok":
        current_distribution = prediction.get("current_distribution", {})
        reference_distribution = prediction.get("reference_distribution", {})
        for label in sorted(set(current_distribution) | set(reference_distribution)):
            prediction_rows.append(
                "<tr>"
                f"<td>{html.escape(str(label))}</td>"
                f"<td>{_format_value(reference_distribution.get(label))}</td>"
                f"<td>{_format_value(current_distribution.get(label))}</td>"
                "</tr>"
            )
    else:
        prediction_rows.append(
            '<tr><td colspan="3">'
            f"{html.escape(str(prediction.get('status', 'not available')))}</td></tr>"
        )

    rows = (
        f"{report.get('reference_quality', {}).get('rows', 0):,}"
        if isinstance(report.get("reference_quality", {}).get("rows", 0), int)
        else "—"
    )
    generated_at = html.escape(str(report.get("generated_at", "")))
    model = report.get("model", {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Monitoring Report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         margin: 0; background: #f5f7fa; color: #1f2933; }}
  header {{ background: #172b4d; color: white; padding: 28px 36px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 0; opacity: .8; }}
  main {{ max-width: 1180px; margin: 24px auto; padding: 0 20px 40px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }}
  .card {{ background: white; border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  .card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #6b778c; }}
  .card .value {{ font-size: 24px; font-weight: 650; margin-top: 6px; }}
  section {{ background: white; border-radius: 10px; padding: 20px 22px; margin-top: 20px;
             box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  h2 {{ margin: 0 0 12px; font-size: 18px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 9px 10px; border-bottom: 1px solid #e4e7eb; }}
  th {{ color: #52606d; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  tr:last-child td {{ border-bottom: none; }}
  .badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
  .badge.ok {{ background: #e3f9e5; color: #1f7a33; }}
  .badge.warning {{ background: #fff3cd; color: #8a6100; }}
  .badge.critical {{ background: #ffe3e3; color: #a61b1b; }}
  .badge.info {{ background: #e3f0ff; color: #1f4e8c; }}
  code {{ background: #eef1f5; padding: 1px 5px; border-radius: 4px; }}
  footer {{ text-align: center; color: #6b778c; font-size: 12px; margin-top: 26px; }}
</style>
</head>
<body>
<header>
  <h1>Model Monitoring Report {_badge(status_text, status_kind)}</h1>
  <p>{html.escape(str(model.get("name", "model")))} v{html.escape(str(model.get("version", "")))} · generated {generated_at}</p>
</header>
<main>
  <div class="cards">
    <div class="card"><div class="label">Reference rows</div><div class="value">{rows}</div></div>
    <div class="card"><div class="label">Current rows</div><div class="value">{current_quality.get("rows", 0)}</div></div>
    <div class="card"><div class="label">Drift share</div><div class="value">{_format_value(summary.get("drift_share"))}</div></div>
    <div class="card"><div class="label">F1 (labelled)</div><div class="value">{_format_value(performance.get("metrics", {}).get("f1"))}</div></div>
    <div class="card"><div class="label">Alerts</div><div class="value">{len(alerts)}</div></div>
  </div>

  <section>
    <h2>Alerts</h2>
    <table>
      <thead><tr><th>Code</th><th>Severity</th><th>Message</th><th>Value</th><th>Threshold</th></tr></thead>
      <tbody>{"".join(alert_rows) or '<tr><td colspan="5">No alerts. All monitored signals are within thresholds.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Data drift by feature</h2>
    <p>Drift threshold: PSI ≥ {_format_value(drift.get("psi_threshold"))}; dataset threshold:
       {_format_value(drift.get("share_threshold"))} of {drift.get("evaluated_features", 0)} features.</p>
    <table>
      <thead><tr><th>Feature</th><th>Kind</th><th>PSI</th><th>p-value</th><th>Status</th></tr></thead>
      <tbody>{"".join(feature_rows) or '<tr><td colspan="5">No features evaluated.</td></tr>'}</tbody>
    </table>
  </section>

  <section>
    <h2>Prediction drift</h2>
    <p>PSI: {_format_value(prediction.get("psi"))} · status:
       {_badge("drifted" if prediction.get("drifted") else "stable", "critical" if prediction.get("drifted") else "ok")}</p>
    <table>
      <thead><tr><th>Class</th><th>Reference share</th><th>Current share</th></tr></thead>
      <tbody>{"".join(prediction_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Production performance</h2>
    <p>{html.escape(str(performance.get("status", "unavailable")))} · matched labelled rows:
       {performance.get("matched_rows", 0)} · degradation threshold:
       {_format_value(performance.get("threshold"))}</p>
    <table>
      <thead><tr><th>Metric</th><th>Current</th><th>Baseline</th><th>Relative drop</th></tr></thead>
      <tbody>{"".join(metric_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>Data quality</h2>
    <table>
      <thead><tr><th>Signal</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Missing cells</td><td>{current_quality.get("missing_cells", 0)} ({_format_value(float(current_quality.get("missing_fraction", 0.0)) * 100, 2)}%)</td></tr>
        <tr><td>Duplicate rows</td><td>{current_quality.get("duplicate_rows", 0)}</td></tr>
        <tr><td>Infinite values</td><td>{current_quality.get("infinite_values", 0)}</td></tr>
        <tr><td>Columns</td><td>{current_quality.get("columns", 0)}</td></tr>
      </tbody>
    </table>
  </section>

  <footer>
    Drift indicates change, not necessarily failure. Confirm with labelled performance before retraining.
  </footer>
</main>
</body>
</html>
"""


def write_html_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(report), encoding="utf-8")
    return path


def write_json_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def generate_evidently_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    path: Path,
) -> tuple[bool, str | None]:
    """Generate an Evidently data-drift report, tolerating API changes.

    Evidently's public API changed between the 0.4 and 0.6+ releases. Both
    variants are attempted; the caller falls back to the built-in HTML report
    when Evidently is unavailable or fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:  # Evidently >= 0.6
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report([DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        snapshot.save_html(str(path))
        return True, None
    except Exception as exc:
        logger.debug("Modern Evidently API unavailable", extra={"error": str(exc)})

    try:  # Evidently 0.4.x
        from evidently.metric_preset import DataDriftPreset
        from evidently.report import Report

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(path))
        return True, None
    except Exception as exc:
        logger.warning("Evidently report generation failed", extra={"error": str(exc)})
        return False, str(exc)


def _prometheus_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _metric_line(
    name: str,
    value: float | int | None,
    labels: dict[str, str] | None = None,
) -> str:
    if value is None:
        return ""
    label_text = ""
    if labels:
        encoded = ",".join(
            f'{key}="{_prometheus_label(str(label_value))}"' for key, label_value in labels.items()
        )
        label_text = f"{{{encoded}}}"
    return f"{name}{label_text} {float(value):.6f}\n"


def write_prometheus_metrics(report: dict[str, Any], path: Path) -> Path:
    """Write gauges in Prometheus textfile format for node_exporter."""
    drift = report.get("data_drift", {})
    performance = report.get("performance", {})
    prediction = report.get("prediction_drift") or {}
    model = report.get("model", {})
    model_labels = {
        "model_name": str(model.get("name", "")),
        "model_version": str(model.get("version", "")),
    }
    alerts = report.get("alerts", [])

    lines = [
        "# HELP mlops_monitoring_last_run_timestamp_seconds Unix timestamp of the latest monitoring run.\n",
        "# TYPE mlops_monitoring_last_run_timestamp_seconds gauge\n",
        _metric_line(
            "mlops_monitoring_last_run_timestamp_seconds", datetime.now(tz=UTC).timestamp()
        ),
        "# HELP mlops_drift_share Share of evaluated features flagged as drifted.\n",
        "# TYPE mlops_drift_share gauge\n",
        _metric_line("mlops_drift_share", drift.get("drift_share"), model_labels),
        "# HELP mlops_dataset_drifted Whether the dataset is considered drifted.\n",
        "# TYPE mlops_dataset_drifted gauge\n",
        _metric_line(
            "mlops_dataset_drifted", 1 if drift.get("dataset_drifted") else 0, model_labels
        ),
        "# HELP mlops_feature_drift Per-feature PSI drift statistic.\n",
        "# TYPE mlops_feature_drift gauge\n",
    ]

    for feature in drift.get("features", []):
        lines.append(
            _metric_line(
                "mlops_feature_drift",
                feature.get("psi"),
                {**model_labels, "feature": str(feature.get("feature", ""))},
            )
        )

    lines.extend(
        [
            "# HELP mlops_prediction_drift_psi PSI of the predicted-class distribution.\n",
            "# TYPE mlops_prediction_drift_psi gauge\n",
            _metric_line("mlops_prediction_drift_psi", prediction.get("psi"), model_labels),
            "# HELP mlops_model_f1 F1 in the monitored labelled window.\n",
            "# TYPE mlops_model_f1 gauge\n",
            _metric_line("mlops_model_f1", performance.get("metrics", {}).get("f1"), model_labels),
            "# HELP mlops_baseline_f1 Baseline F1 used for comparison.\n",
            "# TYPE mlops_baseline_f1 gauge\n",
            _metric_line(
                "mlops_baseline_f1", performance.get("baseline", {}).get("f1"), model_labels
            ),
            "# HELP mlops_performance_drop Largest relative performance drop vs baseline.\n",
            "# TYPE mlops_performance_drop gauge\n",
            _metric_line(
                "mlops_performance_drop", performance.get("max_relative_drop"), model_labels
            ),
            "# HELP mlops_monitoring_rows Number of prediction rows in the window.\n",
            "# TYPE mlops_monitoring_rows gauge\n",
            _metric_line(
                "mlops_monitoring_rows",
                report.get("current_quality", {}).get("rows"),
                model_labels,
            ),
            "# HELP mlops_monitoring_alerts Number of active monitoring alerts by severity.\n",
            "# TYPE mlops_monitoring_alerts gauge\n",
        ]
    )
    for severity in ("warning", "critical"):
        count = sum(1 for alert in alerts if alert.get("severity") == severity)
        lines.append(
            _metric_line("mlops_monitoring_alerts", count, {**model_labels, "severity": severity})
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="utf-8")
    os.replace(temporary, path)
    return path
