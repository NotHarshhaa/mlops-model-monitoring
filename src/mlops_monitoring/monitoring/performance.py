"""Classification metrics and production performance monitoring."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

TRACKED_METRICS = ("f1", "recall", "precision", "accuracy", "roc_auc")


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: np.ndarray | None = None,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Calculate threshold, probability, and per-class classification metrics."""
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if truth.shape[0] != predicted.shape[0]:
        raise ValueError("y_true and y_pred must have the same length")
    if truth.shape[0] == 0:
        raise ValueError("Cannot calculate metrics for an empty dataset")

    if labels is None:
        labels = sorted(set(truth.tolist()) | set(predicted.tolist()), key=str)
    if len(labels) < 2:
        raise ValueError("At least two classes are required to calculate metrics")

    binary = len(labels) == 2
    average = "binary" if binary else "macro"
    pos_label = labels[1] if binary else 1

    precision = precision_score(
        truth, predicted, labels=labels, average=average, pos_label=pos_label, zero_division=0
    )
    recall = recall_score(
        truth, predicted, labels=labels, average=average, pos_label=pos_label, zero_division=0
    )
    f1 = f1_score(
        truth, predicted, labels=labels, average=average, pos_label=pos_label, zero_division=0
    )
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )

    metrics: dict[str, Any] = {
        "accuracy": _safe_float(accuracy_score(truth, predicted)),
        "precision": _safe_float(precision),
        "recall": _safe_float(recall),
        "f1": _safe_float(f1),
        "roc_auc": None,
        "log_loss": None,
        "support": int(truth.shape[0]),
        "labels": [str(label) for label in labels],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "per_class": {},
    }

    if y_proba is not None:
        probabilities = np.asarray(y_proba, dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[0] == truth.shape[0]:
            try:
                if binary:
                    metrics["roc_auc"] = _safe_float(
                        roc_auc_score(truth == labels[1], probabilities[:, 1])
                    )
                else:
                    metrics["roc_auc"] = _safe_float(
                        roc_auc_score(
                            truth,
                            probabilities,
                            labels=labels,
                            multi_class="ovr",
                            average="macro",
                        )
                    )
            except ValueError:
                metrics["roc_auc"] = None
            try:
                metrics["log_loss"] = _safe_float(log_loss(truth, probabilities, labels=labels))
            except ValueError:
                metrics["log_loss"] = None

    for index, label in enumerate(labels):
        metrics["per_class"][str(label)] = {
            "precision": _safe_float(per_precision[index]),
            "recall": _safe_float(per_recall[index]),
            "f1": _safe_float(per_f1[index]),
            "support": int(support[index]),
        }

    return metrics


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten scalar metrics for MLflow logging."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and value is not None:
            flat[name] = float(value)
    return flat


def load_labels(path: Path) -> dict[str, Any]:
    """Load ground-truth labels from CSV or JSONL.

    Expected columns are ``prediction_id`` and ``label`` (``target`` is also
    accepted). When a prediction id appears more than once, the latest value
    wins so corrected labels can be appended over time.
    """
    if not path.exists():
        return {}

    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        records = frame.to_dict(orient="records")
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping malformed label line",
                        extra={"path": str(path), "line_number": line_number},
                    )
                    continue
                if isinstance(record, dict):
                    records.append(record)

    labels: dict[str, Any] = {}
    for record in records:
        prediction_id = record.get("prediction_id") or record.get("id")
        label = record.get("label", record.get("target"))
        if prediction_id is None or label is None:
            continue
        if isinstance(label, float) and label.is_integer():
            label = int(label)
        labels[str(prediction_id)] = label
    return labels


def join_predictions_and_labels(
    predictions: list[dict[str, Any]],
    labels: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Join logged predictions to labels and return matched records plus a skip count."""
    matched: list[dict[str, Any]] = []
    skipped = 0
    for record in predictions:
        prediction_id = record.get("prediction_id")
        if prediction_id is None or "prediction" not in record:
            skipped += 1
            continue
        key = str(prediction_id)
        if key not in labels:
            skipped += 1
            continue
        enriched = dict(record)
        enriched["label"] = labels[key]
        matched.append(enriched)
    return matched, skipped


def _probability_matrix(
    matched: list[dict[str, Any]],
    labels: list[Any],
) -> np.ndarray | None:
    matrix = np.zeros((len(matched), len(labels)), dtype=float)
    for row_index, record in enumerate(matched):
        probabilities = record.get("probabilities")
        if not isinstance(probabilities, dict):
            return None
        for column_index, label in enumerate(labels):
            value = probabilities.get(str(label))
            if value is None:
                return None
            matrix[row_index, column_index] = float(value)
    return matrix


def evaluate_production_performance(
    predictions: list[dict[str, Any]],
    labels: dict[str, Any],
    *,
    baseline_metrics: dict[str, Any] | None = None,
    drop_threshold: float = 0.1,
) -> dict[str, Any]:
    """Measure labelled production performance and compare it with a baseline."""
    matched, skipped = join_predictions_and_labels(predictions, labels)
    result: dict[str, Any] = {
        "status": "no_labels" if not labels else "no_matches",
        "matched_rows": len(matched),
        "unmatched_rows": skipped,
        "threshold": drop_threshold,
        "metrics": {},
        "baseline": baseline_metrics or {},
        "relative_drops": {},
        "max_relative_drop": 0.0,
        "degraded": False,
        "degraded_metrics": [],
    }

    if not labels or not matched:
        return result

    y_true = [record["label"] for record in matched]
    y_pred = [record["prediction"] for record in matched]
    class_labels = sorted(set(y_true) | set(y_pred), key=str)
    probabilities = _probability_matrix(matched, class_labels) if len(class_labels) == 2 else None

    try:
        metrics = classification_metrics(
            y_true,
            y_pred,
            y_proba=probabilities,
            labels=class_labels,
        )
    except ValueError as exc:
        result["status"] = "invalid_labels"
        result["error"] = str(exc)
        return result

    result["status"] = "ok"
    result["metrics"] = metrics

    baseline = baseline_metrics or {}
    drops: dict[str, float] = {}
    degraded_metrics: list[str] = []
    for metric_name in TRACKED_METRICS:
        baseline_value = _safe_float(baseline.get(metric_name))
        current_value = _safe_float(metrics.get(metric_name))
        if baseline_value is None or current_value is None or baseline_value <= 0:
            continue
        drop = max(0.0, (baseline_value - current_value) / baseline_value)
        drops[metric_name] = round(drop, 6)
        if drop > drop_threshold:
            degraded_metrics.append(metric_name)

    result["relative_drops"] = drops
    result["max_relative_drop"] = round(max(drops.values(), default=0.0), 6)
    result["degraded"] = bool(degraded_metrics)
    result["degraded_metrics"] = degraded_metrics
    return result
