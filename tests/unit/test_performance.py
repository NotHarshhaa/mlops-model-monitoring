"""Unit tests for metrics, gates, label joins, and performance degradation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlops_monitoring.monitoring.performance import (
    classification_metrics,
    evaluate_production_performance,
    join_predictions_and_labels,
    load_labels,
)
from mlops_monitoring.training.evaluate import GateConfig, check_gates


def test_perfect_classification_metrics():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    metrics = classification_metrics(y_true, y_pred, y_proba, labels=[0, 1])

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]
    assert metrics["per_class"]["1"]["support"] == 2


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError):
        classification_metrics([], [], None, labels=[0, 1])


def test_gates_pass_and_fail():
    config = GateConfig(min_f1=0.9, min_recall=0.9, min_roc_auc=0.9)
    good = {"f1": 0.95, "recall": 0.94, "roc_auc": 0.97}
    bad = {"f1": 0.85, "recall": 0.94, "roc_auc": 0.97}

    assert check_gates(good, config).passed
    result = check_gates(bad, config)
    assert not result.passed
    assert {failure["name"] for failure in result.failures} == {"min_f1"}


def test_gate_compares_against_baseline():
    config = GateConfig(min_f1=0.5, min_recall=0.5, min_roc_auc=0.5, baseline_tolerance=0.01)
    metrics = {"f1": 0.80, "recall": 0.80, "roc_auc": 0.90}
    baseline = {"f1": 0.90}

    result = check_gates(metrics, config, baseline_metrics=baseline)

    assert not result.passed
    assert any(check["name"] == "baseline_f1" for check in result.failures)


def test_load_labels_supports_jsonl_and_latest_value_wins(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"prediction_id": "a", "label": 0}),
                json.dumps({"prediction_id": "b", "label": 1}),
                json.dumps({"prediction_id": "a", "label": 1}),
            ]
        ),
        encoding="utf-8",
    )

    assert load_labels(path) == {"a": 1, "b": 1}

    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("prediction_id,label\nc,0\n", encoding="utf-8")
    assert load_labels(csv_path) == {"c": 0}


def test_join_predictions_and_labels_counts_unmatched():
    predictions = [
        {"prediction_id": "a", "prediction": 0},
        {"prediction_id": "missing", "prediction": 1},
        {"prediction_id": "b"},
    ]

    matched, skipped = join_predictions_and_labels(predictions, {"a": 0, "b": 1})

    assert len(matched) == 1
    assert matched[0]["label"] == 0
    assert skipped == 2


def test_production_performance_detects_degradation():
    predictions = [
        {
            "prediction_id": str(index),
            "prediction": 0,
            "probabilities": {"0": 0.9, "1": 0.1},
        }
        for index in range(4)
    ]
    labels = {"0": 0, "1": 1, "2": 0, "3": 1}

    result = evaluate_production_performance(
        predictions,
        labels,
        baseline_metrics={"f1": 0.9, "recall": 0.9, "accuracy": 0.9},
        drop_threshold=0.1,
    )

    assert result["status"] == "ok"
    assert result["matched_rows"] == 4
    assert result["degraded"]
    assert result["max_relative_drop"] > 0.1


def test_production_performance_without_labels():
    result = evaluate_production_performance([{"prediction_id": "a", "prediction": 1}], {})

    assert result["status"] == "no_labels"
    assert not result["degraded"]
