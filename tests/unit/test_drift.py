"""Unit tests for drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_monitoring.monitoring.drift import (
    categorical_psi,
    detect_data_drift,
    detect_prediction_drift,
    numeric_psi,
)


def test_numeric_psi_is_zero_for_identical_distributions():
    rng = np.random.default_rng(1)
    reference = rng.normal(size=500)

    assert numeric_psi(reference, reference) < 1e-6


def test_numeric_psi_detects_shifted_distribution():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=500)
    current = rng.normal(loc=2.0, size=500)

    assert numeric_psi(reference, current) > 0.2


def test_numeric_psi_handles_constant_reference():
    assert numeric_psi([1.0] * 50, [1.0] * 50) == 0.0
    assert numeric_psi([1.0] * 50, [2.0] * 50) == 1.0


def test_categorical_psi_detects_new_distribution():
    reference = ["a"] * 80 + ["b"] * 20
    current = ["a"] * 20 + ["b"] * 80

    assert categorical_psi(reference, current) > 0.2


def test_detect_data_drift_flags_dataset():
    rng = np.random.default_rng(3)
    reference = pd.DataFrame(
        {
            "x": rng.normal(size=300),
            "y": rng.normal(size=300),
            "z": rng.normal(size=300),
        }
    )
    current = reference.copy()
    current["x"] += 3.0

    report = detect_data_drift(reference, current, psi_threshold=0.2, share_threshold=0.3)

    assert report["dataset_drifted"]
    assert report["drifted_features"] == ["x"]
    assert report["drift_share"] == round(1 / 3, 6)
    assert report["sufficient_data"]


def test_detect_data_drift_does_not_flag_identical_data():
    rng = np.random.default_rng(4)
    reference = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})

    report = detect_data_drift(reference, reference.copy())

    assert not report["dataset_drifted"]
    assert report["drifted_features"] == []


def test_drift_flags_are_suppressed_for_small_windows():
    rng = np.random.default_rng(5)
    reference = pd.DataFrame({"x": rng.normal(size=200)})
    current = reference.head(5).copy()
    current["x"] += 5.0

    report = detect_data_drift(reference, current, min_samples=20)

    assert not report["sufficient_data"]
    assert not report["dataset_drifted"]


def test_prediction_drift_detects_class_shift():
    reference = [0] * 200 + [1] * 100
    current = [0] * 100 + [1] * 200

    report = detect_prediction_drift(reference, current, threshold=0.2)

    assert report["status"] == "ok"
    assert report["drifted"]
    assert report["psi"] > 0.2
    assert report["current_distribution"]["1"] > report["current_distribution"]["0"]
