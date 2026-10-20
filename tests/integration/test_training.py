"""Integration tests for training and artifact creation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mlops_monitoring.models.loading import load_model_bundle
from mlops_monitoring.training.train import ModelAcceptanceError, train_and_save


def test_training_produces_accepted_artifact(trained_run):
    settings = trained_run["settings"]
    summary = trained_run["summary"]

    assert summary["status"] == "accepted"
    assert summary["gates"]["passed"]
    assert settings.model_artifact_path.exists()
    assert settings.model_metadata_path.exists()
    assert settings.reference_data_path.exists()
    assert settings.evaluation_report_path.exists()
    assert settings.training_summary_path.exists()

    metadata = json.loads(settings.model_metadata_path.read_text(encoding="utf-8"))
    assert metadata["model_name"] == settings.model_name
    assert metadata["feature_specs"]
    assert metadata["metrics"]["f1"] > 0.7
    assert metadata["metadata"]["dataset"]["rows"] > 0

    bundle = load_model_bundle(settings.model_path)
    assert bundle.model_name == summary["model_name"]
    assert bundle.model_version == summary["model_version"]


def test_gate_failure_rejects_candidate(trained_run, demo_frame):
    settings = trained_run["settings"].model_copy(
        update={
            "model_path": trained_run["root"] / "rejected-model",
            "evaluation_report_path": trained_run["root"] / "reports" / "rejected.json",
            "training_summary_path": trained_run["root"] / "reports" / "rejected-summary.json",
            "min_f1": 0.999,
            "min_recall": 0.999,
            "min_roc_auc": 0.999,
        }
    )
    rng = np.random.default_rng(0)
    noisy_frame = demo_frame.copy()
    noisy_frame["target"] = rng.integers(0, 2, size=len(noisy_frame))

    with pytest.raises(ModelAcceptanceError) as exc_info:
        train_and_save(settings, noisy_frame)

    assert not settings.model_artifact_path.exists()
    assert exc_info.value.report["gates"]["passed"] is False
    summary = json.loads(settings.training_summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "rejected"
