"""Shared pytest fixtures."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlops_monitoring.api.main import create_app
from mlops_monitoring.config import Settings
from mlops_monitoring.data.loading import load_breast_cancer_dataset
from mlops_monitoring.models.loading import ModelBundle, load_model_bundle
from mlops_monitoring.training.train import train_and_save


def make_settings(root: Path, **overrides: Any) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    values: dict[str, Any] = {
        "_env_file": None,
        "app_env": "test",
        "log_json": False,
        "dataset_path": root / "breast_cancer.csv",
        "reference_data_path": root / "reference.csv",
        "model_path": root / "model",
        "prediction_log_path": root / "predictions.jsonl",
        "labels_path": root / "labels.jsonl",
        "drift_report_path": root / "reports" / "drift-report.html",
        "monitoring_report_path": root / "reports" / "monitoring-report.json",
        "monitoring_metrics_path": root / "reports" / "monitoring.prom",
        "evaluation_report_path": root / "reports" / "evaluation.json",
        "training_summary_path": root / "reports" / "training-summary.json",
        "mlflow_enabled": False,
        "monitoring_use_evidently": False,
        "min_f1": 0.75,
        "min_recall": 0.70,
        "min_roc_auc": 0.80,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(scope="session")
def demo_frame():
    return load_breast_cancer_dataset()


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory, demo_frame):
    root = tmp_path_factory.mktemp("trained-model")
    settings = make_settings(root)
    summary = train_and_save(settings, demo_frame)
    bundle = load_model_bundle(settings.model_path)
    return {"settings": settings, "summary": summary, "bundle": bundle, "root": root}


@pytest.fixture(scope="session")
def bundle(trained_run) -> ModelBundle:
    return trained_run["bundle"]


@pytest.fixture
def settings_factory(tmp_path):
    counter = itertools.count()

    def factory(**overrides: Any) -> Settings:
        root = tmp_path / f"case-{next(counter)}"
        return make_settings(root, **overrides)

    return factory


@pytest.fixture
def settings(trained_run, tmp_path) -> Settings:
    updated = {
        "prediction_log_path": tmp_path / "predictions.jsonl",
        "labels_path": tmp_path / "labels.jsonl",
        "drift_report_path": tmp_path / "reports" / "drift-report.html",
        "monitoring_report_path": tmp_path / "reports" / "monitoring-report.json",
        "monitoring_metrics_path": tmp_path / "reports" / "monitoring.prom",
    }
    return trained_run["settings"].model_copy(update=updated)


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def features_for_row(bundle):
    def build(row) -> dict[str, Any]:
        features: dict[str, Any] = {}
        for spec in bundle.feature_specs:
            value = row[spec.name]
            if hasattr(value, "item"):
                value = value.item()
            features[spec.name] = float(value) if spec.kind == "numeric" else str(value)
        return features

    return build


@pytest.fixture
def sample_features(demo_frame, features_for_row) -> dict[str, Any]:
    return features_for_row(demo_frame.iloc[0])
