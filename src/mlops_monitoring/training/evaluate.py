"""Model evaluation and acceptance gates."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_monitoring.models.loading import ModelBundle
from mlops_monitoring.monitoring.performance import classification_metrics


@dataclass
class GateConfig:
    min_f1: float = 0.80
    min_recall: float = 0.75
    min_roc_auc: float = 0.85
    baseline_tolerance: float = 0.01


@dataclass
class GateResult:
    passed: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [check for check in self.checks if not check["passed"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "failures": self.failures,
        }


def check_gates(
    metrics: dict[str, Any],
    config: GateConfig,
    *,
    baseline_metrics: dict[str, Any] | None = None,
) -> GateResult:
    """Apply absolute quality gates and a baseline-relative F1 check."""
    checks: list[dict[str, Any]] = []

    for name, threshold in (
        ("f1", config.min_f1),
        ("recall", config.min_recall),
        ("roc_auc", config.min_roc_auc),
    ):
        value = metrics.get(name)
        passed = value is not None and float(value) >= threshold
        checks.append(
            {
                "name": f"min_{name}",
                "passed": bool(passed),
                "value": float(value) if value is not None else None,
                "threshold": threshold,
                "message": (
                    f"{name}={value:.4f} meets minimum {threshold:.4f}"
                    if passed
                    else f"{name} must be at least {threshold:.4f}"
                ),
            }
        )

    if baseline_metrics:
        baseline_f1 = baseline_metrics.get("f1")
        current_f1 = metrics.get("f1")
        if baseline_f1 is not None and current_f1 is not None:
            minimum = float(baseline_f1) - config.baseline_tolerance
            passed = float(current_f1) >= minimum
            checks.append(
                {
                    "name": "baseline_f1",
                    "passed": bool(passed),
                    "value": float(current_f1),
                    "baseline": float(baseline_f1),
                    "tolerance": config.baseline_tolerance,
                    "threshold": round(minimum, 6),
                    "message": (
                        f"f1={current_f1:.4f} is within {config.baseline_tolerance:.4f} "
                        f"of baseline {baseline_f1:.4f}"
                        if passed
                        else f"f1={current_f1:.4f} trails baseline {baseline_f1:.4f} "
                        f"by more than {config.baseline_tolerance:.4f}"
                    ),
                }
            )

    return GateResult(passed=all(check["passed"] for check in checks), checks=checks)


def evaluate_estimator(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted estimator on held-out data."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features) if hasattr(model, "predict_proba") else None
    final_estimator = getattr(model, "named_steps", {}).get("model", model)
    labels = list(getattr(final_estimator, "classes_", sorted(set(target.tolist()))))
    return classification_metrics(
        target.to_numpy(),
        np.asarray(predictions),
        y_proba=probabilities,
        labels=labels,
    )


def evaluate_bundle(
    bundle: ModelBundle,
    frame: pd.DataFrame,
    target_column: str = "target",
) -> dict[str, Any]:
    """Evaluate a persisted model bundle on a labelled data frame."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is not present")
    missing = [feature for feature in bundle.feature_names if feature not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation data is missing required features: {', '.join(missing)}")

    features = frame[bundle.feature_names]
    predictions = bundle.model.predict(features)
    probabilities = (
        bundle.model.predict_proba(features) if hasattr(bundle.model, "predict_proba") else None
    )
    return classification_metrics(
        frame[target_column].to_numpy(),
        np.asarray(predictions),
        y_proba=probabilities,
        labels=bundle.classes,
    )


def build_evaluation_report(
    *,
    model_name: str,
    model_version: str,
    metrics: dict[str, Any],
    gates: GateResult,
    dataset: dict[str, Any],
    candidate: str,
    candidates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_name": model_name,
        "model_version": model_version,
        "candidate": candidate,
        "dataset": dataset,
        "metrics": metrics,
        "gates": gates.to_dict(),
        "candidates": candidates or {},
    }


def save_evaluation_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
