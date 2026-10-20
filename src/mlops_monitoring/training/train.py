"""Reproducible model training, experiment tracking, and artifact creation."""

from __future__ import annotations

import inspect
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlops_monitoring.config import Settings
from mlops_monitoring.data.loading import (
    dataset_fingerprint,
    load_dataset,
    resolve_target_names,
    split_dataset,
)
from mlops_monitoring.data.validation import validate_dataset
from mlops_monitoring.features.preprocessing import (
    build_feature_specs,
    build_model_pipeline,
    feature_statistics,
    split_feature_types,
)
from mlops_monitoring.models.loading import ModelBundle, save_model_bundle
from mlops_monitoring.monitoring.performance import flatten_metrics
from mlops_monitoring.training.evaluate import (
    GateConfig,
    GateResult,
    build_evaluation_report,
    check_gates,
    evaluate_estimator,
    save_evaluation_report,
)

logger = logging.getLogger(__name__)

BASELINE_MODEL = "logistic-regression"

CANDIDATE_FACTORIES: dict[str, Callable[[int], Any]] = {
    BASELINE_MODEL: lambda seed: LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    ),
    "random-forest": lambda seed: RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    ),
}


class ModelAcceptanceError(RuntimeError):
    """Raised when no candidate model passes the configured evaluation gates."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        self.report = report
        super().__init__(message)


@dataclass
class Candidate:
    name: str
    pipeline: Pipeline
    metrics: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)


def train_candidates(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    target_column: str,
    *,
    random_state: int = 42,
    candidates: list[str] | None = None,
) -> tuple[dict[str, Candidate], list[str], list[str]]:
    """Fit each candidate pipeline and evaluate it on the held-out split."""
    numeric_features, categorical_features = split_feature_types(train_frame, target_column)
    feature_names = numeric_features + categorical_features
    selected_names = candidates or list(CANDIDATE_FACTORIES)

    unknown = [name for name in selected_names if name not in CANDIDATE_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown candidate model(s): {', '.join(unknown)}")

    x_train = train_frame[feature_names]
    y_train = train_frame[target_column]
    x_test = test_frame[feature_names]
    y_test = test_frame[target_column]

    fitted: dict[str, Candidate] = {}
    for name in selected_names:
        estimator = CANDIDATE_FACTORIES[name](random_state)
        pipeline = build_model_pipeline(estimator, numeric_features, categorical_features)
        pipeline.fit(x_train, y_train)
        metrics = evaluate_estimator(pipeline, x_test, y_test)
        fitted[name] = Candidate(
            name=name,
            pipeline=pipeline,
            metrics=metrics,
            params=_jsonable_params(estimator.get_params()),
        )
        logger.info(
            "Candidate evaluated",
            extra={
                "candidate": name,
                "f1": metrics.get("f1"),
                "recall": metrics.get("recall"),
                "roc_auc": metrics.get("roc_auc"),
            },
        )

    return fitted, numeric_features, categorical_features


def _jsonable_params(params: dict[str, Any]) -> dict[str, Any]:
    serializable: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            serializable[key] = value
        else:
            serializable[key] = str(value)
    return serializable


def _select_candidate(
    candidates: dict[str, Candidate],
    gates: dict[str, GateResult],
) -> str:
    def score(candidate: Candidate) -> tuple[float, float]:
        f1 = candidate.metrics.get("f1") or 0.0
        roc_auc = candidate.metrics.get("roc_auc") or 0.0
        return float(f1), float(roc_auc)

    passing = [name for name, gate in gates.items() if gate.passed]
    pool = passing or list(candidates)
    selected = max(pool, key=lambda name: score(candidates[name]))
    if not passing:
        logger.warning("No candidate passed all gates; best candidate will be rejected")
    return selected


def _log_to_mlflow(
    candidate: Candidate,
    *,
    settings: Settings,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    evaluation_path: Path,
    dataset_info: dict[str, Any],
) -> dict[str, Any]:
    """Log a training run to MLflow. Failures are reported but never fatal."""
    result: dict[str, Any] = {
        "enabled": settings.mlflow_enabled,
        "tracking_uri": settings.mlflow_uri,
        "run_id": None,
        "experiment_id": None,
        "registered_version": None,
        "error": None,
    }
    if not settings.mlflow_enabled:
        return result

    try:  # pragma: no cover - requires a live or local MLflow store
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient

        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
        if settings.mlflow_uri.startswith("file:"):
            # MLflow 3 blocks the file store by default; allow it for this
            # explicit local-only workflow.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(settings.mlflow_uri)
        experiment = mlflow.set_experiment(settings.mlflow_experiment_name)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=f"{settings.model_name}-{candidate.name}-{timestamp}",
        ) as run:
            mlflow.log_params(
                {
                    **candidate.params,
                    "candidate": candidate.name,
                    "test_size": settings.test_size,
                    "random_state": settings.random_state,
                }
            )
            mlflow.log_metrics(flatten_metrics(candidate.metrics, prefix="test_"))
            mlflow.set_tags(
                {
                    "dataset_fingerprint": dataset_info["fingerprint"],
                    "dataset_rows": str(dataset_info["rows"]),
                    "target_column": settings.target_column,
                    "sklearn_version": sklearn.__version__,
                }
            )
            mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")

            signature = infer_signature(x_train, candidate.pipeline.predict(x_train))
            log_kwargs: dict[str, Any] = {
                "signature": signature,
                "input_example": x_train.head(3),
            }
            log_model_parameters = inspect.signature(mlflow.sklearn.log_model).parameters
            if "skops_trusted_types" in log_model_parameters:
                log_kwargs["skops_trusted_types"] = ["numpy.dtype"]
            if "name" in log_model_parameters:
                log_kwargs["name"] = "model"
            else:  # pragma: no cover - MLflow < 2.17
                log_kwargs["artifact_path"] = "model"
            mlflow.sklearn.log_model(candidate.pipeline, **log_kwargs)

            result["run_id"] = run.info.run_id
            result["experiment_id"] = run.info.experiment_id

            if settings.mlflow_register_model:
                registered = mlflow.register_model(
                    f"runs:/{run.info.run_id}/model", settings.model_name
                )
                client = MlflowClient()
                try:
                    client.set_registered_model_alias(
                        settings.model_name, "champion", registered.version
                    )
                except Exception:  # pragma: no cover - older registry backend
                    logger.warning("Unable to set 'champion' model alias")
                result["registered_version"] = str(registered.version)
    except Exception as exc:
        result["error"] = str(exc)
        logger.warning(
            "MLflow logging skipped because the tracking server is unavailable or misconfigured",
            extra={"error": str(exc), "tracking_uri": settings.mlflow_uri},
        )

    return result


def train_and_save(
    settings: Settings,
    dataset: pd.DataFrame | None = None,
    *,
    candidates: list[str] | None = None,
    force_save: bool = False,
) -> dict[str, Any]:
    """Train, evaluate, log, and persist a model. Returns a summary document."""
    settings.ensure_directories()
    frame = (
        dataset
        if dataset is not None
        else load_dataset(settings.dataset_path, settings.target_column)
    )
    validate_dataset(frame, settings.target_column, strict=True)

    target_names = frame.attrs.get("target_names")
    train_frame, test_frame = split_dataset(
        frame,
        settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    validate_dataset(train_frame, settings.target_column, strict=True)
    validate_dataset(test_frame, settings.target_column, strict=True)

    fitted, numeric_features, categorical_features = train_candidates(
        train_frame,
        test_frame,
        settings.target_column,
        random_state=settings.random_state,
        candidates=candidates,
    )
    if BASELINE_MODEL not in fitted:
        raise ValueError(f"The baseline candidate '{BASELINE_MODEL}' must be included")

    gate_config = GateConfig(
        min_f1=settings.min_f1,
        min_recall=settings.min_recall,
        min_roc_auc=settings.min_roc_auc,
        baseline_tolerance=settings.baseline_tolerance,
    )
    baseline_metrics = fitted[BASELINE_MODEL].metrics
    gates = {
        name: check_gates(candidate.metrics, gate_config, baseline_metrics=baseline_metrics)
        for name, candidate in fitted.items()
    }
    selected_name = _select_candidate(fitted, gates)
    selected = fitted[selected_name]
    selected_gate = gates[selected_name]

    dataset_info = {
        "name": frame.attrs.get("dataset_name", settings.dataset_path.stem),
        "path": str(settings.dataset_path),
        "rows": len(frame),
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "fingerprint": dataset_fingerprint(frame),
        "target_column": settings.target_column,
    }

    version = settings.model_version or datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    report = build_evaluation_report(
        model_name=settings.model_name,
        model_version=version,
        metrics=selected.metrics,
        gates=selected_gate,
        dataset=dataset_info,
        candidate=selected_name,
        candidates={
            name: {
                "metrics": candidate.metrics,
                "params": candidate.params,
                "gates": gates[name].to_dict(),
            }
            for name, candidate in fitted.items()
        },
    )
    evaluation_path = save_evaluation_report(report, settings.evaluation_report_path)

    if not selected_gate.passed and not force_save:
        summary = {
            "status": "rejected",
            "selected_candidate": selected_name,
            "evaluation_report": str(evaluation_path),
            "gates": selected_gate.to_dict(),
        }
        settings.training_summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        raise ModelAcceptanceError(
            f"Candidate '{selected_name}' did not pass the evaluation gates", report
        )

    mlflow_info = _log_to_mlflow(
        selected,
        settings=settings,
        x_train=train_frame[numeric_features + categorical_features],
        y_train=train_frame[settings.target_column],
        evaluation_path=evaluation_path,
        dataset_info=dataset_info,
    )
    if mlflow_info.get("registered_version"):
        version = str(mlflow_info["registered_version"])
        report["model_version"] = version
        save_evaluation_report(report, evaluation_path)

    classes = [
        value.item() if hasattr(value, "item") else value
        for value in selected.pipeline.named_steps["model"].classes_
    ]
    bundle = ModelBundle(
        model=selected.pipeline,
        model_name=settings.model_name,
        model_version=version,
        feature_specs=build_feature_specs(train_frame, numeric_features, categorical_features),
        classes=classes,
        target_names=resolve_target_names(classes, target_names),
        metrics=selected.metrics,
        metadata={
            "trained_at": datetime.now(tz=UTC).isoformat(),
            "candidate": selected_name,
            "dataset": dataset_info,
            "target_column": settings.target_column,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_statistics": feature_statistics(
                train_frame, numeric_features, categorical_features
            ),
            "hyperparameters": selected.params,
            "gates": selected_gate.to_dict(),
            "sklearn_version": sklearn.__version__,
            "mlflow": mlflow_info,
            "reference_data_path": str(settings.reference_data_path),
        },
    )

    train_frame.to_csv(settings.reference_data_path, index=False)
    artifact_path = save_model_bundle(
        bundle,
        settings.model_path,
        metadata_path=settings.model_metadata_path,
    )

    summary = {
        "status": "accepted" if selected_gate.passed else "accepted_with_force",
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "selected_candidate": selected_name,
        "metrics": selected.metrics,
        "gates": selected_gate.to_dict(),
        "candidates": {
            name: {"metrics": candidate.metrics, "gates": gates[name].to_dict()}
            for name, candidate in fitted.items()
        },
        "artifact_path": str(artifact_path),
        "metadata_path": str(settings.model_metadata_path),
        "reference_data_path": str(settings.reference_data_path),
        "evaluation_report": str(evaluation_path),
        "dataset": dataset_info,
        "mlflow": mlflow_info,
    }
    settings.training_summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Training completed",
        extra={
            "model_name": bundle.model_name,
            "model_version": bundle.model_version,
            "candidate": selected_name,
            "f1": selected.metrics.get("f1"),
            "artifact_path": str(artifact_path),
        },
    )
    return summary
