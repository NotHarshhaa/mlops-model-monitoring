"""Central configuration loaded from environment variables and ``.env``."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_project_root() -> Path:
    """Locate the project base directory.

    Prefer ``MLOPS_BASE_DIR`` for explicit deployments, then the nearest parent
    containing this project's ``pyproject.toml``. Fall back to the current
    working directory so an installed package still resolves relative paths
    predictably.
    """
    configured = os.getenv("MLOPS_BASE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "mlops_monitoring"
        ).exists():
            return candidate
    return current


PROJECT_ROOT = _default_project_root()

_PATH_FIELDS = (
    "dataset_path",
    "reference_data_path",
    "model_path",
    "prediction_log_path",
    "labels_path",
    "drift_report_path",
    "monitoring_report_path",
    "monitoring_metrics_path",
    "evaluation_report_path",
    "training_summary_path",
)


def _resolve_path(value: Path) -> Path:
    value = value.expanduser()
    if value.is_absolute():
        return value
    return (PROJECT_ROOT / value).resolve()


class Settings(BaseSettings):
    """Runtime settings.

    Every field can be overridden with an environment variable of the same
    name (case-insensitive) or through the project ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        protected_namespaces=(),
    )

    # Application
    app_env: str = "development"
    log_level: str = "INFO"
    log_json: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"

    # Dataset
    dataset_path: Path = Path("data/breast_cancer.csv")
    target_column: str = "target"
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_state: int = 42
    reference_data_path: Path = Path("data/reference.csv")

    # Model
    model_name: str = "baseline-classifier"
    model_version: str = ""
    model_path: Path = Path("artifacts/model")

    # Inference
    prediction_log_path: Path = Path("artifacts/predictions.jsonl")
    prediction_log_enabled: bool = True
    labels_path: Path = Path("data/labels.jsonl")
    max_batch_size: int = Field(default=1000, gt=0, le=10000)

    # Monitoring
    monitoring_enabled: bool = True
    monitoring_use_evidently: bool = True
    drift_report_path: Path = Path("reports/drift-report.html")
    monitoring_report_path: Path = Path("reports/monitoring-report.json")
    monitoring_metrics_path: Path = Path("reports/monitoring.prom")
    drift_threshold: float = Field(default=0.2, ge=0.0)
    drift_share_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    performance_drop_threshold: float = Field(default=0.1, ge=0.0)

    # Evaluation gates
    min_f1: float = Field(default=0.80, ge=0.0, le=1.0)
    min_recall: float = Field(default=0.75, ge=0.0, le=1.0)
    min_roc_auc: float = Field(default=0.85, ge=0.0, le=1.0)
    baseline_tolerance: float = Field(default=0.01, ge=0.0, le=1.0)

    # MLflow
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str = ""
    mlflow_experiment_name: str = "mlops-model-monitoring"
    mlflow_register_model: bool = False

    # Report locations
    evaluation_report_path: Path = Path("reports/evaluation.json")
    training_summary_path: Path = Path("reports/training-summary.json")

    @model_validator(mode="after")
    def _resolve_paths(self) -> Settings:
        for name in _PATH_FIELDS:
            setattr(self, name, _resolve_path(getattr(self, name)))
        return self

    @property
    def model_artifact_path(self) -> Path:
        """File that stores the serialized model bundle."""
        if self.model_path.suffix in {".joblib", ".pkl"}:
            return self.model_path
        return self.model_path / "model.joblib"

    @property
    def model_metadata_path(self) -> Path:
        return self.model_artifact_path.parent / "metadata.json"

    @property
    def mlflow_uri(self) -> str:
        if self.mlflow_tracking_uri:
            return self.mlflow_tracking_uri
        return (PROJECT_ROOT / "mlruns").as_uri()

    @property
    def cors_origin_list(self) -> list[str]:
        origins = [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return origins or ["*"]

    def ensure_directories(self) -> None:
        """Create local directories required by the application."""
        directories = {
            self.model_artifact_path.parent,
            self.prediction_log_path.parent,
            self.reference_data_path.parent,
            self.labels_path.parent,
            self.drift_report_path.parent,
            self.monitoring_report_path.parent,
            self.monitoring_metrics_path.parent,
            self.evaluation_report_path.parent,
            self.training_summary_path.parent,
        }
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
