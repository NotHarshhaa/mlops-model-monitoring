"""Loading, validating, and using persisted model artifacts."""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1


class ModelArtifactError(RuntimeError):
    """Raised when a model artifact is missing or cannot be loaded."""


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted without a loaded model."""


class FeatureValidationError(ValueError):
    """Raised when an inference payload does not match the model schema."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class FeatureSpec:
    """Expected type and metadata for a single model input feature."""

    name: str
    kind: Literal["numeric", "categorical"]
    dtype: str = "float64"
    minimum: float | None = None
    maximum: float | None = None
    categories: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": self.dtype,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.categories is not None:
            payload["categories"] = self.categories
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> FeatureSpec:
        return cls(
            name=str(payload["name"]),
            kind=payload["kind"],
            dtype=str(payload.get("dtype", "")),
            minimum=payload.get("minimum"),
            maximum=payload.get("maximum"),
            categories=payload.get("categories"),
        )


@dataclass
class ModelBundle:
    """A fitted pipeline plus everything required to serve and audit it."""

    model: Any
    model_name: str
    model_version: str
    feature_specs: list[FeatureSpec]
    classes: list[Any]
    target_names: list[str]
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [spec.name for spec in self.feature_specs]

    @property
    def numeric_features(self) -> list[str]:
        return [spec.name for spec in self.feature_specs if spec.kind == "numeric"]

    @property
    def categorical_features(self) -> list[str]:
        return [spec.name for spec in self.feature_specs if spec.kind == "categorical"]

    def to_payload(self) -> dict[str, Any]:
        return {
            "artifact_version": ARTIFACT_VERSION,
            "model": self.model,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "feature_specs": [spec.to_dict() for spec in self.feature_specs],
            "classes": self.classes,
            "target_names": self.target_names,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ModelBundle:
        if "model" not in payload:
            raise ModelArtifactError("Model artifact does not contain a fitted model")
        return cls(
            model=payload["model"],
            model_name=str(payload.get("model_name", "unknown-model")),
            model_version=str(payload.get("model_version", "unknown")),
            feature_specs=[
                FeatureSpec.from_dict(item) for item in payload.get("feature_specs", [])
            ],
            classes=list(payload.get("classes", [])),
            target_names=[str(name) for name in payload.get("target_names", [])],
            metrics=dict(payload.get("metrics", {})),
            metadata=dict(payload.get("metadata", {})),
        )

    def metadata_document(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "classes": self.classes,
            "target_names": self.target_names,
            "feature_specs": [spec.to_dict() for spec in self.feature_specs],
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


def _resolve_artifact_path(model_path: Path) -> Path:
    if model_path.suffix in {".joblib", ".pkl"}:
        return model_path
    return model_path / "model.joblib"


def save_model_bundle(
    bundle: ModelBundle,
    model_path: Path,
    metadata_path: Path | None = None,
) -> Path:
    """Serialize a model bundle and an optional human-readable metadata file."""
    artifact_path = _resolve_artifact_path(model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    joblib.dump(bundle.to_payload(), temporary_path)
    temporary_path.replace(artifact_path)

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(bundle.metadata_document(), indent=2, default=str),
            encoding="utf-8",
        )

    logger.info(
        "Model bundle saved",
        extra={
            "path": str(artifact_path),
            "model_name": bundle.model_name,
            "model_version": bundle.model_version,
        },
    )
    return artifact_path


def load_model_bundle(model_path: Path) -> ModelBundle:
    """Load a model bundle from disk.

    Only load artifacts produced by a trusted training pipeline: joblib uses
    pickle semantics and must never be pointed at untrusted files.
    """
    artifact_path = _resolve_artifact_path(model_path)
    if not artifact_path.exists():
        raise ModelArtifactError(f"Model artifact not found: {artifact_path}")

    try:
        payload = joblib.load(artifact_path)
    except Exception as exc:  # pragma: no cover - depends on corrupt artifact
        raise ModelArtifactError(f"Unable to load model artifact {artifact_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ModelArtifactError(
            f"Model artifact {artifact_path} has an unsupported format; expected a bundle"
        )
    return ModelBundle.from_payload(payload)


class ModelLoader:
    """Thread-safe, lazily loaded model artifact used by the inference service."""

    def __init__(
        self,
        model_path: Path,
        *,
        default_model_name: str = "baseline-classifier",
        default_model_version: str = "",
    ) -> None:
        self.model_path = model_path
        self.default_model_name = default_model_name
        self.default_model_version = default_model_version
        self._bundle: ModelBundle | None = None
        self._load_error: str | None = None
        self._lock = threading.Lock()

    @property
    def artifact_path(self) -> Path:
        return _resolve_artifact_path(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self._bundle is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self, *, force: bool = False) -> ModelBundle:
        with self._lock:
            if self._bundle is not None and not force:
                return self._bundle
            try:
                bundle = load_model_bundle(self.model_path)
            except ModelArtifactError as exc:
                self._load_error = str(exc)
                raise
            self._bundle = bundle
            self._load_error = None
            logger.info(
                "Model artifact loaded",
                extra={
                    "path": str(self.artifact_path),
                    "model_name": bundle.model_name,
                    "model_version": bundle.model_version,
                    "features": len(bundle.feature_names),
                },
            )
            return bundle

    @property
    def bundle(self) -> ModelBundle:
        if self._bundle is None:
            raise ModelNotLoadedError(self._load_error or "Model is not loaded")
        return self._bundle

    def validate_features(self, features: dict[str, Any]) -> list[str]:
        bundle = self.bundle
        errors: list[str] = []
        expected = set(bundle.feature_names)
        provided = set(features)

        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing:
            errors.append(f"Missing required feature(s): {', '.join(missing)}")
        if unexpected:
            errors.append(f"Unexpected feature(s): {', '.join(unexpected)}")

        for spec in bundle.feature_specs:
            if spec.name not in features:
                continue
            value = features[spec.name]
            if value is None:
                errors.append(f"Feature '{spec.name}' must not be null")
                continue
            if spec.kind == "numeric":
                if isinstance(value, bool) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    errors.append(f"Feature '{spec.name}' must be a number")
                    continue
                numeric = float(value)
                if not math.isfinite(numeric):
                    errors.append(f"Feature '{spec.name}' must be finite")
            else:
                if isinstance(value, (bool, dict, list, tuple)):
                    errors.append(f"Feature '{spec.name}' must be a scalar value")
                    continue
                if spec.categories is not None and str(value) not in spec.categories:
                    errors.append(
                        f"Feature '{spec.name}' has unsupported value '{value}'. "
                        f"Expected one of: {', '.join(spec.categories)}"
                    )
        return errors

    def prepare_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Validate records and construct a typed frame in training column order."""
        bundle = self.bundle
        if not records:
            raise FeatureValidationError(["At least one prediction record is required"])

        rows: list[dict[str, Any]] = []
        all_errors: list[str] = []
        for index, features in enumerate(records):
            if not isinstance(features, dict):
                all_errors.append(f"Record {index}: features must be an object")
                continue
            errors = self.validate_features(features)
            if errors:
                all_errors.extend(f"Record {index}: {error}" for error in errors)
                continue
            row: dict[str, Any] = {}
            for spec in bundle.feature_specs:
                value = features[spec.name]
                row[spec.name] = float(value) if spec.kind == "numeric" else str(value)
            rows.append(row)

        if all_errors:
            raise FeatureValidationError(all_errors)

        return pd.DataFrame(rows, columns=bundle.feature_names)

    def predict_frame(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        bundle = self.bundle
        try:
            predictions = bundle.model.predict(frame)
            probabilities = (
                bundle.model.predict_proba(frame)
                if hasattr(bundle.model, "predict_proba")
                else None
            )
        except Exception as exc:
            raise ModelArtifactError(f"Inference failed: {exc}") from exc
        return np.asarray(predictions), probabilities

    def model_info(self) -> dict[str, Any]:
        bundle = self.bundle
        return {
            "model_name": bundle.model_name,
            "model_version": bundle.model_version,
            "classes": bundle.classes,
            "target_names": bundle.target_names,
            "feature_specs": [spec.to_dict() for spec in bundle.feature_specs],
            "metrics": bundle.metrics,
            "metadata": bundle.metadata,
        }
