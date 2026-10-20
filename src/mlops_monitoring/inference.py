"""Inference orchestration: validation, prediction, logging, and metrics."""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from mlops_monitoring.config import Settings
from mlops_monitoring.models.loading import (
    FeatureValidationError,
    ModelArtifactError,
    ModelLoader,
    ModelNotLoadedError,
)
from mlops_monitoring.observability import metrics
from mlops_monitoring.observability.prediction_log import append_prediction

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    prediction_id: str
    prediction: Any
    confidence: float | None
    probabilities: dict[str, float] | None
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prediction": self.prediction,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round(self.latency_ms, 3),
            "prediction_id": self.prediction_id,
        }
        if self.confidence is not None:
            payload["confidence"] = round(self.confidence, 6)
        if self.probabilities is not None:
            payload["probabilities"] = self.probabilities
        return payload


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


class PredictionService:
    """Coordinates model inference with request logging and Prometheus metrics."""

    def __init__(self, loader: ModelLoader, settings: Settings) -> None:
        self.loader = loader
        self.settings = settings

    def _class_names(self) -> list[str]:
        bundle = self.loader.bundle
        if len(bundle.target_names) == len(bundle.classes):
            return bundle.target_names
        return [str(value) for value in bundle.classes]

    def predict(
        self,
        features: dict[str, Any],
        *,
        request_id: str | None = None,
    ) -> PredictionResult:
        if not self.loader.is_loaded:
            metrics.record_prediction_error("model_unavailable")
            raise ModelNotLoadedError(self.loader.load_error or "Model is not loaded")

        request_id = request_id or uuid.uuid4().hex
        prediction_id = uuid.uuid4().hex
        started = time.perf_counter()

        try:
            frame = self.loader.prepare_frame([features])
        except FeatureValidationError:
            metrics.record_prediction_error("validation")
            raise
        except ModelNotLoadedError:
            metrics.record_prediction_error("model_unavailable")
            raise

        try:
            predictions, probabilities = self.loader.predict_frame(frame)
        except ModelArtifactError:
            metrics.record_prediction_error("inference")
            raise

        bundle = self.loader.bundle
        class_names = self._class_names()
        probability_map: dict[str, float] | None = None
        log_probability_map: dict[str, float] | None = None
        confidence: float | None = None
        if probabilities is not None and len(probabilities) > 0:
            values = [float(value) for value in np.asarray(probabilities[0]).ravel()]
            probability_map = {
                name: round(value, 8) for name, value in zip(class_names, values, strict=False)
            }
            log_probability_map = {
                str(_json_scalar(class_value)): round(value, 8)
                for class_value, value in zip(bundle.classes, values, strict=False)
            }
            confidence = max(values) if values else None

        prediction = _json_scalar(predictions[0])
        latency_seconds = time.perf_counter() - started
        latency_ms = latency_seconds * 1000.0

        metrics.PREDICTION_COUNT.labels(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            prediction=str(prediction),
        ).inc()
        metrics.PREDICTION_LATENCY.labels(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
        ).observe(latency_seconds)
        if confidence is not None:
            metrics.PREDICTION_CONFIDENCE.labels(
                model_name=bundle.model_name,
                model_version=bundle.model_version,
            ).observe(confidence)

        if self.settings.prediction_log_enabled:
            record = {
                "prediction_id": prediction_id,
                "request_id": request_id,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "features": features,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": log_probability_map,
                "latency_ms": round(latency_ms, 3),
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
            }
            try:
                append_prediction(self.settings.prediction_log_path, record)
            except OSError as exc:  # pragma: no cover - disk failure path
                logger.warning("Prediction could not be logged", extra={"error": str(exc)})

        logger.info(
            "prediction_completed",
            extra={
                "request_id": request_id,
                "prediction_id": prediction_id,
                "prediction": prediction,
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
                "latency_ms": round(latency_ms, 3),
            },
        )

        return PredictionResult(
            prediction_id=prediction_id,
            prediction=prediction,
            confidence=confidence,
            probabilities=probability_map,
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            latency_ms=latency_ms,
        )

    def predict_batch(
        self,
        records: list[dict[str, Any]],
        *,
        request_id: str | None = None,
    ) -> list[PredictionResult]:
        return [self.predict(features, request_id=request_id) for features in records]
