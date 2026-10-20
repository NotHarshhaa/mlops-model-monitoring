"""Pydantic request and response contracts for the inference API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from mlops_monitoring.config import Settings

FEATURE_EXAMPLE = {
    "mean_radius": 14.2,
    "mean_texture": 20.4,
    "mean_perimeter": 92.0,
    "mean_area": 650.0,
}


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
        json_schema_extra={"examples": [{"features": FEATURE_EXAMPLE}]},
    )

    features: dict[str, float | int | str] = Field(
        ...,
        min_length=1,
        description="Feature values keyed by the names used during training.",
    )


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    items: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Prediction requests to score in a single call.",
    )


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    prediction: int | float | str
    model_name: str
    model_version: str
    latency_ms: float
    prediction_id: str
    confidence: float | None = None
    probabilities: dict[str, float] | None = None


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[PredictionResponse]
    count: int
    total_latency_ms: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    app_env: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    ready: bool
    model_name: str | None = None
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_version: str
    classes: list[Any]
    target_names: list[str]
    feature_specs: list[dict[str, Any]]
    metrics: dict[str, Any]
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    detail: str | dict[str, Any]
    request_id: str | None = None


def batch_limit(settings: Settings) -> int:
    return settings.max_batch_size
