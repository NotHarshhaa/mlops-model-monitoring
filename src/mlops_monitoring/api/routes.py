"""HTTP routes for health checks, inference, and model metadata."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from mlops_monitoring.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    ReadinessResponse,
)
from mlops_monitoring.inference import PredictionService
from mlops_monitoring.models.loading import ModelLoader
from mlops_monitoring.observability import metrics

router = APIRouter()


def _service(request: Request) -> PredictionService:
    return request.app.state.prediction_service


def _loader(request: Request) -> ModelLoader:
    return request.app.state.model_loader


def _uptime(request: Request) -> float:
    return max(0.0, time.monotonic() - request.app.state.started_at)


@router.get("/health", response_model=HealthResponse, tags=["operations"])
def health(request: Request) -> HealthResponse:
    """Liveness probe. Succeeds even when no model artifact is available."""
    loader = _loader(request)
    bundle = loader.bundle if loader.is_loaded else None
    return HealthResponse(
        status="ok",
        app_env=request.app.state.settings.app_env,
        model_loaded=loader.is_loaded,
        model_name=bundle.model_name if bundle else None,
        model_version=bundle.model_version if bundle else None,
        uptime_seconds=round(_uptime(request), 3),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
    tags=["operations"],
)
def ready(request: Request):
    """Readiness probe. Fails until a usable model artifact is loaded."""
    loader = _loader(request)
    if not loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "ready": False,
                "detail": loader.load_error or "Model artifact is not loaded",
            },
        )
    bundle = loader.bundle
    return ReadinessResponse(
        status="ready",
        ready=True,
        model_name=bundle.model_name,
        model_version=bundle.model_version,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"description": "Invalid feature payload"},
        503: {"description": "Model unavailable"},
    },
    tags=["inference"],
)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """Score one feature payload and log the prediction for monitoring."""
    result = _service(request).predict(
        payload.features,
        request_id=getattr(request.state, "request_id", None),
    )
    return PredictionResponse(**result.to_dict())


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        422: {"description": "Invalid feature payload"},
        503: {"description": "Model unavailable"},
    },
    tags=["inference"],
)
def predict_batch(payload: BatchPredictionRequest, request: Request) -> BatchPredictionResponse:
    """Score a bounded list of feature payloads."""
    settings = request.app.state.settings
    if len(payload.items) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Batch contains {len(payload.items)} items; "
                f"the maximum is {settings.max_batch_size}"
            ),
        )

    results = _service(request).predict_batch(
        [item.features for item in payload.items],
        request_id=getattr(request.state, "request_id", None),
    )
    first = results[0]
    return BatchPredictionResponse(
        predictions=[PredictionResponse(**result.to_dict()) for result in results],
        count=len(results),
        total_latency_ms=round(sum(result.latency_ms for result in results), 3),
        model_name=first.model_name,
        model_version=first.model_version,
    )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    responses={503: {"description": "Model unavailable"}},
    tags=["model"],
)
def model_info(request: Request):
    """Return model metadata, metrics, and the expected feature schema."""
    loader = _loader(request)
    if not loader.is_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "detail": loader.load_error or "Model artifact is not loaded",
                "status": "not_ready",
            },
        )
    return ModelInfoResponse(**loader.model_info())


@router.get("/metrics", include_in_schema=False)
def prometheus_metrics() -> Response:
    """Prometheus exposition endpoint."""
    payload, content_type = metrics.render_metrics()
    return Response(content=payload, media_type=content_type)
