"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlops_monitoring import __version__
from mlops_monitoring.api.routes import router
from mlops_monitoring.config import Settings, get_settings
from mlops_monitoring.inference import PredictionService
from mlops_monitoring.models.loading import (
    FeatureValidationError,
    ModelArtifactError,
    ModelLoader,
    ModelNotLoadedError,
)
from mlops_monitoring.observability import metrics
from mlops_monitoring.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the inference application, optionally with injected settings."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    model_loader = ModelLoader(
        settings.model_path,
        default_model_name=settings.model_name,
        default_model_version=settings.model_version,
    )
    prediction_service = PredictionService(model_loader, settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at = time.monotonic()
        settings.ensure_directories()
        try:
            bundle = model_loader.load()
            metrics.MODEL_LOADED.set(1)
            metrics.set_model_info(bundle.model_name, bundle.model_version, bundle.target_names)
            logger.info(
                "application_started",
                extra={
                    "app_env": settings.app_env,
                    "model_name": bundle.model_name,
                    "model_version": bundle.model_version,
                    "model_path": str(model_loader.artifact_path),
                },
            )
        except ModelArtifactError as exc:
            metrics.MODEL_LOADED.set(0)
            logger.warning(
                "application_started_without_model",
                extra={
                    "app_env": settings.app_env,
                    "model_path": str(model_loader.artifact_path),
                    "error": str(exc),
                },
            )
        yield
        metrics.MODEL_LOADED.set(0)
        logger.info("application_stopped")

    app = FastAPI(
        title="MLOps Model Monitoring API",
        description=(
            "Inference service with validated requests, prediction logging, "
            "model metadata, and Prometheus metrics."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_loader = model_loader
    app.state.prediction_service = prediction_service
    app.state.started_at = time.monotonic()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=settings.cors_origin_list != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def observability_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response = None
        try:
            response = await call_next(request)
            return response
        except Exception:
            logger.exception(
                "request_failed",
                extra={"request_id": request_id, "path": request.url.path},
            )
            raise
        finally:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            path = route.path if route else "unmatched"
            if path != "/metrics":
                status = str(response.status_code) if response is not None else "500"
                metrics.REQUEST_COUNT.labels(method=request.method, path=path, status=status).inc()
                metrics.REQUEST_LATENCY.labels(method=request.method, path=path).observe(duration)
                logger.info(
                    "request_completed",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "duration_ms": round(duration * 1000, 3),
                    },
                )
            if response is not None:
                response.headers["X-Request-ID"] = request_id

    @app.exception_handler(FeatureValidationError)
    async def feature_validation_handler(request: Request, exc: FeatureValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "message": "Invalid prediction payload",
                    "errors": exc.errors,
                },
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": str(exc),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(ModelArtifactError)
    async def model_artifact_handler(request: Request, exc: ModelArtifactError):
        logger.error(
            "model_artifact_error",
            extra={"request_id": getattr(request.state, "request_id", None), "error": str(exc)},
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "The model is temporarily unavailable",
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    app.include_router(router)
    return app


app = create_app()
