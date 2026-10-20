"""Train a model, evaluate it, and persist the deployment artifact."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from mlops_monitoring.config import get_settings
from mlops_monitoring.data.loading import (
    load_breast_cancer_dataset,
    load_dataset,
    save_dataset,
)
from mlops_monitoring.observability.logging import configure_logging
from mlops_monitoring.training.train import ModelAcceptanceError, train_and_save

logger = logging.getLogger(__name__)


def _resolve_path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to a labelled CSV dataset")
    parser.add_argument("--model-name", help="Model name recorded in metadata and MLflow")
    parser.add_argument("--model-version", help="Explicit model version")
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=None,
        help="Candidate models to train (default: logistic-regression random-forest)",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the selected model in the MLflow model registry",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Save the best candidate even when evaluation gates fail",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    dataset_path = _resolve_path(args.dataset, settings.dataset_path)
    if not dataset_path.exists():
        if args.dataset:
            logger.error("Dataset not found", extra={"path": str(dataset_path)})
            return 2
        logger.warning(
            "No dataset configured; materializing the bundled breast cancer demo dataset",
            extra={"path": str(dataset_path)},
        )
        save_dataset(load_breast_cancer_dataset(), dataset_path)

    updates: dict[str, object] = {"dataset_path": dataset_path}
    if args.model_name:
        updates["model_name"] = args.model_name
    if args.model_version:
        updates["model_version"] = args.model_version
    if args.no_mlflow:
        updates["mlflow_enabled"] = False
    if args.register:
        updates["mlflow_register_model"] = True
    settings = settings.model_copy(update=updates)

    try:
        dataset = load_dataset(dataset_path, settings.target_column)
        summary = train_and_save(
            settings,
            dataset,
            candidates=args.candidates,
            force_save=args.force,
        )
    except ModelAcceptanceError as exc:
        print(json.dumps(exc.report.get("gates", {}), indent=2, default=str))
        logger.error("Model rejected by evaluation gates", extra={"error": str(exc)})
        return 1
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Training failed", extra={"error": str(exc)})
        return 2

    print(json.dumps(summary, indent=2, default=str, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
