"""Evaluate a saved model artifact against a labelled dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from mlops_monitoring.config import get_settings
from mlops_monitoring.data.loading import dataset_fingerprint, load_dataset, split_dataset
from mlops_monitoring.models.loading import ModelArtifactError, ModelLoader
from mlops_monitoring.observability.logging import configure_logging
from mlops_monitoring.training.evaluate import (
    GateConfig,
    check_gates,
    evaluate_bundle,
    save_evaluation_report,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="Model artifact file or directory")
    parser.add_argument("--dataset", help="Labelled CSV dataset")
    parser.add_argument("--output", help="Evaluation report JSON path")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate on the full dataset instead of the held-out test split",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Report metrics without failing on acceptance gates",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    model_path = (
        Path(args.model_path).expanduser().resolve() if args.model_path else settings.model_path
    )
    dataset_path = (
        Path(args.dataset).expanduser().resolve() if args.dataset else settings.dataset_path
    )
    output_path = (
        Path(args.output).expanduser().resolve() if args.output else settings.evaluation_report_path
    )

    if not dataset_path.exists():
        logger.error("Dataset not found", extra={"path": str(dataset_path)})
        return 2

    loader = ModelLoader(
        model_path,
        default_model_name=settings.model_name,
        default_model_version=settings.model_version,
    )
    try:
        bundle = loader.load()
    except ModelArtifactError as exc:
        logger.error("Model artifact unavailable", extra={"error": str(exc)})
        return 2

    frame = load_dataset(dataset_path, settings.target_column)
    if args.full:
        evaluation_frame = frame
        split_name = "full"
    else:
        _, evaluation_frame = split_dataset(
            frame,
            settings.target_column,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
        split_name = "test"

    try:
        metrics = evaluate_bundle(bundle, evaluation_frame, settings.target_column)
    except ValueError as exc:
        logger.error("Evaluation failed", extra={"error": str(exc)})
        return 2

    gates = check_gates(
        metrics,
        GateConfig(
            min_f1=settings.min_f1,
            min_recall=settings.min_recall,
            min_roc_auc=settings.min_roc_auc,
            baseline_tolerance=settings.baseline_tolerance,
        ),
        baseline_metrics=bundle.metrics,
    )
    report = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "model_name": bundle.model_name,
        "model_version": bundle.model_version,
        "dataset": {
            "path": str(dataset_path),
            "split": split_name,
            "rows": len(evaluation_frame),
            "fingerprint": dataset_fingerprint(frame),
        },
        "metrics": metrics,
        "gates": gates.to_dict(),
    }
    save_evaluation_report(report, output_path)

    print(json.dumps(report, indent=2, default=str))
    if not gates.passed and not args.no_gate:
        logger.error("Evaluation gates failed", extra={"failures": gates.failures})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
