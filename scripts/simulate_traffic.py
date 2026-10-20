"""Generate synthetic inference traffic, optionally with feature drift.

This is a demonstration utility: it replays held-out rows through the same
prediction service used by the API so the prediction log, metrics, and
monitoring workflow can be exercised locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_monitoring.config import get_settings
from mlops_monitoring.data.loading import load_dataset, split_dataset
from mlops_monitoring.inference import PredictionService
from mlops_monitoring.models.loading import ModelArtifactError, ModelLoader
from mlops_monitoring.observability.logging import configure_logging

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300, help="Number of requests to generate")
    parser.add_argument(
        "--drift-shift",
        type=float,
        default=0.0,
        help="Shift selected numeric features by this many reference standard deviations",
    )
    parser.add_argument(
        "--drift-fraction",
        type=float,
        default=0.5,
        help="Share of numeric features to shift when drift-shift is non-zero",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", help="Labelled CSV dataset")
    parser.add_argument("--model-path", help="Model artifact file or directory")
    parser.add_argument("--predictions", help="Prediction log path")
    parser.add_argument("--labels", help="Ground-truth label log path")
    parser.add_argument("--verbose", action="store_true", help="Log every generated prediction")
    return parser.parse_args(argv)


def _append_label(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(
        settings.log_level if args.verbose else "WARNING",
        json_logs=settings.log_json,
    )

    dataset_path = (
        Path(args.dataset).expanduser().resolve() if args.dataset else settings.dataset_path
    )
    model_path = (
        Path(args.model_path).expanduser().resolve() if args.model_path else settings.model_path
    )
    predictions_path = (
        Path(args.predictions).expanduser().resolve()
        if args.predictions
        else settings.prediction_log_path
    )
    labels_path = Path(args.labels).expanduser().resolve() if args.labels else settings.labels_path

    if not dataset_path.exists():
        logger.error("Dataset not found", extra={"path": str(dataset_path)})
        return 2

    frame = load_dataset(dataset_path, settings.target_column)
    train_frame, test_frame = split_dataset(
        frame,
        settings.target_column,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )

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

    if args.rows < 1:
        logger.error("--rows must be at least 1")
        return 2
    if not 0.0 <= args.drift_fraction <= 1.0:
        logger.error("--drift-fraction must be between 0 and 1")
        return 2

    numeric_features = bundle.numeric_features
    shifted_features: list[str] = []
    if args.drift_shift != 0.0 and numeric_features:
        count = max(1, round(len(numeric_features) * args.drift_fraction))
        rng = np.random.default_rng(args.seed)
        shifted_features = sorted(rng.choice(numeric_features, size=count, replace=False).tolist())
        logger.warning(
            "Applying synthetic feature drift",
            extra={"shift_std": args.drift_shift, "features": shifted_features},
        )

    reference_std = {
        feature: float(pd.to_numeric(train_frame[feature], errors="coerce").std(ddof=0) or 0.0)
        for feature in numeric_features
    }

    sampled = test_frame.sample(
        n=args.rows,
        replace=args.rows > len(test_frame),
        random_state=args.seed,
    )

    service = PredictionService(
        loader,
        settings.model_copy(
            update={
                "prediction_log_path": predictions_path,
                "prediction_log_enabled": True,
            }
        ),
    )

    logged = 0
    for _, row in sampled.iterrows():
        features: dict[str, object] = {}
        for spec in bundle.feature_specs:
            value = row[spec.name]
            if spec.kind == "numeric":
                numeric_value = float(value)
                if spec.name in shifted_features:
                    numeric_value += args.drift_shift * reference_std.get(spec.name, 0.0)
                features[spec.name] = numeric_value
            else:
                features[spec.name] = str(value)

        result = service.predict(features)
        label = row[settings.target_column]
        if hasattr(label, "item"):
            label = label.item()
        if isinstance(label, float) and label.is_integer():
            label = int(label)
        _append_label(
            labels_path,
            {
                "prediction_id": result.prediction_id,
                "label": label,
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
        logged += 1

    print(
        json.dumps(
            {
                "requests": logged,
                "drift_shift_std": args.drift_shift,
                "shifted_features": shifted_features,
                "predictions_path": str(predictions_path),
                "labels_path": str(labels_path),
                "model_name": bundle.model_name,
                "model_version": bundle.model_version,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
