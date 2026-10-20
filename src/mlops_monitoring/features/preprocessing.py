"""Feature type detection, preprocessing, and inference schemas."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_monitoring.models.loading import FeatureSpec

MAX_CATEGORIES = 100


def split_feature_types(
    frame: pd.DataFrame,
    target_column: str = "target",
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature names (target excluded)."""
    numeric: list[str] = []
    categorical: list[str] = []
    for column in frame.columns:
        if column == target_column:
            continue
        series = frame[column]
        if pd_types.is_bool_dtype(series) or pd_types.is_object_dtype(series):
            categorical.append(str(column))
        elif pd_types.is_numeric_dtype(series):
            numeric.append(str(column))
        else:
            categorical.append(str(column))
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build imputation, scaling, and one-hot encoding steps."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("At least one feature column is required")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model_pipeline(
    estimator: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """Combine preprocessing and an estimator into one reproducible pipeline."""
    return Pipeline(
        steps=[
            (
                "preprocessing",
                build_preprocessor(numeric_features, categorical_features),
            ),
            ("model", estimator),
        ]
    )


def build_feature_specs(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[FeatureSpec]:
    """Describe the expected inference payload for every training feature."""
    specs: list[FeatureSpec] = []

    for feature in numeric_features:
        series = pd.to_numeric(frame[feature], errors="coerce").dropna()
        minimum = float(series.min()) if not series.empty else None
        maximum = float(series.max()) if not series.empty else None
        specs.append(
            FeatureSpec(
                name=feature,
                kind="numeric",
                dtype=str(frame[feature].dtype),
                minimum=minimum,
                maximum=maximum,
            )
        )

    for feature in categorical_features:
        values = frame[feature].dropna().astype(str).unique().tolist()
        categories = sorted(values)[:MAX_CATEGORIES] if len(values) <= MAX_CATEGORIES else None
        specs.append(
            FeatureSpec(
                name=feature,
                kind="categorical",
                dtype=str(frame[feature].dtype),
                categories=categories,
            )
        )

    return specs


def numeric_feature_names(
    model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Return transformed feature names when the preprocessing step supports it."""
    preprocessor = model.named_steps.get("preprocessing")
    if preprocessor is None:
        return list(numeric_features)
    try:
        return [str(name) for name in preprocessor.get_feature_names_out()]
    except Exception:  # pragma: no cover - defensive for exotic transformers
        return list(numeric_features)


def feature_statistics(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, dict[str, Any]]:
    """Reference statistics stored with the model for observability."""
    stats: dict[str, dict[str, Any]] = {}
    for feature in numeric_features:
        series = pd.to_numeric(frame[feature], errors="coerce").dropna()
        stats[feature] = {
            "kind": "numeric",
            "count": int(series.size),
            "mean": float(series.mean()) if not series.empty else None,
            "std": float(series.std(ddof=0)) if series.size > 1 else 0.0,
            "min": float(series.min()) if not series.empty else None,
            "max": float(series.max()) if not series.empty else None,
            "missing": int(frame[feature].isna().sum()),
        }
    for feature in categorical_features:
        counts = frame[feature].astype(str).value_counts(dropna=False)
        stats[feature] = {
            "kind": "categorical",
            "count": int(counts.sum()),
            "top_values": {str(key): int(value) for key, value in counts.head(10).items()},
            "missing": int(frame[feature].isna().sum()),
        }
    return stats


def make_inference_example(
    frame: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable example payload from one reference row."""
    features = numeric_features + categorical_features
    if frame.empty or not features:
        return {}
    row = frame.iloc[0]
    example: dict[str, Any] = {}
    for feature in features:
        value = row[feature]
        if isinstance(value, np.generic):
            value = value.item()
        example[feature] = value
    return example
