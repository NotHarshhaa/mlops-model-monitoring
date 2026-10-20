"""Unit tests for preprocessing pipelines and feature metadata."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlops_monitoring.features.preprocessing import (
    build_feature_specs,
    build_model_pipeline,
    feature_statistics,
    split_feature_types,
)


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "numeric": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "category": ["a", "b", "a", None, "b", "a"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )


def test_split_feature_types():
    numeric, categorical = split_feature_types(_mixed_frame())

    assert numeric == ["numeric"]
    assert categorical == ["category"]


def test_pipeline_handles_missing_and_unknown_categories():
    frame = _mixed_frame()
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000), ["numeric"], ["category"])
    pipeline.fit(frame[["numeric", "category"]], frame["target"])

    predictions = pipeline.predict(
        pd.DataFrame({"numeric": [3.0, np.nan], "category": ["unseen", "a"]})
    )

    assert len(predictions) == 2
    assert all(value in {0, 1} for value in predictions)


def test_feature_specs_describe_numeric_and_categorical_inputs():
    frame = _mixed_frame()
    specs = build_feature_specs(frame, ["numeric"], ["category"])
    by_name = {spec.name: spec for spec in specs}

    assert by_name["numeric"].kind == "numeric"
    assert by_name["numeric"].minimum == 1.0
    assert by_name["numeric"].maximum == 6.0
    assert by_name["category"].categories == ["a", "b"]


def test_feature_statistics_include_summary_values():
    frame = _mixed_frame()
    stats = feature_statistics(frame, ["numeric"], ["category"])

    assert stats["numeric"]["mean"] == 3.6
    assert stats["numeric"]["missing"] == 1
    assert stats["category"]["top_values"]["a"] == 3
    assert stats["category"]["missing"] == 1
