"""Unit tests for dataset validation and quality summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_monitoring.data.validation import (
    DataValidationError,
    data_quality_summary,
    validate_dataset,
)


def test_demo_dataset_is_valid(demo_frame):
    report = validate_dataset(demo_frame, strict=True)

    assert report.ok
    assert report.n_rows == len(demo_frame)
    assert report.feature_columns
    assert not report.errors


def test_missing_target_column_is_an_error(demo_frame):
    frame = demo_frame.drop(columns=["target"])

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(frame, target_column="target", strict=True)

    codes = {issue.code for issue in exc_info.value.report.errors}
    assert "missing_target" in codes
    assert "no_features" not in codes


def test_missing_values_are_warnings_when_imputable(demo_frame):
    frame = demo_frame.copy()
    frame.loc[frame.index[:10], frame.columns[0]] = np.nan

    report = validate_dataset(frame, strict=True)

    assert report.ok
    assert any(issue.code == "missing_values" for issue in report.warnings)


def test_infinite_values_are_rejected(demo_frame):
    frame = demo_frame.copy()
    frame.loc[frame.index[0], frame.columns[0]] = np.inf

    report = validate_dataset(frame, strict=False)

    assert not report.ok
    assert any(issue.code == "infinite_values" for issue in report.errors)


def test_single_class_target_is_rejected(demo_frame):
    frame = demo_frame.copy()
    frame["target"] = 1

    report = validate_dataset(frame, strict=False)

    assert any(issue.code == "single_class_target" for issue in report.errors)


def test_data_quality_summary_counts_missing_and_duplicates():
    frame = pd.DataFrame({"a": [1.0, np.nan, 1.0], "b": ["x", "y", "x"]})
    frame = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)

    summary = data_quality_summary(frame)

    assert summary["rows"] == 4
    assert summary["missing_cells"] == 1
    assert summary["duplicate_rows"] == 2
    assert summary["infinite_values"] == 0
