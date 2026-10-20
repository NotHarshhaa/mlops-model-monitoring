"""Dataset validation and data-quality summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api import types as pd_types

Severity = Literal["error", "warning"]


class DataValidationError(ValueError):
    """Raised when a dataset fails blocking validation checks."""

    def __init__(self, report: DataValidationReport) -> None:
        self.report = report
        errors = "; ".join(issue.message for issue in report.errors)
        super().__init__(f"Dataset validation failed: {errors}")


@dataclass(frozen=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    column: str | None = None
    count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.column is not None:
            payload["column"] = self.column
        if self.count is not None:
            payload["count"] = self.count
        return payload


@dataclass
class DataValidationReport:
    n_rows: int
    n_columns: int
    feature_columns: list[str] = field(default_factory=list)
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise DataValidationError(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "feature_columns": self.feature_columns,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def validate_dataset(
    frame: pd.DataFrame,
    target_column: str = "target",
    *,
    max_missing_fraction: float = 0.5,
    require_target: bool = True,
    strict: bool = True,
) -> DataValidationReport:
    """Validate structure, target integrity, missing values, and numeric ranges."""
    issues: list[ValidationIssue] = []

    if frame.empty:
        issues.append(ValidationIssue("error", "empty_dataset", "The dataset contains no rows."))

    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        issues.append(
            ValidationIssue(
                "error",
                "duplicate_columns",
                f"Duplicate column names: {', '.join(map(str, duplicates))}",
            )
        )

    feature_columns = [str(column) for column in frame.columns if column != target_column]

    if require_target and target_column not in frame.columns:
        issues.append(
            ValidationIssue(
                "error",
                "missing_target",
                f"Required target column '{target_column}' is missing.",
                column=target_column,
            )
        )

    if not feature_columns:
        issues.append(
            ValidationIssue(
                "error",
                "no_features",
                "The dataset does not contain any feature columns.",
            )
        )

    if target_column in frame.columns:
        target = frame[target_column]
        missing_target = int(target.isna().sum())
        if missing_target:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_target_values",
                    f"Target column contains {missing_target} missing value(s).",
                    column=target_column,
                    count=missing_target,
                )
            )
        unique_target = target.dropna().unique()
        if len(unique_target) < 2:
            issues.append(
                ValidationIssue(
                    "error",
                    "single_class_target",
                    "The target column must contain at least two classes.",
                    column=target_column,
                )
            )
        elif len(unique_target) > 50:
            issues.append(
                ValidationIssue(
                    "warning",
                    "high_cardinality_target",
                    f"The target column has {len(unique_target)} classes; "
                    "this project is designed for classification tasks.",
                    column=target_column,
                )
            )

    for column in feature_columns:
        series = frame[column]
        missing = int(series.isna().sum())
        missing_fraction = missing / max(len(series), 1)
        if missing_fraction > max_missing_fraction:
            issues.append(
                ValidationIssue(
                    "error",
                    "excessive_missing_values",
                    f"{missing_fraction:.1%} of values are missing "
                    f"(limit {max_missing_fraction:.0%}).",
                    column=column,
                    count=missing,
                )
            )
        elif missing:
            issues.append(
                ValidationIssue(
                    "warning",
                    "missing_values",
                    f"{missing} missing value(s) will be imputed.",
                    column=column,
                    count=missing,
                )
            )

        if len(series.dropna()) <= 1:
            issues.append(
                ValidationIssue(
                    "warning",
                    "constant_or_empty_feature",
                    "Feature has fewer than two non-null values.",
                    column=column,
                )
            )
        elif series.nunique(dropna=True) == 1:
            issues.append(
                ValidationIssue(
                    "warning",
                    "constant_feature",
                    "Feature is constant and will not contribute information.",
                    column=column,
                )
            )

        if pd_types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            infinite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
            if infinite:
                issues.append(
                    ValidationIssue(
                        "error",
                        "infinite_values",
                        f"Feature contains {infinite} infinite value(s).",
                        column=column,
                        count=infinite,
                    )
                )

    report = DataValidationReport(
        n_rows=len(frame),
        n_columns=int(frame.shape[1]),
        feature_columns=feature_columns,
        issues=issues,
    )
    if strict:
        report.raise_if_invalid()
    return report


def data_quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compact data-quality snapshot used in monitoring reports."""
    total_cells = int(frame.shape[0] * frame.shape[1])
    missing_cells = int(frame.isna().sum().sum())
    numeric_frame = frame.select_dtypes(include=[np.number])
    infinite_values = (
        int(np.isinf(numeric_frame.to_numpy(dtype=float)).sum()) if not numeric_frame.empty else 0
    )

    per_column = []
    for column in map(str, frame.columns):
        series = frame[column]
        per_column.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "missing_cells": missing_cells,
        "missing_fraction": round(missing_cells / total_cells, 6) if total_cells else 0.0,
        "duplicate_rows": int(frame.duplicated().sum()),
        "infinite_values": infinite_values,
        "columns_summary": per_column,
    }
