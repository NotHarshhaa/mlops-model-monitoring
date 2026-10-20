"""Data drift and prediction drift detection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas.api import types as pd_types
from scipy import stats

from mlops_monitoring.models.loading import FeatureSpec

MIN_DRIFT_SAMPLES = 20
PSI_EPSILON = 1e-6


def _finite(values: Iterable[Any]) -> np.ndarray:
    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return array[np.isfinite(array)]


def numeric_psi(
    reference: Iterable[Any],
    current: Iterable[Any],
    *,
    buckets: int = 10,
) -> float | None:
    """Population Stability Index for a numeric feature.

    Bins are derived from reference quantiles so the metric is comparable
    across monitoring runs. Returns ``None`` when there is insufficient data.
    """
    expected = _finite(reference)
    actual = _finite(current)
    if expected.size < 2 or actual.size == 0:
        return None

    edges = np.unique(np.quantile(expected, np.linspace(0.0, 1.0, buckets + 1)))
    if edges.size < 2:
        constant = float(edges[0])
        return 0.0 if np.allclose(actual, constant) else 1.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_pct = np.clip(expected_counts / expected.size, PSI_EPSILON, None)
    actual_pct = np.clip(actual_counts / actual.size, PSI_EPSILON, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def categorical_psi(reference: Iterable[Any], current: Iterable[Any]) -> float | None:
    """Population Stability Index for a categorical feature."""
    expected = pd.Series(list(reference)).dropna().astype(str)
    actual = pd.Series(list(current)).dropna().astype(str)
    if expected.empty or actual.empty:
        return None

    categories = sorted(set(expected.unique()) | set(actual.unique()))
    expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)

    expected_values = np.clip(expected_pct.to_numpy(dtype=float), PSI_EPSILON, None)
    actual_values = np.clip(actual_pct.to_numpy(dtype=float), PSI_EPSILON, None)
    return float(
        np.sum((actual_values - expected_values) * np.log(actual_values / expected_values))
    )


def _numeric_summary(values: Iterable[Any], missing: int) -> dict[str, Any]:
    array = _finite(values)
    if array.size == 0:
        return {"count": 0, "missing": missing}
    return {
        "count": int(array.size),
        "missing": missing,
        "mean": round(float(array.mean()), 6),
        "std": round(float(array.std(ddof=0)), 6),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "median": round(float(np.median(array)), 6),
        "p25": round(float(np.quantile(array, 0.25)), 6),
        "p75": round(float(np.quantile(array, 0.75)), 6),
    }


def _categorical_summary(values: Iterable[Any], missing: int) -> dict[str, Any]:
    series = pd.Series(list(values)).dropna().astype(str)
    counts = series.value_counts()
    return {
        "count": int(counts.sum()),
        "missing": missing,
        "top_values": {str(key): int(value) for key, value in counts.head(10).items()},
    }


@dataclass
class FeatureDriftResult:
    feature: str
    kind: str
    method: str
    psi: float | None
    statistic: float | None
    p_value: float | None
    drifted: bool
    reference: dict[str, Any] = field(default_factory=dict)
    current: dict[str, Any] = field(default_factory=dict)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature": self.feature,
            "kind": self.kind,
            "method": self.method,
            "psi": round(self.psi, 6) if self.psi is not None else None,
            "statistic": round(self.statistic, 6) if self.statistic is not None else None,
            "p_value": round(self.p_value, 6) if self.p_value is not None else None,
            "drifted": self.drifted,
            "reference": self.reference,
            "current": self.current,
        }
        if self.note:
            payload["note"] = self.note
        return payload


def detect_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    *,
    feature: str,
    kind: str,
    psi_threshold: float = 0.2,
    sufficient_data: bool = True,
) -> FeatureDriftResult:
    """Compare one feature using PSI plus a supporting statistical test."""
    if kind == "categorical":
        psi = categorical_psi(reference, current)
        expected = reference.dropna().astype(str)
        actual = current.dropna().astype(str)
        statistic: float | None = None
        p_value: float | None = None
        categories = sorted(set(expected.unique()) | set(actual.unique()))
        if categories and not expected.empty and not actual.empty:
            table = np.array(
                [
                    [int((expected == category).sum()) for category in categories],
                    [int((actual == category).sum()) for category in categories],
                ]
            )
            try:
                statistic_value, p_value_value, _, _ = stats.chi2_contingency(table)
                statistic = float(statistic_value)
                p_value = float(p_value_value)
            except ValueError:  # pragma: no cover - degenerate contingency table
                statistic = None
                p_value = None
        result = FeatureDriftResult(
            feature=feature,
            kind=kind,
            method="psi+chi_square",
            psi=psi,
            statistic=statistic,
            p_value=p_value,
            drifted=False,
            reference=_categorical_summary(reference, int(reference.isna().sum())),
            current=_categorical_summary(current, int(current.isna().sum())),
        )
    else:
        psi = numeric_psi(reference, current)
        expected = _finite(reference)
        actual = _finite(current)
        statistic = None
        p_value = None
        if expected.size > 1 and actual.size > 1:
            test = stats.ks_2samp(expected, actual)
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        result = FeatureDriftResult(
            feature=feature,
            kind="numeric",
            method="psi+ks",
            psi=psi,
            statistic=statistic,
            p_value=p_value,
            drifted=False,
            reference=_numeric_summary(reference, int(reference.isna().sum())),
            current=_numeric_summary(current, int(current.isna().sum())),
        )

    if not sufficient_data:
        result.note = "Insufficient current-window samples; drift flag suppressed."
    elif result.psi is not None:
        result.drifted = result.psi >= psi_threshold
        result.note = f"PSI threshold: {psi_threshold}"
    elif result.p_value is not None:
        result.drifted = result.p_value < 0.05
        result.note = "PSI unavailable; significance test used."
    return result


def _resolve_feature_kinds(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_specs: list[FeatureSpec] | None,
    numeric_features: list[str] | None,
    categorical_features: list[str] | None,
) -> tuple[list[tuple[str, str]], list[str]]:
    if feature_specs:
        candidates = [(spec.name, spec.kind) for spec in feature_specs]
    elif numeric_features is not None or categorical_features is not None:
        candidates = [(name, "numeric") for name in numeric_features or []]
        candidates += [(name, "categorical") for name in categorical_features or []]
    else:
        candidates = []
        for column in reference.columns:
            kind = "numeric" if pd_types.is_numeric_dtype(reference[column]) else "categorical"
            candidates.append((str(column), kind))

    common = set(reference.columns) & set(current.columns)
    resolved = [(name, kind) for name, kind in candidates if name in common]
    missing = [name for name, _ in candidates if name not in common]
    return resolved, missing


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    feature_specs: list[FeatureSpec] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    psi_threshold: float = 0.2,
    share_threshold: float = 0.3,
    min_samples: int = MIN_DRIFT_SAMPLES,
) -> dict[str, Any]:
    """Compare feature distributions and summarise dataset-level drift."""
    if current.empty:
        raise ValueError("Current data is empty; cannot calculate drift")
    if reference.empty:
        raise ValueError("Reference data is empty; cannot calculate drift")

    resolved, missing = _resolve_feature_kinds(
        reference,
        current,
        feature_specs=feature_specs,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    if not resolved:
        raise ValueError("No common features found between reference and current data")

    sufficient_data = len(current) >= min_samples
    results = [
        detect_feature_drift(
            reference[name],
            current[name],
            feature=name,
            kind=kind,
            psi_threshold=psi_threshold,
            sufficient_data=sufficient_data,
        )
        for name, kind in resolved
    ]
    drifted = [result.feature for result in results if result.drifted]
    share = len(drifted) / len(results)

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "min_samples": min_samples,
        "sufficient_data": sufficient_data,
        "psi_threshold": psi_threshold,
        "share_threshold": share_threshold,
        "evaluated_features": len(results),
        "drifted_features": drifted,
        "drift_share": round(share, 6),
        "dataset_drifted": bool(sufficient_data and share >= share_threshold),
        "missing_features": missing,
        "features": [result.to_dict() for result in results],
    }


def detect_prediction_drift(
    reference_predictions: Iterable[Any],
    current_predictions: Iterable[Any],
    *,
    threshold: float = 0.2,
    min_samples: int = MIN_DRIFT_SAMPLES,
) -> dict[str, Any]:
    """Compare the distribution of predicted classes."""
    expected = pd.Series(list(reference_predictions)).dropna().astype(str)
    actual = pd.Series(list(current_predictions)).dropna().astype(str)
    if expected.empty or actual.empty:
        return {
            "status": "insufficient_data",
            "sufficient_data": False,
            "threshold": threshold,
            "psi": None,
            "drifted": False,
        }

    categories = sorted(set(expected.unique()) | set(actual.unique()))
    expected_counts = expected.value_counts().reindex(categories, fill_value=0)
    actual_counts = actual.value_counts().reindex(categories, fill_value=0)
    expected_share = expected_counts / max(expected_counts.sum(), 1)
    actual_share = actual_counts / max(actual_counts.sum(), 1)
    psi = categorical_psi(expected, actual)
    sufficient = len(actual) >= min_samples

    return {
        "status": "ok",
        "sufficient_data": sufficient,
        "reference_rows": int(expected.size),
        "current_rows": int(actual.size),
        "threshold": threshold,
        "psi": round(psi, 6) if psi is not None else None,
        "drifted": bool(sufficient and psi is not None and psi >= threshold),
        "reference_distribution": {
            str(key): round(float(value), 6) for key, value in expected_share.items()
        },
        "current_distribution": {
            str(key): round(float(value), 6) for key, value in actual_share.items()
        },
        "reference_counts": {str(key): int(value) for key, value in expected_counts.items()},
        "current_counts": {str(key): int(value) for key, value in actual_counts.items()},
    }
