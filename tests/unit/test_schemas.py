"""Unit tests for API request schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlops_monitoring.api.schemas import BatchPredictionRequest, PredictionRequest


def test_prediction_request_accepts_feature_values():
    request = PredictionRequest(features={"a": 1, "b": 2.5, "c": "text"})

    assert request.features["a"] == 1


def test_prediction_request_rejects_empty_features():
    with pytest.raises(ValidationError):
        PredictionRequest(features={})


def test_prediction_request_rejects_extra_fields():
    with pytest.raises(ValidationError):
        PredictionRequest(features={"a": 1}, unexpected=True)


def test_batch_request_requires_at_least_one_item():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(items=[])

    batch = BatchPredictionRequest(items=[{"features": {"a": 1}}])
    assert batch.items[0].features == {"a": 1}
