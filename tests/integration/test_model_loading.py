"""Integration tests for model artifact loading and schema validation."""

from __future__ import annotations

import pytest
from sklearn.linear_model import LogisticRegression

from mlops_monitoring.data.loading import split_dataset
from mlops_monitoring.features.preprocessing import (
    build_feature_specs,
    build_model_pipeline,
)
from mlops_monitoring.models.loading import (
    FeatureValidationError,
    ModelArtifactError,
    ModelBundle,
    ModelLoader,
    load_model_bundle,
    save_model_bundle,
)


def test_saved_bundle_roundtrip(trained_run, sample_features, features_for_row, demo_frame):
    bundle = load_model_bundle(trained_run["settings"].model_path)

    assert bundle.model_name == trained_run["summary"]["model_name"]
    assert bundle.feature_names
    assert bundle.metrics["f1"] > 0.7

    loader = ModelLoader(trained_run["settings"].model_path)
    loaded = loader.load()
    frame = loader.prepare_frame([sample_features])
    predictions, probabilities = loader.predict_frame(frame)

    assert len(predictions) == 1
    assert probabilities is not None
    assert probabilities.shape == (1, len(loaded.classes))

    second_row = features_for_row(demo_frame.iloc[1])
    assert loader.validate_features(second_row) == []


@pytest.mark.parametrize(
    "mutator,expected",
    [
        (lambda features: features.pop(next(iter(features))), "Missing required feature"),
        (lambda features: features.update({"extra_feature": 1.0}), "Unexpected feature"),
        (lambda features: features.update({"mean_radius": "not-a-number"}), "must be a number"),
        (lambda features: features.update({"mean_radius": None}), "must not be null"),
        (lambda features: features.update({"mean_radius": float("inf")}), "must be finite"),
    ],
)
def test_loader_rejects_invalid_features(trained_run, sample_features, mutator, expected):
    features = dict(sample_features)
    mutator(features)
    loader = ModelLoader(trained_run["settings"].model_path)
    loader.load()

    with pytest.raises(FeatureValidationError) as exc_info:
        loader.prepare_frame([features])

    assert expected in str(exc_info.value)


def test_loader_reports_missing_artifact(tmp_path):
    loader = ModelLoader(tmp_path / "does-not-exist")

    with pytest.raises(ModelArtifactError):
        loader.load()
    assert not loader.is_loaded
    assert loader.load_error is not None


def test_bundle_with_categorical_feature_validates_known_values(tmp_path, demo_frame):
    frame, _ = split_dataset(demo_frame, test_size=0.3, random_state=5)
    frame = frame.head(120).copy()
    frame["category"] = ["alpha", "beta"] * (len(frame) // 2)

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=1000), ["mean_radius"], ["category"]
    )
    pipeline.fit(frame[["mean_radius", "category"]], frame["target"])
    bundle = ModelBundle(
        model=pipeline,
        model_name="categorical-model",
        model_version="test",
        feature_specs=build_feature_specs(frame, ["mean_radius"], ["category"]),
        classes=[0, 1],
        target_names=["no", "yes"],
        metrics={},
    )
    save_model_bundle(bundle, tmp_path / "model")

    loader = ModelLoader(tmp_path / "model")
    loader.load()
    assert loader.validate_features({"mean_radius": 12.0, "category": "alpha"}) == []

    with pytest.raises(FeatureValidationError):
        loader.prepare_frame([{"mean_radius": 12.0, "category": "gamma"}])
