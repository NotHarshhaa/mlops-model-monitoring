"""Unit tests for configuration loading."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlops_monitoring.config import PROJECT_ROOT, Settings


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("DRIFT_THRESHOLD", "0.55")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.example,http://b.example")
    monkeypatch.setenv("MODEL_NAME", "custom-model")

    settings = Settings(_env_file=None)

    assert settings.drift_threshold == 0.55
    assert settings.cors_origin_list == ["http://a.example", "http://b.example"]
    assert settings.model_name == "custom-model"


def test_relative_paths_resolve_against_project_root():
    settings = Settings(_env_file=None)

    assert settings.model_artifact_path == PROJECT_ROOT / "artifacts" / "model" / "model.joblib"
    assert settings.reference_data_path == PROJECT_ROOT / "data" / "reference.csv"


def test_artifact_file_path_is_supported(tmp_path):
    settings = Settings(_env_file=None, model_path=tmp_path / "custom.joblib")

    assert settings.model_artifact_path == tmp_path / "custom.joblib"
    assert settings.model_metadata_path == tmp_path / "metadata.json"


def test_invalid_test_size_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, test_size=1.5)


def test_cors_defaults_to_wildcard():
    settings = Settings(_env_file=None, cors_origins="")

    assert settings.cors_origin_list == ["*"]
