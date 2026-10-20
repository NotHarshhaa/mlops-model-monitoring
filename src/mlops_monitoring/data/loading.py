"""Dataset loading and reproducible train/test splitting."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

DEFAULT_TARGET_COLUMN = "target"
DEMO_DATASET_NAME = "breast-cancer"


def load_breast_cancer_dataset() -> pd.DataFrame:
    """Load the built-in breast cancer dataset as a tidy data frame.

    Using a scikit-learn bundled dataset keeps the reference workflow fully
    offline and reproducible. Replace it with a CSV of your own by pointing
    ``DATASET_PATH`` at that file.
    """
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.data.copy()
    frame.columns = [str(column).strip().replace(" ", "_") for column in frame.columns]
    frame[DEFAULT_TARGET_COLUMN] = bunch.target.astype(int)
    frame = frame.reset_index(drop=True)
    frame.attrs["dataset_name"] = DEMO_DATASET_NAME
    frame.attrs["target_names"] = [str(name) for name in bunch.target_names]
    return frame


def save_dataset(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logger.info("Dataset written", extra={"path": str(path), "rows": len(frame)})
    return path


def load_dataset(
    path: Path,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Load a CSV dataset and verify that the target column is present."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    if target_column not in frame.columns:
        raise ValueError(
            f"Dataset at {path} does not contain the target column '{target_column}'. "
            f"Available columns: {', '.join(map(str, frame.columns))}"
        )
    if frame.empty:
        raise ValueError(f"Dataset at {path} is empty")
    frame.attrs.setdefault("dataset_name", path.stem)
    return frame


def split_dataset(
    frame: pd.DataFrame,
    target_column: str = DEFAULT_TARGET_COLUMN,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified, reproducible train/test split."""
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' is missing")

    class_counts = frame[target_column].value_counts()
    if len(class_counts) < 2:
        raise ValueError("Dataset must contain at least two target classes")
    if class_counts.min() < 2:
        raise ValueError("Every target class needs at least two rows for a stratified split")

    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=random_state,
        stratify=frame[target_column],
    )
    train_frame = train_frame.reset_index(drop=True)
    test_frame = test_frame.reset_index(drop=True)

    for name, value in frame.attrs.items():
        train_frame.attrs.setdefault(name, value)
        test_frame.attrs.setdefault(name, value)

    logger.info(
        "Dataset split complete",
        extra={
            "train_rows": len(train_frame),
            "test_rows": len(test_frame),
            "test_size": test_size,
            "random_state": random_state,
        },
    )
    return train_frame, test_frame


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Stable fingerprint of a dataset's schema and values."""
    hashed = pd.util.hash_pandas_object(frame, index=False).to_numpy()
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(hashed.tobytes())
    return digest.hexdigest()


def resolve_target_names(
    classes: list[object],
    fallback_names: list[str] | None = None,
) -> list[str]:
    """Map raw class values to human-readable display names."""
    if fallback_names and len(fallback_names) == len(classes):
        return [str(name) for name in fallback_names]
    return [str(value) for value in classes]
