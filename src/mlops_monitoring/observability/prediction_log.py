"""Append-only prediction log used as the online monitoring data store."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def append_prediction(path: Path, record: dict[str, Any]) -> None:
    """Append a single prediction record as a JSON line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"logged_at": datetime.now(tz=UTC).isoformat(), **record}
    line = json.dumps(payload, default=_json_default, ensure_ascii=False)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_predictions(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Read prediction records, skipping malformed lines.

    When ``limit`` is provided only the most recent ``limit`` records are
    returned. Invalid lines are logged and ignored so a partially written file
    cannot break monitoring.
    """
    if not path.exists():
        return []

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed prediction log line",
                    extra={"path": str(path), "line_number": line_number},
                )
                continue
            if isinstance(record, dict):
                records.append(record)

    if limit is not None and limit >= 0:
        return records[-limit:] if limit else []
    return records


def count_predictions(path: Path) -> int:
    return len(read_predictions(path))
