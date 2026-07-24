"""Durable, machine-readable per-episode evaluation records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable


SCHEMA_VERSION = 1
REQUIRED_FIELDS = (
    "scene_id",
    "episode_id",
    "instantaneous_pddl_success",
    "ever_pddl_success",
    "crash_type",
    "steps",
    "tokens",
)


def append_episode_result(path: str, record: Dict[str, Any]) -> None:
    """Append one complete JSON record and fsync it before returning."""
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Episode result is missing required fields: {missing}")

    payload = dict(record)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault(
        "recorded_at",
        datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    )

    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    with result_path.open("a", encoding="utf-8") as result_file:
        result_file.write(encoded + "\n")
        result_file.flush()
        os.fsync(result_file.fileno())


def read_episode_results(path: str) -> Iterable[Dict[str, Any]]:
    """Yield complete JSONL records, rejecting malformed non-empty lines."""
    with Path(path).open("r", encoding="utf-8") as result_file:
        for line in result_file:
            if line.strip():
                yield json.loads(line)
