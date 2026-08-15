from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.contracts.schemas import SelectionRecord


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


class JsonArtifactWriter:
    """Append-only event trace plus human-readable per-round artifacts."""

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_dir = output_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.run_dir / "trace.jsonl"

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": _jsonable(payload),
        }
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def write_json(self, relative_path: str, payload: Any) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    def write_selection(self, round_id: int, records: Sequence[SelectionRecord]) -> Path:
        target = self.run_dir / f"round_{round_id:02d}" / "selection.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = [_jsonable(record) for record in records]
        if not rows:
            return target
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return target
