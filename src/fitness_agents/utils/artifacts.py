from __future__ import annotations

import csv
import json
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from fitness_agents.contracts.schemas import SelectionRecord

LOGGER = logging.getLogger("fitness_agents.progress")

_ATOMIC_REPLACE_ATTEMPTS = 8
_ATOMIC_REPLACE_INITIAL_DELAY_S = 0.025
_ATOMIC_REPLACE_MAX_DELAY_S = 0.5


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
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


def _is_status_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float, np.generic))


def _atomic_replace_with_retry(source: Path, target: Path) -> None:
    """Replace ``target`` atomically, tolerating brief Windows/WSL reader locks.

    Files below ``/mnt/*`` can be opened concurrently by Windows indexers or
    monitoring processes.  In that case an otherwise valid ``Path.replace`` may
    fail transiently with ``PermissionError``.  Keep the atomic temp-file
    protocol and retry only that narrow error; persistent permission failures
    are still raised to the caller.
    """

    delay_s = _ATOMIC_REPLACE_INITIAL_DELAY_S
    for attempt in range(1, _ATOMIC_REPLACE_ATTEMPTS + 1):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay_s)
            delay_s = min(delay_s * 2, _ATOMIC_REPLACE_MAX_DELAY_S)


def _format_progress_line(message: str, payload: Mapping[str, Any]) -> str:
    extras: list[str] = []
    for key in (
        "round_id",
        "phase",
        "attempt",
        "completed",
        "total",
        "n_train",
        "n_candidates",
        "n_remaining",
        "model",
        "latency_s",
        "device",
    ):
        if key in payload and payload[key] is not None:
            extras.append(f"{key}={payload[key]}")
    if extras:
        return f"{message} ({', '.join(extras)})"
    return message


class JsonArtifactWriter:
    """Append-only event trace plus human-readable per-round artifacts.

    ``status.json`` is overwritten so operators can watch the current phase without
    parsing the full trace. Stderr progress uses logger ``fitness_agents.progress``.
    """

    def __init__(self, output_root: Path, run_id: str) -> None:
        self.run_id = run_id
        self.run_dir = output_root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.run_dir / "trace.jsonl"
        self.status_path = self.run_dir / "status.json"
        self._started = time.monotonic()
        self._write_lock = threading.RLock()
        self._conversation_sequence = 0
        self._status: dict[str, Any] = {
            "run_id": run_id,
            "phase": "initialized",
            "round_id": 0,
            "message": "run directory created",
        }
        self.write_status(message="run directory created", phase="initialized", round_id=0)

    def write_status(
        self,
        *,
        message: str,
        event_type: str | None = None,
        **payload: Any,
    ) -> Path:
        with self._write_lock:
            phase = payload.get("phase", self._status.get("phase"))
            round_id = payload.get("round_id", self._status.get("round_id", 0))
            detail = {
                str(key): _jsonable(value)
                for key, value in payload.items()
                if key not in {"phase", "round_id", "message"} and _is_status_scalar(value)
            }
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "run_id": self.run_id,
                "phase": phase,
                "round_id": round_id,
                "message": message,
                "elapsed_s": round(time.monotonic() - self._started, 3),
                "event_type": event_type,
            }
            if detail:
                record["detail"] = detail
            self._status = record
            encoded = json.dumps(_jsonable(record), ensure_ascii=False, indent=2, sort_keys=True)
            temporary = self.status_path.with_suffix(".json.tmp")
            temporary.write_text(encoded + "\n", encoding="utf-8")
            _atomic_replace_with_retry(temporary, self.status_path)
        return self.status_path

    def heartbeat(self, message: str, *, log: bool = True, **payload: Any) -> None:
        merged = {**self._status, **payload}
        self.write_status(
            message=message,
            event_type=merged.get("event_type"),
            phase=merged.get("phase"),
            round_id=merged.get("round_id", 0),
            **{
                key: value
                for key, value in payload.items()
                if key not in {"phase", "round_id", "message", "event_type"}
            },
        )
        if log:
            LOGGER.info(_format_progress_line(message, {**self._status, **payload}))

    def report(
        self,
        event_type: str | None,
        *,
        message: str,
        persist: bool = True,
        **payload: Any,
    ) -> None:
        self.write_status(message=message, event_type=event_type, **payload)
        LOGGER.info(_format_progress_line(message, payload))
        if persist and event_type:
            self.event(event_type, {"message": message, **payload})

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": _jsonable(payload),
        }
        with self._write_lock, self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record_prompt_budget(self, payload: Mapping[str, Any]) -> None:
        """Append one size-only budget record to the current round and role artifact."""

        allowed = (
            "role",
            "profile",
            "system_chars",
            "user_chars",
            "assistant_chars",
            "input_chars",
            "field_chars",
            "max_input_chars",
            "remaining_chars",
            "utilization_ratio",
            "budget_band",
            "request_started",
        )
        record = {key: _jsonable(payload.get(key)) for key in allowed}
        round_id = int(payload.get("round_id") or self._status.get("round_id") or 0)
        role = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["role"] or "unknown")).strip("_")
        role = role or "unknown"
        relative = Path(f"round_{round_id:02d}") / "llm" / role / "prompt_budget.json"
        target = self.run_dir / relative
        with self._write_lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            records: list[dict[str, Any]] = []
            if target.is_file():
                existing = json.loads(target.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    records = [dict(item) for item in existing if isinstance(item, dict)]
            records.append(record)
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _atomic_replace_with_retry(temporary, target)

    def record_llm_conversation(self, payload: Mapping[str, Any]) -> None:
        """Write one full conversation with role, stage, and attempt labels."""

        round_id = int(payload.get("round_id") or self._status.get("round_id") or 0)
        role = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", str(payload.get("role") or "unknown")
        ).strip("_") or "unknown"
        stage = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(payload.get("conversation_stage") or "single"),
        ).strip("_") or "single"
        attempt = int(payload.get("attempt") or 0)
        with self._write_lock:
            self._conversation_sequence += 1
            sequence = self._conversation_sequence
            target = (
                self.run_dir
                / f"round_{round_id:02d}"
                / "llm"
                / role
                / "conversations"
                / f"{sequence:05d}_{stage}_attempt-{attempt:02d}.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "conversation_id": f"C{sequence:05d}",
                "conversation_stage": stage,
                **{str(key): _jsonable(value) for key, value in payload.items()},
            }
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _atomic_replace_with_retry(temporary, target)

    def write_json(self, relative_path: str, payload: Any) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return target

    def write_text(self, relative_path: str, content: str) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def write_csv(self, relative_path: str, rows: Sequence[Mapping[str, Any]]) -> Path:
        target = self.run_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = [_jsonable(dict(row)) for row in rows]
        if not serialized:
            target.write_text("", encoding="utf-8")
            return target
        fieldnames = list(dict.fromkeys(key for row in serialized for key in row))
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(serialized)
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
