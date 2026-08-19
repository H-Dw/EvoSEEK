"""Runtime progress for long campaign steps.

Live output is stderr (logger ``fitness_agents.progress``) plus an overwritten
``status.json``. Audit events still go to ``trace.jsonl`` through the artifact writer.
Thinking-token text and prompts are never recorded here.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import threading
import time
from collections.abc import Mapping
from contextvars import ContextVar, Token
from typing import Any, Protocol, Self

LOGGER = logging.getLogger("fitness_agents.progress")
LOGGER.propagate = False

_HEARTBEAT_ENV = "FITNESS_AGENTS_HEARTBEAT_S"
_LOG_LEVEL_ENV = "FITNESS_AGENTS_LOG_LEVEL"
_DEFAULT_HEARTBEAT_S = 15.0


class ProgressTarget(Protocol):
    def heartbeat(self, message: str, *, log: bool = True, **payload: Any) -> None: ...

    def report(
        self,
        event_type: str | None,
        *,
        message: str,
        persist: bool = True,
        **payload: Any,
    ) -> None: ...

    def record_prompt_budget(self, payload: Mapping[str, Any]) -> None: ...


_current: ContextVar[ProgressTarget | None] = ContextVar("fitness_agents_progress", default=None)


def bind_progress(target: ProgressTarget) -> Token[ProgressTarget | None]:
    return _current.set(target)


def reset_progress(token: Token[ProgressTarget | None]) -> None:
    _current.reset(token)


def current_progress() -> ProgressTarget | None:
    return _current.get()


def heartbeat_interval_s() -> float:
    raw = os.environ.get(_HEARTBEAT_ENV)
    if not raw:
        return _DEFAULT_HEARTBEAT_S
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_HEARTBEAT_S
    return value if value > 0 else _DEFAULT_HEARTBEAT_S


def _format_message(message: str, payload: Mapping[str, Any]) -> str:
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


def _log(message: str, payload: Mapping[str, Any] | None = None) -> None:
    LOGGER.info(_format_message(message, payload or {}))


def report_event(
    event_type: str | None,
    *,
    message: str,
    persist: bool = True,
    **payload: Any,
) -> None:
    target = _current.get()
    if target is not None:
        target.report(event_type, message=message, persist=persist, **payload)
        return
    _log(message, payload)


def report_prompt_budget(**payload: Any) -> None:
    """Persist size-only prompt diagnostics without recording prompt contents."""

    allowed = {
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
        "round_id",
    }
    record = {key: value for key, value in payload.items() if key in allowed}
    target = _current.get()
    if target is not None:
        recorder = getattr(target, "record_prompt_budget", None)
        if callable(recorder):
            recorder(record)
        target.report(
            "llm_prompt_budget",
            message=f"LLM prompt budget checked for {record.get('role', 'unknown')}",
            persist=True,
            **record,
        )
        return
    _log("LLM prompt budget checked", record)


def heartbeat(message: str, *, log: bool = True, **payload: Any) -> None:
    target = _current.get()
    if target is not None:
        target.heartbeat(message, log=log, **payload)
        return
    if log:
        _log(message, payload)


def emit_batch_progress(
    label: str,
    *,
    completed: int,
    total: int,
    items_done: int | None = None,
    items_total: int | None = None,
) -> None:
    """Update status every batch; log roughly every 10% plus first and last."""

    if total <= 0:
        return
    stride = max(1, math.ceil(total / 10))
    log = completed <= 1 or completed == total or completed % stride == 0
    parts = [f"{label} {completed}/{total}"]
    if items_done is not None and items_total is not None:
        parts.append(f"items {items_done}/{items_total}")
    heartbeat(
        " ".join(parts),
        log=log,
        completed=completed,
        total=total,
        items_done=items_done,
        items_total=items_total,
    )


class TimedHeartbeat:
    """Stderr/status pulse while a blocking call (LLM, GP fit) has no inner loop."""

    def __init__(self, label: str, *, interval_s: float | None = None) -> None:
        self.label = label
        self.interval_s = heartbeat_interval_s() if interval_s is None else interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def __enter__(self) -> Self:
        if self.interval_s <= 0:
            return self
        self._started = time.monotonic()
        self._thread = threading.Thread(
            target=self._run, name="fitness-agents-heartbeat", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            elapsed = time.monotonic() - self._started
            heartbeat(f"{self.label} still running ({elapsed:.0f}s)")

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=min(1.0, self.interval_s))


def add_logging_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide progress lines on stderr (warnings and errors still appear)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Progress log level (default INFO, or FITNESS_AGENTS_LOG_LEVEL)",
    )
    return parser


def configure_progress_logging(
    *,
    level: int | str | None = None,
    stream: Any | None = None,
    quiet: bool = False,
) -> logging.Logger:
    if quiet:
        resolved: int | str = logging.WARNING
    elif level is not None:
        resolved = level
    else:
        resolved = os.environ.get(_LOG_LEVEL_ENV, "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)
    logger = LOGGER
    logger.setLevel(resolved)
    logger.propagate = False
    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(resolved)
        return logger
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setLevel(resolved)
    handler.setFormatter(
        logging.Formatter("[fitness-agents] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(handler)
    return logger


def configure_from_args(args: argparse.Namespace) -> logging.Logger:
    return configure_progress_logging(
        level=getattr(args, "log_level", None),
        quiet=bool(getattr(args, "quiet", False)),
    )
