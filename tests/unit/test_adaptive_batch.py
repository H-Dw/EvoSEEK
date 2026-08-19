from __future__ import annotations

import threading
import time
from contextvars import ContextVar

import pytest

from fitness_agents.agents.adaptive_batch import adaptive_batch_submit


class _LengthBoundary(RuntimeError):
    pass


def test_adaptive_batch_parallelizes_and_halves_only_failed_work() -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []
    lock = threading.Lock()
    active = 0
    max_active = 0

    def submit(work):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append((work.batch_id, work.item_ids))
        try:
            time.sleep(0.02)
            if len(work.items) == 8:
                raise _LengthBoundary
            return tuple(work.items)
        finally:
            with lock:
                active -= 1

    results = adaptive_batch_submit(
        tuple(f"sample:{index}" for index in range(10)),
        item_id=str,
        submit_batch=submit,
        initial_batch_size=8,
        max_parallel_batches=4,
        should_split_failure=lambda error: isinstance(error, _LengthBoundary),
        role="test-role",
        event_reporter=lambda *_args, **_kwargs: None,
    )

    assert sorted(len(item_ids) for _batch_id, item_ids in calls) == [2, 4, 4, 8]
    assert [item.item_ids for item in results] == [
        tuple(f"sample:{index}" for index in range(4)),
        tuple(f"sample:{index}" for index in range(4, 8)),
        ("sample:8", "sample:9"),
    ]
    assert max_active >= 2


def test_adaptive_batch_rejects_duplicate_item_ids_before_submission() -> None:
    submitted = False

    def submit(_work):
        nonlocal submitted
        submitted = True

    with pytest.raises(ValueError, match="must be unique"):
        adaptive_batch_submit(
            ("sample:1", "sample:1"),
            item_id=str,
            submit_batch=submit,
            initial_batch_size=8,
            max_parallel_batches=2,
            should_split_failure=lambda _error: True,
            role="test-role",
            event_reporter=lambda *_args, **_kwargs: None,
        )

    assert submitted is False


def test_adaptive_batch_propagates_trace_context_to_worker_threads() -> None:
    trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)
    trace_id.set("trace:parent")

    results = adaptive_batch_submit(
        ("sample:1", "sample:2"),
        item_id=str,
        submit_batch=lambda _work: trace_id.get(),
        initial_batch_size=1,
        max_parallel_batches=2,
        should_split_failure=lambda _error: False,
        role="test-role",
        event_reporter=lambda *_args, **_kwargs: None,
    )

    assert [item.output for item in results] == ["trace:parent", "trace:parent"]


def test_adaptive_batch_propagates_a_single_item_size_failure() -> None:
    call_count = 0

    def submit(_work):
        nonlocal call_count
        call_count += 1
        raise _LengthBoundary

    with pytest.raises(_LengthBoundary):
        adaptive_batch_submit(
            ("sample:1",),
            item_id=str,
            submit_batch=submit,
            initial_batch_size=8,
            max_parallel_batches=2,
            should_split_failure=lambda error: isinstance(error, _LengthBoundary),
            role="test-role",
            event_reporter=lambda *_args, **_kwargs: None,
        )

    assert call_count == 1
