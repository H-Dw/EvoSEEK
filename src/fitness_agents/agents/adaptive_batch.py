"""Reusable adaptive parallel submission for homogeneous per-sample LLM work."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from fitness_agents.utils.progress import report_event

ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True)
class AdaptiveBatchWork(Generic[ItemT]):
    """One independently retried batch submission."""

    batch_id: str
    items: tuple[ItemT, ...]
    item_ids: tuple[str, ...]
    split_depth: int


@dataclass(frozen=True)
class AdaptiveBatchResult(Generic[OutputT]):
    """Successful typed output with its runtime-owned coverage metadata."""

    batch_id: str
    item_ids: tuple[str, ...]
    split_depth: int
    output: OutputT


class AdaptiveBatchExecutionError(RuntimeError):
    """Terminal batch failure carrying every sibling result completed so far."""

    def __init__(
        self,
        cause: Exception,
        *,
        completed: Sequence[AdaptiveBatchResult[Any]],
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.completed = tuple(completed)
        for name in (
            "error_code",
            "failure_category",
            "failure_stage",
            "batch_id",
            "sample_ids",
            "validation_paths",
            "input_chars",
            "request_started",
        ):
            if hasattr(cause, name):
                setattr(self, name, getattr(cause, name))


def _split_work(
    work: AdaptiveBatchWork[ItemT],
) -> tuple[AdaptiveBatchWork[ItemT], AdaptiveBatchWork[ItemT]]:
    midpoint = (len(work.items) + 1) // 2
    left_items = work.items[:midpoint]
    right_items = work.items[midpoint:]
    left_ids = work.item_ids[:midpoint]
    right_ids = work.item_ids[midpoint:]
    depth = work.split_depth + 1
    return (
        AdaptiveBatchWork(
            batch_id=f"{work.batch_id}.0",
            items=left_items,
            item_ids=left_ids,
            split_depth=depth,
        ),
        AdaptiveBatchWork(
            batch_id=f"{work.batch_id}.1",
            items=right_items,
            item_ids=right_ids,
            split_depth=depth,
        ),
    )


def adaptive_batch_submit(
    items: Sequence[ItemT],
    *,
    item_id: Callable[[ItemT], str],
    submit_batch: Callable[[AdaptiveBatchWork[ItemT]], OutputT],
    initial_batch_size: int,
    max_parallel_batches: int,
    should_split_failure: Callable[[Exception], bool],
    role: str,
    round_id: int | None = None,
    event_reporter: Callable[..., Any] = report_event,
    preserve_completed_on_failure: bool = False,
) -> tuple[AdaptiveBatchResult[OutputT], ...]:
    """Submit bounded batches in parallel and halve only terminal size failures.

    ``submit_batch`` owns a fresh structured-completion invocation, so transport,
    syntax, schema, semantic, and evidence retries are local to that batch.  This
    coordinator never maintains or consumes a cross-batch retry counter.
    """

    if initial_batch_size < 1:
        raise ValueError("initial_batch_size must be positive")
    if max_parallel_batches < 1:
        raise ValueError("max_parallel_batches must be positive")
    normalized_items = tuple(items)
    if not normalized_items:
        return ()
    item_ids = tuple(str(item_id(item)) for item in normalized_items)
    if any(not value for value in item_ids):
        raise ValueError("adaptive batch item IDs must be non-empty")
    item_counts = Counter(item_ids)
    duplicates = sorted(value for value, count in item_counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"adaptive batch item IDs must be unique: {duplicates}")
    original_position = {value: index for index, value in enumerate(item_ids)}
    frontier = [
        AdaptiveBatchWork(
            batch_id=f"b{index:03d}",
            items=normalized_items[offset : offset + initial_batch_size],
            item_ids=item_ids[offset : offset + initial_batch_size],
            split_depth=0,
        )
        for index, offset in enumerate(range(0, len(normalized_items), initial_batch_size))
    ]
    completed: list[AdaptiveBatchResult[OutputT]] = []
    while frontier:
        next_frontier: list[AdaptiveBatchWork[ItemT]] = []
        terminal_failures: list[Exception] = []
        worker_count = min(max_parallel_batches, len(frontier))
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"{role.replace(':', '-')}-batch",
        ) as executor:
            futures = {}
            for work in frontier:
                event_reporter(
                    "adaptive_batch_submitted",
                    message=(
                        f"{role} submitted batch {work.batch_id} with "
                        f"{len(work.items)} items"
                    ),
                    persist=True,
                    role=role,
                    round_id=round_id,
                    batch_id=work.batch_id,
                    batch_size=len(work.items),
                    split_depth=work.split_depth,
                    retry_scope=f"{role}:{work.batch_id}",
                )
                futures[
                    executor.submit(copy_context().run, submit_batch, work)
                ] = work
            for future in as_completed(futures):
                work = futures[future]
                try:
                    output = future.result()
                except Exception as error:
                    if not should_split_failure(error) or len(work.items) == 1:
                        event_reporter(
                            "adaptive_batch_failed",
                            message=f"{role} batch {work.batch_id} failed without split",
                            persist=True,
                            role=role,
                            round_id=round_id,
                            batch_id=work.batch_id,
                            batch_size=len(work.items),
                            split_depth=work.split_depth,
                            error_code=str(
                                getattr(error, "error_code", type(error).__name__)
                            ),
                            retry_disposition="propagate",
                        )
                        if preserve_completed_on_failure:
                            terminal_failures.append(error)
                            continue
                        raise
                    children = _split_work(work)
                    event_reporter(
                        "adaptive_batch_split",
                        message=(
                            f"{role} batch {work.batch_id} hit a size boundary; "
                            f"retrying as {[len(item.items) for item in children]}"
                        ),
                        persist=True,
                        role=role,
                        round_id=round_id,
                        batch_id=work.batch_id,
                        batch_size=len(work.items),
                        split_depth=work.split_depth,
                        child_batch_sizes=[len(item.items) for item in children],
                        retry_disposition="split_batch",
                    )
                    next_frontier.extend(children)
                    continue
                completed.append(
                    AdaptiveBatchResult(
                        batch_id=work.batch_id,
                        item_ids=work.item_ids,
                        split_depth=work.split_depth,
                        output=output,
                    )
                )
                event_reporter(
                    "adaptive_batch_completed",
                    message=f"{role} completed batch {work.batch_id}",
                    persist=True,
                    role=role,
                    round_id=round_id,
                    batch_id=work.batch_id,
                    batch_size=len(work.items),
                    split_depth=work.split_depth,
                    retry_disposition="completed",
                )
        if terminal_failures:
            raise AdaptiveBatchExecutionError(
                terminal_failures[0], completed=completed
            ) from terminal_failures[0]
        frontier = next_frontier
    covered_ids = tuple(value for item in completed for value in item.item_ids)
    missing = sorted(set(item_ids).difference(covered_ids))
    unexpected = sorted(set(covered_ids).difference(item_ids))
    coverage_counts = Counter(covered_ids)
    repeated = sorted(value for value, count in coverage_counts.items() if count > 1)
    if missing or unexpected or repeated:
        raise ValueError(
            "adaptive batch input coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}, repeated={repeated}"
        )
    completed.sort(key=lambda item: min(original_position[value] for value in item.item_ids))
    return tuple(completed)
