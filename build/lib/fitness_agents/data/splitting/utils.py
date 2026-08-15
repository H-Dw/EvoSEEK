from __future__ import annotations

import json
from collections.abc import Iterable

import pandas as pd

from .hashing import stable_order


def decode_tokens(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        decoded = json.loads(value)
    elif isinstance(value, Iterable):
        decoded = list(value)
    else:
        raise TypeError(f"Cannot decode mutation tokens from {value!r}")
    return tuple(str(token) for token in decoded)


def stratified_shards(
    frame: pd.DataFrame,
    *,
    n_folds: int,
    salt: bytes,
    namespace: str,
    strata: tuple[str, ...],
) -> dict[str, int]:
    missing = set(strata).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing stratification columns: {sorted(missing)}")
    assignment: dict[str, int] = {}
    group_key: str | list[str] = strata[0] if len(strata) == 1 else list(strata)
    for key, group in frame.groupby(group_key, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        ordered = stable_order(
            group["variant_id"].astype(str).tolist(), salt, namespace, *key_tuple
        )
        for index, variant_id in enumerate(ordered):
            assignment[variant_id] = index % n_folds
    return assignment


def allocate_counts(group_sizes: dict[object, int], total: int) -> dict[object, int]:
    if total < 0 or total > sum(group_sizes.values()):
        raise ValueError("Requested count is outside the available population")
    keys = sorted(group_sizes, key=str)
    if not keys:
        return {}
    base = total // len(keys)
    allocation = {key: min(base, group_sizes[key]) for key in keys}
    remaining = total - sum(allocation.values())
    while remaining:
        candidates = [key for key in keys if allocation[key] < group_sizes[key]]
        if not candidates:
            raise ValueError("Unable to allocate requested count across strata")
        key = min(candidates, key=lambda item: (allocation[item] / group_sizes[item], str(item)))
        allocation[key] += 1
        remaining -= 1
    return allocation
