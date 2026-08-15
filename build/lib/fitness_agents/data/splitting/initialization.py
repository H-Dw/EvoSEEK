from __future__ import annotations

import json

import pandas as pd

from .hashing import stable_digest
from .utils import decode_tokens


def _coverage_atoms(row: object) -> set[str]:
    tokens = decode_tokens(row.mutation_tokens)
    decoded = [json.loads(token) for token in tokens]
    positions = sorted(f"{token[1]}:{token[2]}" for token in decoded)
    atoms = {f"token:{token}" for token in tokens}
    atoms.update(f"position:{position}" for position in positions)
    for left_index, left in enumerate(positions):
        for right in positions[left_index + 1 :]:
            atoms.add(f"position_pair:{left}|{right}")
    return atoms


def select_low_order_coverage(
    features: pd.DataFrame, *, budget: int, salt: bytes
) -> set[str]:
    if budget <= 0 or budget > len(features):
        raise ValueError(f"initial budget {budget} is invalid for {len(features)} rows")
    selected: set[str] = set()
    remaining_budget = budget
    for depth in sorted(features["mutation_count"].unique()):
        layer = features.loc[features["mutation_count"] == depth]
        layer_ids = set(layer["variant_id"].astype(str))
        if len(layer) <= remaining_budget:
            selected.update(layer_ids)
            remaining_budget -= len(layer)
            if remaining_budget == 0:
                return selected
            continue
        candidates = {str(row.variant_id): _coverage_atoms(row) for row in layer.itertuples()}
        covered: set[str] = set()
        while remaining_budget:
            best = min(
                candidates,
                key=lambda variant_id: (
                    -len(candidates[variant_id].difference(covered)),
                    stable_digest(salt, "initial_coverage", variant_id),
                    variant_id,
                ),
            )
            selected.add(best)
            covered.update(candidates.pop(best))
            remaining_budget -= 1
        return selected
    if remaining_budget:
        raise ValueError("Not enough variants to satisfy initial budget")
    return selected

