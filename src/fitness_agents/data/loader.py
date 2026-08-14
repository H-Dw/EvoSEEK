from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fitness_agents.contracts.schemas import FitnessObservation, Variant

PUBLIC_REQUIRED = {
    "variant_id",
    "variant",
    "sequence",
    "mutation_notation",
    "mutation_count",
    "split_role",
}
ORACLE_REQUIRED = {"variant_id", "fitness", "split_role"}


@dataclass(frozen=True)
class DatasetBundle:
    initial_variants: list[Variant]
    initial_observations: list[FitnessObservation]
    validation_variants: list[Variant]
    validation_observations: list[FitnessObservation]
    oracle_pool: list[Variant]
    final_test: list[Variant]

    @property
    def public_candidates(self) -> list[Variant]:
        return [*self.oracle_pool]


def _row_to_variant(row: object) -> Variant:
    return Variant(
        variant_id=str(row.variant_id),
        variant=str(row.variant),
        sequence=str(row.sequence),
        mutation_notation=str(row.mutation_notation),
        mutation_count=int(row.mutation_count),
        split_role=str(row.split_role),
    )


def _visible_observations(
    variants: list[Variant], labels: pd.DataFrame, round_revealed: int
) -> list[FitnessObservation]:
    label_map = labels.set_index("variant_id")["fitness"].to_dict()
    return [
        FitnessObservation(
            variant_id=variant.variant_id,
            fitness=float(label_map[variant.variant_id]),
            split_role=variant.split_role,
            round_revealed=round_revealed,
            source="benchmark_initial" if round_revealed == 0 else "validation",
        )
        for variant in variants
    ]


def load_dataset_bundle(public_path: str | Path, oracle_path: str | Path) -> DatasetBundle:
    public = pd.read_csv(public_path)
    labels = pd.read_csv(oracle_path)
    if missing := PUBLIC_REQUIRED.difference(public.columns):
        raise ValueError(f"Public dataset missing columns: {sorted(missing)}")
    if hidden := {"fitness", "raw_fitness", "normalized_fitness"}.intersection(public.columns):
        raise ValueError(f"Public candidate table contains hidden labels: {sorted(hidden)}")
    if missing := ORACLE_REQUIRED.difference(labels.columns):
        raise ValueError(f"Oracle dataset missing columns: {sorted(missing)}")
    if public["variant_id"].duplicated().any() or labels["variant_id"].duplicated().any():
        raise ValueError("Variant IDs must be unique in public and oracle tables")
    if set(public["variant_id"]) != set(labels["variant_id"]):
        raise ValueError("Public and oracle tables have different variant IDs")
    if public.groupby("variant_id")["split_role"].nunique().max() != 1:
        raise ValueError("A variant cannot belong to more than one split")

    by_role: dict[str, list[Variant]] = {}
    for role, frame in public.groupby("split_role", sort=False):
        by_role[str(role)] = [_row_to_variant(row) for row in frame.itertuples(index=False)]
    required_roles = {"initial_observed", "validation", "oracle_pool", "final_test"}
    if missing := required_roles.difference(by_role):
        raise ValueError(f"Missing split roles: {sorted(missing)}")
    initial = by_role["initial_observed"]
    validation = by_role["validation"]
    return DatasetBundle(
        initial_variants=initial,
        initial_observations=_visible_observations(initial, labels, 0),
        validation_variants=validation,
        validation_observations=_visible_observations(validation, labels, -1),
        oracle_pool=by_role["oracle_pool"],
        final_test=by_role["final_test"],
    )

