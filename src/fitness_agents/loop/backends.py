from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fitness_agents.contracts.schemas import FitnessObservation


class CsvOracleBackend:
    """Simulation oracle with one-time reveal and an irreversible final-test gate."""

    def __init__(self, oracle_path: str | Path) -> None:
        labels = pd.read_csv(oracle_path)
        required = {"variant_id", "fitness", "split_role"}
        if missing := required.difference(labels.columns):
            raise ValueError(f"Oracle table missing columns: {sorted(missing)}")
        self._labels = labels.set_index("variant_id")
        self._pool_ids = set(labels.loc[labels["split_role"] == "oracle_pool", "variant_id"])
        self._final_ids = set(labels.loc[labels["split_role"] == "final_test", "variant_id"])
        self._revealed: set[str] = set()
        self._pending: dict[str, tuple[list[str], int]] = {}
        self._collected_runs: set[str] = set()
        self._final_opened = False

    @property
    def revealed_ids(self) -> frozenset[str]:
        return frozenset(self._revealed)

    @property
    def final_opened(self) -> bool:
        return self._final_opened

    def submit(self, variant_ids: Sequence[str], round_id: int) -> str:
        if self._final_opened:
            raise RuntimeError("Campaign is finalized; no more submissions are permitted")
        submitted = list(variant_ids)
        if len(submitted) != len(set(submitted)):
            raise ValueError("A batch cannot contain duplicate variants")
        invalid = set(submitted).difference(self._pool_ids)
        if invalid:
            raise PermissionError(f"Submission contains non-oracle-pool IDs: {sorted(invalid)[:3]}")
        repeated = set(submitted).intersection(self._revealed)
        if repeated:
            raise PermissionError(f"Variants cannot be revealed twice: {sorted(repeated)[:3]}")
        run_id = f"experiment:{round_id}:{uuid.uuid4().hex[:12]}"
        self._pending[run_id] = (submitted, round_id)
        return run_id

    def collect(self, experiment_run_id: str) -> list[FitnessObservation]:
        if experiment_run_id in self._collected_runs:
            raise PermissionError("An experiment run can only be collected once")
        if experiment_run_id not in self._pending:
            raise KeyError(f"Unknown experiment run {experiment_run_id}")
        variant_ids, round_id = self._pending.pop(experiment_run_id)
        self._collected_runs.add(experiment_run_id)
        self._revealed.update(variant_ids)
        return [
            FitnessObservation(
                variant_id=variant_id,
                fitness=float(self._labels.loc[variant_id, "fitness"]),
                split_role="oracle_pool",
                round_revealed=round_id,
                source="csv_oracle",
            )
            for variant_id in variant_ids
        ]

    def open_final_test(self) -> list[FitnessObservation]:
        if self._final_opened:
            raise PermissionError("Final test can only be opened once")
        self._final_opened = True
        return [
            FitnessObservation(
                variant_id=variant_id,
                fitness=float(self._labels.loc[variant_id, "fitness"]),
                split_role="final_test",
                round_revealed=-2,
                source="final_test_once",
            )
            for variant_id in sorted(self._final_ids)
        ]

