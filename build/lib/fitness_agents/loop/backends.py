from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from fitness_agents.contracts.schemas import FitnessObservation


class CsvOracleBackend:
    """Simulation oracle with one-time reveal and an irreversible final-test gate."""

    def __init__(
        self,
        oracle_path: str | Path,
        final_labels_path: str | Path | None = None,
        *,
        query_budget: int | None = None,
    ) -> None:
        labels = pd.read_csv(oracle_path)
        if "target" in labels and "fitness" not in labels:
            labels = labels.rename(columns={"target": "fitness"})
        required = {"variant_id", "fitness"}
        if missing := required.difference(labels.columns):
            raise ValueError(f"Oracle table missing columns: {sorted(missing)}")
        if final_labels_path is None:
            if "split_role" not in labels:
                raise ValueError("Combined oracle table requires split_role")
            final_labels = labels.loc[labels["split_role"] == "final_test"].copy()
            queryable_labels = labels.loc[labels["split_role"].isin(["oracle_pool", "candidate_pool"])].copy()
            self._query_split_role = "oracle_pool"
        else:
            queryable_labels = labels.copy()
            self._query_split_role = "candidate_pool"
            final_labels = pd.read_csv(final_labels_path)
            if "target" in final_labels and "fitness" not in final_labels:
                final_labels = final_labels.rename(columns={"target": "fitness"})
            if missing := required.difference(final_labels.columns):
                raise ValueError(f"Final label table missing columns: {sorted(missing)}")
        if queryable_labels["variant_id"].duplicated().any() or final_labels["variant_id"].duplicated().any():
            raise ValueError("Oracle and final label IDs must be unique")
        if set(queryable_labels["variant_id"]).intersection(final_labels["variant_id"]):
            raise ValueError("Queryable and final-test IDs must be disjoint")
        self._query_labels = queryable_labels.set_index("variant_id")
        self._final_labels = final_labels.set_index("variant_id")
        self._pool_ids = set(queryable_labels["variant_id"])
        self._final_ids = set(final_labels["variant_id"])
        self._query_budget = query_budget
        self._revealed: set[str] = set()
        self._pending: dict[str, tuple[list[str], int]] = {}
        self._collected_runs: set[str] = set()
        self._final_opened = False
        self._last_round = 0

    @classmethod
    def from_fold(
        cls, fold_root: str | Path, *, query_budget: int | None = None
    ) -> CsvOracleBackend:
        root = Path(fold_root)
        if root.name.startswith("fold_") and (root.parent / "manifest.public.json").is_file():
            from fitness_agents.data import load_fold_bundle

            fold_index = int(root.name.removeprefix("fold_"))
            load_fold_bundle(root.parent, fold_index, "oracle")
            load_fold_bundle(root.parent, fold_index, "evaluator")
        return cls(
            root / "oracle/queryable_labels.csv.gz",
            root / "evaluator/final_test_labels.csv.gz",
            query_budget=query_budget,
        )

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
        if not submitted:
            raise ValueError("An experiment batch cannot be empty")
        if round_id <= self._last_round:
            raise ValueError("round_id must increase strictly for each submitted batch")
        if len(submitted) != len(set(submitted)):
            raise ValueError("A batch cannot contain duplicate variants")
        invalid = set(submitted).difference(self._pool_ids)
        if invalid:
            raise PermissionError(f"Submission contains non-oracle-pool IDs: {sorted(invalid)[:3]}")
        repeated = set(submitted).intersection(self._revealed)
        if repeated:
            raise PermissionError(f"Variants cannot be revealed twice: {sorted(repeated)[:3]}")
        pending_ids = {item for ids, _ in self._pending.values() for item in ids}
        if pending := set(submitted).intersection(pending_ids):
            raise PermissionError(f"Variants are already pending: {sorted(pending)[:3]}")
        if self._query_budget is not None:
            used = len(self._revealed) + len(pending_ids)
            if used + len(submitted) > self._query_budget:
                raise PermissionError("Submission exceeds the configured query budget")
        run_id = f"experiment:{round_id}:{uuid.uuid4().hex[:12]}"
        self._pending[run_id] = (submitted, round_id)
        self._last_round = round_id
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
                fitness=float(self._query_labels.loc[variant_id, "fitness"]),
                split_role=self._query_split_role,
                round_revealed=round_id,
                source="csv_oracle",
            )
            for variant_id in variant_ids
        ]

    def open_final_test(self) -> list[FitnessObservation]:
        if self._final_opened:
            raise PermissionError("Final test can only be opened once")
        if self._pending:
            raise RuntimeError("Collect all pending experiments before opening final test")
        self._final_opened = True
        return [
            FitnessObservation(
                variant_id=variant_id,
                fitness=float(self._final_labels.loc[variant_id, "fitness"]),
                split_role="final_test",
                round_revealed=-2,
                source="final_test_once",
            )
            for variant_id in sorted(self._final_ids)
        ]
