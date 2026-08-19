"""Versioned, coverage-checked scoring state for one hypothesis revision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fitness_agents.contracts.schemas import DesignScore, Prediction, Variant


class StateCoverageError(RuntimeError):
    """A selection references IDs missing from one or more versioned score maps."""

    def __init__(
        self,
        *,
        snapshot_version: str,
        missing_by_field: Mapping[str, Sequence[str]],
    ) -> None:
        normalized = {
            field: tuple(sorted(set(values)))
            for field, values in missing_by_field.items()
            if values
        }
        self.snapshot_version = snapshot_version
        self.missing_by_field = normalized
        super().__init__(
            f"scoring snapshot {snapshot_version} has incomplete ID coverage: {normalized}"
        )


@dataclass(frozen=True)
class RoundScoringSnapshot:
    hypothesis_id: str | None
    version: int
    eligible: tuple[Variant, ...]
    design_score_by_id: Mapping[str, DesignScore]
    prediction_by_id: Mapping[str, Prediction]
    model_ranks: Mapping[str, int]
    all_scores: Mapping[str, float]
    acquisition_ranks: Mapping[str, int]
    eligible_ranks: Mapping[str, int]

    @property
    def snapshot_version(self) -> str:
        return f"{self.hypothesis_id or 'no-hypothesis'}:v{self.version}"

    @property
    def eligible_ids(self) -> frozenset[str]:
        return frozenset(item.variant_id for item in self.eligible)

    def assert_eligible_coverage(self) -> None:
        self._assert_coverage(self.eligible_ids)

    def assert_selection_coverage(self, selected_ids: Sequence[str]) -> None:
        selected = frozenset(selected_ids)
        outside = selected.difference(self.eligible_ids)
        missing: dict[str, Sequence[str]] = {}
        if outside:
            missing["eligible"] = tuple(outside)
        try:
            self._assert_coverage(selected)
        except StateCoverageError as error:
            missing.update(error.missing_by_field)
        if missing:
            raise StateCoverageError(
                snapshot_version=self.snapshot_version,
                missing_by_field=missing,
            )

    def _assert_coverage(self, ids: frozenset[str]) -> None:
        maps = {
            "design_score_by_id": self.design_score_by_id,
            "prediction_by_id": self.prediction_by_id,
            "model_ranks": self.model_ranks,
            "all_scores": self.all_scores,
            "acquisition_ranks": self.acquisition_ranks,
            "eligible_ranks": self.eligible_ranks,
        }
        missing = {
            name: tuple(ids.difference(values.keys()))
            for name, values in maps.items()
            if ids.difference(values.keys())
        }
        if missing:
            raise StateCoverageError(
                snapshot_version=self.snapshot_version,
                missing_by_field=missing,
            )
