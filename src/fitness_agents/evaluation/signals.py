from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from fitness_agents.contracts.schemas import (
    CriterionResult,
    CriterionSignal,
    FalsificationCriterion,
    FitnessObservation,
)


def _visible(
    observations: Sequence[FitnessObservation], ids: set[str]
) -> list[FitnessObservation]:
    return [item for item in observations if item.variant_id in ids]


class BatchMedianLiftDetector:
    name = "batch_median_lift"
    version = "1.0.0"

    def evaluate(
        self,
        criterion: FalsificationCriterion,
        observations: Sequence[FitnessObservation],
        context: dict,
    ) -> CriterionResult:
        targets = _visible(observations, set(criterion.target_variant_ids))
        comparators = _visible(observations, set(criterion.comparator_variant_ids))
        observed_ids = tuple(item.variant_id for item in targets + comparators)
        if len(targets) < criterion.min_observations or not comparators:
            return CriterionResult(
                criterion.criterion_id, CriterionSignal.UNRESOLVED, None, None, None,
                observed_ids, "insufficient_data", self.name, self.version,
                "INSUFFICIENT_OBSERVATIONS",
            )
        target_value = float(median(item.fitness for item in targets))
        comparator_value = float(median(item.fitness for item in comparators))
        effect = target_value - comparator_value
        if effect > criterion.support_threshold:
            signal = CriterionSignal.SUPPORT
            reason = "SUPPORT_THRESHOLD_MET"
        elif effect <= criterion.contradiction_threshold:
            signal = CriterionSignal.CONTRADICT
            reason = "CONTRADICTION_THRESHOLD_MET"
        else:
            signal = CriterionSignal.UNRESOLVED
            reason = "INDETERMINATE_EFFECT"
        return CriterionResult(
            criterion.criterion_id, signal, target_value, comparator_value, effect,
            observed_ids, "point_estimate_only", self.name, self.version, reason,
        )


class ThresholdDetector:
    name = "threshold"
    version = "1.0.0"

    def evaluate(
        self,
        criterion: FalsificationCriterion,
        observations: Sequence[FitnessObservation],
        context: dict,
    ) -> CriterionResult:
        targets = _visible(observations, set(criterion.target_variant_ids))
        if len(targets) < criterion.min_observations:
            return CriterionResult(
                criterion.criterion_id, CriterionSignal.UNRESOLVED, None, None, None,
                tuple(item.variant_id for item in targets), "insufficient_data",
                self.name, self.version, "INSUFFICIENT_OBSERVATIONS",
            )
        value = float(median(item.fitness for item in targets))
        if value >= criterion.support_threshold:
            signal, reason = CriterionSignal.SUPPORT, "SUPPORT_THRESHOLD_MET"
        elif value <= criterion.contradiction_threshold:
            signal, reason = CriterionSignal.CONTRADICT, "CONTRADICTION_THRESHOLD_MET"
        else:
            signal, reason = CriterionSignal.UNRESOLVED, "INDETERMINATE_VALUE"
        return CriterionResult(
            criterion.criterion_id, signal, value, None, None,
            tuple(item.variant_id for item in targets), "point_estimate_only",
            self.name, self.version, reason,
        )


class SignalDetectorRegistry:
    def __init__(self) -> None:
        self._detectors = {
            "batch_median_lift": BatchMedianLiftDetector(),
            "threshold": ThresholdDetector(),
        }

    def get(self, name: str):
        try:
            return self._detectors[name]
        except KeyError as error:
            raise ValueError(f"Unknown signal detector {name!r}") from error
