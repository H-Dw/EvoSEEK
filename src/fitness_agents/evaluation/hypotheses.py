from __future__ import annotations

import uuid
from collections.abc import Sequence

from fitness_agents.contracts.schemas import (
    CriterionSignal,
    FalsificationCriterion,
    FalsificationSpec,
    FitnessObservation,
    HypothesisAssessment,
    HypothesisStatus,
)
from fitness_agents.validation.batch import content_hash

from .signals import SignalDetectorRegistry


def preregister_batch_median_test(
    *,
    hypothesis_id: str,
    round_id: int,
    target_variant_ids: Sequence[str],
    visible_observations: Sequence[FitnessObservation],
) -> FalsificationSpec:
    comparator_ids = tuple(item.variant_id for item in visible_observations)
    criterion = FalsificationCriterion(
        criterion_id=f"criterion:{hypothesis_id}:batch_median",
        detector_name="batch_median_lift",
        detector_version="1.0.0",
        target_variant_ids=tuple(target_variant_ids),
        comparator_variant_ids=comparator_ids,
        metric="median_fitness_lift",
        expected_direction="greater",
        support_threshold=0.0,
        contradiction_threshold=0.0,
        min_observations=max(1, len(target_variant_ids)),
        min_replicates=1,
        primary=True,
    )
    payload = {
        "hypothesis_id": hypothesis_id,
        "round_id": round_id,
        "version": "1.0.0",
        "criteria": (criterion,),
        "reduction_policy": "primary_contradiction_first_v1",
        "description": (
            "The selected batch median must exceed the preregistered visible-observation median."
        ),
    }
    return FalsificationSpec(
        spec_id=f"falsification:{hypothesis_id}:v1",
        hypothesis_id=hypothesis_id,
        version="1.0.0",
        registered_at_round=round_id,
        criteria=(criterion,),
        reduction_policy="primary_contradiction_first_v1",
        human_readable_description=(
            payload["description"]
        ),
        pre_registration_hash=content_hash(payload),
    )


def verify_falsification_spec(spec: FalsificationSpec) -> None:
    payload = {
        "hypothesis_id": spec.hypothesis_id,
        "round_id": spec.registered_at_round,
        "version": spec.version,
        "criteria": spec.criteria,
        "reduction_policy": spec.reduction_policy,
        "description": spec.human_readable_description,
    }
    if content_hash(payload) != spec.pre_registration_hash:
        raise PermissionError("FalsificationSpec changed after preregistration")


class DeterministicHypothesisEvaluator:
    version = "1.0.0"

    def __init__(self, registry: SignalDetectorRegistry | None = None) -> None:
        self.registry = registry or SignalDetectorRegistry()

    def evaluate(
        self,
        *,
        spec: FalsificationSpec,
        observations: Sequence[FitnessObservation],
        round_id: int,
    ) -> HypothesisAssessment:
        verify_falsification_spec(spec)
        results = tuple(
            self.registry.get(criterion.detector_name).evaluate(
                criterion, observations, {"round_id": round_id}
            )
            for criterion in spec.criteria
        )
        primary = [
            result
            for criterion, result in zip(spec.criteria, results, strict=True)
            if criterion.primary
        ]
        if any(item.signal is CriterionSignal.CONTRADICT for item in primary):
            status = HypothesisStatus.CONTRADICTED
        elif primary and all(item.signal is CriterionSignal.SUPPORT for item in primary):
            status = HypothesisStatus.SUPPORTED
        else:
            status = HypothesisStatus.INCONCLUSIVE
        decisive = tuple(
            item.criterion_id for item in primary if item.signal is not CriterionSignal.UNRESOLVED
        )
        unresolved = tuple(
            item.criterion_id for item in results if item.signal is CriterionSignal.UNRESOLVED
        )
        observation_ids = tuple(dict.fromkeys(
            observation_id for result in results for observation_id in result.observation_ids
        ))
        payload = {
            "hypothesis_id": spec.hypothesis_id,
            "spec_id": spec.spec_id,
            "round_id": round_id,
            "status": status,
            "results": results,
        }
        return HypothesisAssessment(
            assessment_id=f"assessment:{uuid.uuid4().hex}",
            hypothesis_id=spec.hypothesis_id,
            falsification_spec_id=spec.spec_id,
            round_id=round_id,
            status=status,
            criterion_results=results,
            observation_ids=observation_ids,
            decisive_criterion_ids=decisive,
            unresolved_criterion_ids=unresolved,
            evaluator_version=self.version,
            assessment_hash=content_hash(payload),
        )
