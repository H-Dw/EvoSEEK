from __future__ import annotations

from collections.abc import Sequence

from fitness_agents.contracts.schemas import (
    CriterionSignal,
    FalsificationCriterion,
    FalsificationSpec,
    FitnessObservation,
    Hypothesis,
    HypothesisAssessment,
    HypothesisStatus,
)

from .signals import SignalDetectorRegistry


def preregister_batch_median_test(
    *,
    hypothesis: Hypothesis,
    round_id: int,
    target_variant_ids: Sequence[str],
    visible_observations: Sequence[FitnessObservation],
) -> FalsificationSpec:
    hypothesis_id = hypothesis.hypothesis_id
    template = dict(hypothesis.falsification_template)
    expected_template = {
        "detector": "batch_median_lift",
        "target_relation": "selected_batch",
        "comparator_relation": "pre_round_visible_observations",
        "operator": "greater",
        "threshold_source": "zero_lift",
        "min_observations": "selected_batch_size",
        "missing_data_policy": "INCONCLUSIVE",
        "reduction_policy": "primary_contradiction_first_v1",
    }
    if template != expected_template:
        raise ValueError("UNCOMPILABLE_FALSIFICATION_SPEC")
    if not target_variant_ids:
        raise ValueError("UNCOMPILABLE_FALSIFICATION_SPEC: target set is empty")
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
    return FalsificationSpec(
        spec_id=f"falsification:{hypothesis_id}:v1",
        hypothesis_id=hypothesis_id,
        version="1.0.0",
        registered_at_round=round_id,
        criteria=(criterion,),
        reduction_policy="primary_contradiction_first_v1",
        human_readable_description=hypothesis.falsification_criterion,
        compilation_receipt={
            "compiler_version": "falsification_template_compiler.v1",
            "template_detector": template["detector"],
            "compiled_detector": criterion.detector_name,
            "detector_version": criterion.detector_version,
            "target_relation": template["target_relation"],
            "target_variant_ids": list(criterion.target_variant_ids),
            "comparator_variant_ids": list(criterion.comparator_variant_ids),
            "support_threshold": criterion.support_threshold,
            "contradiction_threshold": criterion.contradiction_threshold,
            "min_observations": criterion.min_observations,
            "missing_data_policy": criterion.missing_data_policy,
            "target_intersection_count": len(target_variant_ids),
            "text_rendered_by_runtime": True,
            "equivalent": True,
        },
    )


def verify_falsification_spec(spec: FalsificationSpec) -> None:
    if not spec.hypothesis_id or not spec.criteria:
        raise PermissionError("FalsificationSpec is incomplete")
    if not spec.compilation_receipt or spec.compilation_receipt.get("equivalent") is not True:
        raise PermissionError("FalsificationSpec lacks a verified compilation receipt")
    primary = tuple(item for item in spec.criteria if item.primary)
    if len(primary) != 1:
        raise PermissionError("FalsificationSpec must contain exactly one compiled primary criterion")
    criterion = primary[0]
    expected = {
        "compiled_detector": criterion.detector_name,
        "detector_version": criterion.detector_version,
        "target_variant_ids": list(criterion.target_variant_ids),
        "comparator_variant_ids": list(criterion.comparator_variant_ids),
        "support_threshold": criterion.support_threshold,
        "contradiction_threshold": criterion.contradiction_threshold,
        "min_observations": criterion.min_observations,
        "missing_data_policy": criterion.missing_data_policy,
    }
    if any(spec.compilation_receipt.get(key) != value for key, value in expected.items()):
        raise PermissionError("FalsificationSpec no longer matches its compilation receipt")


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
        return HypothesisAssessment(
            assessment_id=f"AS{round_id:02d}",
            hypothesis_id=spec.hypothesis_id,
            falsification_spec_id=spec.spec_id,
            round_id=round_id,
            status=status,
            criterion_results=results,
            observation_ids=observation_ids,
            decisive_criterion_ids=decisive,
            unresolved_criterion_ids=unresolved,
            evaluator_version=self.version,
        )
