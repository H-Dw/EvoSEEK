from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from fitness_agents.agents.critic import _compact_critic_context
from fitness_agents.contracts.batch_review import (
    BatchReviewContext,
    PredictionReviewCard,
    batch_diversity_receipt,
    control_feasibility_receipt,
    prediction_review_card,
)
from fitness_agents.contracts.schemas import Hypothesis, Prediction, Variant
from fitness_agents.loop.review import BoundedReviewLoop, ControlFeasibilityError
from fitness_agents.mutation import reserve_hypothesis_negative_controls
from fitness_agents.mutation.conflicts import SequenceConflictDetector
from fitness_agents.validation.batch import BatchHardValidator, build_draft_batch


def _variant(variant_id: str, code: str) -> Variant:
    notation = ";".join(
        f"{source}{position}{target}"
        for source, position, target in zip("VDGV", (39, 40, 41, 54), code, strict=True)
        if source != target
    ) or "WT"
    return Variant(
        variant_id=variant_id,
        variant=code,
        sequence=code,
        mutation_notation=notation,
        mutation_count=sum(a != b for a, b in zip(code, "VDGV", strict=True)),
        split_role="oracle_pool",
    )


def _prediction(model_version: str = "placeholder-canary-sha256-seed11") -> Prediction:
    return Prediction(
        variant_id="v1",
        fitness_mean=9.9,
        fitness_std=8.8,
        interval_90=(-1.0, 1.0),
        ood_score=7.7,
        component_scores={"a": 1.0, "b": 4.0},
        model_version=model_version,
    )


def test_placeholder_prediction_card_hides_all_numeric_values_but_keeps_identity() -> None:
    card = prediction_review_card(
        _prediction(),
        source_kind="placeholder",
        decision_eligible=False,
        calibration_status="not_applicable",
    )
    assert card.prediction_status == "not_evaluated"
    assert card.model_version == "placeholder-canary-sha256-seed11"
    assert card.fitness_mean is None
    assert card.fitness_std is None
    assert card.ood_score is None
    assert card.model_disagreement is None
    with pytest.raises(ValidationError):
        PredictionReviewCard(
            variant_id="v1",
            source_kind="placeholder",
            decision_eligible=True,
            calibration_status="unknown",
            model_version="placeholder",
            prediction_status="evaluated",
            fitness_mean=1.0,
            fitness_std=0.1,
            ood_score=0.2,
            model_disagreement=0.0,
        )


def test_placeholder_prediction_cannot_trigger_model_risk_conflicts() -> None:
    detector = SequenceConflictDetector(
        ood_warning_threshold=0.1,
        model_disagreement_threshold=0.1,
    )
    variant = _variant("v1", "ADGV")
    common = {
        "predictions": {"v1": _prediction()},
        "evidence": {},
        "revealed_ids": set(),
        "pending_ids": set(),
        "allowed_ids": {"v1"},
        "expected_batch_size": 1,
    }

    hidden = detector.detect(
        [variant],
        **common,
        prediction_decision_eligible={"v1": False},
    )
    visible = detector.detect(
        [variant],
        **common,
        prediction_decision_eligible={"v1": True},
    )

    assert not {"HIGH_OOD", "MODEL_DISAGREEMENT"}.intersection(
        item.code for item in hidden
    )
    assert {"HIGH_OOD", "MODEL_DISAGREEMENT"}.issubset(
        item.code for item in visible
    )


def test_compact_critic_context_prefers_typed_prediction_status() -> None:
    card = prediction_review_card(
        _prediction(),
        source_kind="placeholder",
        decision_eligible=False,
        calibration_status="not_applicable",
    )
    context = BatchReviewContext(prediction_status_by_id={"v1": card})
    compact = _compact_critic_context(
        {"predictions": {"v1": _prediction()}, "batch_review_context": context}
    )
    assert compact["predictions"]["v1"]["prediction_status"] == "not_evaluated"
    assert "fitness_mean" not in compact["predictions"]["v1"]
    assert compact["predictions"]["v1"]["model_version"].startswith("placeholder")


def test_control_and_diversity_receipts_distinguish_impossible_from_unsatisfied() -> None:
    empty = control_feasibility_receipt(
        requested_controls=2,
        available_control_ids=(),
        selected_control_ids=(),
    )
    assert empty.reason == "CONTROL_UNIVERSE_EMPTY"
    assert not empty.feasible

    variants = {
        item.variant_id: item
        for item in (
            _variant("v1", "ADGV"),
            _variant("v2", "VAGV"),
            _variant("v3", "VDAV"),
        )
    }
    receipt = batch_diversity_receipt(
        selected_ids=("v1", "v2"),
        candidate_pool_ids=tuple(variants),
        variants_by_id=variants,
        required_minimum_batch_distance=3,
        hypothesis=None,
        position_to_index={39: 0, 40: 1, 41: 2, 54: 3},
    )
    assert receipt.selected.minimum_pairwise_hamming == 2
    assert receipt.threshold_satisfied is False
    assert receipt.threshold_feasible_in_pool is False


def test_disabled_batch_review_scope_omits_control_and_diversity_receipts() -> None:
    context = BatchReviewContext(
        prediction_status_by_id={},
        review_controls=False,
        review_diversity=False,
    )
    assert context.control_feasibility is None
    assert context.diversity is None
    with pytest.raises(ValidationError, match="control_feasibility must be omitted"):
        BatchReviewContext(
            prediction_status_by_id={},
            review_controls=False,
            control_feasibility=control_feasibility_receipt(
                requested_controls=2,
                available_control_ids=(),
                selected_control_ids=(),
            ),
        )


def test_disabled_diversity_review_removes_validator_distance_warning(
    experiment_config,
) -> None:
    disabled = replace(
        experiment_config.critic,
        review_diversity=False,
        min_batch_distance=4,
    )
    validator = BatchHardValidator(experiment_config.task, disabled)
    variants = {"v1": _variant("v1", "ADGV"), "v2": _variant("v2", "AAGV")}
    predictions = {
        variant_id: replace(_prediction("real-model:v1"), variant_id=variant_id)
        for variant_id in variants
    }
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=tuple(variants),
        variants=variants,
        predictions=predictions,
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = validator.validate(
        draft,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=2,
    )
    assert "BATCH_MODE_COLLAPSE" not in {item.code for item in report.conflicts}


def test_only_explicit_hard_residue_constraints_block_a_candidate(
    experiment_config,
) -> None:
    variant = _variant("v1", "ADGV")
    prediction = _prediction("real-model:v1")
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("v1",),
        variants={"v1": variant},
        predictions={"v1": prediction},
        evidence={},
        hypothesis_id="hyp:1",
        falsification_spec=None,
    )
    soft = Hypothesis(
        hypothesis_id="hyp:1",
        statement="Prefer A at position 39.",
        preferred_residues={39: ("A",)},
        evidence_ids=(),
        expected_outcome="Test the preference.",
        falsification_criterion="Reject if the comparison opposes it.",
    )
    validator = BatchHardValidator(experiment_config.task, experiment_config.critic)
    common = {
        "variants": {"v1": variant},
        "predictions": {"v1": prediction},
        "evidence": {},
        "revealed_ids": set(),
        "pending_ids": set(),
        "allowed_ids": {"v1"},
        "expected_batch_size": 1,
    }

    soft_report = validator.validate(draft, hypothesis=soft, **common)
    hard_report = validator.validate(
        draft,
        hypothesis=replace(soft, hard_residue_constraints={39: ("V",)}),
        **common,
    )

    assert "HARD_RESIDUE_CONSTRAINT_VIOLATION" not in {
        item.code for item in soft_report.conflicts
    }
    assert "HARD_RESIDUE_CONSTRAINT_VIOLATION" in {
        item.code for item in hard_report.hard_conflicts
    }


def test_control_feasibility_fails_before_critic_call(experiment_config) -> None:
    class NeverCalledCritic:
        calls = 0

        def review(self, **_kwargs):
            self.calls += 1
            raise AssertionError("control gate must run before the Critic")

    variant = _variant("v1", "ADGV")
    prediction = _prediction(model_version="real-model:v1")
    critic = NeverCalledCritic()
    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=critic,
        max_revision_attempts=1,
    )

    def builder(attempt, parent_id, exclusions, constraints=None):
        del exclusions, constraints
        return build_draft_batch(
            round_id=1,
            review_attempt=attempt,
            candidate_ids=("v1",),
            variants={"v1": variant},
            predictions={"v1": prediction},
            evidence={},
            hypothesis_id=None,
            falsification_spec=None,
            parent_draft_batch_id=parent_id,
        )

    context = BatchReviewContext(
        prediction_status_by_id={
            "v1": prediction_review_card(
                prediction,
                source_kind="real_model",
                decision_eligible=True,
                calibration_status="calibrated",
            )
        },
        control_feasibility=control_feasibility_receipt(
            requested_controls=2,
            available_control_ids=(),
            selected_control_ids=(),
        ),
    )
    with pytest.raises(ControlFeasibilityError, match="CONTROL_UNIVERSE_EMPTY"):
        loop.run(
            draft_builder=builder,
            variants={"v1": variant},
            predictions={"v1": prediction},
            evidence={},
            revealed_ids=set(),
            pending_ids=set(),
            allowed_ids={"v1"},
            expected_batch_size=1,
            review_context_provider=lambda _draft: context,
        )
    assert critic.calls == 0


def test_truncated_strong_pool_reserves_hypothesis_negative_matched_controls() -> None:
    hypothesis = Hypothesis(
        hypothesis_id="hyp:1",
        statement="Prefer A at position 39.",
        preferred_residues={39: ("A",)},
        evidence_ids=(),
        expected_outcome="Test.",
        falsification_criterion="Reject if controls match targets.",
    )
    strong = [_variant(f"strong:{index}", f"A{residue}GV") for index, residue in enumerate("ACDE")]
    controls = [_variant(f"control:{index}", f"V{residue}GV") for index, residue in enumerate("ACDE")]
    reserved = reserve_hypothesis_negative_controls(
        strong,
        [*strong, *controls],
        hypothesis=hypothesis,
        position_to_index={39: 0, 40: 1, 41: 2, 54: 3},
        strong_threshold=0.75,
        required_controls=2,
        candidate_limit=len(strong),
        reserve_multiplier=1,
    )
    assert len(reserved) == len(strong)
    assert len([item for item in reserved if item.variant_id.startswith("control:")]) >= 2
