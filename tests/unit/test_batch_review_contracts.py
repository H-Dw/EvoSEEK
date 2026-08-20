from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from fitness_agents.agents.critic import _compact_critic_context
from fitness_agents.contracts.batch_review import (
    BatchReviewContext,
    CandidateIntentCard,
    PredictionReviewCard,
    batch_diversity_receipt,
    control_feasibility_receipt,
    prediction_review_card,
)
from fitness_agents.contracts.schemas import (
    CandidateIssue,
    CritiqueDecision,
    FalsificationReadiness,
    Hypothesis,
    IssueScope,
    IssueSeverity,
    Prediction,
    RequiredChange,
    RequiredChangeAction,
    ReviewVerdict,
    Variant,
)
from fitness_agents.loop.review import BoundedReviewLoop, ControlFeasibilityError
from fitness_agents.mutation import reserve_hypothesis_negative_controls
from fitness_agents.mutation.conflicts import SequenceConflictDetector
from fitness_agents.validation.batch import (
    BatchHardValidator,
    CritiqueDecisionValidator,
    build_draft_batch,
)


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


def test_hard_residue_issue_requires_explicit_hard_constraint_and_validator_conflict(
    experiment_config,
) -> None:
    variant = _variant("v1", "ADGV")
    prediction = _prediction("real-model:v1")
    prediction = replace(prediction, variant_id="v1")
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("v1",),
        variants={"v1": variant},
        predictions={"v1": prediction},
        evidence={},
        hypothesis_id="H01",
        falsification_spec=None,
    )
    soft = Hypothesis(
        hypothesis_id="H01",
        statement="Prefer A at position 39.",
        preferred_residues={39: ("A",)},
        evidence_ids=(),
        expected_outcome="Test the soft direction.",
        falsification_criterion="Reject if its matched comparison opposes it.",
    )
    hard = replace(soft, hard_residue_constraints={39: ("V",)})
    hard_validator = BatchHardValidator(experiment_config.task, experiment_config.critic)
    common = {
        "variants": {"v1": variant},
        "predictions": {"v1": prediction},
        "evidence": {},
        "revealed_ids": set(),
        "pending_ids": set(),
        "allowed_ids": {"v1"},
        "expected_batch_size": 1,
    }
    soft_report = hard_validator.validate(draft, hypothesis=soft, **common)
    hard_report = hard_validator.validate(draft, hypothesis=hard, **common)

    def decision(conflict_ids: tuple[str, ...]) -> CritiqueDecision:
        return CritiqueDecision(
            decision_id="D01-00",
            draft_batch_id=draft.draft_batch_id,
            round_id=1,
            review_attempt=0,
            verdict=ReviewVerdict.REJECT,
            falsification_readiness=FalsificationReadiness.UNTESTABLE,
            candidate_issues=(
                CandidateIssue(
                    issue_id="I01",
                    candidate_id="v1",
                    scope=IssueScope.RESIDUE,
                    severity=IssueSeverity.BLOCKER,
                    code="HARD_RESIDUE_CONSTRAINT_VIOLATION",
                    claim="The candidate violates an explicit hard residue constraint.",
                    conflict_ids=conflict_ids,
                ),
            ),
            confidence=1.0,
            summary="Reject the deterministic hard conflict.",
        )

    validator = CritiqueDecisionValidator()
    with pytest.raises(ValueError, match="forbidden when hard_residue_constraints is empty"):
        validator.validate(
            decision(()),
            draft=draft,
            report=soft_report,
            visible_evidence_ids=set(),
            hypothesis=soft,
        )
    with pytest.raises(ValueError, match="deterministic hard-conflict ID"):
        validator.validate(
            decision(("C-INVENTED",)),
            draft=draft,
            report=hard_report,
            visible_evidence_ids=set(),
            hypothesis=hard,
        )
    validator.validate(
        decision((hard_report.hard_conflicts[0].conflict_id,)),
        draft=draft,
        report=hard_report,
        visible_evidence_ids=set(),
        hypothesis=hard,
    )


def test_required_residue_map_cannot_be_derived_from_soft_preference(
    experiment_config,
) -> None:
    variant = _variant("v1", "ADGV")
    prediction = replace(_prediction("real-model:v1"), variant_id="v1")
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("v1",),
        variants={"v1": variant},
        predictions={"v1": prediction},
        evidence={},
        hypothesis_id="H01",
        falsification_spec=None,
    )
    hypothesis = Hypothesis(
        hypothesis_id="H01",
        statement="Prefer A at position 39.",
        preferred_residues={39: ("A",)},
        evidence_ids=(),
        expected_outcome="Test the soft direction.",
        falsification_criterion="Reject if its matched comparison opposes it.",
    )
    report = BatchHardValidator(
        experiment_config.task, experiment_config.critic
    ).validate(
        draft,
        variants={"v1": variant},
        predictions={"v1": prediction},
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids={"v1"},
        expected_batch_size=1,
        hypothesis=hypothesis,
    )
    decision = CritiqueDecision(
        decision_id="D01-00",
        draft_batch_id=draft.draft_batch_id,
        round_id=1,
        review_attempt=0,
        verdict=ReviewVerdict.REVISE,
        falsification_readiness=FalsificationReadiness.NEEDS_REVISION,
        required_changes=(
            RequiredChange(
                action=RequiredChangeAction.REPLACE_CANDIDATE,
                target_ids=("v1",),
                parameters={"required_residues_by_position": {"39": ["A"]}},
                rationale="Attempt to convert the soft preference into a hard rule.",
            ),
        ),
        confidence=0.8,
        summary="Revise.",
    )
    with pytest.raises(ValueError, match="cannot be derived from preferred_residues"):
        CritiqueDecisionValidator().validate(
            decision,
            draft=draft,
            report=report,
            visible_evidence_ids=set(),
            hypothesis=hypothesis,
        )


def test_matched_control_may_violate_soft_prior_but_cannot_be_excluded_for_it(
    experiment_config,
) -> None:
    variants = {"target": _variant("target", "ADGV"), "control": _variant("control", "VDGV")}
    predictions = {
        candidate_id: replace(_prediction("real-model:v1"), variant_id=candidate_id)
        for candidate_id in variants
    }
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("target", "control"),
        variants=variants,
        predictions=predictions,
        evidence={},
        hypothesis_id="H01",
        falsification_spec=None,
    )
    hypothesis = Hypothesis(
        hypothesis_id="H01",
        statement="Prefer A at position 39.",
        preferred_residues={39: ("A",)},
        evidence_ids=(),
        expected_outcome="Compare the target with a hypothesis-negative control.",
        falsification_criterion="Reject if the target does not separate from the control.",
    )
    report = BatchHardValidator(
        experiment_config.task, experiment_config.critic
    ).validate(
        draft,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=2,
        hypothesis=hypothesis,
    )
    context = BatchReviewContext(
        prediction_status_by_id={
            candidate_id: prediction_review_card(
                prediction,
                source_kind="real_model",
                decision_eligible=True,
                calibration_status="calibrated",
            )
            for candidate_id, prediction in predictions.items()
        },
        candidate_intent_by_id={
            "target": CandidateIntentCard(candidate_id="target", arm="hypothesis_target"),
            "control": CandidateIntentCard(
                candidate_id="control",
                arm="matched_control",
                matched_to="target",
                allow_hypothesis_mismatch=True,
            ),
        },
        soft_prior_mismatch_ids=("control",),
    )
    validator = CritiqueDecisionValidator()
    validator.validate(
        CritiqueDecision(
            decision_id="D01-00",
            draft_batch_id=draft.draft_batch_id,
            round_id=1,
            review_attempt=0,
            verdict=ReviewVerdict.APPROVE,
            falsification_readiness=FalsificationReadiness.READY,
            confidence=0.9,
            summary="Keep the matched control despite the intentional soft-prior mismatch.",
        ),
        draft=draft,
        report=report,
        visible_evidence_ids=set(),
        hypothesis=hypothesis,
        batch_review_context=context,
    )
    exclude = CritiqueDecision(
        decision_id="D01-00",
        draft_batch_id=draft.draft_batch_id,
        round_id=1,
        review_attempt=0,
        verdict=ReviewVerdict.REVISE,
        falsification_readiness=FalsificationReadiness.NEEDS_REVISION,
        required_changes=(
            RequiredChange(
                action=RequiredChangeAction.EXCLUDE_CANDIDATE,
                target_ids=("control",),
                parameters={},
                rationale="The control does not match preferred_residues.",
            ),
        ),
        confidence=0.8,
        summary="Exclude the soft-prior mismatch.",
    )
    with pytest.raises(ValueError, match="soft prior mismatch cannot trigger exclusion"):
        validator.validate(
            exclude,
            draft=draft,
            report=report,
            visible_evidence_ids=set(),
            hypothesis=hypothesis,
            batch_review_context=context,
        )
    fallback_context = context.model_copy(
        update={
            "candidate_intent_by_id": {
                "target": CandidateIntentCard(candidate_id="target", arm="hypothesis_target"),
                "control": CandidateIntentCard(candidate_id="control", arm="fallback"),
            }
        }
    )
    with pytest.raises(ValueError, match="soft prior mismatch alone cannot trigger"):
        validator.validate(
            exclude,
            draft=draft,
            report=report,
            visible_evidence_ids=set(),
            hypothesis=hypothesis,
            batch_review_context=fallback_context,
        )


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
