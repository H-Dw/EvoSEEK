from __future__ import annotations

from dataclasses import replace

import pytest

from fitness_agents.agents.critic import (
    CriticAgent,
    RuleBasedCriticClient,
    _compact_critic_context,
    load_critic_profile,
)
from fitness_agents.contracts.schemas import (
    CritiqueDecision,
    Evidence,
    FalsificationReadiness,
    FitnessObservation,
    Hypothesis,
    HypothesisStatus,
    Prediction,
    RequiredChange,
    RequiredChangeAction,
    ReviewVerdict,
    Variant,
)
from fitness_agents.evaluation.hypotheses import (
    DeterministicHypothesisEvaluator,
    preregister_batch_median_test,
)
from fitness_agents.loop.backends import ApprovalEnforcingBackend
from fitness_agents.loop.review import BoundedReviewLoop
from fitness_agents.mutation.conflicts import detect_pairwise_epistasis
from fitness_agents.validation.batch import BatchHardValidator, build_draft_batch


def _variant(code: str, variant_id: str) -> Variant:
    positions = (39, 40, 41, 54)
    notation = (
        ";".join(
            f"{source}{position}{target}"
            for source, position, target in zip("VDGV", positions, code, strict=True)
            if source != target
        )
        or "WT"
    )
    return Variant(
        variant_id=variant_id,
        variant=code,
        sequence=code,
        mutation_notation=notation,
        mutation_count=sum(a != b for a, b in zip(code, "VDGV", strict=True)),
        split_role="oracle_pool",
    )


def _prediction(variant_id: str, mean: float) -> Prediction:
    return Prediction(
        variant_id=variant_id,
        fitness_mean=mean,
        fitness_std=0.1,
        interval_90=(mean - 0.2, mean + 0.2),
        ood_score=0.1,
        component_scores={"model_a": mean, "model_b": mean + 0.01},
        model_version="test",
    )


class _ReviseThenApprove:
    provider_name = "scripted"

    def __init__(self) -> None:
        self.calls = 0
        self.rule = RuleBasedCriticClient()

    def review(self, *, context, output_schema):
        self.calls += 1
        if self.calls > 1:
            return self.rule.review(context=context, output_schema=output_schema)
        draft = context["draft"]
        return CritiqueDecision(
            decision_id="decision:revise",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.REVISE,
            falsification_readiness=FalsificationReadiness.READY,
            required_changes=(
                RequiredChange(
                    action=RequiredChangeAction.EXCLUDE_CANDIDATE,
                    target_ids=(draft.candidate_ids[0],),
                    parameters={},
                    rationale="Use the alternate candidate to resolve the injected review concern.",
                ),
            ),
            confidence=0.9,
            summary="Replace the first candidate.",
        )


class _AlwaysApprove:
    provider_name = "unsafe_test_critic"

    def review(self, *, context, output_schema):
        draft = context["draft"]
        return CritiqueDecision(
            decision_id="decision:unsafe-approve",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.APPROVE,
            falsification_readiness=FalsificationReadiness.READY,
            confidence=1.0,
            summary="Approve regardless of deterministic validation.",
        )


def test_bounded_revision_changes_batch_and_creates_approval(experiment_config):
    variants = {"a": _variant("VDGA", "a"), "b": _variant("VDGL", "b")}
    predictions = {"a": _prediction("a", 1.0), "b": _prediction("b", 0.8)}

    def builder(attempt, parent_id, exclusions):
        candidate = next(item for item in ("a", "b") if item not in exclusions)
        return build_draft_batch(
            round_id=1,
            review_attempt=attempt,
            candidate_ids=(candidate,),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id=None,
            falsification_spec=None,
            parent_draft_batch_id=parent_id,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(_ReviseThenApprove(), max_retries=0),
        max_revision_attempts=1,
    )
    result = loop.run(
        draft_builder=builder,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=1,
    )

    assert [item.verdict for item in result.attempts] == [
        ReviewVerdict.REVISE,
        ReviewVerdict.APPROVE,
    ]
    assert result.approved_batch.candidate_ids == ("b",)
    assert result.draft.parent_draft_batch_id is not None

    class RawBackend:
        def submit(self, candidate_ids, round_id):
            assert tuple(candidate_ids) == ("b",)
            assert round_id == 1
            return "experiment:approved"

    backend = ApprovalEnforcingBackend(RawBackend(), loop.gateway)
    assert backend.submit(result.approved_batch) == "experiment:approved"
    with pytest.raises(PermissionError, match="modified"):
        backend.submit(replace(result.approved_batch, candidate_ids=("a",)))


def test_review_loop_notifies_start_before_critic(experiment_config):
    variants = {"a": _variant("VDGA", "a")}
    predictions = {"a": _prediction("a", 1.0)}
    order: list[str] = []

    class _OrderingCritic:
        provider_name = "ordering"

        def review(self, *, context, output_schema):
            order.append("review")
            return RuleBasedCriticClient().review(context=context, output_schema=output_schema)

    def builder(attempt, parent_id, exclusions):
        del attempt, parent_id, exclusions
        return build_draft_batch(
            round_id=1,
            review_attempt=0,
            candidate_ids=("a",),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id=None,
            falsification_spec=None,
            parent_draft_batch_id=None,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(_OrderingCritic(), max_retries=0),
        max_revision_attempts=0,
    )
    loop.run(
        draft_builder=builder,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=1,
        on_attempt_start=lambda draft, report: order.append("start"),
        on_attempt=lambda draft, report, decision: order.append("attempt"),
    )
    assert order == ["start", "review", "attempt"]


def test_approval_backend_rejects_modified_receipt():
    class RawBackend:
        def __init__(self):
            self.calls = 0

        def submit(self, candidate_ids, round_id):
            self.calls += 1
            return "run"

    raw = RawBackend()
    backend = ApprovalEnforcingBackend(raw)
    with pytest.raises(TypeError, match="ApprovedBatch"):
        backend.submit(["a"])


def test_llm_approve_cannot_override_residue_hard_conflict(experiment_config):
    invalid = replace(
        _variant("ADGV", "invalid"),
        mutation_notation="X39A;V39F",
    )
    variants = {invalid.variant_id: invalid}
    predictions = {invalid.variant_id: _prediction(invalid.variant_id, 10.0)}
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=(invalid.variant_id,),
        variants=variants,
        predictions=predictions,
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = BatchHardValidator(experiment_config.task, experiment_config.critic).validate(
        draft,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=1,
    )
    codes = {item.code for item in report.hard_conflicts}
    assert {"FROM_RESIDUE_MISMATCH", "MULTIPLE_EDITS_SAME_POSITION"} <= codes
    with pytest.raises(RuntimeError, match="Critic failed"):
        CriticAgent(_AlwaysApprove(), max_retries=0).review(
            draft=draft,
            variants=variants,
            predictions=predictions,
            evidence={},
            conflict_report=report,
        )


def test_joint_epistasis_requires_all_constituents_and_detects_sign_change():
    unknown = detect_pairwise_epistasis(
        fitness_scale="raw_assay",
        wt_samples=[0.0, 0.0],
        single_a_samples=[1.0, 1.0],
        single_b_samples=None,
        double_samples=[-1.0, -1.0],
    )
    assert unknown.status == "UNKNOWN"
    result = detect_pairwise_epistasis(
        fitness_scale="raw_assay",
        wt_samples=[0.0, 0.0, 0.0],
        single_a_samples=[1.0, 1.0, 1.0],
        single_b_samples=[1.0, 1.0, 1.0],
        double_samples=[-1.0, -1.0, -1.0],
    )
    assert result.sign_epistasis is True
    assert result.reciprocal_sign_epistasis is True


@pytest.mark.parametrize(
    ("target_fitness", "expected"),
    [(1.0, HypothesisStatus.SUPPORTED), (-1.0, HypothesisStatus.CONTRADICTED)],
)
def test_falsification_status_is_computed_after_observation(target_fitness, expected):
    baseline = FitnessObservation("base", 0.0, "initial_observed", 0)
    spec = preregister_batch_median_test(
        hypothesis_id="hyp:1",
        round_id=1,
        target_variant_ids=("target",),
        visible_observations=(baseline,),
    )
    assessment = DeterministicHypothesisEvaluator().evaluate(
        spec=spec,
        observations=(baseline, FitnessObservation("target", target_fitness, "oracle_pool", 1)),
        round_id=1,
    )
    assert assessment.status is expected


def test_falsification_preregistration_cannot_be_changed_after_review():
    baseline = FitnessObservation("base", 0.0, "initial_observed", 0)
    spec = preregister_batch_median_test(
        hypothesis_id="hyp:locked",
        round_id=1,
        target_variant_ids=("target",),
        visible_observations=(baseline,),
    )
    changed = replace(
        spec,
        criteria=(replace(spec.criteria[0], support_threshold=-100.0),),
    )
    with pytest.raises(PermissionError, match="changed after preregistration"):
        DeterministicHypothesisEvaluator().evaluate(
            spec=changed,
            observations=(baseline, FitnessObservation("target", 1.0, "oracle_pool", 1)),
            round_id=1,
        )


def test_scientific_critic_skill_is_structured_english():
    profile = load_critic_profile("scientific_v1")
    assert "## 3. Activation-state routing" in profile
    assert "## 4. Review lenses" in profile
    assert "## 7. Output contract" in profile
    assert "supported or contradicted before results exist" in profile
    assert "REGENERATE_WITH_CONSTRAINTS" in profile
    assert "ADD_CONTROL" in profile
    assert "support`, `mixed`, `oppose`, or" in profile
    assert "open_design" in profile
    assert "executed_kg_tools" in profile
    assert "active_learning" in profile
    for forbidden_prior in ("AAIndex", "Neff", "SASA", "salt-bridge", "hydropathy"):
        assert forbidden_prior not in profile
    assert "physicochemical context" in profile
    assert "evolutionary context" in profile
    assert "structural context" in profile
    assert not any("\u4e00" <= character <= "\u9fff" for character in profile)


class _AddControlThenApprove:
    provider_name = "add_control"

    def __init__(self) -> None:
        self.calls = 0
        self.rule = RuleBasedCriticClient()

    def review(self, *, context, output_schema):
        self.calls += 1
        if self.calls > 1:
            return self.rule.review(context=context, output_schema=output_schema)
        draft = context["draft"]
        return CritiqueDecision(
            decision_id="decision:add-control",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.REVISE,
            falsification_readiness=FalsificationReadiness.READY,
            required_changes=(
                RequiredChange(
                    action=RequiredChangeAction.ADD_CONTROL,
                    target_ids=(),
                    parameters={},
                    rationale="Add a wild-type or single-mutation control.",
                ),
            ),
            confidence=0.8,
            summary="Add controls.",
        )


class _RegenerateHypothesis:
    provider_name = "regenerate"

    def review(self, *, context, output_schema):
        draft = context["draft"]
        return CritiqueDecision(
            decision_id="decision:regenerate",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.REVISE,
            falsification_readiness=FalsificationReadiness.READY,
            required_changes=(
                RequiredChange(
                    action=RequiredChangeAction.REGENERATE_WITH_CONSTRAINTS,
                    target_ids=(),
                    parameters={},
                    rationale="The hypothesis must change before another batch is drafted.",
                ),
            ),
            confidence=0.8,
            summary="Regenerate the hypothesis.",
        )


def test_add_control_revise_rebuilds_batch_instead_of_aborting(experiment_config):
    from fitness_agents.agents.output_guards import RevisionConstraints
    from fitness_agents.loop.review import BoundedReviewLoop

    variants = {"a": _variant("VDGA", "a"), "b": _variant("VDGL", "b")}
    predictions = {"a": _prediction("a", 1.0), "b": _prediction("b", 0.8)}
    seen: list[object] = []

    def builder(attempt, parent_id, exclusions, constraints=None):
        seen.append(constraints)
        candidate = "a" if attempt == 0 else "b"
        del exclusions
        return build_draft_batch(
            round_id=1,
            review_attempt=attempt,
            candidate_ids=(candidate,),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id=None,
            falsification_spec=None,
            parent_draft_batch_id=parent_id,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(_AddControlThenApprove(), max_retries=0),
        max_revision_attempts=1,
    )
    result = loop.run(
        draft_builder=builder,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=1,
    )
    assert result.approved_batch.candidate_ids == ("b",)
    assert any(isinstance(item, RevisionConstraints) and item.require_controls for item in seen[1:])


def test_regenerate_with_constraints_requests_new_hypothesis(experiment_config):
    from fitness_agents.loop.review import BoundedReviewLoop, HypothesisRevisionRequested

    variants = {"a": _variant("VDGA", "a")}
    predictions = {"a": _prediction("a", 1.0)}

    def builder(attempt, parent_id, exclusions, constraints=None):
        del attempt, parent_id, exclusions, constraints
        return build_draft_batch(
            round_id=1,
            review_attempt=0,
            candidate_ids=("a",),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id="hyp:run:r1",
            falsification_spec=None,
            parent_draft_batch_id=None,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(_RegenerateHypothesis(), max_retries=0),
        max_revision_attempts=1,
    )
    with pytest.raises(HypothesisRevisionRequested, match="new hypothesis"):
        loop.run(
            draft_builder=builder,
            variants=variants,
            predictions=predictions,
            evidence={},
            revealed_ids=set(),
            pending_ids=set(),
            allowed_ids=set(variants),
            expected_batch_size=1,
        )


def test_review_loop_passes_hypothesis_snapshot(experiment_config):
    captured: dict[str, object] = {}
    hypothesis = Hypothesis(
        hypothesis_id="hyp:run:r1",
        statement="Prefer aromatics at the mutable sites.",
        preferred_residues={39: ("W",), 40: ("D",), 41: ("G",), 54: ("V",)},
        evidence_ids=("ev:1",),
        expected_outcome="Enrichment relative to random selection.",
        falsification_criterion="Revise if the wet batch median does not improve.",
    )

    class _CaptureHypothesis:
        provider_name = "capture"

        def review(self, *, context, output_schema):
            del output_schema
            captured["hypothesis"] = context.get("hypothesis")
            draft = context["draft"]
            return CritiqueDecision(
                decision_id="decision:capture",
                draft_batch_id=draft.draft_batch_id,
                round_id=draft.round_id,
                review_attempt=draft.review_attempt,
                verdict=ReviewVerdict.APPROVE,
                falsification_readiness=FalsificationReadiness.READY,
                confidence=1.0,
                summary="Approve after capturing the hypothesis snapshot.",
            )

    variants = {"a": _variant("VDGA", "a")}
    predictions = {"a": _prediction("a", 1.0)}

    def builder(attempt, parent_id, exclusions):
        del attempt, parent_id, exclusions
        return build_draft_batch(
            round_id=1,
            review_attempt=0,
            candidate_ids=("a",),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id="hyp:run:r1",
            falsification_spec=None,
            parent_draft_batch_id=None,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(_CaptureHypothesis(), max_retries=0),
        max_revision_attempts=0,
    )
    loop.run(
        draft_builder=builder,
        variants=variants,
        predictions=predictions,
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids=set(variants),
        expected_batch_size=1,
        hypothesis=hypothesis,
    )
    snapshot = captured["hypothesis"]
    assert isinstance(snapshot, dict)
    assert snapshot["hypothesis_id"] == "hyp:run:r1"
    assert snapshot["preferred_residues"]["39"] == ["W"]


def test_compact_critic_context_keeps_channel_sites():
    evidence = Evidence(
        evidence_id="ev:struct:1",
        variant_id="a",
        channel="structure",
        statement="static environment, not a folding claim",
        score=-1.0,
        source_id="structure:test",
        confidence=0.0,
        round_id=1,
        raw_features={
            "sites": {"39": {"contact_count": 4, "sasa": 12.0}},
            "static_context_flag_count": 1,
            "cache_status": "hit",
            "property_accessions": ["unused"],
        },
    )
    compacted = _compact_critic_context(
        {"evidence": {"a": [evidence]}, "context_evidence": [evidence]}
    )
    features = compacted["evidence"]["a"][0]["raw_features"]
    assert features["sites"]["39"]["contact_count"] == 4
    assert "cache_status" not in features
    assert "property_accessions" not in features
    assert compacted["context_evidence"][0]["raw_features"]["sites"]
