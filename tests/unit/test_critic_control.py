from __future__ import annotations

from dataclasses import replace

import pytest

from fitness_agents.agents.critic import (
    CriticAgent,
    RuleBasedCriticClient,
    _compact_critic_context,
    load_critic_profile,
)
from fitness_agents.contracts.batch_review import BatchReviewContext
from fitness_agents.contracts.schemas import (
    ConflictReport,
    CritiqueDecision,
    Evidence,
    FalsificationReadiness,
    FitnessObservation,
    Hypothesis,
    HypothesisStatus,
    IssueScope,
    IssueSeverity,
    MutationConflict,
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
from fitness_agents.validation.batch import (
    BatchHardValidator,
    CritiqueDecisionValidator,
    build_draft_batch,
)


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


def _compiled_hypothesis(hypothesis_id: str) -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        statement="Test a bounded directional prior.",
        preferred_residues={39: ("V",)},
        evidence_ids=(),
        expected_outcome="The selected batch exceeds the pre-round comparator.",
        falsification_criterion=(
            "The selected batch median must exceed the preregistered pre-round "
            "visible-observation median; missing required observations yield INCONCLUSIVE."
        ),
        falsification_template={
            "detector": "batch_median_lift",
            "target_relation": "selected_batch",
            "comparator_relation": "pre_round_visible_observations",
            "operator": "greater",
            "threshold_source": "zero_lift",
            "min_observations": "selected_batch_size",
            "missing_data_policy": "INCONCLUSIVE",
            "reduction_policy": "primary_contradiction_first_v1",
        },
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


def _channel_evidence(evidence_id: str, variant_id: str, channel: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        variant_id=variant_id,
        channel=channel,
        statement=f"{channel} context for {variant_id}",
        score=0.1,
        source_id=f"{channel}:test",
        confidence=0.5,
        round_id=1,
    )


def _multi_channel_evidence(candidate_ids: tuple[str, ...]) -> dict[str, list[Evidence]]:
    channels = ("physchem", "conservation", "structure", "kg")
    return {
        candidate_id: [
            _channel_evidence(f"E:{channel}:{candidate_id}", candidate_id, channel)
            for channel in channels
        ]
        for candidate_id in candidate_ids
    }


def _empty_report(draft) -> ConflictReport:
    return ConflictReport(
        report_id="report:test",
        round_id=draft.round_id,
        conflicts=(),
        validator_version="test",
        draft_batch_id=draft.draft_batch_id,
    )


def test_rule_critic_caps_cited_evidence_for_large_batches(experiment_config) -> None:
    candidate_ids = tuple(f"c{index}" for index in range(8))
    variants = {item: _variant("VDGA", item) for item in candidate_ids}
    predictions = {item: _prediction(item, 1.0) for item in candidate_ids}
    evidence = _multi_channel_evidence(candidate_ids)
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=candidate_ids,
        variants=variants,
        predictions=predictions,
        evidence=evidence,
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = _empty_report(draft)
    decision = RuleBasedCriticClient().review(
        context={"draft": draft, "conflict_report": report, "evidence": evidence},
        output_schema={},
    )
    assert decision.verdict is ReviewVerdict.APPROVE
    assert len(decision.cited_evidence_ids) == 16
    assert "caps applied" in decision.summary
    visible = {entry.evidence_id for entries in evidence.values() for entry in entries}
    CritiqueDecisionValidator().validate(
        decision,
        draft=draft,
        report=report,
        visible_evidence_ids=visible,
    )


def test_rule_critic_caps_issue_sections_with_many_hard_conflicts(experiment_config) -> None:
    candidate_ids = tuple(f"c{index}" for index in range(12))
    variants = {item: _variant("VDGA", item) for item in candidate_ids}
    predictions = {item: _prediction(item, 1.0) for item in candidate_ids}
    conflicts = tuple(
        MutationConflict(
            conflict_id=f"conflict:{index:02d}",
            code="UNKNOWN_CANDIDATE",
            scope=IssueScope.SEQUENCE,
            severity=IssueSeverity.BLOCKER,
            message=f"Synthetic hard conflict {index}",
            candidate_ids=(candidate_ids[index],),
            evidence_ids=(f"ev:{index:02d}",),
            hard=True,
            detector="test",
        )
        for index in range(12)
    )
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=candidate_ids,
        variants=variants,
        predictions=predictions,
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = ConflictReport(
        report_id="report:test",
        round_id=1,
        conflicts=conflicts,
        validator_version="test",
        draft_batch_id=draft.draft_batch_id,
    )
    decision = RuleBasedCriticClient().review(
        context={"draft": draft, "conflict_report": report, "evidence": {}},
        output_schema={},
    )
    assert decision.verdict is ReviewVerdict.REJECT
    assert len(decision.candidate_issues) == 8
    assert "caps applied" in decision.summary
    CritiqueDecisionValidator().validate(
        decision,
        draft=draft,
        report=report,
        visible_evidence_ids={f"ev:{index:02d}" for index in range(12)},
    )


class _BrokenRuleClient(RuleBasedCriticClient):
    provider_name = "broken_rule"

    def review(self, *, context, output_schema, validator=None):
        raise ValueError("synthetic rule failure")


class _BrokenRemoteClient:
    provider_name = "broken_remote"

    def review(self, *, context, output_schema, validator=None):
        raise ConnectionError("remote down")


def _single_candidate_review_kwargs():
    variants = {"a": _variant("VDGA", "a")}
    predictions = {"a": _prediction("a", 1.0)}
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("a",),
        variants=variants,
        predictions=predictions,
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    return {
        "draft": draft,
        "variants": variants,
        "predictions": predictions,
        "evidence": {},
        "conflict_report": _empty_report(draft),
    }


def test_rule_client_failure_closes_with_terminal_abort_decision(experiment_config) -> None:
    decision = CriticAgent(_BrokenRuleClient(), max_retries=0).review(
        **_single_candidate_review_kwargs()
    )
    assert decision.verdict is ReviewVerdict.REJECT
    assert decision.required_changes[0].action is RequiredChangeAction.ABORT_ROUND
    assert "synthetic rule failure" in decision.summary


def test_broken_rule_fallback_still_closes_with_terminal_abort(experiment_config) -> None:
    agent = CriticAgent(
        _BrokenRemoteClient(), max_retries=0, fallback=_BrokenRuleClient()
    )
    decision = agent.review(**_single_candidate_review_kwargs())
    assert agent.fallback_count == 1
    assert decision.verdict is ReviewVerdict.REJECT
    assert decision.required_changes[0].action is RequiredChangeAction.ABORT_ROUND
    assert "synthetic rule failure" in decision.summary


def test_remote_failure_without_fallback_raises_with_cause(experiment_config) -> None:
    with pytest.raises(RuntimeError, match=r"without a configured safe fallback \(ConnectionError"):
        CriticAgent(_BrokenRemoteClient(), max_retries=0).review(
            **_single_candidate_review_kwargs()
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


def test_soft_prior_cannot_create_required_residue_revision(
    experiment_config,
) -> None:
    variants = {
        "a": _variant("VDGA", "a"),
        "a2": _variant("VDGA", "a2"),
    }
    predictions = {
        variant_id: _prediction(variant_id, 1.0)
        for variant_id in variants
    }
    class _ExcludeResidueOnce:
        provider_name = "exclude_residue_once"

        def __init__(self) -> None:
            self.calls = 0

        def review(self, *, context, output_schema):
            del output_schema
            self.calls += 1
            if self.calls != 1:
                raise AssertionError("postcondition must fail before a second Critic call")
            draft = context["draft"]
            return CritiqueDecision(
                decision_id="decision:exclude-v54a",
                draft_batch_id=draft.draft_batch_id,
                round_id=draft.round_id,
                review_attempt=draft.review_attempt,
                verdict=ReviewVerdict.REVISE,
                falsification_readiness=FalsificationReadiness.READY,
                required_changes=(
                    RequiredChange(
                        action=RequiredChangeAction.REPLACE_CANDIDATE,
                        target_ids=("a",),
                        parameters={
                            "excluded_substitutions": [
                                {"position": 54, "from_residue": "V", "to_residue": "A"}
                            ],
                            "required_residues_by_position": {"54": ["V", "L"]},
                            "applies_to_arms": ["fallback"],
                        },
                        rationale="Exclude the V54A substitution from the revised batch.",
                    ),
                ),
                confidence=0.9,
                summary="Replace V54A.",
            )

    client = _ExcludeResidueOnce()

    def builder(
        attempt,
        parent_id,
        exclusions,
        constraints=None,
        revision_feedback=None,
    ):
        del exclusions, constraints
        assert revision_feedback is None
        candidate_id = "a" if attempt == 0 else "a2"
        return build_draft_batch(
            round_id=1,
            review_attempt=attempt,
            candidate_ids=(candidate_id,),
            variants=variants,
            predictions=predictions,
            evidence={},
            hypothesis_id=None,
            falsification_spec=None,
            parent_draft_batch_id=parent_id,
        )

    loop = BoundedReviewLoop(
        validator=BatchHardValidator(experiment_config.task, experiment_config.critic),
        critic=CriticAgent(client, max_retries=0),
        max_revision_attempts=1,
    )
    with pytest.raises(RuntimeError, match="Critic failed"):
        loop.run(
            draft_builder=builder,
            variants=variants,
            predictions=predictions,
            evidence={},
            revealed_ids=set(),
            pending_ids=set(),
            allowed_ids=set(variants),
            expected_batch_size=1,
            position_to_index={39: 0, 40: 1, 41: 2, 54: 3},
        )

    assert client.calls == 1


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
        hypothesis=_compiled_hypothesis("hyp:1"),
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


def test_falsification_spec_uses_structural_validation_without_hash_receipt():
    baseline = FitnessObservation("base", 0.0, "initial_observed", 0)
    spec = preregister_batch_median_test(
        hypothesis=_compiled_hypothesis("hyp:locked"),
        round_id=1,
        target_variant_ids=("target",),
        visible_observations=(baseline,),
    )
    changed = replace(
        spec,
        criteria=(replace(spec.criteria[0], support_threshold=-100.0),),
    )
    with pytest.raises(PermissionError, match="compilation receipt"):
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


def test_disabled_control_review_rejects_out_of_scope_add_control(
    experiment_config,
) -> None:
    variant = _variant("VDGA", "a")
    prediction = _prediction("a", 1.0)
    draft = build_draft_batch(
        round_id=1,
        review_attempt=0,
        candidate_ids=("a",),
        variants={"a": variant},
        predictions={"a": prediction},
        evidence={},
        hypothesis_id=None,
        falsification_spec=None,
    )
    report = BatchHardValidator(
        experiment_config.task,
        replace(experiment_config.critic, review_controls=False),
    ).validate(
        draft,
        variants={"a": variant},
        predictions={"a": prediction},
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids={"a"},
        expected_batch_size=1,
    )
    with pytest.raises(RuntimeError, match="without a configured safe fallback"):
        CriticAgent(_AddControlThenApprove(), max_retries=0).review(
            draft=draft,
            variants={"a": variant},
            predictions={"a": prediction},
            evidence={},
            conflict_report=report,
            batch_review_context=BatchReviewContext(
                prediction_status_by_id={},
                review_controls=False,
            ),
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


def test_compact_critic_context_uses_typed_structure_card_without_mutating_evidence():
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
            "sites": {
                "39": {
                    "status": "ok",
                    "contact_count": 4,
                    "relative_sasa": 0.2,
                    "secondary_structure": "loop",
                    "missing_backbone_atoms": [],
                }
            },
            "static_context_flag_count": 1,
            "resource_id": "rcsb:1PGB",
            "cache_status": "hit",
            "property_accessions": ["unused"],
        },
    )
    compacted = _compact_critic_context(
        {"evidence": {"a": [evidence]}, "context_evidence": [evidence]}
    )
    card = compacted["evidence"]["a"][0]
    features = card["features"]
    assert features["kind"] == "structure"
    assert features["sites"][0]["position"] == 39
    assert features["sites"][0]["contact_count"] == 4
    assert features["sites"][0]["relative_sasa"] == 0.2
    assert "cache_status" not in features
    assert "property_accessions" not in features
    assert "raw_features" not in card
    assert compacted["context_evidence"][0]["features"]["sites"]
    assert compacted["evidence_batch_metadata"]["structure_resource_ids"] == [
        "rcsb:1PGB"
    ]
    assert evidence.raw_features["sites"]["39"]["contact_count"] == 4
    assert evidence.raw_features["property_accessions"] == ["unused"]


def test_compact_critic_context_hoists_repeated_draft_and_conflict_state():
    candidate_ids = ["a", "b"]
    compacted = _compact_critic_context(
        {
            "draft": {
                "candidate_ids": candidate_ids,
                "design_rationales": [
                    {"candidate_id": item, "intended_test": "shared test"}
                    for item in candidate_ids
                ],
                "falsification_spec": {
                    "criteria": [
                        {
                            "criterion_id": "criterion:1",
                            "target_variant_ids": candidate_ids,
                            "comparator_variant_ids": [f"control:{index}" for index in range(64)],
                            "metric": "batch_median",
                        }
                    ]
                },
            },
            "conflict_report": {
                "report_id": "report:1",
                "conflicts": [
                    {
                        "conflict_id": f"conflict:{item}",
                        "code": "EVIDENCE_POLARITY_CONFLICT",
                        "scope": "evidence",
                        "severity": "warning",
                        "message": "Visible evidence is mixed.",
                        "hard": False,
                        "detector": "detector:v1",
                        "candidate_ids": [item],
                        "evidence_ids": [f"ev:{item}"],
                    }
                    for item in candidate_ids
                ],
            },
        }
    )

    draft = compacted["draft"]
    criterion = draft["falsification_spec"]["criteria"][0]
    assert draft["shared_intended_test"] == "shared test"
    assert all("intended_test" not in item for item in draft["design_rationales"])
    assert criterion["target_variant_scope"] == "draft_candidate_ids"
    assert criterion["target_variant_count"] == 2
    assert criterion["comparator_variant_count"] == 64
    assert "comparator_variant_ids" not in criterion
    report = compacted["conflict_report"]
    assert len(report["conflict_templates"]) == 1
    assert len(report["conflicts"]) == 2
    assert all("message" not in item for item in report["conflicts"])
