import json
from dataclasses import replace

import pytest

from fitness_agents.agents.critic import CriticAgent
from fitness_agents.contracts.schemas import (
    BatchRisk,
    CritiqueDecision,
    FalsificationReadiness,
    IssueSeverity,
    RequiredChange,
    RequiredChangeAction,
    ReviewVerdict,
)
from fitness_agents.loop import CampaignRunner, run_campaign


class _RejectingClient:
    provider_name = "rejecting_test_critic"

    def review(self, *, context, output_schema):
        draft = context["draft"]
        return CritiqueDecision(
            decision_id=f"reject:{draft.draft_batch_id}",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.REJECT,
            falsification_readiness=FalsificationReadiness.UNTESTABLE,
            required_changes=(
                RequiredChange(
                    action=RequiredChangeAction.ABORT_ROUND,
                    target_ids=(),
                    parameters={},
                    rationale="Injected rejection for fallback testing.",
                ),
            ),
            confidence=1.0,
            summary="Injected rejection.",
        )


class _InfeasibleResidueRevisionClient:
    provider_name = "infeasible_residue_revision_test_critic"

    def __init__(self) -> None:
        self.calls = 0

    def review(self, *, context, output_schema):
        del output_schema
        self.calls += 1
        if self.calls != 1:
            raise AssertionError("infeasible revision must stop before another Critic call")
        draft = context["draft"]
        return CritiqueDecision(
            decision_id=f"revise:{draft.draft_batch_id}",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=ReviewVerdict.REVISE,
            falsification_readiness=FalsificationReadiness.NEEDS_REVISION,
            batch_level_risks=(
                BatchRisk(
                    risk_id="R01",
                    code="BATCH_MODE_COLLAPSE",
                    severity=IssueSeverity.ERROR,
                    statement="Synthetic independent batch-level revision basis.",
                    candidate_ids=(draft.candidate_ids[0],),
                ),
            ),
            required_changes=(
                RequiredChange(
                    action=RequiredChangeAction.REPLACE_CANDIDATE,
                    target_ids=(draft.candidate_ids[0],),
                    parameters={
                        "excluded_substitutions": [
                            {"position": 39, "to_residue": residue}
                            for residue in "ACDEFGHIKLMNPQRSTVWY"
                        ],
                        "applies_to_arms": ["fallback"],
                    },
                    rationale="Synthetic infeasible position-wide exclusion.",
                ),
            ),
            confidence=1.0,
            summary="Force a deterministically infeasible revision.",
        )


@pytest.mark.integration
def test_campaign_submits_only_after_critic_and_assesses_hypothesis(config_factory):
    config = config_factory(rounds=1, budget_per_round=3, run_label="critic-control")
    summary = run_campaign(config)
    run_dir = config.output_root / summary["run_id"]
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = [item["event_type"] for item in trace]

    assert summary["critique_decisions"] == 1
    assert summary["hypothesis_assessments"] == 1
    assert state["critique_decisions"][0]["verdict"] == "APPROVE"
    assert state["hypothesis_explanations"][0]["hypothesis_id"] == state["hypotheses"][0]["hypothesis_id"]
    assert state["hypothesis_explanations"][0]["critic_role"] == "batch_critic"
    assert state["hypothesis_assessments"][0]["status"] in {
        "SUPPORTED",
        "CONTRADICTED",
        "INCONCLUSIVE",
    }
    assert event_types.index("critique_completed") < event_types.index("batch_approved")
    assert event_types.index("batch_approved") < event_types.index("batch_measured")
    assert (run_dir / "round_01" / "approved_batch.json").is_file()
    assert (run_dir / "round_01" / "hypothesis_assessment.json").is_file()


@pytest.mark.integration
def test_configured_safe_fallback_is_revalidated_before_submission(config_factory):
    base = config_factory(rounds=1, budget_per_round=2, run_label="critic-fallback")
    config = replace(
        base,
        critic=replace(base.critic, on_reject="safe_fallback"),
    )
    summary = CampaignRunner(
        config,
        critic_agent=CriticAgent(_RejectingClient(), max_retries=0),
    ).run()
    trace = [
        json.loads(line)
        for line in (config.output_root / summary["run_id"] / "trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["queries_used"] == 2
    assert summary["rounds_aborted"] == 0
    assert any(item["event_type"] == "critic_fallback_used" for item in trace)


@pytest.mark.integration
def test_round_abort_skips_final_test(config_factory):
    config = config_factory(rounds=1, budget_per_round=2, run_label="critic-abort")
    runner = CampaignRunner(
        config,
        critic_agent=CriticAgent(_RejectingClient(), max_retries=0),
    )
    summary = runner.run()
    run_dir = config.output_root / summary["run_id"]
    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = [item["event_type"] for item in trace if item.get("event_type")]

    assert summary["rounds_aborted"] == 1
    assert summary["queries_used"] == 0
    assert summary["final_prediction_metrics"] is None
    assert summary["finalized"] is True
    assert runner.state.final_test_opened is False
    assert runner.backend.raw_backend.final_opened is False
    assert "round_aborted" in event_types
    assert "final_fit_started" not in event_types
    assert "final_predict_started" not in event_types
    assert (run_dir / "summary.json").is_file()


@pytest.mark.integration
def test_infeasible_residue_revision_writes_shortfall_without_second_critic_call(
    config_factory,
) -> None:
    config = config_factory(
        rounds=1,
        budget_per_round=4,
        candidate_limit=24,
        run_label="revision-constraint-infeasible",
    )
    client = _InfeasibleResidueRevisionClient()
    summary = CampaignRunner(
        config,
        critic_agent=CriticAgent(client, max_retries=0),
    ).run()
    run_dir = config.output_root / summary["run_id"]
    receipt = json.loads(
        (run_dir / "round_01/revision_constraint_infeasible.json").read_text(
            encoding="utf-8"
        )
    )

    assert client.calls == 1
    assert summary["rounds_aborted"] == 1
    assert summary["queries_used"] == 0
    assert receipt["code"] == "REVISION_CONSTRAINT_INFEASIBLE"
    assert receipt["selected_count"] == 0
    assert receipt["shortfall"] == 4
