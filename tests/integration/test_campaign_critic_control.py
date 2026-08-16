import json
from dataclasses import replace

import pytest

from fitness_agents.agents.critic import CriticAgent
from fitness_agents.contracts.schemas import (
    CritiqueDecision,
    FalsificationReadiness,
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
