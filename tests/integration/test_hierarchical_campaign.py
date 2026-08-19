from __future__ import annotations

import json
from dataclasses import replace

import pytest

from fitness_agents.config import load_experiment_config
from fitness_agents.contracts.hypothesis_pipeline import HypothesisPipelineResult
from fitness_agents.loop import CampaignRunner


@pytest.mark.integration
def test_mock_feature_campaign_runs_hierarchical_scientist_critic_graph(tmp_path) -> None:
    base = load_experiment_config(
        "configs/experiments/knowledge_agent_features.example.yaml"
    )
    config = replace(
        base,
        output_root=tmp_path,
    )

    summary = CampaignRunner(config).run()
    run_dir = tmp_path / summary["run_id"]
    pipeline = json.loads(
        (run_dir / "round_01/hypothesis_pipeline.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (run_dir / "completion_manifest.json").read_text(encoding="utf-8")
    )

    assert pipeline["status"] == "SUCCEEDED"
    assert {item["channel"]: item["status"] for item in pipeline["branches"]} == {
        "physchem": "SUCCEEDED",
        "conservation": "SUCCEEDED",
        "structure": "SUCCEEDED",
    }
    assert pipeline["main_review"]["verdict"] == "APPROVE"
    assert summary["run_status"] == "completed"
    assert completion["pass_eligible"] is True


@pytest.mark.integration
def test_failed_required_hypothesis_node_finalizes_artifacts_but_never_passes(
    config_factory,
) -> None:
    config = config_factory(rounds=1, budget_per_round=2, candidate_limit=12)
    runner = CampaignRunner(config)

    class _FailedGraph:
        def run(self, **kwargs):
            del kwargs
            return HypothesisPipelineResult(
                status="FAILED",
                branches=(),
                conflicts=(),
                failure_code="REQUIRED_CHILD_FAILED:structure",
            )

    runner.hypothesis_graph = _FailedGraph()
    summary = runner.run()
    completion = json.loads(
        (config.output_root / summary["run_id"] / "completion_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["finalized"] is True
    assert summary["run_status"] == "failed"
    assert summary["pass_eligible"] is False
    assert completion["evaluation_status"] != "passed"
    assert completion["required_node_failures"]
