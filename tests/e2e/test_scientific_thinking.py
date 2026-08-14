import json
from dataclasses import replace

import pytest

from fitness_agents.evaluation import ScientificThinkingEvaluator
from fitness_agents.loop import run_campaign


@pytest.mark.e2e
def test_scientific_interventions_run_and_are_auditable(config_factory):
    base = config_factory(rounds=1, budget_per_round=4, run_label="science-ref")
    configs = {
        "reference": base,
        "knowledge_ablation": replace(
            base, knowledge_enabled=False, run_label="science-no-knowledge"
        ),
        "score_shuffle": replace(base, score_shuffle=True, run_label="science-shuffle"),
        "evidence_deletion": replace(
            base, evidence_deletion=True, run_label="science-delete-evidence"
        ),
    }
    summaries = {name: run_campaign(config) for name, config in configs.items()}
    report = ScientificThinkingEvaluator().evaluate(
        reference_dir=summaries["reference"]["run_dir"],
        knowledge_ablation_dir=summaries["knowledge_ablation"]["run_dir"],
        score_shuffle_dir=summaries["score_shuffle"]["run_dir"],
        evidence_deletion_dir=summaries["evidence_deletion"]["run_dir"],
    )
    assert set(report["metrics"]) >= {
        "knowledge_ablation_selection_change",
        "score_shuffle_selection_change",
        "evidence_deletion_selection_change",
    }
    assert report["metrics"]["global_rank_tracking_completeness"] == 1.0
    shuffle_state = json.loads(
        (replace(base, score_shuffle=True).output_root / summaries["score_shuffle"]["run_id"] / "state.json").read_text()
    )
    assert all("score_shuffle" in row["intervention_tags"] for row in shuffle_state["selections"])

