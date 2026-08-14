import json

import pytest

from fitness_agents.loop import run_campaign


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "acquisition", "knowledge_enabled", "candidate_limit"),
    [
        ("random", "random", False, 0),
        ("fitness_direct", "greedy", False, 0),
        ("llm_agent", "greedy", False, 40),
        ("knowledge_agent", "greedy", True, 40),
    ],
)
def test_each_baseline_completes_with_global_ranks(
    config_factory, mode, acquisition, knowledge_enabled, candidate_limit
):
    config = config_factory(
        mode=mode,
        acquisition=acquisition,
        knowledge_enabled=knowledge_enabled,
        candidate_limit=candidate_limit,
        rounds=1,
        budget_per_round=3,
        run_label=mode,
    )
    summary = run_campaign(config)
    assert summary["finalized"] is True
    assert summary["queries_used"] == 3
    state = json.loads((config.output_root / summary["run_id"] / "state.json").read_text())
    records = state["selections"]
    assert len(records) == 3
    assert all(record["model_rank_all"] >= 1 for record in records)
    assert all(record["total_candidates"] == 88 for record in records)

