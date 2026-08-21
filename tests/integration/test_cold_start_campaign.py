from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from fitness_agents.config import (
    ExperimentConfig,
    GenerationConfig,
    KnowledgeConfig,
    LearnableParameterSpec,
    ModelConfig,
    PriorScheduleConfig,
    TaskConfig,
)
from fitness_agents.data.gb1 import canonical_mutation_notation, variant_id
from fitness_agents.loop import CampaignRunner

WT = "VDGV"
# Wild type plus controlled singles/doubles/triples/quads so the mutation-order
# filter has a known pool composition to act on.
SINGLES = ["ADGV", "FDGV", "VAGV", "VFGV", "VDAV", "VDFV", "VDGW", "VDGL"]
DOUBLES = ["AAGV", "FAGV", "ADAV", "FDFV", "VAAV", "VFFV", "VDAL", "VDFW"]
TRIPLES = ["AAAV", "FAAV", "AADV", "FAWV", "VAAA", "VFFA", "ADAL", "FDFW"]
QUADS = ["AAAA", "FFAA", "AAFF", "FFFF"]
VALIDATION = ["AAAL", "FFFL", "ALAL", "FLFL"]
FINAL_TEST = ["LLLL", "WWWW", "LLWW", "WWLL"]


def _fitness(code: str) -> float:
    return 1.0 + 0.1 * sum(a != b for a, b in zip(code, WT, strict=True))


def _rows(codes: list[str], role: str) -> list[dict[str, object]]:
    return [
        {
            "variant_id": variant_id(code),
            "variant": code,
            "sequence": code,
            "mutation_notation": canonical_mutation_notation(code),
            "mutation_count": sum(a != b for a, b in zip(code, WT, strict=True)),
            "split_role": role,
            "fitness": _fitness(code),
        }
        for code in codes
    ]


@pytest.fixture
def cold_start_benchmark(tmp_path: Path) -> dict[str, Path]:
    rows = (
        _rows([WT], "initial_observed")
        + _rows(SINGLES[:3], "initial_observed")
        + _rows(SINGLES[3:], "oracle_pool")
        + _rows(DOUBLES, "oracle_pool")
        + _rows(TRIPLES, "oracle_pool")
        + _rows(QUADS, "oracle_pool")
        + _rows(VALIDATION, "validation")
        + _rows(FINAL_TEST, "final_test")
    )
    frame = pd.DataFrame(rows)
    public = tmp_path / "public.csv"
    oracle = tmp_path / "oracle.csv"
    frame.drop(columns="fitness").to_csv(public, index=False)
    frame[["variant_id", "fitness", "split_role"]].to_csv(oracle, index=False)
    return {"public": public, "oracle": oracle, "root": tmp_path}


def _config(benchmark: dict[str, Path], **generation_overrides) -> ExperimentConfig:
    task = TaskConfig(
        task_id="synthetic_gb1_cold_start",
        protein_id="GB1",
        assay_id="synthetic",
        wild_type_sites=WT,
        mutable_positions=[39, 40, 41, 54],
        objective="maximize",
        public_data_path=benchmark["public"],
        oracle_data_path=benchmark["oracle"],
    )
    knowledge = KnowledgeConfig(
        site_profiles={
            39: {"wild_type": "V", "tolerated": ["V", "F"], "structure_risk": 0.2},
            40: {"wild_type": "D", "tolerated": ["D", "W"], "structure_risk": 0.3},
            41: {"wild_type": "G", "tolerated": ["G", "A"], "structure_risk": 0.7},
            54: {"wild_type": "V", "tolerated": ["V", "L"], "structure_risk": 0.3},
        },
        parameters={
            "kg.shrinkage_pseudocount": LearnableParameterSpec(value=3.0),
            "kg.confidence_base": LearnableParameterSpec(value=0.25),
            "kg.support_gain": LearnableParameterSpec(value=0.03),
            "kg.confidence_cap": LearnableParameterSpec(value=0.85),
        },
    )
    return ExperimentConfig(
        mode="knowledge_agent",
        seed=7,
        rounds=2,
        budget_per_round=4,
        candidate_limit=8,
        acquisition="greedy",
        ucb_beta=1.5,
        diversity_lambda=0.1,
        task=task,
        model=ModelConfig(ridge_members=3, extra_trees_estimators=20, bootstrap_fraction=0.8),
        knowledge=knowledge,
        output_root=benchmark["root"] / "runs",
        llm_provider="mock",
        knowledge_enabled=True,
        prior_schedule=PriorScheduleConfig(mode="cold_start", keep_wild_type=True),
        generation=GenerationConfig(**generation_overrides),
    )


def _trace_events(run_dir: Path, event_type: str) -> list[dict[str, object]]:
    return [
        json.loads(line)["payload"]
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event_type"] == event_type
    ]


@pytest.mark.integration
def test_cold_start_campaign_withholds_mutation_priors(cold_start_benchmark):
    config = _config(cold_start_benchmark)

    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])

    started = _trace_events(run_dir, "campaign_started")
    assert len(started) == 1
    assert started[0]["initial_count"] == 1
    assert started[0]["prior_schedule"] == {
        "mode": "cold_start",
        "keep_wild_type": True,
        "withheld_prior_count": 3,
    }

    # The three withheld singles are never revealed: only the wild type plus
    # the two measured batches (2 rounds x budget 4) appear in the state.
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    withheld = {variant_id(code) for code in SINGLES[:3]}
    assert withheld.isdisjoint(state["revealed_variant_ids"])
    assert len(state["revealed_variant_ids"]) == 1 + 2 * 4

    config_record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_record["prior_schedule"]["mode"] == "cold_start"


@pytest.mark.integration
def test_mutation_order_schedule_restricts_round_one_candidates(cold_start_benchmark):
    public = pd.read_csv(cold_start_benchmark["public"])
    count_by_id = public.set_index("variant_id")["mutation_count"].to_dict()
    config = _config(cold_start_benchmark, mutation_order_schedule={1: (1,)})

    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])

    receipt1 = json.loads(
        (run_dir / "round_01/candidate_pool_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt1["mutation_order_filter"] == [1]
    assert receipt1["candidate_ids"]
    assert all(count_by_id[item] == 1 for item in receipt1["candidate_ids"])

    receipt2 = json.loads(
        (run_dir / "round_02/candidate_pool_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt2["mutation_order_filter"] is None

    # Round 1 measurements are singles only; round 2 is unrestricted.
    selection1 = pd.read_csv(run_dir / "round_01" / "selection.csv")
    assert set(selection1["variant_id"].map(count_by_id)) == {1}
    config_record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert config_record["generation"]["mutation_order_schedule"] == {"1": [1]}


@pytest.mark.integration
def test_upfront_prior_schedule_remains_the_default(cold_start_benchmark):
    config = replace(_config(cold_start_benchmark), prior_schedule=PriorScheduleConfig())

    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])

    started = _trace_events(run_dir, "campaign_started")
    assert started[0]["initial_count"] == 4
    assert started[0]["prior_schedule"]["mode"] == "upfront"
    assert started[0]["prior_schedule"]["withheld_prior_count"] == 0
