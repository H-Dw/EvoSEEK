import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import ComponentSpec, DatasetSpec
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.writer import write_split
from fitness_agents.loop import CampaignRunner, run_campaign


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
    if mode == "knowledge_agent":
        assert state["hypotheses"]
        assert state["hypotheses"][0]["evidence_ids"]


def _canonical_from_legacy_fixture(synthetic_benchmark) -> CanonicalDataset:
    public = pd.read_csv(synthetic_benchmark["public"]).drop(columns="split_role")
    oracle = pd.read_csv(synthetic_benchmark["oracle"])
    mutation_tokens = []
    for code in public["variant"]:
        tokens = [
            json.dumps(["GB1_WT", "GB1", position, wild_type, mutant], separators=(",", ":"))
            for position, wild_type, mutant in zip(
                (39, 40, 41, 54), "VDGV", code, strict=True
            )
            if wild_type != mutant
        ]
        mutation_tokens.append(json.dumps(tokens, separators=(",", ":")))
    public["dataset_id"] = "synthetic_fold"
    public["assay_id"] = "synthetic"
    public["backbone_id"] = "GB1_WT"
    public["component_sequences"] = public["variant"].map(
        lambda value: json.dumps([value])
    )
    public["mutation_tokens"] = mutation_tokens
    public["mutated_positions"] = "[]"
    public["source_row_id"] = range(len(public))
    public["source_sha256"] = "synthetic"
    labels = oracle.loc[:, ["variant_id", "fitness"]].rename(columns={"fitness": "target"})
    spec = DatasetSpec(
        dataset_id="synthetic_fold",
        assay_id="synthetic",
        adapter="generic_sequence",
        source=synthetic_benchmark["public"],
        sequence_column="sequence",
        target_column="target",
        components=(ComponentSpec("GB1", "VDGV", (39, 40, 41, 54)),),
    )
    return CanonicalDataset(public, labels, spec, "synthetic")


@pytest.mark.integration
def test_standard_campaign_runs_directly_from_manifest_fold(
    experiment_config, synthetic_benchmark, tmp_path
):
    dataset = _canonical_from_legacy_fixture(synthetic_benchmark)
    request = SplitRequest(
        "al96_closed_loop",
        protocol_version="campaign-test-v1",
        options={
            "initial_budget": 12,
            "test_depth_min": 2,
            "validation_size": 8,
            "validation_strata": ("mutation_count", "backbone_id"),
        },
    )
    result = build_split(dataset, request)
    split_root = write_split(dataset, request, result, tmp_path / "splits")
    task = replace(
        experiment_config.task,
        public_data_path=None,
        oracle_data_path=None,
        split_root=split_root,
        fold_index=0,
        expected_split_strategy="al96_closed_loop",
        expected_protocol_version="campaign-test-v1",
    )
    config = replace(
        experiment_config,
        mode="random",
        acquisition="random",
        knowledge_enabled=False,
        rounds=1,
        budget_per_round=2,
        candidate_limit=0,
        task=task,
        output_root=tmp_path / "runs",
        run_label="manifest-fold",
    )
    with pytest.raises(ValueError, match="Manifest SHA-256"):
        CampaignRunner(
            replace(
                config,
                task=replace(task, expected_manifest_sha256="0" * 64),
            )
        )
    runner = CampaignRunner(config)
    assert runner.bundle.validation_variants == []
    assert runner.bundle.validation_observations == []
    summary = runner.run()
    assert summary["finalized"] is True
    assert summary["queries_used"] == 2
    assert summary["data_source"]["kind"] == "manifest_fold"
    assert summary["data_source"]["fold_index"] == 0
    config_record = json.loads(
        (config.output_root / summary["run_id"] / "config.json").read_text()
    )
    assert config_record["data_source"]["strategy"] == "al96_closed_loop"
    assert "oracle_data_path" not in config_record


@pytest.mark.integration
def test_fold_scheduler_launches_isolated_campaign_processes(
    synthetic_benchmark, tmp_path
):
    dataset = _canonical_from_legacy_fixture(synthetic_benchmark)
    request = SplitRequest(
        "al96_closed_loop",
        protocol_version="scheduler-test-v1",
        options={
            "initial_budget": 12,
            "test_depth_min": 2,
            "validation_size": 8,
            "validation_strata": ("mutation_count", "backbone_id"),
        },
    )
    split_root = write_split(
        dataset,
        request,
        build_split(dataset, request),
        tmp_path / "splits",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        yaml.safe_dump(
            {
                "task_id": "scheduler_test",
                "protein_id": "GB1",
                "assay_id": "synthetic",
                "wild_type_sites": "VDGV",
                "mutable_positions": [39, 40, 41, 54],
                "objective": "maximize",
                "split_root": str(split_root),
                "fold_index": 0,
                "expected_split_strategy": "al96_closed_loop",
                "expected_protocol_version": "scheduler-test-v1",
            }
        ),
        encoding="utf-8",
    )
    experiment_path = tmp_path / "experiment.yaml"
    experiment_path.write_text(
        yaml.safe_dump(
            {
                "mode": "random",
                "seed": 5,
                "rounds": 1,
                "budget_per_round": 1,
                "candidate_limit": 0,
                "acquisition": "random",
                "ucb_beta": 1.5,
                "diversity_lambda": 0.0,
                "task_config": str(task_path),
                "model_config": "configs/model/baseline.yaml",
                "knowledge_config": "configs/knowledge/gb1.yaml",
                "output_root": str(tmp_path / "runs"),
                "llm_provider": "mock",
                "knowledge_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    scheduler_output = tmp_path / "scheduler-output"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_fold_campaigns.py",
            "--config",
            str(experiment_path),
            "--folds",
            "0,1",
            "--max-parallel",
            "2",
            "--output-dir",
            str(scheduler_output),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((scheduler_output / "report.json").read_text(encoding="utf-8"))
    assert report["completed"] == 2
    assert report["failed"] == 0
    comparison = pd.read_csv(scheduler_output / "aggregate/run_comparison.csv")
    assert set(comparison["fold_index"]) == {0, 1}
