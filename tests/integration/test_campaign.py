import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from fitness_agents.config import (
    ActiveLearningConfig,
    AgentQuotaAllocationConfig,
    CalibratedPosteriorConfig,
    GenerationConfig,
    HybridBatchAcquisitionConfig,
)
from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import ComponentSpec, DatasetSpec
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.writer import write_split
from fitness_agents.loop import CampaignRunner, run_campaign


@pytest.mark.integration
def test_agent_quota_allocation_enters_main_loop(config_factory):
    base = config_factory(
        rounds=1,
        budget_per_round=4,
        candidate_limit=40,
        run_label="agent-quota",
    )
    config = replace(
        base,
        generation=GenerationConfig(
            selection_driver="agent_uq",
            quota_allocation=AgentQuotaAllocationConfig(
                enabled=True,
                hypothesis_target=2,
                evidence_prior=1,
                coverage_exploration=1,
                matched_control=0,
            ),
        ),
    )

    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])
    allocation = json.loads(
        (run_dir / "round_01/agent_quota_acquisition.json").read_text(
            encoding="utf-8"
        )
    )

    assert allocation["plugin"] == "agent_uq_quota_v1"
    assert allocation["quotas"] == {
        "hypothesis_target": 2,
        "evidence_prior": 1,
        "coverage_exploration": 1,
        "matched_control": 0,
    }
    assert len(allocation["selected_ids"]) == 4
    approved = json.loads(
        (run_dir / "round_01/agent_quota_acquisition_approved.json").read_text(
            encoding="utf-8"
        )
    )
    assert approved["matches_approved_batch"] is True


@pytest.mark.integration
def test_configured_active_learning_module_enters_main_loop(config_factory):
    base = config_factory(
        rounds=1,
        budget_per_round=4,
        candidate_limit=40,
        run_label="active-learning",
    )
    config = replace(
        base,
        generation=GenerationConfig(selection_driver="active_learning"),
        active_learning=ActiveLearningConfig(
            enabled=True,
            posterior=CalibratedPosteriorConfig(
                predictor_models=(base.model,),
                calibration_fraction=0.25,
                min_calibration_size=4,
                min_training_size=8,
            ),
            acquisition=HybridBatchAcquisitionConfig(
                exploitation_fraction=0.50,
                exploration_fraction=0.25,
                knowledge_fraction=0.25,
                ucb_beta=1.0,
                diversity_lambda=0.10,
            ),
        ),
    )

    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])
    posterior = json.loads(
        (run_dir / "round_01/active_learning_posterior.json").read_text(encoding="utf-8")
    )
    acquisition = json.loads(
        (run_dir / "round_01/active_learning_acquisition.json").read_text(
            encoding="utf-8"
        )
    )
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    scope = json.loads(
        (run_dir / "round_01/prediction_scope_receipt.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["selection_driver"] == "active_learning"
    assert summary["fitness_predictors_used_for_generation"] is True
    assert summary["active_learning_module"] == "lightweight_calibrated_hybrid"
    assert posterior["calibration"]["status"] == "calibrated"
    assert sum(acquisition["selection"]["quotas"].values()) == 4
    assert len(acquisition["selection"]["selected_ids"]) == 4
    assert scope["acquisition_prediction_scope"] == "candidate_pool"
    assert scope["dry_validation_candidate_count"] == 4
    assert {
        item["selection_driver"] for item in state["selections"]
    } == {"active_learning:lightweight_calibrated_hybrid"}


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
def test_each_baseline_completes_with_audited_prediction_scopes(
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
    scope = json.loads(
        (
            config.output_root
            / summary["run_id"]
            / "round_01/prediction_scope_receipt.json"
        ).read_text()
    )
    assert scope["dry_validation_candidate_count"] == 3
    assert scope["approved_batch_size"] == 3
    assert scope["acquisition_prediction_scope"] == (
        "candidate_pool" if mode == "fitness_direct" else "none"
    )
    if mode == "knowledge_agent":
        assert state["hypotheses"]
        assert state["hypotheses"][0]["evidence_ids"]


@pytest.mark.integration
def test_agent_selection_precedes_dry_validation_and_writes_full_kg_outputs(config_factory):
    config = config_factory(
        mode="knowledge_agent",
        acquisition="greedy",
        knowledge_enabled=True,
        rounds=1,
        budget_per_round=2,
        candidate_limit=24,
        run_label="validation-after-design",
    )
    summary = run_campaign(config)
    run_dir = config.output_root / summary["run_id"]
    events = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    event_types = [item["event_type"] for item in events]
    assert event_types.index("batch_initial_selected") < event_types.index(
        "validation_model_fit_started"
    )
    interaction = json.loads((run_dir / "round_01/kg_interaction.json").read_text())
    assert [item["operator"] for item in interaction["packs"]] == [
        "hypothesis_context",
        "query_assay_association",
        "explain_variant",
        "compare_variants",
    ]
    matrix = json.loads((run_dir / "round_01/validation_matrix.json").read_text())
    assert {item["validation_type"] for item in matrix} == {"wet", "dry"}
    scope = json.loads(
        (run_dir / "round_01/prediction_scope_receipt.json").read_text()
    )
    assert scope["acquisition_prediction_scope"] == "none"
    assert scope["dry_validation_scope"] == "draft_selected_candidates_only"
    assert scope["dry_validation_candidate_count"] == 2
    assert scope["approved_batch_size"] == 2
    assert scope["oracle_measurement_scope"] == "approved_batch_only"
    assert min(item["base_weight"] for item in matrix if item["validation_type"] == "wet") > max(
        item["base_weight"] for item in matrix if item["validation_type"] == "dry"
    )
    assert (run_dir / "structured_kg.sqlite").is_file()
    assert (run_dir / "wild_type.json").is_file()
    assert (run_dir / "round_01/top_k.csv").is_file()
    assert (run_dir / "fitness_progress.svg").is_file()
    assert (run_dir / "reasoning.md").is_file()
    assert summary["selection_driver"] == "agent_uq"
    assert summary["fitness_predictors_used_for_generation"] is False


@pytest.mark.integration
def test_disabling_dry_validation_does_not_relabel_agent_utility_as_dry(config_factory):
    config = config_factory(
        mode="knowledge_agent",
        acquisition="greedy",
        knowledge_enabled=True,
        rounds=1,
        budget_per_round=2,
        candidate_limit=20,
        run_label="wet-only-validation",
    )
    config = replace(config, validation=replace(config.validation, enabled=False))

    summary = run_campaign(config)
    run_dir = config.output_root / summary["run_id"]
    matrix = json.loads((run_dir / "round_01/validation_matrix.json").read_text())
    state = json.loads((run_dir / "state.json").read_text())

    assert {item["validation_type"] for item in matrix} == {"wet"}
    assert all(not item["validation_model_versions"] for item in state["selections"])


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
