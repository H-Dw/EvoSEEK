from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from fitness_agents.config import load_experiment_config
from scripts.module_tests.benchmark_gb1_3features_coldstart_validation import (
    CONDITIONS,
    FOLDS,
    assert_three_fold_validation_preflight,
    resumable_runs,
    validation_run_config,
)
from scripts.module_tests.benchmark_gb1_directive_rag import (
    audit_validation_feedback,
)

CONFIG = (
    "configs/experiments/gb1_3features_coldstart_validation_deepseek_v4_pro.yaml"
)


def test_three_feature_cold_start_config_is_wet_only_and_kermut_free(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG)

    assert config.rounds == 3
    assert config.candidate_limit == 32
    assert config.budget_per_round == 16
    assert config.task.split_root is not None
    assert config.prior_schedule.mode == "cold_start"
    assert config.prior_schedule.keep_wild_type is True
    assert (
        config.prior_schedule.no_supported_hypothesis_policy
        == "coverage_exploration"
    )
    assert not config.generation.mutation_order_schedule
    assert config.model.name == "onehot_heterogeneous_ensemble"
    assert config.generation.use_fitness_predictors is False
    assert config.generation.predictor_models == ()
    assert config.validation.enabled is False
    assert config.validation.predictor_models == ()
    assert config.active_learning.enabled is False
    assert config.validation.rethink_mode == "sample"
    assert config.evaluation.top_k == 10
    assert "svg" in config.output.artifacts
    assert config.llm.model == "deepseek-v4-pro"
    assert config.critic.model == "deepseek-v4-pro"
    assert config.hierarchical_hypothesis.enabled is True
    assert config.hierarchical_hypothesis.required_channels == (
        "physchem",
        "conservation",
        "structure",
    )
    assert config.kg_interaction.feature_channels == (
        "physchem",
        "conservation",
        "structure",
    )
    assert config.kg_interaction.feature_variant_limit == 2
    assert config.kg_interaction.max_tool_calls == 18
    assert config.knowledge.providers["conservation"].a3m_path is not None
    assert (
        config.knowledge.providers["structure"].options["resource_id"]
        == "rcsb:1PGB"
    )
    quota = config.generation.quota_allocation
    assert (
        quota.hypothesis_target,
        quota.evidence_prior,
        quota.coverage_exploration,
        quota.matched_control,
    ) == (8, 3, 3, 2)
    assert quota.total == 16

    split_root = tmp_path / "GB1-AL96-5CV-v2"
    split_root.mkdir()
    (split_root / "manifest.public.json").write_text(
        json.dumps(
            {
                "n_folds": 5,
                "strategy": "al96_closed_loop",
                "protocol_version": "GB1-AL96-5CV-v2",
                "options": {"initial_budget": 1},
            }
        ),
        encoding="utf-8",
    )
    for fold in FOLDS:
        fold_root = split_root / f"fold_{fold:02d}"
        fold_root.mkdir()
        (fold_root / "fold_manifest.json").write_text(
            json.dumps(
                {
                    "fold_index": fold,
                    "role_counts": {
                        "initial_observed": 1,
                        "train_observed": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
    config = replace(config, task=replace(config.task, split_root=split_root))
    preflight = assert_three_fold_validation_preflight(config)
    assert all(preflight["checks"].values())
    assert all(preflight["common"]["checks"].values())
    assert preflight["folds"] == [0, 1, 2]


def test_paired_conditions_toggle_only_local_rag(tmp_path: Path) -> None:
    base = load_experiment_config(CONFIG)
    corpus_path = tmp_path / "corpus.sqlite"
    corpus_path.touch()
    local = replace(base.knowledge.local_knowledge, corpus_index_path=corpus_path)
    base = replace(base, knowledge=replace(base.knowledge, local_knowledge=local))

    no_rag = validation_run_config(
        base,
        fold=0,
        condition="kg_3features_base",
        output_root=tmp_path / "output",
    )
    rag = validation_run_config(
        base,
        fold=0,
        condition="kg_3features_rag",
        output_root=tmp_path / "output",
    )

    assert FOLDS == (0, 1, 2)
    assert CONDITIONS == ("kg_3features_base", "kg_3features_rag")
    assert no_rag.task.fold_index == rag.task.fold_index == 0
    assert no_rag.run_label == "V-B-F0"
    assert rag.run_label == "V-R-F0"
    assert no_rag.output_root == tmp_path / "output" / "runs" / "base" / "f0"
    assert rag.output_root == tmp_path / "output" / "runs" / "rag" / "f0"
    assert no_rag.knowledge.local_knowledge.enabled is False
    assert rag.knowledge.local_knowledge.enabled is True
    assert no_rag.kg_interaction.feature_channels == rag.kg_interaction.feature_channels
    assert no_rag.hierarchical_hypothesis == rag.hierarchical_hypothesis
    assert no_rag.generation == rag.generation
    assert no_rag.validation == rag.validation


def test_resume_reuses_only_integrity_complete_existing_runs(tmp_path: Path) -> None:
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    failed_dir = tmp_path / "failed"
    failed_dir.mkdir()
    receipt_path = tmp_path / "paired_validation_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "fold": 0,
                        "condition": "kg_3features_base",
                        "run_dir": str(complete_dir),
                        "integrity": {"completed": True, "feedback_contract": True},
                    },
                    {
                        "fold": 0,
                        "condition": "kg_3features_rag",
                        "run_dir": str(failed_dir),
                        "integrity": {"completed": False},
                    },
                    {
                        "fold": 1,
                        "condition": "kg_3features_rag",
                        "status": "failed",
                        "run_dir": str(failed_dir),
                        "integrity": {"completed": True},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    retained = resumable_runs(receipt_path)

    assert [(item["fold"], item["condition"]) for item in retained] == [
        (0, "kg_3features_base")
    ]


def test_feedback_audit_proves_wet_results_enter_later_rounds(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(CONFIG)
    events = [
        {
            "event_type": "campaign_started",
            "payload": {
                "initial_count": 1,
                "prior_schedule": {
                    "mode": "cold_start",
                    "keep_wild_type": True,
                    "withheld_prior_count": 0,
                },
            },
        },
        *(
            {
                "event_type": "round_started",
                "payload": {"round_id": round_id, "n_observed": count},
            }
            for round_id, count in ((1, 1), (2, 17), (3, 33))
        ),
    ]
    (tmp_path / "trace.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in events), encoding="utf-8"
    )
    records = [
        {"validation_type": "wet", "round_id": round_id}
        for round_id in range(1, 4)
        for _ in range(16)
    ]
    (tmp_path / "validation_records.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    (tmp_path / "summary.json").write_text(
        json.dumps({"actual_batch_sizes": [16, 16, 16]}), encoding="utf-8"
    )
    for round_id in (2, 3):
        round_dir = tmp_path / f"round_{round_id:02d}"
        round_dir.mkdir()
        (round_dir / "design_scores.json").write_text(
            json.dumps([{"prior_score": 0.0}]), encoding="utf-8"
        )

    audit = audit_validation_feedback(
        tmp_path, config, feedback_contract="cold_start_wet_prior"
    )

    assert audit["passed"] is True
    assert audit["visible_observation_counts_by_round"] == [1, 17, 33]
    assert audit["wet_validation_counts_by_round"] == [16, 16, 16]
    assert audit["prior_score_interface_by_round"] == [True, True]
