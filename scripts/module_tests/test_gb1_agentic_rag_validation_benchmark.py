from __future__ import annotations

import json
from pathlib import Path

from fitness_agents.config import load_experiment_config
from scripts.module_tests.benchmark_gb1_agentic_rag_validation import (
    CONDITIONS,
    EXPECTED_MANIFEST_HASH,
    FOLDS,
    aggregate,
    assert_preflight,
    condition_run_config,
    render_report,
    selected_runs,
)

CONFIG = "configs/experiments/gb1_3features_no_rag_vs_agentic_rag_validation_deepseek_v4_pro.yaml"


def test_frozen_config_and_two_conditions_isolate_external_rag(tmp_path: Path) -> None:
    base = load_experiment_config(CONFIG)
    preflight = assert_preflight(base)

    no_rag = condition_run_config(
        base,
        fold=0,
        condition="researcher_no_rag",
        output_root=tmp_path,
    )
    agentic = condition_run_config(
        base,
        fold=0,
        condition="researcher_agentic_rag",
        output_root=tmp_path,
    )

    assert FOLDS == (0, 1, 2)
    assert CONDITIONS == ("researcher_no_rag", "researcher_agentic_rag")
    assert all(preflight["checks"].values())
    assert preflight["index"]["stats"]["manifest_hash"] == EXPECTED_MANIFEST_HASH
    assert base.llm.model == base.critic.model == base.researcher.model == "deepseek-v4-pro"
    assert no_rag.researcher == agentic.researcher
    assert no_rag.kg_interaction == agentic.kg_interaction
    assert no_rag.generation == agentic.generation
    assert no_rag.validation == agentic.validation
    assert no_rag.knowledge.local_knowledge.enabled is False
    assert no_rag.knowledge.local_knowledge.retrieval.query_mode == "fixed"
    assert agentic.knowledge.local_knowledge.enabled is True
    assert agentic.knowledge.local_knowledge.corpus_mode == "read_only_prebuilt"
    assert agentic.knowledge.local_knowledge.retrieval.query_mode == "agentic"
    assert no_rag.kg_interaction.feature_tool_strategy == "agentic"
    assert no_rag.run_label == "N0"
    assert agentic.run_label == "A0"
    assert no_rag.output_root == tmp_path / "r" / "n" / "0"
    assert agentic.output_root == tmp_path / "r" / "a" / "0"


def test_selected_runs_uses_latest_integrity_complete_attempt(tmp_path: Path) -> None:
    older = tmp_path / "older"
    newer = tmp_path / "newer"
    older.mkdir()
    newer.mkdir()
    attempts = [
        {
            "status": "completed",
            "result": {
                "fold": 0,
                "condition": "researcher_no_rag",
                "run_dir": str(older),
                "integrity": {"completed": True},
                "run_id": "older",
            },
        },
        {
            "status": "failed",
            "fold": 0,
            "condition": "researcher_agentic_rag",
        },
        {
            "status": "completed",
            "result": {
                "fold": 0,
                "condition": "researcher_no_rag",
                "run_dir": str(newer),
                "integrity": {"completed": True, "no_fallback": True},
                "run_id": "newer",
            },
        },
    ]

    selected = selected_runs(attempts)

    assert len(selected) == 1
    assert selected[0]["run_id"] == "newer"


def _write_round_ids(run_dir: Path, *, prefix: str) -> None:
    for round_id in range(1, 4):
        round_dir = run_dir / f"round_{round_id:02d}"
        round_dir.mkdir(parents=True)
        candidates = [f"common-{round_id}", f"{prefix}-candidate-{round_id}"]
        approved = [f"common-approved-{round_id}", f"{prefix}-approved-{round_id}"]
        (round_dir / "candidate_pool_receipt.json").write_text(
            json.dumps({"candidate_ids": candidates}), encoding="utf-8"
        )
        (round_dir / "approved_batch.json").write_text(
            json.dumps({"candidate_ids": approved}), encoding="utf-8"
        )


def test_aggregate_reports_paired_effects_and_overlap_counts(tmp_path: Path) -> None:
    runs = []
    for fold in FOLDS:
        no_dir = tmp_path / f"no-{fold}"
        ar_dir = tmp_path / f"ar-{fold}"
        _write_round_ids(no_dir, prefix="no")
        _write_round_ids(ar_dir, prefix="ar")
        runs.extend(
            [
                {
                    "fold": fold,
                    "condition": "researcher_no_rag",
                    "run_dir": str(no_dir),
                    "final_best_seen": 1.0 + fold,
                    "auc_proxy": 0.5 + fold,
                    "researcher": {"rag_queries": 0},
                },
                {
                    "fold": fold,
                    "condition": "researcher_agentic_rag",
                    "run_dir": str(ar_dir),
                    "final_best_seen": 2.0 + fold,
                    "auc_proxy": 1.0 + fold,
                    "researcher": {"rag_queries": 1},
                },
            ]
        )

    result = aggregate(runs)

    assert result["complete_pair_count"] == 3
    assert result["median_paired_final_best_delta"] == 1.0
    assert result["mean_paired_auc_delta"] == 0.5
    assert result["positive_efficacy_supported"] is True
    assert result["total_agentic_rag_queries"] == 3
    assert all(item["candidate_overlap"] == 1 for item in result["candidate_and_approved_overlap"])
    assert all(item["approved_overlap"] == 1 for item in result["candidate_and_approved_overlap"])
    assert result["variant_identities_recorded_in_aggregate"] is False


def test_report_marks_partial_receipt_incomplete_without_variant_ids() -> None:
    receipt = {
        "environment": {
            "python_executable": "python",
            "git_head": "abc",
            "git_dirty_entry_count": 1,
            "config_sha256": "cfg",
        },
        "index": {
            "path": "index.sqlite",
            "sha256": "index",
            "stats": {
                "manifest_hash": EXPECTED_MANIFEST_HASH,
                "schema_version": "local-knowledge-index:v7",
                "documents": 6,
                "chunks": 6,
                "embeddings": 6,
                "knowledge_types": {},
                "facets": {},
            },
        },
        "attempts": [],
        "selected_runs": [],
        "aggregate": aggregate([]),
        "receipt_path": "receipt.json",
    }

    report = render_report(receipt)

    assert "`INCOMPLETE`" in report
    assert "mutation identities" in report
    assert "S-GB1" not in report


def test_partial_no_rag_run_is_not_reported_as_agentic_abstention() -> None:
    result = aggregate(
        [
            {
                "fold": 0,
                "condition": "researcher_no_rag",
                "researcher": {"rag_queries": 0},
            }
        ]
    )

    assert result["all_agentic_rounds_abstained"] is False


def test_report_includes_fail_closed_researcher_reason() -> None:
    receipt = {
        "environment": {},
        "index": {"stats": {}},
        "attempts": [
            {
                "fold": 0,
                "condition": "researcher_agentic_rag",
                "status": "invalid",
                "result": {
                    "fold": 0,
                    "condition": "researcher_agentic_rag",
                    "run_dir": "run-dir",
                    "integrity_failures": ["phase_a_matches_condition"],
                    "researcher": {
                        "rounds": [
                            {
                                "round": 1,
                                "external_decision": "PLAN",
                                "rag_queries": 0,
                                "retrieved_records": 0,
                                "feature_decision": None,
                                "feature_requests": 0,
                                "query_intents": ["boundary"],
                                "sanitized_queries": [],
                                "rejected": [
                                    {
                                        "step_id": "external_retrieval",
                                        "reason": "protected benchmark/task identity",
                                    }
                                ],
                            }
                        ]
                    },
                    "llm": {},
                },
            }
        ],
        "selected_runs": [],
        "aggregate": aggregate([]),
    }

    report = render_report(receipt)

    assert "protected benchmark/task identity" in report
    assert "phase_a_matches_condition" in report


def test_report_includes_redacted_required_node_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "failed-run"
    run_dir.mkdir()
    failure = "round_1:MAIN_NODE_FAILED:OUTPUT_SCHEMA_INVALID: "
    failure += "reason too long [type=string_too_long, "
    failure += "input_value='protected payload', input_type=str]"
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "required_node_failures": [failure]
            }
        ),
        encoding="utf-8",
    )
    receipt = {
        "environment": {},
        "index": {"stats": {}},
        "attempts": [
            {
                "fold": 1,
                "condition": "researcher_no_rag",
                "status": "invalid",
                "result": {
                    "fold": 1,
                    "condition": "researcher_no_rag",
                    "run_dir": str(run_dir),
                    "integrity_failures": ["no_required_node_failure"],
                    "researcher": {"rounds": []},
                    "llm": {},
                },
            }
        ],
        "selected_runs": [],
        "aggregate": aggregate([]),
    }

    report = render_report(receipt)

    assert "MAIN_NODE_FAILED:OUTPUT_SCHEMA_INVALID" in report
    assert "input_value=[REDACTED]" in report
    assert "protected payload" not in report


def test_report_includes_qwen_artifact_statistics(tmp_path: Path) -> None:
    run_dir = tmp_path / "agentic-run"
    round_dir = run_dir / "round_01"
    round_dir.mkdir(parents=True)
    (round_dir / "local_rag_retrieval.json").write_text(
        json.dumps(
            [
                {
                    "policy_decision": {"allowed": True},
                    "chunks": [{"scores": {"reranker": 0.9}}],
                    "records": [{"record_id": "R1"}],
                    "warnings": [],
                },
                {
                    "policy_decision": {"allowed": True},
                    "chunks": [],
                    "records": [],
                    "warnings": ["no_answer_above_retrieval_threshold"],
                },
            ]
        ),
        encoding="utf-8",
    )
    receipt = {
        "environment": {},
        "index": {"stats": {}},
        "attempts": [
            {
                "fold": 0,
                "condition": "researcher_agentic_rag",
                "status": "invalid",
                "result": {
                    "fold": 0,
                    "condition": "researcher_agentic_rag",
                    "run_dir": str(run_dir),
                    "researcher": {"rounds": []},
                    "llm": {},
                },
            }
        ],
        "selected_runs": [],
        "aggregate": aggregate([]),
    }

    report = render_report(receipt)

    assert "## Qwen 查询与重排统计" in report
    assert "| 0 | researcher_agentic_rag | 1 | 2 | 2 | 1 | 1 | 1 | 1 |" in report
