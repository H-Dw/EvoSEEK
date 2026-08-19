from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_routes_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_reasoning_routes.py"
    spec = importlib.util.spec_from_file_location("run_reasoning_routes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run(tmp_path: Path, *, rounds_aborted: int, completed_rounds: int) -> dict:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "structured_kg.sqlite").write_bytes(b"sqlite")
    (run_dir / "knowledge_graph_edges.json").write_text("[]", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "llm_provider": "deepseek",
                "budget_per_round": 16,
                "rounds": 3,
                "critic": {"mode": "remote", "provider": "deepseek"},
                "kg_interaction": {"feature_tool_strategy": "context_only"},
                "generation": {
                    "quota_allocation": {
                        "enabled": True,
                        "quotas": {
                            "hypothesis_target": 8,
                            "evidence_prior": 3,
                            "coverage_exploration": 3,
                            "matched_control": 2,
                        },
                    }
                },
                "knowledge_runtime": {
                    "provider_status": {
                        "physchem": {"status": "disabled"},
                        "conservation": {"status": "disabled"},
                        "structure": {"status": "disabled"},
                    },
                    "local_knowledge": {
                        "enabled": True,
                        "scientist_context_allowed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        json.dumps({"hypotheses": [{"hypothesis_id": "hyp:1"}]}),
        encoding="utf-8",
    )
    metrics = []
    for round_id in range(1, completed_rounds + 1):
        folder = run_dir / f"round_{round_id:02d}"
        folder.mkdir()
        (folder / "kg_interaction.json").write_text(
            json.dumps({"packs": [{"operator": "hypothesis_context"}, {"operator": "query_assay_association"}, {"operator": "query_local_knowledge"}, {"operator": "query_structured_claims"}]}),
            encoding="utf-8",
        )
        (folder / "structured_kg_pre_design.json").write_text(
            json.dumps({"entity_count": 2, "relation_count": 2}),
            encoding="utf-8",
        )
        (folder / "local_rag_retrieval.json").write_text(
            json.dumps({"chunks": ["c1"]}),
            encoding="utf-8",
        )
        (folder / "local_rag_evidence.json").write_text("{}", encoding="utf-8")
        (folder / "kg_truncation_audit.json").write_text(
            json.dumps(
                {
                    "any_truncated": False,
                    "entries": [
                        {"item": "MutationEffectEstimate"},
                        {"item": "HAS_MUTATION"},
                        {"item": "ABOUT_MUTATION"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        if round_id <= completed_rounds:
            (folder / "selection.csv").write_text("variant_id\n", encoding="utf-8")
            (folder / "approved_batch.json").write_text("{}", encoding="utf-8")
        metrics.append({"round_id": round_id, "batch_best_fitness": 1.0})
    if rounds_aborted:
        aborted = run_dir / f"round_{completed_rounds + 1:02d}"
        aborted.mkdir()
        (aborted / "kg_interaction.json").write_text(
            json.dumps({"packs": [{"operator": "hypothesis_context"}]}),
            encoding="utf-8",
        )
        (aborted / "structured_kg_pre_design.json").write_text(
            json.dumps({"entity_count": 1, "relation_count": 1}),
            encoding="utf-8",
        )
        (aborted / "local_rag_retrieval.json").write_text(
            json.dumps({"chunks": ["c1"]}),
            encoding="utf-8",
        )
        (aborted / "local_rag_evidence.json").write_text("{}", encoding="utf-8")
        (aborted / "kg_truncation_audit.json").write_text(
            json.dumps({"any_truncated": False, "entries": [{"item": "MutationEffectEstimate"}, {"item": "HAS_MUTATION"}, {"item": "ABOUT_MUTATION"}]}),
            encoding="utf-8",
        )
    summary = {
        "finalized": True,
        "condition": "rag_kg_none",
        "run_id": "run",
        "run_dir": str(run_dir),
        "rounds_aborted": rounds_aborted,
        "round_metrics": metrics,
        "queries_used": 16 * completed_rounds,
        "selection_driver": "agent_uq",
        "data_source": {"fold_index": 1},
    }
    return summary


def test_incomplete_run_is_not_audit_passed(tmp_path: Path) -> None:
    routes = _load_routes_module()
    spec = routes.RouteSpec(
        route_id="rag_kg_none",
        rag=True,
        channels=(),
        active_learning=False,
        test_goal="test",
    )
    summary = _write_run(tmp_path, rounds_aborted=1, completed_rounds=1)
    audit = routes.audit_run(summary, spec, expected_fold=1)
    failed = {item["name"] for item in audit["checks"] if item["severity"] == "error" and not item["passed"]}
    assert audit["passed"] is False
    assert "rounds_aborted_is_zero" in failed
    assert "completed_rounds_match_config" in failed


def test_complete_three_round_run_can_pass_completion_checks(tmp_path: Path) -> None:
    routes = _load_routes_module()
    spec = routes.RouteSpec(
        route_id="rag_kg_none",
        rag=True,
        channels=(),
        active_learning=False,
        test_goal="test",
    )
    summary = _write_run(tmp_path, rounds_aborted=0, completed_rounds=3)
    audit = routes.audit_run(summary, spec, expected_fold=1)
    names = {
        item["name"]: item["passed"]
        for item in audit["checks"]
        if item["name"]
        in {
            "rounds_aborted_is_zero",
            "completed_rounds_match_config",
            "queries_match_completed_budget",
            "round_01_selection",
            "round_01_approved_batch",
        }
    }
    assert names["rounds_aborted_is_zero"] is True
    assert names["completed_rounds_match_config"] is True
    assert names["queries_match_completed_budget"] is True
    assert names["round_01_selection"] is True
    assert names["round_01_approved_batch"] is True
