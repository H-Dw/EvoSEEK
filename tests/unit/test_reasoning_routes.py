from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from fitness_agents.config import load_experiment_config, project_root


def _load_module():
    path = project_root() / "scripts/run_reasoning_routes.py"
    spec = importlib.util.spec_from_file_location("fitness_agents_reasoning_routes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_matrix_defines_requested_eight_routes_and_first_three_folds():
    module = _load_module()
    matrix = project_root() / "configs/experiments/gb1_reasoning_routes.matrix.yaml"
    _base, folds, seed, parallel, routes = module.load_matrix(matrix)
    assert folds == [0, 1, 2]
    assert seed == 11
    assert parallel == 2
    assert [item.route_id for item in routes] == [
        "rag_kg_none",
        "rag_kg_all",
        "rag_kg_physchem",
        "rag_kg_physchem_structure",
        "rag_kg_physchem_conservation",
        "kg_all",
        "kg_none",
        "rag_kg_all_active_learning",
    ]


def test_apply_route_changes_only_declared_channels_rag_and_selection_driver(tmp_path: Path):
    module = _load_module()
    matrix = project_root() / "configs/experiments/gb1_reasoning_routes.matrix.yaml"
    base_path, _folds, _seed, _parallel, routes = module.load_matrix(matrix)
    base = load_experiment_config(base_path)
    spec = next(item for item in routes if item.route_id == "rag_kg_physchem_structure")
    configured = module.apply_route(base, spec, fold=2, seed=17, output_root=tmp_path)
    assert configured.task.fold_index == 2
    assert configured.seed == 17
    assert configured.rounds == 3
    assert configured.budget_per_round == 16
    assert configured.knowledge.physchem is True
    assert configured.knowledge.conservation is False
    assert configured.knowledge.structure is True
    assert configured.knowledge.kg is True
    assert configured.knowledge.local_knowledge.enabled is True
    assert configured.kg_interaction.feature_channels == ("physchem", "structure")
    assert configured.kg_interaction.feature_tool_strategy == "joint"
    assert configured.kg_interaction.max_tool_calls >= module.required_tool_calls(
        spec, variant_limit=configured.kg_interaction.feature_variant_limit
    )
    assert configured.generation.selection_driver == "agent_uq"
    assert configured.generation.quota_allocation.enabled is True
    assert configured.generation.quota_allocation.quotas() == {
        "hypothesis_target": 8,
        "evidence_prior": 3,
        "coverage_exploration": 3,
        "matched_control": 2,
    }
    assert configured.active_learning.enabled is False
    assert configured.critic.mode == "remote"
    assert configured.critic.provider != "mock"

    none_spec = next(item for item in routes if item.route_id == "rag_kg_none")
    none_config = module.apply_route(base, none_spec, fold=0, seed=11, output_root=tmp_path)
    assert none_config.kg_interaction.feature_tool_strategy == "context_only"
    assert "query_feature_bundle" not in none_config.kg_interaction.enabled_operators
    assert "query_local_knowledge" in none_config.kg_interaction.enabled_operators
    assert "query_assay_association" in none_config.kg_interaction.enabled_operators

    al_spec = next(item for item in routes if item.active_learning)
    al_config = module.apply_route(base, al_spec, fold=0, seed=11, output_root=tmp_path)
    assert al_config.generation.selection_driver == "active_learning"
    assert al_config.active_learning.enabled is True
    assert al_config.generation.quota_allocation.enabled is False
    assert al_config.kg_interaction.feature_tool_strategy == "joint"


def test_apply_route_keeps_tool_budget_above_planned_kg_rag_steps(tmp_path: Path):
    module = _load_module()
    matrix = project_root() / "configs/experiments/gb1_reasoning_routes.matrix.yaml"
    base_path, _folds, _seed, _parallel, routes = module.load_matrix(matrix)
    base = load_experiment_config(base_path)
    assert base.budget_per_round == 16
    assert base.rounds == 3
    for spec in routes:
        configured = module.apply_route(base, spec, fold=0, seed=11, output_root=tmp_path)
        needed = module.required_tool_calls(
            spec, variant_limit=configured.kg_interaction.feature_variant_limit
        )
        assert configured.kg_interaction.max_tool_calls >= needed
        if spec.channels:
            assert configured.kg_interaction.feature_tool_strategy != "context_only"
            assert configured.kg_interaction.feature_channels == spec.channels
        else:
            assert configured.kg_interaction.feature_tool_strategy == "context_only"
        assert "query_assay_association" in configured.kg_interaction.enabled_operators
        assert ("query_local_knowledge" in configured.kg_interaction.enabled_operators) == spec.rag
        assert ("query_structured_claims" in configured.kg_interaction.enabled_operators) == spec.rag


def test_artifact_audit_rejects_missing_expected_feature_channel(tmp_path: Path):
    module = _load_module()
    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_01"
    round_dir.mkdir(parents=True)
    (run_dir / "structured_kg.sqlite").write_bytes(b"sqlite")
    (run_dir / "knowledge_graph_edges.json").write_text("[]", encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "llm_provider": "deepseek",
                "knowledge_runtime": {
                    "provider_status": {
                        "physchem": {"status": "ready"},
                        "conservation": {"status": "disabled"},
                        "structure": {"status": "disabled"},
                    },
                    "local_knowledge": {
                        "enabled": False,
                        "scientist_context_allowed": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "state.json").write_text(
        json.dumps({"hypotheses": [{"id": "h1"}]}), encoding="utf-8"
    )
    (round_dir / "kg_interaction.json").write_text(
        json.dumps(
            {
                "packs": [
                    {"operator": "hypothesis_context"},
                    {"operator": "query_feature_bundle"},
                    {"operator": "query_kg_truncation_audit"},
                ]
            }
        ),
        encoding="utf-8",
    )
    # Deliberately omit physchem from the evidence contract.
    (round_dir / "evidence_contract.json").write_text(
        json.dumps({"channel_counts": {"kg": 2}}), encoding="utf-8"
    )
    (round_dir / "kg_truncation_audit.json").write_text(
        json.dumps(
            {
                "any_truncated": False,
                "entries": [
                    {"item": item} for item in (*module.BASE_AUDIT_ITEMS, "HAS_PHYSCHEM_DELTA")
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = module.RouteSpec("kg_physchem", False, ("physchem",), False, "test")
    summary = {
        "run_dir": str(run_dir),
        "run_id": "run",
        "condition": "kg_physchem",
        "finalized": True,
        "selection_driver": "agent_uq",
        "data_source": {"fold_index": 0},
    }
    report = module.audit_run(summary, spec)
    assert report["passed"] is False
    failed_names = {item["name"] for item in report["checks"] if not item["passed"]}
    assert "evidence_physchem_matches" in failed_names
