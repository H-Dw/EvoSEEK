from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from fitness_agents.config import (
    LeakageGuardConfig,
    LocalKnowledgeConfig,
    LocalKnowledgeIngestionConfig,
    LocalKnowledgeRetrievalConfig,
    LocalKnowledgeRootConfig,
)
from fitness_agents.loop import run_campaign


@pytest.mark.integration
def test_local_knowledge_is_retrieved_and_materialized_in_current_round(
    experiment_config, tmp_path: Path
) -> None:
    root = tmp_path / "local-documents"
    root.mkdir()
    (root / "generic-protein-knowledge.md").write_text(
        "Hydrophobic packing can stabilize protein cores. Binding-interface mutations may "
        "change affinity, while epistasis can reverse an otherwise favorable residue effect.",
        encoding="utf-8",
    )
    local = LocalKnowledgeConfig(
        enabled=True,
        index_path=tmp_path / "shared-local-index.sqlite",
        roots=(
            LocalKnowledgeRootConfig(
                path=root,
                access_policy_mode="synthetic_test",
                runtime_manifest_mode="legacy_compatible",
                include=("**/*.md",),
            ),
        ),
        ingestion=LocalKnowledgeIngestionConfig(chunk_tokens=64, chunk_overlap=8),
        retrieval=LocalKnowledgeRetrievalConfig(mode="lexical", top_k=4),
        leakage_guard=LeakageGuardConfig(enabled=False),
    )
    config = replace(
        experiment_config,
        rounds=1,
        budget_per_round=2,
        candidate_limit=24,
        output_root=tmp_path / "runs",
        run_label="local-rag",
        knowledge=replace(experiment_config.knowledge, local_knowledge=local),
        kg_interaction=replace(experiment_config.kg_interaction, max_tool_calls=6),
    )

    summary = run_campaign(config)
    run_dir = config.output_root / summary["run_id"]
    retrieval = json.loads(
        (run_dir / "round_01/local_rag_retrieval.json").read_text(encoding="utf-8")
    )
    evidence = json.loads(
        (run_dir / "round_01/local_rag_evidence.json").read_text(encoding="utf-8")
    )
    interaction = json.loads(
        (run_dir / "round_01/kg_interaction.json").read_text(encoding="utf-8")
    )

    assert retrieval["chunks"]
    assert evidence
    assert all(item["contributes_to_selection"] is False for item in evidence)
    assert "query_local_knowledge" in [item["operator"] for item in interaction["packs"]]

    connection = sqlite3.connect(run_dir / "structured_kg.sqlite")
    try:
        entity_types = {
            row[0] for row in connection.execute("SELECT DISTINCT entity_type FROM entities")
        }
        predicates = {
            row[0] for row in connection.execute("SELECT DISTINCT predicate FROM relations")
        }
    finally:
        connection.close()
    assert {"Document", "DocumentChunk", "Claim", "Evidence"}.issubset(entity_types)
    assert {"HAS_CHUNK", "ASSERTS", "SUPPORTED_BY_SOURCE"}.issubset(predicates)


@pytest.mark.integration
@pytest.mark.leakage
def test_campaign_leakage_guard_excludes_target_knowledge_from_rag_and_local_kg(
    experiment_config, tmp_path: Path
) -> None:
    root = tmp_path / "guarded-documents"
    root.mkdir()
    (root / "target.md").write_text(
        "GB1 has target-specific binding measurements and mutation rankings.",
        encoding="utf-8",
    )
    (root / "generic.md").write_text(
        "General protein structure and stability can depend on hydrophobic packing and epistasis.",
        encoding="utf-8",
    )
    local = LocalKnowledgeConfig(
        enabled=True,
        index_path=tmp_path / "guarded-index.sqlite",
        roots=(
            LocalKnowledgeRootConfig(
                path=root,
                access_policy_mode="synthetic_test",
                runtime_manifest_mode="legacy_compatible",
                include=("**/*.md",),
            ),
        ),
        ingestion=LocalKnowledgeIngestionConfig(chunk_tokens=64, chunk_overlap=8),
        retrieval=LocalKnowledgeRetrievalConfig(mode="lexical", top_k=4),
        leakage_guard=LeakageGuardConfig(
            enabled=True,
            protected_aliases=("GB1", "protein G B1 domain"),
            strict_aliases_required=True,
        ),
    )
    config = replace(
        experiment_config,
        rounds=1,
        budget_per_round=2,
        candidate_limit=24,
        output_root=tmp_path / "runs",
        run_label="guarded-local-rag",
        task=replace(
            experiment_config.task,
            protein_name="immunoglobulin-binding domain B1 of protein G",
            protein_aliases=("GB1", "protein G B1 domain"),
        ),
        knowledge=replace(experiment_config.knowledge, local_knowledge=local),
        kg_interaction=replace(experiment_config.kg_interaction, max_tool_calls=6),
    )

    summary = run_campaign(config)
    run_dir = config.output_root / summary["run_id"]
    rag_artifact = (run_dir / "round_01/local_rag_retrieval.json").read_text(
        encoding="utf-8"
    )
    assert "gb1" not in rag_artifact.casefold()
    assert "target-specific" not in rag_artifact.casefold()

    connection = sqlite3.connect(run_dir / "structured_kg.sqlite")
    try:
        local_properties = [
            row[0]
            for row in connection.execute(
                "SELECT properties_json FROM entities WHERE source_group = 'local_documents'"
            )
        ]
    finally:
        connection.close()
    assert local_properties
    assert all("gb1" not in item.casefold() for item in local_properties)
    assert all("target-specific" not in item.casefold() for item in local_properties)
