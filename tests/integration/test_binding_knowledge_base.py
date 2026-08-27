from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fitness_agents.config import load_experiment_config
from fitness_agents.local_knowledge import LocalKnowledgeBase


def test_binding_corpus_is_an_independent_typed_retrieval_root(tmp_path) -> None:
    experiment = load_experiment_config("configs/experiments/knowledge_agent_rag.yaml")
    binding_roots = tuple(
        replace(
            root,
            access_policy_mode="synthetic_test",
            runtime_manifest_mode="legacy_compatible",
        )
        for root in experiment.knowledge.local_knowledge.roots
        if Path(root.path).as_posix().endswith("resources/local_knowledge/binding")
    )
    assert len(binding_roots) == 1

    local_config = replace(
        experiment.knowledge.local_knowledge,
        enabled=True,
        index_path=tmp_path / "binding.sqlite",
        corpus_index_path=tmp_path / "binding.sqlite",
        retrieval_overlay_path=tmp_path / "binding-overlay.sqlite",
        roots=binding_roots,
        retrieval=replace(
            experiment.knowledge.local_knowledge.retrieval,
            mode="lexical",
            dense_enabled=False,
            embedding_model_path=None,
            strict_query_language=True,
        ),
    )
    knowledge = LocalKnowledgeBase(
        local_config,
        index_path=local_config.index_path,
        protein_id=experiment.task.protein_id,
        protein_name=experiment.task.protein_name,
        protein_aliases=experiment.task.protein_aliases,
        protein_accessions=experiment.task.protein_accessions,
        reference_sequence=experiment.task.reference_sequence,
    )
    queries = {
        "binding_campaign_definition": (
            "Which binding quantity should be defined before affinity maturation?"
        ),
        "binding_affinity_measurement": (
            "How should multiple ligand concentrations be used to fit KD?"
        ),
        "binding_display_selection": (
            "Which cell-free display supports a transformation-limited library?"
        ),
        "binding_library_design": (
            "How should alanine scanning identify permissive interface positions?"
        ),
        "binding_kinetic_selection": (
            "How should an unlabeled ligand chase prevent rebinding during off-rate selection?"
        ),
        "binding_specificity_counterselection": (
            "How should off-target homologs and matrix components be counterselected?"
        ),
        "binding_round_decision": (
            "How should sequence enrichment be normalized before selecting clones?"
        ),
        "binding_developability_validation": (
            "Which affinity, specificity, aggregation, and stability measurements advance a clone?"
        ),
    }
    try:
        report = knowledge.refresh()
        stats = knowledge.index.stats()
        results = {
            knowledge_type: knowledge.retrieve(
                query=query,
                intent="constraint",
                round_id=1,
                knowledge_types=(knowledge_type,),
            )
            for knowledge_type, query in queries.items()
        }
    finally:
        knowledge.close()

    assert report.indexed_documents == 38
    assert report.quarantined_documents == 0
    assert set(stats["knowledge_types"]) == set(queries)
    for knowledge_type, result in results.items():
        assert result.chunks
        assert {chunk.knowledge_type for chunk in result.chunks} == {knowledge_type}
        for chunk in result.chunks:
            assert "resources\\local_knowledge\\binding" in chunk.artifact_uri
            assert chunk.provenance["metadata"]["front_matter"]["corpus_layer"] == "binding"
