from __future__ import annotations

from dataclasses import replace

from fitness_agents.config import load_experiment_config
from fitness_agents.kg_knowledge import (
    BuildContext,
    KnowledgeGraphBuilder,
    LocalRAGKnowledgeAdapter,
    SQLiteGraphSink,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.plugin_registry import PluginRegistry


def test_directed_evolution_corpus_flows_from_typed_rag_into_kg(tmp_path) -> None:
    experiment = load_experiment_config("configs/experiments/knowledge_agent.yaml")
    local_config = replace(
        experiment.knowledge.local_knowledge,
        enabled=True,
        index_path=tmp_path / "directed-evolution.sqlite",
        corpus_index_path=tmp_path / "directed-evolution.sqlite",
        retrieval_overlay_path=tmp_path / "gb1-overlay.sqlite",
        roots=tuple(
            replace(
                root,
                access_policy_mode="synthetic_test",
                runtime_manifest_mode="legacy_compatible",
            )
            for root in experiment.knowledge.local_knowledge.roots
        ),
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
    sink = SQLiteGraphSink(tmp_path / "structured.sqlite")
    try:
        report = knowledge.refresh()
        stats = knowledge.index.stats()
        result = knowledge.retrieve(
            query="How can reduced codon redundancy lower saturation screening effort?",
            intent="constraint",
            round_id=1,
            knowledge_types=("sequence_safeguards",),
        )
        operational_queries = {
            "assay_engineering": (
                "How should positive and negative controls and Z-prime validate a screen?"
            ),
            "random_mutagenesis_operations": (
                "How should an error-prone PCR pilot measure mutations per gene?"
            ),
            "saturation_mutagenesis_operations": (
                "How should pooled sequencing quality control verify randomized codons?"
            ),
            "specificity_engineering": (
                "How should positive and negative selection evolve substrate specificity?"
            ),
            "machine_learning_operations": (
                "How should uncertainty and predicted fitness select an active-learning batch?"
            ),
            "round_decision_operations": (
                "How should an uncertain substitution be retested in a combinatorial library?"
            ),
        }
        operational_results = {
            knowledge_type: knowledge.retrieve(
                query=query,
                intent="constraint",
                round_id=1,
                knowledge_types=(knowledge_type,),
            )
            for knowledge_type, query in operational_queries.items()
        }
        registry = PluginRegistry("knowledge_adapter")
        registry.register(
            "local_rag",
            LocalRAGKnowledgeAdapter(
                knowledge.guard,
                publication_catalog=knowledge.publication_catalog,
            ),
        )
        built = KnowledgeGraphBuilder(registry, sinks=(sink,), strict=True).build(
            BuildContext(
                run_id="typed-directed-evolution-corpus",
                round_id=1,
                protein_id=experiment.task.protein_id,
                resources={"local_retrieval_results": (result,)},
            )
        )
        claims = sink.query_claims(query="codon redundancy", round_id=1, limit=8)
    finally:
        sink.close()
        knowledge.close()

    expected_types = {
        "amino_acid_properties",
        "substitution_conservativeness",
        "structure_context",
        "mutation_burden",
        "sequence_safeguards",
        "directed_evolution_strategy",
        "history_guided_combination",
        "evidence_applicability",
        "assay_engineering",
        "campaign_definition",
        "continuous_evolution_operations",
        "droplet_screening_operations",
        "machine_learning_operations",
        "random_mutagenesis_operations",
        "recombination_operations",
        "round_decision_operations",
        "saturation_mutagenesis_operations",
        "specificity_engineering",
        "stability_evolvability_operations",
    }
    assert report.indexed_documents >= len(expected_types)
    assert report.quarantined_documents == 0
    assert expected_types.issubset(stats["knowledge_types"])
    assert result.chunks
    assert {item.knowledge_type for item in result.chunks} == {"sequence_safeguards"}
    for knowledge_type, operational_result in operational_results.items():
        assert operational_result.chunks
        assert {item.knowledge_type for item in operational_result.chunks} == {
            knowledge_type
        }
    assert claims
    document_types = {
        item.properties["knowledge_type"]
        for item in built.snapshot.entities
        if item.entity_type == "Document"
    }
    claim_types = {
        knowledge_type
        for item in built.snapshot.entities
        if item.entity_type == "Claim"
        for knowledge_type in item.properties.get(
            "knowledge_types",
            (item.properties.get("knowledge_type"),),
        )
        if knowledge_type
    }
    assert document_types == {"sequence_safeguards"}
    assert claim_types == {"sequence_safeguards"}
    entity_types = {item.entity_type for item in built.snapshot.entities}
    predicates = {item.predicate for item in built.snapshot.relations}
    assert {"Publication", "CitationSupport"}.issubset(entity_types)
    assert {"SUPPORTED_BY_CITATION", "CITES_PUBLICATION"}.issubset(predicates)
