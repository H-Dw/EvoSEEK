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
        index_path=tmp_path / "directed-evolution.sqlite",
        corpus_index_path=tmp_path / "directed-evolution.sqlite",
        retrieval_overlay_path=tmp_path / "gb1-overlay.sqlite",
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
    }
    assert report.indexed_documents >= len(expected_types)
    assert report.quarantined_documents == 0
    assert expected_types.issubset(stats["knowledge_types"])
    assert result.chunks
    assert {item.knowledge_type for item in result.chunks} == {"sequence_safeguards"}
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
        for knowledge_type in item.properties["knowledge_types"]
    }
    assert document_types == {"sequence_safeguards"}
    assert claim_types == {"sequence_safeguards"}
    entity_types = {item.entity_type for item in built.snapshot.entities}
    predicates = {item.predicate for item in built.snapshot.relations}
    assert {"Publication", "CitationSupport"}.issubset(entity_types)
    assert {"SUPPORTED_BY_CITATION", "CITES_PUBLICATION"}.issubset(predicates)
