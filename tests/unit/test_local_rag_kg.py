from __future__ import annotations

from pathlib import Path

from fitness_agents.config import (
    LeakageGuardConfig,
    LocalKnowledgeConfig,
    LocalKnowledgeIngestionConfig,
    LocalKnowledgeRetrievalConfig,
    LocalKnowledgeRootConfig,
)
from fitness_agents.kg_interaction import (
    KGQueryContext,
    KGQueryStep,
    LocalKnowledgeQueryOperator,
    QueryIntent,
    StructuredClaimQueryOperator,
)
from fitness_agents.kg_knowledge import (
    BuildContext,
    KnowledgeGraphBuilder,
    LocalRAGKnowledgeAdapter,
    SQLiteGraphSink,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.plugin_registry import PluginRegistry


def _knowledge(tmp_path: Path) -> LocalKnowledgeBase:
    root = tmp_path / "docs"
    root.mkdir()
    (root / "mechanism.md").write_text(
        """---
title: Amino-acid properties
knowledge_type: amino_acid_properties
evidence_level: textbook_and_primary
citation_keys: [Grantham1974]
---
Hydrophobic substitutions can improve packing, while steric clashes may reduce stability.
""",
        encoding="utf-8",
    )
    config = LocalKnowledgeConfig(
        enabled=True,
        roots=(LocalKnowledgeRootConfig(path=root, include=("**/*.md",)),),
        ingestion=LocalKnowledgeIngestionConfig(chunk_tokens=64, chunk_overlap=8),
        retrieval=LocalKnowledgeRetrievalConfig(mode="lexical", top_k=3),
        leakage_guard=LeakageGuardConfig(enabled=False),
    )
    knowledge = LocalKnowledgeBase(
        config,
        index_path=tmp_path / "local.sqlite",
        protein_id="TARGET",
    )
    knowledge.refresh()
    return knowledge


def test_local_retrieval_materializes_round_visible_claim_graph(tmp_path: Path) -> None:
    knowledge = _knowledge(tmp_path)
    try:
        result = knowledge.retrieve(
            query="hydrophobic packing stability",
            intent="support",
            round_id=2,
        )
        registry = PluginRegistry("knowledge_adapter")
        registry.register("local_rag", LocalRAGKnowledgeAdapter(knowledge.guard))
        sink = SQLiteGraphSink(tmp_path / "structured.sqlite")
        try:
            built = KnowledgeGraphBuilder(registry, sinks=(sink,), strict=True).build(
                BuildContext(
                    run_id="run-local",
                    round_id=2,
                    protein_id="TARGET",
                    resources={"local_retrieval_results": (result,)},
                )
            )
            claims = sink.query_claims(query="hydrophobic", round_id=2, limit=4)
            earlier_claims = sink.query_claims(query="hydrophobic", round_id=1, limit=4)
            snapshots = sink.list_snapshots()
        finally:
            sink.close()
    finally:
        knowledge.close()

    entity_types = {item.entity_type for item in built.snapshot.entities}
    predicates = {item.predicate for item in built.snapshot.relations}
    assert {"Document", "DocumentChunk", "Claim", "Evidence"}.issubset(entity_types)
    assert {"HAS_CHUNK", "ASSERTS", "SUPPORTED_BY_SOURCE", "DERIVED_FROM"}.issubset(
        predicates
    )
    assert claims
    assert not earlier_claims
    assert snapshots[0]["snapshot_id"].startswith("kgsnapshot:")
    assert snapshots[0]["round_id"] == 2
    document = next(
        item for item in built.snapshot.entities if item.entity_type == "Document"
    )
    chunk = next(
        item for item in built.snapshot.entities if item.entity_type == "DocumentChunk"
    )
    claim = next(item for item in built.snapshot.entities if item.entity_type == "Claim")
    assert document.properties["knowledge_type"] == "amino_acid_properties"
    assert document.properties["metadata"]["citation_keys"] == ["Grantham1974"]
    assert chunk.properties["knowledge_type"] == "amino_acid_properties"
    assert claim.properties["knowledge_types"] == ["amino_acid_properties"]


class _EngineFacade:
    def __init__(self, knowledge: LocalKnowledgeBase, sink: SQLiteGraphSink) -> None:
        self.knowledge = knowledge
        self.sink = sink

    def retrieve_local_knowledge(self, **kwargs):
        result = self.knowledge.retrieve(
            query=kwargs["query"],
            intent=kwargs["intent"],
            round_id=kwargs["round_id"],
            anchors=kwargs["anchors"],
            top_k=kwargs["top_k"],
            knowledge_types=kwargs.get("knowledge_types", ()),
        )
        return result, self.knowledge.evidence_from_result(result)

    def query_structured_claims(self, **kwargs):
        return self.sink.query_claims(**kwargs)


def test_local_and_structured_claim_operators_return_evidence_packs(tmp_path: Path) -> None:
    knowledge = _knowledge(tmp_path)
    sink = SQLiteGraphSink(tmp_path / "operator-structured.sqlite")
    facade = _EngineFacade(knowledge, sink)
    context = KGQueryContext(run_id="run", round_id=1, max_rows=4)
    try:
        local_pack = LocalKnowledgeQueryOperator(facade).execute(
            KGQueryStep(
                "local",
                "query_local_knowledge",
                QueryIntent.SUPPORT,
                {
                    "query": "hydrophobic packing",
                    "knowledge_types": ["amino_acid_properties"],
                    "limit": 3,
                },
            ),
            context,
        )
        assert local_pack.evidence
        assert local_pack.metadata["index_manifest_hash"] != "unbuilt"
        assert local_pack.metadata["knowledge_types"] == ["amino_acid_properties"]

        result = knowledge.retrieve(
            query="hydrophobic packing",
            intent="support",
            round_id=1,
        )
        registry = PluginRegistry("knowledge_adapter")
        registry.register("local_rag", LocalRAGKnowledgeAdapter(knowledge.guard))
        KnowledgeGraphBuilder(registry, sinks=(sink,), strict=True).build(
            BuildContext(
                run_id="run",
                round_id=1,
                protein_id="TARGET",
                resources={"local_retrieval_results": (result,)},
            )
        )
        claim_pack = StructuredClaimQueryOperator(facade).execute(
            KGQueryStep(
                "claims",
                "query_structured_claims",
                QueryIntent.SUPPORT,
                {"query": "hydrophobic", "limit": 3},
            ),
            context,
        )
        assert claim_pack.facts
    finally:
        sink.close()
        knowledge.close()
