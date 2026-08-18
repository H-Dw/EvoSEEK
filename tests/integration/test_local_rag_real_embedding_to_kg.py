from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from fitness_agents.config import load_experiment_config
from fitness_agents.kg_knowledge import (
    BuildContext,
    KnowledgeGraphBuilder,
    LocalRAGKnowledgeAdapter,
    SQLiteGraphSink,
)
from fitness_agents.local_knowledge import LocalKnowledgeBase
from fitness_agents.plugin_registry import PluginRegistry


@pytest.mark.integration
def test_real_sentence_transformer_dense_retrieval_materializes_into_kg(
    tmp_path: Path,
) -> None:
    raw_model_path = os.environ.get("FITNESS_RAG_TEST_MODEL")
    if not raw_model_path:
        pytest.skip("FITNESS_RAG_TEST_MODEL is required for the real embedding integration test")
    model_path = Path(raw_model_path).resolve()
    if not model_path.is_dir():
        pytest.fail(f"FITNESS_RAG_TEST_MODEL is not a directory: {model_path}")

    experiment = load_experiment_config("configs/experiments/knowledge_agent.yaml")
    local_config = replace(
        experiment.knowledge.local_knowledge,
        index_path=tmp_path / "dense-local-knowledge.sqlite",
        corpus_index_path=tmp_path / "dense-local-knowledge.sqlite",
        retrieval_overlay_path=tmp_path / "dense-overlay.sqlite",
        retrieval=replace(
            experiment.knowledge.local_knowledge.retrieval,
            mode="hybrid",
            dense_enabled=True,
            embedding_model_path=model_path,
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
        knowledge.refresh()
        counts = knowledge.index.stats()
        assert counts["chunks"] > 0
        assert counts["embeddings"] == counts["chunks"]

        row = knowledge.index.connection.execute(
            "SELECT dimension, vector FROM embeddings LIMIT 1"
        ).fetchone()
        assert row is not None
        vector = np.frombuffer(row["vector"], dtype=np.float32)
        assert len(vector) == row["dimension"] == knowledge.embedding_backend.dimension
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-4)

        result = knowledge.retrieve(
            query="Why can combining mutations produce negative epistasis?",
            intent="diagnostic",
            round_id=2,
            top_k=5,
        )
        assert result.chunks
        expected = [
            item
            for item in result.chunks[:3]
            if item.knowledge_type == "history_guided_combination"
        ]
        assert expected
        assert "dense" in expected[0].scores

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
                run_id="real-embedding-test",
                round_id=2,
                protein_id=experiment.task.protein_id,
                resources={"local_retrieval_results": (result,)},
            )
        )
        entity_types = {item.entity_type for item in built.snapshot.entities}
        predicates = {item.predicate for item in built.snapshot.relations}
        assert {"Document", "DocumentChunk", "Claim", "Evidence"}.issubset(entity_types)
        assert {"HAS_CHUNK", "ASSERTS", "SUPPORTED_BY_SOURCE", "DERIVED_FROM"}.issubset(
            predicates
        )
        assert sink.query_claims(query="epistasis", round_id=2, limit=8)
        assert not sink.query_claims(query="epistasis", round_id=1, limit=8)
    finally:
        sink.close()
        knowledge.close()


def test_current_selection_flag_is_explicitly_rejected_before_calibration() -> None:
    experiment = load_experiment_config("configs/experiments/knowledge_agent.yaml")
    with pytest.raises(
        ValueError,
        match="requires calibrated_candidate_projection",
    ):
        replace(
            experiment.knowledge.local_knowledge.kg_update,
            contributes_to_selection=True,
        )
