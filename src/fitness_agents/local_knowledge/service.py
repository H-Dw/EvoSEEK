from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fitness_agents.config import LocalKnowledgeConfig
from fitness_agents.contracts.schemas import Evidence

from .contracts import LeakagePolicyContext, RetrievalRequest, RetrievalResult
from .embeddings import (
    SentenceTransformerEmbeddingBackend,
    SentenceTransformerRerankerBackend,
)
from .index import SQLiteLocalKnowledgeIndex
from .leakage import TargetLeakageGuard
from .retriever import LocalHybridRetriever

DEFAULT_GENERIC_TERMS = (
    "protein structure and stability",
    "binding interface mutation effects",
    "physicochemical substitution mechanisms",
    "epistasis and residue interactions",
    "protein property optimization",
)


class LocalKnowledgeBase:
    name = "local_knowledge"

    def __init__(
        self,
        config: LocalKnowledgeConfig,
        *,
        index_path: str | Path,
        protein_id: str,
        protein_name: str | None = None,
        protein_aliases: tuple[str, ...] = (),
        protein_accessions: tuple[str, ...] = (),
        reference_sequence: str | None = None,
    ) -> None:
        if not config.enabled:
            raise ValueError("LocalKnowledgeBase requires enabled configuration")
        self.config = config
        self.protein_id = protein_id
        self.guard = TargetLeakageGuard(
            config.leakage_guard,
            protein_name=protein_name,
            protein_id=protein_id,
            aliases=protein_aliases,
            accessions=protein_accessions,
            reference_sequence=reference_sequence,
        )
        self.embedding_backend = None
        self.reranker_backend = None
        if config.retrieval.dense_enabled:
            self.embedding_backend = SentenceTransformerEmbeddingBackend(
                config.retrieval.embedding_model_path  # type: ignore[arg-type]
            )
        if config.retrieval.reranker_model_path is not None:
            self.reranker_backend = SentenceTransformerRerankerBackend(
                config.retrieval.reranker_model_path
            )
        self.index = SQLiteLocalKnowledgeIndex(index_path)
        self.retriever = LocalHybridRetriever(
            self.index,
            config,
            guard=self.guard,
            embedding_backend=self.embedding_backend,
            reranker_backend=self.reranker_backend,
        )
        self.last_build_report = None

    def refresh(self):
        self.last_build_report = self.index.build(
            self.config,
            guard=self.guard,
            embedding_backend=self.embedding_backend,
        )
        return self.last_build_report

    def retrieve(
        self,
        *,
        query: str,
        intent: str,
        round_id: int,
        anchors: Sequence[str] = DEFAULT_GENERIC_TERMS,
        top_k: int | None = None,
        token_budget: int | None = None,
        knowledge_types: Sequence[str] = (),
    ) -> RetrievalResult:
        anchor_tuple = tuple(str(item) for item in anchors)
        knowledge_type_tuple = tuple(
            dict.fromkeys(
                str(item).strip().casefold()
                for item in knowledge_types
                if str(item).strip()
            )
        )
        query_payload = json.dumps(
            [
                self.index.manifest_hash,
                round_id,
                intent,
                query,
                anchor_tuple,
                knowledge_type_tuple,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        query_id = f"localq:{hashlib.sha256(query_payload.encode()).hexdigest()[:20]}"
        policy_context = LeakagePolicyContext(
            enabled=self.guard.enabled,
            policy_version="target-leakage-guard:v1",
            protected_terms_hash=self.guard.protected_terms_hash,
            generic_terms=anchor_tuple,
        )
        return self.retriever.retrieve(
            RetrievalRequest(
                query_id=query_id,
                round_id=round_id,
                intent=intent,
                query=query,
                anchors=anchor_tuple,
                top_k=top_k or self.config.retrieval.top_k,
                token_budget=token_budget or self.config.retrieval.token_budget,
                filters={"knowledge_types": knowledge_type_tuple},
                policy_context=policy_context,
            )
        )

    def prefetch_round(
        self,
        *,
        round_id: int,
        objective: str,
        assay_conditions: dict[str, Any] | None = None,
        anchors: Sequence[str] = (),
    ) -> RetrievalResult:
        generic = tuple(dict.fromkeys((*DEFAULT_GENERIC_TERMS, *(str(item) for item in anchors))))
        condition_terms = tuple(
            f"assay condition {key} {value}"
            for key, value in sorted((assay_conditions or {}).items())
        )
        if self.guard.enabled:
            query = "; ".join((f"optimization objective {objective}", *condition_terms, *generic))
        else:
            query = "; ".join(
                (f"protein {self.protein_id}", f"optimization objective {objective}", *condition_terms, *generic)
            )
        return self.retrieve(
            query=query,
            intent="round_prefetch",
            round_id=round_id,
            anchors=generic,
        )

    def evidence_from_result(self, result: RetrievalResult) -> tuple[Evidence, ...]:
        claims_by_chunk = {
            chunk_id: claim
            for claim in result.claims
            for chunk_id in claim.evidence_chunk_ids
        }
        output = []
        for chunk in result.chunks:
            claim = claims_by_chunk.get(chunk.chunk_id)
            score = float(chunk.scores.get("rrf", 0.0))
            confidence = min(1.0, max(0.05, score * 60.0))
            output.append(
                Evidence(
                    evidence_id=f"ev:local_rag:{chunk.chunk_id.split(':', 1)[-1]}",
                    variant_id=f"context:{self.protein_id}",
                    channel="local_rag",
                    statement=chunk.text,
                    score=0.0,
                    source_id=f"localdoc:{chunk.document_id}:{chunk.chunk_id}",
                    confidence=confidence,
                    round_id=result.round_id,
                    evidence_type="retrieved_document",
                    raw_features={
                        "retrieval_scores": chunk.scores,
                        "knowledge_type": chunk.knowledge_type,
                    },
                    quality_status="unverified",
                    applicability="generic_or_other_protein_context",
                    contributes_to_selection=self.config.kg_update.contributes_to_selection,
                    warnings=(
                        "retrieved_context_not_causal",
                        "cross_context_applicability_requires_review",
                        *result.warnings,
                    ),
                    provenance={
                        **chunk.provenance,
                        "knowledge_type": chunk.knowledge_type,
                        "artifact_uri": chunk.artifact_uri,
                        "artifact_span": [chunk.start_offset, chunk.end_offset],
                        "section_path": list(chunk.section_path),
                        "index_manifest_hash": result.index_manifest_hash,
                        "policy_decision": result.policy_decision,
                        "sanitized_query": result.sanitized_query,
                    },
                    claim_id=claim.claim_id if claim else None,
                    polarity=claim.polarity if claim else "neutral",
                    source_group=chunk.source_group,
                    artifact_uri=chunk.artifact_uri,
                    artifact_span=(chunk.start_offset, chunk.end_offset),
                    valid_from_round=result.round_id,
                )
            )
        return tuple(output)

    def close(self) -> None:
        self.index.close()
