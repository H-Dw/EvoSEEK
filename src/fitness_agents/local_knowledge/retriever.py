from __future__ import annotations

import hashlib
from collections import defaultdict

from fitness_agents.config import LocalKnowledgeConfig

from .contracts import KnowledgeClaim, RetrievalRequest, RetrievalResult, RetrievedChunk
from .index import SQLiteLocalKnowledgeIndex
from .leakage import TargetLeakageGuard
from .prompt_safety import instruction_like_markers
from .protocols import EmbeddingBackend, RerankerBackend


class LocalHybridRetriever:
    name = "local-hybrid-retriever:v1"

    def __init__(
        self,
        index: SQLiteLocalKnowledgeIndex,
        config: LocalKnowledgeConfig,
        *,
        guard: TargetLeakageGuard,
        embedding_backend: EmbeddingBackend | None = None,
        reranker_backend: RerankerBackend | None = None,
    ) -> None:
        self.index = index
        self.config = config
        self.guard = guard
        self.embedding_backend = embedding_backend
        self.reranker_backend = reranker_backend

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        original_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
        raw_knowledge_types = request.filters.get("knowledge_types", ())
        if isinstance(raw_knowledge_types, str):
            raw_knowledge_types = (raw_knowledge_types,)
        knowledge_types = tuple(
            dict.fromkeys(
                str(item).strip().casefold()
                for item in raw_knowledge_types
                if str(item).strip()
            )
        )
        generic_terms = (
            request.policy_context.generic_terms
            if request.policy_context is not None
            else request.anchors
        )
        decision = self.guard.sanitize_query(request.query, generic_terms=generic_terms)
        policy_decision = {
            **decision.public_dict(),
            "filters": {"knowledge_types": list(knowledge_types)},
        }
        if not decision.allowed:
            result = RetrievalResult(
                query_id=request.query_id,
                round_id=request.round_id,
                original_query_hash=original_hash,
                sanitized_query="",
                policy_decision=policy_decision,
                chunks=(),
                claims=(),
                warnings=("query_rejected_by_target_leakage_guard",),
                index_manifest_hash=self.index.manifest_hash,
            )
            self._record(request, result)
            return result

        mode = self.config.retrieval.mode
        rankings: list[tuple[str, tuple[tuple[str, float], ...]]] = []
        if mode in {"lexical", "hybrid"}:
            rankings.append(
                (
                    "lexical",
                    self.index.lexical_search(
                        decision.sanitized_query,
                        limit=self.config.retrieval.lexical_candidates,
                        knowledge_types=knowledge_types,
                    ),
                )
            )
        if mode in {"dense", "hybrid"}:
            if self.embedding_backend is None:
                raise RuntimeError("Dense retrieval requested without an embedding backend")
            rankings.append(
                (
                    "dense",
                    self.index.dense_search(
                        decision.sanitized_query,
                        limit=self.config.retrieval.dense_candidates,
                        embedding_backend=self.embedding_backend,
                        knowledge_types=knowledge_types,
                    ),
                )
            )
        fused: dict[str, float] = defaultdict(float)
        score_details: dict[str, dict[str, float]] = defaultdict(dict)
        for channel, ranking in rankings:
            for rank, (chunk_id, raw_score) in enumerate(ranking, start=1):
                fused[chunk_id] += 1.0 / (60.0 + rank)
                score_details[chunk_id][channel] = raw_score
                score_details[chunk_id][f"{channel}_rank"] = float(rank)
        ordered_ids = tuple(
            item[0] for item in sorted(fused.items(), key=lambda item: (-item[1], item[0]))
        )
        records = self.index.get_chunks(ordered_ids)
        if self.reranker_backend is not None and ordered_ids:
            rerank_ids = tuple(item for item in ordered_ids if item in records)
            rerank_scores = self.reranker_backend.score(
                decision.sanitized_query, [records[item]["text"] for item in rerank_ids]
            )
            if len(rerank_scores) != len(rerank_ids):
                raise RuntimeError("Reranker backend returned an unexpected score count")
            for chunk_id, score in zip(rerank_ids, rerank_scores, strict=True):
                score_details[chunk_id]["reranker"] = float(score)
            ordered_ids = tuple(
                sorted(
                    rerank_ids,
                    key=lambda item: (-score_details[item]["reranker"], -fused[item], item),
                )
            )
        selected: list[RetrievedChunk] = []
        warnings: list[str] = []
        by_document: dict[str, int] = defaultdict(int)
        used_tokens = 0
        for chunk_id in ordered_ids:
            record = records.get(chunk_id)
            if record is None:
                continue
            matches = self.guard.validate_result(
                text=record["text"], path=record["artifact_uri"]
            )
            if matches:
                warnings.append(f"result_quarantined:{chunk_id}")
                continue
            document_id = record["document_id"]
            if by_document[document_id] >= self.config.retrieval.max_chunks_per_document:
                continue
            if used_tokens + record["token_count"] > request.token_budget and selected:
                continue
            scores = {**score_details[chunk_id], "rrf": fused[chunk_id]}
            instruction_markers = instruction_like_markers(record["text"])
            if instruction_markers:
                warnings.append(f"instruction_like_content:{chunk_id}")
            selected.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    text=record["text"],
                    artifact_uri=record["artifact_uri"],
                    section_path=record["section_path"],
                    start_offset=record["start_offset"],
                    end_offset=record["end_offset"],
                    source_group=record["source_group"],
                    knowledge_type=record["knowledge_type"],
                    scores=scores,
                    provenance={
                        "file_hash": record["file_hash"],
                        "knowledge_type": record["knowledge_type"],
                        "metadata": record["metadata"],
                        "retriever": self.name,
                        "query_id": request.query_id,
                        "instruction_like_markers": instruction_markers,
                    },
                )
            )
            used_tokens += record["token_count"]
            by_document[document_id] += 1
            if len(selected) >= min(request.top_k, self.config.retrieval.top_k):
                break

        claims = tuple(
            KnowledgeClaim(
                claim_id=f"claim:{hashlib.sha256(f'retrieval-only:v1|{item.chunk_id}|{item.text}'.encode()).hexdigest()[:24]}",
                statement=item.text,
                subject=None,
                predicate=None,
                object=None,
                polarity="neutral",
                applicability={
                    "scope": "retrieved_context",
                    "verified": False,
                    "knowledge_type": item.knowledge_type,
                },
                confidence=min(1.0, max(0.0, item.scores.get("rrf", 0.0) * 60.0)),
                evidence_chunk_ids=(item.chunk_id,),
            )
            for item in selected[: self.config.kg_update.max_claims_per_round]
        )
        result = RetrievalResult(
            query_id=request.query_id,
            round_id=request.round_id,
            original_query_hash=original_hash,
            sanitized_query=decision.sanitized_query,
            policy_decision=policy_decision,
            chunks=tuple(selected),
            claims=claims,
            warnings=tuple(sorted(set(warnings))),
            index_manifest_hash=self.index.manifest_hash,
        )
        self._record(request, result)
        return result

    def _record(self, request: RetrievalRequest, result: RetrievalResult) -> None:
        self.index.record_retrieval(
            query_id=request.query_id,
            round_id=request.round_id,
            original_query_hash=result.original_query_hash,
            sanitized_query=result.sanitized_query,
            policy=result.policy_decision,
            result_chunk_ids=tuple(item.chunk_id for item in result.chunks),
        )
