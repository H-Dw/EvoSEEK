from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from fitness_agents.config import LocalKnowledgeConfig

from .contracts import (
    KnowledgeClaim,
    KnowledgeRecord,
    RetrievalRequest,
    RetrievalResult,
    RetrievedChunk,
)
from .index import SQLiteLocalKnowledgeIndex
from .leakage import TargetLeakageGuard
from .overlay import SQLiteRetrievalOverlay
from .prompt_safety import instruction_like_markers
from .protocols import EmbeddingBackend, RerankerBackend


class LocalHybridRetriever:
    name = "local-hybrid-retriever:v2"

    def __init__(
        self,
        index: SQLiteLocalKnowledgeIndex,
        overlay: SQLiteRetrievalOverlay,
        config: LocalKnowledgeConfig,
        *,
        guard: TargetLeakageGuard,
        embedding_backend: EmbeddingBackend | None = None,
        reranker_backend: RerankerBackend | None = None,
    ) -> None:
        self.index = index
        self.overlay = overlay
        self.config = config
        self.guard = guard
        self.embedding_backend = embedding_backend
        self.reranker_backend = reranker_backend

    @staticmethod
    def _retrieval_confidence(
        scores: dict[str, float],
        threshold: float,
        *,
        reranker_score_kind: str = "raw_logit",
    ) -> float:
        if "reranker" in scores:
            if reranker_score_kind == "probability":
                return min(1.0, max(0.0, scores["reranker"]))
            return min(1.0, max(0.0, 1.0 / (1.0 + pow(2.718281828, -scores["reranker"]))))
        dense = scores.get("dense")
        if dense is not None:
            denominator = max(1e-6, 1.0 - threshold)
            return min(1.0, max(0.0, (dense - threshold) / denominator))
        return 0.20

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        original_hash = hashlib.sha256(request.query.encode("utf-8")).hexdigest()
        if self.config.retrieval.strict_query_language and re.search(
            r"[\u3400-\u9fff]", request.query
        ):
            result = RetrievalResult(
                query_id=request.query_id,
                round_id=request.round_id,
                original_query_hash=original_hash,
                sanitized_query="",
                policy_decision={"allowed": False, "reason": "query_language_must_be_english"},
                chunks=(),
                claims=(),
                warnings=("query_language_must_be_english",),
                index_manifest_hash=self.index.manifest_hash,
            )
            self._record(request, result)
            return result

        raw_knowledge_types = request.filters.get("knowledge_types", ())
        if isinstance(raw_knowledge_types, str):
            raw_knowledge_types = (raw_knowledge_types,)
        knowledge_types = tuple(
            dict.fromkeys(
                str(item).strip().casefold() for item in raw_knowledge_types if str(item).strip()
            )
        )
        facet_filters: dict[str, tuple[str, ...]] = {}
        for name, raw_values in request.filters.items():
            if name == "knowledge_types":
                continue
            values = raw_values if isinstance(raw_values, (list, tuple, set)) else (raw_values,)
            normalized = tuple(
                dict.fromkeys(str(item).strip() for item in values if str(item).strip())
            )
            if normalized:
                facet_filters[str(name)] = normalized
        generic_terms = (
            request.policy_context.generic_terms
            if request.policy_context is not None
            else request.anchors
        )
        decision = self.guard.sanitize_query(request.query, generic_terms=generic_terms)
        policy_decision = {
            **decision.public_dict(),
            "filters": {
                "knowledge_types": list(knowledge_types),
                **{key: list(value) for key, value in facet_filters.items()},
            },
            "corpus_manifest_hash": self.index.manifest_hash,
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
                        facets=facet_filters,
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
                        facets=facet_filters,
                        minimum_similarity=self.config.retrieval.minimum_dense_similarity,
                        max_exact_chunks=self.config.retrieval.max_exact_dense_chunks,
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
        if mode == "hybrid" and self.config.retrieval.require_dense_match_for_hybrid:
            fused = {
                chunk_id: score
                for chunk_id, score in fused.items()
                if "dense" in score_details[chunk_id]
            }
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
            minimum = self.config.retrieval.minimum_reranker_score
            ordered_ids = tuple(
                sorted(
                    (
                        item
                        for item in rerank_ids
                        if minimum is None or score_details[item]["reranker"] >= minimum
                    ),
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
            if not self.overlay.is_document_allowed(
                record["document_id"], corpus_manifest_hash=self.index.manifest_hash
            ):
                warnings.append(f"result_blocked_by_task_overlay:{chunk_id}")
                continue
            matches = self.guard.validate_result(text=record["text"], path=record["artifact_uri"])
            if matches:
                warnings.append(f"result_quarantined:{chunk_id}")
                continue
            instruction_markers = instruction_like_markers(record["text"])
            if instruction_markers:
                warnings.append(f"instruction_like_content:{chunk_id}")
                if self.config.retrieval.instruction_content_policy == "reject":
                    continue
            document_id = record["document_id"]
            if by_document[document_id] >= self.config.retrieval.max_chunks_per_document:
                continue
            if used_tokens + record["token_count"] > request.token_budget and selected:
                continue
            scores = {**score_details[chunk_id], "rrf": fused[chunk_id]}
            scores["retrieval_confidence"] = self._retrieval_confidence(
                scores,
                self.config.retrieval.minimum_dense_similarity,
                reranker_score_kind=(
                    getattr(self.reranker_backend, "score_kind", "raw_logit")
                    if self.reranker_backend is not None
                    else "raw_logit"
                ),
            )
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
                    facets=record.get("facets", {}),
                )
            )
            used_tokens += record["token_count"]
            by_document[document_id] += 1
            if len(selected) >= min(request.top_k, self.config.retrieval.top_k):
                break
        if not selected:
            warnings.append("no_answer_above_retrieval_threshold")

        claim_rows = (
            self._claim_from_chunk(item)
            for item in selected[: self.config.kg_update.max_claims_per_round]
        )
        claims = tuple(item for item in claim_rows if item is not None)
        record_rows = (
            self._record_from_chunk(item)
            for item in selected[: self.config.kg_update.max_claims_per_round]
        )
        records = tuple(item for item in record_rows if item is not None)
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
            records=records,
        )
        self._record(request, result)
        return result

    @staticmethod
    def _claim_from_chunk(item: RetrievedChunk) -> KnowledgeClaim | None:
        metadata = item.provenance.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("record_type") == "atomic_claim":
            raw_support = metadata.get("citation_support", [])
            citation_support = tuple(
                dict(entry) for entry in raw_support if isinstance(entry, dict)
            )
            applicability = metadata.get("applicability", {})
            if not isinstance(applicability, dict):
                applicability = {"scope": str(applicability)}
            declared_confidence = float(metadata.get("confidence", 0.70))
            return KnowledgeClaim(
                claim_id=str(metadata["claim_id"]),
                statement=str(metadata["statement"]),
                subject=str(metadata["subject"]),
                predicate=str(metadata["predicate"]),
                object=str(metadata["object"]),
                polarity=str(metadata.get("polarity", "support")),
                applicability=applicability,
                # Scientific confidence belongs to the verified claim record.
                # Retrieval relevance remains available only in RetrievedChunk.scores.
                confidence=declared_confidence,
                evidence_chunk_ids=(item.chunk_id,),
                claim_kind=str(metadata.get("claim_kind", "scientific_prior")),
                citation_support=citation_support,
                selection_eligible=metadata.get("selection_eligible") is True,
                extraction_version="atomic-claim:v1",
            )
        if metadata.get("record_type") in {"logic_unit", "knowledge_decision_card"}:
            return None
        return KnowledgeClaim(
            claim_id=(
                "claim:"
                + hashlib.sha256(
                    f"retrieval-only:v2|{item.chunk_id}|{item.text}".encode()
                ).hexdigest()[:24]
            ),
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
            confidence=float(item.scores.get("retrieval_confidence", 0.0)),
            evidence_chunk_ids=(item.chunk_id,),
            extraction_version="retrieval-only:v2",
        )

    @staticmethod
    def _record_from_chunk(item: RetrievedChunk) -> KnowledgeRecord | None:
        metadata = item.provenance.get("metadata", {})
        if not isinstance(metadata, dict):
            return None
        record_type = str(metadata.get("record_type", ""))
        if record_type not in {
            "atomic_claim",
            "logic_unit",
            "knowledge_decision_card",
        }:
            return None
        record_id = str(
            metadata.get("record_id")
            or metadata.get("claim_id")
            or metadata.get("logic_unit_id")
            or metadata.get("decision_card_id")
            or ""
        )
        retrieval_text = str(
            metadata.get("retrieval_text")
            or metadata.get("statement")
            or item.text
        )
        scientific_quality = metadata.get("scientific_quality", {})
        task_applicability = metadata.get("task_applicability", {})
        boundary_conditions = metadata.get("boundary_conditions", ())
        counterclaims = metadata.get("counterclaims", ())
        abstain_if = metadata.get("abstain_if", ())
        payload = metadata.get("record_payload", {})
        return KnowledgeRecord(
            record_id=record_id,
            record_type=record_type,
            retrieval_text=retrieval_text,
            knowledge_type=item.knowledge_type,
            permission=str(metadata.get("permission", "explanation_only")),
            scientific_quality=(
                dict(scientific_quality) if isinstance(scientific_quality, dict) else {}
            ),
            task_applicability=(
                dict(task_applicability) if isinstance(task_applicability, dict) else {}
            ),
            boundary_conditions=tuple(str(value) for value in boundary_conditions),
            counterclaims=tuple(str(value) for value in counterclaims),
            abstain_if=tuple(str(value) for value in abstain_if),
            facets=item.facets,
            evidence_chunk_ids=(item.chunk_id,),
            canonical_record_hash=(
                str(metadata.get("source_record_hash"))
                if metadata.get("source_record_hash")
                else None
            ),
            payload=dict(payload) if isinstance(payload, dict) else {},
        )

    def _record(self, request: RetrievalRequest, result: RetrievalResult) -> None:
        self.overlay.record_retrieval(
            query_id=request.query_id,
            round_id=request.round_id,
            original_query_hash=result.original_query_hash,
            sanitized_query=result.sanitized_query,
            policy=result.policy_decision,
            result_chunk_ids=tuple(item.chunk_id for item in result.chunks),
            corpus_manifest_hash=self.index.manifest_hash,
        )
