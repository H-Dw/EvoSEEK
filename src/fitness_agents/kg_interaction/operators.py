from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, Protocol

from .contracts import EvidencePack, KGQueryContext, KGQueryStep


class QueryOperator(Protocol):
    name: str

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack: ...


def _dict_tuple(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(dict(item) for item in value if isinstance(item, Mapping))


def _stable_query_id(operator: str, query_ids: Sequence[str]) -> str:
    payload = json.dumps([operator, *query_ids], sort_keys=True).encode("utf-8")
    return f"kgq:{hashlib.sha256(payload).hexdigest()[:16]}"


def _collect_provenance(evidence: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: dict[str, dict[str, Any]] = {}
    for item in evidence:
        source_id = item.get("source_id")
        if source_id:
            unique[str(source_id)] = {
                "source_id": str(source_id),
                "evidence_type": item.get("evidence_type", "unknown"),
            }
    return tuple(unique[key] for key in sorted(unique))


class HypothesisContextOperator:
    name = "hypothesis_context"

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        requested = int(step.arguments.get("limit", context.max_rows))
        result = self.tool.hypothesis_context(
            round_id=context.round_id,
            limit=min(requested, context.max_rows),
        )
        facts = []
        for item in _dict_tuple(result.get("beneficial_site_residues")):
            facts.append({"fact_type": "residue_aggregate", **item})
        for item in _dict_tuple(result.get("top_visible_observations")):
            facts.append({"fact_type": "measurement", **item})
        for item in _dict_tuple(result.get("top_knowledge_evidence")):
            facts.append({"fact_type": "computed_evidence", **item})
        for item in _dict_tuple(result.get("validation_prior")):
            facts.append({"fact_type": "validation_prior", **item})
        predictions = _dict_tuple(result.get("current_candidate_predictions"))
        evidence = tuple(
            dict(item)
            for prediction in predictions
            for item in prediction.get("evidence", ())
            if isinstance(item, Mapping)
        )
        return EvidencePack(
            query_id=str(result.get("query_id", _stable_query_id(self.name, ()))),
            operator=self.name,
            as_of_round=context.round_id,
            facts=tuple(facts[: context.max_rows]),
            predictions=predictions[: context.max_rows],
            evidence=evidence[: context.max_rows],
            supporting_paths=_dict_tuple(result.get("supporting_paths"))[: context.max_rows],
            counterevidence=_dict_tuple(result.get("counterevidence"))[: context.max_rows],
            directional_signals=_dict_tuple(result.get("directional_signals"))[
                : context.max_rows
            ],
            caveats=tuple(str(item) for item in result.get("caveats", ()))[: context.max_rows],
            provenance=_collect_provenance(evidence)[: context.max_rows],
            metadata={
                "prior_hypotheses": list(result.get("prior_hypotheses", ())),
                "visibility_rule": result.get("visibility_rule"),
            },
        )


class ExplainVariantOperator:
    name = "explain_variant"

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        variant_id = str(step.arguments["variant_id"])
        result = self.tool.explain_variant(variant_id, round_id=context.round_id)
        evidence = _dict_tuple(result.get("evidence"))
        counterevidence = tuple(item for item in evidence if float(item.get("score", 0.0)) < 0.0)
        return EvidencePack(
            query_id=str(result.get("query_id", _stable_query_id(self.name, (variant_id,)))),
            operator=self.name,
            as_of_round=context.round_id,
            facts=(
                *_dict_tuple(result.get("visible_observations")),
                *(
                    {"fact_type": "validation_record", **item}
                    for item in _dict_tuple(result.get("validation_history"))
                ),
            )[: context.max_rows],
            predictions=_dict_tuple(result.get("predictions"))[: context.max_rows],
            evidence=evidence[: context.max_rows],
            supporting_paths=_dict_tuple(result.get("supporting_paths"))[: context.max_rows],
            counterevidence=counterevidence[: context.max_rows],
            caveats=tuple(str(item) for item in result.get("caveats", ()))[: context.max_rows],
            provenance=_collect_provenance(evidence)[: context.max_rows],
            metadata={
                "variant_id": variant_id,
                "found": bool(result.get("found", False)),
                "mutation_notation": result.get("mutation_notation"),
            },
        )


class CompareVariantsOperator:
    """Deterministically compare variants through the existing safe graph tool."""

    name = "compare_variants"

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        variant_ids = tuple(str(item) for item in step.arguments.get("variant_ids", ()))[
            : context.max_rows
        ]
        results = [
            self.tool.explain_variant(variant_id, round_id=context.round_id)
            for variant_id in variant_ids
        ]
        query_ids = [str(item.get("query_id", "")) for item in results]
        facts: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        counterevidence: list[dict[str, Any]] = []
        for result in results:
            variant_id = str(result.get("variant_id", ""))
            facts.append(
                {
                    "variant_id": variant_id,
                    "found": bool(result.get("found", False)),
                    "mutation_notation": result.get("mutation_notation"),
                    "visible_observations": list(result.get("visible_observations", ())),
                    "validation_history": list(result.get("validation_history", ())),
                }
            )
            predictions.extend(
                {"variant_id": variant_id, **item}
                for item in _dict_tuple(result.get("predictions"))
            )
            for item in _dict_tuple(result.get("evidence")):
                record = {"variant_id": variant_id, **item}
                evidence.append(record)
                if float(item.get("score", 0.0)) < 0.0:
                    counterevidence.append(record)
        query_id = _stable_query_id(self.name, query_ids)
        if hasattr(self.tool, "graph"):
            query_id = self.tool.graph.record_agent_query(
                self.name,
                round_id=context.round_id,
                parameters={"variant_ids": list(variant_ids)},
                result={
                    "child_query_ids": query_ids,
                    "fact_count": len(facts),
                    "prediction_count": len(predictions),
                    "evidence_count": len(evidence),
                },
            )
        return EvidencePack(
            query_id=query_id,
            operator=self.name,
            as_of_round=context.round_id,
            facts=tuple(facts[: context.max_rows]),
            predictions=tuple(predictions[: context.max_rows]),
            evidence=tuple(evidence[: context.max_rows]),
            counterevidence=tuple(counterevidence[: context.max_rows]),
            provenance=_collect_provenance(evidence)[: context.max_rows],
            metadata={"variant_ids": list(variant_ids), "child_query_ids": query_ids},
        )


class FeatureEvidenceOperator:
    """Expose one scientific feature channel without granting raw database access."""

    def __init__(self, name: str, channel: str, tool: Any) -> None:
        self.name = name
        self.channel = channel
        self.tool = tool

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        variant_id = str(step.arguments["variant_id"])
        result = self.tool.feature_evidence(
            variant_id,
            channel=self.channel,
            round_id=context.round_id,
        )
        evidence = _dict_tuple(result.get("evidence"))[: context.max_rows]
        return EvidencePack(
            query_id=str(result["query_id"]),
            operator=self.name,
            as_of_round=context.round_id,
            evidence=evidence,
            counterevidence=tuple(
                item for item in evidence if float(item.get("score", 0.0)) < 0.0
            ),
            caveats=tuple(
                warning
                for item in evidence
                for warning in item.get("warnings", ())
            )[: context.max_rows],
            provenance=tuple(
                dict(item.get("provenance", {})) for item in evidence
            )[: context.max_rows],
            metadata={"variant_id": variant_id, "channel": self.channel},
        )


class EvidenceProvenanceOperator:
    name = "query_evidence_provenance"

    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        evidence_id = str(step.arguments["evidence_id"])
        result = self.tool.evidence_provenance(evidence_id, round_id=context.round_id)
        facts = ({key: value for key, value in result.items() if key != "tool"},)
        return EvidencePack(
            query_id=str(result["query_id"]),
            operator=self.name,
            as_of_round=context.round_id,
            facts=facts,
            provenance=(dict(result.get("provenance", {})),) if result.get("found") else (),
            caveats=tuple(str(item) for item in result.get("warnings", ())),
            metadata={"evidence_id": evidence_id, "found": bool(result.get("found"))},
        )


class LocalKnowledgeQueryOperator:
    """Retrieve policy-filtered local documents through the project-owned source."""

    name = "query_local_knowledge"

    def __init__(self, knowledge_engine: Any) -> None:
        self.knowledge_engine = knowledge_engine

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        query = str(step.arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query_local_knowledge requires a non-empty query")
        anchors = tuple(str(item) for item in step.arguments.get("anchors", ()))
        raw_knowledge_types = step.arguments.get("knowledge_types", ())
        if isinstance(raw_knowledge_types, str):
            raw_knowledge_types = (raw_knowledge_types,)
        knowledge_types = tuple(str(item) for item in raw_knowledge_types)
        requested = int(step.arguments.get("limit", context.max_rows))
        result, evidence = self.knowledge_engine.retrieve_local_knowledge(
            query=query,
            intent=step.intent.value,
            round_id=context.round_id,
            anchors=anchors,
            top_k=min(requested, context.max_rows),
            knowledge_types=knowledge_types,
            stage=True,
        )
        evidence_payload = tuple(asdict(item) for item in evidence[: context.max_rows])
        facts = tuple(
            {
                "fact_type": "local_knowledge_claim",
                "claim_id": item.claim_id,
                "statement": item.statement,
                "polarity": item.polarity,
                "applicability": item.applicability,
                "confidence": item.confidence,
                "evidence_chunk_ids": item.evidence_chunk_ids,
            }
            for item in result.claims[: context.max_rows]
        )
        return EvidencePack(
            query_id=result.query_id,
            operator=self.name,
            as_of_round=context.round_id,
            facts=facts,
            evidence=evidence_payload,
            caveats=result.warnings[: context.max_rows],
            provenance=tuple(
                {
                    "evidence_id": item.evidence_id,
                    "source_id": item.source_id,
                    **item.provenance,
                }
                for item in evidence[: context.max_rows]
            ),
            metadata={
                "sanitized_query": result.sanitized_query,
                "policy_decision": result.policy_decision,
                "index_manifest_hash": result.index_manifest_hash,
                "knowledge_types": list(knowledge_types),
                "staged_for_round_commit": True,
            },
        )


class StructuredClaimQueryOperator:
    """Read round-visible claims from the existing structured KG projection."""

    name = "query_structured_claims"

    def __init__(self, knowledge_engine: Any) -> None:
        self.knowledge_engine = knowledge_engine

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        query = str(step.arguments.get("query", "")).strip()
        requested = int(step.arguments.get("limit", context.max_rows))
        claims = self.knowledge_engine.query_structured_claims(
            query=query,
            round_id=context.round_id,
            limit=min(requested, context.max_rows),
        )
        query_id = _stable_query_id(
            self.name, tuple(str(item["entity_id"]) for item in claims)
        )
        evidence_ids = tuple(
            str(evidence_id)
            for claim in claims
            for evidence_id in claim.get("evidence_ids", ())
        )
        return EvidencePack(
            query_id=query_id,
            operator=self.name,
            as_of_round=context.round_id,
            facts=claims,
            supporting_paths=tuple(
                {
                    "claim_id": item["entity_id"],
                    "relation_ids": item.get("supporting_relation_ids", ()),
                    "evidence_ids": item.get("evidence_ids", ()),
                }
                for item in claims
            ),
            provenance=tuple(
                {
                    "claim_id": item["entity_id"],
                    "source_ids": item.get("source_ids", ()),
                    "source_group": item.get("source_group"),
                    "valid_from_round": item.get("valid_from_round"),
                }
                for item in claims
            ),
            metadata={"query": query, "evidence_ids": evidence_ids},
        )


class CallableQueryOperator:
    """Adapter for project-specific retrievers without changing the controller."""

    def __init__(
        self,
        name: str,
        function: Callable[[KGQueryStep, KGQueryContext], EvidencePack],
    ) -> None:
        self.name = name
        self.function = function

    def execute(self, step: KGQueryStep, context: KGQueryContext) -> EvidencePack:
        pack = self.function(step, context)
        if pack.operator != self.name:
            raise ValueError(f"Operator {self.name!r} returned pack for {pack.operator!r}")
        return pack
