from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
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
            facts=tuple(facts),
            predictions=predictions,
            evidence=evidence,
            supporting_paths=_dict_tuple(result.get("supporting_paths")),
            counterevidence=_dict_tuple(result.get("counterevidence")),
            directional_signals=_dict_tuple(result.get("directional_signals")),
            caveats=tuple(str(item) for item in result.get("caveats", ())),
            provenance=_collect_provenance(evidence),
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
            facts=_dict_tuple(result.get("visible_observations")),
            predictions=_dict_tuple(result.get("predictions")),
            evidence=evidence,
            supporting_paths=_dict_tuple(result.get("supporting_paths")),
            counterevidence=counterevidence,
            caveats=tuple(str(item) for item in result.get("caveats", ())),
            provenance=_collect_provenance(evidence),
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
        return EvidencePack(
            query_id=_stable_query_id(self.name, query_ids),
            operator=self.name,
            as_of_round=context.round_id,
            facts=tuple(facts),
            predictions=tuple(predictions),
            evidence=tuple(evidence),
            counterevidence=tuple(counterevidence),
            provenance=_collect_provenance(evidence),
            metadata={"variant_ids": list(variant_ids), "child_query_ids": query_ids},
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
