"""Fail-closed validation and execution projection for Researcher plans."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fitness_agents.config import ResearcherConfig
from fitness_agents.contracts.researcher import (
    FEATURE_FOCUS_BY_CHANNEL,
    ExternalRetrievalPlan,
    FeatureEvidencePlan,
)

from .contracts import KGQueryStep, QueryIntent

_VIRAL_TERMS = re.compile(
    r"\b(?:virus|viral|virion|phage|bacteriophage|retrovirus|coronavirus)\b",
    re.IGNORECASE,
)
_MUTATION_IDENTITY = re.compile(r"\b[A-Z]\d{1,6}[A-Z]\b")
_CHANNEL_OPERATOR = {
    "physchem": "query_physchem_delta",
    "conservation": "query_evolutionary_profile",
    "structure": "query_structure_environment",
}


@dataclass(frozen=True)
class ValidatedRetrievalNeed:
    need_id: str
    intent: str
    query: str
    facets: dict[str, tuple[str, ...]]
    top_k: int
    rationale: str


class ResearcherPlanningController:
    """Validate model plans against request-local authority and hard budgets."""

    def __init__(
        self,
        config: ResearcherConfig,
        *,
        mutable_positions: Sequence[int],
        facet_catalog: Mapping[str, Sequence[str]],
        enabled_feature_channels: Sequence[str] = (
            "physchem",
            "conservation",
            "structure",
        ),
        forbidden_query_terms: Sequence[str] = (),
    ) -> None:
        self.config = config
        self.mutable_positions = frozenset(int(item) for item in mutable_positions)
        self.facet_catalog = {
            str(key): frozenset(str(item) for item in values)
            for key, values in facet_catalog.items()
        }
        self.enabled_feature_channels = frozenset(enabled_feature_channels)
        self.forbidden_query_terms = tuple(
            str(item).strip().casefold()
            for item in forbidden_query_terms
            if str(item).strip()
        )

    def validate_external_plan(
        self, plan: ExternalRetrievalPlan
    ) -> tuple[ValidatedRetrievalNeed, ...]:
        plan = ExternalRetrievalPlan.model_validate(plan)
        if plan.decision == "ABSTAIN":
            return ()
        if len(plan.needs) > self.config.max_rag_queries:
            raise ValueError("Researcher retrieval plan exceeds the per-round query budget")
        output: list[ValidatedRetrievalNeed] = []
        for need in plan.needs:
            query = need.scientific_question.strip()
            lowered = query.casefold()
            if _VIRAL_TERMS.search(query):
                raise ValueError("Researcher query requests prohibited viral-protein material")
            if _MUTATION_IDENTITY.search(query):
                raise ValueError("Researcher query contains a mutation identity")
            if any(term in lowered for term in self.forbidden_query_terms):
                raise ValueError("Researcher query contains a protected benchmark/task identity")
            if need.top_k > self.config.rag_top_k_per_query:
                raise ValueError("Researcher query exceeds top_k budget")
            facets: dict[str, tuple[str, ...]] = {}
            for name, values in need.facets.items():
                if name not in self.facet_catalog:
                    raise ValueError(f"Researcher requested unknown facet {name!r}")
                normalized = tuple(dict.fromkeys(str(item) for item in values))
                outside = set(normalized).difference(self.facet_catalog[name])
                if outside:
                    raise ValueError(
                        f"Researcher requested out-of-catalog {name} values: {sorted(outside)}"
                    )
                if normalized:
                    facets[name] = normalized
            output.append(
                ValidatedRetrievalNeed(
                    need_id=need.need_id,
                    intent=need.intent,
                    query=query,
                    facets=facets,
                    top_k=need.top_k,
                    rationale=need.rationale,
                )
            )
        return tuple(output)

    def validate_feature_plan(
        self,
        plan: FeatureEvidencePlan,
        *,
        sample_id_to_variant_id: Mapping[str, str],
    ) -> tuple[KGQueryStep, ...]:
        plan = FeatureEvidencePlan.model_validate(plan)
        if plan.decision == "ABSTAIN":
            return ()
        if len(plan.needs) > self.config.max_feature_requests:
            raise ValueError("Researcher feature plan exceeds the request budget")
        requested_samples = {item.sample_id for item in plan.needs}
        if len(requested_samples) > self.config.max_feature_variants:
            raise ValueError("Researcher feature plan exceeds the visible-sample budget")
        steps: list[KGQueryStep] = []
        for need in plan.needs:
            if need.sample_id not in sample_id_to_variant_id:
                raise ValueError(f"Researcher requested unknown sample {need.sample_id!r}")
            if need.channel not in self.enabled_feature_channels:
                raise ValueError(f"Researcher requested disabled channel {need.channel!r}")
            outside = set(need.positions).difference(self.mutable_positions)
            if outside:
                raise ValueError(
                    f"Researcher requested positions outside task scope: {sorted(outside)}"
                )
            if not set(need.focus).issubset(FEATURE_FOCUS_BY_CHANNEL[need.channel]):
                raise ValueError("Researcher requested an illegal feature focus")
            variant_id = sample_id_to_variant_id[need.sample_id]
            steps.append(
                KGQueryStep(
                    step_id=f"researcher_{need.request_id}",
                    operator=_CHANNEL_OPERATOR[need.channel],
                    intent=QueryIntent.EXPLAIN,
                    arguments={
                        "variant_id": variant_id,
                        "positions": list(need.positions),
                        "projection": list(need.focus),
                    },
                    depends_on=("context",),
                    rationale=need.rationale,
                )
            )
        return tuple(steps)


def stable_payload_hash(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
