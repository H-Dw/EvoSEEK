from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from fitness_agents.agents.remote_llm import (
    complete_json,
    create_openai_client,
    resolve_base_url,
    resolve_model,
)
from fitness_agents.contracts.schemas import Evidence, Hypothesis

HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "hypothesis_id",
        "statement",
        "preferred_residues",
        "evidence_ids",
        "expected_outcome",
        "falsification_criterion",
    ],
    "properties": {
        "hypothesis_id": {"type": "string"},
        "statement": {"type": "string"},
        "preferred_residues": {"type": "object"},
        "evidence_ids": {"type": "array", "items": {"type": "string"}},
        "expected_outcome": {"type": "string"},
        "falsification_criterion": {"type": "string"},
        "parent_hypothesis_id": {"type": ["string", "null"]},
    },
}


class MockScientistLLMClient:
    """Deterministic offline scientist used for reproducible tests.

    It produces falsifiable, evidence-linked hypotheses from *visible observations only*. This is a
    harness mock, not a claim that a rule engine is a language model.
    """

    provider_name = "mock"

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
    ) -> Hypothesis:
        observations = list(sanitized_context.get("visible_observations", []))
        if not observations:
            preferred = {39: ("V",), 40: ("D",), 41: ("G",), 54: ("V",)}
        else:
            ranked = sorted(observations, key=lambda item: item["measured_fitness"], reverse=True)
            elite = ranked[: max(4, len(ranked) // 3)]
            preferred = {}
            for index, position in enumerate((39, 40, 41, 54)):
                residue_values: dict[str, list[float]] = defaultdict(list)
                for item in elite:
                    residue_values[item["variant"][index]].append(item["measured_fitness"])
                order = sorted(
                    residue_values,
                    key=lambda residue: (
                        sum(residue_values[residue]) / len(residue_values[residue]),
                        len(residue_values[residue]),
                        residue,
                    ),
                    reverse=True,
                )
                preferred[position] = tuple(order[:2])

        graph_context = sanitized_context.get("knowledge_graph", {})
        graph_preferences: dict[int, list[str]] = defaultdict(list)
        for item in graph_context.get("beneficial_site_residues", []):
            position = int(item["position"])
            residue = str(item["residue"])
            if residue not in graph_preferences[position]:
                graph_preferences[position].append(residue)
        interaction_context = sanitized_context.get("kg_interaction", {})
        for pack in interaction_context.get("packs", []):
            for item in pack.get("facts", []):
                if item.get("fact_type") != "residue_aggregate":
                    continue
                position = int(item["position"])
                residue = str(item["residue"])
                if residue not in graph_preferences[position]:
                    graph_preferences[position].append(residue)
        for position in (39, 40, 41, 54):
            merged = graph_preferences[position] + list(preferred.get(position, ()))
            preferred[position] = tuple(dict.fromkeys(merged))[:2]

        ranked_evidence = sorted(
            evidence, key=lambda item: (item.confidence * abs(item.score), item.evidence_id), reverse=True
        )
        evidence_ids = tuple(item.evidence_id for item in ranked_evidence[:8])
        round_id = int(sanitized_context["round_id"])
        parent = sanitized_context.get("previous_hypothesis_id")
        residue_text = ", ".join(
            f"{position}:{'/'.join(residues)}" for position, residues in preferred.items()
        )
        evidence_source = (
            "Visible observations and the audited multi-step KG interaction"
            if interaction_context
            else (
                "Visible observations and the audited knowledge-graph query"
                if graph_context
                else "Visible elite observations"
            )
        )
        return Hypothesis(
            hypothesis_id=f"hyp:{sanitized_context['run_id']}:r{round_id}",
            statement=(
                f"{evidence_source} support testing residue preferences {residue_text}; "
                "retain batch diversity to probe epistasis."
            ),
            preferred_residues=preferred,
            evidence_ids=evidence_ids,
            expected_outcome="The proposed batch should enrich high-fitness variants relative to random selection.",
            falsification_criterion=(
                "Reject or revise if the revealed batch median fails to exceed the pre-round observed "
                "median, or if preferred-residue variants underperform matched alternatives."
            ),
            parent_hypothesis_id=parent,
        )


def _preferred_residues(raw: Any) -> dict[int, tuple[str, ...]]:
    if not isinstance(raw, dict):
        raise TypeError("preferred_residues must be a JSON object keyed by site number")
    parsed: dict[int, tuple[str, ...]] = {}
    for position, residues in raw.items():
        values = residues if isinstance(residues, (list, tuple)) else [residues]
        parsed[int(position)] = tuple(str(item) for item in values)
    return parsed


class OpenAICompatibleLLMClient:
    """OpenAI-compatible Chat Completions adapter. API keys are read only from the environment."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        thinking: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.model = resolve_model(model, provider=provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.client = create_openai_client(api_key=api_key, base_url=base_url, provider=provider)

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
    ) -> Hypothesis:
        evidence_payload = [entry.__dict__ for entry in evidence[:80]]
        payload = complete_json(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a protein-engineering hypothesis agent. Use only supplied visible "
                        "measurements and cited evidence. Do not invent fitness values. Reply with "
                        "a single JSON object that matches this schema: "
                        + json.dumps(output_schema, ensure_ascii=False)
                        + " preferred_residues keys must be the integer site numbers 39, 40, 41, "
                        "and 54. Do not include markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"context": sanitized_context, "evidence": evidence_payload},
                        ensure_ascii=False,
                    ),
                },
            ],
            schema=output_schema,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
        )
        return Hypothesis(
            hypothesis_id=str(payload["hypothesis_id"]),
            statement=str(payload["statement"]),
            preferred_residues=_preferred_residues(payload["preferred_residues"]),
            evidence_ids=tuple(str(item) for item in payload["evidence_ids"]),
            expected_outcome=str(payload["expected_outcome"]),
            falsification_criterion=str(payload["falsification_criterion"]),
            parent_hypothesis_id=payload.get("parent_hypothesis_id"),
        )


def create_llm_client(provider: str, **kwargs: Any):
    if provider == "mock":
        return MockScientistLLMClient()
    if provider in {"openai", "openai_compatible", "deepseek"}:
        if provider == "deepseek":
            kwargs.setdefault("provider", "deepseek")
            kwargs.setdefault("base_url", resolve_base_url(kwargs.get("base_url"), provider="deepseek"))
            kwargs.setdefault("model", resolve_model(kwargs.get("model"), provider="deepseek"))
        return OpenAICompatibleLLMClient(**kwargs)
    raise ValueError(f"Unknown LLM provider {provider!r}")
