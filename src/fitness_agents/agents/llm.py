from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

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

        ranked_evidence = sorted(
            evidence, key=lambda item: (item.confidence * abs(item.score), item.evidence_id), reverse=True
        )
        evidence_ids = tuple(item.evidence_id for item in ranked_evidence[:8])
        round_id = int(sanitized_context["round_id"])
        parent = sanitized_context.get("previous_hypothesis_id")
        residue_text = ", ".join(
            f"{position}:{'/'.join(residues)}" for position, residues in preferred.items()
        )
        return Hypothesis(
            hypothesis_id=f"hyp:{sanitized_context['run_id']}:r{round_id}",
            statement=(
                f"Visible elite observations support testing residue preferences {residue_text}; "
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


class OpenAICompatibleLLMClient:
    """Optional structured-output adapter. API keys are read only from the environment."""

    provider_name = "openai_compatible"

    def __init__(self, *, model: str | None = None, base_url: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install requirements/llm.txt to use a remote LLM") from error
        api_key = os.environ.get("FITNESS_AGENTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set FITNESS_AGENTS_LLM_API_KEY or OPENAI_API_KEY")
        self.model = model or os.environ.get("FITNESS_AGENTS_LLM_MODEL", "gpt-5-mini")
        self.client = OpenAI(api_key=api_key, base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
    ) -> Hypothesis:
        evidence_payload = [entry.__dict__ for entry in evidence[:80]]
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a protein-engineering hypothesis agent. Use only supplied visible "
                        "measurements and cited evidence. Do not invent fitness values. Return a "
                        "falsifiable hypothesis as JSON."
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
            text={
                "format": {
                    "type": "json_schema",
                    "name": "protein_hypothesis",
                    "strict": True,
                    "schema": output_schema,
                }
            },
        )
        payload = json.loads(response.output_text)
        return Hypothesis(
            hypothesis_id=str(payload["hypothesis_id"]),
            statement=str(payload["statement"]),
            preferred_residues={
                int(position): tuple(residues)
                for position, residues in payload["preferred_residues"].items()
            },
            evidence_ids=tuple(payload["evidence_ids"]),
            expected_outcome=str(payload["expected_outcome"]),
            falsification_criterion=str(payload["falsification_criterion"]),
            parent_hypothesis_id=payload.get("parent_hypothesis_id"),
        )


def create_llm_client(provider: str):
    if provider == "mock":
        return MockScientistLLMClient()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleLLMClient()
    raise ValueError(f"Unknown LLM provider {provider!r}")

