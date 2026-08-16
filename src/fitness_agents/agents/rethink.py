from __future__ import annotations

import hashlib
import json
from typing import Any

from fitness_agents.agents.remote_llm import (
    complete_json,
    create_openai_client,
    resolve_base_url,
    resolve_model,
)
from fitness_agents.contracts.schemas import ReThinkReflection

RETHINK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reflections"],
    "properties": {
        "reflections": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "variant_id",
                    "verdict",
                    "summary",
                    "positive_findings",
                    "negative_findings",
                    "revised_reason",
                    "next_round_advice",
                ],
                "properties": {
                    "variant_id": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["support", "conflict", "mixed", "inconclusive"],
                    },
                    "summary": {"type": "string"},
                    "positive_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "negative_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "revised_reason": {"type": "string"},
                    "next_round_advice": {"type": "string"},
                },
            },
        }
    },
}


def _reflection_id(run_id: str, round_id: int, variant_id: str) -> str:
    digest = hashlib.sha256(f"{run_id}|{round_id}|{variant_id}".encode()).hexdigest()[:16]
    return f"rethink:{digest}"


def _parse_reflections(
    payload: dict[str, Any], *, run_id: str, round_id: int, provider: str
) -> tuple[ReThinkReflection, ...]:
    return tuple(
        ReThinkReflection(
            reflection_id=_reflection_id(run_id, round_id, str(item["variant_id"])),
            variant_id=str(item["variant_id"]),
            round_id=round_id,
            verdict=str(item["verdict"]),
            summary=str(item["summary"]),
            positive_findings=tuple(str(value) for value in item["positive_findings"]),
            negative_findings=tuple(str(value) for value in item["negative_findings"]),
            revised_reason=str(item["revised_reason"]),
            next_round_advice=str(item["next_round_advice"]),
            provider=provider,
        )
        for item in payload.get("reflections", ())
    )


class MockReThinkClient:
    """Deterministic offline reflection with the same structured output as the remote role."""

    provider_name = "mock_rethink"

    def reflect_round(self, *, context: dict[str, Any]) -> tuple[ReThinkReflection, ...]:
        baseline = float(context.get("visible_baseline", 0.0))
        items: list[dict[str, Any]] = []
        for item in context.get("candidates", ()):
            wet = float(item["wet_value"])
            dry = [float(entry["value"]) for entry in item.get("dry_validations", ())]
            wet_support = wet > baseline
            dry_mean = sum(dry) / len(dry) if dry else None
            dry_agrees = dry_mean is None or (dry_mean > baseline) == wet_support
            if wet_support and dry_agrees:
                verdict = "support"
            elif not wet_support and dry_agrees:
                verdict = "conflict"
            else:
                verdict = "mixed"
            positives = []
            negatives = []
            if wet_support:
                positives.append("Wet validation exceeded the pre-round visible baseline.")
            else:
                negatives.append("Wet validation did not exceed the pre-round visible baseline.")
            if dry_mean is not None and dry_agrees:
                positives.append("Dry validation agreed with the wet direction.")
            elif dry_mean is not None:
                negatives.append("Dry and wet validation directions disagreed.")
            items.append(
                {
                    "variant_id": item["variant_id"],
                    "verdict": verdict,
                    "summary": (
                        f"Recommendation reason is {verdict} by wet/dry directional checks; "
                        f"wet={wet:.4f}, baseline={baseline:.4f}."
                    ),
                    "positive_findings": positives,
                    "negative_findings": negatives,
                    "revised_reason": (
                        str(item.get("agent_reason", ""))
                        + " Treat this as round-specific evidence, not a universal residue effect."
                    ),
                    "next_round_advice": (
                        "Retain related mutations with uncertainty-aware exploration."
                        if wet_support
                        else "Down-weight this rationale and test matched alternatives."
                    ),
                }
            )
        return _parse_reflections(
            {"reflections": items},
            run_id=str(context["run_id"]),
            round_id=int(context["round_id"]),
            provider=self.provider_name,
        )


class OpenAICompatibleReThinkClient:
    provider_name = "openai_compatible_rethink"

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

    def reflect_round(self, *, context: dict[str, Any]) -> tuple[ReThinkReflection, ...]:
        payload = complete_json(
            client=self.client,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the ReThink Agent in an iterative protein-design campaign. "
                        "For every candidate, assess whether the original recommendation reason "
                        "is supported, contradicted, mixed, or inconclusive using supplied wet and "
                        "dry validation. Wet measurements are authoritative; dry model values are "
                        "lower-fidelity evidence. Do not invent measurements. Return JSON matching "
                        + json.dumps(RETHINK_SCHEMA, ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
            ],
            schema=RETHINK_SCHEMA,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
        )
        return _parse_reflections(
            payload,
            run_id=str(context["run_id"]),
            round_id=int(context["round_id"]),
            provider=self.provider_name,
        )


def create_rethink_client(provider: str, **kwargs: Any):
    if provider == "mock":
        return MockReThinkClient()
    if provider in {"openai", "openai_compatible", "deepseek"}:
        if provider == "deepseek":
            kwargs.setdefault("provider", "deepseek")
            kwargs.setdefault(
                "base_url", resolve_base_url(kwargs.get("base_url"), provider="deepseek")
            )
            kwargs.setdefault("model", resolve_model(kwargs.get("model"), provider="deepseek"))
        return OpenAICompatibleReThinkClient(**kwargs)
    raise ValueError(f"Unknown ReThink provider {provider!r}")
