from __future__ import annotations

import hashlib
import json
from typing import Any

from fitness_agents.agents.remote_llm import (
    create_openai_client,
    resolve_base_url,
    resolve_model,
)
from fitness_agents.contracts.agent_io import AgentTraceContext, ReThinkContextInput
from fitness_agents.contracts.schemas import ReThinkReflection

from .output_contracts import ReThinkOutput, validate_rethink_payload
from .profile_loader import load_role_profile
from .structured_completion import complete_structured
from .transports import OpenAICompatibleChatTransport

RETHINK_SCHEMA: dict[str, Any] = ReThinkOutput.model_json_schema()


def _parse_reflections(
    payload: dict[str, Any], *, run_id: str, round_id: int, provider: str
) -> tuple[ReThinkReflection, ...]:
    return ReThinkOutput.model_validate(payload).to_reflections(
        run_id=run_id,
        round_id=round_id,
        provider=provider,
    )


class MockReThinkClient:
    """Deterministic offline reflection with the same structured output as the remote role."""

    provider_name = "mock_rethink"

    def reflect_round(self, *, context: ReThinkContextInput) -> tuple[ReThinkReflection, ...]:
        context = ReThinkContextInput.model_validate(context).model_dump(mode="json")
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


class NativeReThinkClient:
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
        profile: str = "scientific_v1",
    ) -> None:
        self.model = resolve_model(model, provider=provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        role_profile = load_role_profile("rethink", profile)
        self.profile_name = profile
        self.profile = role_profile.instructions
        self.profile_sha256 = role_profile.sha256
        self.client = create_openai_client(api_key=api_key, base_url=base_url, provider=provider)
        self.transport = OpenAICompatibleChatTransport(self.client)

    def reflect_round(self, *, context: ReThinkContextInput) -> tuple[ReThinkReflection, ...]:
        validated_context = ReThinkContextInput.model_validate(context)
        trace_context = AgentTraceContext(
            run_id=validated_context.run_id,
            round_id=validated_context.round_id,
            role="rethink",
            request_id=(
                f"rethink:{validated_context.run_id}:r{validated_context.round_id}"
            ),
        )
        output = complete_structured(
            client=self.client,
            transport=self.transport,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        self.profile + "\nReturn JSON matching "
                        + json.dumps(RETHINK_SCHEMA, ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": validated_context.model_dump_json()},
            ],
            output_type=ReThinkOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            contextual_validator=lambda value: validate_rethink_payload(
                value, expected_variant_ids=validated_context.expected_variant_ids
            ),
            trace_context={
                **trace_context.model_dump(mode="json"),
                "profile": self.profile_name,
                "profile_sha256": self.profile_sha256,
                "schema_name": "ReThinkOutput",
                "context_sha256": hashlib.sha256(
                    validated_context.model_dump_json().encode()
                ).hexdigest(),
            },
        )
        return _parse_reflections(
            output.model_dump(mode="json"),
            run_id=validated_context.run_id,
            round_id=validated_context.round_id,
            provider=self.provider_name,
        )


OpenAICompatibleReThinkClient = NativeReThinkClient


def create_rethink_client(provider: str, **kwargs: Any):
    if "runtime" in kwargs:
        runtime = str(kwargs.pop("runtime"))
        if runtime != "chat_completions":
            raise ValueError(f"Removed Agents SDK runtime is not supported: {runtime!r}")
    if provider == "mock":
        return MockReThinkClient()
    if provider in {"openai", "openai_compatible", "deepseek"}:
        if provider == "deepseek":
            kwargs.setdefault("provider", "deepseek")
            kwargs.setdefault(
                "base_url", resolve_base_url(kwargs.get("base_url"), provider="deepseek")
            )
            kwargs.setdefault("model", resolve_model(kwargs.get("model"), provider="deepseek"))
        return NativeReThinkClient(**kwargs)
    raise ValueError(f"Unknown ReThink provider {provider!r}")
