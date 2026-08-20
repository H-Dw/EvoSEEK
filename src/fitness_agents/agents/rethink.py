from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from fitness_agents.agents.adaptive_batch import AdaptiveBatchWork, adaptive_batch_submit
from fitness_agents.agents.remote_llm import (
    RemoteLLMCompletionError,
    create_openai_client,
    resolve_base_url,
    resolve_model,
)
from fitness_agents.contracts.agent_io import AgentTraceContext, ReThinkContextInput
from fitness_agents.contracts.schemas import ReThinkReflection
from fitness_agents.utils.progress import report_event

from .output_contracts import ReThinkOutput, validate_rethink_payload
from .profile_loader import load_role_profile
from .short_ids import ShortIdMap, rewrite_exact_ids
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
        max_transport_retries: int = 2,
        max_truncation_retries: int = 1,
        max_syntax_retries: int = 1,
        max_schema_retries: int = 2,
        max_semantic_retries: int = 1,
        max_unknown_evidence_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
        request_timeout_seconds: float = 120.0,
        allow_unknown_evidence_stripping: bool = False,
        max_input_chars: int | None = None,
        reasoning_batch_size: int = 8,
        max_parallel_batches: int = 4,
    ) -> None:
        self.model = resolve_model(model, provider=provider)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_transport_retries = max_transport_retries
        self.max_truncation_retries = max_truncation_retries
        self.max_syntax_retries = max_syntax_retries
        self.max_schema_retries = max_schema_retries
        self.max_semantic_retries = max_semantic_retries
        self.max_unknown_evidence_retries = max_unknown_evidence_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.allow_unknown_evidence_stripping = allow_unknown_evidence_stripping
        self.max_input_chars = max_input_chars
        if not 1 <= reasoning_batch_size <= 8:
            raise ValueError("ReThink reasoning_batch_size must be between 1 and 8")
        if max_parallel_batches < 1:
            raise ValueError("ReThink max_parallel_batches must be positive")
        self.reasoning_batch_size = reasoning_batch_size
        self.max_parallel_batches = max_parallel_batches
        role_profile = load_role_profile("rethink", profile)
        self.profile_name = profile
        self.profile = role_profile.instructions
        self.client = create_openai_client(
            api_key=api_key,
            base_url=base_url,
            provider=provider,
            request_timeout_seconds=request_timeout_seconds,
        )
        self.transport = OpenAICompatibleChatTransport(self.client)

    @staticmethod
    def _is_truncation(error: Exception) -> bool:
        return isinstance(error, RemoteLLMCompletionError) and (
            error.error_code == "OUTPUT_TRUNCATED"
        )

    def _reflect_batch(
        self,
        *,
        context: ReThinkContextInput,
        batch_id: str,
        split_depth: int,
    ) -> tuple[ReThinkReflection, ...]:
        id_map = ShortIdMap.build(
            tuple(str(item["variant_id"]) for item in context.candidates), prefix="S"
        )
        alias_context = context.model_copy(
            update={
                "candidates": rewrite_exact_ids(context.candidates, id_map),
            }
        )
        expected_aliases = alias_context.expected_variant_ids
        trace_context = AgentTraceContext(
            run_id=context.run_id,
            round_id=context.round_id,
            role="rethink",
            request_id=(
                f"rethink:{context.run_id}:r{context.round_id}:{batch_id}"
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
                        "Use only the request-local S labels in candidates; local code maps them "
                        "back to canonical sample records. "
                        + json.dumps(ReThinkOutput.model_json_schema(), ensure_ascii=False)
                    ),
                },
                {"role": "user", "content": alias_context.model_dump_json()},
            ],
            output_type=ReThinkOutput,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            retries=0,
            transport_retries=getattr(self, "max_transport_retries", 2),
            truncation_retries=getattr(self, "max_truncation_retries", 1),
            syntax_retries=getattr(self, "max_syntax_retries", 1),
            schema_retries=getattr(self, "max_schema_retries", 2),
            semantic_retries=getattr(self, "max_semantic_retries", 1),
            unknown_evidence_retries=getattr(
                self, "max_unknown_evidence_retries", 1
            ),
            retry_backoff_seconds=getattr(self, "retry_backoff_seconds", 0.0),
            allow_unknown_evidence_stripping=getattr(
                self, "allow_unknown_evidence_stripping", False
            ),
            max_input_chars=getattr(self, "max_input_chars", None),
            separate_json_render=True,
            repair_hints={
                "reflections[].variant_id": tuple(expected_aliases)
            },
            contextual_validator=lambda value: validate_rethink_payload(
                value, expected_variant_ids=expected_aliases
            ),
            reasoning_truncation_retries=0,
            preserve_reasoning_on_retry=True,
            trace_context={
                **trace_context.model_dump(mode="json"),
                "profile": self.profile_name,
                "schema_name": "ReThinkOutput",
                "retry_scope": f"rethink:{batch_id}",
                "rethink_batch_id": batch_id,
                "rethink_batch_size": len(context.candidates),
                "rethink_split_depth": split_depth,
            },
        )
        reflections = _parse_reflections(
            output.model_dump(mode="json"),
            run_id=context.run_id,
            round_id=context.round_id,
            provider=self.provider_name,
        )
        return tuple(
            replace(item, variant_id=id_map.decode(item.variant_id))
            for item in reflections
        )

    def reflect_round(self, *, context: ReThinkContextInput) -> tuple[ReThinkReflection, ...]:
        validated_context = ReThinkContextInput.model_validate(context)
        candidates = tuple(validated_context.candidates)
        if not candidates:
            return ()
        by_variant_id: dict[str, ReThinkReflection] = {}
        batches = adaptive_batch_submit(
            candidates,
            item_id=lambda item: str(item["variant_id"]),
            submit_batch=lambda work: self._reflect_batch_work(
                context=validated_context,
                work=work,
            ),
            initial_batch_size=getattr(self, "reasoning_batch_size", 8),
            max_parallel_batches=getattr(self, "max_parallel_batches", 4),
            should_split_failure=self._is_truncation,
            role="rethink",
            round_id=validated_context.round_id,
            event_reporter=report_event,
        )
        for batch in batches:
            for reflection in batch.output:
                if reflection.variant_id in by_variant_id:
                    raise ValueError(
                        "Adaptive ReThink batches returned duplicate variant_id "
                        f"{reflection.variant_id!r}"
                    )
                by_variant_id[reflection.variant_id] = reflection
        expected_ids = tuple(str(item["variant_id"]) for item in candidates)
        missing = sorted(set(expected_ids).difference(by_variant_id))
        unexpected = sorted(set(by_variant_id).difference(expected_ids))
        if missing or unexpected:
            raise ValueError(
                "Adaptive ReThink batch coverage mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )
        return tuple(
            replace(by_variant_id[item], reflection_id=f"R{validated_context.round_id:02d}-{index:02d}")
            for index, item in enumerate(expected_ids, start=1)
        )

    def _reflect_batch_work(
        self,
        *,
        context: ReThinkContextInput,
        work: AdaptiveBatchWork[dict[str, Any]],
    ) -> tuple[ReThinkReflection, ...]:
        batch_context = context.model_copy(update={"candidates": list(work.items)})
        return self._reflect_batch(
            context=batch_context,
            batch_id=work.batch_id,
            split_depth=work.split_depth,
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
