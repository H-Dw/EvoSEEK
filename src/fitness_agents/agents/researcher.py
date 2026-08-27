"""Provider-neutral two-stage Researcher planner.

The Researcher emits typed plans.  Tool execution remains exclusively in the
local controller/runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any

from fitness_agents.contracts.researcher import (
    ExternalRetrievalPlan,
    FeatureEvidencePlan,
    ResearcherContextInput,
)

from .profile_loader import load_role_profile
from .remote_llm import create_openai_client, resolve_model
from .structured_completion import complete_structured
from .transports import OpenAICompatibleChatTransport

_IDENTITY_NEUTRAL_REPAIR_HINT = (
    "Rewrite every scientific_question using only identity-neutral phrases such as "
    "'the current protein' and 'the current assay'. Do not copy or infer any protein, "
    "assay, dataset, accession, benchmark, task, or run identity from the context."
)


def _schema_hash(output_type: type[Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            output_type.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class NativeResearcherClient:
    """Strict OpenAI-compatible planner with no direct tool interface."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        profile: str = "evidence_planner_v1",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        reasoning_effort: str | None = None,
        thinking: str | None = "disabled",
        max_input_chars: int = 30000,
        request_timeout_seconds: float = 120.0,
        max_transport_retries: int = 2,
        max_truncation_retries: int = 1,
        max_syntax_retries: int = 1,
        max_schema_retries: int = 2,
        max_semantic_retries: int = 1,
        retry_backoff_seconds: float = 1.0,
        transport: Any | None = None,
    ) -> None:
        if temperature != 0.0:
            raise ValueError("Researcher temperature must be zero")
        self.model = resolve_model(model, provider=provider)
        self.profile_name = profile
        role_profile = load_role_profile("researcher", profile)
        self.profile = role_profile.instructions
        self.profile_version = str(role_profile.metadata.get("version", "unknown"))
        self.profile_hash = hashlib.sha256(self.profile.encode("utf-8")).hexdigest()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.max_input_chars = max_input_chars
        self.max_transport_retries = max_transport_retries
        self.max_truncation_retries = max_truncation_retries
        self.max_syntax_retries = max_syntax_retries
        self.max_schema_retries = max_schema_retries
        self.max_semantic_retries = max_semantic_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        if transport is None:
            client = create_openai_client(
                api_key=api_key,
                base_url=base_url,
                provider=provider,
                request_timeout_seconds=request_timeout_seconds,
            )
            self.client = client
            self.transport = OpenAICompatibleChatTransport(client)
        else:
            self.client = None
            self.transport = transport
        self._external_plan_validator: Callable[[ExternalRetrievalPlan], Any] | None = None

    def bind_external_plan_validator(
        self,
        validator: Callable[[ExternalRetrievalPlan], Any],
    ) -> None:
        """Bind the runtime controller inside the bounded semantic-retry boundary."""

        self._external_plan_validator = validator

    @staticmethod
    def schema_hash(output_type: type[Any]) -> str:
        return _schema_hash(output_type)

    def _complete(
        self,
        context: ResearcherContextInput,
        output_type: type[Any],
    ) -> Any:
        phase_instruction = (
            "Return an external evidence-gap retrieval plan."
            if context.phase == "external_retrieval"
            else "Return a feature evidence projection plan."
        )
        contextual_validator = None
        repair_hints = None
        if context.phase == "external_retrieval" and self._external_plan_validator is not None:

            def validate_external(value: dict[str, Any]) -> dict[str, Any]:
                plan = ExternalRetrievalPlan.model_validate(value)
                self._external_plan_validator(plan)
                return value

            contextual_validator = validate_external
            repair_hints = {
                "runtime_invariant": (_IDENTITY_NEUTRAL_REPAIR_HINT,)
            }
        return complete_structured(
            client=self.client,
            transport=self.transport,
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        self.profile
                        + "\n\n"
                        + phase_instruction
                        + " Return exactly one JSON object matching this schema: "
                        + json.dumps(
                            output_type.model_json_schema(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        context.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            output_type=output_type,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort=self.reasoning_effort,
            thinking=self.thinking,
            retries=0,
            transport_retries=self.max_transport_retries,
            truncation_retries=self.max_truncation_retries,
            syntax_retries=self.max_syntax_retries,
            schema_retries=self.max_schema_retries,
            semantic_retries=self.max_semantic_retries,
            unknown_evidence_retries=0,
            retry_backoff_seconds=self.retry_backoff_seconds,
            max_input_chars=self.max_input_chars,
            separate_json_render=False,
            contextual_validator=contextual_validator,
            repair_hints=repair_hints,
            trace_context={
                "role": "researcher",
                "profile": self.profile_name,
                "profile_version": self.profile_version,
                "round_id": context.round_id,
                "phase": context.phase,
            },
        )

    def plan_external(self, context: ResearcherContextInput) -> ExternalRetrievalPlan:
        if context.phase != "external_retrieval":
            raise ValueError("plan_external requires a Phase A context")
        return self._complete(context, ExternalRetrievalPlan)

    def plan_features(self, context: ResearcherContextInput) -> FeatureEvidencePlan:
        if context.phase != "feature_evidence":
            raise ValueError("plan_features requires a Phase B context")
        return self._complete(context, FeatureEvidencePlan)


class MockResearcherClient:
    """Deterministic queued planner for unit and integration tests."""

    profile_name = "evidence_planner_v1"
    profile_version = "1.0.0"
    profile_hash = hashlib.sha256(b"mock-researcher:v1").hexdigest()

    def __init__(
        self,
        *,
        external_plans: Sequence[ExternalRetrievalPlan] = (),
        feature_plans: Sequence[FeatureEvidencePlan] = (),
    ) -> None:
        self.external_plans = list(external_plans)
        self.feature_plans = list(feature_plans)
        self.calls: list[ResearcherContextInput] = []

    @staticmethod
    def schema_hash(output_type: type[Any]) -> str:
        return _schema_hash(output_type)

    def plan_external(self, context: ResearcherContextInput) -> ExternalRetrievalPlan:
        self.calls.append(context)
        if not self.external_plans:
            return ExternalRetrievalPlan(
                decision="ABSTAIN",
                evidence_gap="",
                abstention_reason="No configured mock evidence gap.",
            )
        return self.external_plans.pop(0)

    def plan_features(self, context: ResearcherContextInput) -> FeatureEvidencePlan:
        self.calls.append(context)
        if not self.feature_plans:
            return FeatureEvidencePlan(
                decision="ABSTAIN",
                abstention_reason="No configured mock feature need.",
            )
        return self.feature_plans.pop(0)


def create_researcher_client(provider: str, **kwargs: Any) -> Any:
    if provider == "mock":
        allowed = {"external_plans", "feature_plans"}
        return MockResearcherClient(
            **{key: value for key, value in kwargs.items() if key in allowed}
        )
    if provider in {"openai", "openai_compatible", "deepseek"}:
        return NativeResearcherClient(provider=provider, **kwargs)
    raise ValueError(f"Unknown Researcher provider {provider!r}")
