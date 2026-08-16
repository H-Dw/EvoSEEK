"""Optional OpenAI Agents SDK adapters with DeepSeek Chat Completions compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from fitness_agents.contracts.schemas import Evidence, Hypothesis, ReThinkReflection
from fitness_agents.kg_interaction import KGToolSession, QueryIntent
from fitness_agents.utils.progress import report_event

from .llm import load_scientist_profile
from .output_contracts import HypothesisOutput, ReThinkOutput
from .remote_llm import resolve_api_key, resolve_base_url, resolve_model, uses_deepseek

OutputT = TypeVar("OutputT")


class _CompletionsProxy:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    async def create(self, **kwargs: Any) -> Any:
        response_format = kwargs.get("response_format")
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            kwargs["response_format"] = {"type": "json_object"}
            messages = [dict(item) for item in kwargs.get("messages", ())]
            schema_instruction = (
                "Return one JSON object matching this exact schema. Do not omit required keys: "
                + json.dumps(schema, ensure_ascii=False)
            )
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = (
                    str(messages[0].get("content", "")) + "\n\n" + schema_instruction
                )
            else:
                messages.insert(0, {"role": "system", "content": schema_instruction})
            kwargs["messages"] = messages
        return await self._delegate.create(**kwargs)


class _ChatProxy:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.completions = _CompletionsProxy(delegate.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


class _DeepSeekAsyncClientProxy:
    """Downgrade SDK json_schema requests to DeepSeek json_object, retaining local validation."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.chat = _ChatProxy(delegate.chat)
        self.base_url = delegate.base_url

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)


def _trace_id(metadata: dict[str, Any], attempt: int) -> str:
    raw = json.dumps([metadata, attempt], sort_keys=True, default=str).encode()
    return "trace_" + hashlib.sha256(raw).hexdigest()[:32]


class SDKStructuredRuntime:
    """Runs one typed SDK role; traces are observational and disabled unless opted in."""

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
        sdk_tracing_enabled: bool = False,
        sdk_max_turns: int = 6,
        sdk_model_retries: int = 2,
    ) -> None:
        try:
            from agents import ModelSettings
            from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
            from openai import AsyncOpenAI
            from openai.types.shared import Reasoning
        except ImportError as error:
            raise RuntimeError(
                "Install the 'agents-sdk' optional dependency to use runtime=agents_sdk"
            ) from error

        self.model_name = resolve_model(model, provider=provider)
        resolved_base = resolve_base_url(base_url, provider=provider)
        client: Any = AsyncOpenAI(
            api_key=resolve_api_key(api_key),
            base_url=resolved_base,
        )
        if uses_deepseek(self.model_name, resolved_base):
            client = _DeepSeekAsyncClientProxy(client)
        self.strict_function_tools = not uses_deepseek(
            self.model_name, resolved_base
        ) or str(resolved_base or "").rstrip("/").endswith("/beta")
        self.model = OpenAIChatCompletionsModel(
            model=self.model_name,
            openai_client=client,
        )
        reasoning = Reasoning(effort=reasoning_effort) if reasoning_effort else None
        extra_body = {"thinking": {"type": thinking}} if thinking else None
        self.model_settings = ModelSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning=reasoning,
            parallel_tool_calls=False,
            extra_body=extra_body,
        )
        self.tracing_enabled = sdk_tracing_enabled
        self.max_turns = sdk_max_turns
        self.model_retries = sdk_model_retries

    def run(
        self,
        *,
        role_name: str,
        instructions: str,
        input_payload: dict[str, Any],
        output_type: type[OutputT],
        tools: list[Any],
        trace_context: dict[str, Any],
        validate_output: Callable[[OutputT], None] | None = None,
    ) -> OutputT:
        from agents import Agent, RunConfig, Runner

        agent = Agent(
            name=role_name,
            instructions=instructions,
            model=self.model,
            model_settings=self.model_settings,
            tools=tools,
            handoffs=[],
            output_type=output_type,
        )
        last_error: Exception | None = None
        base_input = json.dumps(input_payload, ensure_ascii=False)
        for attempt in range(self.model_retries + 1):
            trace_id = _trace_id(trace_context, attempt)
            report_event(
                "sdk_agent_started",
                message=f"SDK {role_name} attempt {attempt + 1}",
                attempt=attempt,
                trace_id=trace_id,
                **trace_context,
            )
            try:
                run_input = base_input
                if last_error is not None:
                    run_input += (
                        "\n\nThe previous output failed validation. Return a complete corrected "
                        f"object. Error type: {type(last_error).__name__}."
                    )
                result = Runner.run_sync(
                    agent,
                    run_input,
                    max_turns=self.max_turns,
                    run_config=RunConfig(
                        tracing_disabled=not self.tracing_enabled,
                        trace_include_sensitive_data=False,
                        workflow_name=f"fitness-agents:{role_name}",
                        trace_id=trace_id,
                        group_id=str(trace_context.get("run_id", "")),
                        trace_metadata=dict(trace_context),
                    ),
                )
                output = result.final_output_as(output_type, raise_if_incorrect_type=True)
                if validate_output is not None:
                    validate_output(output)
                report_event(
                    "sdk_agent_completed",
                    message=f"SDK {role_name} completed",
                    attempt=attempt,
                    trace_id=trace_id,
                    **trace_context,
                )
                return output
            except Exception as error:  # noqa: BLE001 - typed provider boundary retries
                last_error = error
                report_event(
                    "sdk_agent_retry" if attempt < self.model_retries else "sdk_agent_failed",
                    message=f"SDK {role_name} validation/provider failure",
                    attempt=attempt,
                    trace_id=trace_id,
                    error_type=type(error).__name__,
                    **trace_context,
                )
        raise RuntimeError(f"SDK {role_name} failed") from last_error


def build_scientist_kg_tools(
    session: KGToolSession | None, *, strict_mode: bool = True
) -> list[Any]:
    if session is None:
        return []
    from agents import function_tool

    @function_tool(strict_mode=strict_mode)
    def kg_hypothesis_context(limit: int) -> dict[str, Any]:
        """Return bounded visible KG context for the current campaign round."""

        return session.call("hypothesis_context", QueryIntent.CONTEXT, {"limit": limit})

    @function_tool(strict_mode=strict_mode)
    def kg_explain_variant(variant_id: str) -> dict[str, Any]:
        """Explain one in-scope variant using only round-visible KG records."""

        return session.call(
            "explain_variant", QueryIntent.EXPLAIN, {"variant_id": variant_id}
        )

    @function_tool(strict_mode=strict_mode)
    def kg_compare_variants(variant_ids: list[str]) -> dict[str, Any]:
        """Compare in-scope variants through the bounded KG controller."""

        return session.call(
            "compare_variants", QueryIntent.COMPARE, {"variant_ids": variant_ids}
        )

    return [kg_hypothesis_context, kg_explain_variant, kg_compare_variants]


class AgentsSDKScientistLLMClient:
    provider_name = "agents_sdk_scientist"
    supports_kg_tools = True

    def __init__(
        self,
        *,
        profile: str = "scientific_v1",
        sdk_runtime: SDKStructuredRuntime | None = None,
        **runtime_kwargs: Any,
    ) -> None:
        self.profile_name = profile
        self.profile = load_scientist_profile(profile)
        self.profile_sha256 = hashlib.sha256(self.profile.encode()).hexdigest()
        self.runtime = sdk_runtime or SDKStructuredRuntime(**runtime_kwargs)

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
        kg_tool_session: KGToolSession | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        from .scientist import assert_sanitized

        del output_schema
        assert_sanitized(sanitized_context)
        expected_id = str(sanitized_context["expected_hypothesis_id"])
        expected_parent_id = sanitized_context.get("previous_hypothesis_id")
        allowed_evidence_ids = frozenset(entry.evidence_id for entry in evidence[:80])
        metadata = dict(trace_context or {})
        if kg_tool_session is not None:
            scope = sorted(kg_tool_session.context.allowed_variant_ids or ())
            metadata["variant_ids"] = scope
            metadata["variant_scope_count"] = len(scope)
            metadata["variant_scope_sha256"] = hashlib.sha256(
                json.dumps(scope).encode()
            ).hexdigest()
        output = self.runtime.run(
            role_name="ScientistAgent",
            instructions=self.profile,
            input_payload={
                "context": sanitized_context,
                "evidence": [entry.__dict__ for entry in evidence[:80]],
            },
            output_type=HypothesisOutput,
            tools=build_scientist_kg_tools(
                kg_tool_session,
                strict_mode=getattr(self.runtime, "strict_function_tools", True),
            ),
            trace_context=metadata,
            validate_output=lambda item: item.to_hypothesis(
                expected_hypothesis_id=expected_id,
                expected_parent_hypothesis_id=expected_parent_id,
                allowed_evidence_ids=allowed_evidence_ids,
            ),
        )
        return output.to_hypothesis(
            expected_hypothesis_id=expected_id,
            expected_parent_hypothesis_id=expected_parent_id,
            allowed_evidence_ids=allowed_evidence_ids,
        )


class AgentsSDKReThinkClient:
    provider_name = "agents_sdk_rethink"

    def __init__(
        self,
        *,
        sdk_runtime: SDKStructuredRuntime | None = None,
        **runtime_kwargs: Any,
    ) -> None:
        self.runtime = sdk_runtime or SDKStructuredRuntime(**runtime_kwargs)

    def reflect_round(self, *, context: dict[str, Any]) -> tuple[ReThinkReflection, ...]:
        run_id = str(context["run_id"])
        round_id = int(context["round_id"])
        expected_variants = {str(item["variant_id"]) for item in context.get("candidates", ())}

        def validate(output: ReThinkOutput) -> None:
            actual = {item.variant_id for item in output.reflections}
            if actual != expected_variants:
                raise ValueError("ReThink output must cover exactly the supplied candidate IDs")

        output = self.runtime.run(
            role_name="ReThinkAgent",
            instructions=(
                "You are the post-validation ReThink role. Use only the wet and dry values in the "
                "current CampaignRunner context. Wet values are authoritative; dry predictions are "
                "lower-fidelity. Do not invent data or call tools. Return one typed reflection per "
                "supplied candidate. You cannot submit experiments, query an oracle or final test, "
                "write the KG, approve a batch, or modify campaign state."
            ),
            input_payload=context,
            output_type=ReThinkOutput,
            tools=[],
            trace_context={
                "run_id": run_id,
                "round_id": round_id,
                "variant_ids": sorted(expected_variants),
                "role": "rethink",
            },
            validate_output=validate,
        )
        return output.to_reflections(
            run_id=run_id,
            round_id=round_id,
            provider=self.provider_name,
        )
