"""OpenAI Agents SDK adapters for Scientist and ReThink.

CampaignRunner remains the scientific state machine. These clients only execute the
authorized cognitive step: structured JSON plus optional round-scoped KG function tools.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from fitness_agents.agents.llm import HYPOTHESIS_SCHEMA, load_scientist_profile
from fitness_agents.agents.output_contracts import (
    HypothesisOutput,
    ReThinkOutput,
    validate_hypothesis_payload,
    validate_rethink_payload,
)
from fitness_agents.agents.remote_llm import (
    extract_json_object,
    load_project_env,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
    uses_deepseek,
)
from fitness_agents.agents.rethink import RETHINK_SCHEMA, _parse_reflections
from fitness_agents.contracts.schemas import Evidence, Hypothesis, ReThinkReflection
from fitness_agents.kg_interaction.contracts import QueryIntent
from fitness_agents.utils.progress import TimedHeartbeat, report_event

_SDK_IMPORT_ERROR = (
    "Install the agents-sdk extra before using llm.runtime=agents_sdk: "
    "pip install 'fitness-agents[agents-sdk]'"
)


def _import_agents() -> Any:
    try:
        import agents
    except ImportError as error:
        raise RuntimeError(_SDK_IMPORT_ERROR) from error
    return agents


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def _trace_metadata(trace_context: dict[str, Any] | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key, value in (trace_context or {}).items():
        if value is None:
            continue
        metadata[str(key)] = str(value)
    return metadata


def _payload_from_result(result: Any) -> dict[str, Any]:
    output = getattr(result, "final_output", result)
    if isinstance(output, (HypothesisOutput, ReThinkOutput)):
        by_alias = isinstance(output, HypothesisOutput)
        return output.model_dump(mode="json", by_alias=by_alias)
    if hasattr(output, "model_dump"):
        dumped = output.model_dump(mode="json", by_alias=True)
        if isinstance(dumped, dict):
            return dumped
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        return extract_json_object(output)
    raise TypeError(f"SDK agent returned unsupported output type {type(output)!r}")


def _validation_detail(error: Exception) -> str:
    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            entries = errors(include_input=False, include_url=False)
            summary = [
                {
                    "location": ".".join(str(part) for part in item.get("loc", ())),
                    "type": item.get("type"),
                    "message": item.get("msg"),
                }
                for item in entries[:12]
            ]
            return json.dumps(summary, ensure_ascii=False)
        except (TypeError, ValueError):
            pass
    return f"{type(error).__name__}: {error}"


def build_kg_function_tools(session: Any) -> list[Any]:
    """Wrap a round-scoped KGToolSession as Agents SDK function tools."""

    agents = _import_agents()

    @agents.function_tool
    def hypothesis_context(limit: int = 12) -> dict[str, Any]:
        """Read round-visible residue aggregates, measurements, and computed evidence."""
        return _json_safe(
            session.call("hypothesis_context", QueryIntent.CONTEXT, {"limit": int(limit)})
        )

    @agents.function_tool
    def explain_variant(variant_id: str) -> dict[str, Any]:
        """Explain one currently visible variant. Hidden oracle labels are not available."""
        return _json_safe(
            session.call(
                "explain_variant",
                QueryIntent.EXPLAIN,
                {"variant_id": str(variant_id)},
            )
        )

    @agents.function_tool
    def compare_variants(variant_ids: list[str]) -> dict[str, Any]:
        """Compare two or more currently visible variants and surface counterevidence."""
        cleaned = [str(item) for item in variant_ids]
        if len(cleaned) < 2:
            raise ValueError("compare_variants requires at least two variant_ids")
        return _json_safe(
            session.call(
                "compare_variants",
                QueryIntent.COMPARE,
                {"variant_ids": cleaned},
            )
        )

    return [hypothesis_context, explain_variant, compare_variants]


class _AgentsSDKRoleClient:
    """Shared Chat Completions + structured-JSON runner for one campaign role."""

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
        profile: str = "scientific_v1",
    ) -> None:
        load_project_env()
        self.provider = provider
        self.model = resolve_model(model, provider=provider)
        self.base_url = resolve_base_url(base_url, provider=provider)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.sdk_tracing_enabled = bool(sdk_tracing_enabled)
        self.sdk_max_turns = int(sdk_max_turns)
        self.sdk_model_retries = int(sdk_model_retries)
        self.profile_name = profile
        self._openai_client: Any | None = None
        self._model: Any | None = None

    def _ensure_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._openai_client is not None:
            return self._model, self._openai_client
        agents = _import_agents()
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=resolve_api_key(self.api_key),
            base_url=self.base_url,
            max_retries=0,
        )
        self._openai_client = client
        self._model = agents.OpenAIChatCompletionsModel(
            model=self.model,
            openai_client=client,
        )
        return self._model, client

    def _model_settings(self, *, thinking: str | None) -> Any:
        agents = _import_agents()
        extra_args: dict[str, Any] = {}
        extra_body: dict[str, Any] = {}
        deepseek = uses_deepseek(self.model, self.base_url)
        effort = self.reasoning_effort
        if deepseek:
            effort = effort or "high"
            thinking = thinking or "enabled"
        if deepseek and effort:
            extra_args["reasoning_effort"] = effort
        if deepseek and thinking:
            extra_body["thinking"] = {"type": thinking}
        return agents.ModelSettings(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_args=extra_args or None,
            extra_body=extra_body or None,
        )

    def _run_sdk(
        self,
        *,
        role: str,
        instructions: str,
        user_payload: dict[str, Any],
        output_model: type[HypothesisOutput | ReThinkOutput],
        validator: Callable[[dict[str, Any]], dict[str, Any]],
        kg_tool_session: Any | None,
        trace_context: dict[str, Any] | None,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        agents = _import_agents()
        model, _client = self._ensure_model()
        if not self.sdk_tracing_enabled:
            agents.set_tracing_disabled(True)

        tools = build_kg_function_tools(kg_tool_session) if kg_tool_session is not None else []
        use_structured = not uses_deepseek(self.model, self.base_url)
        output_type: Any | None = None
        if use_structured:
            output_type = agents.AgentOutputSchema(output_model, strict_json_schema=False)

        system = (
            instructions
            + "\n\nReply with a single JSON object that matches this schema: "
            + json.dumps(schema, ensure_ascii=False)
            + " Do not include markdown."
        )
        user_text = json.dumps(user_payload, ensure_ascii=False)
        input_items: list[dict[str, str]] = [{"role": "user", "content": user_text}]
        current_thinking = self.thinking
        last_error: Exception | None = None
        metadata = _trace_metadata(trace_context)

        for attempt in range(self.sdk_model_retries + 1):
            agent = agents.Agent(
                name=role,
                instructions=system,
                model=model,
                model_settings=self._model_settings(thinking=current_thinking),
                tools=tools,
                output_type=output_type,
            )
            run_config = agents.RunConfig(
                tracing_disabled=not self.sdk_tracing_enabled,
                workflow_name=f"fitness-agents:{role}",
                trace_metadata=metadata or None,
            )
            report_event(
                "sdk_request_started",
                message=f"SDK {role} {self.model} attempt {attempt + 1}/{self.sdk_model_retries + 1}",
                model=self.model,
                role=role,
                attempt=attempt,
                max_turns=self.sdk_max_turns,
                kg_tools=bool(tools),
            )
            started = time.perf_counter()
            result = None
            try:
                with TimedHeartbeat(f"SDK {role} {self.model} attempt {attempt + 1}"):
                    result = agents.Runner.run_sync(
                        agent,
                        input_items,
                        max_turns=self.sdk_max_turns,
                        run_config=run_config,
                    )
                payload = validator(_payload_from_result(result))
                report_event(
                    "sdk_request_completed",
                    message=f"SDK {role} {self.model} completed",
                    model=self.model,
                    role=role,
                    attempt=attempt,
                    latency_s=round(time.perf_counter() - started, 3),
                )
                return payload
            except Exception as error:  # noqa: BLE001 - retry schema/provider failures together
                last_error = error
                report_event(
                    "sdk_request_retry",
                    message=f"SDK {role} {self.model} retry ({type(error).__name__})",
                    model=self.model,
                    role=role,
                    attempt=attempt,
                    error_type=type(error).__name__,
                    latency_s=round(time.perf_counter() - started, 3),
                )
                if uses_deepseek(self.model, self.base_url) and current_thinking == "enabled":
                    current_thinking = "disabled"
                if attempt < self.sdk_model_retries:
                    detail = _validation_detail(error)
                    correction = (
                        "The previous JSON failed the required output contract: "
                        f"{detail}. Return a complete corrected JSON object with every "
                        "required key and no Markdown."
                    )
                    to_input_list = getattr(result, "to_input_list", None)
                    if callable(to_input_list):
                        input_items = list(to_input_list()) + [
                            {"role": "user", "content": correction}
                        ]
                    else:
                        input_items = [
                            {"role": "user", "content": user_text},
                            {"role": "user", "content": correction},
                        ]
                continue
        report_event(
            "sdk_request_failed",
            message=f"SDK {role} {self.model} failed",
            model=self.model,
            role=role,
            error_type=type(last_error).__name__ if last_error is not None else "RuntimeError",
        )
        raise RuntimeError(f"Agents SDK {role} JSON completion failed") from last_error


class AgentsSDKScientistLLMClient(_AgentsSDKRoleClient):
    provider_name = "agents_sdk"
    supports_kg_tools = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.profile = load_scientist_profile(self.profile_name)
        self.profile_sha256 = hashlib.sha256(self.profile.encode()).hexdigest()

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
        kg_tool_session: Any | None = None,
        trace_context: dict[str, Any] | None = None,
    ) -> Hypothesis:
        del output_schema
        evidence_payload = [entry.__dict__ for entry in evidence[:80]]
        expected_id = str(sanitized_context["expected_hypothesis_id"])
        expected_parent_id = sanitized_context.get("previous_hypothesis_id")
        allowed_evidence_ids = frozenset(entry.evidence_id for entry in evidence[:80])
        if kg_tool_session is not None:
            allowed_evidence_ids = None
        payload = self._run_sdk(
            role="scientist",
            instructions=self.profile,
            user_payload={"context": sanitized_context, "evidence": evidence_payload},
            output_model=HypothesisOutput,
            validator=lambda value: validate_hypothesis_payload(
                value,
                expected_hypothesis_id=expected_id,
                expected_parent_hypothesis_id=expected_parent_id,
                allowed_evidence_ids=allowed_evidence_ids,
            ),
            kg_tool_session=kg_tool_session,
            trace_context=trace_context,
            schema=HYPOTHESIS_SCHEMA,
        )
        return HypothesisOutput.model_validate(payload).to_hypothesis(
            expected_hypothesis_id=expected_id,
            expected_parent_hypothesis_id=expected_parent_id,
            allowed_evidence_ids=allowed_evidence_ids,
        )


class AgentsSDKReThinkClient(_AgentsSDKRoleClient):
    provider_name = "agents_sdk_rethink"

    def reflect_round(self, *, context: dict[str, Any]) -> tuple[ReThinkReflection, ...]:
        payload = self._run_sdk(
            role="rethink",
            instructions=(
                "You are the ReThink Agent in an iterative protein-design campaign. "
                "For every candidate, assess whether the original recommendation reason "
                "is supported, contradicted, mixed, or inconclusive using supplied wet and "
                "dry validation. Wet measurements are authoritative; dry model values are "
                "lower-fidelity evidence. Do not invent measurements."
            ),
            user_payload=context,
            output_model=ReThinkOutput,
            validator=validate_rethink_payload,
            kg_tool_session=None,
            trace_context={
                "run_id": context.get("run_id"),
                "round_id": context.get("round_id"),
                "role": "rethink",
            },
            schema=RETHINK_SCHEMA,
        )
        return _parse_reflections(
            payload,
            run_id=str(context["run_id"]),
            round_id=int(context["round_id"]),
            provider=self.provider_name,
        )
