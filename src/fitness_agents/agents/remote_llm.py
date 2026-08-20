"""Shared OpenAI-compatible chat client used by Scientist and Critic.

DeepSeek V4 is accessed through the Chat Completions API, not the OpenAI Responses API.
Keys are read from the process environment or a gitignored project ``.env`` file.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fitness_agents.utils.progress import TimedHeartbeat, report_event, report_prompt_budget

from .output_guards import (
    MAX_OUTPUT_TOKENS,
    ContentFilteredError,
    EmptyLLMOutputError,
    OutputTruncatedError,
    PromptBudgetExceededError,
    ProviderResourceError,
    TokenBudgetPolicy,
    UnexpectedFinishReasonError,
    classify_output_failure,
    json_salvage,
    retry_instruction,
    validation_error_entries,
)
from .transports import ChatTransport

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"

_completion_receipt: ContextVar[dict[str, Any] | None] = ContextVar(
    "fitness_agents_llm_completion_receipt", default=None
)


def reset_completion_receipt() -> None:
    """Clear request metadata in the current role context before a completion."""

    _completion_receipt.set(None)


def completion_receipt_snapshot() -> dict[str, Any]:
    """Return size/status metadata for the latest completion in this role context."""

    return dict(_completion_receipt.get() or {})


def _record_completion_receipt(
    *,
    input_chars: int,
    request_started: bool,
    failure_category: str | None,
) -> None:
    _completion_receipt.set(
        {
            "input_chars": input_chars,
            "failure_category": failure_category,
            "request_started": request_started,
        }
    )


class RemoteLLMCompletionError(RuntimeError):
    """Structured terminal failure for one bounded remote completion."""

    def __init__(
        self,
        error_code: str,
        *,
        failure_category: str,
        input_chars: int,
        request_started: bool,
        detail: str,
    ) -> None:
        super().__init__(f"{error_code}: {detail}")
        self.error_code = error_code
        self.failure_category = failure_category
        self.input_chars = input_chars
        self.request_started = request_started


def _json_char_count(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))


def _measure_field_chars(
    value: Any,
    *,
    prefix: str,
    output: dict[str, int],
    depth: int = 0,
) -> None:
    if depth >= 4:
        return
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw_key))[:80] or "field"
            path = f"{prefix}.{key}" if prefix else key
            output[path] = output.get(path, 0) + _json_char_count(item)
            _measure_field_chars(item, prefix=path, output=output, depth=depth + 1)
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        keys = tuple(dict.fromkeys(str(key) for item in value for key in item))[:64]
        for raw_key in keys:
            key = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_key)[:80] or "field"
            path = f"{prefix}[].{key}"
            output[path] = output.get(path, 0) + sum(
                _json_char_count(item[raw_key]) for item in value if raw_key in item
            )


def prompt_budget_record(
    messages: list[dict[str, str]],
    *,
    max_input_chars: int | None,
    trace_context: dict[str, Any] | None,
    request_started: bool,
) -> dict[str, Any]:
    field_chars: dict[str, int] = {}
    for item in messages:
        role = str(item.get("role", "unknown"))
        content = str(item.get("content", ""))
        try:
            parsed = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        _measure_field_chars(parsed, prefix=role, output=field_chars)
    largest_fields = sorted(field_chars.items(), key=lambda item: (-item[1], item[0]))[:128]
    trace = trace_context or {}
    input_chars = sum(len(str(item.get("content", ""))) for item in messages)
    utilization_ratio = (
        round(input_chars / max_input_chars, 6) if max_input_chars else None
    )
    budget_band = None
    if utilization_ratio is not None:
        budget_band = (
            "exceeded"
            if utilization_ratio > 1.0
            else "critical"
            if utilization_ratio >= 0.9
            else "warning"
            if utilization_ratio >= 0.8
            else "normal"
        )
    return {
        "role": str(trace.get("role") or "unknown"),
        "profile": trace.get("profile"),
        "system_chars": sum(
            len(str(item.get("content", ""))) for item in messages if item.get("role") == "system"
        ),
        "user_chars": sum(
            len(str(item.get("content", ""))) for item in messages if item.get("role") == "user"
        ),
        "assistant_chars": sum(
            len(str(item.get("content", "")))
            for item in messages
            if item.get("role") == "assistant"
        ),
        "input_chars": input_chars,
        "field_chars": dict(largest_fields),
        "max_input_chars": max_input_chars,
        "remaining_chars": (
            max_input_chars - input_chars if max_input_chars is not None else None
        ),
        "utilization_ratio": utilization_ratio,
        "budget_band": budget_band,
        "request_started": request_started,
        "round_id": trace.get("round_id"),
    }


def _terminal_failure(
    error: Exception | None,
    *,
    failure: Any | None,
    disposition: RequestFailureDisposition | None,
    input_chars: int,
    request_started: bool,
) -> RemoteLLMCompletionError:
    if isinstance(error, PromptBudgetExceededError):
        code = "PROMPT_BUDGET_EXCEEDED"
        category = "budget"
    elif isinstance(error, ContentFilteredError):
        code = "CONTENT_FILTERED"
        category = "output"
    elif isinstance(error, UnexpectedFinishReasonError):
        code = "OUTPUT_FINISH_REASON_INVALID"
        category = "output"
    elif isinstance(error, ProviderResourceError):
        code = "PROVIDER_RESOURCE_UNAVAILABLE"
        category = "transport"
    else:
        status_code = _exception_status_code(error) if error is not None else None
        if status_code is not None:
            code = f"HTTP_{status_code}"
            category = "transport"
        elif failure is not None and failure.kind == "schema":
            code = "OUTPUT_SCHEMA_INVALID"
            category = "output"
        elif failure is not None and failure.kind == "semantic":
            code = "OUTPUT_SEMANTIC_INVALID"
            category = "output"
        elif failure is not None and failure.kind == "syntax":
            code = "OUTPUT_JSON_INVALID"
            category = "output"
        elif failure is not None and failure.kind == "truncated":
            code = "OUTPUT_TRUNCATED"
            category = "output"
        elif failure is not None and failure.kind == "unknown_evidence":
            code = "OUTPUT_EVIDENCE_IDS_INVALID"
            category = "output"
        elif failure is not None and failure.kind == "empty":
            code = "OUTPUT_EMPTY"
            category = "output"
        elif isinstance(error, TimeoutError):
            code = "TRANSPORT_TIMEOUT"
            category = "transport"
        elif isinstance(error, ConnectionError):
            code = "TRANSPORT_CONNECTION_ERROR"
            category = "transport"
        elif disposition is not None and disposition.category == "transport":
            code = "TRANSPORT_ERROR"
            category = "transport"
        else:
            code = "OUTPUT_INVALID"
            category = "output"
    return RemoteLLMCompletionError(
        code,
        failure_category=category,
        input_chars=input_chars,
        request_started=request_started,
        detail=str(error)[:800] if error is not None else "remote completion failed",
    )


def load_project_env(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from ``.env`` without overwriting existing process env."""

    env_path = Path(path) if path is not None else None
    if env_path is None:
        try:
            from fitness_agents.config import project_root

            env_path = project_root() / ".env"
        except FileNotFoundError:
            return
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_secret(value: object, *env_names: str) -> str | None:
    if isinstance(value, str) and value.startswith("env:"):
        return os.environ.get(value[4:]) or None
    if isinstance(value, str) and value and not value.startswith("REPLACE_WITH_"):
        return value
    for name in env_names:
        resolved = os.environ.get(name)
        if resolved:
            return resolved
    return None


def resolve_api_key(explicit: str | None = None) -> str:
    load_project_env()
    key = resolve_secret(
        explicit,
        "FITNESS_AGENTS_LLM_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not key:
        raise RuntimeError(
            "Set FITNESS_AGENTS_LLM_API_KEY, DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY"
        )
    return key


def resolve_base_url(explicit: str | None = None, *, provider: str | None = None) -> str | None:
    load_project_env()
    if explicit and not str(explicit).startswith("REPLACE_WITH_"):
        return str(explicit)
    for name in ("FITNESS_AGENTS_LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"):
        value = os.environ.get(name)
        if value:
            return value
    if provider == "deepseek" or (os.environ.get("FITNESS_AGENTS_LLM_MODEL") or "").startswith(
        "deepseek"
    ):
        return DEEPSEEK_DEFAULT_BASE_URL
    return None


def resolve_model(explicit: str | None = None, *, provider: str | None = None) -> str:
    load_project_env()
    if explicit:
        return explicit
    env_model = os.environ.get("FITNESS_AGENTS_LLM_MODEL")
    if env_model:
        return env_model
    if provider == "deepseek":
        return DEEPSEEK_DEFAULT_MODEL
    return "gpt-5-mini"


def uses_deepseek(model: str, base_url: str | None) -> bool:
    blob = f"{model} {base_url or ''}".lower()
    return "deepseek" in blob


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    provider: str | None = None,
    request_timeout_seconds: float = 120.0,
):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install requirements/llm.txt to use a remote LLM") from error
    return OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=resolve_base_url(base_url, provider=provider),
        max_retries=0,
        timeout=request_timeout_seconds,
    )


@dataclass(frozen=True)
class RequestFailureDisposition:
    """Whether a provider/transport failure can consume the transport budget."""

    category: Literal["transport", "output"]
    retryable: bool
    status_code: int | None = None


_RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NON_RETRYABLE_HTTP_STATUS = frozenset({400, 401, 402, 403, 404, 405, 422})


def _exception_status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if value is None:
        value = getattr(getattr(error, "response", None), "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_request_failure(error: Exception) -> RequestFailureDisposition:
    """Classify API/transport failures without conflating them with JSON repair."""

    if isinstance(
        error,
        (ContentFilteredError, PromptBudgetExceededError, UnexpectedFinishReasonError),
    ):
        return RequestFailureDisposition("output", False, None)
    status_code = _exception_status_code(error)
    if status_code is not None:
        if status_code in _RETRYABLE_HTTP_STATUS or status_code >= 500:
            return RequestFailureDisposition("transport", True, status_code)
        if status_code in _NON_RETRYABLE_HTTP_STATUS or 400 <= status_code < 500:
            return RequestFailureDisposition("transport", False, status_code)
    error_name = type(error).__name__.lower()
    if isinstance(error, (TimeoutError, ConnectionError)) or any(
        marker in error_name
        for marker in ("timeout", "connection", "ratelimit", "internalserver")
    ):
        return RequestFailureDisposition("transport", True, status_code)
    return RequestFailureDisposition("output", True, status_code)


def extract_json_object(text: str) -> dict[str, Any]:
    """Accept one complete JSON object, with cosmetic fence/comma repair only."""

    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, count=1, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        salvaged = json_salvage(payload)
        if salvaged is not None:
            return salvaged
        raise
    if isinstance(parsed, dict):
        return parsed
    raise TypeError("Remote LLM JSON payload is not an object")


def _message_content(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        joined = "".join(parts).strip()
        if joined:
            return joined
    return ""


def _usage_payload(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    payload: dict[str, Any] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }
    if details is not None:
        payload["reasoning_tokens"] = getattr(details, "reasoning_tokens", None)
    return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class OutputRetryBudgets:
    """Independent repair budgets; one failure class cannot consume another."""

    truncated: int = 1
    syntax: int = 1
    schema: int = 2
    semantic: int = 1
    unknown_evidence: int = 1
    empty: int = 1
    other: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "truncated": self.truncated,
            "syntax": self.syntax,
            "schema": self.schema,
            "semantic": self.semantic,
            "unknown_evidence": self.unknown_evidence,
            "empty": self.empty,
            "other": self.other,
        }


def _private_raw_capture(
    content: str,
    *,
    trace_fields: dict[str, Any],
    attempt: int,
) -> str | None:
    """Optionally persist raw visible output outside normal artifacts and traces."""

    private_dir = os.environ.get("FITNESS_AGENTS_PRIVATE_LLM_OUTPUT_DIR")
    if not private_dir or not content:
        return None
    role = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(trace_fields.get("role") or "unknown"))
    target_dir = Path(private_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{role}.attempt-{attempt}.txt"
    target.write_text(content, encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass
    return str(target)


def complete_json(
    *,
    client: Any | None,
    transport: ChatTransport | None = None,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    retries: int = 2,
    transport_retries: int | None = None,
    truncation_retries: int | None = None,
    syntax_retries: int | None = None,
    schema_retries: int | None = None,
    semantic_retries: int | None = None,
    unknown_evidence_retries: int | None = None,
    retry_backoff_seconds: float = 0.0,
    allow_unknown_evidence_stripping: bool = False,
    max_input_chars: int | None = None,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    repair_hints: dict[str, tuple[str, ...] | list[str]] | None = None,
    trace_context: dict[str, Any] | None = None,
    preserve_thinking_on_retry: bool = False,
) -> dict[str, Any]:
    """Request one typed object with separate transport/repair retry budgets."""

    # Callers include the generated schema in their prompt. DeepSeek's compatibility path supports
    # json_object rather than OpenAI's json_schema response format, so validation is local.
    schema = schema or {}
    # Retained as a config/API compatibility field. Formal extraction never
    # rewrites identifier-bearing payloads after validation.
    del allow_unknown_evidence_stripping
    load_project_env()
    token_budget = max_tokens or int(os.environ.get("FITNESS_AGENTS_LLM_MAX_TOKENS", "20000"))
    if not 1 <= token_budget <= MAX_OUTPUT_TOKENS:
        raise ValueError(
            f"max_tokens must be between 1 and {MAX_OUTPUT_TOKENS}; got {token_budget}"
        )
    if transport is None and client is None:
        raise ValueError("A chat transport or compatible client is required")
    connection = transport if transport is not None else client
    client_base = str(getattr(connection, "base_url", "") or "")
    env_base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("FITNESS_AGENTS_LLM_BASE_URL")
    deepseek = uses_deepseek(model, client_base) or (
        not client_base and uses_deepseek(model, env_base)
    )
    # DeepSeek-only env defaults must not leak into Qwen/DashScope Chat Completions.
    if deepseek:
        effort = reasoning_effort or os.environ.get("FITNESS_AGENTS_LLM_REASONING_EFFORT") or "high"
        thinking_mode = thinking or os.environ.get("FITNESS_AGENTS_LLM_THINKING") or "enabled"
    else:
        effort = reasoning_effort
        thinking_mode = thinking
    # Provider adapters may otherwise infer an effort/default that silently
    # re-enables reasoning. Disabled thinking always means no effort field.
    if thinking_mode == "disabled":
        effort = None

    if retries < 0:
        raise ValueError("retries must be non-negative")
    transport_retry_limit = retries if transport_retries is None else transport_retries
    budgets = OutputRetryBudgets(
        truncated=1 if truncation_retries is None else truncation_retries,
        syntax=1 if syntax_retries is None else syntax_retries,
        schema=2 if schema_retries is None else schema_retries,
        semantic=1 if semantic_retries is None else semantic_retries,
        unknown_evidence=(
            1 if unknown_evidence_retries is None else unknown_evidence_retries
        ),
        empty=1,
        other=0,
    )
    if transport_retry_limit < 0 or any(value < 0 for value in budgets.as_dict().values()):
        raise ValueError("transport and output retry budgets must be non-negative")
    if retry_backoff_seconds < 0:
        raise ValueError("retry_backoff_seconds must be non-negative")

    reset_completion_receipt()
    last_error: Exception | None = None
    last_failure: Any | None = None
    last_disposition: RequestFailureDisposition | None = None
    policy = TokenBudgetPolicy(budget=token_budget, thinking=thinking_mode, effort=effort)
    current_messages = list(messages)
    trace_fields = dict(trace_context or {})
    transport_retries_used = 0
    output_retries_used = {kind: 0 for kind in budgets.as_dict()}
    request_attempt = 0
    request_started_any = False
    input_chars = 0
    max_external_attempts = 1 + transport_retry_limit + sum(budgets.as_dict().values())
    while request_attempt < max_external_attempts:
        input_chars = sum(len(str(item.get("content", ""))) for item in current_messages)
        if max_input_chars is not None and input_chars > max_input_chars:
            last_error = PromptBudgetExceededError(
                f"projected prompt has {input_chars} characters; limit is {max_input_chars}"
            )
            last_failure = classify_output_failure(
                last_error, finish_reason=None, content="", usage={}
            )
            last_disposition = classify_request_failure(last_error)
            report_prompt_budget(
                **prompt_budget_record(
                    current_messages,
                    max_input_chars=max_input_chars,
                    trace_context=trace_fields,
                    request_started=False,
                )
            )
            report_event(
                "llm_prompt_budget_exceeded",
                message=f"LLM request {model} exceeded prompt preflight budget",
                model=model,
                input_chars=input_chars,
                max_input_chars=max_input_chars,
                **trace_fields,
            )
            break
        budget_record = prompt_budget_record(
            current_messages,
            max_input_chars=max_input_chars,
            trace_context=trace_fields,
            request_started=True,
        )
        report_prompt_budget(**budget_record)
        if budget_record["budget_band"] in {"warning", "critical"}:
            report_event(
                "llm_prompt_budget_high_water",
                message=f"LLM request {model} reached prompt high-water mark",
                model=model,
                input_chars=budget_record["input_chars"],
                max_input_chars=max_input_chars,
                utilization_ratio=budget_record["utilization_ratio"],
                budget_band=budget_record["budget_band"],
                **trace_fields,
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": current_messages,
            "stream": False,
            "max_tokens": policy.budget,
            "response_format": {"type": "json_object"},
        }
        if not (deepseek and policy.thinking == "enabled"):
            kwargs["temperature"] = temperature
        extra_body: dict[str, Any] = {}
        if deepseek and policy.effort:
            kwargs["reasoning_effort"] = policy.effort
        if deepseek and policy.thinking:
            extra_body["thinking"] = {"type": policy.thinking}
        if extra_body:
            kwargs["extra_body"] = extra_body
        report_event(
            "llm_request_started",
            message=(
                f"LLM request {model} attempt {request_attempt + 1}/"
                f"{max_external_attempts}"
            ),
            model=model,
            attempt=request_attempt,
            thinking=policy.thinking,
            max_tokens=policy.budget,
            input_chars=input_chars,
                reasoning_effort=policy.effort,
                retry_budget={
                    "limits": budgets.as_dict(),
                    "consumed": dict(output_retries_used),
                    "transport_limit": transport_retry_limit,
                    "transport_consumed": transport_retries_used,
                },
            **trace_fields,
        )
        started = time.perf_counter()
        finish_reason = None
        content = ""
        usage: dict[str, Any] = {}
        try:
            request_started_any = True
            with TimedHeartbeat(f"LLM {model} attempt {request_attempt + 1}"):
                response = (
                    transport.create_chat_completion(**kwargs)
                    if transport is not None
                    else client.chat.completions.create(**kwargs)
                )
            choice = response.choices[0]
            finish_reason = getattr(choice, "finish_reason", None)
            usage = _usage_payload(response)
            content = _message_content(choice.message)
            normalized_finish = str(finish_reason or "").lower()
            if normalized_finish in {"length", "max_tokens", "max_output_tokens"}:
                raise OutputTruncatedError(
                    f"Remote LLM hit {finish_reason}; partial output is never accepted"
                )
            if normalized_finish == "content_filter":
                raise ContentFilteredError("Remote LLM completion was content-filtered")
            if normalized_finish == "insufficient_system_resource":
                raise ProviderResourceError(
                    "Remote LLM reported insufficient system resources"
                )
            if normalized_finish in {"tool_calls", "function_call"}:
                raise UnexpectedFinishReasonError(
                    f"JSON-only role returned unexpected finish_reason={finish_reason}"
                )
            if normalized_finish and normalized_finish != "stop":
                raise UnexpectedFinishReasonError(
                    f"Unsupported finish_reason={finish_reason}"
                )
            if not content:
                raise EmptyLLMOutputError("Remote LLM returned empty message content")
            payload = extract_json_object(content)
            if validator is not None:
                payload = validator(payload)
            report_event(
                "llm_request_completed",
                message=f"LLM request {model} completed",
                model=model,
                attempt=request_attempt,
                thinking=policy.thinking,
                latency_s=round(time.perf_counter() - started, 3),
                finish_reason=finish_reason,
                disposition="accepted",
                retry_budget={
                    "limits": budgets.as_dict(),
                    "consumed": dict(output_retries_used),
                    "transport_limit": transport_retry_limit,
                    "transport_consumed": transport_retries_used,
                },
                **usage,
                **trace_fields,
            )
            _record_completion_receipt(
                input_chars=input_chars,
                request_started=True,
                failure_category=None,
            )
            return payload
        except Exception as error:  # noqa: BLE001 - retry JSON/thinking failures
            last_error = error
            disposition = classify_request_failure(error)
            failure = classify_output_failure(
                error, finish_reason=finish_reason, content=content, usage=usage
            )
            last_disposition = disposition
            last_failure = failure
            failure_budget = budgets.as_dict().get(failure.kind, 0)
            will_retry = (
                disposition.retryable
                and (
                    transport_retries_used < transport_retry_limit
                    if disposition.category == "transport"
                    else output_retries_used.get(failure.kind, 0) < failure_budget
                )
            )
            private_raw_path = (
                _private_raw_capture(
                    content,
                    trace_fields=trace_fields,
                    attempt=request_attempt,
                )
                if content
                else None
            )
            report_event(
                "llm_request_retry" if will_retry else "llm_request_rejected",
                message=(
                    f"LLM request {model} "
                    f"{'retry' if will_retry else 'rejected'} "
                    f"({failure.kind}:{type(error).__name__})"
                ),
                model=model,
                attempt=request_attempt,
                thinking=policy.thinking,
                error_type=type(error).__name__,
                failure_kind=failure.kind,
                failure_category=disposition.category,
                retryable=disposition.retryable,
                will_retry=will_retry,
                status_code=disposition.status_code,
                finish_reason=finish_reason,
                content_length=failure.content_length,
                braces_balanced=failure.braces_balanced,
                decode_position=failure.decode_position,
                validation_errors=validation_error_entries(error),
                private_raw_path=private_raw_path,
                disposition="retry" if will_retry else "rejected",
                retry_budget={
                    "limits": budgets.as_dict(),
                    "consumed": dict(output_retries_used),
                    "transport_limit": transport_retry_limit,
                    "transport_consumed": transport_retries_used,
                },
                latency_s=round(time.perf_counter() - started, 3),
                **usage,
                **trace_fields,
            )
            if disposition.category == "transport":
                if not will_retry:
                    break
                transport_retries_used += 1
                if retry_backoff_seconds:
                    time.sleep(
                        retry_backoff_seconds * (2 ** (transport_retries_used - 1))
                    )
            else:
                if not will_retry:
                    break
                output_retries_used[failure.kind] += 1
                if (
                    deepseek
                    and policy.thinking == "enabled"
                    and not preserve_thinking_on_retry
                ):
                    policy.thinking = "disabled"
                if policy.thinking == "disabled":
                    policy.effort = None
                policy.apply(
                    failure,
                    deepseek=deepseek,
                    preserve_thinking=preserve_thinking_on_retry,
                )
                retry_messages = list(messages)
                if content and failure.kind != "truncated":
                    retry_messages.append({"role": "assistant", "content": content})
                retry_messages.append(
                    {
                        "role": "user",
                        "content": retry_instruction(
                            failure,
                            error=error,
                            schema=schema,
                            repair_hints=repair_hints,
                        ),
                    }
                )
                current_messages = retry_messages
            request_attempt += 1
            continue
        request_attempt += 1
    report_event(
        "llm_request_failed",
        message=f"LLM request {model} failed",
        model=model,
        error_type=type(last_error).__name__ if last_error is not None else "RuntimeError",
        disposition="failed",
        retry_budget={
            "limits": budgets.as_dict(),
            "consumed": dict(output_retries_used),
            "transport_limit": transport_retry_limit,
            "transport_consumed": transport_retries_used,
        },
        **trace_fields,
    )
    terminal = _terminal_failure(
        last_error,
        failure=last_failure,
        disposition=last_disposition,
        input_chars=input_chars,
        request_started=request_started_any,
    )
    _record_completion_receipt(
        input_chars=terminal.input_chars,
        request_started=terminal.request_started,
        failure_category=terminal.failure_category,
    )
    raise terminal from last_error
