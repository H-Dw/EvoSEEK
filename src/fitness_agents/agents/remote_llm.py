"""Shared OpenAI-compatible chat client used by Scientist and Critic.

DeepSeek V4 is accessed through the Chat Completions API, not the OpenAI Responses API.
Keys are read from the process environment or a gitignored project ``.env`` file.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from fitness_agents.utils.progress import TimedHeartbeat, report_event

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"


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
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
    )
    if not key:
        raise RuntimeError(
            "Set FITNESS_AGENTS_LLM_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY"
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


def create_openai_client(*, api_key: str | None = None, base_url: str | None = None, provider: str | None = None):
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install requirements/llm.txt to use a remote LLM") from error
    return OpenAI(
        api_key=resolve_api_key(api_key),
        base_url=resolve_base_url(base_url, provider=provider),
    )


def extract_json_object(text: str) -> dict[str, Any]:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, count=1, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Remote LLM response did not contain a JSON object")
    parsed = json.loads(payload[start : end + 1])
    if not isinstance(parsed, dict):
        raise TypeError("Remote LLM JSON payload is not an object")
    return parsed


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


def complete_json(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    """Request JSON via Chat Completions, with DeepSeek thinking-mode extras when configured."""

    del schema  # Included in the prompt by callers; DeepSeek only supports json_object.
    load_project_env()
    token_budget = max_tokens or int(os.environ.get("FITNESS_AGENTS_LLM_MAX_TOKENS", "16384"))
    effort = reasoning_effort or os.environ.get("FITNESS_AGENTS_LLM_REASONING_EFFORT")
    thinking_mode = thinking or os.environ.get("FITNESS_AGENTS_LLM_THINKING")
    client_base = str(getattr(client, "base_url", "") or "")
    deepseek = uses_deepseek(model, client_base) or uses_deepseek(
        model, os.environ.get("OPENAI_BASE_URL") or os.environ.get("FITNESS_AGENTS_LLM_BASE_URL")
    )
    if deepseek:
        effort = effort or "high"
        thinking_mode = thinking_mode or "enabled"

    last_error: Exception | None = None
    current_thinking = thinking_mode
    for attempt in range(retries + 1):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": token_budget,
            "response_format": {"type": "json_object"},
        }
        extra_body: dict[str, Any] = {}
        if deepseek and effort:
            kwargs["reasoning_effort"] = effort
        if deepseek and current_thinking:
            extra_body["thinking"] = {"type": current_thinking}
        if extra_body:
            kwargs["extra_body"] = extra_body
        report_event(
            "llm_request_started",
            message=f"LLM request {model} attempt {attempt + 1}/{retries + 1}",
            model=model,
            attempt=attempt,
            thinking=current_thinking,
            max_tokens=token_budget,
        )
        started = time.perf_counter()
        try:
            with TimedHeartbeat(f"LLM {model} attempt {attempt + 1}"):
                response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            content = _message_content(message)
            if not content:
                raise ValueError("Remote LLM returned empty message content")
            payload = extract_json_object(content)
            report_event(
                "llm_request_completed",
                message=f"LLM request {model} completed",
                model=model,
                attempt=attempt,
                thinking=current_thinking,
                latency_s=round(time.perf_counter() - started, 3),
                finish_reason=getattr(response.choices[0], "finish_reason", None),
                **_usage_payload(response),
            )
            return payload
        except Exception as error:  # noqa: BLE001 - retry JSON/thinking failures
            last_error = error
            report_event(
                "llm_request_retry",
                message=f"LLM request {model} retry ({type(error).__name__})",
                model=model,
                attempt=attempt,
                thinking=current_thinking,
                error_type=type(error).__name__,
                latency_s=round(time.perf_counter() - started, 3),
            )
            if deepseek and current_thinking == "enabled":
                current_thinking = "disabled"
            continue
    report_event(
        "llm_request_failed",
        message=f"LLM request {model} failed",
        model=model,
        error_type=type(last_error).__name__ if last_error is not None else "RuntimeError",
    )
    raise RuntimeError("Remote LLM JSON completion failed") from last_error
