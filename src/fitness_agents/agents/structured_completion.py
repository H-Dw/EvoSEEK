"""Provider-neutral structured completion over project-owned Chat Completions clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel

from .remote_llm import complete_json
from .transports import ChatTransport

OutputT = TypeVar("OutputT", bound=BaseModel)


def complete_structured(
    *,
    client: Any | None,
    transport: ChatTransport | None = None,
    model: str,
    messages: list[dict[str, str]],
    output_type: type[OutputT],
    contextual_validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    thinking: str | None = None,
    retries: int = 2,
    trace_context: dict[str, Any] | None = None,
) -> OutputT:
    """Parse, validate and context-check inside one bounded retry boundary."""

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = output_type.model_validate(payload).model_dump(mode="json", by_alias=True)
        return contextual_validator(normalized) if contextual_validator else normalized

    payload = complete_json(
        client=client,
        transport=transport,
        model=model,
        messages=messages,
        schema=output_type.model_json_schema(),
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        retries=retries,
        validator=validate,
        trace_context=trace_context,
    )
    return output_type.model_validate(payload)
