"""Provider-neutral structured completion over project-owned Chat Completions clients."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .output_guards import SemanticOutputValidationError, UnknownEvidenceIdsError
from .remote_llm import complete_json
from .transports import ChatTransport

OutputT = TypeVar("OutputT", bound=BaseModel)


def _render_constraint_summary(schema: dict[str, Any]) -> str:
    """Expose model-relevant size constraints without maintaining a second schema."""

    constraints: list[str] = []

    def walk(node: Any, path: str, root: dict[str, Any]) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            node = root.get("$defs", {}).get(ref.rsplit("/", 1)[-1], node)
        for name, child in (node.get("properties") or {}).items():
            child_path = f"{path}.{name}" if path else str(name)
            if isinstance(child, dict):
                if "maxLength" in child:
                    constraints.append(f"{child_path}: max {child['maxLength']} characters")
                if "maxItems" in child:
                    constraints.append(f"{child_path}: max {child['maxItems']} items")
                walk(child, child_path, root)
        items = node.get("items")
        if isinstance(items, dict):
            walk(items, f"{path}[]" if path else "[]", root)
        for key in ("anyOf", "oneOf", "allOf"):
            for child in node.get(key, ()) if isinstance(node.get(key), list) else ():
                walk(child, path, root)

    walk(schema, "", schema)
    return "; ".join(dict.fromkeys(constraints)) or "Follow every schema constraint exactly."


def _semantic_paths(error: Exception) -> tuple[str, ...]:
    message = str(error).lower()
    paths: list[str] = []
    for marker, path in (
        ("hypothesis id", "hypothesis_id"),
        ("parent", "parent_hypothesis_id"),
        ("evidence", "evidence_ids"),
        ("position", "preferred_residues"),
        ("residue", "preferred_residues"),
        ("variant", "variant_id"),
        ("channel", "channel"),
    ):
        if marker in message:
            paths.append(path)
    return tuple(dict.fromkeys(paths)) or ("runtime_invariant",)


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
    transport_retries: int | None = None,
    truncation_retries: int | None = None,
    syntax_retries: int | None = None,
    schema_retries: int | None = None,
    semantic_retries: int | None = None,
    unknown_evidence_retries: int | None = None,
    empty_retries: int | None = None,
    retry_backoff_seconds: float = 0.0,
    allow_unknown_evidence_stripping: bool = False,
    max_input_chars: int | None = None,
    separate_json_render: bool = False,
    repair_hints: dict[str, tuple[str, ...] | list[str]] | None = None,
    trace_context: dict[str, Any] | None = None,
    reasoning_truncation_retries: int | None = None,
    preserve_reasoning_on_retry: bool = True,
) -> OutputT:
    """Parse, validate and context-check inside one bounded retry boundary."""

    def validate(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = output_type.model_validate(payload).model_dump(mode="json", by_alias=True)
        if contextual_validator is None:
            return normalized
        try:
            return contextual_validator(normalized)
        except (ValidationError, UnknownEvidenceIdsError, SemanticOutputValidationError):
            raise
        except (TypeError, ValueError) as error:
            raise SemanticOutputValidationError(
                str(error), paths=_semantic_paths(error)
            ) from error

    generated_schema = output_type.model_json_schema()
    effective_messages = messages
    effective_thinking = thinking
    effective_effort = reasoning_effort
    reasoning_then_render = separate_json_render and thinking != "disabled"
    if reasoning_then_render:
        draft = complete_json(
            client=client,
            transport=transport,
            model=model,
            messages=messages,
            schema={"type": "object"},
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking="enabled",
            retries=retries,
            transport_retries=transport_retries,
            truncation_retries=(
                truncation_retries
                if reasoning_truncation_retries is None
                else reasoning_truncation_retries
            ),
            syntax_retries=syntax_retries,
            schema_retries=0,
            semantic_retries=0,
            unknown_evidence_retries=0,
            empty_retries=empty_retries,
            retry_backoff_seconds=retry_backoff_seconds,
            max_input_chars=max_input_chars,
            trace_context={**(trace_context or {}), "completion_stage": "reasoning_draft"},
            preserve_thinking_on_retry=preserve_reasoning_on_retry,
        )
        try:
            normalized_draft = validate(draft)
        except (ValidationError, UnknownEvidenceIdsError, SemanticOutputValidationError):
            normalized_draft = None
        if normalized_draft is not None:
            return output_type.model_validate(normalized_draft)
        effective_messages = [
            *(item for item in messages if item.get("role") == "system"),
            {
                "role": "assistant",
                "content": json.dumps(draft, ensure_ascii=False, separators=(",", ":")),
            },
            {
                "role": "user",
                "content": (
                    "Render the preceding draft as exactly one JSON object matching this "
                    "Pydantic-generated schema. Preserve scientific meaning, verdict/action values, "
                    "and every identifier. You MAY compress only free-text prose so it fits the "
                    "generated length limits; do not truncate mechanically or add claims. Check "
                    "these generated high-risk constraints before replying: "
                    + _render_constraint_summary(generated_schema)
                    + ". Return no commentary. Schema: "
                    + json.dumps(generated_schema, ensure_ascii=False, separators=(",", ":"))
                ),
            },
        ]
        effective_thinking = "disabled"
        effective_effort = None

    payload = complete_json(
        client=client,
        transport=transport,
        model=model,
        messages=effective_messages,
        schema=generated_schema,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=effective_effort,
        thinking=effective_thinking,
        retries=retries,
        transport_retries=transport_retries,
        truncation_retries=truncation_retries,
        syntax_retries=syntax_retries,
        schema_retries=schema_retries,
        semantic_retries=semantic_retries,
        unknown_evidence_retries=unknown_evidence_retries,
        empty_retries=empty_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        allow_unknown_evidence_stripping=allow_unknown_evidence_stripping,
        max_input_chars=max_input_chars,
        validator=validate,
        repair_hints=repair_hints,
        trace_context={
            **(trace_context or {}),
            "completion_stage": "json_render" if reasoning_then_render else "single",
        },
    )
    return output_type.model_validate(payload)
