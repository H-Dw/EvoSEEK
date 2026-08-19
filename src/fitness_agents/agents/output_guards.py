"""LangChain-style output guards without a LangChain dependency.

RetryOutputParser / OutputFixingParser / token-budget behaviour is implemented
against the project Chat Completions client. Salvage never invents scientific fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens", "max_output_tokens"})
MAX_OUTPUT_TOKENS = 32768
ALLOWED_EVIDENCE_RETRY_CAP = 24
FAILURE_KINDS = ("truncated", "syntax", "schema", "unknown_evidence", "empty", "other")
FailureKind = Literal["truncated", "syntax", "schema", "unknown_evidence", "empty", "other"]


class OutputTruncatedError(ValueError):
    """Visible completion ended before a complete JSON object was emitted."""


class EmptyLLMOutputError(ValueError):
    """Remote LLM returned no visible message content."""


class UnknownEvidenceIdsError(ValueError):
    """Hypothesis cited evidence IDs that are not visible to the role."""

    def __init__(
        self,
        unknown: list[str],
        allowed: frozenset[str] | None,
        *,
        stripped_payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"evidence_ids contains identifiers not visible to the role: {sorted(unknown)}"
        )
        self.unknown = tuple(sorted(unknown))
        self.allowed = tuple(sorted(allowed or ())[:ALLOWED_EVIDENCE_RETRY_CAP])
        self.stripped_payload = stripped_payload


@dataclass(frozen=True)
class OutputFailure:
    kind: FailureKind
    message: str
    finish_reason: str | None = None
    content_length: int = 0
    braces_balanced: bool = True
    decode_position: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None


@dataclass
class TokenBudgetPolicy:
    budget: int
    thinking: str | None
    effort: str | None = None
    max_budget: int = MAX_OUTPUT_TOKENS

    def apply(self, failure: OutputFailure, *, deepseek: bool) -> None:
        if failure.kind != "truncated":
            return
        if deepseek:
            self.thinking = "disabled"
            self.effort = "low"
        raised = max(self.budget + 4096, int(self.budget * 1.5))
        self.budget = min(self.max_budget, raised)


def strip_markdown_fences(text: str) -> str:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload, count=1, flags=re.IGNORECASE)
        payload = re.sub(r"\s*```$", "", payload)
    return payload.strip()


def _braces_balanced(text: str) -> bool:
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False
    for char in text:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth_obj += 1
        elif char == "}":
            depth_obj -= 1
            if depth_obj < 0:
                return False
        elif char == "[":
            depth_arr += 1
        elif char == "]":
            depth_arr -= 1
            if depth_arr < 0:
                return False
    return depth_obj == 0 and depth_arr == 0 and not in_string


def _close_truncated_json(fragment: str) -> str | None:
    depth_obj = 0
    depth_arr = 0
    in_string = False
    escape = False
    for char in fragment:
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth_obj += 1
        elif char == "}":
            depth_obj -= 1
        elif char == "[":
            depth_arr += 1
        elif char == "]":
            depth_arr -= 1
    if in_string or depth_obj < 0 or depth_arr < 0:
        return None
    return fragment + ("]" * depth_arr) + ("}" * depth_obj)


def json_salvage(text: str) -> dict[str, Any] | None:
    """Conservative repair: fences, trailing commas, and truncated close-braces only."""

    payload = strip_markdown_fences(text)
    if not payload:
        return None
    stripped_commas = re.sub(r",(\s*[}\]])", r"\1", payload)

    def _load(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start < 0 or end <= start:
                return None
            try:
                parsed = json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, dict) else None

    parsed = _load(stripped_commas)
    if parsed is not None:
        return parsed
    start = stripped_commas.find("{")
    if start < 0:
        return None
    closed = _close_truncated_json(stripped_commas[start:])
    if closed is None:
        return None
    return _load(closed)


def classify_output_failure(
    error: Exception,
    *,
    finish_reason: str | None,
    content: str,
    usage: dict[str, Any] | None = None,
) -> OutputFailure:
    usage = usage or {}
    content = content or ""
    decode_position = getattr(error, "pos", None)
    if isinstance(error, json.JSONDecodeError):
        decode_position = error.pos
    balanced = _braces_balanced(strip_markdown_fences(content)) if content else True
    truncated = (
        str(finish_reason or "").lower() in TRUNCATION_FINISH_REASONS
        or isinstance(error, OutputTruncatedError)
        or (
            isinstance(error, json.JSONDecodeError)
            and content
            and error.pos >= max(0, len(content) - 2)
        )
        or (content.startswith("{") and not balanced)
    )
    if isinstance(error, EmptyLLMOutputError) or (
        isinstance(error, ValueError) and "empty message content" in str(error).lower()
    ):
        kind: FailureKind = "empty"
    elif isinstance(error, UnknownEvidenceIdsError):
        kind = "unknown_evidence"
    elif truncated:
        kind = "truncated"
    elif isinstance(error, json.JSONDecodeError) or "JSON object" in str(error):
        kind = "syntax"
    elif type(error).__name__ in {"ValidationError", "ValueError"}:
        kind = "schema"
    else:
        kind = "other"
    return OutputFailure(
        kind=kind,
        message=str(error) or type(error).__name__,
        finish_reason=finish_reason,
        content_length=len(content),
        braces_balanced=balanced,
        decode_position=decode_position if isinstance(decode_position, int) else None,
        completion_tokens=usage.get("completion_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
    )


def retry_instruction(
    failure: OutputFailure,
    *,
    error: Exception,
    allowed_evidence_ids: tuple[str, ...] = (),
) -> str:
    parts = [
        f"The previous JSON failed ({failure.kind}): {failure.message[:800]}.",
        "Return one complete JSON object with every required key and no Markdown.",
        "Keep statement/summary/expected_outcome/falsification_criterion at or under 400 characters.",
    ]
    if failure.finish_reason:
        parts.append(f"finish_reason={failure.finish_reason}.")
    if failure.decode_position is not None:
        parts.append(f"JSON parse failed at character {failure.decode_position}.")
    parts.append(f"content_chars={failure.content_length}.")
    if failure.kind == "truncated":
        parts.append(
            "The previous completion was truncated. Disable hidden reasoning and emit compact JSON only."
        )
    unknown = getattr(error, "unknown", ())
    allowed = getattr(error, "allowed", allowed_evidence_ids)
    if unknown:
        parts.append(f"Drop these non-visible evidence_ids: {list(unknown)}.")
    if allowed:
        parts.append(f"Cite only these evidence_id values: {list(allowed)}.")
        parts.append("If none apply, return evidence_ids=[]. Never invent ev: identifiers.")
    return " ".join(parts)


def validation_detail(error: Exception) -> str:
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
    if isinstance(error, json.JSONDecodeError):
        return (
            f"JSONDecodeError: {error.msg} at column {error.pos} "
            f"(doc length {len(error.doc) if error.doc else 0})"
        )
    text = str(error).strip()
    if text:
        return f"{type(error).__name__}: {text[:800]}"
    return type(error).__name__


@dataclass
class RevisionConstraints:
    """Machine-executable batch constraints derived from a critic REVISE decision."""

    require_controls: bool = False
    increase_diversity: bool = False
    add_exploration: bool = False
    reduce_mutation_depth: bool = False
    regenerate_hypothesis: bool = False

    def merge(self, other: "RevisionConstraints") -> "RevisionConstraints":
        return RevisionConstraints(
            require_controls=self.require_controls or other.require_controls,
            increase_diversity=self.increase_diversity or other.increase_diversity,
            add_exploration=self.add_exploration or other.add_exploration,
            reduce_mutation_depth=self.reduce_mutation_depth or other.reduce_mutation_depth,
            regenerate_hypothesis=self.regenerate_hypothesis or other.regenerate_hypothesis,
        )


def critic_revision_payload(
    *,
    decision: Any,
    rejected_hypothesis: Any,
) -> dict[str, Any]:
    preferred = getattr(rejected_hypothesis, "preferred_residues", {}) or {}
    serialised = {
        str(site): list(residues) for site, residues in preferred.items()
    }
    changes = []
    for change in getattr(decision, "required_changes", ()) or ():
        action = getattr(change.action, "value", change.action)
        changes.append(
            {
                "action": action,
                "target_ids": list(getattr(change, "target_ids", ()) or ()),
                "rationale": getattr(change, "rationale", ""),
            }
        )
    return {
        "verdict": getattr(getattr(decision, "verdict", None), "value", "REVISE"),
        "summary": getattr(decision, "summary", "") or "",
        "required_changes": changes,
        "rejected_hypothesis_id": getattr(rejected_hypothesis, "hypothesis_id", None),
        "rejected_statement": getattr(rejected_hypothesis, "statement", None),
        "rejected_preferred_residues": serialised,
    }
