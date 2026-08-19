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
MAX_OUTPUT_TOKENS = 20000
ALLOWED_EVIDENCE_RETRY_CAP = 24
FAILURE_KINDS = (
    "truncated",
    "syntax",
    "schema",
    "semantic",
    "unknown_evidence",
    "empty",
    "other",
)
FailureKind = Literal[
    "truncated",
    "syntax",
    "schema",
    "semantic",
    "unknown_evidence",
    "empty",
    "other",
]


class OutputTruncatedError(ValueError):
    """Visible completion ended before a complete JSON object was emitted."""


class EmptyLLMOutputError(ValueError):
    """Remote LLM returned no visible message content."""


class ContentFilteredError(RuntimeError):
    """Provider blocked the visible completion; retrying the same request is unsafe."""


class UnexpectedFinishReasonError(RuntimeError):
    """The provider ended in a state incompatible with a JSON-only role call."""


class ProviderResourceError(TimeoutError):
    """Provider declared a transient resource shortage."""


class PromptBudgetExceededError(RuntimeError):
    """Projected prompt exceeded the configured preflight character budget."""


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
        self.paths = ("evidence_ids",)
        self.allowed_values = {"evidence_ids[]": self.allowed}
        self.stripped_payload = stripped_payload


class SemanticOutputValidationError(ValueError):
    """Typed JSON passed its schema but failed a runtime/domain invariant."""

    def __init__(
        self,
        message: str,
        *,
        paths: tuple[str, ...] = (),
        allowed_values: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        super().__init__(message)
        self.paths = paths
        self.allowed_values = allowed_values or {}


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

    def apply(
        self,
        failure: OutputFailure,
        *,
        deepseek: bool,
        preserve_thinking: bool = False,
    ) -> None:
        if failure.kind != "truncated":
            return
        if deepseek and not preserve_thinking:
            self.thinking = "disabled"
            # DeepSeek maps low/medium reasoning effort back to high.  Once a
            # completion is truncated, disable thinking and omit the effort
            # field altogether so the compact retry cannot silently re-enable
            # an expensive reasoning path.
            self.effort = None
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


def json_salvage(text: str) -> dict[str, Any] | None:
    """Repair cosmetic JSON only; never extract prose or synthesize closure."""

    payload = strip_markdown_fences(text)
    if not payload:
        return None
    stripped_commas = re.sub(r",(\s*[}\]])", r"\1", payload)

    if not _braces_balanced(stripped_commas):
        return None

    def _load(candidate: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    parsed = _load(stripped_commas)
    return parsed


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
    elif isinstance(error, SemanticOutputValidationError):
        kind = "semantic"
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
    schema: dict[str, Any] | None = None,
    repair_hints: dict[str, tuple[str, ...] | list[str]] | None = None,
    allowed_evidence_ids: tuple[str, ...] = (),
) -> str:
    parts = [
        f"The previous JSON failed ({failure.kind}): {failure.message[:800]}.",
        "Return exactly one balanced, complete JSON object and no Markdown.",
    ]
    if failure.finish_reason:
        parts.append(f"finish_reason={failure.finish_reason}.")
    if failure.decode_position is not None:
        parts.append(f"JSON parse failed at character {failure.decode_position}.")
    parts.append(f"content_chars={failure.content_length}.")
    if failure.kind == "truncated":
        parts.append(
            "The previous completion was truncated; do not copy it or guess its missing suffix. "
            "Emit a fresh compact object."
        )
    else:
        parts.append("Modify only the fields named by the validation errors.")
        details = validation_error_entries(error)
        if details:
            parts.append(
                "Validation errors (field path/type/message): "
                + json.dumps(details, ensure_ascii=False, separators=(",", ":"))
                + "."
            )
    unknown = getattr(error, "unknown", ())
    allowed = getattr(error, "allowed", allowed_evidence_ids)
    if unknown:
        parts.append(f"Drop these non-visible evidence_ids: {list(unknown)}.")
    if allowed:
        parts.append(f"Cite only these evidence_id values: {list(allowed)}.")
        parts.append("If none apply, return evidence_ids=[]. Never invent ev: identifiers.")
    constraints = schema_constraints(schema or {})
    explicit = dict(repair_hints or {})
    explicit.update(getattr(error, "allowed_values", {}) or {})
    for path, values in sorted(explicit.items()):
        compact = tuple(str(item) for item in values)[:32]
        if compact:
            constraints[path] = {"allowed": list(compact)}
    if constraints:
        parts.append(
            "Current field constraints: "
            + json.dumps(constraints, ensure_ascii=False, separators=(",", ":"))
            + "."
        )
    parts.append(
        "Do not change verdict, action, or identifier fields unless their exact path is "
        "listed as invalid."
    )
    return " ".join(parts)


def validation_error_entries(error: Exception) -> list[dict[str, str]]:
    """Return bounded Pydantic-style errors without echoing invalid inputs."""

    errors = getattr(error, "errors", None)
    if callable(errors):
        try:
            raw_entries = errors(include_input=False, include_url=False)
        except TypeError:
            raw_entries = errors()
        return [
            {
                "path": ".".join(str(part) for part in item.get("loc", ())) or "$",
                "type": str(item.get("type") or "validation_error"),
                "message": str(item.get("msg") or "invalid value")[:240],
            }
            for item in raw_entries[:16]
        ]
    paths = getattr(error, "paths", ())
    return [
        {"path": str(path), "type": type(error).__name__, "message": str(error)[:240]}
        for path in paths[:16]
    ]


def schema_constraints(schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract compact enum/length constraints from a generated JSON Schema."""

    definitions = schema.get("$defs", {}) if isinstance(schema, dict) else {}
    output: dict[str, dict[str, Any]] = {}

    def walk(node: Any, path: str, seen: frozenset[str] = frozenset()) -> None:
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            if name not in seen:
                walk(definitions.get(name), path, seen.union({name}))
            return
        item: dict[str, Any] = {}
        if isinstance(node.get("enum"), list):
            item["allowed"] = node["enum"][:32]
        if "const" in node:
            item["allowed"] = [node["const"]]
        for key in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
            if key in node:
                item[key] = node[key]
        if item and path:
            output[path] = item
        for key, child in (node.get("properties") or {}).items():
            child_path = f"{path}.{key}" if path else str(key)
            walk(child, child_path, seen)
        if "items" in node:
            walk(node["items"], f"{path}[]", seen)
        for child in node.get("anyOf", ()) or node.get("oneOf", ()) or ():
            walk(child, path, seen)

    walk(schema, "")
    return dict(sorted(output.items())[:64])


def validation_detail(error: Exception) -> str:
    entries = validation_error_entries(error)
    if entries:
        return json.dumps(entries, ensure_ascii=False)
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
    required_control_count: int | None = None
    minimum_batch_distance: int | None = None
    add_exploration: bool = False
    reduce_mutation_depth: bool = False
    regenerate_hypothesis: bool = False

    def merge(self, other: RevisionConstraints) -> RevisionConstraints:
        return RevisionConstraints(
            require_controls=self.require_controls or other.require_controls,
            increase_diversity=self.increase_diversity or other.increase_diversity,
            required_control_count=max(
                (
                    item
                    for item in (
                        self.required_control_count,
                        other.required_control_count,
                    )
                    if item is not None
                ),
                default=None,
            ),
            minimum_batch_distance=max(
                (
                    item
                    for item in (
                        self.minimum_batch_distance,
                        other.minimum_batch_distance,
                    )
                    if item is not None
                ),
                default=None,
            ),
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
