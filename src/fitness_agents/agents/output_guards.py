"""LangChain-style output guards without a LangChain dependency.

RetryOutputParser / OutputFixingParser / token-budget behaviour is implemented
against the project Chat Completions client. Salvage never invents scientific fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
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


CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")
REVISION_ARMS = frozenset(
    {
        "hypothesis_target",
        "evidence_prior",
        "coverage_exploration",
        "matched_control",
        "fallback",
    }
)
DEFAULT_RESIDUE_CONSTRAINT_ARMS = frozenset(
    {
        "hypothesis_target",
        "evidence_prior",
        "coverage_exploration",
        "fallback",
    }
)


@dataclass(frozen=True, order=True)
class ResidueSubstitutionConstraint:
    position: int
    to_residue: str
    from_residue: str | None = None

    def __post_init__(self) -> None:
        if self.position <= 0:
            raise ValueError("substitution position must be positive")
        if self.to_residue not in CANONICAL_RESIDUES:
            raise ValueError("substitution to_residue must be canonical")
        if self.from_residue is not None and self.from_residue not in CANONICAL_RESIDUES:
            raise ValueError("substitution from_residue must be canonical")

    @property
    def token(self) -> str:
        prefix = self.from_residue or ""
        return f"{prefix}{self.position}{self.to_residue}"


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
    excluded_substitutions: tuple[ResidueSubstitutionConstraint, ...] = ()
    required_residues_by_position: dict[int, tuple[str, ...]] = field(
        default_factory=dict
    )
    applies_to_arms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown_arms = set(self.applies_to_arms).difference(REVISION_ARMS)
        if unknown_arms:
            raise ValueError(f"unknown revision arms: {sorted(unknown_arms)}")
        for position, residues in self.required_residues_by_position.items():
            if position <= 0:
                raise ValueError("required residue constraint positions must be positive")
            invalid = set(residues).difference(CANONICAL_RESIDUES)
            if invalid:
                raise ValueError(
                    f"required residue constraints contain non-canonical residues: {sorted(invalid)}"
                )

    @property
    def has_residue_constraints(self) -> bool:
        return bool(self.excluded_substitutions or self.required_residues_by_position)

    def applies_to(self, arm: str) -> bool:
        if not self.has_residue_constraints:
            return False
        scoped = set(self.applies_to_arms) or set(DEFAULT_RESIDUE_CONSTRAINT_ARMS)
        return arm in scoped

    def variant_violations(
        self,
        variant: Any,
        *,
        arm: str,
        position_to_index: dict[int, int],
        wild_type_by_position: dict[int, str] | None = None,
    ) -> tuple[str, ...]:
        if not self.applies_to(arm):
            return ()
        sequence = str(getattr(variant, "variant", ""))
        violations: list[str] = []
        for rule in self.excluded_substitutions:
            index = position_to_index.get(rule.position)
            if index is None or index >= len(sequence):
                violations.append(f"unmapped:{rule.token}")
                continue
            if sequence[index] != rule.to_residue:
                continue
            if (
                rule.from_residue is not None
                and wild_type_by_position is not None
                and wild_type_by_position.get(rule.position) != rule.from_residue
            ):
                continue
            violations.append(f"excluded:{rule.token}")
        for position, allowed in self.required_residues_by_position.items():
            index = position_to_index.get(position)
            if index is None or index >= len(sequence):
                violations.append(f"unmapped:{position}")
            elif sequence[index] not in allowed:
                violations.append(
                    f"required:{position}:{','.join(sorted(allowed))}"
                )
        return tuple(violations)

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "excluded_substitutions": [
                {
                    "position": item.position,
                    "from_residue": item.from_residue,
                    "to_residue": item.to_residue,
                }
                for item in self.excluded_substitutions
            ],
            "required_residues_by_position": {
                str(position): list(residues)
                for position, residues in sorted(
                    self.required_residues_by_position.items()
                )
            },
            "applies_to_arms": list(self.applies_to_arms),
        }

    def merge(self, other: RevisionConstraints) -> RevisionConstraints:
        required: dict[int, tuple[str, ...]] = {}
        for position in set(self.required_residues_by_position).union(
            other.required_residues_by_position
        ):
            left = self.required_residues_by_position.get(position)
            right = other.required_residues_by_position.get(position)
            if left is not None and right is not None:
                required[position] = tuple(sorted(set(left).intersection(right)))
            else:
                required[position] = tuple(left or right or ())
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
            excluded_substitutions=tuple(
                sorted(
                    set(self.excluded_substitutions).union(
                        other.excluded_substitutions
                    )
                )
            ),
            required_residues_by_position=required,
            applies_to_arms=tuple(
                sorted(set(self.applies_to_arms).union(other.applies_to_arms))
            ),
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
