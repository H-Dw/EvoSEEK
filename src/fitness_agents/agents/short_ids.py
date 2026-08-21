"""Request-scoped model identifier projection and recovery.

Canonical identifiers stay in local runtime state. A model sees only compact,
opaque aliases that are valid for one logical structured-completion transaction.
The same bridge must be reused by reasoning, render, repair and transport retry
attempts belonging to that transaction.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Literal

IdResolutionMode = Literal["exact", "normalize", "unique_near"]


class LocalIdResolutionError(ValueError):
    """Base class for a request-local identifier that cannot be resolved safely."""


class UnresolvedLocalIdError(LocalIdResolutionError):
    """No allowed request-local identifier matches the returned value."""


class AmbiguousLocalIdError(LocalIdResolutionError):
    """More than one allowed identifier could match the returned value."""


class LocalIdNamespaceError(LocalIdResolutionError):
    """A returned identifier belongs to another request-local namespace."""


@dataclass(frozen=True)
class IdResolutionReceipt:
    scope_id: str
    field_path: str
    namespace: str
    raw_token: str
    normalized_token: str
    resolved_alias: str
    strategy: IdResolutionMode
    edit_distance: int
    candidate_count: int
    corrected: bool


@dataclass(frozen=True)
class FieldIdPolicy:
    """Resolution policy for one schema path such as ``reflections[].variant_id``."""

    namespace: str
    mode: IdResolutionMode = "exact"


def _edit_distance_at_most_one(left: str, right: str) -> int | None:
    """Return edit distance 0/1, or ``None`` when it is greater than one."""

    if left == right:
        return 0
    if abs(len(left) - len(right)) > 1:
        return None
    if len(left) == len(right):
        mismatches = sum(a != b for a, b in zip(left, right, strict=True))
        return 1 if mismatches == 1 else None
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    short_index = long_index = 0
    skipped = False
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        if skipped:
            return None
        skipped = True
        long_index += 1
    return 1


@dataclass(frozen=True)
class ShortIdMap:
    """Bidirectional request-local alias map for one typed namespace."""

    alias_to_value: dict[str, str]
    prefix: str = ""

    def __post_init__(self) -> None:
        aliases = tuple(self.alias_to_value)
        values = tuple(self.alias_to_value.values())
        if len(aliases) != len(set(aliases)) or len(values) != len(set(values)):
            raise ValueError("ShortIdMap must be one-to-one")
        if any(not alias or not value for alias, value in self.alias_to_value.items()):
            raise ValueError("ShortIdMap identifiers must not be empty")
        if self.prefix and any(not alias.startswith(self.prefix) for alias in aliases):
            raise ValueError("ShortIdMap aliases must use the configured prefix")

    @classmethod
    def build(
        cls,
        values: list[str] | tuple[str, ...],
        *,
        prefix: str,
        minimum_width: int = 2,
    ) -> ShortIdMap:
        ordered = tuple(dict.fromkeys(str(value) for value in values if str(value)))
        normalized_prefix = str(prefix).strip().upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9]*", normalized_prefix):
            raise ValueError("ShortIdMap prefix must be an uppercase identifier namespace")
        width = max(minimum_width, len(str(len(ordered))))
        return cls(
            {
                f"{normalized_prefix}{index:0{width}d}": value
                for index, value in enumerate(ordered, start=1)
            },
            prefix=normalized_prefix,
        )

    @property
    def value_to_alias(self) -> dict[str, str]:
        return {value: alias for alias, value in self.alias_to_value.items()}

    def encode(self, value: str, *, strict: bool = False) -> str:
        raw = str(value)
        alias = self.value_to_alias.get(raw)
        if alias is not None:
            return alias
        if strict:
            raise UnresolvedLocalIdError(
                f"Canonical identifier is not present in this request namespace: {raw!r}"
            )
        return raw

    def decode(self, value: str, *, strict: bool = False) -> str:
        raw = str(value)
        canonical = self.alias_to_value.get(raw)
        if canonical is not None:
            return canonical
        if strict:
            raise UnresolvedLocalIdError(
                f"Unknown request-local identifier for namespace {self.prefix!r}: {raw!r}"
            )
        return raw

    def prose_alias_pattern(self) -> re.Pattern[str]:
        """Match padded request-local aliases such as ``E01`` or ``E33``.

        Canonical IDs like ``E1:kg:sha256:...`` use a single digit plus ``:`` and
        are intentionally excluded.
        """

        return re.compile(rf"\b{re.escape(self.prefix)}\d{{2,}}\b")

    def unknown_aliases_in_text(self, text: str) -> tuple[str, ...]:
        """Return padded aliases in prose that are not in this request map."""

        found = self.prose_alias_pattern().findall(str(text))
        return tuple(
            dict.fromkeys(
                alias for alias in found if alias not in self.alias_to_value
            )
        )

    def strip_unknown_aliases_in_text(self, text: str) -> str:
        """Drop padded aliases that are not in this request map.

        Prior-round ``E33`` labels have no identity here. Current-map aliases
        such as ``E01`` are kept.
        """

        def _replace(match: re.Match[str]) -> str:
            token = match.group(0)
            return token if token in self.alias_to_value else ""

        cleaned = self.prose_alias_pattern().sub(_replace, str(text))
        cleaned = re.sub(r"\s*[/,;]\s*(?=[/,;]|$)", " ", cleaned)
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\s+\.", ".", cleaned)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned.strip(" ,;/")

    def expand_aliases_in_text(self, text: str) -> str:
        """Replace whole-token aliases inside prose with canonical identifiers.

        ``decode`` only matches an entire string. Interpretation sentences can
        embed request-local labels such as ``S01`` next to mutation tokens, and
        those must be expanded with the same batch map used for typed ID fields.
        Longer aliases are substituted first so ``S10`` is not eaten by ``S1``.
        """

        output = str(text)
        for alias, canonical in sorted(
            self.alias_to_value.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if alias == canonical:
                continue
            output = re.sub(rf"\b{re.escape(alias)}\b", canonical, output)
        return output

    def collapse_canonicals_in_text(self, text: str) -> str:
        """Replace canonical identifiers inside prose with this request's aliases.

        Typed ID fields are rewritten as whole strings. Finding statements still
        embed canonical sample or evidence IDs after per-batch expansion, and
        those must collapse with the same map the Critic uses for arrays.
        Longer canonicals are substituted first.
        """

        output = str(text)
        for alias, canonical in sorted(
            self.alias_to_value.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        ):
            if alias == canonical:
                continue
            output = re.sub(
                rf"(?<![A-Za-z0-9]){re.escape(canonical)}(?![A-Za-z0-9])",
                alias,
                output,
            )
        return output

    def prompt_map(self, labels: dict[str, str] | None = None) -> dict[str, str]:
        """Return alias-to-readable-label rows without exposing canonical IDs."""

        labels = labels or {}
        return {
            alias: labels.get(value, alias)
            for alias, value in self.alias_to_value.items()
        }

    def normalize_alias(self, value: str) -> str:
        """Apply deterministic formatting repair without changing identity."""

        normalized = unicodedata.normalize("NFKC", str(value)).strip()
        while (
            len(normalized) >= 2
            and normalized[0] in "'\"`"
            and normalized[-1] == normalized[0]
        ):
            normalized = normalized[1:-1].strip()
        normalized = normalized.upper()
        if self.prefix:
            match = re.fullmatch(
                rf"{re.escape(self.prefix)}[-_ ]?(\d+)", normalized
            )
            if match:
                widths = {len(alias) - len(self.prefix) for alias in self.alias_to_value}
                if len(widths) == 1:
                    normalized = f"{self.prefix}{int(match.group(1)):0{next(iter(widths))}d}"
        return normalized

    def resolve(
        self,
        value: str,
        *,
        scope_id: str,
        field_path: str,
        namespace: str,
        mode: IdResolutionMode,
    ) -> tuple[str, IdResolutionReceipt]:
        raw = str(value)
        if raw in self.alias_to_value:
            return self.alias_to_value[raw], IdResolutionReceipt(
                scope_id=scope_id,
                field_path=field_path,
                namespace=namespace,
                raw_token=raw,
                normalized_token=raw,
                resolved_alias=raw,
                strategy="exact",
                edit_distance=0,
                candidate_count=1,
                corrected=False,
            )
        normalized = self.normalize_alias(raw)
        if mode in {"normalize", "unique_near"} and normalized in self.alias_to_value:
            return self.alias_to_value[normalized], IdResolutionReceipt(
                scope_id=scope_id,
                field_path=field_path,
                namespace=namespace,
                raw_token=raw,
                normalized_token=normalized,
                resolved_alias=normalized,
                strategy="normalize",
                edit_distance=0,
                candidate_count=1,
                corrected=True,
            )
        if mode == "unique_near":
            candidates = [
                alias
                for alias in self.alias_to_value
                if alias.startswith(self.prefix)
                and _edit_distance_at_most_one(normalized, alias) == 1
            ]
            if len(candidates) == 1:
                alias = candidates[0]
                return self.alias_to_value[alias], IdResolutionReceipt(
                    scope_id=scope_id,
                    field_path=field_path,
                    namespace=namespace,
                    raw_token=raw,
                    normalized_token=normalized,
                    resolved_alias=alias,
                    strategy="unique_near",
                    edit_distance=1,
                    candidate_count=1,
                    corrected=True,
                )
            if len(candidates) > 1:
                raise AmbiguousLocalIdError(
                    f"Ambiguous request-local identifier {raw!r} at {field_path}; "
                    f"candidates={sorted(candidates)}"
                )
        raise UnresolvedLocalIdError(
            f"Unknown request-local identifier {raw!r} at {field_path} "
            f"for namespace {namespace!r}"
        )


def _format_path(path: tuple[str, ...]) -> str:
    output = ""
    for part in path:
        if part == "[]":
            output += "[]"
        else:
            output += ("." if output else "") + part
    return output


def _walk_id_fields(value: Any, *, path: tuple[str, ...], transform: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _walk_id_fields(item, path=(*path, str(key)), transform=transform)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _walk_id_fields(item, path=(*path, "[]"), transform=transform)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _walk_id_fields(item, path=(*path, "[]"), transform=transform)
            for item in value
        )
    return transform(value, path)


@dataclass
class RequestScopedIdBridge:
    """Typed ID bridge owned by one logical LLM completion transaction."""

    scope_id: str
    role: str
    schema_name: str
    namespaces: dict[str, ShortIdMap]
    field_policies: dict[str, FieldIdPolicy]
    resolution_receipts: list[IdResolutionReceipt] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        *,
        scope_id: str,
        role: str,
        schema_name: str,
        namespace_values: dict[str, list[str] | tuple[str, ...]],
        field_policies: dict[str, FieldIdPolicy],
        minimum_width: int = 2,
    ) -> RequestScopedIdBridge:
        namespaces = {
            namespace: ShortIdMap.build(
                values, prefix=namespace, minimum_width=minimum_width
            )
            for namespace, values in namespace_values.items()
        }
        missing = {
            policy.namespace
            for policy in field_policies.values()
            if policy.namespace not in namespaces
        }
        if missing:
            raise ValueError(f"ID field policies reference missing namespaces: {sorted(missing)}")
        return cls(
            scope_id=str(scope_id),
            role=str(role),
            schema_name=str(schema_name),
            namespaces=namespaces,
            field_policies=dict(field_policies),
        )

    def encode_projection(self, payload: Any) -> Any:
        def transform(value: Any, path: tuple[str, ...]) -> Any:
            policy = self.field_policies.get(_format_path(path))
            if policy is None or value is None:
                return value
            if not isinstance(value, str):
                raise TypeError(f"ID field {_format_path(path)} must contain a string")
            return self.namespaces[policy.namespace].encode(value, strict=True)

        return _walk_id_fields(payload, path=(), transform=transform)

    def decode_and_validate(self, payload: Any, *, record_receipts: bool = True) -> Any:
        receipts: list[IdResolutionReceipt] = []

        def transform(value: Any, path: tuple[str, ...]) -> Any:
            field_path = _format_path(path)
            policy = self.field_policies.get(field_path)
            if policy is None or value is None:
                return value
            if not isinstance(value, str):
                raise TypeError(f"ID field {field_path} must contain a string")
            for other_namespace, other_map in self.namespaces.items():
                if other_namespace == policy.namespace:
                    continue
                if value in other_map.alias_to_value:
                    raise LocalIdNamespaceError(
                        f"Identifier {value!r} at {field_path} belongs to namespace "
                        f"{other_namespace!r}, expected {policy.namespace!r}"
                    )
            canonical, receipt = self.namespaces[policy.namespace].resolve(
                value,
                scope_id=self.scope_id,
                field_path=field_path,
                namespace=policy.namespace,
                mode=policy.mode,
            )
            receipts.append(receipt)
            return canonical

        decoded = _walk_id_fields(payload, path=(), transform=transform)
        if record_receipts:
            self.resolution_receipts.extend(receipts)
        return decoded

    def prompt_maps(self, labels: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
        labels = labels or {}
        return {
            namespace: mapping.prompt_map(labels.get(namespace))
            for namespace, mapping in self.namespaces.items()
        }

    def audit_payload(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "role": self.role,
            "schema_name": self.schema_name,
            "namespaces": {
                namespace: dict(mapping.alias_to_value)
                for namespace, mapping in self.namespaces.items()
            },
            "field_policies": {
                path: {"namespace": policy.namespace, "mode": policy.mode}
                for path, policy in self.field_policies.items()
            },
            "resolution_receipts": [receipt.__dict__ for receipt in self.resolution_receipts],
        }


def rewrite_exact_ids(value: Any, *maps: ShortIdMap, decode: bool = False) -> Any:
    """Legacy exact rewrite helper retained while role callers are migrated."""

    def rewrite(item: str) -> str:
        output = item
        for mapping in maps:
            output = mapping.decode(output) if decode else mapping.encode(output)
        return output

    if isinstance(value, dict):
        return {
            rewrite(str(key)): rewrite_exact_ids(item, *maps, decode=decode)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [rewrite_exact_ids(item, *maps, decode=decode) for item in value]
    if isinstance(value, tuple):
        return tuple(rewrite_exact_ids(item, *maps, decode=decode) for item in value)
    if isinstance(value, str):
        return rewrite(value)
    return value


__all__ = [
    "AmbiguousLocalIdError",
    "FieldIdPolicy",
    "IdResolutionReceipt",
    "LocalIdNamespaceError",
    "LocalIdResolutionError",
    "RequestScopedIdBridge",
    "ShortIdMap",
    "UnresolvedLocalIdError",
    "rewrite_exact_ids",
]
