"""Runtime-owned short identifier maps for model-facing payloads.

The campaign keeps canonical identifiers in local state.  Language models see
only compact aliases and local code expands returned aliases before semantic
validation.  This avoids asking a generative model to reproduce long opaque
identifiers exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ShortIdMap:
    """Bidirectional, request-local alias map."""

    alias_to_value: dict[str, str]

    @classmethod
    def build(
        cls,
        values: list[str] | tuple[str, ...],
        *,
        prefix: str,
    ) -> ShortIdMap:
        ordered = tuple(dict.fromkeys(str(value) for value in values if str(value)))
        width = max(2, len(str(len(ordered))))
        return cls(
            {
                f"{prefix}{index:0{width}d}": value
                for index, value in enumerate(ordered, start=1)
            }
        )

    @property
    def value_to_alias(self) -> dict[str, str]:
        return {value: alias for alias, value in self.alias_to_value.items()}

    def encode(self, value: str) -> str:
        return self.value_to_alias.get(str(value), str(value))

    def decode(self, value: str) -> str:
        return self.alias_to_value.get(str(value), str(value))

    def prompt_map(self, labels: dict[str, str] | None = None) -> dict[str, str]:
        """Return alias-to-readable-label rows without exposing canonical IDs."""

        labels = labels or {}
        return {
            alias: labels.get(value, alias)
            for alias, value in self.alias_to_value.items()
        }


def rewrite_exact_ids(value: Any, *maps: ShortIdMap, decode: bool = False) -> Any:
    """Recursively rewrite exact dictionary keys and scalar ID values."""

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


__all__ = ["ShortIdMap", "rewrite_exact_ids"]
