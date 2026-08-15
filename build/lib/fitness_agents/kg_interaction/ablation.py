from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InteractionAblationConfig:
    """Switches for isolating each part of LLM-KG interaction."""

    enabled_operators: frozenset[str] | None = None
    max_tool_calls: int = 2
    use_counterevidence: bool = True
    stop_when_sufficient: bool = True
    read_only: bool = True

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")

    def operator_enabled(self, name: str) -> bool:
        return self.enabled_operators is None or name in self.enabled_operators

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> InteractionAblationConfig:
        enabled = value.get("enabled_operators")
        return cls(
            enabled_operators=frozenset(enabled) if enabled is not None else None,
            max_tool_calls=int(value.get("max_tool_calls", 2)),
            use_counterevidence=bool(value.get("use_counterevidence", True)),
            stop_when_sufficient=bool(value.get("stop_when_sufficient", True)),
            read_only=bool(value.get("read_only", True)),
        )
