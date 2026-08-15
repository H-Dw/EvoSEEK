from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

T = TypeVar("T")


class DuplicatePluginError(ValueError):
    """Raised when two pluggable components use the same registry name."""


class UnknownPluginError(KeyError):
    """Raised when a requested pluggable component is not registered."""


class PluginRegistry(Generic[T]):
    """Small dependency-free registry shared by the optional KG modules."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._plugins: dict[str, T] = {}

    def register(self, name: str, plugin: T, *, replace: bool = False) -> T:
        normalized = name.strip()
        if not normalized:
            raise ValueError(f"{self.kind} plugin name must not be empty")
        if normalized in self._plugins and not replace:
            raise DuplicatePluginError(f"{self.kind} plugin {normalized!r} is already registered")
        self._plugins[normalized] = plugin
        return plugin

    def get(self, name: str) -> T:
        try:
            return self._plugins[name]
        except KeyError as exc:
            available = ", ".join(self.names()) or "<none>"
            raise UnknownPluginError(
                f"Unknown {self.kind} plugin {name!r}; available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    def select(self, names: Iterable[str] | None = None) -> tuple[T, ...]:
        selected = self.names() if names is None else tuple(names)
        return tuple(self.get(name) for name in selected)
