"""Allow-listed role client construction with no provider logic in CampaignRunner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fitness_agents.contracts.interfaces import LLMClient, ReThinkClient


@dataclass(frozen=True)
class RoleClientBundle:
    scientist: LLMClient
    rethink: ReThinkClient


class ClientRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., RoleClientBundle]] = {}

    def register(self, provider: str, factory: Callable[..., RoleClientBundle]) -> None:
        if provider in self._factories:
            raise ValueError(f"Duplicate client provider {provider!r}")
        self._factories[provider] = factory

    def create(self, provider: str, **kwargs: Any) -> RoleClientBundle:
        try:
            factory = self._factories[provider]
        except KeyError as error:
            raise ValueError(f"Unknown LLM provider {provider!r}") from error
        return factory(**kwargs)


def create_role_client_bundle(
    provider: str,
    *,
    rethink_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RoleClientBundle:
    """Build both cognitive roles from one explicit provider allowlist."""

    from .llm import create_llm_client
    from .rethink import create_rethink_client

    registry = ClientRegistry()

    def build(**settings: Any) -> RoleClientBundle:
        rethink_settings = {**settings, **dict(rethink_options or {})}
        return RoleClientBundle(
            scientist=create_llm_client(provider, **settings),
            rethink=create_rethink_client(provider, **rethink_settings),
        )

    registry.register(provider, build)
    return registry.create(provider, **kwargs)
