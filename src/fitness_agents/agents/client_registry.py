"""Allow-listed role client construction with no provider logic in CampaignRunner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from fitness_agents.contracts.interfaces import (
    HypothesisReThinkClient,
    LLMClient,
    ReThinkClient,
)


@dataclass(frozen=True)
class RoleClientBundle:
    scientist: LLMClient
    rethink: ReThinkClient | HypothesisReThinkClient


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
    rethink_mode: str = "sample",
    rethink_options: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> RoleClientBundle:
    """Build both cognitive roles while preserving the sample-mode default."""

    from .llm import create_llm_client
    from .rethink import create_hypothesis_rethink_client, create_rethink_client

    if rethink_mode not in {"sample", "hypothesis"}:
        raise ValueError("rethink_mode must be sample or hypothesis")

    registry = ClientRegistry()

    def build(**settings: Any) -> RoleClientBundle:
        rethink_settings = {**settings, **dict(rethink_options or {})}
        if rethink_mode == "sample" and "parallel_dimension_groups" in rethink_settings:
            rethink_settings.setdefault(
                "dimension_parallel",
                rethink_settings.pop("parallel_dimension_groups"),
            )
        rethink_factory = (
            create_hypothesis_rethink_client
            if rethink_mode == "hypothesis"
            else create_rethink_client
        )
        return RoleClientBundle(
            scientist=create_llm_client(provider, **settings),
            rethink=rethink_factory(provider, **rethink_settings),
        )

    registry.register(provider, build)
    return registry.create(provider, **kwargs)
