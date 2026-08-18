from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fitness_agents.config import ActiveLearningConfig, ModelConfig
from fitness_agents.plugin_registry import PluginRegistry

from .module import LightweightCalibratedHybridModule

ActiveLearningFactory = Callable[..., LightweightCalibratedHybridModule]
MODULES: PluginRegistry[ActiveLearningFactory] = PluginRegistry("active_learning_module")


def _lightweight_factory(
    config: ActiveLearningConfig,
    *,
    fallback_model: ModelConfig,
    predictor_factory: Callable[..., Any],
    seed: int,
) -> LightweightCalibratedHybridModule:
    return LightweightCalibratedHybridModule(
        config,
        fallback_model=fallback_model,
        predictor_factory=predictor_factory,
        seed=seed,
    )


MODULES.register(LightweightCalibratedHybridModule.name, _lightweight_factory)


def register_active_learning_module(name: str, factory: ActiveLearningFactory) -> None:
    MODULES.register(name, factory)


def create_active_learning_module(
    config: ActiveLearningConfig,
    *,
    fallback_model: ModelConfig,
    predictor_factory: Callable[..., Any],
    seed: int,
) -> LightweightCalibratedHybridModule:
    factory = MODULES.get(config.module)
    return factory(
        config,
        fallback_model=fallback_model,
        predictor_factory=predictor_factory,
        seed=seed,
    )


def available_active_learning_modules() -> tuple[str, ...]:
    return MODULES.names()

