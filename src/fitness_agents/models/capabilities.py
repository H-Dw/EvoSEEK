"""Capability resolution without constructing heavyweight model backends."""

from __future__ import annotations

from fitness_agents.config import ModelConfig
from fitness_agents.contracts.capabilities import PredictorCapabilities


def is_library_scale_predictor(config: ModelConfig) -> bool:
    """Return whether ``config`` can score a full remaining GB1-scale pool.

    Kermut/ESM-2 is reserved for the selected wet-lab batch, an optional
    bounded acquisition screen, and held-out final-test. It must not be used
    to score the remaining candidate library.
    """

    name = (config.name or "").casefold()
    factory = (config.backend_factory or "").casefold()
    return "kermut" not in name and "kermut" not in factory


def predictor_capabilities(config: ModelConfig) -> PredictorCapabilities:
    """Return fail-closed capabilities for one configured predictor.

    Built-in feature providers own their declarations. Optional external
    backends must declare both capabilities explicitly in model configuration;
    an absent declaration is treated as unsupported.
    """

    if config.name == "onehot_heterogeneous_ensemble":
        if config.feature_provider == "full_sequence_onehot":
            return PredictorCapabilities(
                supports_full_sequence=True,
                supports_generated_sequences=True,
            )
        return PredictorCapabilities()
    if config.capabilities is not None:
        return config.capabilities
    return PredictorCapabilities()

