from __future__ import annotations

from collections.abc import Callable

from fitness_agents.contracts.interfaces import FeatureProvider

from .gb1 import GB1OneHotPairwiseProvider
from .sequence import FullSequenceOneHotProvider

FEATURE_PROVIDERS: dict[str, Callable[[], FeatureProvider]] = {
    "gb1_onehot": lambda: GB1OneHotPairwiseProvider(include_pairwise=False),
    "gb1_onehot_pairwise": lambda: GB1OneHotPairwiseProvider(include_pairwise=True),
    "full_sequence_onehot": FullSequenceOneHotProvider,
}


def register_feature_provider(name: str, factory: Callable[[], FeatureProvider]) -> None:
    if not name or name in FEATURE_PROVIDERS:
        raise ValueError(f"Feature provider name must be new and non-empty: {name!r}")
    FEATURE_PROVIDERS[name] = factory


def create_feature_provider(name: str) -> FeatureProvider:
    try:
        return FEATURE_PROVIDERS[name]()
    except KeyError as error:
        raise ValueError(
            f"Unknown feature provider {name!r}; available={sorted(FEATURE_PROVIDERS)}"
        ) from error
