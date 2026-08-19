from __future__ import annotations

from collections.abc import Callable

from fitness_agents.config import ModelConfig
from fitness_agents.contracts.interfaces import FitnessPredictor
from fitness_agents.features import create_feature_provider

from .capabilities import predictor_capabilities
from .ensemble import OneHotHeterogeneousEnsemble
from .external import create_external_predictor

PredictorFactory = Callable[[ModelConfig, int], FitnessPredictor]


def _onehot_ensemble(config: ModelConfig, seed: int) -> FitnessPredictor:
    provider = create_feature_provider(config.feature_provider)
    return OneHotHeterogeneousEnsemble(
        provider,
        ridge_members=config.ridge_members,
        extra_trees_estimators=config.extra_trees_estimators,
        ridge_alpha=config.ridge_alpha,
        bootstrap_fraction=config.bootstrap_fraction,
        conformal_alpha=config.conformal_alpha,
        include_gaussian_process=config.include_gaussian_process,
        seed=seed,
        capabilities=predictor_capabilities(config),
    )


PREDICTORS: dict[str, PredictorFactory] = {
    "onehot_heterogeneous_ensemble": _onehot_ensemble,
    "kermut": lambda config, seed: create_external_predictor("kermut", config, seed),
    "proteinnpt": lambda config, seed: create_external_predictor("proteinnpt", config, seed),
    "prosst": lambda config, seed: create_external_predictor("prosst", config, seed),
    "pythia_ppi": lambda config, seed: create_external_predictor("pythia_ppi", config, seed),
}


def register_predictor(name: str, factory: PredictorFactory) -> None:
    if not name or name in PREDICTORS:
        raise ValueError(f"Predictor name must be new and non-empty: {name!r}")
    PREDICTORS[name] = factory


def create_predictor(config: ModelConfig, *, seed: int) -> FitnessPredictor:
    try:
        return PREDICTORS[config.name](config, seed)
    except KeyError as error:
        raise ValueError(f"Unknown predictor {config.name!r}; available={sorted(PREDICTORS)}") from error


def available_predictors() -> tuple[str, ...]:
    return tuple(sorted(PREDICTORS))
