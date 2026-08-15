from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fitness_agents.config import ModelConfig
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant

from .device import resolve_device

SUPPORTED_EXTERNAL_MODELS = frozenset({"kermut", "proteinnpt", "prosst", "pythia_ppi"})


class ExternalPredictorConfigurationError(RuntimeError):
    """Raised when an optional external predictor has not been wired to a backend."""


@dataclass(frozen=True)
class ExternalModelContext:
    """Stable constructor contract passed to model-specific plugin factories."""

    model_name: str
    device: str
    batch_size: int
    seed: int
    checkpoint: str | None
    options: Mapping[str, Any]


def _load_factory(path: str):
    module_name, separator, attribute = path.partition(":")
    if not separator or not module_name or not attribute:
        raise ExternalPredictorConfigurationError(
            "backend_factory must use 'python.module:factory_name' syntax; "
            f"received {path!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ExternalPredictorConfigurationError(
            f"Could not import predictor backend module {module_name!r} for {path!r}. "
            "Install that model's optional environment/package first."
        ) from error
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ExternalPredictorConfigurationError(
            f"Predictor backend module {module_name!r} has no factory {attribute!r}"
        ) from error
    if not callable(factory):
        raise ExternalPredictorConfigurationError(f"Predictor backend {path!r} is not callable")
    return factory


class ExternalPredictorAdapter:
    """Lazy bridge from the core FitnessPredictor contract to optional model packages.

    A backend factory receives :class:`ExternalModelContext` and returns an object exposing the
    same ``fit`` and ``predict`` methods as ``FitnessPredictor``. Its ``predict`` method may return
    full ``Prediction`` instances or dictionaries with at least ``variant_id`` and either
    ``fitness_mean`` or ``score``. All returned means must already follow the project convention:
    higher means better assay fitness.
    """

    def __init__(self, model_name: str, config: ModelConfig, seed: int) -> None:
        if model_name not in SUPPORTED_EXTERNAL_MODELS:
            raise ValueError(f"Unsupported external model {model_name!r}")
        if config.batch_size < 1:
            raise ValueError("model batch_size must be at least one")
        if not config.backend_factory:
            raise ExternalPredictorConfigurationError(
                f"Predictor {model_name!r} is registered but its optional backend is not configured. "
                "Set backend_factory: 'your_package.module:create_backend' in the model YAML."
            )
        resolved_device = resolve_device(
            config.device, allow_fallback=config.allow_device_fallback
        )
        context = ExternalModelContext(
            model_name=model_name,
            device=resolved_device,
            batch_size=config.batch_size,
            seed=seed,
            checkpoint=config.checkpoint,
            options=dict(config.options),
        )
        factory = _load_factory(config.backend_factory)
        backend = factory(context)
        for method in ("fit", "predict"):
            if not callable(getattr(backend, method, None)):
                raise ExternalPredictorConfigurationError(
                    f"Backend {config.backend_factory!r} must implement {method}()"
                )
        self.model_name = model_name
        self.context = context
        self.backend = backend
        backend_version = str(getattr(backend, "model_version", "plugin"))
        self.model_version = f"{model_name}:{backend_version}:{resolved_device}"

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> ExternalPredictorAdapter:
        self.backend.fit(
            variants,
            observations,
            validation_variants,
            validation_observations,
        )
        return self

    def _coerce_prediction(self, value: Prediction | Mapping[str, Any]) -> Prediction:
        if isinstance(value, Prediction):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                f"{self.model_name} backend predictions must be Prediction or mapping, "
                f"received {type(value).__name__}"
            )
        variant_id = str(value["variant_id"])
        raw_mean = value.get("fitness_mean", value.get("score"))
        if raw_mean is None:
            raise ValueError(
                f"{self.model_name} prediction for {variant_id!r} has neither fitness_mean nor score"
            )
        mean = float(raw_mean)
        std = float(value.get("fitness_std", 0.0))
        interval_value = value.get("interval_90")
        interval = (
            (float(interval_value[0]), float(interval_value[1]))
            if interval_value is not None
            else (mean - 1.645 * std, mean + 1.645 * std)
        )
        components = {
            str(key): float(component)
            for key, component in dict(
                value.get("component_scores", {self.model_name: mean})
            ).items()
        }
        return Prediction(
            variant_id=variant_id,
            fitness_mean=mean,
            fitness_std=std,
            interval_90=interval,
            ood_score=float(value.get("ood_score", 0.0)),
            component_scores=components,
            model_version=str(value.get("model_version", self.model_version)),
            is_measured=bool(value.get("is_measured", False)),
        )

    @staticmethod
    def _validate_prediction(prediction: Prediction) -> None:
        numeric = (
            prediction.fitness_mean,
            prediction.fitness_std,
            prediction.interval_90[0],
            prediction.interval_90[1],
            prediction.ood_score,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError(f"Non-finite prediction for {prediction.variant_id!r}")
        if prediction.fitness_std < 0:
            raise ValueError(f"Negative fitness_std for {prediction.variant_id!r}")
        if prediction.interval_90[0] > prediction.interval_90[1]:
            raise ValueError(f"Reversed interval_90 for {prediction.variant_id!r}")

    def predict(self, variants: Sequence[Variant]) -> list[Prediction]:
        if not variants:
            return []
        raw = self.backend.predict(variants)
        predictions = [self._coerce_prediction(value) for value in raw]
        expected_ids = [variant.variant_id for variant in variants]
        actual_ids = [prediction.variant_id for prediction in predictions]
        if len(set(actual_ids)) != len(actual_ids):
            raise ValueError(f"{self.model_name} backend returned duplicate variant IDs")
        if set(actual_ids) != set(expected_ids):
            missing = sorted(set(expected_ids).difference(actual_ids))
            unexpected = sorted(set(actual_ids).difference(expected_ids))
            raise ValueError(
                f"{self.model_name} backend prediction IDs do not match the request; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        by_id = {prediction.variant_id: prediction for prediction in predictions}
        ordered = [by_id[variant_id] for variant_id in expected_ids]
        for prediction in ordered:
            self._validate_prediction(prediction)
        return ordered


def create_external_predictor(
    model_name: str, config: ModelConfig, seed: int
) -> ExternalPredictorAdapter:
    return ExternalPredictorAdapter(model_name, config, seed)
