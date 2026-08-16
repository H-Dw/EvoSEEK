from __future__ import annotations

from copy import deepcopy

import numpy as np
from common import (
    ensure,
    load_config,
    parse_args,
    placeholder,
    resolve_output,
    write_legacy_benchmark,
    write_result,
)

from fitness_agents.config import ModelConfig
from fitness_agents.data import load_dataset_bundle
from fitness_agents.evaluation.metrics import prediction_metrics
from fitness_agents.features import create_feature_provider
from fitness_agents.models import available_predictors, create_predictor
from fitness_agents.utils.progress import configure_progress_logging


def _run_predictor(config: ModelConfig, bundle, seed: int):
    predictor = create_predictor(config, seed=seed)
    predictor.fit(
        bundle.initial_variants,
        bundle.initial_observations,
        bundle.validation_variants,
        bundle.validation_observations,
    )
    predictions = predictor.predict(bundle.oracle_pool[:12])
    ensure(len(predictions) == 12, f"{config.name} returned the wrong prediction count")
    ensure(all(np.isfinite(item.fitness_mean) for item in predictions), "Non-finite mean")
    ensure(all(item.fitness_std > 0 for item in predictions), "Uncertainty must be positive")
    return predictions


def _has_nested_placeholder(value: object) -> bool:
    if placeholder(value):
        return True
    if isinstance(value, dict):
        return any(_has_nested_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_nested_placeholder(item) for item in value)
    return False


def main() -> None:
    args = parse_args("configs/module_tests/predictive_model.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    paths = write_legacy_benchmark(output / "input", seed=int(config["seed"]))
    bundle = load_dataset_bundle(paths["public"], paths["oracle"])

    baseline_config = ModelConfig(**config["baseline"])
    provider = create_feature_provider(baseline_config.feature_provider)
    train_features = provider.fit(bundle.initial_variants).transform(bundle.initial_variants)
    candidate_features = provider.transform(bundle.oracle_pool[: int(config["prediction_size"])])
    ensure(train_features.ndim == 2, "Feature provider did not return a matrix")
    ensure(train_features.shape[1] == candidate_features.shape[1], "Feature width drifted")

    baseline_predictions = _run_predictor(baseline_config, bundle, int(config["seed"]))
    candidate_truth = [
        item
        for item in load_dataset_bundle(paths["public"], paths["oracle"]).initial_observations
        if item.variant_id in {prediction.variant_id for prediction in baseline_predictions}
    ]
    # Candidate labels are intentionally not exposed by DatasetBundle; use a separate evaluator view.
    import pandas as pd

    label_frame = pd.read_csv(paths["oracle"]).set_index("variant_id")
    from fitness_agents.contracts.schemas import FitnessObservation

    candidate_truth = [
        FitnessObservation(
            prediction.variant_id,
            float(label_frame.loc[prediction.variant_id, "fitness"]),
            "evaluator_only",
            -2,
            "module_test_evaluator",
        )
        for prediction in baseline_predictions
    ]
    metrics = prediction_metrics(baseline_predictions, candidate_truth)
    ensure(metrics["n"] == len(baseline_predictions), "Metrics alignment dropped predictions")

    expected_external = set(config["external_contract_test"]["model_names"])
    ensure(expected_external.issubset(available_predictors()), "External model aliases are missing")
    external_results: dict[str, object] = {}
    requested_ids = [item.variant_id for item in bundle.oracle_pool[:6]]
    for index, name in enumerate(config["external_contract_test"]["model_names"]):
        external_config = ModelConfig(
            name=str(name),
            device="cpu",
            backend_factory=str(config["external_contract_test"]["backend_factory"]),
            checkpoint=str(config["checkpoint_placeholders"].get(str(name), "contract-test-unused")),
            options={"contract_test": True},
        )
        predictions = _run_predictor(external_config, bundle, int(config["seed"]) + index + 1)[:6]
        ensure(
            [item.variant_id for item in predictions] == requested_ids,
            f"External adapter did not restore request order for {name}",
        )
        external_results[str(name)] = {
            "model_version": predictions[0].model_version,
            "prediction_count": len(predictions),
            "checkpoint_value_was_not_loaded": True,
        }

    real_result: dict[str, object] = {"enabled": False, "status": "skipped"}
    real_raw = deepcopy(config["real_external_model"])
    enabled = bool(real_raw.pop("enabled", False))
    if enabled:
        configure_progress_logging()
        ensure(
            not _has_nested_placeholder(real_raw),
            "Replace all real_external_model placeholders before enabling it",
        )
        real_predictions = _run_predictor(
            ModelConfig(**real_raw), bundle, int(config["seed"]) + 100
        )
        real_result = {
            "enabled": True,
            "status": "passed",
            "model_version": real_predictions[0].model_version,
        }

    write_result(
        output,
        "predictive_model",
        {
            "config": config["_config_path"],
            "feature_shapes": {
                "train": train_features.shape,
                "candidate": candidate_features.shape,
            },
            "baseline_model_version": baseline_predictions[0].model_version,
            "baseline_metrics": metrics,
            "external_contracts": external_results,
            "real_external_model": real_result,
        },
    )


if __name__ == "__main__":
    main()

