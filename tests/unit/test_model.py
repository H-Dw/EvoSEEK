import numpy as np

from fitness_agents.data import load_dataset_bundle
from fitness_agents.models import create_predictor


def test_ensemble_returns_mean_uncertainty_and_components(experiment_config):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path, experiment_config.task.oracle_data_path
    )
    model = create_predictor(experiment_config.model, seed=3)
    model.fit(
        bundle.initial_variants,
        bundle.initial_observations,
        bundle.validation_variants,
        bundle.validation_observations,
    )
    predictions = model.predict(bundle.oracle_pool[:5])
    assert len(predictions) == 5
    assert all(item.fitness_std > 0 for item in predictions)
    assert all("ridge" in item.component_scores for item in predictions)
    assert all(np.isfinite(item.interval_90).all() for item in predictions)

