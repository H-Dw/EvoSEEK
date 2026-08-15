from __future__ import annotations

import numpy as np
import pytest

from fitness_agents.config import ModelConfig
from fitness_agents.data import load_dataset_bundle
from fitness_agents.models import create_predictor

torch = pytest.importorskip("torch", reason="Kermut optional runtime is not installed")
pytest.importorskip("gpytorch", reason="Kermut optional runtime is not installed")
kermut_core = pytest.importorskip(
    "fitness_agents.models.backends.kermut_core",
    reason="Kermut optional runtime is not installed",
)
CompositeKernel = kermut_core.CompositeKernel
Tokenizer = kermut_core.Tokenizer


def test_kermut_fails_before_loading_esm_when_structure_resources_are_missing():
    config = ModelConfig(
        name="kermut",
        device="cpu",
        backend_factory="fitness_agents.models.backends.kermut:create_backend",
        options={"feature_mode": "live_esm2"},
    )

    with pytest.raises(RuntimeError, match="conditional_probs_path.*coords_path"):
        create_predictor(config, seed=3)


def test_kermut_multiply_composition_has_a_valid_learned_scale():
    tokens = Tokenizer()(["CAAA", "DAAA"])
    embeddings = torch.eye(2)
    conditional = torch.full((4, 20), 0.05)
    coords = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    kernel = CompositeKernel(
        Tokenizer()("AAAA"),
        conditional,
        coords,
        composition="multiply",
    )

    covariance = kernel((tokens, embeddings)).to_dense()

    assert covariance.shape == (2, 2)
    assert torch.isfinite(covariance).all()


def _write_feature_resources(tmp_path, variants):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_index = {amino_acid: index for index, amino_acid in enumerate(alphabet)}
    embeddings = np.zeros((len(variants), len(variants[0].variant) * 20), dtype=np.float32)
    zero_shot = np.zeros(len(variants), dtype=np.float32)
    for row, variant in enumerate(variants):
        for position, amino_acid in enumerate(variant.variant):
            embeddings[row, position * 20 + aa_index[amino_acid]] = 1.0
        zero_shot[row] = -0.2 * variant.mutation_count
    feature_path = tmp_path / "kermut_features.npz"
    np.savez_compressed(
        feature_path,
        variant_ids=np.asarray([variant.variant_id for variant in variants]),
        embeddings=embeddings,
        zero_shot=zero_shot,
    )

    rng = np.random.default_rng(17)
    # Mimic official full-protein resources while the benchmark variants are four-site strings.
    conditional = rng.dirichlet(np.ones(20), size=60).astype(np.float32)
    coords = np.stack(
        [np.arange(60, dtype=np.float32) * 3.8, np.zeros(60), np.zeros(60)],
        axis=1,
    )
    conditional_path = tmp_path / "conditional_probs.npy"
    coords_path = tmp_path / "coords.npy"
    np.save(conditional_path, conditional, allow_pickle=False)
    np.save(coords_path, coords, allow_pickle=False)
    return feature_path, conditional_path, coords_path


@pytest.mark.integration
def test_real_kermut_backend_fits_official_gp_and_predicts(experiment_config, tmp_path):
    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path,
        experiment_config.task.oracle_data_path,
    )
    all_variants = [
        *bundle.initial_variants,
        *bundle.validation_variants,
        *bundle.oracle_pool,
        *bundle.final_test,
    ]
    feature_path, conditional_path, coords_path = _write_feature_resources(
        tmp_path, all_variants
    )
    config = ModelConfig(
        name="kermut",
        device="cpu",
        batch_size=3,
        backend_factory="fitness_agents.models.backends.kermut:create_backend",
        options={
            "feature_mode": "precomputed",
            "wild_type_sequence": "VDGV",
            "precomputed_features_path": str(feature_path),
            "conditional_probs_path": str(conditional_path),
            "coords_path": str(coords_path),
            "resource_positions": [39, 40, 41, 54],
            "positions_are_one_indexed": True,
            "n_steps": 3,
            "learning_rate": 0.05,
            "progress_bar": False,
        },
    )
    predictor = create_predictor(config, seed=19)
    predictor.fit(
        bundle.initial_variants,
        bundle.initial_observations,
        bundle.validation_variants,
        bundle.validation_observations,
    )
    predictions = predictor.predict(bundle.oracle_pool[:5])
    assert len(predictions) == 5
    assert all(np.isfinite(item.fitness_mean) for item in predictions)
    assert all(item.fitness_std > 0 for item in predictions)
    assert all("esm2_zero_shot" in item.component_scores for item in predictions)
    assert all("official-main" in item.model_version for item in predictions)
