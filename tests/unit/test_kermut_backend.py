from __future__ import annotations

import pickle

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


class _FairEsmLikeArgs:
    def __init__(self, arch: str) -> None:
        self.arch = arch


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


def test_pytorch_weights_only_rejects_pickled_config_objects(tmp_path):
    """fair-esm checkpoints pickle config objects; PyTorch 2.6+ weights_only rejects them."""
    payload = {
        "args": _FairEsmLikeArgs("roberta_large"),
        "model": {"weight": torch.tensor([1.0])},
    }
    path = tmp_path / "esm2_t33_650M_UR50D.pt"
    torch.save(payload, path)

    with pytest.raises((RuntimeError, pickle.UnpicklingError)):
        torch.load(str(path), map_location="cpu", weights_only=True)


def test_torch_load_trusted_esm_defaults_to_cpu_without_weights_only(tmp_path):
    from fitness_agents.models.backends.kermut_features import torch_load_trusted_esm

    recorded: dict[str, object] = {}

    class FakeSerialization:
        @staticmethod
        def add_safe_globals(values):
            recorded["safe_globals"] = list(values)

    class FakeTorch:
        serialization = FakeSerialization()

        @staticmethod
        def load(path, map_location=None, weights_only=None):
            recorded["path"] = path
            recorded["map_location"] = map_location
            recorded["weights_only"] = weights_only
            return {"loaded": True}

    path = tmp_path / "esm2_t33_650M_UR50D.pt"
    path.write_bytes(b"placeholder")
    payload = torch_load_trusted_esm(FakeTorch(), path)

    assert payload == {"loaded": True}
    assert recorded["map_location"] == "cpu"
    assert recorded["weights_only"] is False
    assert recorded["safe_globals"][0].__name__ == "Namespace"


def test_torch_load_trusted_esm_allows_fair_esm_namespace(tmp_path):
    import argparse

    from fitness_agents.models.backends.kermut_features import torch_load_trusted_esm

    payload = {
        "args": argparse.Namespace(arch="roberta_large"),
        "model": {"weight": torch.tensor([1.0])},
    }
    path = tmp_path / "esm2_t33_650M_UR50D.pt"
    torch.save(payload, path)

    loaded = torch_load_trusted_esm(torch, path)
    assert loaded["args"].arch == "roberta_large"
    assert torch.equal(loaded["model"]["weight"], torch.tensor([1.0]))


def test_patch_fair_esm_cpu_loaders_replaces_local_and_hub_helpers():
    from fitness_agents.models.backends.kermut_features import (
        DEFAULT_CHECKPOINT_MAP_LOCATION,
        patch_fair_esm_cpu_loaders,
    )

    class FakePretrained:
        load_model_and_alphabet_local = staticmethod(lambda path: ("unpatched-local", path))
        load_hub_workaround = staticmethod(lambda url: ("unpatched-hub", url))

    class FakeHub:
        @staticmethod
        def load_state_dict_from_url(url, progress=True, map_location=None, weights_only=None):
            return {
                "url": url,
                "progress": progress,
                "map_location": map_location,
                "weights_only": weights_only,
            }

        @staticmethod
        def get_dir():
            return "/tmp/torch-hub"

    class FakeTorch:
        hub = FakeHub()

    pretrained = FakePretrained()
    patch_fair_esm_cpu_loaders(pretrained, FakeTorch())
    payload = pretrained.load_hub_workaround("https://example.invalid/esm2.pt")
    assert payload["map_location"] == DEFAULT_CHECKPOINT_MAP_LOCATION
    assert payload["weights_only"] is False
    assert payload["progress"] is False

    patch_fair_esm_cpu_loaders(pretrained, FakeTorch())
    assert pretrained.load_hub_workaround("https://example.invalid/esm2.pt")["url"].endswith("esm2.pt")


def test_live_esm2_uses_local_checkpoint_loader(monkeypatch, tmp_path):
    from fitness_agents.models.backends import kermut_features

    checkpoint = tmp_path / "esm2_t33_650M_UR50D.pt"
    checkpoint.write_bytes(b"placeholder")
    called: dict[str, object] = {}

    class DummyModel:
        def eval(self):
            return self

        def to(self, device):
            del device
            return self

    class DummyAlphabet:
        def get_batch_converter(self):
            return lambda batch: batch

    def fake_load(pretrained, torch_module, path):
        del pretrained, torch_module
        called["path"] = path
        return DummyModel(), DummyAlphabet()

    monkeypatch.setattr(kermut_features, "load_fair_esm_local", fake_load)
    monkeypatch.setattr(kermut_features, "patch_fair_esm_cpu_loaders", lambda *args, **kwargs: None)
    source = kermut_features.LiveESM2KermutFeatures(
        device="cpu",
        batch_size=1,
        checkpoint=str(checkpoint),
        options={"cache_dir": None},
    )
    source._ensure_runtime()
    assert called["path"] == checkpoint


def test_precomputed_store_lookup_skips_live_esm(tmp_path, experiment_config):
    from fitness_agents.models.backends.kermut_features import LiveESM2KermutFeatures

    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path,
        experiment_config.task.oracle_data_path,
    )
    variants = bundle.initial_variants[:4]
    feature_path, _conditional_path, _coords_path = _write_feature_resources(tmp_path, variants)
    source = LiveESM2KermutFeatures(
        device="cpu",
        batch_size=1,
        checkpoint=None,
        options={
            "precomputed_features_path": str(feature_path),
            "cache_dir": None,
        },
    )

    def fail_if_loaded() -> None:
        raise AssertionError("offline ESM-2 store hits must not load fair-esm")

    source._ensure_runtime = fail_if_loaded  # type: ignore[method-assign]
    embeddings, zero_shot = source.encode(variants, "VDGV")
    assert embeddings.shape[0] == 4
    assert zero_shot.shape == (4,)
    assert np.isfinite(embeddings).all()
    assert np.isfinite(zero_shot).all()


def test_official_esm2_checkpoint_unpickles_under_pytorch_weights_only():
    from fitness_agents.models.backends.kermut_features import (
        DEFAULT_ESM2_CHECKPOINT,
        resolve_esm_checkpoint_path,
        torch_load_trusted_esm,
    )

    try:
        path = resolve_esm_checkpoint_path(DEFAULT_ESM2_CHECKPOINT)
    except FileNotFoundError:
        pytest.skip("Official ESM-2 650M checkpoint is not cached locally")
    payload = torch_load_trusted_esm(torch, path)
    assert isinstance(payload, dict)
    assert "model" in payload
