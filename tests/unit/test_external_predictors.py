from __future__ import annotations

import sys
from types import ModuleType
from typing import ClassVar

import pytest

from fitness_agents.config import ModelConfig, project_root, read_yaml
from fitness_agents.loop import run_campaign
from fitness_agents.models import available_predictors, create_predictor
from fitness_agents.models.external import ExternalPredictorConfigurationError


class _FakeExternalBackend:
    model_version = "fake-v1"
    fit_calls: ClassVar[list[tuple[int, int, int]]] = []

    def __init__(self, context) -> None:
        self.context = context
        self.fitted = False

    def fit(
        self,
        variants,
        observations,
        validation_variants=None,
        validation_observations=None,
    ) -> None:
        assert len(variants) == len(observations)
        self.fit_calls.append(
            (
                len(variants),
                len(validation_variants or []),
                len(validation_observations or []),
            )
        )
        self.fitted = True

    def predict(self, variants):
        assert self.fitted
        # Deliberately reverse the backend output to verify that the adapter restores request order.
        return [
            {
                "variant_id": variant.variant_id,
                "score": float(index),
                "fitness_std": 0.2,
                "component_scores": {"raw_external_score": float(index)},
            }
            for index, variant in reversed(list(enumerate(variants)))
        ]


def test_external_model_aliases_are_config_selectable():
    assert {"kermut", "proteinnpt", "prosst", "pythia_ppi"}.issubset(
        available_predictors()
    )


def test_external_adapter_defaults_to_cpu_and_passes_options(monkeypatch):
    module = ModuleType("fake_kermut_backend")
    module.create_backend = _FakeExternalBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ModelConfig(
        name="kermut",
        backend_factory="fake_kermut_backend:create_backend",
        options={"structure_path": "fake.pdb"},
    )
    predictor = create_predictor(config, seed=13)
    assert predictor.context.device == "cpu"
    assert predictor.context.options["structure_path"] == "fake.pdb"


def test_external_adapter_round_trip(monkeypatch, experiment_config):
    module = ModuleType("fake_prosst_backend")
    module.create_backend = _FakeExternalBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = ModelConfig(
        name="prosst",
        backend_factory="fake_prosst_backend:create_backend",
        batch_size=4,
    )
    predictor = create_predictor(config, seed=5)

    from fitness_agents.data import load_dataset_bundle

    bundle = load_dataset_bundle(
        experiment_config.task.public_data_path,
        experiment_config.task.oracle_data_path,
    )
    predictor.fit(bundle.initial_variants, bundle.initial_observations)
    requested = bundle.oracle_pool[:3]
    predictions = predictor.predict(requested)
    assert [item.variant_id for item in predictions] == [item.variant_id for item in requested]
    assert all(item.fitness_std == pytest.approx(0.2) for item in predictions)
    assert predictions[0].component_scores["raw_external_score"] == pytest.approx(0.0)
    assert predictions[0].model_version.startswith("prosst:")


def test_campaign_uses_selected_external_predictor_and_validation(monkeypatch, config_factory):
    module = ModuleType("fake_proteinnpt_backend")
    module.create_backend = _FakeExternalBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    _FakeExternalBackend.fit_calls = []
    model = ModelConfig(
        name="proteinnpt",
        backend_factory="fake_proteinnpt_backend:create_backend",
        batch_size=4,
    )
    summary = run_campaign(
        config_factory(
            model=model,
            mode="fitness_direct",
            acquisition="greedy",
            knowledge_enabled=False,
            rounds=1,
            budget_per_round=3,
        )
    )
    assert summary["finalized"] is True
    assert len(_FakeExternalBackend.fit_calls) == 2
    assert all(validation_count == 16 for _, validation_count, _ in _FakeExternalBackend.fit_calls)
    assert all(label_count == 16 for _, _, label_count in _FakeExternalBackend.fit_calls)


def test_external_model_without_backend_fails_with_actionable_message():
    with pytest.raises(ExternalPredictorConfigurationError, match="backend_factory"):
        create_predictor(ModelConfig(name="pythia_ppi"), seed=1)


@pytest.mark.parametrize("name", ["kermut", "proteinnpt", "prosst", "pythia_ppi"])
def test_external_model_config_templates_are_cpu_first(name):
    raw = read_yaml(project_root() / f"configs/model/{name}.yaml")
    config = ModelConfig(**raw)
    assert config.name == name
    assert config.device == "cpu"
    assert config.allow_device_fallback is False
