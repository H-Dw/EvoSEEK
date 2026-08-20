from fitness_agents.config import ModelConfig
from fitness_agents.models.capabilities import is_library_scale_predictor


def test_kermut_is_not_library_scale():
    kermut = ModelConfig(
        name="kermut",
        backend_factory="fitness_agents.models.backends.kermut:create_backend",
    )
    assert is_library_scale_predictor(ModelConfig()) is True
    assert is_library_scale_predictor(kermut) is False
