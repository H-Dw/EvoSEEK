import numpy as np

from fitness_agents.contracts.schemas import Variant
from fitness_agents.features import GB1OneHotPairwiseProvider, hamming_distance


def _variant(code: str) -> Variant:
    return Variant(code, code, code, code, 0, "test")


def test_pairwise_feature_shape_and_activity():
    provider = GB1OneHotPairwiseProvider()
    matrix = provider.transform([_variant("VDGV"), _variant("FWAL")])
    assert matrix.shape == (2, 2480)
    assert np.allclose(matrix.sum(axis=1), 10.0)  # four additive + six pairs


def test_hamming_distance():
    assert hamming_distance("VDGV", "VDGV") == 0
    assert hamming_distance("VDGV", "FWAL") == 4

