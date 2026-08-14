from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np

from fitness_agents.contracts.schemas import Variant
from fitness_agents.data.gb1 import AMINO_ACIDS

AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("Hamming distance requires equal-length strings")
    return sum(a != b for a, b in zip(left, right, strict=True))


class GB1OneHotPairwiseProvider:
    """Transparent GB1 encoding with additive and pairwise epistasis terms."""

    name = "gb1_onehot_pairwise"

    def __init__(self, include_pairwise: bool = True) -> None:
        self.include_pairwise = include_pairwise
        self._pairs = tuple(combinations(range(4), 2))

    def fit(self, variants: Sequence[Variant]) -> GB1OneHotPairwiseProvider:
        self.transform(variants[:1])
        return self

    @property
    def output_dim(self) -> int:
        additive = 4 * len(AMINO_ACIDS)
        pairwise = len(self._pairs) * len(AMINO_ACIDS) ** 2 if self.include_pairwise else 0
        return additive + pairwise

    def transform(self, variants: Sequence[Variant]) -> np.ndarray:
        matrix = np.zeros((len(variants), self.output_dim), dtype=np.float32)
        additive_size = 4 * len(AMINO_ACIDS)
        for row, item in enumerate(variants):
            if len(item.variant) != 4 or any(aa not in AA_INDEX for aa in item.variant):
                raise ValueError(f"Invalid GB1 four-site code: {item.variant!r}")
            indices = [AA_INDEX[aa] for aa in item.variant]
            for position, aa_index in enumerate(indices):
                matrix[row, position * 20 + aa_index] = 1.0
            if self.include_pairwise:
                for pair_index, (left, right) in enumerate(self._pairs):
                    offset = additive_size + pair_index * 400
                    matrix[row, offset + indices[left] * 20 + indices[right]] = 1.0
        return matrix

