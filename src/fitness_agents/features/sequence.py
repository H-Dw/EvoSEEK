"""Transparent fixed-length full-sequence features for open-design baselines."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fitness_agents.contracts.schemas import Variant
from fitness_agents.data.gb1 import AMINO_ACIDS

AA_INDEX = {residue: index for index, residue in enumerate(AMINO_ACIDS)}


class FullSequenceOneHotProvider:
    """Additive one-hot encoding over ``Variant.sequence`` at every residue."""

    name = "full_sequence_onehot"

    def __init__(self) -> None:
        self.sequence_length: int | None = None

    def code_for(self, variant: Variant) -> str:
        return variant.sequence

    def fit(self, variants: Sequence[Variant]) -> FullSequenceOneHotProvider:
        if not variants:
            raise ValueError("Full-sequence features require at least one variant")
        lengths = {len(self.code_for(item)) for item in variants}
        if len(lengths) != 1:
            raise ValueError("Full-sequence features require fixed-length sequences")
        self.sequence_length = lengths.pop()
        self.transform(variants[:1])
        return self

    @property
    def output_dim(self) -> int:
        if self.sequence_length is None:
            raise RuntimeError("Full-sequence feature provider must be fitted first")
        return self.sequence_length * len(AMINO_ACIDS)

    def transform(self, variants: Sequence[Variant]) -> np.ndarray:
        if self.sequence_length is None:
            if variants:
                raise RuntimeError("Full-sequence feature provider must be fitted first")
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.zeros((len(variants), self.output_dim), dtype=np.float32)
        for row, item in enumerate(variants):
            sequence = self.code_for(item)
            if len(sequence) != self.sequence_length or any(
                residue not in AA_INDEX for residue in sequence
            ):
                raise ValueError(
                    f"Invalid fixed-length canonical protein sequence for {item.variant_id}"
                )
            for position, residue in enumerate(sequence):
                matrix[row, position * len(AMINO_ACIDS) + AA_INDEX[residue]] = 1.0
        return matrix
