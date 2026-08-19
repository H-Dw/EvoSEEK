"""Runtime capability contracts for fitness predictors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictorCapabilities:
    """Declare whether a predictor can score de novo complete sequences."""

    supports_full_sequence: bool = False
    supports_generated_sequences: bool = False

    @property
    def supports_open_design(self) -> bool:
        return self.supports_full_sequence and self.supports_generated_sequences

