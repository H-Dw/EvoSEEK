from __future__ import annotations

from dataclasses import dataclass

from fitness_agents.contracts.schemas import Prediction


@dataclass(frozen=True)
class PosteriorCalibrationSummary:
    plugin: str
    status: str
    visible_observations: int
    training_observations: int
    calibration_observations: int
    model_versions: tuple[str, ...]
    model_weights: tuple[float, ...]
    bias: float
    variance_scale: float
    conformal_radius: float
    conformal_alpha: float
    refit_full: bool


@dataclass(frozen=True)
class CalibratedPosteriorResult:
    predictions: tuple[Prediction, ...]
    calibration: PosteriorCalibrationSummary


@dataclass(frozen=True)
class HybridCandidateScore:
    variant_id: str
    exploitation: float
    exploration: float
    knowledge: float
    composite: float
    fitness_mean: float
    fitness_std: float
    ood_score: float
    knowledge_score: float


@dataclass(frozen=True)
class HybridScoreResult:
    scores: tuple[HybridCandidateScore, ...]

    def composite_by_id(self) -> dict[str, float]:
        return {item.variant_id: item.composite for item in self.scores}

    def by_id(self) -> dict[str, HybridCandidateScore]:
        return {item.variant_id: item for item in self.scores}


@dataclass(frozen=True)
class HybridBatchSelection:
    plugin: str
    selected_ids: tuple[str, ...]
    quotas: dict[str, int]
    selected_by_arm: dict[str, tuple[str, ...]]
    effective_fractions: dict[str, float]
    knowledge_available: bool

