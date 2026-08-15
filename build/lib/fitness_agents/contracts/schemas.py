from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CampaignPhase(str, Enum):
    INITIALIZED = "initialized"
    MODEL_FIT = "model_fit"
    PROPOSED = "proposed"
    SELECTED = "selected"
    MEASURED = "measured"
    FINALIZED = "finalized"


@dataclass(frozen=True)
class Variant:
    variant_id: str
    variant: str
    sequence: str
    mutation_notation: str
    mutation_count: int
    split_role: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitnessObservation:
    variant_id: str
    fitness: float
    split_role: str
    round_revealed: int
    source: str = "experiment"


@dataclass(frozen=True)
class Prediction:
    variant_id: str
    fitness_mean: float
    fitness_std: float
    interval_90: tuple[float, float]
    ood_score: float
    component_scores: dict[str, float]
    model_version: str
    is_measured: bool = False


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    variant_id: str
    channel: str
    statement: str
    score: float
    source_id: str
    confidence: float
    round_id: int
    evidence_type: str = "computed"


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    preferred_residues: dict[int, tuple[str, ...]]
    evidence_ids: tuple[str, ...]
    expected_outcome: str
    falsification_criterion: str
    parent_hypothesis_id: str | None = None


@dataclass(frozen=True)
class SelectionRecord:
    variant_id: str
    round_id: int
    selection_order: int
    model_rank_all: int
    acquisition_rank_all: int
    eligible_rank: int
    total_candidates: int
    eligible_candidates: int
    fitness_mean: float
    fitness_std: float
    acquisition_score: float
    knowledge_score: float
    evidence_ids: tuple[str, ...]
    hypothesis_id: str | None
    reason: str
    intervention_tags: tuple[str, ...] = ()


@dataclass
class CampaignState:
    run_id: str
    mode: str
    seed: int
    round_id: int = 0
    phase: CampaignPhase = CampaignPhase.INITIALIZED
    observed: list[FitnessObservation] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    selections: list[SelectionRecord] = field(default_factory=list)
    revealed_variant_ids: set[str] = field(default_factory=set)
    final_test_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        value["revealed_variant_ids"] = sorted(self.revealed_variant_ids)
        return value

