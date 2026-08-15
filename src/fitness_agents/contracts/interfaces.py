from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .schemas import (
    ApprovedBatch,
    CampaignState,
    ConflictReport,
    CritiqueDecision,
    DraftBatch,
    Evidence,
    FalsificationCriterion,
    FitnessObservation,
    Hypothesis,
    HypothesisAssessment,
    Prediction,
    SelectionRecord,
    Variant,
)


class FeatureProvider(Protocol):
    name: str

    def fit(self, variants: Sequence[Variant]) -> FeatureProvider: ...

    def transform(self, variants: Sequence[Variant]) -> np.ndarray: ...


class FitnessPredictor(Protocol):
    model_version: str

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> FitnessPredictor: ...

    def predict(self, variants: Sequence[Variant]) -> list[Prediction]: ...


class CandidateGenerator(Protocol):
    name: str

    def generate(
        self,
        candidates: Sequence[Variant],
        state: CampaignState,
        hypothesis: Hypothesis | None,
        evidence: dict[str, list[Evidence]],
        limit: int,
    ) -> list[Variant]: ...


class EvidenceProvider(Protocol):
    channel: str

    def evaluate(self, variant: Variant, *, round_id: int) -> Evidence: ...


class KnowledgeGraphTool(Protocol):
    """Allow-listed, read-only graph queries available to scientist agents."""

    tool_name: str

    def hypothesis_context(
        self,
        *,
        round_id: int,
        limit: int | None = None,
    ) -> dict[str, Any]: ...

    def explain_variant(self, variant_id: str, *, round_id: int) -> dict[str, Any]: ...


class AcquisitionPolicy(Protocol):
    name: str

    def score(
        self,
        predictions: Sequence[Prediction],
        knowledge_scores: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, float]: ...

    def select(
        self,
        variants: Sequence[Variant],
        predictions: Sequence[Prediction],
        scores: dict[str, float],
        budget: int,
        diversity_lambda: float,
    ) -> list[str]: ...


class ExperimentBackend(Protocol):
    def submit(self, variant_ids: Sequence[str], round_id: int) -> str: ...

    def collect(self, experiment_run_id: str) -> list[FitnessObservation]: ...

    def open_final_test(self) -> list[FitnessObservation]: ...


class ApprovedExperimentBackend(Protocol):
    def submit(self, batch: ApprovedBatch) -> str: ...

    def collect(self, experiment_run_id: str) -> list[FitnessObservation]: ...

    def open_final_test(self) -> list[FitnessObservation]: ...


class LLMClient(Protocol):
    provider_name: str

    def generate_hypothesis(
        self,
        *,
        sanitized_context: dict[str, Any],
        evidence: Sequence[Evidence],
        output_schema: dict[str, Any],
    ) -> Hypothesis: ...


class CriticClient(Protocol):
    provider_name: str

    def review(self, *, context: dict[str, Any], output_schema: dict[str, Any]) -> CritiqueDecision: ...


class BatchValidator(Protocol):
    version: str

    def validate(self, draft: DraftBatch, **context: Any) -> ConflictReport: ...


class SignalDetector(Protocol):
    name: str
    version: str

    def evaluate(
        self,
        criterion: FalsificationCriterion,
        observations: Sequence[FitnessObservation],
        context: dict[str, Any],
    ) -> Any: ...


class HypothesisEvaluator(Protocol):
    version: str

    def evaluate(self, **context: Any) -> HypothesisAssessment: ...


class ArtifactWriter(Protocol):
    run_dir: Path

    def event(self, event_type: str, payload: dict[str, Any]) -> None: ...

    def write_json(self, relative_path: str, payload: Any) -> Path: ...

    def write_selection(self, round_id: int, records: Sequence[SelectionRecord]) -> Path: ...
