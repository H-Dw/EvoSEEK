"""Typed, deterministic context supplied to the batch Critic."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fitness_agents.contracts.schemas import Hypothesis, Prediction, Variant

PredictionSourceKind = Literal[
    "placeholder",
    "dry_validation",
    "active_posterior",
    "real_model",
]
PredictionStatus = Literal["evaluated", "not_evaluated"]
CalibrationStatus = Literal[
    "not_applicable",
    "unknown",
    "uncalibrated",
    "calibrated",
]


class PredictionReviewCard(BaseModel):
    """Decision-scoped prediction view; ineligible values never reach the LLM."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    variant_id: str = Field(min_length=1, max_length=160)
    source_kind: PredictionSourceKind
    decision_eligible: bool
    calibration_status: CalibrationStatus
    model_version: str = Field(min_length=1, max_length=240)
    prediction_status: PredictionStatus
    fitness_mean: float | None = None
    fitness_std: float | None = Field(default=None, ge=0)
    ood_score: float | None = Field(default=None, ge=0)
    model_disagreement: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def enforce_decision_visibility(self) -> PredictionReviewCard:
        values = (
            self.fitness_mean,
            self.fitness_std,
            self.ood_score,
            self.model_disagreement,
        )
        if not self.decision_eligible:
            if self.prediction_status != "not_evaluated" or any(
                item is not None for item in values
            ):
                raise ValueError(
                    "decision-ineligible predictions must hide numeric values"
                )
        elif self.prediction_status != "evaluated" or any(
            item is None for item in values
        ):
            raise ValueError(
                "decision-eligible predictions require complete numeric values"
            )
        if self.source_kind == "placeholder" and self.decision_eligible:
            raise ValueError("placeholder predictions cannot be decision eligible")
        return self


def prediction_review_card(
    prediction: Prediction,
    *,
    source_kind: PredictionSourceKind,
    decision_eligible: bool,
    calibration_status: CalibrationStatus,
) -> PredictionReviewCard:
    components = prediction.component_scores
    numeric_components = [
        float(item)
        for item in components.values()
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    disagreement = (
        max(numeric_components) - min(numeric_components)
        if len(numeric_components) >= 2
        else 0.0
    )
    if not decision_eligible:
        return PredictionReviewCard(
            variant_id=prediction.variant_id,
            source_kind=source_kind,
            decision_eligible=False,
            calibration_status=calibration_status,
            model_version=prediction.model_version,
            prediction_status="not_evaluated",
        )
    return PredictionReviewCard(
        variant_id=prediction.variant_id,
        source_kind=source_kind,
        decision_eligible=True,
        calibration_status=calibration_status,
        model_version=prediction.model_version,
        prediction_status="evaluated",
        fitness_mean=float(prediction.fitness_mean),
        fitness_std=float(prediction.fitness_std),
        ood_score=float(prediction.ood_score),
        model_disagreement=float(disagreement),
    )


class AssayControl(BaseModel):
    """A repeatable assay control, separate from novel candidate eligibility."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    control_id: str = Field(min_length=1, max_length=160)
    control_kind: Literal["wild_type", "known_neutral", "reference_variant"]
    variant_id: str = Field(min_length=1, max_length=160)
    mutation_notation: str = Field(min_length=1, max_length=240)
    sequence_sha256: str = Field(min_length=64, max_length=64)
    repeat_measurement: Literal[True] = True
    consumes_candidate_quota: Literal[False] = False


class ControlFeasibilityReceipt(BaseModel):
    """Pre-LLM proof that the requested control arm is executable."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    requested_controls: int = Field(ge=0)
    available_controls: int = Field(ge=0)
    selected_controls: int = Field(ge=0)
    reason: Literal[
        "NOT_REQUIRED",
        "FEASIBLE",
        "CONTROL_UNIVERSE_EMPTY",
        "CONTROL_SHORTFALL",
    ]
    available_control_ids_sha256: str = Field(min_length=64, max_length=64)
    selected_control_ids: tuple[str, ...] = ()
    assay_controls: tuple[AssayControl, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.reason in {"NOT_REQUIRED", "FEASIBLE"}


class BatchDiversityMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    minimum_pairwise_hamming: int = Field(ge=0)
    mean_pairwise_hamming: float = Field(ge=0)
    unique_mutation_position_patterns: int = Field(ge=0)
    position_entropy: float = Field(ge=0)
    residue_entropy: float = Field(ge=0)
    hypothesis_mode_coverage: int = Field(ge=0)


class BatchDiversityDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    minimum_pairwise_hamming: int
    mean_pairwise_hamming: float
    unique_mutation_position_patterns: int
    position_entropy: float
    residue_entropy: float
    hypothesis_mode_coverage: int


class BatchDiversityReceipt(BaseModel):
    """Deterministic diversity contract for one proposed batch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    receipt_id: str = Field(min_length=1, max_length=160)
    required_minimum_batch_distance: int = Field(ge=0)
    selected: BatchDiversityMetrics
    candidate_pool: BatchDiversityMetrics
    pool_estimated_max_minimum_pairwise_hamming: int = Field(ge=0)
    threshold_feasible_in_pool: bool
    threshold_satisfied: bool
    previous_receipt_id: str | None = Field(default=None, max_length=160)
    revision_delta: BatchDiversityDelta | None = None


class BatchReviewContext(BaseModel):
    """Runtime-owned facts visible to the batch Critic."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    prediction_status_by_id: dict[str, PredictionReviewCard]
    review_controls: bool = True
    review_diversity: bool = True
    control_feasibility: ControlFeasibilityReceipt | None = None
    diversity: BatchDiversityReceipt | None = None

    @model_validator(mode="after")
    def enforce_review_scope(self) -> BatchReviewContext:
        if not self.review_controls and self.control_feasibility is not None:
            raise ValueError(
                "control_feasibility must be omitted when control review is disabled"
            )
        if not self.review_diversity and self.diversity is not None:
            raise ValueError("diversity must be omitted when diversity review is disabled")
        return self


def _ids_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(sorted(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def control_feasibility_receipt(
    *,
    requested_controls: int,
    available_control_ids: Sequence[str],
    selected_control_ids: Sequence[str],
    assay_controls: Sequence[AssayControl] = (),
) -> ControlFeasibilityReceipt:
    available = tuple(sorted(set(available_control_ids)))
    selected = tuple(dict.fromkeys(selected_control_ids))
    total_available = len(available) + len(assay_controls)
    total_selected = len(selected) + len(assay_controls)
    if requested_controls == 0:
        reason = "NOT_REQUIRED"
    elif total_available == 0:
        reason = "CONTROL_UNIVERSE_EMPTY"
    elif total_available < requested_controls or total_selected < requested_controls:
        reason = "CONTROL_SHORTFALL"
    else:
        reason = "FEASIBLE"
    return ControlFeasibilityReceipt(
        requested_controls=requested_controls,
        available_controls=total_available,
        selected_controls=total_selected,
        reason=reason,
        available_control_ids_sha256=_ids_sha256(available),
        selected_control_ids=selected,
        assay_controls=tuple(assay_controls),
    )


_EDIT_RE = re.compile(r"[A-Z](\d+)([A-Z])")


def _edits(variant: Variant) -> tuple[tuple[int, str], ...]:
    if variant.mutation_notation == "WT":
        return ()
    return tuple(
        (int(position), residue)
        for position, residue in _EDIT_RE.findall(variant.mutation_notation)
    )


def _entropy(values: Sequence[Any]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = float(sum(counts.values()))
    return float(
        -sum((count / total) * math.log2(count / total) for count in counts.values())
    )


def _pairwise_distances(variants: Sequence[Variant]) -> tuple[int, ...]:
    return tuple(
        sum(left != right for left, right in zip(a.variant, b.variant, strict=True))
        for index, a in enumerate(variants)
        for b in variants[index + 1 :]
    )


def _metrics(
    variants: Sequence[Variant],
    *,
    hypothesis: Hypothesis | None,
    position_to_index: Mapping[int, int],
) -> BatchDiversityMetrics:
    distances = _pairwise_distances(variants)
    edits = [_edits(item) for item in variants]
    position_values = [position for item in edits for position, _ in item]
    residue_values = [(position, residue) for item in edits for position, residue in item]
    modes: set[tuple[bool, ...]] = set()
    if hypothesis is not None:
        for variant in variants:
            modes.add(
                tuple(
                    variant.variant[position_to_index[position]] in residues
                    for position, residues in sorted(hypothesis.preferred_residues.items())
                    if position in position_to_index
                )
            )
    return BatchDiversityMetrics(
        minimum_pairwise_hamming=min(distances, default=0),
        mean_pairwise_hamming=(sum(distances) / len(distances) if distances else 0.0),
        unique_mutation_position_patterns=len(
            {tuple(position for position, _ in item) for item in edits}
        ),
        position_entropy=_entropy(position_values),
        residue_entropy=_entropy(residue_values),
        hypothesis_mode_coverage=len(modes),
    )


def _estimated_maximum_minimum_distance(
    variants: Sequence[Variant], batch_size: int
) -> int:
    if len(variants) < 2 or batch_size < 2:
        return 0
    target = min(batch_size, len(variants))
    start = max(
        variants,
        key=lambda item: (
            sum(
                sum(a != b for a, b in zip(item.variant, other.variant, strict=True))
                for other in variants
            ),
            item.variant_id,
        ),
    )
    selected = [start]
    remaining = {item.variant_id: item for item in variants if item != start}
    while remaining and len(selected) < target:
        choice = max(
            remaining.values(),
            key=lambda item: (
                min(
                    sum(a != b for a, b in zip(item.variant, chosen.variant, strict=True))
                    for chosen in selected
                ),
                item.variant_id,
            ),
        )
        selected.append(choice)
        remaining.pop(choice.variant_id)
    return min(_pairwise_distances(selected), default=0)


def batch_diversity_receipt(
    *,
    selected_ids: Sequence[str],
    candidate_pool_ids: Sequence[str],
    variants_by_id: Mapping[str, Variant],
    required_minimum_batch_distance: int,
    hypothesis: Hypothesis | None,
    position_to_index: Mapping[int, int],
    previous: BatchDiversityReceipt | None = None,
) -> BatchDiversityReceipt:
    selected = [variants_by_id[item] for item in selected_ids]
    pool = [variants_by_id[item] for item in candidate_pool_ids]
    selected_metrics = _metrics(
        selected, hypothesis=hypothesis, position_to_index=position_to_index
    )
    pool_metrics = _metrics(
        pool, hypothesis=hypothesis, position_to_index=position_to_index
    )
    estimated_max = _estimated_maximum_minimum_distance(pool, len(selected))
    delta = None
    if previous is not None:
        delta = BatchDiversityDelta(
            minimum_pairwise_hamming=(
                selected_metrics.minimum_pairwise_hamming
                - previous.selected.minimum_pairwise_hamming
            ),
            mean_pairwise_hamming=(
                selected_metrics.mean_pairwise_hamming
                - previous.selected.mean_pairwise_hamming
            ),
            unique_mutation_position_patterns=(
                selected_metrics.unique_mutation_position_patterns
                - previous.selected.unique_mutation_position_patterns
            ),
            position_entropy=(
                selected_metrics.position_entropy - previous.selected.position_entropy
            ),
            residue_entropy=(
                selected_metrics.residue_entropy - previous.selected.residue_entropy
            ),
            hypothesis_mode_coverage=(
                selected_metrics.hypothesis_mode_coverage
                - previous.selected.hypothesis_mode_coverage
            ),
        )
    digest = hashlib.sha256(
        json.dumps(
            {
                "selected_ids": list(selected_ids),
                "candidate_pool_ids": sorted(candidate_pool_ids),
                "required": required_minimum_batch_distance,
                "previous": previous.receipt_id if previous else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:16]
    return BatchDiversityReceipt(
        receipt_id=f"batch-diversity:{digest}",
        required_minimum_batch_distance=required_minimum_batch_distance,
        selected=selected_metrics,
        candidate_pool=pool_metrics,
        pool_estimated_max_minimum_pairwise_hamming=estimated_max,
        threshold_feasible_in_pool=estimated_max >= required_minimum_batch_distance,
        threshold_satisfied=(
            selected_metrics.minimum_pairwise_hamming
            >= required_minimum_batch_distance
        ),
        previous_receipt_id=previous.receipt_id if previous else None,
        revision_delta=delta,
    )


__all__ = [
    "AssayControl",
    "BatchDiversityReceipt",
    "BatchReviewContext",
    "ControlFeasibilityReceipt",
    "PredictionReviewCard",
    "batch_diversity_receipt",
    "control_feasibility_receipt",
    "prediction_review_card",
]
