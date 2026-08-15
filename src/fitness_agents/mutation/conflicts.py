from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from fitness_agents.contracts.schemas import (
    Evidence,
    IssueScope,
    IssueSeverity,
    MutationConflict,
    Prediction,
    Variant,
)

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _conflict_id(code: str, candidate_ids: Sequence[str]) -> str:
    material = f"{code}|{'|'.join(sorted(candidate_ids))}"
    return f"conflict:{hashlib.sha256(material.encode()).hexdigest()[:16]}"


def _hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        return max(len(left), len(right))
    return sum(a != b for a, b in zip(left, right, strict=True))


class ResidueConflictDetector:
    """Deterministic checks for local, discrete mutation-edit conflicts."""

    name = "residue_conflicts"
    version = "1.0.0"

    def detect(
        self,
        variants: Sequence[Variant],
        *,
        wild_type_sites: str,
        mutable_positions: Sequence[int],
    ) -> list[MutationConflict]:
        conflicts: list[MutationConflict] = []
        for variant in variants:
            codes: list[tuple[str, str]] = []
            if len(variant.variant) != len(wild_type_sites):
                codes.append(("RESIDUE_LENGTH_MISMATCH", "Variant-site code length differs from WT"))
            if set(variant.variant).difference(AMINO_ACIDS):
                codes.append(("INVALID_AMINO_ACID", "Variant contains a non-canonical amino acid"))
            if len(mutable_positions) != len(wild_type_sites):
                codes.append(("MUTABLE_POSITION_MAPPING_INVALID", "Mutable-position mapping does not match WT"))
            if len(variant.variant) == len(wild_type_sites):
                actual_depth = sum(
                    left != right
                    for left, right in zip(variant.variant, wild_type_sites, strict=True)
                )
                if actual_depth != variant.mutation_count:
                    codes.append(
                        ("MUTATION_DEPTH_MISMATCH", "Declared mutation_count differs from the sequence")
                    )
                expected_tokens = {
                    f"{source}{position}{target}"
                    for source, position, target in zip(
                        wild_type_sites, mutable_positions, variant.variant, strict=True
                    )
                    if source != target
                }
                actual_tokens = (
                    variant.mutation_notation.split(";")
                    if variant.mutation_notation and variant.mutation_notation != "WT"
                    else []
                )
                seen_positions: dict[int, str] = {}
                source_by_position = dict(zip(mutable_positions, wild_type_sites, strict=True))
                target_by_position = dict(zip(mutable_positions, variant.variant, strict=True))
                for token in actual_tokens:
                    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
                    if match is None:
                        codes.append(("INVALID_MUTATION_NOTATION", "Mutation notation is malformed"))
                        continue
                    source, raw_position, target = match.groups()
                    position = int(raw_position)
                    if position not in source_by_position:
                        codes.append(("FORBIDDEN_POSITION", "Mutation targets a non-mutable position"))
                        continue
                    if position in seen_positions:
                        codes.append(
                            ("MULTIPLE_EDITS_SAME_POSITION", "Multiple edits target the same residue")
                        )
                    seen_positions[position] = target
                    if source != source_by_position[position]:
                        codes.append(("FROM_RESIDUE_MISMATCH", "Mutation source residue differs from WT"))
                    if target != target_by_position[position]:
                        codes.append(("TO_RESIDUE_MISMATCH", "Mutation target differs from the sequence"))
                if set(actual_tokens) != expected_tokens:
                    codes.append(
                        ("MUTATION_NOTATION_MISMATCH", "Mutation notation does not match the sequence")
                    )
            for code, message in codes:
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id(code, (variant.variant_id,)),
                        code=code,
                        scope=IssueScope.RESIDUE,
                        severity=IssueSeverity.BLOCKER,
                        message=message,
                        candidate_ids=(variant.variant_id,),
                        hard=True,
                        detector=f"{self.name}:{self.version}",
                    )
                )
        return conflicts


class SequenceConflictDetector:
    """Complete-sequence and batch checks that cannot be inferred from one residue alone."""

    name = "sequence_conflicts"
    version = "1.0.0"

    def __init__(
        self,
        *,
        ood_warning_threshold: float | None = None,
        model_disagreement_threshold: float | None = None,
        min_batch_distance: int = 1,
    ) -> None:
        self.ood_warning_threshold = ood_warning_threshold
        self.model_disagreement_threshold = model_disagreement_threshold
        self.min_batch_distance = min_batch_distance

    def detect(
        self,
        variants: Sequence[Variant],
        *,
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        revealed_ids: set[str],
        pending_ids: set[str],
        allowed_ids: set[str],
        expected_batch_size: int,
    ) -> list[MutationConflict]:
        conflicts: list[MutationConflict] = []
        ids = [item.variant_id for item in variants]
        sequences = [item.sequence for item in variants]
        hard_cases = (
            ("EMPTY_BATCH", not ids, "Experiment batch is empty", ()),
            ("INCOMPLETE_BATCH", len(ids) != expected_batch_size, "Batch size differs from budget", tuple(ids)),
            ("DUPLICATE_CANDIDATE", len(ids) != len(set(ids)), "Batch repeats a candidate ID", tuple(ids)),
            ("DUPLICATE_SEQUENCE", len(sequences) != len(set(sequences)), "Batch repeats a complete sequence", tuple(ids)),
            (
                "INCONSISTENT_SEQUENCE_LENGTH",
                len({len(item) for item in sequences}) > 1,
                "Complete sequences in one batch have inconsistent lengths",
                tuple(ids),
            ),
            ("UNKNOWN_CANDIDATE", bool(set(ids).difference(allowed_ids)), "Batch contains an unknown candidate", tuple(ids)),
            ("ALREADY_OBSERVED", bool(set(ids).intersection(revealed_ids)), "Batch contains an observed candidate", tuple(ids)),
            ("ALREADY_PENDING", bool(set(ids).intersection(pending_ids)), "Batch contains a pending candidate", tuple(ids)),
        )
        for code, active, message, candidate_ids in hard_cases:
            if active:
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id(code, candidate_ids),
                        code=code,
                        scope=IssueScope.BATCH,
                        severity=IssueSeverity.BLOCKER,
                        message=message,
                        candidate_ids=candidate_ids,
                        hard=True,
                        detector=f"{self.name}:{self.version}",
                    )
                )

        for variant in variants:
            prediction = predictions.get(variant.variant_id)
            if prediction is None:
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id("MISSING_PREDICTION", (variant.variant_id,)),
                        code="MISSING_PREDICTION",
                        scope=IssueScope.SEQUENCE,
                        severity=IssueSeverity.BLOCKER,
                        message="Complete-sequence prediction is missing",
                        candidate_ids=(variant.variant_id,),
                        hard=True,
                        detector=f"{self.name}:{self.version}",
                    )
                )
                continue
            if (
                self.ood_warning_threshold is not None
                and prediction.ood_score >= self.ood_warning_threshold
            ):
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id("HIGH_OOD", (variant.variant_id,)),
                        code="HIGH_OOD",
                        scope=IssueScope.SEQUENCE,
                        severity=IssueSeverity.WARNING,
                        message="Prediction is outside the calibrated in-distribution region",
                        candidate_ids=(variant.variant_id,),
                        hard=False,
                        detector=f"{self.name}:{self.version}",
                    )
                )
            components = list(prediction.component_scores.values())
            if (
                self.model_disagreement_threshold is not None
                and len(components) >= 2
                and float(np.std(components)) >= self.model_disagreement_threshold
            ):
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id("MODEL_DISAGREEMENT", (variant.variant_id,)),
                        code="MODEL_DISAGREEMENT",
                        scope=IssueScope.SEQUENCE,
                        severity=IssueSeverity.WARNING,
                        message="Model components disagree on the complete sequence",
                        candidate_ids=(variant.variant_id,),
                        hard=False,
                        detector=f"{self.name}:{self.version}",
                    )
                )
            bundle = evidence.get(variant.variant_id, ())
            if any(item.score > 0 for item in bundle) and any(item.score < 0 for item in bundle):
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id("EVIDENCE_POLARITY_CONFLICT", (variant.variant_id,)),
                        code="EVIDENCE_POLARITY_CONFLICT",
                        scope=IssueScope.EVIDENCE,
                        severity=IssueSeverity.WARNING,
                        message="Visible evidence contains both supporting and opposing signals",
                        candidate_ids=(variant.variant_id,),
                        evidence_ids=tuple(item.evidence_id for item in bundle),
                        hard=False,
                        detector=f"{self.name}:{self.version}",
                    )
                )

        if len(variants) > 1 and self.min_batch_distance > 0:
            minimum = min(
                _hamming(left.sequence, right.sequence)
                for index, left in enumerate(variants)
                for right in variants[index + 1 :]
            )
            if minimum < self.min_batch_distance:
                conflicts.append(
                    MutationConflict(
                        conflict_id=_conflict_id("BATCH_MODE_COLLAPSE", tuple(ids)),
                        code="BATCH_MODE_COLLAPSE",
                        scope=IssueScope.BATCH,
                        severity=IssueSeverity.WARNING,
                        message="Batch sequence diversity is below the configured minimum",
                        candidate_ids=tuple(ids),
                        hard=False,
                        detector=f"{self.name}:{self.version}",
                    )
                )
        return conflicts


@dataclass(frozen=True)
class EpistasisResult:
    status: str
    fitness_scale: str
    epsilon_mean: float | None
    interval_90: tuple[float, float] | None
    sign_epistasis: bool | None
    reciprocal_sign_epistasis: bool | None
    reason_code: str


def detect_pairwise_epistasis(
    *,
    fitness_scale: str,
    wt_samples: Sequence[float] | None,
    single_a_samples: Sequence[float] | None,
    single_b_samples: Sequence[float] | None,
    double_samples: Sequence[float] | None,
) -> EpistasisResult:
    """Evaluate epistasis from aligned joint posterior samples on one declared fitness scale."""

    if not fitness_scale.strip():
        raise ValueError("fitness_scale must be explicitly declared")
    values = (wt_samples, single_a_samples, single_b_samples, double_samples)
    if any(item is None for item in values):
        return EpistasisResult(
            "UNKNOWN", fitness_scale, None, None, None, None, "MISSING_CONSTITUENT"
        )
    arrays = tuple(np.asarray(item, dtype=float) for item in values)
    lengths = {len(item) for item in arrays}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        return EpistasisResult(
            "UNKNOWN", fitness_scale, None, None, None, None, "UNALIGNED_POSTERIOR"
        )
    wt, a, b, double = arrays
    epsilon = double - a - b + wt
    delta_a_wt = a - wt
    delta_a_given_b = double - b
    delta_b_wt = b - wt
    delta_b_given_a = double - a
    sign_a = float(np.mean(delta_a_wt)) * float(np.mean(delta_a_given_b)) < 0
    sign_b = float(np.mean(delta_b_wt)) * float(np.mean(delta_b_given_a)) < 0
    return EpistasisResult(
        status="DETECTED" if sign_a or sign_b else "ESTIMATED",
        fitness_scale=fitness_scale,
        epsilon_mean=float(np.mean(epsilon)),
        interval_90=(float(np.quantile(epsilon, 0.05)), float(np.quantile(epsilon, 0.95))),
        sign_epistasis=sign_a or sign_b,
        reciprocal_sign_epistasis=sign_a and sign_b,
        reason_code="OK",
    )
