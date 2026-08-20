"""Fail-closed validation for generated full-sequence designs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from fitness_agents.config import CriticConfig
from fitness_agents.contracts.design import ResolvedDesignSpace
from fitness_agents.contracts.schemas import (
    ConflictReport,
    DraftBatch,
    Evidence,
    Hypothesis,
    IssueScope,
    IssueSeverity,
    MutationConflict,
    Prediction,
    Variant,
)
from fitness_agents.mutation.conflicts import AMINO_ACIDS, SequenceConflictDetector
from fitness_agents.mutation.notation import InvalidMutationNotation, parse_mutation_notation

def _conflict(
    code: str,
    message: str,
    *,
    candidate_ids: Sequence[str] = (),
    scope: IssueScope = IssueScope.SEQUENCE,
    detector: str,
) -> MutationConflict:
    return MutationConflict(
        conflict_id=f"C-{code}-{'-'.join(candidate_ids) or 'GLOBAL'}",
        code=code,
        scope=scope,
        severity=IssueSeverity.BLOCKER,
        message=message,
        candidate_ids=tuple(candidate_ids),
        hard=True,
        detector=detector,
    )


class OpenDesignHardValidator:
    """Validate generated sequences against one resolved mutation authority.

    This intentionally does not use ``ResidueConflictDetector`` because that
    detector interprets ``Variant.variant`` as a compact mutable-site code.
    Open design carries the complete sequence in both ``variant`` and
    ``sequence``.
    """

    version = "1.0.0"

    def __init__(
        self,
        design_space: ResolvedDesignSpace,
        critic: CriticConfig,
        *,
        mutation_depth: int = 1,
    ) -> None:
        self.design_space = design_space
        self.mutation_depth = mutation_depth
        self.sequence = SequenceConflictDetector(
            ood_warning_threshold=critic.ood_warning_threshold,
            model_disagreement_threshold=critic.model_disagreement_threshold,
            min_batch_distance=(
                critic.min_batch_distance if critic.review_diversity else 0
            ),
        )

    def _candidate_conflicts(self, variant: Variant) -> list[MutationConflict]:
        detector = f"open_design_contract:{self.version}"
        candidate_ids = (variant.variant_id,)
        output: list[MutationConflict] = []

        def add(code: str, message: str, scope: IssueScope = IssueScope.SEQUENCE) -> None:
            output.append(
                _conflict(
                    code,
                    message,
                    candidate_ids=candidate_ids,
                    scope=scope,
                    detector=detector,
                )
            )

        reference = self.design_space.reference_sequence
        sequence = variant.sequence
        if variant.variant != sequence:
            add("FULL_SEQUENCE_ALIAS_MISMATCH", "variant and sequence must contain the same full sequence")
        if len(sequence) != len(reference):
            add("REFERENCE_LENGTH_MISMATCH", "Generated sequence length differs from the reference")
            return output
        if set(sequence).difference(AMINO_ACIDS):
            add("INVALID_AMINO_ACID", "Generated sequence contains a non-canonical amino acid")
        derived = []
        for position in self.design_space.computation_positions:
            index = self.design_space.position_to_sequence_index[position]
            wild_type = reference[index]
            mutant = sequence[index]
            if mutant != wild_type:
                derived.append((position, wild_type, mutant))
                if position not in self.design_space.allowed_mutation_positions:
                    add(
                        "FORBIDDEN_POSITION",
                        f"Mutation targets position {position}, which is outside the resolved design space",
                        IssueScope.RESIDUE,
                    )
                if mutant not in self.design_space.allowed_residues:
                    add(
                        "FORBIDDEN_MUTANT_RESIDUE",
                        f"Residue {mutant} is not allowed at position {position}",
                        IssueScope.RESIDUE,
                    )
        if not derived:
            add("NO_MUTATION", "Open-design candidates must differ from the reference")
        if len(derived) > self.mutation_depth:
            add(
                "MUTATION_DEPTH_EXCEEDED",
                f"Generated sequence has {len(derived)} edits; allowed depth is {self.mutation_depth}",
                IssueScope.RESIDUE,
            )
        if variant.mutation_count != len(derived):
            add(
                "MUTATION_DEPTH_MISMATCH",
                "Declared mutation_count differs from the full-sequence edit count",
                IssueScope.RESIDUE,
            )

        try:
            parsed = parse_mutation_notation(variant.mutation_notation)
        except InvalidMutationNotation:
            add("INVALID_MUTATION_NOTATION", "Mutation notation is malformed", IssueScope.RESIDUE)
            parsed = None
        if parsed is not None:
            identities = [item.identity for item in parsed]
            if len({item.position for item in parsed}) != len(parsed):
                add(
                    "MULTIPLE_EDITS_SAME_POSITION",
                    "Mutation notation contains multiple edits for one position",
                    IssueScope.RESIDUE,
                )
            if set(identities) != set(derived) or len(identities) != len(derived):
                add(
                    "MUTATION_NOTATION_MISMATCH",
                    "Mutation notation does not exactly match edits derived from the full sequence",
                    IssueScope.RESIDUE,
                )
        return output

    def validate(
        self,
        draft: DraftBatch,
        *,
        variants: Mapping[str, Variant],
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        revealed_ids: set[str],
        pending_ids: set[str],
        allowed_ids: set[str],
        expected_batch_size: int,
        prediction_decision_eligible: Mapping[str, bool] | None = None,
        hypothesis: Hypothesis | None = None,
    ) -> ConflictReport:
        detector = f"open_design_contract:{self.version}"
        conflicts: list[MutationConflict] = []
        reference = self.design_space.reference_sequence
        if set(reference).difference(AMINO_ACIDS):
            conflicts.append(
                _conflict(
                    "INVALID_REFERENCE_SEQUENCE",
                    "Reference sequence contains a non-canonical amino acid",
                    scope=IssueScope.SYSTEM,
                    detector=detector,
                )
            )
        if hypothesis is not None and hypothesis.hard_residue_constraints:
            for candidate_id in draft.candidate_ids:
                if candidate_id not in variants:
                    continue
                candidate = variants[candidate_id]
                violates = any(
                    position not in self.design_space.position_to_sequence_index
                    or candidate.sequence[
                        self.design_space.position_to_sequence_index[position]
                    ]
                    not in allowed
                    for position, allowed in hypothesis.hard_residue_constraints.items()
                )
                if violates:
                    conflicts.append(
                        _conflict(
                            "HARD_RESIDUE_CONSTRAINT_VIOLATION",
                            "Candidate violates explicit hard_residue_constraints; soft preferences are not a gate",
                            candidate_ids=(candidate_id,),
                            scope=IssueScope.RESIDUE,
                            detector=detector,
                        )
                    )
        missing_variants = tuple(item for item in draft.candidate_ids if item not in variants)
        if missing_variants:
            conflicts.append(
                _conflict(
                    "MISSING_GENERATED_SEQUENCE",
                    "Draft references generated candidates without full sequences",
                    candidate_ids=missing_variants,
                    detector=detector,
                )
            )
        selected = [variants[item] for item in draft.candidate_ids if item in variants]
        for variant in selected:
            conflicts.extend(self._candidate_conflicts(variant))
        conflicts.extend(
            self.sequence.detect(
                selected,
                predictions=predictions,
                evidence=evidence,
                revealed_ids=revealed_ids,
                pending_ids=pending_ids,
                allowed_ids=allowed_ids,
                expected_batch_size=expected_batch_size,
                prediction_decision_eligible=prediction_decision_eligible,
            )
        )

        visible_evidence_ids = {
            item.evidence_id
            for candidate_id in draft.candidate_ids
            for item in evidence.get(candidate_id, ())
        }
        missing_evidence = {
            evidence_id
            for rationale in draft.design_rationales
            for evidence_id in rationale.evidence_ids
            if evidence_id not in visible_evidence_ids
        }
        if missing_evidence:
            conflicts.append(
                _conflict(
                    "MISSING_RATIONALE_EVIDENCE",
                    "Design rationale cites evidence outside the frozen snapshot",
                    scope=IssueScope.EVIDENCE,
                    detector=f"evidence_reference:{self.version}",
                )
            )
        return ConflictReport(
            report_id=f"V{draft.round_id:02d}-{draft.review_attempt:02d}",
            round_id=draft.round_id,
            conflicts=tuple(conflicts),
            validator_version=self.version,
            draft_batch_id=draft.draft_batch_id,
        )
