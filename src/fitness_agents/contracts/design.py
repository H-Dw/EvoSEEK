"""Typed contracts for candidates generated outside a benchmark pool."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schemas import Variant


@dataclass(frozen=True)
class ResolvedDesignSpace:
    """The single, immutable resolution of an open-design position policy.

    ``computation_positions`` describes the full sequence available to feature and
    evidence providers. ``allowed_mutation_positions`` is the narrower authority
    boundary shared by proposal, Scientist output guards, validation, and audit
    artifacts.
    """

    reference_sequence: str
    computation_positions: tuple[int, ...]
    allowed_mutation_positions: tuple[int, ...]
    position_to_sequence_index: dict[int, int]
    position_policy: str
    policy_include_positions: tuple[int, ...]
    policy_exclude_positions: tuple[int, ...]
    allowed_residues: tuple[str, ...]
    proposer: str
    mutation_depth: int

    def __post_init__(self) -> None:
        computation = set(self.computation_positions)
        allowed = set(self.allowed_mutation_positions)
        if not self.reference_sequence:
            raise ValueError("resolved design space requires a reference sequence")
        if not allowed:
            raise ValueError("resolved design space requires allowed mutation positions")
        if not allowed.issubset(computation):
            raise ValueError("allowed mutation positions must be inside computation positions")
        if set(self.position_to_sequence_index) != computation:
            raise ValueError("position mapping must cover the complete computation context")

    @property
    def reference_length(self) -> int:
        return len(self.reference_sequence)

    def residue_at(self, position: int) -> str:
        return self.reference_sequence[self.position_to_sequence_index[position]]

    @property
    def allowed_wild_type_sites(self) -> str:
        return "".join(self.residue_at(item) for item in self.allowed_mutation_positions)

    def public_dict(self) -> dict[str, Any]:
        return {
            "reference_sequence": self.reference_sequence,
            "reference_length": self.reference_length,
            "computation_positions": list(self.computation_positions),
            "allowed_mutation_positions": list(self.allowed_mutation_positions),
            "position_to_sequence_index": {
                str(key): value for key, value in self.position_to_sequence_index.items()
            },
            "position_policy": self.position_policy,
            "policy_include_positions": list(self.policy_include_positions),
            "policy_exclude_positions": list(self.policy_exclude_positions),
            "allowed_residues": list(self.allowed_residues),
            "proposer": self.proposer,
            "mutation_depth": self.mutation_depth,
        }


@dataclass(frozen=True)
class SequenceEdit:
    position: int
    wild_type: str
    mutant: str

    @property
    def notation(self) -> str:
        return f"{self.wild_type}{self.position}{self.mutant}"


@dataclass(frozen=True)
class SequenceProposal:
    proposal_id: str
    reference_sequence: str
    sequence: str
    edits: tuple[SequenceEdit, ...]
    proposer: str
    position_policy: str

    @property
    def mutation_notation(self) -> str:
        return ";".join(item.notation for item in self.edits) or "WT"

    def to_variant(self) -> Variant:
        """Expose a full-sequence Variant to existing model/evidence interfaces."""

        return Variant(
            variant_id=self.proposal_id,
            variant=self.sequence,
            sequence=self.sequence,
            mutation_notation=self.mutation_notation,
            mutation_count=len(self.edits),
            split_role="open_design_candidate",
        )

    def public_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["mutation_notation"] = self.mutation_notation
        return output


@dataclass(frozen=True)
class RankedSequenceDesign:
    proposal: SequenceProposal
    fitness_mean: float
    fitness_std: float
    interval_90: tuple[float, float]
    ood_score: float
    acquisition_score: float
    knowledge_score: float
    hypothesis_prior: float
    structure_constraint: float
    acquisition_arm: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            **self.proposal.public_dict(),
            "fitness_mean": self.fitness_mean,
            "fitness_std": self.fitness_std,
            "interval_90": self.interval_90,
            "ood_score": self.ood_score,
            "acquisition_score": self.acquisition_score,
            "knowledge_score": self.knowledge_score,
            "hypothesis_prior": self.hypothesis_prior,
            "structure_constraint": self.structure_constraint,
            "acquisition_arm": self.acquisition_arm,
        }
