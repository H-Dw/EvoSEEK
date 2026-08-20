"""Pluggable proposal operators for full-reference sequence design."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fitness_agents.config import DesignerConfig
from fitness_agents.contracts.design import (
    ResolvedDesignSpace,
    SequenceEdit,
    SequenceProposal,
)
from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.plugin_registry import PluginRegistry
from fitness_agents.protein_features import ProteinTaskContext


class OpenDesignProposer(Protocol):
    name: str

    def propose(self) -> list[SequenceProposal]: ...


def resolve_design_positions(
    context: ProteinTaskContext, config: DesignerConfig
) -> tuple[int, ...]:
    all_positions = tuple(context.mutable_positions)
    available = set(all_positions)
    include = set(config.include_positions)
    exclude = set(config.exclude_positions)
    unknown = sorted((include | exclude).difference(available))
    if unknown:
        raise ValueError(f"designer positions are outside the reference sequence: {unknown}")
    if config.position_policy == "all":
        selected = all_positions
    elif config.position_policy == "all_except":
        selected = tuple(item for item in all_positions if item not in exclude)
    elif config.position_policy == "include":
        selected = tuple(item for item in all_positions if item in include)
    else:
        selected = tuple(item for item in all_positions if item in include or not include)
    selected = tuple(item for item in selected if item not in exclude)
    if not selected:
        raise ValueError("designer position policy resolved to an empty sequence space")
    return selected


def resolve_design_space(
    context: ProteinTaskContext, config: DesignerConfig
) -> ResolvedDesignSpace:
    """Resolve the position policy exactly once at the orchestration boundary."""

    return ResolvedDesignSpace(
        reference_sequence=context.full_sequence,
        computation_positions=tuple(context.mutable_positions),
        allowed_mutation_positions=resolve_design_positions(context, config),
        position_to_sequence_index=dict(context.position_to_sequence_index),
        position_policy=config.position_policy,
        policy_include_positions=tuple(config.include_positions),
        policy_exclude_positions=tuple(config.exclude_positions),
        allowed_residues=tuple(config.allowed_residues),
        proposer=config.proposer,
        mutation_depth=config.mutation_depth,
    )


class AllPositionSubstitutionProposer:
    """Enumerate every allowed non-WT single substitution over selected positions."""

    name = "all_position_substitution"

    def __init__(
        self,
        context: ProteinTaskContext,
        config: DesignerConfig,
        design_space: ResolvedDesignSpace,
    ) -> None:
        self.context = context
        self.config = config
        self.design_space = design_space
        self.positions = design_space.allowed_mutation_positions

    def propose(self) -> list[SequenceProposal]:
        proposals: list[SequenceProposal] = []
        reference = self.context.full_sequence
        proposal_index = 0
        for position in self.positions:
            sequence_index = self.context.position_to_sequence_index[position]
            wild_type = reference[sequence_index]
            for mutant in self.config.allowed_residues:
                if mutant == wild_type:
                    continue
                proposal_index += 1
                sequence = reference[:sequence_index] + mutant + reference[sequence_index + 1 :]
                proposals.append(
                    SequenceProposal(
                        proposal_id=f"P{proposal_index:05d}",
                        reference_sequence=reference,
                        sequence=sequence,
                        edits=(SequenceEdit(position, wild_type, mutant),),
                        proposer=self.name,
                        position_policy=self.config.position_policy,
                    )
                )
        return proposals


OPEN_DESIGN_PROPOSERS: PluginRegistry[type[AllPositionSubstitutionProposer]] = PluginRegistry(
    "open_design_proposer"
)
OPEN_DESIGN_PROPOSERS.register(
    AllPositionSubstitutionProposer.name, AllPositionSubstitutionProposer
)


def create_open_design_proposer(
    config: DesignerConfig,
    context: ProteinTaskContext,
    design_space: ResolvedDesignSpace,
) -> OpenDesignProposer:
    proposer_type = OPEN_DESIGN_PROPOSERS.get(config.proposer)
    return proposer_type(context, config, design_space)


def normalize_visible_variants(
    variants: Sequence[Variant],
    observations: Sequence[FitnessObservation],
    *,
    source_context: ProteinTaskContext,
    open_context: ProteinTaskContext,
) -> tuple[list[Variant], list[FitnessObservation]]:
    """Project visible variants onto the configured reference without changing label IDs."""

    observation_by_id = {item.variant_id: item for item in observations}
    output: list[Variant] = []
    normalized_observations: list[FitnessObservation] = []
    reference = open_context.full_sequence
    for variant in variants:
        if variant.variant_id not in observation_by_id:
            continue
        if len(variant.sequence) == len(reference):
            sequence = variant.sequence.upper()
        elif len(variant.variant) == len(reference):
            sequence = variant.variant.upper()
        elif len(variant.variant) == len(source_context.mutable_positions):
            sequence = source_context.full_sequence_for_variant(variant.variant.upper())
        else:
            raise ValueError(
                f"Visible variant {variant.variant_id} cannot be projected onto the reference"
            )
        if len(sequence) != len(reference):
            raise ValueError("Normalized visible sequences must match the reference length")
        edits = [
            SequenceEdit(position, reference[index], residue)
            for position, index in open_context.position_to_sequence_index.items()
            if (residue := sequence[index]) != reference[index]
        ]
        output.append(
            Variant(
                variant_id=variant.variant_id,
                variant=sequence,
                sequence=sequence,
                mutation_notation=";".join(item.notation for item in edits) or "WT",
                mutation_count=len(edits),
                split_role=variant.split_role,
            )
        )
        normalized_observations.append(observation_by_id[variant.variant_id])
    if len(output) != len(observation_by_id):
        missing = sorted(set(observation_by_id).difference(item.variant_id for item in output))
        raise ValueError(f"Visible observations lack {len(missing)} input variants")
    return output, normalized_observations
