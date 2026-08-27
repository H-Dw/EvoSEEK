"""Shared soft-prior scoring with wild-type-neutral edit semantics."""

from __future__ import annotations

from collections.abc import Mapping

from fitness_agents.contracts.schemas import Hypothesis, Variant


def actionable_preferred_residues(
    hypothesis: Hypothesis | None,
    wild_type_by_position: Mapping[int, str] | None,
) -> dict[int, tuple[str, ...]]:
    """Return only residue preferences that represent an actual edit.

    A native residue may be useful as a control or explanatory boundary, but
    retaining it is not an intervention and must not earn a candidate-ranking
    bonus.  Callers without a wild-type map retain the historical behaviour.
    """

    if hypothesis is None:
        return {}
    wild_type = dict(wild_type_by_position or {})
    output: dict[int, tuple[str, ...]] = {}
    for position, residues in hypothesis.preferred_residues.items():
        native = wild_type.get(position)
        actionable = tuple(
            residue for residue in residues if native is None or residue != native
        )
        if actionable:
            output[position] = actionable
    return output


def hypothesis_edit_match_count(
    variant: Variant,
    hypothesis: Hypothesis | None,
    position_to_index: Mapping[int, int],
    wild_type_by_position: Mapping[int, str] | None = None,
) -> int:
    """Count preferred residues only where the candidate is genuinely edited."""

    actionable = actionable_preferred_residues(
        hypothesis,
        wild_type_by_position,
    )
    wild_type = dict(wild_type_by_position or {})
    matches = 0
    for position, residues in actionable.items():
        if position not in position_to_index:
            continue
        residue = variant.variant[position_to_index[position]]
        if residue != wild_type.get(position) and residue in residues:
            matches += 1
    return matches


def hypothesis_edit_match_fraction(
    variant: Variant,
    hypothesis: Hypothesis | None,
    position_to_index: Mapping[int, int],
    wild_type_by_position: Mapping[int, str] | None = None,
) -> float:
    """Return the fraction of actionable preferred positions matched by edits."""

    actionable = actionable_preferred_residues(
        hypothesis,
        wild_type_by_position,
    )
    tested = sum(position in position_to_index for position in actionable)
    if tested == 0:
        return 0.0
    return float(
        hypothesis_edit_match_count(
            variant,
            hypothesis,
            position_to_index,
            wild_type_by_position,
        )
        / tested
    )
