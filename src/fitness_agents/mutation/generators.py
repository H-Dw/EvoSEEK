from __future__ import annotations

import hashlib
import heapq
from collections.abc import Callable, Sequence

from fitness_agents.contracts.schemas import CampaignState, Evidence, Hypothesis, Variant


def _hypothesis_matches(
    variant: Variant,
    hypothesis: Hypothesis | None,
    position_to_index: dict[int, int],
) -> int:
    if hypothesis is None:
        return 0
    return sum(
        variant.variant[position_to_index[position]] in residues
        for position, residues in hypothesis.preferred_residues.items()
        if position in position_to_index
    )


def _proposal_key(
    variant: Variant,
    *,
    state: CampaignState,
    namespace: str,
) -> bytes:
    """Return a label-blind, seed/fold/round-specific ordering key."""

    material = (
        f"candidate-proposal:v1|{namespace}|seed={state.seed}|"
        f"round={state.round_id}|{variant.variant_id}"
    )
    return hashlib.sha256(material.encode("utf-8")).digest()


class EnumeratingCandidateGenerator:
    name = "stable_uniform"

    def __init__(
        self,
        position_to_index: dict[int, int] | None = None,
        *,
        sampling_namespace: str = "closed_pool",
    ) -> None:
        self.position_to_index = dict(position_to_index or {})
        self.sampling_namespace = sampling_namespace

    def generate(
        self,
        candidates: Sequence[Variant],
        state: CampaignState,
        hypothesis: Hypothesis | None,
        evidence: dict[str, list[Evidence]],
        limit: int,
    ) -> list[Variant]:
        if limit <= 0:
            raise ValueError("closed-pool candidate generation requires a positive limit")
        target = min(limit, len(candidates))
        return heapq.nsmallest(
            target,
            candidates,
            key=lambda item: _proposal_key(
                item,
                state=state,
                namespace=self.sampling_namespace,
            ),
        )


class HypothesisCandidateGenerator:
    name = "hypothesis_filtered"

    def __init__(
        self,
        position_to_index: dict[int, int] | None = None,
        *,
        sampling_namespace: str = "closed_pool",
    ) -> None:
        self.position_to_index = dict(position_to_index or {})
        self.sampling_namespace = sampling_namespace

    def generate(
        self,
        candidates: Sequence[Variant],
        state: CampaignState,
        hypothesis: Hypothesis | None,
        evidence: dict[str, list[Evidence]],
        limit: int,
    ) -> list[Variant]:
        ranked = sorted(
            candidates,
            key=lambda item: (
                _hypothesis_matches(item, hypothesis, self.position_to_index),
                _proposal_key(
                    item,
                    state=state,
                    namespace=self.sampling_namespace,
                ),
            ),
            reverse=True,
        )
        if limit <= 0:
            raise ValueError("closed-pool candidate generation requires a positive limit")
        return ranked[: max(limit, 1)]


class KnowledgeCandidateGenerator:
    name = "knowledge_filtered"

    def __init__(
        self,
        position_to_index: dict[int, int] | None = None,
        *,
        sampling_namespace: str = "closed_pool",
    ) -> None:
        self.position_to_index = dict(position_to_index or {})
        self.sampling_namespace = sampling_namespace

    def generate(
        self,
        candidates: Sequence[Variant],
        state: CampaignState,
        hypothesis: Hypothesis | None,
        evidence: dict[str, list[Evidence]],
        limit: int,
    ) -> list[Variant]:
        if evidence:
            evidenced = [item for item in candidates if item.variant_id in evidence]
            if limit > 0 and len(evidenced) >= limit:
                candidates = evidenced

        def evidence_score(item: Variant) -> float:
            bundle = [
                entry
                for entry in evidence.get(item.variant_id, [])
                if entry.contributes_to_selection
            ]
            if not bundle:
                return 0.0
            denominator = sum(max(entry.confidence, 1e-6) for entry in bundle)
            return sum(entry.score * max(entry.confidence, 1e-6) for entry in bundle) / denominator

        ranked = sorted(
            candidates,
            key=lambda item: (
                _hypothesis_matches(item, hypothesis, self.position_to_index),
                evidence_score(item),
                _proposal_key(
                    item,
                    state=state,
                    namespace=self.sampling_namespace,
                ),
            ),
            reverse=True,
        )
        if limit <= 0:
            raise ValueError("closed-pool candidate generation requires a positive limit")
        return ranked[: max(limit, 1)]


CANDIDATE_GENERATORS: dict[str, Callable[[], object]] = {
    "random": EnumeratingCandidateGenerator,
    "fitness_direct": EnumeratingCandidateGenerator,
    "llm_agent": HypothesisCandidateGenerator,
    "knowledge_agent": KnowledgeCandidateGenerator,
}


def register_candidate_generator(mode: str, factory: Callable[[], object]) -> None:
    if not mode or mode in CANDIDATE_GENERATORS:
        raise ValueError(f"Candidate-generator mode must be new and non-empty: {mode!r}")
    CANDIDATE_GENERATORS[mode] = factory


def create_candidate_generator(
    mode: str,
    *,
    position_to_index: dict[int, int] | None = None,
    sampling_namespace: str = "closed_pool",
):
    try:
        return CANDIDATE_GENERATORS[mode](
            position_to_index,
            sampling_namespace=sampling_namespace,
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown experiment mode {mode!r}; available={sorted(CANDIDATE_GENERATORS)}"
        ) from error
