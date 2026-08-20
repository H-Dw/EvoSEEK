from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

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


def _proposal_order(
    candidates: Sequence[Variant],
    *,
    state: CampaignState,
    namespace: str,
) -> dict[str, int]:
    """Return a deterministic seeded tie-break order without content hashes."""

    ordered_ids = sorted(item.variant_id for item in candidates)
    namespace_seed = sum(
        (index + 1) * ord(character) for index, character in enumerate(namespace)
    )
    rng = np.random.default_rng(
        int(state.seed) * 1009 + int(state.round_id) * 9176 + namespace_seed
    )
    permutation = rng.permutation(len(ordered_ids))
    return {
        ordered_ids[int(source_index)]: len(ordered_ids) - rank
        for rank, source_index in enumerate(permutation)
    }


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
        ordered = sorted(candidates, key=lambda item: item.variant_id)
        namespace_seed = sum(
            (index + 1) * ord(character)
            for index, character in enumerate(self.sampling_namespace)
        )
        rng = np.random.default_rng(
            int(state.seed) * 1009 + int(state.round_id) * 9176 + namespace_seed
        )
        indices = rng.permutation(len(ordered))[:target]
        return [ordered[int(index)] for index in indices]


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
        proposal_order = _proposal_order(
            candidates, state=state, namespace=self.sampling_namespace
        )
        ranked = sorted(
            candidates,
            key=lambda item: (
                _hypothesis_matches(item, hypothesis, self.position_to_index),
                proposal_order[item.variant_id],
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

        proposal_order = _proposal_order(
            candidates, state=state, namespace=self.sampling_namespace
        )
        ranked = sorted(
            candidates,
            key=lambda item: (
                _hypothesis_matches(item, hypothesis, self.position_to_index),
                evidence_score(item),
                proposal_order[item.variant_id],
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
