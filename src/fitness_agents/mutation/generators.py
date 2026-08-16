from __future__ import annotations

from collections.abc import Callable, Sequence

from fitness_agents.contracts.schemas import CampaignState, Evidence, Hypothesis, Variant


def _hypothesis_matches(variant: Variant, hypothesis: Hypothesis | None) -> int:
    if hypothesis is None:
        return 0
    index_by_position = {39: 0, 40: 1, 41: 2, 54: 3}
    return sum(
        variant.variant[index_by_position[position]] in residues
        for position, residues in hypothesis.preferred_residues.items()
        if position in index_by_position
    )


class EnumeratingCandidateGenerator:
    name = "enumerating"

    def generate(
        self,
        candidates: Sequence[Variant],
        state: CampaignState,
        hypothesis: Hypothesis | None,
        evidence: dict[str, list[Evidence]],
        limit: int,
    ) -> list[Variant]:
        ranked = list(candidates)
        if limit <= 0:
            return ranked
        return ranked[: max(limit, 1)]


class HypothesisCandidateGenerator:
    name = "hypothesis_filtered"

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
            key=lambda item: (_hypothesis_matches(item, hypothesis), item.variant_id),
            reverse=True,
        )
        if limit <= 0:
            return ranked
        return ranked[: max(limit, 1)]


class KnowledgeCandidateGenerator:
    name = "knowledge_filtered"

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
            bundle = evidence.get(item.variant_id, [])
            if not bundle:
                return 0.0
            denominator = sum(max(entry.confidence, 1e-6) for entry in bundle)
            return sum(entry.score * max(entry.confidence, 1e-6) for entry in bundle) / denominator

        ranked = sorted(
            candidates,
            key=lambda item: (
                _hypothesis_matches(item, hypothesis),
                evidence_score(item),
                item.variant_id,
            ),
            reverse=True,
        )
        if limit <= 0:
            return ranked
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


def create_candidate_generator(mode: str):
    try:
        return CANDIDATE_GENERATORS[mode]()
    except KeyError as error:
        raise ValueError(
            f"Unknown experiment mode {mode!r}; available={sorted(CANDIDATE_GENERATORS)}"
        ) from error
