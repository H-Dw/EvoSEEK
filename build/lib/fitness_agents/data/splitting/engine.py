from __future__ import annotations

from fitness_agents.data.canonical import CanonicalDataset

from .al96 import build_al96
from .contracts import SplitRequest, SplitResult
from .flip_ood import build_flip_ood
from .mutation_ood import build_mutation_ood

STRATEGIES = {
    "al96_closed_loop": build_al96,
    "flip_static_ood": build_flip_ood,
    "mutation_identity_ood": build_mutation_ood,
}


def build_split(dataset: CanonicalDataset, request: SplitRequest) -> SplitResult:
    try:
        builder = STRATEGIES[request.strategy]
    except KeyError as error:
        raise ValueError(f"Unknown split strategy: {request.strategy}") from error
    result = builder(dataset, request)
    if len(result.folds) != request.n_folds:
        raise AssertionError("Split strategy did not return the requested number of folds")
    return result

