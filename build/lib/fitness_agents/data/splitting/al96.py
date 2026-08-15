from __future__ import annotations

from fitness_agents.data.canonical import CanonicalDataset

from .contracts import FoldSplit, SplitRequest, SplitResult
from .hashing import effective_salt, stable_order
from .initialization import select_low_order_coverage
from .utils import allocate_counts, stratified_shards


def build_al96(dataset: CanonicalDataset, request: SplitRequest) -> SplitResult:
    features = dataset.features.copy()
    salt = effective_salt(request.public_salt, request.seed)
    budget = int(request.options.get("initial_budget", 96))
    test_depth_min = int(request.options.get("test_depth_min", 3))
    validation_size = int(request.options.get("validation_size", 384))
    strata = tuple(request.options.get("validation_strata", ("mutation_count", "backbone_id")))
    initial_ids = select_low_order_coverage(features, budget=budget, salt=salt)
    deployable = features.loc[
        (~features["variant_id"].isin(initial_ids))
        & (features["mutation_count"] >= test_depth_min)
    ]
    if deployable.empty:
        raise ValueError("AL closed-loop requires non-empty deployable high-order variants")
    outer = stratified_shards(
        deployable,
        n_folds=request.n_folds,
        salt=salt,
        namespace="al96_outer_test",
        strata=("mutation_count", "backbone_id"),
    )
    folds: list[FoldSplit] = []
    for fold_index in range(request.n_folds):
        test_ids = {variant_id for variant_id, shard in outer.items() if shard == fold_index}
        validation_eligible = features.loc[
            (~features["variant_id"].isin(initial_ids | test_ids))
            & (features["mutation_count"] >= test_depth_min)
        ]
        if validation_size >= len(validation_eligible):
            raise ValueError("validation_size must leave at least one queryable candidate")
        group_key: str | list[str] = strata[0] if len(strata) == 1 else list(strata)
        groups = {
            key: len(group) for key, group in validation_eligible.groupby(group_key, sort=True)
        }
        allocation = allocate_counts(groups, validation_size)
        validation_ids: set[str] = set()
        for key, group in validation_eligible.groupby(group_key, sort=True):
            key_tuple = key if isinstance(key, tuple) else (key,)
            ordered = stable_order(
                group["variant_id"].astype(str).tolist(),
                salt,
                "al96_validation",
                fold_index,
                *key_tuple,
            )
            validation_ids.update(ordered[: allocation[key]])
        assignments = features.loc[:, ["variant_id"]].copy()
        assignments["fold_index"] = fold_index
        assignments["split_role"] = "candidate_pool"
        assignments.loc[assignments["variant_id"].isin(initial_ids), "split_role"] = (
            "initial_observed"
        )
        assignments.loc[assignments["variant_id"].isin(validation_ids), "split_role"] = (
            "benchmark_validation"
        )
        assignments.loc[assignments["variant_id"].isin(test_ids), "split_role"] = "final_test"
        assignments["queryable"] = assignments["split_role"].eq("candidate_pool")
        assignments["label_visibility"] = assignments["split_role"].map(
            {
                "initial_observed": "agent",
                "candidate_pool": "oracle_on_query",
                "benchmark_validation": "controller",
                "final_test": "evaluator",
            }
        )
        folds.append(
            FoldSplit(
                fold_index,
                assignments,
                {
                    "test_depth_min": test_depth_min,
                    "initial_budget": budget,
                    "validation_size": validation_size,
                },
            )
        )
    return SplitResult(
        request.strategy,
        tuple(folds),
        {"label_conditioned_population": False, "initial_ids_shared_across_folds": True},
    )

