from __future__ import annotations

import pandas as pd

from fitness_agents.data.canonical import CanonicalDataset

from .contracts import FoldSplit, SplitRequest, SplitResult
from .hashing import effective_salt, stable_order
from .utils import stratified_shards

DEPTH_RULES = {"one_vs_rest": 1, "two_vs_rest": 2, "three_vs_rest": 3}


def _population(
    dataset: CanonicalDataset, request: SplitRequest, salt: bytes
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    mode = str(request.options.get("population", "full"))
    if mode == "full":
        return dataset.features.copy(), None, False
    if mode != "flip_keep":
        raise ValueError(f"Unknown population: {mode}")
    if not request.allow_label_dependent_membership:
        raise PermissionError("flip_keep requires --allow-label-dependent-membership")
    merged = dataset.features.merge(dataset.labels, on="variant_id", validate="one_to_one")
    high = merged.loc[merged["target"] > 0.5]
    low = merged.loc[merged["target"] <= 0.5]
    low_count = min(len(low), len(high) // 2)
    low_ids = stable_order(
        low["variant_id"].astype(str).tolist(), salt, "flip_keep_low"
    )[:low_count]
    selected = set(high["variant_id"].astype(str)) | set(low_ids)
    return dataset.features.loc[dataset.features["variant_id"].isin(selected)].copy(), merged, True


def build_flip_ood(dataset: CanonicalDataset, request: SplitRequest) -> SplitResult:
    salt = effective_salt(request.public_salt, request.seed)
    features, merged, label_conditioned = _population(dataset, request, salt)
    rule = str(request.options.get("ood_rule", "two_vs_rest"))
    if rule in DEPTH_RULES:
        threshold = DEPTH_RULES[rule]
        train_mask = features["mutation_count"] <= threshold
    elif rule == "low_vs_high":
        if not request.allow_label_dependent_membership:
            raise PermissionError("low_vs_high requires --allow-label-dependent-membership")
        if merged is None:
            merged = features.merge(dataset.labels, on="variant_id", validate="one_to_one")
        wt_targets = merged.loc[merged["mutation_count"] == 0, "target"]
        if wt_targets.empty:
            raise ValueError("low_vs_high requires at least one WT/zero-mutation row")
        wt_target = float(wt_targets.mean())
        target_by_id = merged.set_index("variant_id")["target"]
        train_mask = features["variant_id"].map(target_by_id).lt(wt_target)
        label_conditioned = True
    else:
        raise ValueError(f"Unknown OOD rule: {rule}")
    train_domain = features.loc[train_mask].copy()
    test_domain = features.loc[~train_mask].copy()
    if train_domain.empty or test_domain.empty:
        raise ValueError("OOD rule must produce non-empty train and test domains")
    anchors = set(
        train_domain.loc[train_domain["mutation_count"] == 0, "variant_id"].astype(str)
    )
    eligible = train_domain.loc[~train_domain["variant_id"].isin(anchors)]
    if len(eligible) < request.n_folds:
        raise ValueError("Training domain is too small for the requested validation folds")
    shards = stratified_shards(
        eligible,
        n_folds=request.n_folds,
        salt=salt,
        namespace=f"flip_ood:{rule}",
        strata=("mutation_count", "backbone_id"),
    )
    test_ids = set(test_domain["variant_id"].astype(str))
    folds: list[FoldSplit] = []
    for fold_index in range(request.n_folds):
        validation_ids = {key for key, value in shards.items() if value == fold_index}
        assignments = features.loc[:, ["variant_id"]].copy()
        assignments["fold_index"] = fold_index
        assignments["split_role"] = "train_observed"
        assignments.loc[
            assignments["variant_id"].isin(validation_ids), "split_role"
        ] = "benchmark_validation"
        assignments.loc[assignments["variant_id"].isin(test_ids), "split_role"] = "final_test"
        assignments["queryable"] = False
        assignments["label_visibility"] = assignments["split_role"].map(
            {
                "train_observed": "agent",
                "benchmark_validation": "controller",
                "final_test": "evaluator",
            }
        )
        folds.append(FoldSplit(fold_index, assignments, {"ood_rule": rule}))
    return SplitResult(
        request.strategy,
        tuple(folds),
        {
            "ood_rule": rule,
            "population": request.options.get("population", "full"),
            "label_conditioned_population": label_conditioned,
            "eligible_for_closed_loop": False,
        },
    )

