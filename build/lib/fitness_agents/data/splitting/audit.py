from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from fitness_agents.data.canonical import TARGET_PROXY_COLUMNS, CanonicalDataset

from .contracts import SplitResult
from .utils import decode_tokens


def _tokens_for_ids(features: pd.DataFrame, variant_ids: set[str]) -> set[str]:
    tokens: set[str] = set()
    for value in features.loc[features["variant_id"].isin(variant_ids), "mutation_tokens"]:
        tokens.update(decode_tokens(value))
    return tokens


def audit_split(dataset: CanonicalDataset, result: SplitResult) -> dict[str, Any]:
    expected_ids = set(dataset.features["variant_id"].astype(str))
    fold_reports: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    for fold in result.folds:
        assignments = fold.assignments
        if assignments["variant_id"].duplicated().any():
            hard_failures.append(f"fold_{fold.fold_index:02d}: duplicate variant assignment")
        actual_ids = set(assignments["variant_id"].astype(str))
        if not actual_ids.issubset(expected_ids):
            hard_failures.append(f"fold_{fold.fold_index:02d}: unknown variant IDs")
        by_role = {
            str(key): int(value)
            for key, value in assignments["split_role"].value_counts().sort_index().items()
        }
        queryable_roles = set(assignments.loc[assignments["queryable"], "split_role"])
        if queryable_roles.difference({"candidate_pool"}):
            hard_failures.append(
                f"fold_{fold.fold_index:02d}: non-candidate role marked queryable"
            )
        fold_reports.append(
            {
                "fold_index": fold.fold_index,
                "rows": len(assignments),
                "role_counts": by_role,
                "queryable_count": int(assignments["queryable"].sum()),
            }
        )

    if result.strategy == "al96_closed_loop":
        initial_sets = [
            set(fold.assignments.loc[fold.assignments["split_role"] == "initial_observed", "variant_id"])
            for fold in result.folds
        ]
        if any(value != initial_sets[0] for value in initial_sets[1:]):
            hard_failures.append("AL initial set differs across folds")
        tests = [
            set(fold.assignments.loc[fold.assignments["split_role"] == "final_test", "variant_id"])
            for fold in result.folds
        ]
        for left in range(len(tests)):
            for right in range(left + 1, len(tests)):
                if tests[left].intersection(tests[right]):
                    hard_failures.append("AL outer final tests are not disjoint")
                    break
        test_depth_min = int(result.folds[0].metadata["test_depth_min"])
        expected_outer = set(
            dataset.features.loc[
                dataset.features["mutation_count"] >= test_depth_min, "variant_id"
            ].astype(str)
        )
        if set().union(*tests) != expected_outer:
            hard_failures.append("AL outer final tests do not cover the deployable universe")
    elif result.strategy == "flip_static_ood":
        tests = [
            tuple(sorted(fold.assignments.loc[fold.assignments["split_role"] == "final_test", "variant_id"]))
            for fold in result.folds
        ]
        if any(value != tests[0] for value in tests[1:]):
            hard_failures.append("FLIP-compatible OOD test differs across folds")
        validation_counts: Counter[str] = Counter()
        for fold in result.folds:
            validation_counts.update(
                fold.assignments.loc[
                    fold.assignments["split_role"] == "benchmark_validation", "variant_id"
                ].astype(str)
            )
        if any(count != 1 for count in validation_counts.values()):
            hard_failures.append("FLIP training-domain validation rows do not rotate exactly once")
        expected_validation = set(
            result.folds[0].assignments.loc[
                result.folds[0].assignments["split_role"].isin(
                    ["train_observed", "benchmark_validation"]
                ),
                "variant_id",
            ].astype(str)
        )
        anchors = set(
            dataset.features.loc[
                (dataset.features["variant_id"].isin(expected_validation))
                & (dataset.features["mutation_count"] == 0),
                "variant_id",
            ].astype(str)
        )
        if set(validation_counts) != expected_validation.difference(anchors):
            hard_failures.append("FLIP validation rotation does not cover its full eligible domain")
    elif result.strategy == "mutation_identity_ood":
        for fold in result.folds:
            assignments = fold.assignments
            train_ids = set(
                assignments.loc[assignments["split_role"] == "train_observed", "variant_id"]
            )
            validation_ids = set(
                assignments.loc[
                    assignments["split_role"] == "benchmark_validation", "variant_id"
                ]
            )
            test_ids = set(
                assignments.loc[assignments["split_role"] == "final_test", "variant_id"]
            )
            train_tokens = _tokens_for_ids(dataset.features, train_ids)
            validation_tokens = _tokens_for_ids(dataset.features, validation_ids)
            test_tokens = _tokens_for_ids(dataset.features, test_ids)
            identity_shards = result.metadata["identity_shards"]
            test_shard = fold.fold_index
            validation_shard = (fold.fold_index + 1) % len(result.folds)
            heldout_test = {
                token for token, shard in identity_shards.items() if shard == test_shard
            }
            heldout_validation = {
                token for token, shard in identity_shards.items() if shard == validation_shard
            }
            if heldout_test.intersection(train_tokens | validation_tokens):
                hard_failures.append(
                    f"fold_{fold.fold_index:02d}: test identities leaked to train/validation"
                )
            if heldout_validation.intersection(train_tokens):
                hard_failures.append(
                    f"fold_{fold.fold_index:02d}: validation identities leaked to train"
                )
            if not heldout_test.intersection(test_tokens):
                hard_failures.append(
                    f"fold_{fold.fold_index:02d}: final test lacks held-out test identities"
                )

    hidden = TARGET_PROXY_COLUMNS.intersection(
        column.lower() for column in dataset.features.columns
    )
    if hidden:
        hard_failures.append(f"canonical features expose target proxies: {sorted(hidden)}")
    return {
        "valid": not hard_failures,
        "strategy": result.strategy,
        "hard_failures": hard_failures,
        "folds": fold_reports,
        "label_conditioned_population": bool(
            result.metadata.get("label_conditioned_population", False)
        ),
    }


def assert_audit_passed(report: dict[str, Any]) -> None:
    if not report["valid"]:
        raise ValueError("Split leakage audit failed: " + "; ".join(report["hard_failures"]))
