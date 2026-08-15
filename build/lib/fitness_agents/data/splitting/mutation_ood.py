from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from fitness_agents.data.canonical import CanonicalDataset

from .contracts import FoldSplit, SplitRequest, SplitResult
from .hashing import effective_salt, stable_digest
from .utils import decode_tokens


def _token_group(token: str) -> tuple[str, int]:
    import json

    decoded = json.loads(token)
    return str(decoded[1]), int(decoded[2])


def _assign_identity_shards(
    features: pd.DataFrame, request: SplitRequest, salt: bytes
) -> dict[str, int]:
    support: Counter[str] = Counter()
    for value in features["mutation_tokens"]:
        support.update(decode_tokens(value))
    by_group: dict[tuple[str, int], list[str]] = defaultdict(list)
    for token in support:
        by_group[_token_group(token)].append(token)
    insufficient = {group: len(tokens) for group, tokens in by_group.items() if len(tokens) < request.n_folds}
    if insufficient:
        preview = ", ".join(f"{key}={value}" for key, value in sorted(insufficient.items())[:8])
        raise ValueError(
            "Mutation-identity OOD requires at least n_folds identities per component/position; "
            f"insufficient groups: {preview}"
        )
    assignment: dict[str, int] = {}
    bin_support = [0] * request.n_folds
    bin_count = [0] * request.n_folds
    for group, tokens in sorted(by_group.items()):
        ordered = sorted(
            tokens,
            key=lambda token: (
                -support[token],
                stable_digest(salt, "mutation_identity", *group, token),
                token,
            ),
        )
        group_counts = [0] * request.n_folds
        for token in ordered:
            shard = min(
                range(request.n_folds),
                key=lambda index: (
                    group_counts[index],
                    bin_support[index],
                    bin_count[index],
                    stable_digest(salt, "mutation_bin", *group, token, index),
                ),
            )
            assignment[token] = shard
            group_counts[shard] += 1
            bin_support[shard] += support[token]
            bin_count[shard] += 1
    return assignment


def build_mutation_ood(dataset: CanonicalDataset, request: SplitRequest) -> SplitResult:
    features = dataset.features.copy()
    salt = effective_salt(request.public_salt, request.seed)
    identity_shards = _assign_identity_shards(features, request, salt)
    policy = str(request.options.get("mutation_row_policy", "contains_unseen"))
    if policy not in {"contains_unseen", "pure_group_only"}:
        raise ValueError(f"Unknown mutation_row_policy: {policy}")
    folds: list[FoldSplit] = []
    for fold_index in range(request.n_folds):
        test_shard = fold_index
        validation_shard = (fold_index + 1) % request.n_folds
        roles: list[str] = []
        for value in features["mutation_tokens"]:
            tokens = decode_tokens(value)
            shards = {identity_shards[token] for token in tokens}
            if not shards:
                role = "train_observed"
            elif policy == "pure_group_only":
                if shards == {test_shard}:
                    role = "final_test"
                elif shards == {validation_shard}:
                    role = "benchmark_validation"
                elif test_shard not in shards and validation_shard not in shards:
                    role = "train_observed"
                else:
                    role = "quarantine"
            elif test_shard in shards and validation_shard in shards:
                role = "quarantine"
            elif test_shard in shards:
                role = "final_test"
            elif validation_shard in shards:
                role = "benchmark_validation"
            else:
                role = "train_observed"
            roles.append(role)
        assignments = features.loc[:, ["variant_id"]].copy()
        assignments["fold_index"] = fold_index
        assignments["split_role"] = roles
        assignments["queryable"] = False
        assignments["label_visibility"] = assignments["split_role"].map(
            {
                "train_observed": "agent",
                "benchmark_validation": "controller",
                "final_test": "evaluator",
                "quarantine": "none",
            }
        )
        counts = assignments["split_role"].value_counts()
        required = {"train_observed", "benchmark_validation", "final_test"}
        if missing := required.difference(counts.index):
            raise ValueError(f"Mutation OOD fold {fold_index} has empty roles: {sorted(missing)}")
        folds.append(
            FoldSplit(
                fold_index,
                assignments,
                {
                    "mutation_row_policy": policy,
                    "test_identity_count": sum(
                        shard == test_shard for shard in identity_shards.values()
                    ),
                    "validation_identity_count": sum(
                        shard == validation_shard for shard in identity_shards.values()
                    ),
                },
            )
        )
    return SplitResult(
        request.strategy,
        tuple(folds),
        {
            "label_conditioned_population": False,
            "identity_shards": identity_shards,
            "mutation_row_policy": policy,
        },
    )

