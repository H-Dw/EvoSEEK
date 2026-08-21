from __future__ import annotations

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.loader import load_fold_bundle
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.audit import audit_split
from fitness_agents.data.splitting.writer import write_split


def _assignment_signature(result) -> list[list[tuple[str, str]]]:
    return [
        list(
            fold.assignments.sort_values("variant_id")
            .loc[:, ["variant_id", "split_role"]]
            .itertuples(index=False, name=None)
        )
        for fold in result.folds
    ]


def test_al_closed_loop_is_fivefold_label_blind_and_covers_outer_test(synthetic_landscape):
    dataset = synthetic_landscape
    request = SplitRequest(
        "al96_closed_loop",
        options={
            "initial_budget": 26,
            "test_depth_min": 3,
            "validation_size": 20,
            "validation_strata": ("mutation_count", "backbone_id"),
        },
    )
    result = build_split(dataset, request)
    assert len(result.folds) == 5
    initial = result.folds[0].assignments.query("split_role == 'initial_observed'")
    depth = dataset.features.set_index("variant_id")["mutation_count"]
    assert initial["variant_id"].map(depth).value_counts().to_dict() == {1: 20, 2: 5, 0: 1}
    tests = [
        set(fold.assignments.query("split_role == 'final_test'")["variant_id"])
        for fold in result.folds
    ]
    assert not any(tests[left] & tests[right] for left in range(5) for right in range(left + 1, 5))
    expected = set(dataset.features.loc[dataset.features["mutation_count"] >= 3, "variant_id"])
    assert set().union(*tests) == expected

    permuted = CanonicalDataset(
        dataset.features.sample(frac=1, random_state=4).reset_index(drop=True),
        dataset.labels.assign(target=dataset.labels["target"].sample(frac=1, random_state=7).to_numpy()),
        dataset.spec,
        dataset.source_sha256,
    )
    assert _assignment_signature(build_split(permuted, request)) == _assignment_signature(result)


def test_flip_static_ood_has_fixed_test_and_rotating_validation(synthetic_landscape):
    dataset = synthetic_landscape
    request = SplitRequest(
        "flip_static_ood", options={"ood_rule": "two_vs_rest", "population": "full"}
    )
    result = build_split(dataset, request)
    tests = [
        set(fold.assignments.query("split_role == 'final_test'")["variant_id"])
        for fold in result.folds
    ]
    assert all(test == tests[0] for test in tests[1:])
    expected = set(dataset.features.loc[dataset.features["mutation_count"] > 2, "variant_id"])
    assert tests[0] == expected
    validation_counts: dict[str, int] = {}
    for fold in result.folds:
        for variant_id in fold.assignments.query("split_role == 'benchmark_validation'")["variant_id"]:
            validation_counts[variant_id] = validation_counts.get(variant_id, 0) + 1
    non_wt_train = set(
        dataset.features.loc[dataset.features["mutation_count"].between(1, 2), "variant_id"]
    )
    assert set(validation_counts) == non_wt_train
    assert set(validation_counts.values()) == {1}
    assert audit_split(dataset, result)["valid"] is True


def test_mutation_identity_ood_quarantines_mixed_rows_without_leakage(
    synthetic_landscape,
):
    dataset = synthetic_landscape
    result = build_split(dataset, SplitRequest("mutation_identity_ood"))
    assert len(result.folds) == 5
    assert all(
        "quarantine" in set(fold.assignments["split_role"]) for fold in result.folds
    )
    report = audit_split(dataset, result)
    assert report["valid"] is True, report["hard_failures"]


def test_writer_outputs_capability_views_and_loader_enforces_them(
    tmp_path, synthetic_landscape
):
    dataset = synthetic_landscape
    request = SplitRequest(
        "al96_closed_loop",
        protocol_version="test-v1",
        options={
            "initial_budget": 26,
            "test_depth_min": 3,
            "validation_size": 20,
            "validation_strata": ("mutation_count", "backbone_id"),
        },
    )
    result = build_split(dataset, request)
    output = write_split(dataset, request, result, tmp_path / "outputs")
    agent = load_fold_bundle(output, 0, "agent")
    assert agent.observed is not None and "target" in agent.observed
    assert agent.candidates is not None and "target" not in agent.candidates
    assert agent.validation is None and agent.queryable_labels is None
    assert agent.final_inputs is None and agent.final_labels is None

    oracle = load_fold_bundle(output, 0, "oracle")
    assert oracle.queryable_labels is not None
    assert oracle.final_labels is None
    evaluator = load_fold_bundle(output, 0, "evaluator")
    assert evaluator.final_labels is not None
    assert evaluator.queryable_labels is None
    assert len(list(output.glob("fold_*"))) == 5


def test_al_closed_loop_initial_budget_one_keeps_only_wild_type(synthetic_landscape):
    dataset = synthetic_landscape
    options = {
        "test_depth_min": 3,
        "validation_size": 20,
        "validation_strata": ("mutation_count", "backbone_id"),
    }
    base = build_split(dataset, SplitRequest("al96_closed_loop", options={**options, "initial_budget": 26}))
    cold = build_split(dataset, SplitRequest("al96_closed_loop", options={**options, "initial_budget": 1}))

    depth = dataset.features.set_index("variant_id")["mutation_count"]
    single_count = int((depth == 1).sum())
    for fold in cold.folds:
        initial = fold.assignments.query("split_role == 'initial_observed'")
        assert len(initial) == 1
        assert depth[initial["variant_id"].iloc[0]] == 0
        pool_depths = fold.assignments.query("split_role == 'candidate_pool'")["variant_id"].map(depth)
        # A WT-only prior leaves every single mutant in the queryable pool.
        assert int((pool_depths == 1).sum()) == single_count
    # The smaller prior never touches the HD>=3 deployable population, so the
    # test and validation composition matches the full-prior build exactly.
    for cold_fold, base_fold in zip(cold.folds, base.folds, strict=True):
        for role in ("final_test", "benchmark_validation"):
            cold_ids = set(cold_fold.assignments.query("split_role == @role")["variant_id"])
            base_ids = set(base_fold.assignments.query("split_role == @role")["variant_id"])
            assert cold_ids == base_ids
    assert audit_split(dataset, cold)["valid"] is True
