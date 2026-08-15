from __future__ import annotations

from dataclasses import replace

import numpy as np
from common import (
    ensure,
    load_config,
    make_evidence,
    make_predictions,
    parse_args,
    resolve_output,
    variant_grid,
    write_result,
)

from fitness_agents.acquisition import create_policy
from fitness_agents.contracts.schemas import CampaignState, Evidence, Hypothesis, Variant
from fitness_agents.mutation import create_candidate_generator
from fitness_agents.mutation.conflicts import (
    ResidueConflictDetector,
    SequenceConflictDetector,
    detect_pairwise_epistasis,
)


def main() -> None:
    args = parse_args("configs/module_tests/design_acquisition.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    candidates = variant_grid()[:12]
    predictions = make_predictions(candidates)
    prediction_map = {item.variant_id: item for item in predictions}
    evidence = make_evidence(candidates)
    evidence[candidates[1].variant_id].append(
        Evidence(
            "ev:opposing:1",
            candidates[1].variant_id,
            "opposing_channel",
            "Opposing source for polarity-conflict coverage.",
            0.6,
            "module-test:source-c",
            0.7,
            1,
        )
    )
    hypothesis = Hypothesis(
        "hyp:design-module",
        "Test A/W/A/I-enriched candidates while preserving diversity.",
        {39: ("A",), 40: ("W",), 41: ("A",), 54: ("I",)},
        tuple(item.evidence_id for item in evidence[candidates[0].variant_id]),
        "Selected variants should exceed the visible baseline.",
        "Reject if selected median does not exceed the baseline median.",
    )
    state = CampaignState("module-design", "knowledge_agent", int(config["seed"]), round_id=1)

    generator_results: dict[str, list[str]] = {}
    for mode in ("random", "fitness_direct", "llm_agent", "knowledge_agent"):
        generated = create_candidate_generator(mode).generate(
            candidates,
            state,
            hypothesis,
            evidence,
            int(config["candidate_limit"]),
        )
        ensure(generated, f"{mode} candidate generator returned no candidates")
        ensure(len({item.variant_id for item in generated}) == len(generated), "Duplicate generation")
        generator_results[mode] = [item.variant for item in generated]

    knowledge_scores = {
        variant_id: float(np.mean([item.score for item in bundle]))
        for variant_id, bundle in evidence.items()
    }
    policy_results: dict[str, object] = {}
    for offset, policy_name in enumerate(("random", "greedy", "ucb", "thompson", "ts")):
        policy = create_policy(
            policy_name,
            beta=float(config["ucb_beta"]),
            knowledge_weight=float(config["knowledge_weight"]),
        )
        scores = policy.score(
            predictions,
            knowledge_scores,
            np.random.default_rng(int(config["seed"]) + offset),
        )
        selected = policy.select(
            candidates,
            predictions,
            scores,
            int(config["batch_budget"]),
            float(config["diversity_lambda"]),
        )
        ensure(len(selected) == int(config["batch_budget"]), f"{policy_name} budget mismatch")
        policy_results[policy_name] = {
            "selected_ids": selected,
            "selected_codes": [
                next(item.variant for item in candidates if item.variant_id == identifier)
                for identifier in selected
            ],
            "top_score": max(scores.values()),
        }

    valid_residue_conflicts = ResidueConflictDetector().detect(
        candidates[:3], wild_type_sites="VDGV", mutable_positions=(39, 40, 41, 54)
    )
    ensure(not valid_residue_conflicts, "Valid variants failed residue validation")
    invalid = Variant(
        "invalid",
        "VDGX",
        "VDGX",
        "D40E;D40W;BAD",
        0,
        "oracle_pool",
    )
    invalid_residue_conflicts = ResidueConflictDetector().detect(
        [invalid], wild_type_sites="VDGV", mutable_positions=(39, 40, 41, 54)
    )
    ensure(invalid_residue_conflicts, "Invalid residue edits were not detected")

    selected_variants = candidates[:3]
    warned_prediction = replace(
        predictions[1],
        ood_score=0.95,
        component_scores={"model_a": -0.8, "model_b": 0.9},
    )
    warned_predictions = dict(prediction_map)
    warned_predictions[candidates[1].variant_id] = warned_prediction
    thresholds = config["conflict_thresholds"]
    sequence_conflicts = SequenceConflictDetector(
        ood_warning_threshold=float(thresholds["ood_warning"]),
        model_disagreement_threshold=float(thresholds["model_disagreement"]),
        min_batch_distance=int(thresholds["min_batch_distance"]),
    ).detect(
        selected_variants,
        predictions=warned_predictions,
        evidence=evidence,
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids={item.variant_id for item in candidates},
        expected_batch_size=len(selected_variants),
    )
    sequence_codes = {item.code for item in sequence_conflicts}
    ensure("HIGH_OOD" in sequence_codes, "OOD warning was not emitted")
    ensure("MODEL_DISAGREEMENT" in sequence_codes, "Model-disagreement warning was not emitted")
    ensure("EVIDENCE_POLARITY_CONFLICT" in sequence_codes, "Evidence conflict was not emitted")

    missing_prediction_conflicts = SequenceConflictDetector().detect(
        [candidates[0]],
        predictions={},
        evidence={},
        revealed_ids=set(),
        pending_ids=set(),
        allowed_ids={candidates[0].variant_id},
        expected_batch_size=1,
    )
    ensure(
        any(item.code == "MISSING_PREDICTION" and item.hard for item in missing_prediction_conflicts),
        "Missing predictions were not treated as a hard conflict",
    )

    rng = np.random.default_rng(int(config["seed"]))
    complete_epistasis = detect_pairwise_epistasis(
        fitness_scale="raw_assay",
        wt_samples=rng.normal(0.0, 0.05, 200),
        single_a_samples=rng.normal(0.3, 0.05, 200),
        single_b_samples=rng.normal(0.2, 0.05, 200),
        double_samples=rng.normal(-0.1, 0.05, 200),
    )
    missing_epistasis = detect_pairwise_epistasis(
        fitness_scale="raw_assay",
        wt_samples=[0.0],
        single_a_samples=[0.3],
        single_b_samples=None,
        double_samples=[0.4],
    )
    ensure(complete_epistasis.status == "DETECTED", "Sign epistasis was not detected")
    ensure(missing_epistasis.status == "UNKNOWN", "Missing constituent was not marked unknown")

    write_result(
        output,
        "design_acquisition",
        {
            "config": config["_config_path"],
            "candidate_generators": generator_results,
            "acquisition_policies": policy_results,
            "invalid_residue_conflict_codes": sorted({item.code for item in invalid_residue_conflicts}),
            "sequence_conflict_codes": sorted(sequence_codes),
            "epistasis": {
                "complete": complete_epistasis,
                "missing_constituent": missing_epistasis,
            },
        },
    )


if __name__ == "__main__":
    main()

