"""Isolated live-API smoke test for hypothesis-level ReThink.

The production experiment config is loaded and asserted to remain Kermut, but the
CampaignRunner is not created. Dry predictions come from an independent one-hot
fixture so this smoke test cannot activate Kermut.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fitness_agents.agents.rethink import NativeReThinkClient, build_round_evidence_digest
from fitness_agents.config import ModelConfig, load_experiment_config
from fitness_agents.contracts.agent_io import HypothesisReflectionContextInput
from fitness_agents.data import load_dataset_bundle
from fitness_agents.models import create_predictor
from fitness_agents.utils.artifacts import JsonArtifactWriter
from fitness_agents.utils.progress import (
    bind_progress,
    configure_progress_logging,
    reset_progress,
)

REQUIRED_DIMENSIONS = {
    "measured_function",
    "edit_level_direction",
    "sequence_interaction_context",
    "structural_context",
    "evolutionary_context",
    "physicochemical_context",
    "feasibility_developability",
    "uncertainty_domain_shift",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--production-config",
        default="configs/experiments/knowledge_agent.yaml",
    )
    parser.add_argument("--fixture-config", default="configs/model/baseline.yaml")
    parser.add_argument("--output-root", default="artifacts/rethink_real_api")
    return parser.parse_args()


def _fixture_context(root: Path, production, fixture_config: ModelConfig):
    bundle = load_dataset_bundle(
        production.task.public_data_path,
        production.task.oracle_data_path,
    )
    fixture = create_predictor(fixture_config, seed=20260821)
    fixture.fit(
        bundle.initial_variants,
        bundle.initial_observations,
        bundle.validation_variants,
        bundle.validation_observations,
    )
    candidates = tuple(bundle.oracle_pool[:4])
    predictions = tuple(fixture.predict(candidates))
    if len(predictions) != 4 or not all(
        np.isfinite(item.fitness_mean) and item.fitness_std > 0
        for item in predictions
    ):
        raise AssertionError("one-hot fixture did not produce four finite predictions")

    oracle = pd.read_csv(production.task.oracle_data_path).set_index("variant_id")
    visible_baseline = float(
        np.median([item.fitness for item in bundle.initial_observations])
    )
    wet = {
        item.variant_id: float(oracle.loc[item.variant_id, "fitness"])
        for item in candidates
    }
    target_ids = tuple(item.variant_id for item in candidates[:2])
    control_ids = tuple(item.variant_id for item in candidates[2:])
    target_mean = float(np.mean([wet[item] for item in target_ids]))
    control_mean = float(np.mean([wet[item] for item in control_ids]))
    effect = target_mean - control_mean
    status = "SUPPORTED" if effect > 0 else "CONTRADICTED"
    signal = "SUPPORT" if status == "SUPPORTED" else "CONTRADICT"
    prediction_by_id = {item.variant_id: item for item in predictions}
    observation_cards = []
    for index, variant in enumerate(candidates):
        prediction = prediction_by_id[variant.variant_id]
        observation_cards.append(
            {
                "variant_id": variant.variant_id,
                "mutation_notation": variant.mutation_notation,
                "evidence_ids": [],
                "wet_value": wet[variant.variant_id],
                "dry_validations": [
                    {
                        "value": prediction.fitness_mean,
                        "uncertainty": prediction.fitness_std,
                        "ood_score": prediction.ood_score,
                        "model_version": prediction.model_version,
                        "source_kind": "dry_validation",
                        "decision_eligible": False,
                        "calibration_status": "uncalibrated",
                        "prediction_status": "evaluated",
                    }
                ],
                "intent_arm": (
                    "hypothesis_target" if index < 2 else "matched_control"
                ),
                "matched_to": (
                    candidates[index + 2].variant_id
                    if index < 2
                    else candidates[index - 2].variant_id
                ),
                "allow_hypothesis_mismatch": False,
                "falsification_role": "target" if index < 2 else "comparator",
            }
        )
    criterion_receipts = [
        {
            "criterion_id": "criterion:onehot-live:target_vs_control",
            "signal": signal,
            "metric_value": target_mean,
            "comparator_value": control_mean,
            "effect_size": effect,
            "observation_ids": [item.variant_id for item in candidates],
            "qc_status": "ok",
            "detector_name": "fixture_target_control_mean_delta",
            "detector_version": "1.0.0",
            "reason_code": (
                "target_above_control" if effect > 0 else "target_not_above_control"
            ),
        }
    ]
    digest = build_round_evidence_digest(
        observation_cards,
        visible_baseline=visible_baseline,
        optimization_direction="higher_is_better",
        criterion_receipts=criterion_receipts,
    )
    context = HypothesisReflectionContextInput.model_validate(
        {
            "run_id": "rethink-live-onehot-20260821",
            "round_id": 1,
            "visible_baseline": visible_baseline,
            "baseline_receipt": {
                "value": visible_baseline,
                "statistic": "pre_round_visible_median",
                "source": "revealed_observations_before_current_round",
            },
            "measurement_contract": {
                "assay_id": production.task.assay_id,
                "fitness_scale": production.task.fitness_scale,
                "optimization_direction": "higher_is_better",
            },
            "approved_hypothesis": {
                "hypothesis_id": "H-onehot-live-01",
                "statement": (
                    "The first one-hot fixture arm has higher measured fitness "
                    "than its matched-control arm."
                ),
                "expected_outcome": (
                    "The target-arm mean exceeds the matched-control mean."
                ),
                "falsification_criterion": (
                    "Contradict the hypothesis when the target-arm mean does not "
                    "exceed the matched-control mean."
                ),
                "evidence_ids": [],
            },
            "final_critic_decision": {
                "decision_id": "D-onehot-live-01",
                "verdict": "APPROVE",
                "summary": (
                    "The fixture defines explicit target and matched-control arms "
                    "with a deterministic criterion."
                ),
                "cited_evidence_ids": [],
            },
            "hypothesis_assessment": {
                "assessment_id": "AS-onehot-live-01",
                "hypothesis_id": "H-onehot-live-01",
                "falsification_spec_id": "FS-onehot-live-01",
                "status": status,
                "criterion_results": criterion_receipts,
                "observation_ids": [item.variant_id for item in candidates],
                "decisive_criterion_ids": [
                    "criterion:onehot-live:target_vs_control"
                ],
                "unresolved_criterion_ids": [],
                "evaluator_version": "fixture_target_control_mean_delta.v1",
            },
            "falsification_spec": {
                "spec_id": "FS-onehot-live-01",
                "hypothesis_id": "H-onehot-live-01",
                "version": "1.0.0",
                "reduction_policy": "single_primary_criterion",
                "criteria": [
                    {
                        "criterion_id": "criterion:onehot-live:target_vs_control",
                        "detector_name": "fixture_target_control_mean_delta",
                        "metric": "mean_fitness_delta",
                        "expected_direction": "greater",
                        "target_variant_ids": list(target_ids),
                        "comparator_variant_ids": list(control_ids),
                        "min_observations": 4,
                        "missing_data_policy": "INCONCLUSIVE",
                        "primary": True,
                    }
                ],
            },
            "round_evidence_digest": digest.model_dump(mode="json"),
        }
    )
    fixture_receipt = {
        "sample_count": len(candidates),
        "visible_baseline": visible_baseline,
        "target_mean": target_mean,
        "control_mean": control_mean,
        "effect_size": effect,
        "assessment_status": status,
        "variant_ids": [item.variant_id for item in candidates],
        "predictions": [asdict(item) for item in predictions],
        "wet_values": wet,
    }
    return context, fixture_receipt


def main() -> None:
    args = _arguments()
    root = Path(__file__).resolve().parents[2]
    production = load_experiment_config(root / args.production_config)
    if production.model.name != "kermut":
        raise AssertionError("production experiment config must remain Kermut")
    fixture_config = ModelConfig(
        **yaml.safe_load((root / args.fixture_config).read_text(encoding="utf-8"))
    )
    if fixture_config.name != "onehot_heterogeneous_ensemble":
        raise AssertionError("live ReThink smoke test requires an isolated one-hot fixture")
    context, fixture_receipt = _fixture_context(root, production, fixture_config)
    llm = production.llm
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    writer = JsonArtifactWriter(root / args.output_root, stamp)
    output = writer.run_dir
    configure_progress_logging()
    writer.write_json("rethink_input.json", context)
    writer.write_json(
        "preflight_receipt.json",
        {
            "production_experiment_config": args.production_config,
            "production_runner_invoked": False,
            "production_model_config": production.model.name,
            "fixture_model_config": args.fixture_config,
            "fixture_model_name": fixture_config.name,
            "kermut_activated": False,
            "provider": llm.provider,
            "model": llm.model,
            "prompt_profile": llm.profile,
            **fixture_receipt,
        },
    )
    client = NativeReThinkClient(
        model=llm.model,
        base_url=llm.base_url,
        provider=llm.provider,
        temperature=llm.temperature,
        max_tokens=llm.rethink_max_tokens,
        render_max_tokens=llm.rethink_render_max_tokens,
        reasoning_effort=llm.rethink_reasoning_effort,
        thinking=llm.rethink_thinking,
        profile=llm.profile,
        max_transport_retries=llm.max_transport_retries,
        max_truncation_retries=llm.max_truncation_retries,
        max_syntax_retries=llm.max_syntax_retries,
        max_schema_retries=llm.max_schema_retries,
        max_semantic_retries=llm.max_semantic_retries,
        max_unknown_evidence_retries=llm.max_unknown_evidence_retries,
        retry_backoff_seconds=llm.retry_backoff_seconds,
        request_timeout_seconds=llm.request_timeout_seconds,
        allow_unknown_evidence_stripping=llm.allow_unknown_evidence_stripping,
        max_input_chars=llm.max_input_chars,
        max_parallel_batches=llm.rethink_max_parallel_batches,
        max_calls_per_round=llm.rethink_max_calls_per_round,
        call_reserve=llm.rethink_call_reserve,
        parallel_dimension_groups=llm.rethink_parallel_dimension_groups,
    )
    started = time.perf_counter()
    progress_token = bind_progress(writer)
    try:
        reflection = client.reflect_hypothesis(context=context)
    except Exception as error:
        writer.write_json(
            "live_api_failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise
    finally:
        reset_progress(progress_token)
    elapsed = time.perf_counter() - started
    if reflection is None:
        raise AssertionError("live ReThink unexpectedly returned NOT_APPLICABLE")
    if (
        reflection.hypothesis_id != "H-onehot-live-01"
        or reflection.assessment_id != "AS-onehot-live-01"
        or reflection.assessment_status != fixture_receipt["assessment_status"]
    ):
        raise AssertionError("live ReThink changed runtime-owned hypothesis fields")
    if not reflection.advisory_only or reflection.selection_eligible:
        raise AssertionError("live ReThink violated its advisory-only boundary")
    if len(client.last_dimension_groups) != 4:
        raise AssertionError("live ReThink did not return exactly four dimension groups")
    if len(reflection.dimension_assessments) != 8 or {
        item["dimension"] for item in reflection.dimension_assessments
    } != REQUIRED_DIMENSIONS:
        raise AssertionError("live ReThink did not cover all eight dimensions exactly once")
    degraded_dimensions = [
        item["dimension"]
        for item in reflection.dimension_assessments
        if item["quality_status"] != "model"
    ]
    writer.write_json("rethink_dimension_groups.json", client.last_dimension_groups)
    writer.write_json("hypothesis_reflection.json", reflection)
    if degraded_dimensions:
        writer.write_json(
            "live_api_failure.json",
            {
                "error_type": "DegradedDimensionGroups",
                "degraded_dimensions": degraded_dimensions,
                "elapsed_seconds": elapsed,
            },
        )
        print(
            json.dumps(
                {
                    "status": "failed",
                    "artifact_dir": str(output),
                    "degraded_dimensions": degraded_dimensions,
                },
                ensure_ascii=False,
            )
        )
        raise AssertionError("one or more live dimension groups degraded to fallback")
    if not all(
        item["quality_status"] == "model"
        for item in reflection.dimension_assessments
    ):
        raise AssertionError("one or more live dimension groups degraded to fallback")

    receipt = {
        "test_kind": "isolated_rethink_live_api",
        "production_experiment_config": args.production_config,
        "production_runner_invoked": False,
        "production_model_config": production.model.name,
        "fixture_model_config": args.fixture_config,
        "fixture_model_name": fixture_config.name,
        "kermut_activated": False,
        "provider": llm.provider,
        "model": llm.model,
        "prompt_profile": llm.profile,
        "logical_dimension_group_calls": len(client.last_dimension_groups),
        "dimension_count": len(reflection.dimension_assessments),
        "quality_status": reflection.quality_status,
        "elapsed_seconds": elapsed,
        **fixture_receipt,
    }
    writer.write_json("fixture_receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": "passed",
                "artifact_dir": str(output),
                "production_model": production.model.name,
                "fixture_model": fixture_config.name,
                "kermut_activated": False,
                "provider": llm.provider,
                "model": llm.model,
                "assessment_status": reflection.assessment_status,
                "logical_dimension_group_calls": len(client.last_dimension_groups),
                "dimension_count": len(reflection.dimension_assessments),
                "quality_status": reflection.quality_status,
                "elapsed_seconds": round(elapsed, 3),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
