"""Live three-round acceptance test for hypothesis-level ReThink.

The campaign uses DeepSeek for Scientist, Critic, and ReThink, and the Qwen
embedding/reranker APIs for read-only RAG. GB1 benchmark measurements are the
only round-level fitness values. No fitness model, including Kermut, is created;
the mandatory final evaluator receives a benchmark-truth adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.loop import CampaignRunner
from fitness_agents.utils.progress import configure_progress_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/gb1_rethink_hypothesis_live_api.yaml"
RESTRICTED_RELATIVE_PATH = (
    "claims/continuous_evolution_operations/maintain_mutagenesis_and_host_flow.md"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/gb1-rethink-hypothesis-live-api",
    )
    return parser.parse_args()


def _require_live_credentials() -> None:
    load_project_env(PROJECT_ROOT)
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
        if not resolve_secret(f"env:{name}", name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required live-API environment variables: " + ", ".join(missing)
        )


def _benchmark_truth(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["variant_id"]: float(row["fitness"]) for row in csv.DictReader(handle)}


class BenchmarkTruthEvaluator:
    """Final-evaluation adapter that only replays authoritative benchmark values."""

    model_version = "gb1-benchmark-truth:v1"

    def __init__(self, truth: dict[str, float]) -> None:
        self._truth = truth
        self.fit_calls = 0

    def fit(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
        validation_variants: Sequence[Variant] | None = None,
        validation_observations: Sequence[FitnessObservation] | None = None,
    ) -> BenchmarkTruthEvaluator:
        del variants, observations, validation_variants, validation_observations
        self.fit_calls += 1
        return self

    def predict(self, variants: Sequence[Variant]) -> list[Prediction]:
        return [
            Prediction(
                variant_id=variant.variant_id,
                fitness_mean=self._truth[variant.variant_id],
                fitness_std=0.0,
                interval_90=(self._truth[variant.variant_id], self._truth[variant.variant_id]),
                ood_score=0.0,
                component_scores={"benchmark_truth": self._truth[variant.variant_id]},
                model_version=self.model_version,
                is_measured=True,
            )
            for variant in variants
        ]


def _live_config(path: Path, output_root: Path) -> ExperimentConfig:
    config = load_experiment_config(path)
    local = config.knowledge.local_knowledge
    corpus_path = local.corpus_index_path or local.index_path
    if corpus_path is None or not corpus_path.is_file():
        raise FileNotFoundError("The prebuilt Qwen RAG corpus index is missing")
    if not any(RESTRICTED_RELATIVE_PATH in root.exclude for root in local.roots):
        raise AssertionError("The configured RAG roots do not exclude the restricted path")
    overlay = output_root / "rag-overlays" / f"live-{time.time_ns()}.sqlite"
    local = replace(
        local,
        corpus_mode="read_only_prebuilt",
        retrieval_overlay_path=overlay,
    )
    return replace(
        config,
        output_root=output_root,
        knowledge=replace(config.knowledge, local_knowledge=local),
    )


def _assert_preflight(config: ExperimentConfig) -> None:
    retrieval = config.knowledge.local_knowledge.retrieval
    checks = {
        "three rounds": config.rounds == 3,
        "hypothesis ReThink enabled": (
            config.validation.rethink_enabled
            and config.validation.rethink_mode == "hypothesis"
        ),
        "DeepSeek Scientist/ReThink": config.llm.provider == "deepseek",
        "remote DeepSeek Critic": (
            config.critic.enabled
            and config.critic.mode == "remote"
            and config.critic.provider == "deepseek"
        ),
        "Kermut absent": config.model.name != "kermut",
        "active learning disabled": not config.active_learning.enabled,
        "generation predictors disabled": (
            not config.generation.use_fitness_predictors
            and not config.generation.predictor_models
            and config.generation.predictor_weight == 0.0
        ),
        "dry validation disabled": (
            not config.validation.enabled and not config.validation.predictor_models
        ),
        "read-only prebuilt RAG": (
            config.knowledge.local_knowledge.enabled
            and config.knowledge.local_knowledge.corpus_mode == "read_only_prebuilt"
        ),
        "Qwen API embedding": (
            retrieval.mode == "hybrid"
            and retrieval.dense_enabled
            and retrieval.embedding_backend == "api"
            and retrieval.embedding_api_config is not None
            and retrieval.embedding_model_path is None
        ),
        "Qwen API reranker": (
            retrieval.reranker_backend == "api"
            and retrieval.reranker_api_config is not None
            and retrieval.reranker_model_path is None
        ),
    }
    failed = [label for label, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Live campaign preflight failed: " + ", ".join(failed))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_campaign(
    config: ExperimentConfig,
    summary: dict[str, Any],
    truth: dict[str, float],
    factory_calls: list[dict[str, Any]],
    evaluators: list[BenchmarkTruthEvaluator],
) -> dict[str, Any]:
    run_dir = Path(summary["run_dir"])
    if summary["run_status"] != "completed" or summary["rounds_aborted"] != 0:
        raise AssertionError("The live campaign did not complete all rounds")
    if summary["hypothesis_reflections"] != 3 or summary["rethink_reflections"] != 0:
        raise AssertionError("The campaign did not retain exactly three aggregate reflections")
    if summary["fitness_predictors_used_for_generation"]:
        raise AssertionError("A fitness predictor was used for generation")
    if summary["required_node_failures"] or summary["fallback_nodes"]:
        raise AssertionError("The live campaign used a required-node or fallback path")
    expected_factory_call = {
        "seed": config.seed + config.rounds + 1,
        "requested_model": config.model.name,
    }
    if factory_calls != [expected_factory_call]:
        raise AssertionError("A fitness model was requested outside final evaluation")
    if len(evaluators) != 1 or evaluators[0].fit_calls != 1:
        raise AssertionError("The benchmark-truth final evaluator was not used exactly once")

    rag_receipts: list[dict[str, Any]] = []
    dimension_quality: list[str] = []
    for round_id in range(1, config.rounds + 1):
        round_dir = run_dir / f"round_{round_id:02d}"
        scope = _read_json(round_dir / "prediction_scope_receipt.json")
        if (
            scope["acquisition_prediction_scope"] != "none"
            or scope["acquisition_prediction_count"] != 0
            or scope["dry_validation_scope"] != "disabled"
            or scope["dry_validation_candidate_count"] != 0
        ):
            raise AssertionError(f"Round {round_id} activated a fitness prediction path")

        rag = _read_json(round_dir / "local_rag_retrieval.json")
        if not rag.get("query_id") or not rag.get("chunks"):
            raise AssertionError(f"Round {round_id} did not return live RAG chunks")
        rag_receipts.append(
            {
                "round_id": round_id,
                "query_id": rag["query_id"],
                "chunk_count": len(rag["chunks"]),
            }
        )

        groups = _read_json(round_dir / "rethink_dimension_groups.json")
        if len(groups) != 4:
            raise AssertionError(f"Round {round_id} did not return four ReThink groups")
        assessments = [item for group in groups for item in group["dimension_assessments"]]
        if {item["dimension"] for item in assessments} != REQUIRED_DIMENSIONS:
            raise AssertionError(f"Round {round_id} did not cover all eight dimensions")
        dimension_quality.extend(item["quality_status"] for item in assessments)
        if any(item["quality_status"] != "model" for item in assessments):
            raise AssertionError(f"Round {round_id} degraded a ReThink dimension to fallback")

        validations = _read_json(round_dir / "validation_matrix.json")
        if len(validations) != config.budget_per_round:
            raise AssertionError(f"Round {round_id} returned an unexpected wet batch size")
        for item in validations:
            if item["validation_type"] != "wet" or item["model_version"] is not None:
                raise AssertionError(f"Round {round_id} created a dry validation record")
            if item["value"] != truth[item["variant_id"]]:
                raise AssertionError(f"Round {round_id} did not use benchmark truth")

    with sqlite3.connect(run_dir / "knowledge_graph.sqlite") as connection:
        aggregate_count = connection.execute(
            "SELECT COUNT(*) FROM hypothesis_reflections"
        ).fetchone()[0]
        sample_link_count = connection.execute(
            "SELECT COUNT(*) FROM validation_records WHERE reflection_id IS NOT NULL"
        ).fetchone()[0]
    if aggregate_count != 3 or sample_link_count != 0:
        raise AssertionError("KG reflection persistence was not collection-level only")

    trace = [
        json.loads(line)
        for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if sum(item.get("event_type") == "local_knowledge_retrieved" for item in trace) != 3:
        raise AssertionError("The trace does not contain three RAG retrieval events")
    if sum(item.get("event_type") == "rethink_completed" for item in trace) != 3:
        raise AssertionError("The trace does not contain three ReThink completion events")

    return {
        "status": "passed",
        "test_kind": "gb1_hypothesis_rethink_live_api",
        "run_id": summary["run_id"],
        "run_dir": str(run_dir),
        "rounds": config.rounds,
        "wet_measurements": config.rounds * config.budget_per_round,
        "deepseek_scientist_model": config.llm.model,
        "deepseek_critic_model": config.critic.model,
        "qwen_embedding_model": retrieval_model_id(
            config.knowledge.local_knowledge.retrieval.embedding_api_config
        ),
        "qwen_reranker_model": retrieval_model_id(
            config.knowledge.local_knowledge.retrieval.reranker_api_config
        ),
        "rag_receipts": rag_receipts,
        "rethink_group_calls": config.rounds * 4,
        "rethink_dimension_assessments": len(dimension_quality),
        "hypothesis_reflections_in_kg": aggregate_count,
        "sample_reflection_links_in_kg": sample_link_count,
        "fitness_model_activated": False,
        "kermut_activated": False,
        "final_mse": summary["final_prediction_metrics"]["mse"],
        "final_rmse": summary["final_prediction_metrics"]["rmse"],
    }


def retrieval_model_id(config: Any) -> str | None:
    return None if config is None else str(config.model)


def main() -> None:
    args = _arguments()
    _require_live_credentials()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    config = _live_config(args.config, output_root)
    _assert_preflight(config)
    if config.task.oracle_data_path is None:
        raise AssertionError("GB1 live acceptance requires the benchmark oracle CSV")
    truth = _benchmark_truth(config.task.oracle_data_path)
    factory_calls: list[dict[str, Any]] = []
    evaluators: list[BenchmarkTruthEvaluator] = []

    def benchmark_truth_factory(model_config: Any, *, seed: int) -> BenchmarkTruthEvaluator:
        factory_calls.append({"seed": seed, "requested_model": model_config.name})
        evaluator = BenchmarkTruthEvaluator(truth)
        evaluators.append(evaluator)
        return evaluator

    configure_progress_logging()
    started = time.perf_counter()
    summary = CampaignRunner(
        config,
        predictor_factory=benchmark_truth_factory,
    ).run()
    receipt = _assert_campaign(config, summary, truth, factory_calls, evaluators)
    receipt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
