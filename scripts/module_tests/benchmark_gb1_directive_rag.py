"""Run a blinded paired GB1 no-local-RAG versus directive-RAG benchmark.

The script never exposes variant identities or per-variant fitness to the
research iteration. Benchmark truth is injected only through the mandatory
final evaluator. Kermut and every generation-time fitness predictor are blocked.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.contracts.schemas import FitnessObservation, Prediction, Variant
from fitness_agents.local_knowledge.api_backends import build_embedding_backend
from fitness_agents.local_knowledge.index import SQLiteLocalKnowledgeIndex
from fitness_agents.utils.progress import configure_progress_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/gb1_directive_rag_benchmark.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/gb1-directive-rag-benchmark"
DEFAULT_SEEDS = (11, 23, 37, 53, 71)
CARD_ROOT_ID_PREFIX = "GB1_DIRECTIVE_"
FEEDBACK_CONTRACTS = ("standard", "cold_start_wet_prior")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--canary-condition", choices=("no_rag", "rag"))
    parser.add_argument("--canary-rounds", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--feedback-contract",
        choices=FEEDBACK_CONTRACTS,
        default="standard",
        help=(
            "Audit the cross-round observation contract. cold_start_wet_prior "
            "requires one WT observation before round 1 and exactly one wet batch "
            "to become visible before each later round."
        ),
    )
    return parser.parse_args()


def require_credentials() -> None:
    load_project_env(PROJECT_ROOT / ".env")
    missing = [
        name
        for name in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY")
        if not resolve_secret(f"env:{name}", name)
    ]
    if missing:
        raise RuntimeError(
            "Missing required live-API environment variables: " + ", ".join(missing)
        )


def benchmark_truth(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["variant_id"]: float(row["fitness"]) for row in csv.DictReader(handle)}


class BenchmarkTruthEvaluator:
    """Final-only adapter for authoritative GB1 benchmark values."""

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


def assert_static_preflight(
    config: ExperimentConfig, *, feedback_contract: str = "standard"
) -> dict[str, Any]:
    if feedback_contract not in FEEDBACK_CONTRACTS:
        raise ValueError(f"Unknown feedback contract: {feedback_contract}")
    local = config.knowledge.local_knowledge
    retrieval = local.retrieval
    corpus_path = local.corpus_index_path or local.index_path
    configured_predictor_names = {
        config.model.name.casefold(),
        *(item.name.casefold() for item in config.generation.predictor_models),
        *(item.name.casefold() for item in config.validation.predictor_models),
        *(
            item.name.casefold()
            for item in config.active_learning.posterior.predictor_models
        ),
    }
    checks = {
        "baseline_model_not_kermut": config.model.name.lower() != "kermut",
        "kermut_absent_from_configured_predictors": "kermut"
        not in configured_predictor_names,
        "active_learning_disabled": not config.active_learning.enabled,
        "generation_predictors_disabled": (
            not config.generation.use_fitness_predictors
            and not config.generation.predictor_models
            and config.generation.predictor_weight == 0.0
        ),
        "dry_validation_disabled": (
            not config.validation.enabled and not config.validation.predictor_models
        ),
        "fallback_policy_none": config.critic.fallback_policy == "none",
        "critic_semantic_repair_bounded": config.critic.max_semantic_retries == 2,
        "three_rounds": config.rounds == 3,
        "local_bundle_only": (
            bool(local.roots)
            and all(root.root_id.startswith(CARD_ROOT_ID_PREFIX) for root in local.roots)
            and all(root.include == ("cards/*.md",) for root in local.roots)
        ),
        "target_identity_context_is_explicit": (
            local.leakage_guard.enabled
            and local.leakage_guard.allow_target_identity_context
            and local.leakage_guard.quarantine_target_documents
            and local.leakage_guard.block_target_entities
        ),
        "qwen_embedding_and_reranker": (
            retrieval.mode == "hybrid"
            and retrieval.embedding_backend == "api"
            and retrieval.reranker_backend == "api"
            and retrieval.embedding_api_config is not None
            and retrieval.reranker_api_config is not None
        ),
        "benchmark_truth_available": (
            (
                config.task.oracle_data_path is not None
                and config.task.oracle_data_path.is_file()
            )
            or (
                config.task.split_root is not None
                and (config.task.split_root / "manifest.public.json").is_file()
            )
        ),
        "bundle_validation_passed": all(
            (
                (root.path / "validation-receipt.json").is_file()
                and json.loads(
                    (root.path / "validation-receipt.json").read_text(encoding="utf-8")
                ).get("status")
                == "passed"
            )
            for root in local.roots
        ),
    }
    if feedback_contract == "cold_start_wet_prior":
        three_features = ("physchem", "conservation", "structure")
        checks.update(
            {
                "cold_start_wild_type_only": (
                    config.prior_schedule.mode == "cold_start"
                    and config.prior_schedule.keep_wild_type
                ),
                "three_feature_knowledge_enabled": all(
                    (
                        config.knowledge.physchem,
                        config.knowledge.conservation,
                        config.knowledge.structure,
                    )
                ),
                "three_feature_tools_enabled": (
                    tuple(config.kg_interaction.feature_channels) == three_features
                    and tuple(config.hierarchical_hypothesis.required_channels)
                    == three_features
                    and config.hierarchical_hypothesis.enabled
                ),
                "deepseek_v4_pro_scientist_and_critic": (
                    config.llm.provider == "deepseek"
                    and config.llm.model == "deepseek-v4-pro"
                    and config.critic.provider == "deepseek"
                    and config.critic.model == "deepseek-v4-pro"
                ),
                "wet_feedback_has_selection_authority": (
                    config.knowledge_enabled
                    and config.validation.wet_weight > 0
                    and config.generation.prior_weight > 0
                ),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Benchmark preflight failed: " + ", ".join(failed))
    return {
        "checks": checks,
        "corpus_path": str(corpus_path) if corpus_path is not None else None,
        "feedback_contract": feedback_contract,
    }


def audit_validation_feedback(
    run_dir: Path,
    config: ExperimentConfig,
    *,
    feedback_contract: str,
) -> dict[str, Any]:
    """Audit feedback counts without exposing variant identities or labels."""

    if feedback_contract == "standard":
        return {"contract": feedback_contract, "passed": True, "checks": {}}
    if feedback_contract != "cold_start_wet_prior":
        raise ValueError(f"Unknown feedback contract: {feedback_contract}")

    trace_events = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trace_events.append(json.loads(line))
    campaign_started = next(
        item for item in trace_events if item.get("event_type") == "campaign_started"
    )
    round_started = sorted(
        (
            item
            for item in trace_events
            if item.get("event_type") == "round_started"
        ),
        key=lambda item: int(item["payload"]["round_id"]),
    )
    initial_count = int(campaign_started["payload"]["initial_count"])
    visible_counts = [int(item["payload"]["n_observed"]) for item in round_started]
    expected_visible_counts = [
        1 + index * config.budget_per_round for index in range(config.rounds)
    ]

    validation_records = json.loads(
        (run_dir / "validation_records.json").read_text(encoding="utf-8")
    )
    wet_counts_by_round = [
        sum(
            1
            for item in validation_records
            if item.get("validation_type") == "wet"
            and int(item.get("round_id") or 0) == round_id
        )
        for round_id in range(1, config.rounds + 1)
    ]
    prior_score_interface_by_round = []
    for round_id in range(2, config.rounds + 1):
        path = run_dir / f"round_{round_id:02d}" / "design_scores.json"
        scores = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
        prior_score_interface_by_round.append(
            bool(scores) and all("prior_score" in item for item in scores)
        )

    actual_batch_sizes = [
        int(item)
        for item in json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))[
            "actual_batch_sizes"
        ]
    ]
    prior_schedule = campaign_started["payload"].get("prior_schedule") or {}
    checks = {
        "initial_visible_count_is_one": initial_count == 1,
        "cold_start_schedule_recorded": (
            prior_schedule.get("mode") == "cold_start"
            and prior_schedule.get("keep_wild_type") is True
        ),
        "round_visible_counts_match_feedback": visible_counts
        == expected_visible_counts,
        "each_round_measured_full_wet_batch": wet_counts_by_round
        == [config.budget_per_round] * config.rounds,
        "actual_batch_sizes_match_budget": actual_batch_sizes
        == [config.budget_per_round] * config.rounds,
        "later_rounds_expose_wet_prior_score_interface": all(
            prior_score_interface_by_round
        ),
    }
    return {
        "contract": feedback_contract,
        "passed": all(checks.values()),
        "checks": checks,
        "initial_visible_count": initial_count,
        "visible_observation_counts_by_round": visible_counts,
        "wet_validation_counts_by_round": wet_counts_by_round,
        "prior_score_interface_by_round": prior_score_interface_by_round,
    }


def declared_card_count(config: ExperimentConfig) -> int:
    count = 0
    for root in config.knowledge.local_knowledge.roots:
        receipt_path = root.path / "validation-receipt.json"
        if not receipt_path.is_file():
            raise AssertionError(f"Directive bundle receipt is missing: {receipt_path}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "passed":
            raise AssertionError(f"Directive bundle receipt did not pass: {receipt_path}")
        count += sum(
            1
            for item in receipt.get("cards", ())
            if str(item.get("path", "")).replace("\\", "/").startswith("cards/")
        )
    if count < 1:
        raise AssertionError("Directive bundle declares no validated cards")
    return count


def build_index(config: ExperimentConfig) -> dict[str, Any]:
    local = config.knowledge.local_knowledge
    corpus_path = local.corpus_index_path or local.index_path
    if corpus_path is None:
        raise AssertionError("Directive RAG requires an explicit corpus path")
    expected_documents = declared_card_count(config)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    rebuild_reason: str | None = None
    if corpus_path.exists():
        index = SQLiteLocalKnowledgeIndex(corpus_path, read_only=True)
        try:
            stats = index.stats()
            if (
                int(stats.get("documents", 0)) < expected_documents
                or int(stats.get("chunks", 0)) < expected_documents
            ):
                raise AssertionError("Directive corpus is unexpectedly incomplete")
            index.assert_runtime_binding(local)
            return {"status": "reused", "stats": stats, "path": str(corpus_path)}
        except (AssertionError, RuntimeError) as error:
            # A corpus can have the expected row counts while still predating or
            # mismatching the active access-policy/runtime-manifest binding.
            # Rebuild to a sidecar and atomically replace it only after success.
            rebuild_reason = str(error)
        finally:
            index.close()

    sidecar = corpus_path.with_name(f"{corpus_path.name}.building-{time.time_ns()}")
    backend = build_embedding_backend(local.retrieval)
    index = SQLiteLocalKnowledgeIndex(sidecar)
    try:
        report = index.build(local, embedding_backend=backend)
    except Exception:
        index.close()
        if sidecar.is_file():
            sidecar.unlink()
        raise
    else:
        index.close()
        sidecar.replace(corpus_path)
    if (
        report.indexed_documents < expected_documents
        or report.indexed_chunks < expected_documents
    ):
        raise AssertionError("Directive corpus is unexpectedly incomplete")
    return {
        "status": "rebuilt" if rebuild_reason is not None else "built",
        "rebuild_reason": rebuild_reason,
        "report": asdict(report),
        "path": str(corpus_path),
    }


def run_config(
    base: ExperimentConfig,
    *,
    seed: int,
    condition: str,
    output_root: Path,
) -> ExperimentConfig:
    local = base.knowledge.local_knowledge
    if condition == "rag":
        corpus_path = local.corpus_index_path or local.index_path
        if corpus_path is None or not corpus_path.is_file():
            raise FileNotFoundError("Prebuilt directive RAG corpus is missing")
        local = replace(
            local,
            enabled=True,
            corpus_mode="read_only_prebuilt",
            retrieval_overlay_path=(
                output_root / "overlays" / f"rag-seed-{seed}-{time.time_ns()}.sqlite"
            ),
        )
    elif condition == "no_rag":
        local = replace(local, enabled=False)
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return replace(
        base,
        seed=seed,
        condition=condition,
        run_label=f"GB1-DIRECTIVE-{condition.upper()}-S{seed}",
        output_root=output_root / "runs" / condition / f"seed-{seed}",
        knowledge=replace(base.knowledge, local_knowledge=local),
    )


def card_retrieval_summary(
    run_dir: Path,
    rounds: int,
    intended_knowledge_types_by_round: dict[int, tuple[str, ...]],
) -> dict[str, Any]:
    chunk_counts: list[int] = []
    card_rounds = 0
    retrieved_card_ids: set[str] = set()
    retrieved_card_ids_by_round: list[list[str]] = []
    intended_card_match_by_round: list[bool] = []
    for round_id in range(1, rounds + 1):
        path = run_dir / f"round_{round_id:02d}" / "local_rag_retrieval.json"
        if not path.is_file():
            chunk_counts.append(0)
            retrieved_card_ids_by_round.append([])
            intended_card_match_by_round.append(False)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = payload.get("chunks", [])
        chunk_counts.append(len(chunks))
        round_has_card = False
        round_card_ids: set[str] = set()
        round_knowledge_types: set[str] = set()
        for chunk in chunks:
            provenance = chunk.get("provenance", {}).get("metadata", {})
            root_id = str(provenance.get("root_id", ""))
            relative_path = str(provenance.get("relative_path", ""))
            normalized_path = relative_path.replace("\\", "/")
            if root_id.startswith(CARD_ROOT_ID_PREFIX) and normalized_path.startswith("cards/"):
                round_has_card = True
                card_id = Path(relative_path).stem
                retrieved_card_ids.add(card_id)
                round_card_ids.add(card_id)
                round_knowledge_types.add(str(chunk.get("knowledge_type", "")).casefold())
        intended_types = set(intended_knowledge_types_by_round.get(round_id, ()))
        intended_match = round_has_card and (
            not intended_types or bool(round_knowledge_types & intended_types)
        )
        card_rounds += int(intended_match)
        retrieved_card_ids_by_round.append(sorted(round_card_ids))
        intended_card_match_by_round.append(intended_match)
    return {
        "chunk_counts_by_round": chunk_counts,
        "rounds_with_intended_card": card_rounds,
        "run_retrieved_intended_card": card_rounds == rounds,
        "retrieved_card_ids": sorted(retrieved_card_ids),
        "retrieved_card_ids_by_round": retrieved_card_ids_by_round,
        "intended_card_match_by_round": intended_card_match_by_round,
    }


def execute_run(
    config: ExperimentConfig,
    truth: dict[str, float],
    *,
    feedback_contract: str = "standard",
) -> dict[str, Any]:
    # Keep the full campaign stack lazy so configuration-only dry runs do not
    # require scientific runtime dependencies such as SciPy.
    from fitness_agents.loop import CampaignRunner

    factory_calls: list[dict[str, Any]] = []
    evaluators: list[BenchmarkTruthEvaluator] = []

    def evaluator_factory(model_config: Any, *, seed: int) -> BenchmarkTruthEvaluator:
        factory_calls.append({"requested_model": model_config.name, "seed": seed})
        evaluator = BenchmarkTruthEvaluator(truth)
        evaluators.append(evaluator)
        return evaluator

    started = time.perf_counter()
    summary = CampaignRunner(config, predictor_factory=evaluator_factory).run()
    round_best = [float(item["best_seen_fitness"]) for item in summary["round_metrics"]]
    run_dir = Path(summary["run_dir"])
    feedback = audit_validation_feedback(
        run_dir, config, feedback_contract=feedback_contract
    )
    expected_factory = [
        {
            "requested_model": config.model.name,
            "seed": config.seed + config.rounds + 1,
        }
    ]
    integrity = {
        "completed": summary["run_status"] == "completed" and summary["rounds_aborted"] == 0,
        "no_generation_predictor": not summary["fitness_predictors_used_for_generation"],
        "no_fallback": not summary["fallback_nodes"],
        "no_required_node_failure": not summary["required_node_failures"],
        "final_evaluator_only": (
            factory_calls == expected_factory
            and len(evaluators) == 1
            and evaluators[0].fit_calls == 1
        ),
        "kermut_not_activated": config.model.name.lower() != "kermut",
        "validation_feedback_contract": bool(feedback["passed"]),
    }
    failed_integrity = [name for name, passed in integrity.items() if not passed]
    if failed_integrity:
        raise RuntimeError(
            "Run integrity failed: " + ", ".join(sorted(failed_integrity))
        )
    receipt: dict[str, Any] = {
        "seed": config.seed,
        "condition": config.condition,
        "run_id": summary["run_id"],
        "run_dir": str(run_dir),
        "round_best_seen": round_best,
        "final_best_seen": round_best[-1],
        "auc_proxy": statistics.fmean(round_best),
        "integrity": integrity,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "feedback": feedback,
    }
    if config.condition == "rag":
        receipt["retrieval"] = card_retrieval_summary(
            run_dir,
            config.rounds,
            config.knowledge.local_knowledge.retrieval.runtime_knowledge_types_by_round,
        )
    return receipt


def aggregate(runs: list[dict[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    by_pair: dict[int, dict[str, dict[str, Any]]] = {seed: {} for seed in seeds}
    for run in runs:
        if run.get("status") == "failed":
            continue
        by_pair[int(run["seed"])][str(run["condition"])] = run
    complete_pairs = {
        seed: pair for seed, pair in by_pair.items() if {"no_rag", "rag"} <= pair.keys()
    }
    paired = [
        {
            "seed": seed,
            "final_best_delta": pair["rag"]["final_best_seen"]
            - pair["no_rag"]["final_best_seen"],
            "auc_delta": pair["rag"]["auc_proxy"] - pair["no_rag"]["auc_proxy"],
        }
        for seed, pair in sorted(complete_pairs.items())
    ]
    integrity_ok = all(
        all(value is True for value in run["integrity"].values())
        for run in runs
        if run.get("status") != "failed"
    ) and len(runs) == 2 * len(seeds)
    rag_card_hits = sum(
        int(run.get("retrieval", {}).get("run_retrieved_intended_card", False))
        for run in runs
        if run.get("condition") == "rag" and run.get("status") != "failed"
    )
    median_final = (
        statistics.median(item["final_best_delta"] for item in paired) if paired else None
    )
    mean_auc = statistics.fmean(item["auc_delta"] for item in paired) if paired else None
    success = (
        len(paired) == len(seeds)
        and median_final is not None
        and median_final > 0.0
        and mean_auc is not None
        and mean_auc > 0.0
        and integrity_ok
        and rag_card_hits >= 4
    )
    return {
        "paired_deltas": paired,
        "complete_pair_count": len(paired),
        "median_paired_final_best_delta": median_final,
        "mean_paired_auc_delta": mean_auc,
        "rag_runs_with_intended_card": rag_card_hits,
        "runtime_integrity_passed": integrity_ok,
        "success": success,
    }


def main() -> None:
    args = arguments()
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"Frozen seed set is {DEFAULT_SEEDS}; received {seeds}")
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    base = load_experiment_config(config_path)
    preflight = assert_static_preflight(base, feedback_contract=args.feedback_contract)
    if args.dry_run:
        print(json.dumps({"status": "dry_run_passed", "preflight": preflight}, indent=2))
        return

    require_credentials()
    index_receipt = build_index(base) if args.build_index else {"status": "not_requested"}
    corpus_path = base.knowledge.local_knowledge.corpus_index_path
    if corpus_path is None or not corpus_path.is_file():
        raise FileNotFoundError("Directive index is missing; rerun with --build-index")
    if base.task.oracle_data_path is None:
        raise AssertionError("GB1 benchmark truth path is missing")
    truth = benchmark_truth(base.task.oracle_data_path)
    configure_progress_logging()

    output_root.mkdir(parents=True, exist_ok=True)
    if args.canary_condition is not None:
        canary_root = output_root / "canary" / str(time.time_ns())
        canary_config = run_config(
            base,
            seed=DEFAULT_SEEDS[0],
            condition=args.canary_condition,
            output_root=canary_root,
        )
        canary_config = replace(canary_config, rounds=args.canary_rounds)
        canary_receipt = {
            "schema_version": "gb1-directive-canary:v1",
            "scored": False,
            "condition": args.canary_condition,
            "result": execute_run(
                canary_config,
                truth,
                feedback_contract=args.feedback_contract,
            ),
        }
        canary_path = canary_root / "canary_receipt.json"
        canary_path.parent.mkdir(parents=True, exist_ok=True)
        canary_path.write_text(
            json.dumps(canary_receipt, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(canary_receipt, indent=2))
        return

    runs: list[dict[str, Any]] = []
    execution_order: list[dict[str, Any]] = []
    failure_seen = False
    for index, seed in enumerate(seeds):
        conditions = ("no_rag", "rag") if index % 2 == 0 else ("rag", "no_rag")
        for condition in conditions:
            execution_order.append({"seed": seed, "condition": condition})
            config = run_config(
                base,
                seed=seed,
                condition=condition,
                output_root=output_root,
            )
            try:
                runs.append(
                    execute_run(
                        config,
                        truth,
                        feedback_contract=args.feedback_contract,
                    )
                )
            except Exception as error:  # noqa: BLE001 - persist a complete paired audit
                runs.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                )
                failure_seen = True
                break
        if failure_seen:
            break

    receipt = {
        "schema_version": "gb1-directive-paired-benchmark:v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "seeds": list(seeds),
        "execution_order": execution_order,
        "preflight": preflight,
        "feedback_contract": args.feedback_contract,
        "index": index_receipt,
        "runs": runs,
        "aggregate": aggregate(runs, seeds),
        "research_feedback_scope": "aggregate_blinded_only",
        "variant_identities_in_receipt": False,
        "kermut_activated": False,
    }
    receipt_path = output_root / "paired_benchmark_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
