"""Run the three-fold GB1 3features no-RAG/RAG cold-start validation.

The protocol is fixed to the established validation example: three folds,
three rounds, 32 candidates scored per round, and 16 wet measurements revealed
per round. The only data-visibility change is a WT-only initial observation.
Kermut, generation-time fitness predictors, active learning, and dry validation
are prohibited by preflight and runtime integrity checks.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fitness_agents.agents.remote_llm import load_project_env, resolve_secret
from fitness_agents.config import ExperimentConfig, load_experiment_config
from fitness_agents.utils.progress import configure_progress_logging
from scripts.module_tests.benchmark_gb1_directive_rag import (
    assert_static_preflight,
    audit_validation_feedback,
    build_index,
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/gb1_3features_coldstart_validation_deepseek_v4_pro.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "artifacts/gb1-3features-coldstart-validation-deepseek-v4-pro"
)
FOLDS = (0, 1, 2)
EXPECTED_MANIFEST_FOLDS = 5
CONDITIONS = ("kg_3features_base", "kg_3features_rag")
FEEDBACK_CONTRACT = "cold_start_wet_prior"
EXPECTED_ROUNDS = 3
EXPECTED_CANDIDATES = 32
EXPECTED_WET_BUDGET = 16


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only prior campaigns that already passed every integrity check.",
    )
    parser.add_argument("--dry-run", action="store_true")
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


def assert_three_fold_validation_preflight(config: ExperimentConfig) -> dict[str, Any]:
    common = assert_static_preflight(config, feedback_contract=FEEDBACK_CONTRACT)
    split_root = config.task.split_root
    manifest_path = split_root / "manifest.public.json" if split_root is not None else None
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path is not None and manifest_path.is_file()
        else {}
    )
    fold_manifests: list[dict[str, Any]] = []
    if split_root is not None:
        for fold in FOLDS:
            fold_manifest_path = split_root / f"fold_{fold:02d}" / "fold_manifest.json"
            if fold_manifest_path.is_file():
                fold_manifests.append(
                    json.loads(fold_manifest_path.read_text(encoding="utf-8"))
                )
    quota = config.generation.quota_allocation
    configured_predictors = {
        config.model.name.casefold(),
        *(item.name.casefold() for item in config.generation.predictor_models),
        *(item.name.casefold() for item in config.validation.predictor_models),
        *(
            item.name.casefold()
            for item in config.active_learning.posterior.predictor_models
        ),
    }
    checks = {
        "manifest_backed_five_fold_dataset": (
            split_root is not None
            and manifest.get("n_folds") == EXPECTED_MANIFEST_FOLDS
        ),
        "only_first_three_folds_are_selected": tuple(FOLDS) == (0, 1, 2),
        "split_initial_budget_is_wt_only": (
            (manifest.get("options") or {}).get("initial_budget") == 1
            and len(fold_manifests) == len(FOLDS)
            and all(
                (item.get("role_counts") or {}).get("initial_observed") == 1
                and not (item.get("role_counts") or {}).get("train_observed", 0)
                for item in fold_manifests
            )
        ),
        "expected_split_protocol": (
            manifest.get("strategy") == config.task.expected_split_strategy
            and manifest.get("protocol_version") == config.task.expected_protocol_version
        ),
        "three_rounds": config.rounds == EXPECTED_ROUNDS,
        "candidate_pool_is_32": config.candidate_limit == EXPECTED_CANDIDATES,
        "wet_validation_budget_is_16": config.budget_per_round == EXPECTED_WET_BUDGET,
        "original_agent_uq_quota_is_8_3_3_2": (
            quota.enabled
            and quota.hypothesis_target == 8
            and quota.evidence_prior == 3
            and quota.coverage_exploration == 3
            and quota.matched_control == 2
            and quota.total == EXPECTED_WET_BUDGET
        ),
        "candidate_pool_is_not_round_restricted": (
            not config.generation.mutation_order_schedule
        ),
        "only_wt_is_initially_visible": (
            config.prior_schedule.mode == "cold_start"
            and config.prior_schedule.keep_wild_type
        ),
        "wt_only_abstention_uses_coverage_exploration": (
            config.prior_schedule.no_supported_hypothesis_policy
            == "coverage_exploration"
        ),
        "kermut_absent": "kermut" not in configured_predictors,
        "wet_only_feedback": (
            not config.validation.enabled
            and not config.validation.predictor_models
            and not config.generation.use_fitness_predictors
            and not config.generation.predictor_models
            and not config.active_learning.enabled
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            "Three-fold validation preflight failed: " + ", ".join(sorted(failed))
        )
    return {
        "common": common,
        "checks": checks,
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "folds": list(FOLDS),
        "conditions": list(CONDITIONS),
    }


def validation_run_config(
    base: ExperimentConfig,
    *,
    fold: int,
    condition: str,
    output_root: Path,
) -> ExperimentConfig:
    if fold not in FOLDS:
        raise ValueError(f"Validation fold must be one of {FOLDS}")
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown validation condition: {condition}")
    local = base.knowledge.local_knowledge
    rag_enabled = condition == "kg_3features_rag"
    if rag_enabled:
        corpus_path = local.corpus_index_path or local.index_path
        if corpus_path is None or not corpus_path.is_file():
            raise FileNotFoundError("Prebuilt directive RAG corpus is missing")
        local = replace(
            local,
            enabled=True,
            corpus_mode="read_only_prebuilt",
            retrieval_overlay_path=(
                output_root
                / "overlays"
                / f"kg-3features-rag-fold-{fold:02d}-{time.time_ns()}.sqlite"
            ),
        )
    else:
        local = replace(local, enabled=False)
    return replace(
        base,
        task=replace(base.task, fold_index=fold),
        condition=condition,
        # Keep Windows paths below legacy MAX_PATH even for deeply nested LLM
        # conversation and request-local ID-bridge artifacts.
        run_label=f"V-{'R' if rag_enabled else 'B'}-F{fold}",
        output_root=(
            output_root
            / "runs"
            / ("rag" if rag_enabled else "base")
            / f"f{fold}"
        ),
        knowledge=replace(base.knowledge, local_knowledge=local),
    )


def _round_candidate_counts(run_dir: Path, rounds: int) -> list[int]:
    return [
        int(
            json.loads(
                (run_dir / f"round_{round_id:02d}" / "candidate_pool_receipt.json").read_text(
                    encoding="utf-8"
                )
            )["actual_candidate_count"]
        )
        for round_id in range(1, rounds + 1)
    ]


def _rag_rounds_present(run_dir: Path, rounds: int) -> list[bool]:
    return [
        (run_dir / f"round_{round_id:02d}" / "local_rag_retrieval.json").is_file()
        for round_id in range(1, rounds + 1)
    ]


def execute_run(config: ExperimentConfig) -> dict[str, Any]:
    from fitness_agents.loop import CampaignRunner

    started = time.perf_counter()
    summary = CampaignRunner(config).run()
    run_dir = Path(summary["run_dir"])
    feedback = audit_validation_feedback(
        run_dir, config, feedback_contract=FEEDBACK_CONTRACT
    )
    candidate_counts = _round_candidate_counts(run_dir, config.rounds)
    rag_rounds = _rag_rounds_present(run_dir, config.rounds)
    config_record = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    expected_rag = config.condition == "kg_3features_rag"
    integrity = {
        "completed": summary["run_status"] == "completed"
        and summary["rounds_aborted"] == 0,
        "three_rounds": len(summary["round_metrics"]) == EXPECTED_ROUNDS,
        "candidate_pool_32_each_round": candidate_counts
        == [EXPECTED_CANDIDATES] * EXPECTED_ROUNDS,
        "wet_validation_16_each_round": summary["actual_batch_sizes"]
        == [EXPECTED_WET_BUDGET] * EXPECTED_ROUNDS,
        "feedback_contract": bool(feedback["passed"]),
        "rag_artifacts_match_condition": all(rag_rounds)
        if expected_rag
        else not any(rag_rounds),
        "no_generation_predictor": not summary["fitness_predictors_used_for_generation"],
        "active_learning_disabled": config.active_learning.enabled is False,
        "dry_validation_disabled": config.validation.enabled is False,
        "no_fallback": not summary["fallback_nodes"],
        "no_required_node_failure": not summary["required_node_failures"],
        "kermut_not_configured_or_activated": (
            config.model.name.casefold() != "kermut"
            and config_record.get("model") != "kermut"
        ),
        "deepseek_v4_pro_roles": (
            (config_record.get("llm") or {}).get("model") == "deepseek-v4-pro"
            and (config_record.get("critic") or {}).get("model")
            == "deepseek-v4-pro"
        ),
        "fold_matches": summary["data_source"].get("fold_index")
        == config.task.fold_index,
    }
    failed = [name for name, passed in integrity.items() if not passed]
    if failed:
        raise RuntimeError("Run integrity failed: " + ", ".join(sorted(failed)))

    round_metrics = summary["round_metrics"]
    round_best_seen = [float(item["best_seen_fitness"]) for item in round_metrics]
    return {
        "fold": config.task.fold_index,
        "condition": config.condition,
        "run_id": summary["run_id"],
        "run_dir": str(run_dir),
        "round_best_seen": round_best_seen,
        "round_batch_best": [float(item["batch_best_fitness"]) for item in round_metrics],
        "round_batch_mean": [float(item["batch_mean_fitness"]) for item in round_metrics],
        "final_best_seen": round_best_seen[-1],
        "auc_proxy": statistics.fmean(round_best_seen),
        "feedback": feedback,
        "candidate_counts_by_round": candidate_counts,
        "integrity": integrity,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    paired = []
    for fold in FOLDS:
        by_condition = {
            str(item.get("condition")): item
            for item in runs
            if item.get("status") != "failed" and int(item.get("fold", -1)) == fold
        }
        if set(CONDITIONS) <= by_condition.keys():
            no_rag = by_condition["kg_3features_base"]
            rag = by_condition["kg_3features_rag"]
            paired.append(
                {
                    "fold": fold,
                    "final_best_delta": float(rag["final_best_seen"])
                    - float(no_rag["final_best_seen"]),
                    "auc_delta": float(rag["auc_proxy"])
                    - float(no_rag["auc_proxy"]),
                }
            )
    final_deltas = [item["final_best_delta"] for item in paired]
    auc_deltas = [item["auc_delta"] for item in paired]
    return {
        "paired_deltas": paired,
        "complete_pair_count": len(paired),
        "median_paired_final_best_delta": (
            statistics.median(final_deltas) if final_deltas else None
        ),
        "mean_paired_auc_delta": statistics.fmean(auc_deltas) if auc_deltas else None,
        "runtime_integrity_passed": (
            len(paired) == len(FOLDS)
            and all(all(item["integrity"].values()) for item in runs if "integrity" in item)
        ),
    }


def resumable_runs(receipt_path: Path) -> list[dict[str, Any]]:
    """Load one integrity-complete result per fold/condition from a prior receipt."""

    if not receipt_path.is_file():
        return []
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    retained: dict[tuple[int, str], dict[str, Any]] = {}
    for item in receipt.get("runs", []):
        if item.get("status") == "failed":
            continue
        integrity = item.get("integrity")
        run_dir = item.get("run_dir")
        try:
            fold = int(item.get("fold", -1))
        except (TypeError, ValueError):
            continue
        condition = str(item.get("condition", ""))
        if (
            fold not in FOLDS
            or condition not in CONDITIONS
            or not isinstance(integrity, dict)
            or not integrity
            or not all(bool(value) for value in integrity.values())
            or not run_dir
            or not Path(run_dir).is_dir()
        ):
            continue
        retained[(fold, condition)] = item
    return [retained[key] for key in sorted(retained)]


def _write_receipt(
    path: Path,
    *,
    config_path: Path,
    preflight: dict[str, Any],
    index: dict[str, Any],
    runs: list[dict[str, Any]],
) -> None:
    receipt = {
        "schema_version": "gb1-3features-coldstart-validation:v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "folds": list(FOLDS),
        "conditions": list(CONDITIONS),
        "protocol": {
            "rounds": EXPECTED_ROUNDS,
            "candidate_limit": EXPECTED_CANDIDATES,
            "wet_validation_budget": EXPECTED_WET_BUDGET,
            "initial_visible_observations": 1,
            "feedback_contract": FEEDBACK_CONTRACT,
        },
        "preflight": preflight,
        "index": index,
        "runs": runs,
        "aggregate": aggregate(runs),
        "variant_identities_in_receipt": False,
        "kermut_activated": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    config_path = (
        args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    ).resolve()
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else PROJECT_ROOT / args.output_root
    ).resolve()
    base = load_experiment_config(config_path)
    preflight = assert_three_fold_validation_preflight(base)
    if args.dry_run:
        print(
            json.dumps(
                {"status": "dry_run_passed", "preflight": preflight},
                indent=2,
                default=str,
            )
        )
        return

    require_credentials()
    index = build_index(base) if args.build_index else {"status": "not_requested"}
    corpus_path = base.knowledge.local_knowledge.corpus_index_path
    if corpus_path is None or not corpus_path.is_file():
        raise FileNotFoundError("Directive index is missing; rerun with --build-index")
    configure_progress_logging()

    receipt_path = output_root / "paired_validation_receipt.json"
    runs = resumable_runs(receipt_path) if args.resume else []
    completed = {
        (int(item["fold"]), str(item["condition"]))
        for item in runs
    }
    for fold in FOLDS:
        conditions = CONDITIONS if fold % 2 == 0 else tuple(reversed(CONDITIONS))
        for condition in conditions:
            if (fold, condition) in completed:
                continue
            config = validation_run_config(
                base, fold=fold, condition=condition, output_root=output_root
            )
            try:
                runs.append(execute_run(config))
            except Exception as error:  # noqa: BLE001 - persist complete audit state
                runs.append(
                    {
                        "fold": fold,
                        "condition": condition,
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error)[:1000],
                    }
                )
            _write_receipt(
                receipt_path,
                config_path=config_path,
                preflight=preflight,
                index=index,
                runs=runs,
            )

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps(receipt, indent=2))
    if receipt["aggregate"]["complete_pair_count"] != len(FOLDS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
