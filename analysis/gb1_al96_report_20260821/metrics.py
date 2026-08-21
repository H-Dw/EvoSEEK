"""Metric extraction, aggregation, ranking inputs, and protocol audits."""

from __future__ import annotations

import json
import sqlite3
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CONDITION_ORDER,
    DELTA_COMPARISONS,
    DELTA_METRICS,
    EXPECTED_BATCH_SIZE,
    EXPECTED_FOLDS,
    EXPECTED_ROUNDS,
    FINAL_VISIBLE_OBSERVATIONS,
    INITIAL_OBSERVATIONS,
    METRIC_DIRECTIONS,
    THREE_FEATURE_CONDITIONS,
    TOTAL_QUERY_BUDGET,
)
from io_artifacts import RunArtifact, read_json


def _eligible_runs(runs: list[RunArtifact]) -> list[RunArtifact]:
    return [run for run in runs if run.eligible and run.condition in CONDITION_ORDER]


def build_round_metrics(runs: list[RunArtifact]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in _eligible_runs(runs):
        state = read_json(run.path / "state.json")
        observed = state.get("observed", [])
        initial = [row for row in observed if int(row["round_revealed"]) == 0]
        if len(initial) != INITIAL_OBSERVATIONS:
            raise ValueError(
                f"{run.run_id}: expected {INITIAL_OBSERVATIONS} initial observations, "
                f"found {len(initial)}"
            )
        initial_values = np.asarray([float(row["fitness"]) for row in initial])
        rows.append(
            {
                "condition": run.condition,
                "fold": run.fold,
                "seed": run.seed,
                "assignment_sha256": run.assignment_sha256,
                "run_id": run.run_id,
                "round_id": 0,
                "query_budget": 0,
                "visible_observations": INITIAL_OBSERVATIONS,
                "best_seen_fitness": float(initial_values.max()),
                "visible_mean_fitness": float(initial_values.mean()),
                "batch_best_fitness": np.nan,
                "batch_mean_fitness": np.nan,
                "batch_median_fitness": np.nan,
            }
        )
        summary_rounds = run.summary.get("round_metrics") or []
        if len(summary_rounds) != len(EXPECTED_ROUNDS):
            raise ValueError(f"{run.run_id}: incomplete summary round metrics")
        for metric in summary_rounds:
            round_id = int(metric["round_id"])
            row = {
                "condition": run.condition,
                "fold": run.fold,
                "seed": run.seed,
                "assignment_sha256": run.assignment_sha256,
                "run_id": run.run_id,
                "round_id": round_id,
                "query_budget": round_id * EXPECTED_BATCH_SIZE,
                "visible_observations": INITIAL_OBSERVATIONS
                + round_id * EXPECTED_BATCH_SIZE,
            }
            for key in (
                "best_seen_fitness",
                "visible_mean_fitness",
                "batch_best_fitness",
                "batch_mean_fitness",
                "batch_median_fitness",
            ):
                row[key] = float(metric[key])
            expected_best = max(
                float(item["fitness"])
                for item in observed
                if int(item["round_revealed"]) <= round_id
            )
            if not np.isclose(row["best_seen_fitness"], expected_best):
                raise ValueError(
                    f"{run.run_id} round {round_id}: best_seen mismatch "
                    f"{row['best_seen_fitness']} != {expected_best}"
                )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "fold", "round_id"])
        .reset_index(drop=True)
    )


def build_final_metrics(
    runs: list[RunArtifact], round_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in _eligible_runs(runs):
        trajectory = round_metrics[
            (round_metrics["condition"] == run.condition)
            & (round_metrics["fold"] == run.fold)
        ].sort_values("round_id")
        if trajectory["round_id"].tolist() != [0, 1, 2, 3]:
            raise ValueError(f"{run.run_id}: malformed round trajectory")
        x = trajectory["query_budget"].to_numpy(dtype=float)
        y = trajectory["best_seen_fitness"].to_numpy(dtype=float)
        if int(x[-1]) != TOTAL_QUERY_BUDGET:
            raise ValueError(f"{run.run_id}: unexpected total query budget")
        round_3 = trajectory.iloc[-1]
        final_prediction = run.summary.get("final_prediction_metrics") or {}
        row = {
            "condition": run.condition,
            "fold": run.fold,
            "seed": run.seed,
            "assignment_sha256": run.assignment_sha256,
            "run_id": run.run_id,
            "initial_best_seen": float(y[0]),
            "final_best_seen": float(y[-1]),
            "delta_best_seen": float(y[-1] - y[0]),
            "best_seen_aulc": float(np.trapezoid(y, x) / TOTAL_QUERY_BUDGET),
            "r3_batch_best": float(round_3["batch_best_fitness"]),
            "r3_batch_mean": float(round_3["batch_mean_fitness"]),
            "r3_batch_median": float(round_3["batch_median_fitness"]),
            "queries_used": int(run.summary.get("queries_used", -1)),
            "final_visible_observations": FINAL_VISIBLE_OBSERVATIONS,
            "selection_driver": run.summary.get("selection_driver"),
            "active_learning_module": run.summary.get("active_learning_module"),
            "fitness_predictors_used_for_generation": run.summary.get(
                "fitness_predictors_used_for_generation"
            ),
        }
        for metric in (
            "spearman",
            "pearson",
            "mse",
            "rmse",
            "ndcg",
            "top_k_hit",
            "top_k_recall",
            "regret_at_k",
            "interval_90_coverage",
            "gaussian_nll",
            "n",
        ):
            row[metric] = float(final_prediction[metric])
        row["interval_90_coverage_error"] = abs(
            row["interval_90_coverage"] - 0.90
        )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(["condition", "fold"])
        .reset_index(drop=True)
    )


def aggregate_final_metrics(final_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITION_ORDER:
        group = final_metrics[final_metrics["condition"] == condition]
        for metric, direction in METRIC_DIRECTIONS.items():
            values = group[metric].astype(float)
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "direction": direction,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "n": int(values.count()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_round_metrics(round_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = (
        "best_seen_fitness",
        "visible_mean_fitness",
        "batch_best_fitness",
        "batch_mean_fitness",
        "batch_median_fitness",
    )
    rows = []
    for (condition, round_id), group in round_metrics.groupby(
        ["condition", "round_id"], sort=False
    ):
        for metric in metric_columns:
            values = group[metric].dropna().astype(float)
            if values.empty:
                continue
            rows.append(
                {
                    "condition": condition,
                    "round_id": int(round_id),
                    "query_budget": int(group["query_budget"].iloc[0]),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def build_fold_deltas(final_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison_id, condition, reference in DELTA_COMPARISONS:
        base = final_metrics[final_metrics["condition"] == reference].set_index("fold")
        current = final_metrics[final_metrics["condition"] == condition].set_index("fold")
        for fold in sorted(set(base.index) & set(current.index)):
            for metric in DELTA_METRICS:
                rows.append(
                    {
                        "comparison_id": comparison_id,
                        "condition": condition,
                        "reference": reference,
                        "fold": int(fold),
                        "metric": metric,
                        "condition_value": float(current.loc[fold, metric]),
                        "reference_value": float(base.loc[fold, metric]),
                        "delta": float(current.loc[fold, metric] - base.loc[fold, metric]),
                    }
                )
    return pd.DataFrame(rows)


def build_feature_channel_audit(runs: list[RunArtifact]) -> pd.DataFrame:
    """Verify that all three feature channels executed and passed their child critics."""

    rows = []
    channels = ("physchem", "conservation", "structure")
    feature_runs = [
        run
        for run in _eligible_runs(runs)
        if run.condition in THREE_FEATURE_CONDITIONS
    ]
    for run in feature_runs:
        for round_id in EXPECTED_ROUNDS:
            round_dir = run.path / f"round_{round_id:02d}"
            interaction = read_json(round_dir / "kg_interaction.json")
            executed_steps = [str(value) for value in interaction.get("executed_steps", [])]
            for channel in channels:
                review_path = (
                    round_dir / "subreviews" / channel / "attempt_01.json"
                )
                review = read_json(review_path)
                analysis = review.get("analysis") or {}
                covered_sample_ids = {
                    str(sample_id)
                    for batch in review.get("analysis_batches", [])
                    for sample_id in batch.get("sample_ids", [])
                }
                conversations = sorted(
                    (
                        round_dir
                        / "llm"
                        / f"subscientist_{channel}"
                        / "conversations"
                    ).glob("*.json")
                )
                accepted_conversations = 0
                for path in conversations:
                    payload = read_json(path)
                    if str(payload.get("disposition", "")).lower() == "accepted":
                        accepted_conversations += 1
                rows.append(
                    {
                        "condition": run.condition,
                        "fold": run.fold,
                        "round_id": round_id,
                        "channel": channel,
                        "feature_step_executed": any(
                            step.startswith(f"feature_{channel}_")
                            for step in executed_steps
                        ),
                        "critic_disposition": review.get("disposition"),
                        "request_started": bool(review.get("request_started")),
                        "analysis_summary": analysis.get("analysis_summary"),
                        "finding_count": len(analysis.get("findings", [])),
                        "candidate_hypothesis_count": len(
                            analysis.get("candidate_hypotheses", [])
                        ),
                        "evidence_id_count": len(analysis.get("evidence_ids", [])),
                        "fact_id_count": len(analysis.get("fact_ids", [])),
                        "covered_sample_count": len(covered_sample_ids),
                        "subscientist_conversation_count": len(conversations),
                        "accepted_conversation_count": accepted_conversations,
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id", "channel"]
    ).reset_index(drop=True)
    expected_rows = (
        len(THREE_FEATURE_CONDITIONS)
        * len(EXPECTED_FOLDS)
        * len(EXPECTED_ROUNDS)
        * len(channels)
    )
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} feature-channel audit rows, found {len(frame)}"
        )
    checks = {
        "feature_step_executed": frame["feature_step_executed"].eq(True),
        "critic_disposition": frame["critic_disposition"].eq("APPROVED"),
        "request_started": frame["request_started"].eq(True),
        "accepted_conversation_count": frame["accepted_conversation_count"].gt(0),
    }
    failed = [name for name, values in checks.items() if not bool(values.all())]
    if failed:
        raise ValueError(f"Feature-channel execution audit failed: {failed}")
    return frame


def _structured_kg_counts(db_path: Any) -> tuple[int, int]:
    path = str(db_path)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        entities = int(connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0])
        relations = int(
            connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        )
    return entities, relations


def build_new_condition_runtime_audit(runs: list[RunArtifact]) -> pd.DataFrame:
    """Audit the implemented boundary of agent_only and kg_3features_base."""

    rows = []
    target_conditions = ("agent_only", "kg_3features_base")
    for run in _eligible_runs(runs):
        if run.condition not in target_conditions:
            continue
        config = read_json(run.path / "config.json")
        runtime = config.get("knowledge_runtime") or {}
        local_knowledge = runtime.get("local_knowledge") or {}
        channel_flags = config.get("knowledge_channels") or {}
        enabled_channels = sorted(
            str(channel) for channel, enabled in channel_flags.items() if bool(enabled)
        )
        entities, relations = _structured_kg_counts(run.path / "structured_kg.sqlite")
        selected_evidence_records = 0
        for round_id in EXPECTED_ROUNDS:
            evidence_path = run.path / f"round_{round_id:02d}" / "selected_evidence.json"
            if evidence_path.exists():
                evidence = read_json(evidence_path)
                selected_evidence_records += sum(len(value) for value in evidence.values())
        knowledge_enabled = bool(config.get("knowledge_enabled"))
        kg_interaction_enabled = bool((config.get("kg_interaction") or {}).get("enabled"))
        hierarchical_enabled = bool(
            (config.get("hierarchical_hypothesis") or {}).get("enabled")
        )
        local_rag_enabled = bool(local_knowledge.get("enabled"))
        remote_context_enabled = bool(local_knowledge.get("allow_remote_context"))
        active_learning_enabled = bool(
            (config.get("active_learning") or {}).get("enabled")
        )
        if run.condition == "agent_only":
            contract_ok = (
                not knowledge_enabled
                and not kg_interaction_enabled
                and not hierarchical_enabled
                and not local_rag_enabled
                and not remote_context_enabled
                and not enabled_channels
                and entities == 0
                and relations == 0
                and selected_evidence_records == 0
            )
        else:
            contract_ok = (
                knowledge_enabled
                and kg_interaction_enabled
                and hierarchical_enabled
                and not local_rag_enabled
                and not remote_context_enabled
                and set(enabled_channels)
                == {"conservation", "kg", "physchem", "structure"}
                and entities > 0
                and relations > 0
                and selected_evidence_records == 48
            )
        rows.append(
            {
                "condition": run.condition,
                "fold": run.fold,
                "knowledge_enabled": knowledge_enabled,
                "kg_interaction_enabled": kg_interaction_enabled,
                "hierarchical_hypothesis_enabled": hierarchical_enabled,
                "local_rag_enabled": local_rag_enabled,
                "allow_remote_context": remote_context_enabled,
                "active_learning_enabled": active_learning_enabled,
                "enabled_channels": ",".join(enabled_channels),
                "kg_entity_count": entities,
                "kg_relation_count": relations,
                "selected_evidence_records": selected_evidence_records,
                "condition_contract_ok": contract_ok,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["condition", "fold"]).reset_index(drop=True)
    if len(frame) != len(target_conditions) * len(EXPECTED_FOLDS):
        raise ValueError(f"Expected 6 new-condition audit rows, found {len(frame)}")
    if not frame["condition_contract_ok"].eq(True).all():
        failed = frame.loc[
            ~frame["condition_contract_ok"].eq(True), ["condition", "fold"]
        ].to_dict("records")
        raise ValueError(f"New-condition runtime contract failed: {failed}")
    return frame


def build_feature_rag_interaction_deltas(final_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute fold-aligned feature-by-RAG difference-in-differences."""

    indexed = final_metrics.set_index(["condition", "fold"])
    rows = []
    for fold in EXPECTED_FOLDS:
        for metric in DELTA_METRICS:
            feature_without_rag = float(
                indexed.loc[("kg_3features_base", fold), metric]
                - indexed.loc[("kg_base", fold), metric]
            )
            feature_with_rag = float(
                indexed.loc[("kg_3features_rag", fold), metric]
                - indexed.loc[("kg_base_rag", fold), metric]
            )
            rag_without_features = float(
                indexed.loc[("kg_base_rag", fold), metric]
                - indexed.loc[("kg_base", fold), metric]
            )
            rag_with_features = float(
                indexed.loc[("kg_3features_rag", fold), metric]
                - indexed.loc[("kg_3features_base", fold), metric]
            )
            rows.append(
                {
                    "fold": fold,
                    "metric": metric,
                    "feature_effect_without_rag": feature_without_rag,
                    "feature_effect_with_rag": feature_with_rag,
                    "rag_effect_without_features": rag_without_features,
                    "rag_effect_with_features": rag_with_features,
                    "feature_by_rag_interaction": feature_with_rag
                    - feature_without_rag,
                }
            )
    return pd.DataFrame(rows)


def aggregate_feature_rag_interaction_deltas(
    interaction: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    value_columns = (
        "feature_effect_without_rag",
        "feature_effect_with_rag",
        "rag_effect_without_features",
        "rag_effect_with_features",
        "feature_by_rag_interaction",
    )
    for metric, group in interaction.groupby("metric", sort=False):
        for effect in value_columns:
            values = group[effect].astype(float)
            rows.append(
                {
                    "metric": metric,
                    "effect": effect,
                    "mean_delta": float(values.mean()),
                    "sd_delta": float(values.std(ddof=1)),
                    "min_delta": float(values.min()),
                    "max_delta": float(values.max()),
                    "positive_folds": int((values > 0).sum()),
                    "zero_folds": int((values == 0).sum()),
                    "negative_folds": int((values < 0).sum()),
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def aggregate_fold_deltas(fold_deltas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparison_id, condition, reference, metric), group in fold_deltas.groupby(
        ["comparison_id", "condition", "reference", "metric"], sort=False
    ):
        values = group["delta"].astype(float)
        rows.append(
            {
                "comparison_id": comparison_id,
                "condition": condition,
                "reference": reference,
                "metric": metric,
                "mean_delta": float(values.mean()),
                "sd_delta": float(values.std(ddof=1)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "positive_folds": int((values > 0).sum()),
                "zero_folds": int((values == 0).sum()),
                "negative_folds": int((values < 0).sum()),
                "n": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_pool_overlap(runs: list[RunArtifact]) -> pd.DataFrame:
    pools: dict[tuple[str, int, int], set[str]] = {}
    for run in _eligible_runs(runs):
        for round_id in EXPECTED_ROUNDS:
            receipt = read_json(
                run.path / f"round_{round_id:02d}" / "candidate_pool_receipt.json"
            )
            pools[(run.condition, run.fold, round_id)] = set(receipt["candidate_ids"])
    rows = []
    for fold in sorted({key[1] for key in pools}):
        for round_id in EXPECTED_ROUNDS:
            available = [
                condition
                for condition in CONDITION_ORDER
                if (condition, fold, round_id) in pools
            ]
            for condition_a, condition_b in combinations(available, 2):
                pool_a = pools[(condition_a, fold, round_id)]
                pool_b = pools[(condition_b, fold, round_id)]
                overlap = len(pool_a & pool_b)
                union = len(pool_a | pool_b)
                rows.append(
                    {
                        "fold": fold,
                        "round_id": round_id,
                        "condition_a": condition_a,
                        "condition_b": condition_b,
                        "n_a": len(pool_a),
                        "n_b": len(pool_b),
                        "overlap": overlap,
                        "jaccard": overlap / union if union else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def aggregate_candidate_pool_overlap(pool_overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (condition_a, condition_b), group in pool_overlap.groupby(
        ["condition_a", "condition_b"], sort=False
    ):
        values = group["jaccard"].astype(float)
        rows.append(
            {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "mean_jaccard": float(values.mean()),
                "sd_jaccard": float(values.std(ddof=1)),
                "min_jaccard": float(values.min()),
                "max_jaccard": float(values.max()),
                "n_fold_round_pairs": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def build_active_learning_audit(runs: list[RunArtifact]) -> pd.DataFrame:
    rows = []
    for run in _eligible_runs(runs):
        if run.condition != "kg_base_al":
            continue
        for round_id in EXPECTED_ROUNDS:
            round_dir = run.path / f"round_{round_id:02d}"
            acquisition = read_json(round_dir / "active_learning_acquisition.json")
            posterior = read_json(round_dir / "active_learning_posterior.json")
            selection = acquisition["selection"]
            calibration = posterior["calibration"]
            quotas = selection["quotas"]
            selected_by_arm = selection["selected_by_arm"]
            rows.append(
                {
                    "condition": run.condition,
                    "fold": run.fold,
                    "round_id": round_id,
                    "module": posterior.get("module"),
                    "plugin": selection.get("plugin"),
                    "calibration_status": calibration.get("status"),
                    "visible_observations": calibration.get("visible_observations"),
                    "training_observations": calibration.get("training_observations"),
                    "calibration_observations": calibration.get(
                        "calibration_observations"
                    ),
                    "exploitation_quota": quotas.get("exploitation"),
                    "exploration_quota": quotas.get("exploration"),
                    "knowledge_quota": quotas.get("knowledge"),
                    "exploitation_selected": len(
                        selected_by_arm.get("exploitation", [])
                    ),
                    "exploration_selected": len(selected_by_arm.get("exploration", [])),
                    "knowledge_selected": len(selected_by_arm.get("knowledge", [])),
                    "selected_total": len(selection.get("selected_ids", [])),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["fold", "round_id"]).reset_index(drop=True)
    expected_visible = 96 + (frame["round_id"].astype(int) - 1) * 16
    checks = {
        "calibration_status": frame["calibration_status"].eq("calibrated"),
        "visible_observations": frame["visible_observations"].astype(int).eq(
            expected_visible
        ),
        "exploitation_quota": frame["exploitation_quota"].astype(int).eq(8),
        "exploration_quota": frame["exploration_quota"].astype(int).eq(4),
        "knowledge_quota": frame["knowledge_quota"].astype(int).eq(4),
        "exploitation_selected": frame["exploitation_selected"].astype(int).eq(8),
        "exploration_selected": frame["exploration_selected"].astype(int).eq(4),
        "knowledge_selected": frame["knowledge_selected"].astype(int).eq(4),
        "selected_total": frame["selected_total"].astype(int).eq(16),
    }
    failed = [name for name, values in checks.items() if not bool(values.all())]
    if failed:
        raise ValueError(f"Active-learning execution audit failed: {failed}")
    return frame


def compact_metric_summary(aggregate: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in METRIC_DIRECTIONS:
        subset = aggregate[aggregate["metric"] == metric]
        summary[metric] = {
            row["condition"]: {
                "mean": float(row["mean"]),
                "sd": float(row["sd"]),
                "n": int(row["n"]),
            }
            for _, row in subset.iterrows()
        }
    return summary
