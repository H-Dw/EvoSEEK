"""Diagnose why cumulative best rises while batch mean and median decline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import CONDITION_ORDER, EXPECTED_ROUNDS, REPO_ROOT
from io_artifacts import RunArtifact, read_json


KG_CONDITIONS = ("kg_base", "kg_base_rag", "kg_base_al")
LOW_FITNESS_THRESHOLD = 0.05
HIGH_FITNESS_THRESHOLD = 2.0
INITIAL_BEST = 4.07312564853


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    return float(x.astype(float).corr(y.astype(float)))


def _distribution_row(values: pd.Series) -> dict[str, float]:
    values = values.astype(float)
    max_index = values.idxmax()
    without_max = values.drop(max_index)
    positive_sum = float(values.clip(lower=0).sum())
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "q10": float(values.quantile(0.10)),
        "q25": float(values.quantile(0.25)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q75": float(values.quantile(0.75)),
        "q90": float(values.quantile(0.90)),
        "max": float(values.max()),
        "sd": float(values.std(ddof=1)),
        "mean_without_batch_max": float(without_max.mean()),
        "max_minus_median": float(values.max() - values.median()),
        "top1_positive_mass_share": (
            float(values.max() / positive_sum) if positive_sum > 0 else np.nan
        ),
        "fraction_le_0_05": float((values <= LOW_FITNESS_THRESHOLD).mean()),
        "fraction_ge_2": float((values >= HIGH_FITNESS_THRESHOLD).mean()),
        "fraction_above_initial_best": float((values > INITIAL_BEST).mean()),
    }


def build_batch_distribution_by_fold(
    candidates: pd.DataFrame, round_metrics: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = candidates[candidates["condition"].isin(KG_CONDITIONS)].copy()
    for (condition, fold, round_id), group in data.groupby(
        ["condition", "fold", "round_id"], sort=False
    ):
        metrics = round_metrics[
            (round_metrics["condition"] == condition)
            & (round_metrics["fold"] == fold)
        ].set_index("round_id")
        current = metrics.loc[int(round_id)]
        previous = metrics.loc[int(round_id) - 1]
        record = group.loc[group["wet_fitness"].idxmax()]
        row: dict[str, Any] = {
            "condition": condition,
            "fold": int(fold),
            "round_id": int(round_id),
            "previous_best_seen": float(previous["best_seen_fitness"]),
            "best_seen_fitness": float(current["best_seen_fitness"]),
            "best_seen_increment": float(
                current["best_seen_fitness"] - previous["best_seen_fitness"]
            ),
            "new_record": bool(
                current["best_seen_fitness"] > previous["best_seen_fitness"] + 1e-12
            ),
            "round_max_variant": record["variant"],
            "round_max_variant_id": record["variant_id"],
            "round_max_fitness": float(record["wet_fitness"]),
        }
        row.update(_distribution_row(group["wet_fitness"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def build_pooled_batch_distribution(
    candidates: pd.DataFrame,
    round_metrics: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = candidates[candidates["condition"].isin(KG_CONDITIONS)].copy()
    for (condition, round_id), group in data.groupby(
        ["condition", "round_id"], sort=False
    ):
        fold_rows = by_fold[
            (by_fold["condition"] == condition)
            & (by_fold["round_id"] == round_id)
        ]
        metrics = round_metrics[
            (round_metrics["condition"] == condition)
            & (round_metrics["round_id"] == round_id)
        ]
        row: dict[str, Any] = {
            "condition": condition,
            "round_id": int(round_id),
            "folds": int(fold_rows["fold"].nunique()),
            "new_record_folds": int(fold_rows["new_record"].sum()),
            "best_seen_mean": float(metrics["best_seen_fitness"].mean()),
            "best_seen_sd": float(metrics["best_seen_fitness"].std(ddof=1)),
            "mean_of_fold_batch_means": float(metrics["batch_mean_fitness"].mean()),
            "mean_of_fold_batch_medians": float(
                metrics["batch_median_fitness"].mean()
            ),
        }
        row.update(_distribution_row(group["wet_fitness"]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id"]
    ).reset_index(drop=True)


def build_score_alignment(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = candidates[candidates["condition"].isin(KG_CONDITIONS)].copy()
    for (condition, fold, round_id), group in data.groupby(
        ["condition", "fold", "round_id"], sort=False
    ):
        rows.append(
            {
                "condition": condition,
                "fold": int(fold),
                "round_id": int(round_id),
                "selected_predicted_mean": float(group["fitness_mean"].mean()),
                "selected_predicted_median": float(group["fitness_mean"].median()),
                "selected_uncertainty_mean": float(group["fitness_std"].mean()),
                "acquisition_mean": float(group["acquisition_score"].mean()),
                "knowledge_mean": float(group["knowledge_score"].mean()),
                "wet_mean": float(group["wet_fitness"].mean()),
                "wet_median": float(group["wet_fitness"].median()),
                "dry_minus_wet_mean": float(group["dry_wet_gap"].mean()),
                "prediction_wet_pearson": _safe_corr(
                    group["fitness_mean"], group["wet_fitness"]
                ),
                "acquisition_wet_pearson": _safe_corr(
                    group["acquisition_score"], group["wet_fitness"]
                ),
                "knowledge_wet_pearson": _safe_corr(
                    group["knowledge_score"], group["wet_fitness"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def aggregate_score_alignment(score_alignment: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        column
        for column in score_alignment.columns
        if column not in {"condition", "fold", "round_id"}
    ]
    rows = []
    for (condition, round_id), group in score_alignment.groupby(
        ["condition", "round_id"], sort=False
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "round_id": int(round_id),
            "n_folds": int(group["fold"].nunique()),
        }
        for column in value_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_sd"] = float(group[column].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id"]
    ).reset_index(drop=True)


def build_arm_outcomes(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = candidates[candidates["condition"].isin(KG_CONDITIONS)].copy()
    for (condition, fold, round_id, run_id), group in data.groupby(
        ["condition", "fold", "round_id", "run_id"], sort=False
    ):
        run_path = REPO_ROOT / str(group["run_path"].iloc[0])
        round_dir = run_path / f"round_{int(round_id):02d}"
        if condition == "kg_base_al":
            receipt = read_json(round_dir / "active_learning_acquisition.json")
            selection_receipt = receipt["selection"]
            selected_by_arm = dict(selection_receipt["selected_by_arm"])
            receipt_selected_ids = set(selection_receipt["selected_ids"])
            receipt_type = "active_learning_8_4_4"
        else:
            receipt = read_json(round_dir / "agent_quota_acquisition.json")
            selected_by_arm = dict(receipt["selected_by_arm"])
            receipt_selected_ids = set(receipt["selected_ids"])
            receipt_type = "agent_uq_8_3_3_2"
        assigned_ids = {
            variant_id for ids in selected_by_arm.values() for variant_id in ids
        }
        unassigned_ids = sorted(receipt_selected_ids - assigned_ids)
        if unassigned_ids:
            selected_by_arm["fallback_fill"] = unassigned_ids
        all_ids = [variant_id for ids in selected_by_arm.values() for variant_id in ids]
        if len(all_ids) != 16 or len(set(all_ids)) != 16:
            raise ValueError(
                f"{run_id} round {round_id}: acquisition arms do not partition 16 IDs"
            )
        if set(all_ids) != receipt_selected_ids or receipt_selected_ids != set(
            group["variant_id"]
        ):
            raise ValueError(
                f"{run_id} round {round_id}: arm IDs differ from revealed batch"
            )
        for arm, variant_ids in selected_by_arm.items():
            arm_rows = group[group["variant_id"].isin(variant_ids)]
            rows.append(
                {
                    "condition": condition,
                    "fold": int(fold),
                    "round_id": int(round_id),
                    "run_id": run_id,
                    "receipt_type": receipt_type,
                    "arm": arm,
                    "n": int(len(arm_rows)),
                    "wet_mean": float(arm_rows["wet_fitness"].mean()),
                    "wet_median": float(arm_rows["wet_fitness"].median()),
                    "wet_max": float(arm_rows["wet_fitness"].max()),
                    "predicted_mean": float(arm_rows["fitness_mean"].mean()),
                    "fraction_le_0_05": float(
                        (arm_rows["wet_fitness"] <= LOW_FITNESS_THRESHOLD).mean()
                    ),
                    "fraction_ge_2": float(
                        (arm_rows["wet_fitness"] >= HIGH_FITNESS_THRESHOLD).mean()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id", "arm"]
    ).reset_index(drop=True)


def aggregate_arm_outcomes(arm_outcomes: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "wet_mean",
        "wet_median",
        "wet_max",
        "predicted_mean",
        "fraction_le_0_05",
        "fraction_ge_2",
    )
    rows = []
    for (condition, round_id, arm), group in arm_outcomes.groupby(
        ["condition", "round_id", "arm"], sort=False
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "round_id": int(round_id),
            "arm": arm,
            "n_folds": int(group["fold"].nunique()),
            "quota_per_fold_mean": float(group["n"].mean()),
            "quota_per_fold_min": int(group["n"].min()),
            "quota_per_fold_max": int(group["n"].max()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id", "arm"]
    ).reset_index(drop=True)


def build_arm_mix_summary(arm_outcomes: pd.DataFrame) -> pd.DataFrame:
    """Compare the full batch with its primary optimization-oriented arms."""

    rows = []
    for (condition, round_id), group in arm_outcomes.groupby(
        ["condition", "round_id"], sort=False
    ):
        if condition == "kg_base_al":
            core_names = {"exploitation", "knowledge"}
        else:
            core_names = {"hypothesis_target"}
        core = group[group["arm"].isin(core_names)]
        supporting = group[~group["arm"].isin(core_names)]

        def weighted_mean(frame: pd.DataFrame) -> float:
            return float((frame["wet_mean"] * frame["n"]).sum() / frame["n"].sum())

        full_mean = weighted_mean(group)
        core_mean = weighted_mean(core)
        supporting_mean = weighted_mean(supporting)
        rows.append(
            {
                "condition": condition,
                "round_id": int(round_id),
                "core_arms": "+".join(sorted(core_names)),
                "core_candidate_fraction": float(core["n"].sum() / group["n"].sum()),
                "full_batch_wet_mean": full_mean,
                "core_arms_wet_mean": core_mean,
                "supporting_arms_wet_mean": supporting_mean,
                "batch_mix_gap_core_minus_full": float(core_mean - full_mean),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id"]
    ).reset_index(drop=True)


def build_candidate_pool_diagnostics(runs: list[RunArtifact]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = [
        run for run in runs if run.eligible and run.condition in KG_CONDITIONS
    ]
    for run in eligible:
        previous_pool: set[str] | None = None
        initial_catalog: int | None = None
        for round_id in EXPECTED_ROUNDS:
            round_dir = run.path / f"round_{round_id:02d}"
            receipt = read_json(round_dir / "candidate_pool_receipt.json")
            pool = set(receipt["candidate_ids"])
            if initial_catalog is None:
                initial_catalog = int(receipt["catalog_candidate_count"])
            overlap = len(pool & previous_pool) if previous_pool is not None else 0
            union = len(pool | previous_pool) if previous_pool is not None else 0
            design_scores = read_json(round_dir / "design_scores.json")
            score_frame = pd.DataFrame(design_scores)
            selected_ids = set(
                pd.read_csv(round_dir / "selection.csv")["variant_id"].astype(str)
            )
            score_frame["selected"] = score_frame["variant_id"].isin(selected_ids)
            selected_scores = score_frame[score_frame["selected"]]
            unselected_scores = score_frame[~score_frame["selected"]]
            catalog_count = int(receipt["catalog_candidate_count"])
            rows.append(
                {
                    "condition": run.condition,
                    "fold": run.fold,
                    "round_id": round_id,
                    "catalog_candidate_count": catalog_count,
                    "catalog_removed_since_round1": int(initial_catalog - catalog_count),
                    "catalog_removed_fraction": float(
                        (initial_catalog - catalog_count) / initial_catalog
                    ),
                    "pool_size": int(len(pool)),
                    "previous_round_pool_overlap": int(overlap),
                    "previous_round_pool_jaccard": (
                        float(overlap / union) if union else np.nan
                    ),
                    "sampling_strategy": receipt.get("sampling_strategy"),
                    "pool_utility_mean": float(score_frame["utility"].mean()),
                    "pool_utility_max": float(score_frame["utility"].max()),
                    "selected_utility_mean": float(selected_scores["utility"].mean()),
                    "unselected_utility_mean": float(
                        unselected_scores["utility"].mean()
                    ),
                    "pool_evidence_mean": float(score_frame["evidence_score"].mean()),
                    "pool_uncertainty_mean": float(score_frame["uncertainty"].mean()),
                }
            )
            previous_pool = pool
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def build_motif_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    data = candidates[candidates["condition"].isin(KG_CONDITIONS)].copy()
    variant = data["variant"].astype(str)
    data["preferred_39"] = variant.str[0].isin(["I", "L"])
    data["preferred_40"] = variant.str[1].isin(["W", "Y", "F", "H"])
    data["preferred_41"] = variant.str[2].eq("G")
    data["preferred_54"] = variant.str[3].isin(["C", "A", "V"])
    preferred_columns = [
        "preferred_39",
        "preferred_40",
        "preferred_41",
        "preferred_54",
    ]
    data["preferred_site_count"] = data[preferred_columns].sum(axis=1)
    rows = []
    for (condition, fold, round_id), group in data.groupby(
        ["condition", "fold", "round_id"], sort=False
    ):
        g41 = group[group["preferred_41"]]
        non_g41 = group[~group["preferred_41"]]
        rows.append(
            {
                "condition": condition,
                "fold": int(fold),
                "round_id": int(round_id),
                "preferred_site_count_mean": float(
                    group["preferred_site_count"].mean()
                ),
                "fraction_position_41_G": float(group["preferred_41"].mean()),
                "fraction_all_four_preferred": float(
                    (group["preferred_site_count"] == 4).mean()
                ),
                "wet_mean_position_41_G": (
                    float(g41["wet_fitness"].mean()) if len(g41) else np.nan
                ),
                "wet_mean_position_41_non_G": (
                    float(non_g41["wet_fitness"].mean()) if len(non_g41) else np.nan
                ),
                "preferred_count_wet_pearson": _safe_corr(
                    group["preferred_site_count"], group["wet_fitness"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def aggregate_motif_diagnostics(motif_by_fold: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        column
        for column in motif_by_fold.columns
        if column not in {"condition", "fold", "round_id"}
    ]
    rows = []
    for (condition, round_id), group in motif_by_fold.groupby(
        ["condition", "round_id"], sort=False
    ):
        row: dict[str, Any] = {
            "condition": condition,
            "round_id": int(round_id),
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id"]
    ).reset_index(drop=True)


def selected_candidate_diagnostics(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "fold",
        "round_id",
        "selection_order",
        "variant_id",
        "variant",
        "mutation_notation",
        "wet_fitness",
        "fitness_mean",
        "fitness_std",
        "acquisition_score",
        "knowledge_score",
        "selection_driver",
    ]
    return candidates[candidates["condition"].isin(KG_CONDITIONS)][columns].copy()


def compact_diagnostic_summary(
    pooled: pd.DataFrame,
    score_summary: pd.DataFrame,
    pool_diagnostics: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in KG_CONDITIONS:
        p = pooled[pooled["condition"] == condition].set_index("round_id")
        s = score_summary[score_summary["condition"] == condition].set_index(
            "round_id"
        )
        pools = pool_diagnostics[pool_diagnostics["condition"] == condition]
        result[condition] = {
            "record_folds_by_round": {
                str(round_id): int(p.loc[round_id, "new_record_folds"])
                for round_id in EXPECTED_ROUNDS
            },
            "round1_to_round3": {
                "pooled_median_change": float(p.loc[3, "median"] - p.loc[1, "median"]),
                "near_zero_fraction_change": float(
                    p.loc[3, "fraction_le_0_05"]
                    - p.loc[1, "fraction_le_0_05"]
                ),
                "high_fraction_change": float(
                    p.loc[3, "fraction_ge_2"] - p.loc[1, "fraction_ge_2"]
                ),
                "selected_predicted_mean_change": float(
                    s.loc[3, "selected_predicted_mean_mean"]
                    - s.loc[1, "selected_predicted_mean_mean"]
                ),
            },
            "maximum_catalog_removed_fraction": float(
                pools["catalog_removed_fraction"].max()
            ),
        }
    return result
