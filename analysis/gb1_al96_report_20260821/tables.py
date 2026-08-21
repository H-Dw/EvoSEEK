"""Publication-facing Markdown tables with declared metric directions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import CONDITION_ORDER, METRIC_DIRECTIONS


DISPLAY_NAMES = {
    "random": "Random",
    "fitness_direct": "Kermut direct",
    "agent_only": "Agent only",
    "kg_base": "KG base",
    "kg_3features_base": "KG + 3-channel",
    "kg_base_rag": "KG + RAG",
    "kg_base_al": "KG + active learning",
    "kg_3features_rag": "KG + 3-channel + RAG",
}

COMPARISON_NAMES = {
    "kg_memory": "KG base − Agent only",
    "three_channels_without_rag": "3-channel − KG base",
    "rag_without_three_channels": "KG + RAG − KG base",
    "three_channels_with_rag": "3-channel + RAG − KG + RAG",
    "rag_with_three_channels": "3-channel + RAG − 3-channel",
    "active_learning": "KG + active learning − KG base",
}


def _format_value(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def _decorate_ranks(values: dict[str, tuple[float, float]], direction: str) -> dict[str, str]:
    finite = {
        key: value for key, value in values.items() if np.isfinite(float(value[0]))
    }
    unique = sorted(
        {float(value[0]) for value in finite.values()},
        reverse=direction == "higher",
    )
    first = unique[0] if unique else None
    second = unique[1] if len(unique) > 1 else None
    decorated: dict[str, str] = {}
    for condition, (mean, sd) in values.items():
        text = _format_value(mean, sd)
        if first is not None and np.isclose(mean, first, rtol=1e-9, atol=1e-12):
            text = f"**{text}**"
        elif second is not None and np.isclose(mean, second, rtol=1e-9, atol=1e-12):
            text = f"<u>{text}</u>"
        decorated[condition] = text
    return decorated


def _metric_table(
    aggregate: pd.DataFrame,
    metrics: list[tuple[str, str, int]],
) -> str:
    headers = ["策略"] + [label for _, label, _ in metrics]
    directions = ["方向"] + [
        "↑" if METRIC_DIRECTIONS[metric] == "higher" else "↓"
        for metric, _, _ in metrics
    ]
    formatted: dict[str, dict[str, str]] = {}
    for metric, _, digits in metrics:
        subset = aggregate[aggregate["metric"] == metric]
        values = {
            str(row["condition"]): (float(row["mean"]), float(row["sd"]))
            for _, row in subset.iterrows()
        }
        ranked = _decorate_ranks(values, METRIC_DIRECTIONS[metric])
        if digits != 3:
            ranked = {
                condition: value.replace(
                    _format_value(*values[condition], digits=3),
                    _format_value(*values[condition], digits=digits),
                )
                for condition, value in ranked.items()
            }
        formatted[metric] = ranked
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
        "| " + " | ".join(directions) + " |",
    ]
    for condition in CONDITION_ORDER:
        cells = [DISPLAY_NAMES[condition]] + [
            formatted[metric][condition] for metric, _, _ in metrics
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_performance_tables(aggregate: pd.DataFrame) -> str:
    discovery = [
        ("final_best_seen", "最终 best-seen", 3),
        ("delta_best_seen", "best-seen 增量", 3),
        ("best_seen_aulc", "best-seen AULC", 3),
        ("r3_batch_best", "R3 batch best", 3),
        ("r3_batch_mean", "R3 batch mean", 3),
        ("r3_batch_median", "R3 batch median", 3),
    ]
    ranking = [
        ("spearman", "Spearman", 3),
        ("pearson", "Pearson", 3),
        ("ndcg", "NDCG@10", 3),
        ("top_k_hit", "Top-k hit", 3),
        ("top_k_recall", "Top-k recall", 3),
        ("regret_at_k", "Regret@10", 3),
    ]
    error = [
        ("mse", "MSE", 3),
        ("rmse", "RMSE", 3),
        ("interval_90_coverage_error", "\\|Coverage−0.90\\|", 3),
        ("gaussian_nll", "Gaussian NLL", 3),
    ]
    return "\n\n".join(
        [
            "# Performance tables",
            "三折均值 ± 样本标准差。粗体为第一名，下划线为第二名；箭头给出指标优化方向。",
            "## Wet-fitness discovery metrics\n\n" + _metric_table(aggregate, discovery),
            "## Isolated-test ranking metrics\n\n" + _metric_table(aggregate, ranking),
            "## Isolated-test error and calibration metrics\n\n" + _metric_table(aggregate, error),
        ]
    ) + "\n"


def build_status_table(run_status: pd.DataFrame) -> str:
    lines = [
        "# Completion matrix",
        "",
        "| 条件 | fold 0 | fold 1 | fold 2 | 正式纳入 |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITION_ORDER:
        subset = run_status[run_status["condition"] == condition]
        fold_cells = []
        for fold in (0, 1, 2):
            rows = subset[subset["fold"] == fold]
            ok = bool(rows["eligible"].any()) if not rows.empty else False
            fold_cells.append("通过" if ok else "失败/未纳入")
        included = int(subset["eligible"].sum())
        lines.append(
            f"| {DISPLAY_NAMES[condition]} | "
            + " | ".join(fold_cells)
            + f" | {included}/3 |"
        )
    superseded = run_status[
        run_status["exclusion_reason"].eq("superseded_failed_run")
    ]
    lines.extend(
        [
            "",
            "> `kg_3features_rag`、`kg_3features_base` 与 `agent_only` 均以新完成的三折结果正式纳入；旧目录中的失败 `kg_3features_rag` 运行仅作为被替代的审计记录。"
            if len(superseded) == 3
            else "",
            "",
        ]
    )
    return "\n".join(lines)


def build_ablation_tables(
    delta_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    feature_audit: pd.DataFrame,
    runtime_audit: pd.DataFrame,
) -> str:
    """Build ready-to-paste module, interaction, and execution-audit tables."""

    metric_labels = {
        "final_best_seen": "Final best-seen Δ",
        "best_seen_aulc": "AULC Δ",
        "r3_batch_mean": "R3 mean Δ",
        "r3_batch_median": "R3 median Δ",
    }
    lines = [
        "# Ablation and runtime-audit tables",
        "",
        "三折 fold-aligned 差值，均值 ± 样本标准差；括号内为正/持平/负 folds。",
        "",
        "## Module comparisons",
        "",
        "| Comparison | Final best-seen Δ | AULC Δ | R3 mean Δ | R3 median Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    metric_order = tuple(metric_labels)
    for comparison_id, label in COMPARISON_NAMES.items():
        group = delta_summary[delta_summary["comparison_id"] == comparison_id]
        cells = []
        for metric in metric_order:
            row = group[group["metric"] == metric].iloc[0]
            cells.append(
                f"{float(row['mean_delta']):+.3f} ± {float(row['sd_delta']):.3f} "
                f"({int(row['positive_folds'])}/{int(row['zero_folds'])}/{int(row['negative_folds'])})"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Feature-by-RAG interaction",
            "",
            "| Metric | Feature effect without RAG | Feature effect with RAG | Interaction |",
            "|---|---:|---:|---:|",
        ]
    )
    for metric in metric_order:
        group = interaction_summary[interaction_summary["metric"] == metric].set_index(
            "effect"
        )
        cells = []
        for effect in (
            "feature_effect_without_rag",
            "feature_effect_with_rag",
            "feature_by_rag_interaction",
        ):
            row = group.loc[effect]
            cells.append(
                f"{float(row['mean_delta']):+.3f} ± {float(row['sd_delta']):.3f}"
            )
        lines.append(f"| {metric_labels[metric]} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Three-channel execution audit",
            "",
            "| Condition | Channel | Approved paths | Covered samples | Findings | Candidate hypotheses |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    grouped = feature_audit.groupby(["condition", "channel"], sort=False)
    for (condition, channel), group in grouped:
        approved = int(group["critic_disposition"].eq("APPROVED").sum())
        lines.append(
            f"| {condition} | {channel} | {approved}/{len(group)} | "
            f"{int(group['covered_sample_count'].sum())} | "
            f"{int(group['finding_count'].sum())} | "
            f"{int(group['candidate_hypothesis_count'].sum())} |"
        )

    lines.extend(
        [
            "",
            "## New-condition runtime boundary",
            "",
            "| Condition | KG enabled | Hierarchical | Local RAG | Enabled channels | KG entities | KG relations | Selection evidence / fold |",
            "|---|---:|---:|---:|---|---:|---:|---:|",
        ]
    )
    for condition, group in runtime_audit.groupby("condition", sort=False):
        entity_range = (
            f"{int(group['kg_entity_count'].min())}–{int(group['kg_entity_count'].max())}"
        )
        relation_range = (
            f"{int(group['kg_relation_count'].min())}–{int(group['kg_relation_count'].max())}"
        )
        evidence_values = "/".join(
            str(int(value)) for value in group.sort_values("fold")["selected_evidence_records"]
        )
        channels = str(group["enabled_channels"].fillna("").iloc[0]) or "none"
        lines.append(
            f"| {condition} | {bool(group['knowledge_enabled'].all())} | "
            f"{bool(group['hierarchical_hypothesis_enabled'].all())} | "
            f"{bool(group['local_rag_enabled'].all())} | {channels} | "
            f"{entity_range} | {relation_range} | {evidence_values} |"
        )
    return "\n".join(lines) + "\n"
