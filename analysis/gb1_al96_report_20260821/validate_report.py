"""Check that the manuscript chapter is synchronized with generated outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    CONDITION_ORDER,
    FIGURE_DIR,
    MUTATION_BEHAVIOR_DIR,
    MUTATION_BEHAVIOR_SOURCE_DIR,
    REPO_ROOT,
    SOURCE_DATA_DIR,
)


REPORT_PATH = REPO_ROOT / "docs" / "GB1实验报告-问题定义与无泄漏评估协议.md"


def _chapter_four(text: str) -> str:
    start = text.index("## 4. 主要结果、对照实验与消融矩阵")
    end = text.index("## 5. 局限、未来工作及来源声明")
    return text[start:end]


def main() -> int:
    report = REPORT_PATH.read_text(encoding="utf-8")
    chapter = _chapter_four(report)
    aggregate = pd.read_csv(SOURCE_DATA_DIR / "final_metrics_mean_sd.csv")
    discovery_metrics = (
        "final_best_seen",
        "delta_best_seen",
        "best_seen_aulc",
        "r3_batch_best",
        "r3_batch_mean",
        "r3_batch_median",
    )
    missing_values = []
    for condition in CONDITION_ORDER:
        for metric in discovery_metrics:
            row = aggregate[
                (aggregate["condition"] == condition)
                & (aggregate["metric"] == metric)
            ].iloc[0]
            token = f"{float(row['mean']):.3f} ± {float(row['sd']):.3f}"
            if token not in chapter:
                missing_values.append(f"{condition}:{metric}:{token}")
    forbidden = (
        "knowledge_agent_qwen_rag",
        "best-seen@96",
        "图 2（结果位）",
        "图 3（结果位）",
    )
    stale = [token for token in forbidden if token in chapter]
    figure_links = (
        "../analysis/gb1_al96_report_20260821/outputs/figures/figure2_fitness_trajectories.png",
        "../analysis/gb1_al96_report_20260821/outputs/figures/figure3_module_deltas.png",
    )
    missing_links = [link for link in figure_links if link not in chapter]
    missing_figures = [
        name
        for name in ("figure2_fitness_trajectories.png", "figure3_module_deltas.png")
        if not (FIGURE_DIR / name).exists()
    ]
    behavior = pd.read_csv(MUTATION_BEHAVIOR_SOURCE_DIR / "round_behavior_by_fold.csv")
    behavior_errors = []
    if len(behavior) != 45:
        behavior_errors.append(f"round_rows={len(behavior)}")
    invariant_columns = {
        "candidate_pool_size": 32,
        "selected_batch_size": 16,
        "candidate_depth_1_count": 0,
        "selected_depth_1_count": 0,
        "new_complete_variant_count_vs_pre_round_visible": 16,
        "new_mutated_position_count_vs_pre_round_visible": 0,
        "new_position_residue_pair_count_vs_pre_round_visible": 0,
    }
    for column, expected in invariant_columns.items():
        if column not in behavior or not (behavior[column] == expected).all():
            behavior_errors.append(f"{column}!={expected}")
    expected_round_three_depths = {
        "kg_base": (5, 25, 18),
        "kg_base_rag": (9, 26, 13),
        "kg_base_al": (5, 38, 5),
        "fitness_direct": (2, 14, 32),
        "random": (2, 9, 37),
    }
    round_three = behavior[behavior["round_id"] == 3]
    missing_round_three_rows = []
    for condition, expected_depths in expected_round_three_depths.items():
        group = round_three[round_three["condition"] == condition]
        observed_depths = tuple(
            int(group[f"selected_depth_{depth}_count"].sum())
            for depth in (2, 3, 4)
        )
        if len(group) != 3 or observed_depths != expected_depths:
            behavior_errors.append(
                f"{condition}:round3_depths={observed_depths},rows={len(group)}"
            )
        row_token = (
            f"| `{condition}` | {expected_depths[0]}/48 | "
            f"{expected_depths[1]}/48 | {expected_depths[2]}/48 |"
        )
        if row_token not in report:
            missing_round_three_rows.append(row_token)
    behavior_markers = (
        "### 2.6 闭池候选生成与逐轮选择语义",
        "#### 4.1.1 每轮突变推荐与选择行为审计",
        "### 附件 C｜Scientist 偏好、候选池位点与实际入选位点",
        "而不是亲本—子代式的开放序列进化",
        "这些数据不支持“越靠后的轮次，实际选择的新突变点越少”",
        "AL96 初始观测共包含 96 个变体",
        "正式三轮候选池和入选批次均不包含单突变",
        "**表 C2｜五种策略第三轮入选批次的突变深度构成。**",
        "`41=[G]`",
    )
    missing_behavior_markers = [
        marker for marker in behavior_markers if marker not in report
    ]
    behavior_outputs = (
        MUTATION_BEHAVIOR_DIR / "mutation_behavior_analysis.md",
        MUTATION_BEHAVIOR_DIR / "mutation_behavior_manifest.json",
        MUTATION_BEHAVIOR_SOURCE_DIR / "position_sets_by_fold_round.csv",
        MUTATION_BEHAVIOR_SOURCE_DIR / "kg_position_sets_wide.csv",
        MUTATION_BEHAVIOR_SOURCE_DIR / "candidate_pool_variants.csv",
        MUTATION_BEHAVIOR_SOURCE_DIR / "selected_variant_lineage_audit.csv",
    )
    missing_behavior_outputs = [
        str(path) for path in behavior_outputs if not path.exists()
    ]
    if (
        missing_values
        or stale
        or missing_links
        or missing_figures
        or behavior_errors
        or missing_behavior_markers
        or missing_round_three_rows
        or missing_behavior_outputs
    ):
        raise ValueError(
            "Report validation failed: "
            f"missing_values={missing_values}, stale={stale}, "
            f"missing_links={missing_links}, missing_figures={missing_figures}, "
            f"behavior_errors={behavior_errors}, "
            f"missing_behavior_markers={missing_behavior_markers}, "
            f"missing_round_three_rows={missing_round_three_rows}, "
            f"missing_behavior_outputs={missing_behavior_outputs}"
        )
    print(
        f"Report validation passed: {len(CONDITION_ORDER) * len(discovery_metrics)} "
        "discovery values, 2 figure links, 45 mutation-behavior rounds, and "
        "5 round-3 mutation-depth rows checked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
