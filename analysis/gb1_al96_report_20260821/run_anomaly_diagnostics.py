"""Reproducible entry point for the best-fitness versus batch-distribution anomaly."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from anomaly_diagnostics import (
    KG_CONDITIONS,
    aggregate_arm_outcomes,
    aggregate_motif_diagnostics,
    aggregate_score_alignment,
    build_arm_mix_summary,
    build_arm_outcomes,
    build_batch_distribution_by_fold,
    build_candidate_pool_diagnostics,
    build_motif_diagnostics,
    build_pooled_batch_distribution,
    build_score_alignment,
    compact_diagnostic_summary,
    selected_candidate_diagnostics,
)
from anomaly_plots import plot_anomaly_diagnostics
from cases import build_candidate_table
from config import (
    ANOMALY_DIR,
    ANOMALY_FIGURE_DIR,
    ANOMALY_SOURCE_DATA_DIR,
    REPO_ROOT,
)
from io_artifacts import discover_runs, sha256_file, validate_analysis_matrix
from metrics import build_round_metrics


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _build_report(
    by_fold: pd.DataFrame,
    pooled: pd.DataFrame,
    score_summary: pd.DataFrame,
    arm_summary: pd.DataFrame,
    arm_mix: pd.DataFrame,
    pool_diagnostics: pd.DataFrame,
    motif_summary: pd.DataFrame,
) -> str:
    def row(frame: pd.DataFrame, condition: str, round_id: int) -> pd.Series:
        return frame[
            (frame["condition"] == condition) & (frame["round_id"] == round_id)
        ].iloc[0]

    lines = [
        "# Best fitness 上升而 batch mean/median 下降：异常诊断",
        "",
        "## Figure contract",
        "",
        "- Core conclusion：best-seen 是不可下降的累计极值；本批实验的后期上升由极少数新纪录候选驱动，而批次主体同时向低 fitness 移动。",
        "- Figure archetype：quantitative grid。",
        "- Backend：Python/matplotlib。",
        "- Statistics：3 folds；候选分布每个 condition × round 合并 48 个已揭示候选；不进行显著性检验。",
        "- Reviewer risk：未获得未选中候选的 wet 标签，因此不能把下降严格归因于候选池真值耗竭。",
        "",
        "## 诊断结论",
        "",
        "### 1. 这首先是累计极值与批次分布的统计口径差异",
        "",
        "`best_seen_fitness` 定义为截至当前轮所有已测候选的累计最大值，因此数学上只能持平或上升；batch mean 和 median 只描述当轮 16 个新候选。一个候选刷新纪录即可抬高 best-seen，即使其余 15 个候选整体变差。实际新纪录也高度稀疏：",
        "",
        "| 条件 | Round 1 新纪录 folds | Round 2 | Round 3 |",
        "|---|---:|---:|---:|",
    ]
    for condition in KG_CONDITIONS:
        values = [int(row(pooled, condition, r)["new_record_folds"]) for r in (1, 2, 3)]
        lines.append(
            f"| `{condition}` | {values[0]}/3 | {values[1]}/3 | {values[2]}/3 |"
        )
    round3_records = by_fold[
        (by_fold["round_id"] == 3) & by_fold["new_record"]
    ].sort_values(["condition", "fold"])
    record_descriptions = [
        f"`{item.condition}` fold {int(item.fold)} 的 `{item.round_max_variant}`"
        f"（{float(item.round_max_fitness):.3f}）"
        for item in round3_records.itertuples()
    ]
    record_sentence = "和".join(record_descriptions)
    lines.extend(
        [
            "",
            f"因此，图中的‘逐轮上升’不代表每个 fold 或整批候选持续改善。Round 3 只有 {record_sentence} 刷新既有纪录；`kg_base_rag` 在 Round 3 没有任何 fold 刷新纪录。",
            "",
            "### 2. mean 和 median 下降来自批次主体真实下移，而非单个离群值造成",
            "",
            "| 条件 | pooled median R1 → R3 | fitness ≤0.05 | fitness ≥2 | 去除每轮最大值后的 mean R1 → R3 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in KG_CONDITIONS:
        r1 = row(pooled, condition, 1)
        r3 = row(pooled, condition, 3)
        lines.append(
            f"| `{condition}` | {r1['median']:.3f} → {r3['median']:.3f} | "
            f"{_pct(r1['fraction_le_0_05'])} → {_pct(r3['fraction_le_0_05'])} | "
            f"{_pct(r1['fraction_ge_2'])} → {_pct(r3['fraction_ge_2'])} | "
            f"{r1['mean_without_batch_max']:.3f} → {r3['mean_without_batch_max']:.3f} |"
        )
    lines.extend(
        [
            "",
            "三个条件均出现低值质量增加、高值质量减少。尤其是 `kg_base`，≤0.05 的候选从 14.6% 增至 41.7%，≥2 的候选从 72.9% 降至 27.1%；即使删除每轮最高候选，mean 仍从 2.316 降至 1.051。这证明下降是批次主体变化，而不是最高点把统计量‘拉坏’。",
            "",
            "### 3. 后期入选候选在 dry predictor 看来也更弱",
            "",
            "| 条件 | selected predicted mean R1 → R3 | wet mean R1 → R3 | acquisition–wet Pearson R3 |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in KG_CONDITIONS:
        r1 = row(score_summary, condition, 1)
        r3 = row(score_summary, condition, 3)
        lines.append(
            f"| `{condition}` | {r1['selected_predicted_mean_mean']:.3f} → "
            f"{r3['selected_predicted_mean_mean']:.3f} | {r1['wet_mean_mean']:.3f} → "
            f"{r3['wet_mean_mean']:.3f} | {r3['acquisition_wet_pearson_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "这排除了‘模型仍认为后期批次同样优秀，只是测量偶然变差’这一解释。三条路线的 selected predicted mean 都下降；同时 `kg_base_rag` Round 3 的 acquisition–wet 相关接近零，说明综合 acquisition 中的知识、先验、覆盖不确定性和控制配额未能继续稳定映射到 wet fitness。",
            "",
            "### 4. 固定分臂使批次目标不等于纯粹最大化当轮均值",
            "",
            "`kg_base`/`kg_base_rag` 的请求配额为 8 个 hypothesis-target、3 个 evidence-prior、3 个 coverage-exploration 和 2 个 matched-control，出现 shortfall 时由 fallback 补齐；`kg_base_al` 的请求配额为 8 个 exploitation、4 个 exploration 和 4 个 knowledge。后半批候选承担覆盖、证伪和对照功能，本来就不保证具有最高即时 fitness。各分臂 wet 结果已输出到 `acquisition_arm_outcomes_mean_sd.csv`。",
            "",
            "### 5. acquisition 分臂放大下降，但核心优化臂本身也在变弱",
            "",
            "| 条件 | 核心臂 wet mean R1 → R3 | 支持臂 wet mean R1 → R3 | R3 核心臂相对全批次差值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in KG_CONDITIONS:
        r1 = row(arm_mix, condition, 1)
        r3 = row(arm_mix, condition, 3)
        lines.append(
            f"| `{condition}` | {r1['core_arms_wet_mean']:.3f} → "
            f"{r3['core_arms_wet_mean']:.3f} | {r1['supporting_arms_wet_mean']:.3f} → "
            f"{r3['supporting_arms_wet_mean']:.3f} | +{r3['batch_mix_gap_core_minus_full']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`kg_base_al` 的 exploration arm 下降最明显（2.552 → 0.136），Round 3 有 58.3% exploration 候选 ≤0.05；若只看 exploitation + knowledge，Round 3 mean 为 2.461，高于全批次 1.880。类似地，`kg_base` 的 coverage-exploration 从 2.845 降至 0.374，`kg_base_rag` 的 matched-control 在 Round 3 只有 0.081。尽管如此，核心臂也同步下降：`kg_base` hypothesis-target 为 2.716 → 1.599，`kg_base_al` exploitation + knowledge 为 3.591 → 2.461。因此，探索/控制配额是放大器，不是唯一原因。",
            "",
            "### 6. 后期序列组成更少满足早期高值残基模式",
            "",
            "| 条件 | 平均 preferred-site count R1 → R3 | position 41=G 比例 | R3 wet mean：41G / non-G |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in KG_CONDITIONS:
        r1 = row(motif_summary, condition, 1)
        r3 = row(motif_summary, condition, 3)
        lines.append(
            f"| `{condition}` | {r1['preferred_site_count_mean_mean']:.2f} → "
            f"{r3['preferred_site_count_mean_mean']:.2f} | "
            f"{_pct(r1['fraction_position_41_G_mean'])} → "
            f"{_pct(r3['fraction_position_41_G_mean'])} | "
            f"{r3['wet_mean_position_41_G_mean']:.2f} / "
            f"{r3['wet_mean_position_41_non_G_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "`kg_base` 与 `kg_base_rag` 中 41G 比例均降至 56%，而 Round 3 的 41G 候选平均 fitness 明显高于 non-G。`kg_base_al` 保持了较高的 41G 比例，但平均 preferred-site count 仍下降。该组成漂移与 batch body 下移一致。这里的 preferred-site 集合属于事后描述性诊断，不是独立验证的因果规则；它也不应被转化为硬过滤，因为 `LWAA` 等非 G41 高值变体证明了上位性例外存在。",
            "",
            "### 7. 不能把下降解释为 119k 全局候选空间被耗尽",
            "",
        ]
    )
    max_removed = float(pool_diagnostics["catalog_removed_fraction"].max())
    mean_overlap = float(
        pool_diagnostics["previous_round_pool_jaccard"].dropna().mean()
    )
    lines.extend(
        [
            "| 条件 | pool utility mean R1 → R3 | selected utility mean R1 → R3 | unselected utility mean R1 → R3 |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in KG_CONDITIONS:
        group = pool_diagnostics[pool_diagnostics["condition"] == condition]
        r1 = group[group["round_id"] == 1]
        r3 = group[group["round_id"] == 3]
        lines.append(
            f"| `{condition}` | {r1['pool_utility_mean'].mean():.3f} → "
            f"{r3['pool_utility_mean'].mean():.3f} | "
            f"{r1['selected_utility_mean'].mean():.3f} → "
            f"{r3['selected_utility_mean'].mean():.3f} | "
            f"{r1['unselected_utility_mean'].mean():.3f} → "
            f"{r3['unselected_utility_mean'].mean():.3f} |"
        )
    lines.extend(
        [
            "",
            "utility 的量纲在条件内解释；`kg_base_al` 使用秩归一化 utility，不能与其他条件横向比较。每个条件中 selected utility 仍高于 unselected utility，说明选择器按自身 dry 目标在小池内实现了偏好；但 `kg_base`/`kg_base_rag` 的 pool 和 selected utility 总体下降，且 dry–wet 对齐在后期变弱。",
            "",
            f"每轮实际只从约 119,442 个可用候选中抽取 32 个并选择 16 个。到 Round 3，目录仅减少 32 个，即 {100 * max_removed:.3f}%；相邻轮 32-candidate pools 的平均 Jaccard 为 {mean_overlap:.3f}。因此，全局候选空间耗竭不是数据支持的主因。更符合证据的解释是：首轮优先抓住明显的高值模式；后续小候选池构成改变，且 acquisition 继续为探索、知识验证和控制分配预算，使入选集合的预测质量与实际质量同步下移。由于未选中候选没有 wet 标签，无法进一步区分‘小池本身变难’与‘dry 目标对 wet fitness 的排序变差’各占多少。",
            "",
            "## 论文表述建议",
            "",
            "不应写成‘best fitness 随轮次持续改善，但 batch quality 异常下降’。更准确的表述是：",
            "",
            "> 累计 best-seen 在少数 fold 中被稀疏的新纪录候选继续抬高，而当轮批次的中位数、高值候选比例和 dry-predicted mean 同时下降，说明峰值发现与批次富集发生分离。该现象主要来自累计极值的单调定义、首轮高值候选的前置捕获，以及后续轮次对探索/知识/控制预算的持续分配；现有数据不支持全局候选空间耗竭这一解释。",
            "",
        ]
    )
    return "\n".join(lines)


def _build_figure_qa_notes() -> str:
    return """# Figure 4 QA notes

- Core conclusion: cumulative best-seen can rise through sparse record events while the selected-batch distribution shifts downward.
- Archetype: four-panel quantitative grid; all panels were generated by Python/matplotlib.
- Final canvas: 7.2 × 5.1 inches (double-column layout).
- Font: Arial with sans-serif fallback; panel labels are bold lowercase.
- Color: condition color plus marker/linestyle encodings; no rainbow scale.
- Editable exports: SVG (`svg.fonttype=none`) and PDF (`pdf.fonttype=42`).
- Raster exports: 300 dpi PNG preview and 600 dpi RGB TIFF.

## Statistical legend minimum

- n definition: three independent campaign folds per condition; 16 newly selected candidates per fold and round.
- biological replicates: not applicable; these are in-silico lookup evaluations against fixed GB1 labels.
- technical replicates: none.
- center statistic: per-fold or pooled arithmetic mean where labelled; box plots show median and interquartile range.
- spread/interval: box-plot IQR and full candidate points; no inferential interval is shown.
- test: none, because n=3 folds is treated descriptively.
- multiple-comparison correction: not applicable.
- p-value display: none.
- source data: `batch_distribution_pooled.csv`, `selected_candidate_diagnostics.csv`, and `score_alignment_mean_sd.csv`.

## Machine-learning reporting

- split: AL96 initial observations with three rounds of 16 queried candidates; fold definitions follow the campaign artifacts.
- number of folds: 3.
- metric definitions: best-seen is the cumulative maximum; wet mean/median use only the current round's 16 revealed candidates.
- variability: fold counts in panel a; pooled 48-candidate distributions in panels b–c; three-fold means in panel d.
- baseline: the figure compares `kg_base`, `kg_base_rag`, and `kg_base_al`; it is a mechanism diagnostic rather than a baseline leaderboard.
"""


def main() -> int:
    ANOMALY_SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ANOMALY_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    validate_analysis_matrix(runs)
    round_metrics = build_round_metrics(runs)
    candidates = build_candidate_table(runs)
    by_fold = build_batch_distribution_by_fold(candidates, round_metrics)
    pooled = build_pooled_batch_distribution(candidates, round_metrics, by_fold)
    score_by_fold = build_score_alignment(candidates)
    score_summary = aggregate_score_alignment(score_by_fold)
    arm_by_fold = build_arm_outcomes(candidates)
    arm_summary = aggregate_arm_outcomes(arm_by_fold)
    arm_mix = build_arm_mix_summary(arm_by_fold)
    pool_diagnostics = build_candidate_pool_diagnostics(runs)
    motif_diagnostics = build_motif_diagnostics(candidates)
    motif_summary = aggregate_motif_diagnostics(motif_diagnostics)
    selected = selected_candidate_diagnostics(candidates)

    outputs = {
        "batch_distribution_by_fold.csv": by_fold,
        "batch_distribution_pooled.csv": pooled,
        "score_alignment_by_fold.csv": score_by_fold,
        "score_alignment_mean_sd.csv": score_summary,
        "acquisition_arm_outcomes_by_fold.csv": arm_by_fold,
        "acquisition_arm_outcomes_mean_sd.csv": arm_summary,
        "acquisition_arm_mix_summary.csv": arm_mix,
        "candidate_pool_diagnostics.csv": pool_diagnostics,
        "motif_diagnostics_by_fold.csv": motif_diagnostics,
        "motif_diagnostics_mean_sd.csv": motif_summary,
        "selected_candidate_diagnostics.csv": selected,
    }
    for name, frame in outputs.items():
        _write_csv(frame, ANOMALY_SOURCE_DATA_DIR / name)
    figure_paths = plot_anomaly_diagnostics(
        ANOMALY_SOURCE_DATA_DIR, ANOMALY_FIGURE_DIR
    )
    report = _build_report(
        by_fold,
        pooled,
        score_summary,
        arm_summary,
        arm_mix,
        pool_diagnostics,
        motif_summary,
    )
    report_path = ANOMALY_DIR / "anomaly_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    qa_path = ANOMALY_DIR / "figure4_qa.md"
    qa_path.write_text(_build_figure_qa_notes(), encoding="utf-8")
    declared_output_paths = [
        report_path,
        qa_path,
        *(ANOMALY_SOURCE_DATA_DIR / name for name in outputs),
        *figure_paths,
    ]
    manifest = {
        "analysis_id": "best_vs_batch_anomaly",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": list(KG_CONDITIONS),
        "folds": 3,
        "rounds": 3,
        "selected_candidates": int(len(selected)),
        "summary": compact_diagnostic_summary(
            pooled, score_summary, pool_diagnostics
        ),
        "outputs": [
            {
                "path": path.relative_to(ANOMALY_DIR).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(declared_output_paths)
        ],
        "figure_paths": [
            path.relative_to(ANOMALY_DIR).as_posix() for path in figure_paths
        ],
    }
    _write_json(manifest, ANOMALY_DIR / "anomaly_manifest.json")
    print(
        f"Anomaly diagnostics complete: {len(selected)} KG candidates, "
        f"{len(outputs)} source-data tables"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
