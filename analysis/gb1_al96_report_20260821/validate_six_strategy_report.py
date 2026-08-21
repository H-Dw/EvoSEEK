"""Validate the six-condition manuscript against generated source data and figures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CONDITION_ORDER, FIGURE_DIR, REPO_ROOT, SOURCE_DATA_DIR


REPORT_PATH = (
    REPO_ROOT / "docs" / "GB1实验报告-AL96六策略补充分析修订稿-20260821.md"
)


def _chapter_four(text: str) -> str:
    start = text.index("## 4. 主要结果、对照实验与消融矩阵")
    end = text.index("## 5. 局限、未来工作及来源声明")
    return text[start:end]


def main() -> int:
    report = REPORT_PATH.read_text(encoding="utf-8")
    chapter = _chapter_four(report)
    normalized = chapter.replace("−", "-")
    aggregate = pd.read_csv(SOURCE_DATA_DIR / "final_metrics_mean_sd.csv")
    missing_values = []
    for condition in CONDITION_ORDER:
        for _, row in aggregate[aggregate["condition"] == condition].iterrows():
            token = f"{float(row['mean']):.3f} ± {float(row['sd']):.3f}"
            if token not in normalized:
                missing_values.append(f"{condition}:{row['metric']}:{token}")

    forbidden = (
        "失败/未纳入",
        "0/3（占位）",
        "五种策略的 fitness 轨迹",
        "Cases are selected by deterministic rules from all 432",
    )
    stale = [token for token in forbidden if token in chapter]
    required_markers = (
        "18 个 run",
        "六种策略",
        "kg_3features_rag` | 通过 | 通过 | 通过 | 3/3",
        "27 条路径全部执行",
        "576 个已选候选",
        "三通道正例",
        "三通道反例",
        "相对 `kg_base_rag`",
        "不能说明这些观察已被转化为更准确的定向进化方向",
        "Data Availability（待补仓库标识后方可投稿）",
        "[I20]",
    )
    missing_markers = [token for token in required_markers if token not in report]

    figure_stems = ("figure2_fitness_trajectories", "figure3_module_deltas")
    missing_figures = [
        f"{stem}.{suffix}"
        for stem in figure_stems
        for suffix in ("svg", "pdf", "png", "tiff")
        if not (FIGURE_DIR / f"{stem}.{suffix}").exists()
    ]
    svg_errors = []
    for stem in figure_stems:
        svg = (FIGURE_DIR / f"{stem}.svg").read_text(encoding="utf-8")
        if "<text" not in svg:
            svg_errors.append(f"{stem}:no_editable_text")

    feature = pd.read_csv(SOURCE_DATA_DIR / "feature_channel_audit.csv")
    feature_errors = []
    if len(feature) != 27:
        feature_errors.append(f"rows={len(feature)}")
    if not feature["feature_step_executed"].astype(bool).all():
        feature_errors.append("feature_step_not_executed")
    if not feature["critic_disposition"].eq("APPROVED").all():
        feature_errors.append("critic_not_approved")
    if int(feature["candidate_hypothesis_count"].sum()) != 0:
        feature_errors.append("unexpected_child_hypotheses")
    selected = pd.read_csv(SOURCE_DATA_DIR / "selected_candidates.csv")
    feature_selected = selected[selected["condition"].eq("kg_3features_rag")]
    if len(feature_selected) != 144:
        feature_errors.append(f"selected_candidates={len(feature_selected)}")
    if set(feature_selected["evidence_channel"].dropna()) != {"kg"}:
        feature_errors.append("unexpected_selection_evidence_channel")
    if set(feature_selected["evidence_type"].dropna()) != {"measured_aggregate"}:
        feature_errors.append("unexpected_selection_evidence_type")
    if not feature_selected["evidence_count"].eq(1).all():
        feature_errors.append("unexpected_selection_evidence_count")

    if missing_values or stale or missing_markers or missing_figures or svg_errors or feature_errors:
        raise ValueError(
            "Six-strategy report validation failed: "
            f"missing_values={missing_values}, stale={stale}, "
            f"missing_markers={missing_markers}, missing_figures={missing_figures}, "
            f"svg_errors={svg_errors}, feature_errors={feature_errors}"
        )
    print(
        "Six-strategy report validation passed: "
        f"{len(aggregate)} aggregate metric rows, 8 figure exports, "
        "27 feature-channel rows, 4 Prompt/KG cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
