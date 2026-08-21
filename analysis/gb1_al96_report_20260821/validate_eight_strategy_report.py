"""Validate the eight-condition manuscript against generated source data."""

from __future__ import annotations

import json

import pandas as pd

from config import CONDITION_ORDER, FIGURE_DIR, REPO_ROOT, SOURCE_DATA_DIR


REPORT_PATH = (
    REPO_ROOT / "docs" / "GB1实验报告-AL96八策略补充分析修订稿-20260821.md"
)
CASE_PATH = (
    REPO_ROOT
    / "analysis"
    / "gb1_al96_report_20260821"
    / "outputs"
    / "case_studies"
    / "selected_cases.json"
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
        "六种策略",
        "18 个 run",
        "576 个已选候选",
        "27 条三通道路径",
        "0/3（占位）",
    )
    stale = [token for token in forbidden if token in chapter]
    required_markers = (
        "24 个 run",
        "八种策略",
        "`agent_only` | 通过 | 通过 | 通过 | 3/3",
        "`kg_3features_base` | 通过 | 通过 | 通过 | 3/3",
        "54 条 fold–round–channel 路径",
        "864 个已选候选",
        "Experimental Memory",
        "feature-by-RAG",
        "不支持加性协同",
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

    status = pd.read_csv(SOURCE_DATA_DIR / "run_status.csv")
    status_errors = []
    if int(status["eligible"].sum()) != 24:
        status_errors.append(f"eligible={int(status['eligible'].sum())}")
    if int((~status["eligible"].astype(bool)).sum()) != 3:
        status_errors.append(
            f"excluded={int((~status['eligible'].astype(bool)).sum())}"
        )

    feature = pd.read_csv(SOURCE_DATA_DIR / "feature_channel_audit.csv")
    audit_errors = []
    if len(feature) != 54:
        audit_errors.append(f"feature_rows={len(feature)}")
    if set(feature["condition"]) != {"kg_3features_base", "kg_3features_rag"}:
        audit_errors.append("feature_conditions")
    if not feature["feature_step_executed"].astype(bool).all():
        audit_errors.append("feature_step_not_executed")
    if not feature["critic_disposition"].eq("APPROVED").all():
        audit_errors.append("critic_not_approved")
    if int(feature["candidate_hypothesis_count"].sum()) != 0:
        audit_errors.append("unexpected_child_hypotheses")

    selected = pd.read_csv(SOURCE_DATA_DIR / "selected_candidates.csv")
    feature_selected = selected[
        selected["condition"].isin(("kg_3features_base", "kg_3features_rag"))
    ]
    if len(feature_selected) != 288:
        audit_errors.append(f"feature_selected={len(feature_selected)}")
    if set(feature_selected["evidence_channel"].dropna()) != {"kg"}:
        audit_errors.append("unexpected_feature_selection_channel")
    if set(feature_selected["evidence_type"].dropna()) != {"measured_aggregate"}:
        audit_errors.append("unexpected_feature_selection_type")
    if not feature_selected["evidence_count"].eq(1).all():
        audit_errors.append("unexpected_feature_selection_count")
    agent_selected = selected[selected["condition"].eq("agent_only")]
    if len(agent_selected) != 144 or not agent_selected["evidence_count"].eq(0).all():
        audit_errors.append("agent_only_selection_evidence")

    runtime = pd.read_csv(SOURCE_DATA_DIR / "new_condition_runtime_audit.csv")
    if len(runtime) != 6 or not runtime["condition_contract_ok"].astype(bool).all():
        audit_errors.append("new_condition_runtime_contract")
    interaction = pd.read_csv(
        SOURCE_DATA_DIR / "feature_rag_interaction_deltas.csv"
    )
    if len(interaction) != 12:
        audit_errors.append(f"interaction_rows={len(interaction)}")

    cases = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    expected_case_ids = {
        "positive",
        "negative",
        "feature_rag_positive",
        "feature_rag_negative",
        "feature_base_positive",
        "feature_base_negative",
        "agent_only_positive",
        "agent_only_negative",
    }
    case_ids = {str(case["case_id"]) for case in cases}
    if len(cases) != 8 or case_ids != expected_case_ids:
        audit_errors.append(f"case_ids={sorted(case_ids)}")

    if (
        missing_values
        or stale
        or missing_markers
        or missing_figures
        or svg_errors
        or status_errors
        or audit_errors
    ):
        raise ValueError(
            "Eight-strategy report validation failed: "
            f"missing_values={missing_values}, stale={stale}, "
            f"missing_markers={missing_markers}, missing_figures={missing_figures}, "
            f"svg_errors={svg_errors}, status_errors={status_errors}, "
            f"audit_errors={audit_errors}"
        )
    print(
        "Eight-strategy report validation passed: "
        f"{len(aggregate)} aggregate metric rows, 8 figure exports, "
        "54 feature-channel rows, 6 runtime-audit rows, 8 cases."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

