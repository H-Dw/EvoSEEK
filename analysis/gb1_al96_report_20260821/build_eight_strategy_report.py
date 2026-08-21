"""Build an eight-condition manuscript without overwriting the six-condition draft."""

from __future__ import annotations

from pathlib import Path

from config import PACKAGE_ROOT, REPO_ROOT


SOURCE_REPORT = (
    REPO_ROOT / "docs" / "GB1实验报告-AL96六策略补充分析修订稿-20260821.md"
)
CHAPTER_FOUR = PACKAGE_ROOT / "report_chapter4_eight_strategy.md"
OUTPUT_REPORT = (
    REPO_ROOT / "docs" / "GB1实验报告-AL96八策略补充分析修订稿-20260821.md"
)


OLD_LIMITATION_OPENING = (
    "研究背景提出的预期效果是由任务约束和实现接口推导出的可检验假设。当前三折结果显示，"
    "四条 KG 路线在 fixed-budget 高值发现和推荐批次质量上均高于 `random` 与 "
    "`fitness_direct`；其中 `kg_base` 的最终 best-seen 最高，`kg_base_al` 的 AULC "
    "与末轮 batch mean/median 最高。主动学习的批次级增益跨三折方向一致，外部 RAG 的收益混合；"
    "`kg_3features_rag` 的 27 条三通道路径均真实执行并通过子 Critic，但相对 `kg_base_rag` "
    "没有形成增量 wet-fitness 优势。由于 seed 和逐轮候选池未严格配对，且 n=3，本报告不把这些"
    "描述性差异解释为组件因果效应。层级 Multi-Agent、ReThink、不确定性以及三通道信息进入 "
    "acquisition 的独立贡献仍需固定 seed、候选池和同折开关对照后判定。"
)

NEW_LIMITATION_OPENING = (
    "研究背景提出的预期效果是由任务约束和实现接口推导出的可检验假设。当前三折结果显示，"
    "`agent_only` 已能提高批次富集与有限峰值发现；在此基础上，`kg_base` 的最终 best-seen 和 "
    "AULC 在 3/3 folds 均提高，说明 Experimental Memory 的优势主要体现在累计高值发现。"
    "`kg_3features_base` 获得最高 final best-seen、AULC 和 R3 batch best，`kg_base_al` 则保持最高"
    "末轮 batch mean/median。外部 RAG 的收益在有无三通道时均为混合方向，三通道与 RAG 也没有"
    "显示稳定的加性协同。由于每个条件只有三个 fold，且逐轮候选池未严格配对，本报告不把这些"
    "描述性差异解释为组件因果效应；三通道单项贡献、ReThink 和 UQ 仍需固定候选池、重复 seed "
    "与同折开关对照后判定。"
)

OLD_I17 = (
    "[I17] 本节数值来自三个只读实验包：[`random`/`fitness_direct` artifacts]"
    "(../artifacts/random-fitness-direct-s42-al96-collected-20260820T102640Z/)、"
    "[`kg_base`/`kg_base_rag`/`kg_base_al` artifacts]"
    "(../artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/)与正式完成的"
    "[`kg_3features_rag` artifacts](../artifacts/hierarchical-scientist-kg_3features_rag/)。"
    "纳入规则、指标公式、源数据、案例筛选和图表由[模块化 Python 分析包]"
    "(../analysis/gb1_al96_report_20260821/)及其[`analysis_summary.json`]"
    "(../analysis/gb1_al96_report_20260821/outputs/analysis_summary.json)记录。"
)

NEW_I17 = (
    "[I17] 本节数值来自四个只读实验包：[`random`/`fitness_direct` artifacts]"
    "(../artifacts/random-fitness-direct-s42-al96-collected-20260820T102640Z/)、"
    "[`kg_base`/`kg_base_rag`/`kg_base_al` artifacts]"
    "(../artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/)、"
    "[`kg_3features_rag` artifacts](../artifacts/hierarchical-scientist-kg_3features_rag/)与"
    "[`kg_3features_base`/`agent_only` artifacts]"
    "(../artifacts/hierarchical-scientist-kg_3features_base_agent_only/)。纳入规则、指标公式、"
    "源数据、案例筛选和图表由[模块化 Python 分析包]"
    "(../analysis/gb1_al96_report_20260821/)及其[`analysis_summary.json`]"
    "(../analysis/gb1_al96_report_20260821/outputs/analysis_summary.json)记录。"
)

OLD_I20 = (
    "[I20] 三通道执行证据见[`feature_channel_audit.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/feature_channel_audit.csv)、"
    "候选级直接证据见[`selected_candidates.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/selected_candidates.csv)，"
    "全局与三通道正反案例的模型可见 Prompt、最终输出和 KG 子图见"
    "[`selected_cases.md`](../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.md)"
    "与[`selected_cases.json`](../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.json)。"
)

NEW_I20 = (
    "[I20] 两组三通道的 54 条执行记录见[`feature_channel_audit.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/feature_channel_audit.csv)，"
    "新增条件的运行边界见[`new_condition_runtime_audit.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/new_condition_runtime_audit.csv)，"
    "feature-by-RAG 交互见[`feature_rag_interaction_deltas_mean_sd.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/feature_rag_interaction_deltas_mean_sd.csv)，"
    "候选级直接证据见[`selected_candidates.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/selected_candidates.csv)。"
    "八个正反案例的模型可见 Prompt、最终输出和 KG 子图见[`selected_cases.md`]"
    "(../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.md)与"
    "[`selected_cases.json`](../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.json)。"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one {label}, found {text.count(old)}")
    return text.replace(old, new, 1)


def main() -> int:
    report = SOURCE_REPORT.read_text(encoding="utf-8")
    chapter = CHAPTER_FOUR.read_text(encoding="utf-8").rstrip() + "\n\n"
    start_marker = "## 4. 主要结果、对照实验与消融矩阵"
    end_marker = "## 5. 局限、未来工作及来源声明"
    start = report.index(start_marker)
    end = report.index(end_marker)
    report = report[:start] + chapter + report[end:]
    report = replace_once(
        report, OLD_LIMITATION_OPENING, NEW_LIMITATION_OPENING, "limitation opening"
    )
    report = replace_once(report, OLD_I17, NEW_I17, "I17 implementation source")
    report = replace_once(report, OLD_I20, NEW_I20, "I20 audit source")
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

