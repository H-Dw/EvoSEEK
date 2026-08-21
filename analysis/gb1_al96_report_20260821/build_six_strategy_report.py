"""Build a new full manuscript by replacing the results chapter mechanically."""

from __future__ import annotations

from pathlib import Path

from config import PACKAGE_ROOT, REPO_ROOT


SOURCE_REPORT = REPO_ROOT / "docs" / "GB1实验报告-问题定义与无泄漏评估协议.md"
CHAPTER_FOUR = PACKAGE_ROOT / "report_chapter4_six_strategy.md"
OUTPUT_REPORT = (
    REPO_ROOT / "docs" / "GB1实验报告-AL96六策略补充分析修订稿-20260821.md"
)


OLD_LIMITATION_OPENING = (
    "研究背景提出的预期效果是由任务约束和实现接口推导出的可检验假设。当前三折结果显示，"
    "三条 KG 路线在 fixed-budget 高值发现和推荐批次质量上均高于 `random` 与 "
    "`fitness_direct`；其中 `kg_base` 的最终 best-seen 最高，`kg_base_al` 的 AULC "
    "与末轮 batch mean/median 最高。主动学习的批次级增益跨三折方向一致，外部 RAG 的增益"
    "较小且折间不稳定。由于 seed 和逐轮候选池未严格配对，且 n=3，本报告不把这些描述性差异"
    "解释为组件因果效应。层级 Multi-Agent、ReThink、不确定性和三通道特征的独立贡献仍需固定 "
    "seed、候选池和同折对照后判定。"
)

NEW_LIMITATION_OPENING = (
    "研究背景提出的预期效果是由任务约束和实现接口推导出的可检验假设。当前三折结果显示，"
    "四条 KG 路线在 fixed-budget 高值发现和推荐批次质量上均高于 `random` 与 "
    "`fitness_direct`；其中 `kg_base` 的最终 best-seen 最高，`kg_base_al` 的 AULC "
    "与末轮 batch mean/median 最高。主动学习的批次级增益跨三折方向一致，外部 RAG 的收益混合；"
    "`kg_3features_rag` 的 27 条三通道路径均真实执行并通过子 Critic，但相对 `kg_base_rag` "
    "没有形成增量 wet-fitness 优势。由于 seed 和逐轮候选池未严格配对，且 n=3，本报告不把这些"
    "描述性差异解释为组件因果效应。层级 Multi-Agent、ReThink、不确定性以及三通道信息进入 "
    "acquisition 的独立贡献仍需固定 seed、候选池和同折开关对照后判定。"
)

OLD_I17 = (
    "[I17] 本节数值来自本轮提供的两个只读实验包：[`random`/`fitness_direct` artifacts]"
    "(../artifacts/random-fitness-direct-s42-al96-collected-20260820T102640Z/)与"
    "[`kg_base`/`kg_base_rag`/`kg_base_al` artifacts]"
    "(../artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/)。纳入规则、指标公式、"
    "源数据、案例筛选和图表由[模块化 Python 分析包](../analysis/gb1_al96_report_20260821/)及其"
    "[`analysis_summary.json`](../analysis/gb1_al96_report_20260821/outputs/analysis_summary.json)记录。"
)

NEW_I17 = (
    "[I17] 本节数值来自三个只读实验包：[`random`/`fitness_direct` artifacts]"
    "(../artifacts/random-fitness-direct-s42-al96-collected-20260820T102640Z/)、"
    "[`kg_base`/`kg_base_rag`/`kg_base_al` artifacts]"
    "(../artifacts/hierarchical-scientist-kg_base_kg_base_rag_kg_base_al/)与正式完成的"
    "[`kg_3features_rag` artifacts](../artifacts/hierarchical-scientist-kg_3features_rag/)。"
    "纳入规则、指标公式、源数据、案例筛选和图表由[模块化 Python 分析包]"
    "(../analysis/gb1_al96_report_20260821/)及其[`analysis_summary.json`]"
    "(../analysis/gb1_al96_report_20260821/outputs/analysis_summary.json)记录。"
)

I20 = (
    "[I20] 三通道执行证据见[`feature_channel_audit.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/feature_channel_audit.csv)、"
    "候选级直接证据见[`selected_candidates.csv`]"
    "(../analysis/gb1_al96_report_20260821/outputs/source_data/selected_candidates.csv)，"
    "全局与三通道正反案例的模型可见 Prompt、最终输出和 KG 子图见"
    "[`selected_cases.md`](../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.md)"
    "与[`selected_cases.json`](../analysis/gb1_al96_report_20260821/outputs/case_studies/selected_cases.json)。"
)

AVAILABILITY_DRAFT = """### 5.1 数据与代码可用性（投稿前工作稿）

当前内部复现链已经建立：原始 campaign artifacts 保持只读，处理后的 figure/table source data、案例审计、图件、输入文件 SHA-256 和输出哈希均保存在仓库内的 `analysis/gb1_al96_report_20260821/outputs/`；分析代码位于同名分析包。该本地路径只提供内部可追溯性，不构成公开、持久的数据访问方式。

**Data Availability（待补仓库标识后方可投稿）**

> The raw campaign artifacts, processed source data underlying all figures and tables, and case-level audit records supporting this study will be deposited in **[repository]** under **[DOI/accession]**. Reused GB1 data will be cited with the exact source, release and access identifier. A public repository record and persistent identifier have not yet been assigned; this statement is therefore not submission-ready.

**Code Availability（待补版本标识后方可投稿）**

> The analysis code used to validate runs, compute fold-level and aggregate metrics, generate figures and tables, and extract Prompt–KG cases will be archived at **[repository]** under **[release/DOI]**. The archived release will include an environment specification, execution instructions and checksums linking each source-data file to its input artifacts.

投稿前仍需确认公开仓库、DOI/登录号、代码 release、许可证，以及复用 GB1 数据的正式版本与引用。若原始远程模型交互记录受供应商条款限制，应分别公开可再分发的结构化审计记录，并说明不可公开字段、控制方和访问条件。

"""


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
    report = replace_once(report, "## 参考文献", AVAILABILITY_DRAFT + "## 参考文献", "references marker")
    report = replace_once(
        report,
        "\n## 附件\n",
        "\n" + I20 + "\n\n## 附件\n",
        "attachments marker",
    )
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
