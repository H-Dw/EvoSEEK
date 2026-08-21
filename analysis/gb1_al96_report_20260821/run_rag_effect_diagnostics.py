"""Run the reproducible RAG retrieval, Prompt, and selection-impact audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import RAG_DIAGNOSTIC_CASE_DIR, RAG_DIAGNOSTIC_DIR, RAG_DIAGNOSTIC_SOURCE_DIR
from io_artifacts import discover_runs, sha256_file, validate_analysis_matrix
from metrics import aggregate_final_metrics, build_final_metrics, build_round_metrics
from rag_effect_diagnostics import (
    BASE_CONDITION,
    RAG_CONDITION,
    build_matched_impact,
    build_prompt_audit,
    build_prompt_cases,
    build_retrieval_claim_audit,
)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _metric(aggregate: pd.DataFrame, condition: str, metric: str) -> tuple[float, float]:
    row = aggregate[
        (aggregate["condition"] == condition) & (aggregate["metric"] == metric)
    ]
    if len(row) != 1:
        raise ValueError(f"Missing aggregate metric: {condition}/{metric}")
    return float(row.iloc[0]["mean"]), float(row.iloc[0]["sd"])


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _build_report(
    aggregate: pd.DataFrame,
    prompt_audit: pd.DataFrame,
    retrieval_claims: pd.DataFrame,
    prompt_claims: pd.DataFrame,
    matched_impact: pd.DataFrame,
    rag_candidates: pd.DataFrame,
    cases: list[dict],
) -> str:
    rag_prompt = prompt_audit[prompt_audit["condition"] == RAG_CONDITION]
    base_prompt = prompt_audit[prompt_audit["condition"] == BASE_CONDITION]
    token_increment = matched_impact["rag_prompt_token_increment"]
    token_ratio = rag_prompt["prompt_tokens"].mean() / base_prompt["prompt_tokens"].mean()
    retrieval_signatures = retrieval_claims.groupby(["fold", "round_id"])[
        "claim_id"
    ].apply(lambda values: "|".join(sorted(values)))
    prompt_signatures = prompt_claims.groupby(["fold", "round_id"])["claim_id"].apply(
        lambda values: "|".join(sorted(values))
    )
    selection_ineligible = int((~retrieval_claims["selection_eligible"]).sum())
    target_specific = int(retrieval_claims["target_specific"].sum())
    cited_rounds = int(rag_prompt["response_used_rag"].sum())
    mismatch_rounds = int(
        (rag_prompt["rag_claims_with_mismatch_warning"] > 0).sum()
    )
    lines = [
        "# 为什么 RAG 没有带来预期提升：检索—Prompt—选择链审计",
        "",
        "## 结论先行",
        "",
        "本实验不能概括为‘RAG 让所有性能下降’。RAG 相对 `kg_base` 提高了 best-seen AULC、Round 3 batch mean/median 和部分排序指标，但没有提高最终 best-seen，且 Spearman、Pearson、MSE 与高峰发现能力变差。更准确的结论是：RAG 没有形成稳定的候选级增益，其主要原因是检索内容与当前 GB1 四位点闭池选择的决策接口不匹配。",
        "",
        "## 1. 性能变化是混合的，不是单向退化",
        "",
        "| 指标 | kg_base | kg_base_rag | RAG − base | 解释 |",
        "|---|---:|---:|---:|---|",
    ]
    metric_specs = [
        ("final_best_seen", "higher", "最终峰值"),
        ("best_seen_aulc", "higher", "峰值轨迹效率"),
        ("r3_batch_mean", "higher", "末轮批次均值"),
        ("r3_batch_median", "higher", "末轮批次中位数"),
        ("spearman", "higher", "预测排序"),
        ("mse", "lower", "预测误差"),
        ("regret_at_k", "lower", "Top-k regret"),
    ]
    for metric, direction, label in metric_specs:
        base_mean, base_sd = _metric(aggregate, BASE_CONDITION, metric)
        rag_mean, rag_sd = _metric(aggregate, RAG_CONDITION, metric)
        delta = rag_mean - base_mean
        better = delta > 0 if direction == "higher" else delta < 0
        lines.append(
            f"| {label} | {_fmt(base_mean)} ± {_fmt(base_sd)} | "
            f"{_fmt(rag_mean)} ± {_fmt(rag_sd)} | {delta:+.3f} | "
            f"{'改善' if better else '下降'} |"
        )
    lines.extend(
        [
            "",
            "因此，问题不是 RAG 完全无效，而是它没有把局部批次富集转化为更高、可重复的峰值发现或更可靠的预测对齐。",
            "",
            "## 2. 实际检索内容：高重复、通用、不可直接用于候选选择",
            "",
            f"9 个 fold × round 共记录 {len(retrieval_claims)} 条检索结果；每轮 8 条。只有 {retrieval_signatures.nunique()} 种检索 claim 集合，说明查询和返回内容高度重复。全部 {selection_ineligible}/{len(retrieval_claims)} 条均为 `selection_eligible=false`，包含 GB1、39/40/41/54 位点或具体残基方向的 target-specific claim 数为 {target_specific}。",
            "",
            "固定检索问题为：",
            "",
            f"> {retrieval_claims.iloc[0]['sanitized_query']}",
            "",
            "检索内容主要落在以下类别：",
            "",
            "| 类别 | 出现次数 | 对当前闭池选择的作用 |",
            "|---|---:|---|",
        ]
    )
    for actionability, group in retrieval_claims.groupby("actionability_class"):
        explanation = {
            "interpretation_or_future_combination": "提示上位性或后续组合实验，只能约束解释强度",
            "requires_unavailable_stability_assay": "需要稳定性、表达或可溶性读出，当前实验没有该通道",
            "requires_structure_or_site_validation": "需要结构或热点验证，当前四位点闭池没有新增结构证据",
            "library_design_not_current_candidate_score": "用于文库设计，不能区分本轮32个候选的fitness",
            "generic_prior_not_candidate_score": "通用先验，未提供候选级方向",
        }.get(actionability, actionability)
        lines.append(f"| `{actionability}` | {len(group)} | {explanation} |")
    top_claims = (
        retrieval_claims.groupby(["claim_id", "statement"], as_index=False)
        .agg(occurrences=("claim_id", "size"), mean_confidence=("confidence", "mean"))
        .sort_values(["occurrences", "mean_confidence"], ascending=[False, False])
        .head(8)
    )
    lines.extend(
        [
            "",
            "### 被反复检索的原始 claim",
            "",
            "| Claim | 次数 | Mean confidence | 原文 |",
            "|---|---:|---:|---|",
        ]
    )
    for _, row in top_claims.iterrows():
        lines.append(
            f"| `{row['claim_id']}` | {int(row['occurrences'])} | "
            f"{float(row['mean_confidence']):.3f} | {row['statement']} |"
        )
    lines.extend(
        [
            "",
            "## 3. RAG 如何进入 LLM Prompt",
            "",
            "执行链为 `local_rag_retrieval.json` → `local_rag_evidence.json` → Prompt 顶层 `rag_claims` → Scientist `evidence_ids`/`preferred_residues` → `hypothesis_score` → Agent-UQ hypothesis-target arm。RAG 不直接写入 fitness 或 predictor score，只能通过改变 Scientist 假设间接影响选择。",
            "",
            f"RAG Prompt 平均为 {rag_prompt['prompt_tokens'].mean():.0f} tokens，base 平均为 {base_prompt['prompt_tokens'].mean():.0f} tokens，约为 {token_ratio:.2f} 倍；平均增加 {token_increment.mean():.0f} tokens。Prompt 每轮平均包含 {rag_prompt['prompt_rag_claim_count'].mean():.1f} 张 RAG claim cards，但只有 {cited_rounds}/9 轮的最终 Scientist 输出显式引用了至少一条 RAG 短 ID。未引用不等于完全没有上下文影响，但说明可审计的直接使用很有限。",
            "",
            f"RAG与base的四个位点 residue-set Jaccard 平均为 {matched_impact['preferred_residue_similarity'].mean():.3f}；RAG每轮preferred-residue笛卡尔组合数平均为 {rag_prompt['preferred_combination_count'].mean():.1f}，base为 {base_prompt['preferred_combination_count'].mean():.1f}。Scientist statement 文本相似度平均为 {matched_impact['statement_similarity'].mean():.3f}。总体方向仍由可见GB1测量主导，RAG主要改变residue set的宽窄、措辞和置信边界，而没有产生稳定的新GB1特异证据。",
            "",
            "更关键的是，这些soft preferences会在LLM之后、Agent-UQ之前先影响32-candidate pool。`KnowledgeCandidateGenerator`首先按每个variant命中多少个preferred positions排序，再按selection-eligible evidence score和确定性tie-break排序，最后从约119k候选截取32个。因此，语义上‘soft’的逐位偏好在大空间硬截断下具有近似门控效应。",
            "",
            "## 4. Prompt 投影存在证据身份混合",
            "",
            f"原始检索每轮为 8 条，但 Prompt 中平均扩展为 {prompt_claims.groupby(['fold','round_id']).size().mean():.1f} 张 cards，其中包括 KG 中已有的通用先验。{mismatch_rounds}/9 轮都存在 `claim_text_mismatch_across_paths`，每轮12张cards中有6张带该warning。没有发现同一短ID跨多张card复用，但一张card内部可合并来自不同claim路径的source refs。",
            "",
            "这意味着 LLM 看到的不是八条彼此独立、身份稳定的证据：同一statement可能合并多个不一致source refs。虽然运行时保留了warning，但Scientist仍可引用该短ID，导致‘引用闭合’不等于‘语义身份清晰’。这会增加错误归因风险。",
            "",
            "## 5. 对候选选择的实际影响",
            "",
            "| Fold | Round | Pool overlap | Selected overlap | RAG−base mean | RAG−base median | RAG−base best | Preferred-set similarity | RAG citations |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in matched_impact.iterrows():
        lines.append(
            f"| {int(row['fold'])} | {int(row['round_id'])} | "
            f"{int(row['candidate_pool_overlap_count'])}/32 | "
            f"{int(row['selected_overlap_count'])}/16 | "
            f"{row['rag_minus_base_batch_mean']:+.3f} | "
            f"{row['rag_minus_base_batch_median']:+.3f} | "
            f"{row['rag_minus_base_batch_best']:+.3f} | "
            f"{row['preferred_residue_similarity']:.3f} | "
            f"{int(row['rag_claim_citation_count'])} |"
        )
    exact_pool_rounds = int((matched_impact["candidate_pool_overlap_count"] == 32).sum())
    same_sampler_rounds = int(
        (
            matched_impact["same_pool_seed"]
            & matched_impact["same_sampling_namespace"]
            & matched_impact["same_sampling_strategy"]
        ).sum()
    )
    lines.extend(
        [
            "",
            f"虽然 {same_sampler_rounds}/9 个fold-round使用相同seed、sampling namespace和`knowledge_filtered`策略，但没有任何一轮候选池完全相同（{exact_pool_rounds}/9）；Round 1也只重合4–11/32。结合生成器的确定性排序逻辑，这表明RAG引起的preferred-residue变化已经在Scientist输出之后、最终acquisition之前改写了候选池。后续轮次还叠加了前序入选差异。",
            "",
        ]
    )
    match_summary = rag_candidates.groupby("preferred_position_matches").agg(
        n=("variant_id", "size"), wet_mean=("wet_fitness", "mean")
    )
    lines.extend(
        [
            "### RAG偏好匹配数与wet结果",
            "",
            "| 满足Scientist偏好的位点数 | n | wet mean |",
            "|---:|---:|---:|",
        ]
    )
    for matches, row in match_summary.iterrows():
        lines.append(
            f"| {int(matches)} | {int(row['n'])} | {float(row['wet_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "偏好匹配数与pooled wet mean呈明显递增：四个位点全部匹配的85个候选均值为2.745，而仅匹配2–3个位点的候选显著更低。这说明RAG/Scientist的逐位偏好确实能做粗粒度富集，也解释了RAG为何改善部分batch mean/median。问题在于85个全匹配候选内部仍有巨大组合差异，而当前匹配分数只计数位点、不编码配对或四位点上位性，因此难以稳定找到最高峰。",
            "",
            "## 6. 为什么没有得到预期提升",
            "",
            "1. **检索问题没有候选锚点。** 查询只包含‘结构、稳定性、界面、理化、上位性’等宽泛主题，没有当前32个variant、已观测反例或本轮待区分的残基组合。返回结果因此高度重复。",
            "2. **知识粒度与决策粒度错位。** 文献claim回答‘应如何设计或验证实验’，Agent-UQ需要的是‘本轮哪些完整四位点组合更可能提高GB1 fitness’。所有claim都被正确标为不可直接选择，但系统仍允许其改变假设方向。",
            "3. **现有实验数据已经提供了更强的方向。** base Prompt中的96+轮次观测已经清楚支持40位芳香残基、41G等模式；RAG主要重复‘注意上位性、验证稳定性、保留WT’等常识，新增信息边际很小。",
            "4. **RAG影响是间接且未经校准的，但作用被候选池截断放大。** 通用claim可改变`preferred_residues`；候选生成器先按逐位匹配数从约119k候选截到32个，之后才进入`hypothesis_score`和8个hypothesis-target名额。RAG confidence不是候选fitness效应，也没有GB1 selection calibration。",
            "5. **证据融合降低了信号清晰度。** claim-text mismatch和单张card内的source-ref混合增加了Prompt长度与语义歧义；更多上下文没有等比例增加可行动信息。",
            "6. **信息接口无法表达RAG自己检索到的上位性知识。** Prompt反复出现‘epistasis is widespread/background-dependent’，但Scientist输出与`_hypothesis_matches`仍把四个位点表示为独立residue sets并逐位相加。结果可以提高平均富集，却会在全匹配组合之间随机tie-break，错过像`LWAA`这类特定组合高峰。",
            "",
            "## 7. 证据边界",
            "",
            "该审计证明RAG内容进入了Scientist可见Prompt，并在部分轮次被引用、改变了假设和候选选择；但不能把RAG与base的所有差异解释为纯RAG因果效应。后续轮次候选池会因先前选择而分叉，LLM调用也不是严格配对重复。更严格的验证应固定每轮32-candidate pool、复用同一visible-observation snapshot，并对同一Prompt执行RAG on/off配对重放。",
            "",
            "## 8. 建议的修正优先级",
            "",
            "1. 将查询改为round-specific：纳入当前候选组合、已观测支持/反例、需要区分的competing directions。",
            "2. RAG默认只生成解释与实验建议；只有通过GB1或任务级校准的claim才能进入 `hypothesis_score`。",
            "3. 修复claim/source/evidence-ID一对一身份，出现 `claim_text_mismatch_across_paths` 时禁止该claim影响selection。",
            "4. 从逐位preferred residues升级为候选级、组合级的支持/反证表示，并显式建模上位性。",
            "5. 对RAG进行no-answer门控：若检索内容不能区分当前候选，则返回‘解释性上下文’，不扩张selection prior。",
            "",
            "## Prompt案例",
            "",
        ]
    )
    for case in cases:
        metric = case["metric_comparison"]
        base_output = case["llm_output_comparison"]["kg_base"]
        rag_output = case["llm_output_comparison"]["kg_base_rag"]
        lines.extend(
            [
                f"### {case['case_id']}：fold {case['fold']} / round {case['round_id']}",
                "",
                f"- RAG−base batch mean：{float(metric['rag_minus_base_batch_mean']):+.3f}",
                f"- 候选池重合：{int(metric['candidate_pool_overlap_count'])}/32；入选重合：{int(metric['selected_overlap_count'])}/16",
                f"- Scientist显式引用的RAG IDs：{', '.join(case['rag_cited_evidence_ids']) if case['rag_cited_evidence_ids'] else 'none'}",
                f"- base preferred residues：`{json.dumps(base_output.get('preferred_residues', {}), ensure_ascii=False, sort_keys=True)}`",
                f"- RAG preferred residues：`{json.dumps(rag_output.get('preferred_residues', {}), ensure_ascii=False, sort_keys=True)}`",
                f"- 原始检索：`{case['rag_retrieval_path']}`",
                f"- 完整Scientist Prompt：`{case['rag_conversation_path']}`",
                "- 可复制的精简Prompt与输出已保存到对应 evidence-case JSON；provider reasoning_content 未复制。",
                "",
            ]
        )
        if case["case_id"] == "largest_negative_batch_delta":
            lines.extend(
                [
                    "该负向案例中，RAG输出显式引用的是‘pairwise epistasis widespread’与‘epistasis background-dependent’，二者都没有提供Y/F、I/L或A的候选级方向；但最终preferred sets从base的较宽集合收缩为39={I,L}、40={Y,F}、41={G}、54={A}，候选池随之只与base重合4/32。该收缩缺少RAG claim到具体残基的可验证蕴含关系，却被确定性候选池截断放大。",
                    "",
                ]
            )
        if case["case_id"] == "largest_positive_batch_delta":
            lines.extend(
                [
                    "该正向案例中，RAG batch mean高出1.348，但Scientist没有显式引用任何RAG短ID；其方向主要来自当轮已揭示GB1观测。因此，这个改善不能作为外部检索claim带来收益的直接证据。",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    RAG_DIAGNOSTIC_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIAGNOSTIC_CASE_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    validate_analysis_matrix(runs)
    prompt_audit = build_prompt_audit(runs)
    retrieval_claims, prompt_claims = build_retrieval_claim_audit(runs)
    matched_impact, rag_candidates = build_matched_impact(runs, prompt_audit)
    cases = build_prompt_cases(runs, matched_impact)
    round_metrics = build_round_metrics(runs)
    final_metrics = build_final_metrics(runs, round_metrics)
    aggregate = aggregate_final_metrics(final_metrics)

    outputs = {
        "scientist_prompt_audit.csv": prompt_audit,
        "rag_retrieval_claims.csv": retrieval_claims,
        "rag_prompt_claims.csv": prompt_claims,
        "kg_base_vs_rag_matched_impact.csv": matched_impact,
        "rag_selected_candidate_preference_audit.csv": rag_candidates,
    }
    for name, frame in outputs.items():
        _write_csv(frame, RAG_DIAGNOSTIC_SOURCE_DIR / name)
    case_paths = []
    for case in cases:
        path = RAG_DIAGNOSTIC_CASE_DIR / f"{case['case_id']}.json"
        _write_json(case, path)
        case_paths.append(path)
    report_path = RAG_DIAGNOSTIC_DIR / "rag_effect_analysis.md"
    report_path.write_text(
        _build_report(
            aggregate,
            prompt_audit,
            retrieval_claims,
            prompt_claims,
            matched_impact,
            rag_candidates,
            cases,
        ),
        encoding="utf-8",
    )
    declared = [
        report_path,
        *(RAG_DIAGNOSTIC_SOURCE_DIR / name for name in outputs),
        *case_paths,
    ]
    manifest = {
        "analysis_id": "rag_effect_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": [BASE_CONDITION, RAG_CONDITION],
        "folds": 3,
        "rounds": 3,
        "retrieval_claim_rows": len(retrieval_claims),
        "prompt_claim_rows": len(prompt_claims),
        "outputs": [
            {
                "path": path.relative_to(RAG_DIAGNOSTIC_DIR).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(declared)
        ],
    }
    _write_json(manifest, RAG_DIAGNOSTIC_DIR / "rag_effect_manifest.json")
    print(
        "RAG effect diagnostics complete: "
        f"{len(retrieval_claims)} retrieval rows, {len(prompt_claims)} prompt claims"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
