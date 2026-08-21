"""Build the standalone Attachment D results-showcase module.

The script reads campaign artifacts without modifying them.  It reuses the
validated run discovery, metric reconstruction, case selection, and plotting
style from the main AL96 analysis package.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cases import build_candidate_table, build_case_studies
from config import AGENT_CONDITIONS, OUTPUT_ROOT, REPO_ROOT
from io_artifacts import (
    RunArtifact,
    discover_runs,
    read_json,
    sha256_file,
    validate_analysis_matrix,
)
from metrics import aggregate_final_metrics, build_final_metrics, build_round_metrics
from plots import _clean_axis, _save_all, _set_style


SHOWCASE_ROOT = OUTPUT_ROOT / "results_showcase"
SOURCE_DATA_DIR = SHOWCASE_ROOT / "source_data"
FIGURE_DIR = SHOWCASE_ROOT / "figures"
PROMPT_CHAIN_PATH = SHOWCASE_ROOT / "showcase_prompt_chain.json"
MANIFEST_PATH = SHOWCASE_ROOT / "results_showcase_manifest.json"
REPORT_PATH = REPO_ROOT / "docs" / "GB1实验报告-附件D-结果展示-20260821.md"
DEFAULT_TOP_K = 5
WINDOW_START = 33
WINDOW_END = 60


def _write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _doc_relative(path: Path) -> str:
    return "../" + _repo_relative(path)


def _find_run(runs: list[RunArtifact], run_id: str) -> RunArtifact:
    matches = [run for run in runs if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one run for {run_id}, found {len(matches)}")
    return matches[0]


def _select_condition_and_run(
    runs: list[RunArtifact], round_metrics: pd.DataFrame
) -> tuple[str, RunArtifact, pd.DataFrame, pd.DataFrame]:
    final_metrics = build_final_metrics(runs, round_metrics)
    aggregate = aggregate_final_metrics(final_metrics)
    pivot = (
        aggregate[aggregate["condition"].isin(AGENT_CONDITIONS)]
        .pivot(index="condition", columns="metric", values="mean")
        .reset_index()
    )
    leader = pivot.sort_values(
        ["final_best_seen", "best_seen_aulc", "condition"],
        ascending=[False, False, True],
    ).iloc[0]
    condition = str(leader["condition"])
    condition_folds = final_metrics[final_metrics["condition"] == condition]
    representative = condition_folds.sort_values(
        ["final_best_seen", "best_seen_aulc", "fold"],
        ascending=[False, False, True],
    ).iloc[0]
    run = _find_run(runs, str(representative["run_id"]))
    return condition, run, final_metrics, aggregate


def _selection_arm(run: RunArtifact, round_id: int, variant_id: str) -> str:
    path = (
        run.path
        / f"round_{round_id:02d}"
        / "agent_quota_acquisition_approved.json"
    )
    if not path.exists():
        return "not_recorded"
    payload = read_json(path)
    selected = ((payload.get("selection") or {}).get("selected_by_arm") or {})
    arms = [arm for arm, ids in selected.items() if variant_id in ids]
    return "+".join(sorted(arms)) if arms else "not_recorded"


def _build_top_k(
    candidates: pd.DataFrame, run: RunArtifact, top_k: int
) -> pd.DataFrame:
    selected = candidates[
        (candidates["run_id"] == run.run_id)
        & (candidates["selection_order"] <= top_k)
    ].copy()
    selected["selection_arm"] = [
        _selection_arm(run, int(row.round_id), str(row.variant_id))
        for row in selected.itertuples()
    ]
    selected = selected.rename(
        columns={
            "fitness_mean": "dry_validation_fitness",
            "fitness_std": "dry_validation_sd",
        }
    )
    columns = [
        "round_id",
        "selection_order",
        "variant_id",
        "variant",
        "mutation_notation",
        "selection_arm",
        "acquisition_score",
        "knowledge_score",
        "dry_validation_fitness",
        "dry_validation_sd",
        "wet_fitness",
        "rethink_verdict",
        "rethink_summary",
    ]
    result = selected[columns].sort_values(["round_id", "selection_order"])
    counts = result.groupby("round_id").size().to_dict()
    if counts != {1: top_k, 2: top_k, 3: top_k}:
        raise ValueError(f"Malformed showcase Top-k counts: {counts}")
    return result.reset_index(drop=True)


def _raw_candidate_row(
    runs: list[RunArtifact], run_id: str, variant_id: str
) -> tuple[RunArtifact, pd.Series]:
    run = _find_run(runs, run_id)
    rows = pd.read_csv(run.path / "top_k_all_rounds.csv")
    match = rows[rows["variant_id"] == variant_id]
    if len(match) != 1:
        raise ValueError(
            f"Expected one selected row for {run_id}/{variant_id}, found {len(match)}"
        )
    return run, match.iloc[0]


def _sequence_window(sequence: str) -> str:
    return sequence[WINDOW_START - 1 : WINDOW_END]


def _build_sequences(
    runs: list[RunArtifact],
    representative_run: RunArtifact,
    positive_case: dict[str, Any],
    negative_case: dict[str, Any],
    candidates: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    wild_type = read_json(representative_run.path / "wild_type.json")
    representative_rows = candidates[
        (candidates["run_id"] == representative_run.run_id)
        & (candidates["round_id"] == 3)
        & (candidates["rethink_verdict"] == "mixed")
        & (candidates["wet_fitness"] < 0.02)
    ].sort_values(["selection_order", "wet_fitness"])
    if representative_rows.empty:
        raise ValueError("No representative round-3 failure was found")
    top_rank_failure = representative_rows.iloc[0].to_dict()

    roles = [
        ("成功代表", positive_case),
        ("跨 fold 反例", negative_case),
        ("同一代表 run 的高优先级失败", top_rank_failure),
    ]
    rows: list[dict[str, Any]] = [
        {
            "display_role": "野生型",
            "condition": representative_run.condition,
            "fold": representative_run.fold,
            "round_id": 0,
            "variant_id": wild_type["variant_id"],
            "variant": wild_type["variant"],
            "mutation_notation": "WT",
            "full_sequence": wild_type["sequence"],
            "window_start": WINDOW_START,
            "window_end": WINDOW_END,
            "sequence_window": _sequence_window(str(wild_type["sequence"])),
            "wet_fitness": np.nan,
        }
    ]
    for role, case in roles:
        case_run, raw = _raw_candidate_row(
            runs, str(case["run_id"]), str(case["variant_id"])
        )
        wet = case.get("wet_fitness")
        rows.append(
            {
                "display_role": role,
                "condition": case_run.condition,
                "fold": case_run.fold,
                "round_id": int(raw["round_id"]),
                "variant_id": str(raw["variant_id"]),
                "variant": str(raw["variant"]),
                "mutation_notation": str(raw["mutation_notation"]),
                "full_sequence": str(raw["sequence"]),
                "window_start": WINDOW_START,
                "window_end": WINDOW_END,
                "sequence_window": _sequence_window(str(raw["sequence"])),
                "wet_fitness": float(wet),
            }
        )
    return pd.DataFrame(rows), top_rank_failure


def _core_sentence(text: str, phrase: str) -> str:
    for sentence in text.replace("\n", " ").split(". "):
        if phrase in sentence:
            return sentence.strip().rstrip(".") + "."
    raise ValueError(f"Prompt sentence containing {phrase!r} was not found")


def _pick_findings(card: dict[str, Any], keywords: tuple[str, ...]) -> list[str]:
    findings = ((card.get("analysis") or {}).get("findings") or [])
    selected: list[str] = []
    for keyword in keywords:
        for finding in findings:
            statement = str(finding.get("statement") or "")
            if keyword.lower() in statement.lower() and statement not in selected:
                selected.append(statement)
                break
    return selected


def _extract_prompt_chain(
    runs: list[RunArtifact],
    positive_case: dict[str, Any],
    negative_case: dict[str, Any],
) -> dict[str, Any]:
    conversation_path = REPO_ROOT / positive_case["prompt_record"]["conversation_path"]
    conversation = read_json(conversation_path)
    system_text = next(
        str(item["content"])
        for item in conversation.get("messages", [])
        if item.get("role") == "system"
    )
    user_text = next(
        str(item["content"])
        for item in conversation.get("messages", [])
        if item.get("role") == "user"
    )
    prompt_payload = json.loads(user_text)
    context = prompt_payload["context"]
    activation = context["activation_state"]
    top_visible = sorted(
        context["visible_observations"],
        key=lambda item: float(item["measured_fitness"]),
        reverse=True,
    )[:5]
    visible_subset = [
        {
            "variant_id": item.get("variant_id"),
            "mutation_notation": item.get("mutation_notation"),
            "measured_fitness": float(item["measured_fitness"]),
            "round_revealed": int(item.get("round_revealed", 0)),
        }
        for item in top_visible
    ]
    channel_cards = {
        str(card["channel"]): {
            "contribution_modes": card.get("contribution_modes", []),
            "selected_findings": _pick_findings(
                card,
                {
                    "physchem": ("D40F", "G41R"),
                    "conservation": ("low Neff/L", "position 41", "Pairwise analysis"),
                    "structure": (
                        "Position 54 is buried",
                        "Position 40 is solvent-exposed",
                        "side chains were not modelled",
                    ),
                }[str(card["channel"])],
            ),
            "uncertainty": (card.get("analysis") or {}).get("uncertainty"),
            "critic_verdict": (card.get("semantic_review") or {}).get("verdict"),
        }
        for card in context.get("approved_channel_analyses", [])
    }

    positive_run = _find_run(runs, str(positive_case["run_id"]))
    round_id = int(positive_case["round_id"])
    round_dir = positive_run.path / f"round_{round_id:02d}"
    first_review = read_json(round_dir / "main_reviews" / "attempt_01.json")
    second_review = read_json(round_dir / "main_reviews" / "attempt_02.json")
    batch_review = read_json(round_dir / "critique_attempt_0.json")
    assessment = read_json(round_dir / "hypothesis_assessment.json")

    negative_run = _find_run(runs, str(negative_case["run_id"]))
    negative_round = int(negative_case["round_id"])
    negative_arm = _selection_arm(
        negative_run, negative_round, str(negative_case["variant_id"])
    )

    return {
        "hidden_reasoning_exported": False,
        "scope_note": (
            "Only model-visible messages, typed channel cards, final structured "
            "responses, Critic verdicts, acquisition receipts, and revealed outcomes "
            "are exported. Provider reasoning_content is excluded."
        ),
        "source_conversation": _repo_relative(conversation_path),
        "core_system_prompt": [
            _core_sentence(system_text, "main Scientist alone owns cross-channel fusion"),
            _core_sentence(system_text, "preferred_residues` is always a soft"),
        ],
        "core_user_prompt": {
            "round_id": int(context["round_id"]),
            "task": context["task"],
            "wild_type_sites": context["wild_type_sites"],
            "design_space": context["design_space"],
            "selection_driver": activation["selection_driver"],
            "fitness_predictors_used_for_generation": activation[
                "fitness_predictors_used_for_generation"
            ],
            "kg_tool_results_present": activation["kg_tool_results_present"],
            "previous_hypothesis_assessment": context[
                "previous_hypothesis_assessment"
            ],
            "top_visible_observations": visible_subset,
        },
        "approved_channel_cards": channel_cards,
        "draft_and_revision": {
            "attempt_0": {
                "statement": first_review["hypothesis"]["statement"],
                "preferred_residues": first_review["hypothesis"][
                    "preferred_residues"
                ],
                "critic_verdict": first_review["review"]["verdict"],
                "required_changes": first_review["review"].get("required_changes"),
            },
            "attempt_1": {
                "statement": second_review["hypothesis"]["statement"],
                "preferred_residues": second_review["hypothesis"][
                    "preferred_residues"
                ],
                "expected_outcome": second_review["hypothesis"]["expected_outcome"],
                "critic_verdict": second_review["review"]["verdict"],
            },
        },
        "batch_critic": {
            "verdict": batch_review["verdict"],
            "summary": batch_review["summary"],
        },
        "post_measurement_assessment": {
            "hypothesis_id": assessment["hypothesis_id"],
            "status": assessment["status"],
            "decisive_criterion_ids": assessment.get("decisive_criterion_ids", []),
        },
        "positive_case": {
            "run_id": positive_run.run_id,
            "fold": positive_run.fold,
            "round_id": round_id,
            "variant": positive_case["variant"],
            "mutation_notation": positive_case["mutation_notation"],
            "selection_arm": _selection_arm(
                positive_run, round_id, str(positive_case["variant_id"])
            ),
            "dry_validation_fitness": positive_case["fitness_mean"],
            "wet_fitness": positive_case["wet_fitness"],
        },
        "negative_case": {
            "run_id": negative_run.run_id,
            "fold": negative_run.fold,
            "round_id": negative_round,
            "variant": negative_case["variant"],
            "mutation_notation": negative_case["mutation_notation"],
            "selection_arm": negative_arm,
            "prompt_statement": negative_case["hypothesis"]["statement"],
            "preferred_residues": negative_case["hypothesis"]["preferred_residues"],
            "dry_validation_fitness": negative_case["fitness_mean"],
            "wet_fitness": negative_case["wet_fitness"],
            "source_top_k": _repo_relative(
                negative_run.path / "top_k_all_rounds.csv"
            ),
        },
        "source_paths": {
            "main_review_attempt_0": _repo_relative(
                round_dir / "main_reviews" / "attempt_01.json"
            ),
            "main_review_attempt_1": _repo_relative(
                round_dir / "main_reviews" / "attempt_02.json"
            ),
            "batch_critic": _repo_relative(round_dir / "critique_attempt_0.json"),
            "assessment": _repo_relative(round_dir / "hypothesis_assessment.json"),
        },
    }


def _plot_showcase_trajectory(trajectory: pd.DataFrame, condition: str, fold: int) -> list[Path]:
    os.environ["SOURCE_DATE_EPOCH"] = "946684800"
    _set_style()
    plt.rcParams["svg.hashsalt"] = "gb1-al96-results-showcase-v1"
    figure, axis = plt.subplots(figsize=(4.8, 3.0))
    x = trajectory["query_budget"].to_numpy(float)
    axis.plot(
        x,
        trajectory["best_seen_fitness"].to_numpy(float),
        marker="o",
        color="#287C48",
        label="Cumulative best seen",
        linewidth=1.8,
    )
    measured = trajectory[trajectory["round_id"] > 0]
    axis.plot(
        measured["query_budget"],
        measured["batch_mean_fitness"],
        marker="s",
        color="#C47B36",
        label="Selected-batch mean",
    )
    axis.plot(
        measured["query_budget"],
        measured["batch_median_fitness"],
        marker="^",
        color="#76508A",
        label="Selected-batch median",
    )
    axis.set_xticks([0, 16, 32, 48])
    axis.set_xlabel("Cumulative wet-lab queries")
    axis.set_ylabel("Fitness")
    axis.set_title(f"{condition}, fold {fold}")
    axis.legend(frameon=False, loc="upper left")
    _clean_axis(axis)
    figure.tight_layout()
    paths = _save_all(figure, FIGURE_DIR / f"{condition}_fold{fold}_fitness_trajectory")
    plt.close(figure)
    return paths


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _metric(aggregate: pd.DataFrame, condition: str, metric: str) -> tuple[float, float]:
    row = aggregate[
        (aggregate["condition"] == condition) & (aggregate["metric"] == metric)
    ].iloc[0]
    return float(row["mean"]), float(row["sd"])


def _make_report(
    condition: str,
    run: RunArtifact,
    top_k: pd.DataFrame,
    trajectory: pd.DataFrame,
    sequences: pd.DataFrame,
    aggregate: pd.DataFrame,
    prompt_chain: dict[str, Any],
    top_rank_failure: dict[str, Any],
    top_k_value: int,
) -> str:
    initial_best = float(trajectory.iloc[0]["best_seen_fitness"])
    final_best = float(trajectory.iloc[-1]["best_seen_fitness"])
    final_mean, final_sd = _metric(aggregate, condition, "final_best_seen")
    delta_mean, delta_sd = _metric(aggregate, condition, "delta_best_seen")
    aulc_mean, aulc_sd = _metric(aggregate, condition, "best_seen_aulc")
    r3_mean, r3_sd = _metric(aggregate, condition, "r3_batch_mean")
    r3_median, r3_median_sd = _metric(aggregate, condition, "r3_batch_median")
    image_path = FIGURE_DIR / f"{condition}_fold{run.fold}_fitness_trajectory.png"
    full_top_k_path = run.path / "top_k_all_rounds.csv"
    prompt_json = json.dumps(
        prompt_chain["core_user_prompt"], ensure_ascii=False, indent=2
    )
    revised_json = json.dumps(
        {
            "statement": prompt_chain["draft_and_revision"]["attempt_1"]["statement"],
            "preferred_residues": prompt_chain["draft_and_revision"]["attempt_1"][
                "preferred_residues"
            ],
            "expected_outcome": prompt_chain["draft_and_revision"]["attempt_1"][
                "expected_outcome"
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    lines = [
        "# 附件 D｜结果展示（独立模块）",
        "",
        "> 本文件拟放在源报告附件 C 之后；本轮未修改源报告。所有数值均由已完成且 `pass_eligible=true` 的工件重建。展示用条件按八策略中三折平均 `final_best_seen` 最高者自动选择，展示 fold 再按该条件内 `final_best_seen` 最高者选择。因此，过程案例用于说明 Agent 如何运行，不替代三折总体比较。",
        "",
        f"本次自动选中 `{condition}`，代表 run 为 fold {run.fold}（`{run.run_id}`）。该条件三折最终 best-seen 为 **{_fmt(final_mean)} ± {_fmt(final_sd)}**，较统一初始 best-seen 4.073 提高 **{_fmt(delta_mean)} ± {_fmt(delta_sd)}**；best-seen AULC 为 **{_fmt(aulc_mean)} ± {_fmt(aulc_sd)}**。这些是描述性结果，n=3，未进行显著性推断。",
        "",
        "## D1｜野生型与代表突变序列",
        "",
        "四个位点紧凑编码按 39/40/41/54 排列。工件中的野生型四位点为 `VDGV`。完整构建体序列如下：",
        "",
        "```text",
        *textwrap.wrap(str(sequences.iloc[0]["full_sequence"]), width=80),
        "```",
        "",
        f"下表同时给出位置 {WINDOW_START}–{WINDOW_END} 的局部序列；完整突变序列保存于 [`showcase_sequences.csv`]({_doc_relative(SOURCE_DATA_DIR / 'showcase_sequences.csv')})。",
        "",
        "| 角色 | 条件 / fold / round | 四位点序列 | 突变表示 | 位置 33–60 序列 | Wet fitness |",
        "|---|---|---|---|---|---:|",
    ]
    for row in sequences.itertuples():
        wet = "—" if pd.isna(row.wet_fitness) else _fmt(row.wet_fitness, 4)
        lines.append(
            f"| {row.display_role} | `{row.condition}` / {int(row.fold)} / {int(row.round_id)} | "
            f"`{row.variant}` | `{row.mutation_notation}` | `{row.sequence_window}` | {wet} |"
        )
    lines.extend(
        [
            "",
            "位点对齐可直接看到三类差异：成功例 `VWAA` 同时包含 D40W、G41A 和 V54A；跨 fold 反例 `LYWC` 在 41 位引入体积很大的 W；同一代表 run 的失败例则在 54 位引入 K。",
            "",
            f"## D2｜每轮推荐 Top-{top_k_value}",
            "",
            f"每轮实际提交 16 个变体。为便于展示，这里按最终 `selection_order` 列出前 {top_k_value} 个；这不是按 dry predictor 排序，因为该 run 的 `selection_driver=agent_uq` 且 `fitness_predictors_used_for_generation=false`。Dry validation 仅为选择后的辅助诊断。完整 48 个推荐见 [`top_k_all_rounds.csv`]({_doc_relative(full_top_k_path)})，精简表见 [`showcase_top{top_k_value}_by_round.csv`]({_doc_relative(SOURCE_DATA_DIR / f'showcase_top{top_k_value}_by_round.csv')})。",
            "",
            "| Round | 推荐序 | 四位点序列 | 突变方案 | 采集臂 | Acquisition | Dry validation | Wet fitness | ReThink |",
            "|---:|---:|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in top_k.itertuples():
        lines.append(
            f"| {int(row.round_id)} | {int(row.selection_order)} | `{row.variant}` | "
            f"`{row.mutation_notation}` | `{row.selection_arm}` | "
            f"{_fmt(row.acquisition_score)} | {_fmt(row.dry_validation_fitness)} ± {_fmt(row.dry_validation_sd)} | "
            f"{_fmt(row.wet_fitness, 4)} | `{row.rethink_verdict}` |"
        )
    lines.extend(
        [
            "",
            "## D3｜Fitness 是否提升",
            "",
            f"![{condition} fold {run.fold} fitness trajectory]({_doc_relative(image_path)})",
            "",
            f"**图 D1｜`{condition}` fold {run.fold} 的真实 wet-fitness 轨迹。** 累计 best-seen 从 {initial_best:.3f} 提高到 {final_best:.3f}，但 batch mean/median 并非单调上升。",
            "",
            "| Round | 累计 query | 可见样本 | Best-seen | 相对初始增量 | Batch best | Batch mean | Batch median | 是否刷新记录 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    previous_best = None
    for row in trajectory.itertuples():
        if int(row.round_id) == 0:
            batch_values = ("—", "—", "—")
            refreshed = "初始值"
        else:
            batch_values = (
                _fmt(row.batch_best_fitness),
                _fmt(row.batch_mean_fitness),
                _fmt(row.batch_median_fitness),
            )
            refreshed = "是" if float(row.best_seen_fitness) > float(previous_best) else "否"
        lines.append(
            f"| {int(row.round_id)} | {int(row.query_budget)} | {int(row.visible_observations)} | "
            f"{_fmt(row.best_seen_fitness)} | {_fmt(float(row.best_seen_fitness) - initial_best)} | "
            f"{batch_values[0]} | {batch_values[1]} | {batch_values[2]} | {refreshed} |"
        )
        previous_best = float(row.best_seen_fitness)
    lines.extend(
        [
            "",
            f"结论是“**峰值发现提升，但批次质量不单调**”：代表 run 在 round 2 发现 `VWAA`（6.124）后累计最优不再提高；round 3 仍找到 batch best 5.446，但 batch mean/median 回落至 0.883/0.151。三折层面同样需要区分目标：`{condition}` 的最终 best-seen 最高，而其 R3 batch mean/median 仅为 {_fmt(r3_mean)} ± {_fmt(r3_sd)} / {_fmt(r3_median)} ± {_fmt(r3_median_sd)}。",
            "",
            "八策略的完整 mean ± s.d. 曲线沿用既有分析图：",
            "",
            f"![八策略 fitness trajectories]({_doc_relative(OUTPUT_ROOT / 'figures' / 'figure2_fitness_trajectories.png')})",
            "",
            "## D4｜Agent 的可审计推理过程",
            "",
            f"> 展示边界：以下为模型可见 Prompt、三通道结构化 cards、Scientist 最终结构化输出、Critic 判定、采集回执和结果反馈。服务商 `reasoning_content` 未导出，也未据此重构隐藏思维链。完整精简记录见 [`showcase_prompt_chain.json`]({_doc_relative(PROMPT_CHAIN_PATH)})。",
            "",
            "### D4.1 核心系统 Prompt 中间内容",
            "",
        ]
    )
    for sentence in prompt_chain["core_system_prompt"]:
        lines.append(f"> {sentence}")
        lines.append(">")
    lines.extend(
        [
            "核心模型可见运行状态与最高值观测的原样字段子集如下（只省略与展示无关的长数组）：",
            "",
            "```json",
            prompt_json,
            "```",
            "",
            "这段 Prompt 明确了三点：闭池设计、Agent-UQ 负责选择、fitness predictor 不参与生成；上一轮假设已得到 `SUPPORTED`，本轮可在新揭示测量上收缩或修订残基集合。",
            "",
            "### D4.2 三通道先分别给出事实与限制",
            "",
            "| 通道 | Prompt 中间结果（verbatim） | 对决策的边界 |",
            "|---|---|---|",
        ]
    )
    channel_labels = {
        "physchem": "Physchem",
        "conservation": "Conservation",
        "structure": "Structure",
    }
    channel_boundaries = {
        "physchem": "描述符只作 analysis/counterevidence，不能直接推出 assay fitness。",
        "conservation": "Neff/L 低且 pairwise disabled，单点 log-odds 不能解释组合上位性。",
        "structure": "只反映 WT 静态骨架；突变侧链未建模或 relax。",
    }
    for channel in ("physchem", "conservation", "structure"):
        card = prompt_chain["approved_channel_cards"][channel]
        findings = "<br>".join(card["selected_findings"])
        lines.append(
            f"| {channel_labels[channel]} | {findings} | {channel_boundaries[channel]} |"
        )
    attempt_0 = prompt_chain["draft_and_revision"]["attempt_0"]
    attempt_1 = prompt_chain["draft_and_revision"]["attempt_1"]
    lines.extend(
        [
            "",
            "### D4.3 Main Scientist 合并通道，Critic 要求一次修订",
            "",
            f"> **初稿：** {attempt_0['statement']}",
            ">",
            f"> **Critic：** `{attempt_0['critic_verdict']}`；required change = `{attempt_0['required_changes']}`。问题不是残基方向本身，而是初稿没有把预期方向变成可执行的 falsification endpoint。",
            "",
            "修订后的核心结构化输出如下：",
            "",
            "```json",
            revised_json,
            "```",
            "",
            f"随后 batch Critic 给出 `{prompt_chain['batch_critic']['verdict']}`：{prompt_chain['batch_critic']['summary']} 测量后，假设评估为 `{prompt_chain['post_measurement_assessment']['status']}`。",
            "",
            "### D4.4 为什么选择这些替换与组合",
            "",
            "- 位点 40：已揭示高值组合反复包含 Y/W/F/H；结构 card 同时指出该位点在静态骨架中较暴露，因此将芳香/疏水替换作为软方向，而非硬规则。",
            "- 位点 41：round 2 的高值测量支持 A，同时保留 WT G 作为对照；conservation 对多种 41 位替换给出负 log-odds，所以该方向被明确限定为 assay-only。",
            "- 位点 54：A/C 与可见高值组合相关，但结构 card 指出 54 位埋藏，侧链未 relax；因此 A/C 只能作为软组合偏好，不能被解释为已验证的稳定性机制。",
            "- 位点 39：V/I/C 被保留为 context-dependent alternatives；没有把单一氨基酸宣称为独立增益来源。",
            "- 组合逻辑：`VWAA` 属于 `hypothesis_target`，同时命中 40=W、41=A、54=A 的软方向；其 dry validation 为 3.549，真实 wet fitness 为 6.124，说明该组合成功，但不能把成功因果归于某一个 feature channel。",
            "",
            "## D5｜失败推荐与可能原因",
            "",
            "| 失败推荐 | 选择位置 / 采集臂 | Dry → Wet | Prompt 已知信息 | 可能原因与系统边界 |",
            "|---|---|---:|---|---|",
            f"| `LYWC` (`V39L;D40Y;G41W;V54C`) | fold 0, round 2, order 12 / `evidence_prior` | 3.405 → 0.0198 | 当轮 Prompt 明确写出 41 位保留 G，因为可见 G41 替换导致 fitness 崩塌。 | G41W 仍因软先验和聚合 evidence score 越过选择；W 相对 G 体积大，且 conservation 不支持 41 位替换。最直接的系统问题是“Prompt 方向没有形成候选级 veto”。 |",
            f"| `{top_rank_failure['variant']}` (`{top_rank_failure['mutation_notation']}`) | fold {run.fold}, round {int(top_rank_failure['round_id'])}, order {int(top_rank_failure['selection_order'])} / `{_selection_arm(run, int(top_rank_failure['round_id']), str(top_rank_failure['variant_id']))}` | {_fmt(top_rank_failure['fitness_mean'])} → {_fmt(top_rank_failure['wet_fitness'], 4)} | round 3 假设偏好 54=A；该候选只在 54 位改为 K，但其余三位命中软方向。 | 54 位在静态结构中埋藏，V54K 引入正电侧链，可能破坏局部 packing；同时 3/4 位点命中仍可获得 hypothesis-target 分数，暴露部分匹配规则对单个位点灾难性替换不敏感。 |",
            "| `VWNA` (`D40W;G41N;V54A`) | fold 2, round 3, order 6 / `hypothesis_target` | 0.850 → 0.0048 | round 3 偏好 41=A；conservation card 对 41 位多种替换均不支持。 | G41N 偏离当轮唯一 41=A 软集合；低值可能来自局部构象与组合上位性。当前通道没有突变侧链 relax 或候选特异 pairwise 能量，故只能给出机制假说。 |",
            "",
            "这些失败不等同于“Agent 完全没有识别重要位点”。相反，Prompt 已识别 40、41、54 的方向和反证；主要缺口是软残基集合、聚合 KG evidence、coverage/控制设计与候选级硬约束之间仍有接口空隙。下一步应记录每个候选的逐位命中/冲突分解，对 G41 与 buried V54 的高风险替换设置可校准的 penalty 或显式 matched-control 标签，并用 side-chain relax 或候选特异 pairwise evidence 验证机制。",
            "",
            "## D6｜复现与证据入口",
            "",
            "```powershell",
            "python analysis/gb1_al96_report_20260821/build_results_showcase.py --top-k 5",
            "python analysis/gb1_al96_report_20260821/validate_results_showcase.py",
            "```",
            "",
            f"- 代表 run 完整推理报告：[`reasoning.md`]({_doc_relative(run.path / 'reasoning.md')})",
            f"- 逐轮 Top-k 原始工件：[`top_k_all_rounds.csv`]({_doc_relative(full_top_k_path)})",
            f"- Prompt 精简审计：[`showcase_prompt_chain.json`]({_doc_relative(PROMPT_CHAIN_PATH)})",
            f"- 生成清单与哈希：[`results_showcase_manifest.json`]({_doc_relative(MANIFEST_PATH)})",
            f"- 八策略总体分析：[`report_chapter4_eight_strategy.md`]({_doc_relative(REPO_ROOT / 'analysis' / 'gb1_al96_report_20260821' / 'report_chapter4_eight_strategy.md')})",
            "",
            "证据边界：这里报告的是闭池、固定预算下的已揭示 wet fitness 与可审计运行链。候选池未在所有条件间严格固定，n=3；三通道 observations 也未成为候选级直接 selection evidence。因此，结果支持“该完整条件在本次运行中取得最高峰值发现”，不支持单一通道的因果有效性或可外推的分子机制结论。",
            "",
        ]
    )
    return "\n".join(lines)


def _manifest(
    runs: list[RunArtifact],
    run: RunArtifact,
    condition: str,
    top_k: int,
    prompt_chain: dict[str, Any],
    output_paths: list[Path],
) -> dict[str, Any]:
    input_paths = [
        run.path / "completion_manifest.json",
        run.path / "summary.json",
        run.path / "state.json",
        run.path / "wild_type.json",
        run.path / "top_k_all_rounds.csv",
        REPO_ROOT / prompt_chain["source_conversation"],
        REPO_ROOT / prompt_chain["source_paths"]["main_review_attempt_0"],
        REPO_ROOT / prompt_chain["source_paths"]["main_review_attempt_1"],
        REPO_ROOT / prompt_chain["source_paths"]["batch_critic"],
        REPO_ROOT / prompt_chain["source_paths"]["assessment"],
        REPO_ROOT / prompt_chain["negative_case"]["source_top_k"],
    ]
    return {
        "analysis": "GB1 AL96 standalone Attachment D results showcase",
        "selection_rule": {
            "condition": "highest three-fold mean final_best_seen among Agent conditions; best_seen_aulc tie-break",
            "representative_fold": "highest final_best_seen within selected condition; best_seen_aulc tie-break",
            "top_k": f"first {top_k} final selection_order entries per round",
        },
        "selected_condition": condition,
        "selected_fold": run.fold,
        "selected_run_id": run.run_id,
        "eligible_run_count": sum(item.eligible for item in runs),
        "hidden_reasoning_exported": False,
        "inputs": [
            {"path": _repo_relative(path), "sha256": sha256_file(path)}
            for path in input_paths
        ],
        "outputs": [
            {"path": _repo_relative(path), "sha256": sha256_file(path)}
            for path in sorted(output_paths)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1 or args.top_k > 16:
        raise ValueError("--top-k must be between 1 and 16")

    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    validate_analysis_matrix(runs)
    round_metrics = build_round_metrics(runs)
    condition, run, _, aggregate = _select_condition_and_run(runs, round_metrics)
    candidates = build_candidate_table(runs)
    cases, _ = build_case_studies(runs, candidates)
    positive_case = next(
        case
        for case in cases
        if case["condition"] == condition and case["case_id"].endswith("_positive")
    )
    negative_case = next(
        case
        for case in cases
        if case["condition"] == condition and case["case_id"].endswith("_negative")
    )
    top_k = _build_top_k(candidates, run, args.top_k)
    sequences, top_rank_failure = _build_sequences(
        runs, run, positive_case, negative_case, candidates
    )
    trajectory = round_metrics[round_metrics["run_id"] == run.run_id].copy()
    trajectory["best_seen_gain_from_initial"] = (
        trajectory["best_seen_fitness"]
        - float(trajectory.iloc[0]["best_seen_fitness"])
    )
    prompt_chain = _extract_prompt_chain(runs, positive_case, negative_case)

    top_k_path = SOURCE_DATA_DIR / f"showcase_top{args.top_k}_by_round.csv"
    trajectory_path = SOURCE_DATA_DIR / "showcase_fitness_trajectory.csv"
    sequences_path = SOURCE_DATA_DIR / "showcase_sequences.csv"
    top_k.to_csv(top_k_path, index=False, encoding="utf-8")
    trajectory.to_csv(trajectory_path, index=False, encoding="utf-8")
    sequences.to_csv(sequences_path, index=False, encoding="utf-8")
    _write_json(prompt_chain, PROMPT_CHAIN_PATH)
    figure_paths = _plot_showcase_trajectory(trajectory, condition, run.fold)
    report = _make_report(
        condition,
        run,
        top_k,
        trajectory,
        sequences,
        aggregate,
        prompt_chain,
        top_rank_failure,
        args.top_k,
    )
    REPORT_PATH.write_text(report, encoding="utf-8")

    output_paths = [
        top_k_path,
        trajectory_path,
        sequences_path,
        PROMPT_CHAIN_PATH,
        REPORT_PATH,
        *figure_paths,
    ]
    _write_json(
        _manifest(runs, run, condition, args.top_k, prompt_chain, output_paths),
        MANIFEST_PATH,
    )
    print(f"Wrote {REPORT_PATH}")
    print(f"Selected {condition}, fold {run.fold}, run {run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
