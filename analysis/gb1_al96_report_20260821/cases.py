"""Traceable candidate-level case selection and Prompt/KG extraction."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    AGENT_CONDITIONS,
    CONDITION_ORDER,
    EXPECTED_ROUNDS,
    KG_CONDITIONS,
    REPO_ROOT,
)
from io_artifacts import RunArtifact, query_structured_kg, read_json


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    return [str(item) for item in parsed] if isinstance(parsed, list) else [str(parsed)]


def _compact_text(value: Any, limit: int = 1800) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_candidate_table(runs: list[RunArtifact]) -> pd.DataFrame:
    """Join selected candidates to wet values, human-readable variants, and evidence."""

    rows: list[dict[str, Any]] = []
    eligible = [
        run
        for run in runs
        if run.eligible and run.condition in CONDITION_ORDER
    ]
    for run in eligible:
        state = read_json(run.path / "state.json")
        wet_by_id = {
            str(item["variant_id"]): float(item["fitness"])
            for item in state.get("observed", [])
            if int(item.get("round_revealed", -1)) > 0
        }
        selected = pd.read_csv(run.path / "top_k_all_rounds.csv")
        for round_id in EXPECTED_ROUNDS:
            evidence_path = run.path / f"round_{round_id:02d}" / "selected_evidence.json"
            evidence_map = read_json(evidence_path) if evidence_path.exists() else {}
            round_rows = selected[selected["round_id"] == round_id]
            for _, item in round_rows.iterrows():
                variant_id = str(item["variant_id"])
                evidence = evidence_map.get(variant_id, [])
                evidence_ids = _parse_list(item.get("evidence_ids"))
                first = evidence[0] if evidence else {}
                rows.append(
                    {
                        "condition": run.condition,
                        "fold": run.fold,
                        "seed": run.seed,
                        "run_id": run.run_id,
                        "round_id": round_id,
                        "selection_order": int(item["selection_order"]),
                        "variant_id": variant_id,
                        "variant": item.get("variant"),
                        "mutation_notation": item.get("mutation_notation"),
                        "mutation_count": int(item.get("mutation_count", 0)),
                        "wet_fitness": wet_by_id.get(variant_id, np.nan),
                        "fitness_mean": float(item.get("fitness_mean", np.nan)),
                        "fitness_std": float(item.get("fitness_std", np.nan)),
                        "dry_wet_gap": float(item.get("fitness_mean", np.nan))
                        - wet_by_id.get(variant_id, np.nan),
                        "acquisition_score": float(
                            item.get("acquisition_score", np.nan)
                        ),
                        "knowledge_score": float(item.get("knowledge_score", np.nan)),
                        "design_score": float(item.get("design_score", np.nan)),
                        "selection_driver": item.get("selection_driver"),
                        "hypothesis_id": item.get("hypothesis_id"),
                        "reason": item.get("reason"),
                        "rethink_verdict": item.get("rethink_verdict"),
                        "rethink_summary": item.get("rethink_summary"),
                        "evidence_count": len(evidence),
                        "evidence_ids": json.dumps(
                            evidence_ids, ensure_ascii=False, separators=(",", ":")
                        ),
                        "evidence_type": first.get("evidence_type"),
                        "evidence_channel": first.get("channel"),
                        "evidence_source_group": first.get("source_group"),
                        "evidence_quality_status": first.get("quality_status"),
                        "evidence_statement": first.get("statement"),
                        "run_path": _repo_relative(run.path),
                    }
                )
    frame = pd.DataFrame(rows)
    expected = len(eligible) * len(EXPECTED_ROUNDS) * 16
    if len(frame) != expected:
        raise ValueError(f"Expected {expected} selected candidates, found {len(frame)}")
    if frame["wet_fitness"].isna().any():
        missing = frame.loc[frame["wet_fitness"].isna(), "variant_id"].tolist()
        raise ValueError(f"Missing wet fitness for selected candidates: {missing[:3]}")
    return frame.sort_values(
        ["condition", "fold", "round_id", "selection_order"]
    ).reset_index(drop=True)


def _zscore(series: pd.Series) -> pd.Series:
    sd = float(series.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series.astype(float) - float(series.mean())) / sd


def select_case_rows(candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select global KG and condition-specific success/failure cases."""

    audit = candidates[candidates["condition"].isin(AGENT_CONDITIONS)].copy()
    audit["wet_percentile_within_condition"] = audit.groupby("condition")[
        "wet_fitness"
    ].rank(pct=True, method="average")
    audit["acquisition_percentile_within_condition"] = audit.groupby("condition")[
        "acquisition_score"
    ].rank(pct=True, method="average")
    audit["surprise_score"] = (
        _zscore(audit["dry_wet_gap"].fillna(0))
        + 0.5 * _zscore(audit["acquisition_score"].fillna(0))
        + 0.25 * _zscore(audit["knowledge_score"].fillna(0))
    )
    core_kg_audit = audit[
        audit["condition"].isin(("kg_base", "kg_base_rag", "kg_base_al"))
    ]
    positive_index = core_kg_audit.sort_values(
        ["wet_fitness", "acquisition_score", "evidence_count", "variant_id"],
        ascending=[False, False, False, True],
    ).index[0]
    negative_pool = core_kg_audit[
        (core_kg_audit["wet_percentile_within_condition"] <= 0.25)
        & (core_kg_audit["acquisition_percentile_within_condition"] >= 0.75)
    ]
    if negative_pool.empty:
        negative_pool = core_kg_audit.nlargest(
            max(1, len(core_kg_audit) // 4), "dry_wet_gap"
        )
    negative_index = negative_pool.sort_values(
        ["surprise_score", "dry_wet_gap", "variant_id"],
        ascending=[False, False, True],
    ).index[0]
    condition_case_indices: list[tuple[str, int]] = []
    for condition, prefix in (
        ("kg_3features_rag", "feature_rag"),
        ("kg_3features_base", "feature_base"),
        ("agent_only", "agent_only"),
    ):
        group = audit[audit["condition"] == condition]
        condition_positive = group.sort_values(
            ["wet_fitness", "acquisition_score", "evidence_count", "variant_id"],
            ascending=[False, False, False, True],
        ).index[0]
        condition_negative_pool = group[
            (group["wet_percentile_within_condition"] <= 0.25)
            & (group["acquisition_percentile_within_condition"] >= 0.75)
        ]
        if condition_negative_pool.empty:
            condition_negative_pool = group.nlargest(
                max(1, len(group) // 4), "dry_wet_gap"
            )
        condition_negative = condition_negative_pool.sort_values(
            ["surprise_score", "dry_wet_gap", "variant_id"],
            ascending=[False, False, True],
        ).index[0]
        condition_case_indices.extend(
            [
                (f"{prefix}_positive", condition_positive),
                (f"{prefix}_negative", condition_negative),
            ]
        )
    audit["selected_case"] = ""
    audit.loc[positive_index, "selected_case"] = "positive"
    audit.loc[negative_index, "selected_case"] = "negative"
    for case_id, index in condition_case_indices:
        audit.loc[index, "selected_case"] = case_id
    selected_pairs = [
        ("positive", positive_index),
        ("negative", negative_index),
        *condition_case_indices,
    ]
    selected = audit.loc[[index for _, index in selected_pairs]].copy()
    selected["case_id"] = [case_id for case_id, _ in selected_pairs]
    return selected.reset_index(drop=True), audit.reset_index(drop=True)


def build_condition_shortlist(audit: pd.DataFrame) -> pd.DataFrame:
    """Keep one success and one surprise failure for each Agent/KG condition."""

    rows = []
    for condition in AGENT_CONDITIONS:
        group = audit[audit["condition"] == condition]
        positive = group.sort_values(
            ["wet_fitness", "acquisition_score", "variant_id"],
            ascending=[False, False, True],
        ).iloc[0]
        negative_pool = group[
            (group["wet_percentile_within_condition"] <= 0.25)
            & (group["acquisition_percentile_within_condition"] >= 0.75)
        ]
        if negative_pool.empty:
            negative_pool = group.nlargest(max(1, len(group) // 4), "dry_wet_gap")
        negative = negative_pool.sort_values(
            ["surprise_score", "dry_wet_gap", "variant_id"],
            ascending=[False, False, True],
        ).iloc[0]
        for case_type, item in (("positive", positive), ("negative", negative)):
            record = item.to_dict()
            record["shortlist_case"] = case_type
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["condition", "shortlist_case"], ascending=[True, False]
    ).reset_index(drop=True)


def _find_run(runs: list[RunArtifact], run_id: str) -> RunArtifact:
    matches = [run for run in runs if run.run_id == run_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one run for {run_id}, found {len(matches)}")
    return matches[0]


def _read_first_json(paths: list[Path]) -> tuple[Path | None, dict[str, Any]]:
    for path in paths:
        try:
            return path, read_json(path)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
    return None, {}


def _extract_prompt_record(round_dir: Path) -> dict[str, Any]:
    conversation_dir = round_dir / "llm" / "scientist" / "conversations"
    path, conversation = _read_first_json(sorted(conversation_dir.glob("*.json")))
    messages = conversation.get("messages", [])
    visible_messages = [
        {
            "role": message.get("role"),
            "content_excerpt": _compact_text(message.get("content"), 1800),
        }
        for message in messages
        if message.get("role") in {"system", "user"}
    ]
    return {
        "conversation_path": _repo_relative(path) if path else None,
        "conversation_stage": conversation.get("conversation_stage"),
        "disposition": conversation.get("disposition"),
        "model": conversation.get("model"),
        "visible_input_messages": visible_messages,
        "response_excerpt": _compact_text(conversation.get("response_content"), 2200),
        "hidden_reasoning_exported": False,
    }


def _extract_feature_prompt_records(round_dir: Path) -> list[dict[str, Any]]:
    records = []
    for channel in ("physchem", "conservation", "structure"):
        conversation_dir = (
            round_dir / "llm" / f"subscientist_{channel}" / "conversations"
        )
        accepted = []
        for path in sorted(conversation_dir.glob("*.json")):
            payload = read_json(path)
            if str(payload.get("disposition", "")).lower() == "accepted":
                accepted.append((path, payload))
        if not accepted:
            continue
        path, conversation = accepted[-1]
        messages = conversation.get("messages", [])
        records.append(
            {
                "channel": channel,
                "conversation_path": _repo_relative(path),
                "model": conversation.get("model"),
                "visible_input_messages": [
                    {
                        "role": message.get("role"),
                        "content_excerpt": _compact_text(
                            message.get("content"), 1200
                        ),
                    }
                    for message in messages
                    if message.get("role") in {"system", "user"}
                ],
                "response_excerpt": _compact_text(
                    conversation.get("response_content"), 1600
                ),
                "hidden_reasoning_exported": False,
            }
        )
    return records


def _hypothesis_for_case(state: dict[str, Any], hypothesis_id: str) -> dict[str, Any]:
    for hypothesis in state.get("hypotheses", []):
        if str(hypothesis.get("hypothesis_id")) == str(hypothesis_id):
            return hypothesis
    return {}


def _kg_interaction_excerpt(round_dir: Path) -> dict[str, Any]:
    path = round_dir / "kg_interaction.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    packs = payload.get("packs", [])
    first_pack = packs[0] if packs else {}
    return {
        "path": _repo_relative(path),
        "executed_steps": payload.get("executed_steps", []),
        "skipped_steps": payload.get("skipped_steps", []),
        "stop_reason": payload.get("stop_reason"),
        "fact_count_first_pack": len(first_pack.get("facts", [])),
        "evidence_count_first_pack": len(first_pack.get("evidence", [])),
        "directional_signal_count_first_pack": len(
            first_pack.get("directional_signals", [])
        ),
        "first_facts": first_pack.get("facts", [])[:4],
        "caveats": first_pack.get("caveats", [])[:4],
    }


def build_case_studies(
    runs: list[RunArtifact], candidates: pd.DataFrame
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    selected, audit = select_case_rows(candidates)
    cases: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        run = _find_run(runs, str(row["run_id"]))
        round_id = int(row["round_id"])
        round_dir = run.path / f"round_{round_id:02d}"
        state = read_json(run.path / "state.json")
        evidence_path = round_dir / "selected_evidence.json"
        evidence_map = read_json(evidence_path) if evidence_path.exists() else {}
        evidence = evidence_map.get(str(row["variant_id"]), [])
        db_path = run.path / "structured_kg.sqlite"
        subgraph = (
            query_structured_kg(db_path, str(row["variant_id"]))
            if db_path.exists()
            else {"entities": [], "relations": []}
        )
        case = {
            "case_id": row["case_id"],
            "selection_rule": (
                "maximum observed wet fitness across eligible KG selections"
                if row["case_id"] == "positive"
                else (
                    "low-wet/high-acquisition KG candidate with maximum surprise score"
                    if row["case_id"] == "negative"
                    else (
                        f"maximum observed wet fitness within {row['condition']}"
                        if str(row["case_id"]).endswith("_positive")
                        else f"{row['condition']} low-wet/high-acquisition candidate with maximum surprise score"
                    )
                )
            ),
            "condition": row["condition"],
            "fold": int(row["fold"]),
            "round_id": round_id,
            "run_id": row["run_id"],
            "variant_id": row["variant_id"],
            "variant": row["variant"],
            "mutation_notation": row["mutation_notation"],
            "wet_fitness": float(row["wet_fitness"]),
            "fitness_mean": float(row["fitness_mean"]),
            "fitness_std": float(row["fitness_std"]),
            "dry_wet_gap": float(row["dry_wet_gap"]),
            "acquisition_score": float(row["acquisition_score"]),
            "knowledge_score": float(row["knowledge_score"]),
            "hypothesis": _hypothesis_for_case(
                state, str(row.get("hypothesis_id", ""))
            ),
            "selected_evidence": evidence,
            "kg_subgraph": subgraph,
            "kg_interaction": _kg_interaction_excerpt(round_dir),
            "prompt_record": _extract_prompt_record(round_dir),
            "feature_prompt_records": _extract_feature_prompt_records(round_dir),
            "source_paths": {
                "state": _repo_relative(run.path / "state.json"),
                "top_k": _repo_relative(run.path / "top_k_all_rounds.csv"),
                "selected_evidence": _repo_relative(
                    evidence_path
                ),
                "structured_kg": _repo_relative(db_path),
            },
        }
        cases.append(case)
    return cases, audit


def cases_to_markdown(cases: list[dict[str, Any]]) -> str:
    lines = [
        "# Prompt/KG case audit",
        "",
        f"Cases are selected by deterministic rules from all {len(AGENT_CONDITIONS) * 3 * 3 * 16} Agent/KG-selected candidates. "
        "Only model-visible inputs and final responses are exported; hidden reasoning is excluded.",
        "",
    ]
    for case in cases:
        label = {
            "positive": "全局符合预期案例",
            "negative": "全局偏离预期案例",
            "feature_rag_positive": "三通道 + RAG 符合预期案例",
            "feature_rag_negative": "三通道 + RAG 偏离预期案例",
            "feature_base_positive": "无 RAG 三通道符合预期案例",
            "feature_base_negative": "无 RAG 三通道偏离预期案例",
            "agent_only_positive": "Agent-only 符合预期案例",
            "agent_only_negative": "Agent-only 偏离预期案例",
        }[case["case_id"]]
        lines.extend(
            [
                f"## {label}: {case['condition']} / fold {case['fold']} / round {case['round_id']}",
                "",
                f"- 变体：`{case['variant']}`（{case['mutation_notation']}）",
                f"- 实测 fitness：{case['wet_fitness']:.4f}",
                f"- 预测 fitness：{case['fitness_mean']:.4f} ± {case['fitness_std']:.4f}",
                f"- acquisition / knowledge：{case['acquisition_score']:.4f} / {case['knowledge_score']:.4f}",
                f"- 自动选择规则：{case['selection_rule']}",
                "",
                "### Scientist 输入摘录",
                "",
            ]
        )
        for message in case["prompt_record"]["visible_input_messages"]:
            lines.extend(
                [
                    f"**{message['role']}**",
                    "",
                    f"> {message['content_excerpt']}",
                    "",
                ]
            )
        lines.extend(
            [
                "### Scientist 最终输出摘录",
                "",
                f"> {case['prompt_record']['response_excerpt']}",
                "",
                "### 证据与 KG 摘要",
                "",
                f"- 选中证据记录数：{len(case['selected_evidence'])}",
                f"- 子图实体 / 关系数：{len(case['kg_subgraph']['entities'])} / {len(case['kg_subgraph']['relations'])}",
                f"- KG 交互首包事实数：{case['kg_interaction'].get('fact_count_first_pack', 0)}",
                "",
            ]
        )
        if case["feature_prompt_records"]:
            lines.extend(["### 三通道 Sub-Scientist 输入输出摘录", ""])
            for record in case["feature_prompt_records"]:
                lines.extend(
                    [
                        f"**{record['channel']} channel**",
                        "",
                        f"> 输入：{record['visible_input_messages'][-1]['content_excerpt']}",
                        "",
                        f"> 输出：{record['response_excerpt']}",
                        "",
                    ]
                )
        for evidence in case["selected_evidence"][:3]:
            lines.append(
                f"- `{evidence.get('evidence_id')}`: {evidence.get('statement')} "
                f"(type={evidence.get('evidence_type')}, quality={evidence.get('quality_status')})"
            )
        lines.append("")
    return "\n".join(lines)


def shortlist_to_markdown(shortlist: pd.DataFrame) -> str:
    lines = [
        "# Condition-level case shortlist",
        "",
        "| Condition | Case | Fold | Round | Variant | Mutation | Predicted fitness | Wet fitness | Acquisition | Knowledge |",
        "|---|---|---:|---:|---|---|---:|---:|---:|---:|",
    ]
    for _, row in shortlist.iterrows():
        lines.append(
            f"| {row['condition']} | {row['shortlist_case']} | {int(row['fold'])} | "
            f"{int(row['round_id'])} | {row['variant']} | {row['mutation_notation']} | "
            f"{float(row['fitness_mean']):.4f} ± {float(row['fitness_std']):.4f} | "
            f"{float(row['wet_fitness']):.4f} | {float(row['acquisition_score']):.4f} | "
            f"{float(row['knowledge_score']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive cases maximize wet fitness within each condition. Negative cases use the predeclared low-wet/high-acquisition surprise rule.",
            "",
        ]
    )
    return "\n".join(lines)
