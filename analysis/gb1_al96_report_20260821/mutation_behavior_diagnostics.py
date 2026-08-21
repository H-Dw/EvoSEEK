"""Audit closed-pool mutation-site, candidate-pool, and batch-selection semantics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cases import build_candidate_table
from config import CONDITION_ORDER, EXPECTED_ROUNDS, REPO_ROOT
from io_artifacts import RunArtifact, read_json


MUTABLE_POSITIONS = (39, 40, 41, 54)
WILD_TYPE_SITES = "VDGV"
KG_CONDITIONS = ("kg_base", "kg_base_rag", "kg_base_al")
SPLIT_ROOT = (
    REPO_ROOT
    / "data"
    / "processed"
    / "splits"
    / "GB1"
    / "al96_closed_loop"
    / "GB1-AL96-5CV-v1"
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _variant_positions(variant: str) -> set[int]:
    if len(variant) != len(WILD_TYPE_SITES):
        raise ValueError(f"Expected a four-site GB1 variant, found {variant!r}")
    return {
        position
        for position, wild_type, residue in zip(
            MUTABLE_POSITIONS, WILD_TYPE_SITES, variant, strict=True
        )
        if residue != wild_type
    }


def _substitution_pairs(variant: str) -> set[str]:
    return {
        f"{position}{residue}"
        for position, wild_type, residue in zip(
            MUTABLE_POSITIONS, WILD_TYPE_SITES, variant, strict=True
        )
        if residue != wild_type
    }


def _mutation_notation(variant: str) -> str:
    tokens = [
        f"{wild_type}{position}{residue}"
        for position, wild_type, residue in zip(
            MUTABLE_POSITIONS, WILD_TYPE_SITES, variant, strict=True
        )
        if residue != wild_type
    ]
    return ";".join(tokens) if tokens else "WT"


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def _is_strict_add_one_child(child: str, parent: str) -> bool:
    """True only when child retains the parent and adds one WT->mutant edit."""

    added = 0
    for wild_type, parent_residue, child_residue in zip(
        WILD_TYPE_SITES, parent, child, strict=True
    ):
        if parent_residue != wild_type:
            if child_residue != parent_residue:
                return False
        elif child_residue != wild_type:
            added += 1
    return added == 1


def _catalog(fold: int) -> pd.DataFrame:
    path = SPLIT_ROOT / f"fold_{fold:02d}" / "agent" / "candidate_pool.csv.gz"
    frame = pd.read_csv(
        path,
        usecols=["variant_id", "variant", "mutation_count", "mutated_positions"],
    )
    if frame["variant_id"].duplicated().any():
        raise ValueError(f"Duplicate candidate IDs in {path}")
    return frame


def _initial_variants(fold: int) -> pd.DataFrame:
    path = (
        SPLIT_ROOT
        / f"fold_{fold:02d}"
        / "agent"
        / "initial_or_train_observed.csv.gz"
    )
    return pd.read_csv(path, usecols=["variant_id", "variant", "mutation_count"])


def _accepted_scientist_output(round_dir: Path) -> tuple[Path, dict[str, Any]]:
    conversation_dir = round_dir / "llm" / "scientist" / "conversations"
    paths = sorted(conversation_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No Scientist conversation in {conversation_dir}")
    records = [(path, read_json(path)) for path in paths]
    accepted = [
        item
        for item in records
        if item[1].get("disposition") == "accepted"
        and item[1].get("conversation_stage") == "reasoning_draft"
    ]
    path, record = (accepted or records)[-1]
    return path, json.loads(record["response_content"])


def _round_pool(
    run: RunArtifact,
    round_id: int,
    catalog: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    receipt_path = run.path / f"round_{round_id:02d}" / "candidate_pool_receipt.json"
    receipt = read_json(receipt_path)
    ids = [str(value) for value in receipt.get("candidate_ids", [])]
    catalog_by_id = catalog.set_index("variant_id", drop=False)
    missing = sorted(set(ids) - set(catalog_by_id.index))
    if missing:
        raise ValueError(f"Candidate IDs missing from split catalog: {missing[:3]}")
    frame = catalog_by_id.loc[ids].reset_index(drop=True).copy()
    frame["pool_order"] = np.arange(1, len(frame) + 1)
    if len(frame) != int(receipt["actual_candidate_count"]):
        raise ValueError(f"Pool receipt mismatch: {receipt_path}")
    return receipt, frame


def build_mutation_behavior_tables(
    runs: list[RunArtifact],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build round, position-set, candidate-pool, and lineage audit tables."""

    selected_all = build_candidate_table(runs)
    eligible = sorted(
        [run for run in runs if run.eligible and run.condition in CONDITION_ORDER],
        key=lambda item: (CONDITION_ORDER.index(item.condition), item.fold),
    )
    catalogs = {fold: _catalog(fold) for fold in {run.fold for run in eligible}}
    initials = {fold: _initial_variants(fold) for fold in {run.fold for run in eligible}}

    round_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    pool_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []

    for run in eligible:
        catalog = catalogs[run.fold]
        initial = initials[run.fold]
        initial_variants = [str(value) for value in initial["variant"]]
        initial_positions = set().union(*(_variant_positions(value) for value in initial_variants))
        initial_pairs = set().union(*(_substitution_pairs(value) for value in initial_variants))
        initial_ids = set(str(value) for value in initial["variant_id"])
        if len(initial) != 96 or len(initial_positions) != 4 or len(initial_pairs) != 76:
            raise ValueError(
                f"Unexpected AL96 initial coverage for fold {run.fold}: "
                f"n={len(initial)} positions={len(initial_positions)} pairs={len(initial_pairs)}"
            )

        historical_selected_ids: set[str] = set()
        historical_selected_positions: set[int] = set()
        historical_selected_pairs: set[str] = set()
        previous_round_variants: list[str] = []

        for round_id in EXPECTED_ROUNDS:
            round_dir = run.path / f"round_{round_id:02d}"
            receipt, pool = _round_pool(run, round_id, catalog)
            selected = selected_all[
                (selected_all["run_id"] == run.run_id)
                & (selected_all["round_id"] == round_id)
            ].copy()
            if len(selected) != 16:
                raise ValueError(f"Expected 16 selected variants for {run.run_id}/r{round_id}")
            selected_ids = set(str(value) for value in selected["variant_id"])
            if not selected_ids.issubset(set(pool["variant_id"].astype(str))):
                raise ValueError(f"Selected batch escaped the round pool: {run.run_id}/r{round_id}")

            preferred: dict[int, list[str]] = {}
            conversation_path: Path | None = None
            if run.condition in KG_CONDITIONS:
                conversation_path, response = _accepted_scientist_output(round_dir)
                preferred = {
                    int(position): [str(residue) for residue in residues]
                    for position, residues in (response.get("preferred_residues") or {}).items()
                }

            pool_variants = [str(value) for value in pool["variant"]]
            selected_variants = [str(value) for value in selected["variant"]]
            pool_positions = set().union(*(_variant_positions(value) for value in pool_variants))
            selected_positions = set().union(
                *(_variant_positions(value) for value in selected_variants)
            )
            selected_pairs = set().union(
                *(_substitution_pairs(value) for value in selected_variants)
            )
            selected_depths = selected["mutation_count"].astype(int)
            pool_depths = pool["mutation_count"].astype(int)
            new_ids_vs_visible = selected_ids - initial_ids - historical_selected_ids
            new_positions_vs_visible = selected_positions - initial_positions - historical_selected_positions
            new_pairs_vs_visible = selected_pairs - initial_pairs - historical_selected_pairs
            new_positions_vs_campaign = selected_positions - historical_selected_positions
            new_pairs_vs_campaign = selected_pairs - historical_selected_pairs

            hamming_one_count = 0
            strict_add_one_count = 0
            if previous_round_variants:
                hamming_one_count = sum(
                    any(_hamming(child, parent) == 1 for parent in previous_round_variants)
                    for child in selected_variants
                )
                strict_add_one_count = sum(
                    any(
                        _is_strict_add_one_child(child, parent)
                        for parent in previous_round_variants
                    )
                    for child in selected_variants
                )

            round_rows.append(
                {
                    "condition": run.condition,
                    "fold": run.fold,
                    "round_id": round_id,
                    "run_id": run.run_id,
                    "sampling_strategy": receipt.get("sampling_strategy"),
                    "catalog_remaining": int(receipt["catalog_candidate_count"]),
                    "candidate_pool_size": len(pool),
                    "selected_batch_size": len(selected),
                    "candidate_pool_mutated_positions_json": _json(sorted(pool_positions)),
                    "selected_mutated_positions_json": _json(sorted(selected_positions)),
                    "candidate_pool_unique_mutated_position_count": len(pool_positions),
                    "selected_unique_mutated_position_count": len(selected_positions),
                    "scientist_preferred_position_count": len(preferred),
                    "scientist_covers_all_configured_positions": set(preferred) == set(MUTABLE_POSITIONS),
                    "scientist_preferred_residues_json": _json(preferred),
                    "candidate_depth_1_count": int((pool_depths == 1).sum()),
                    "candidate_depth_2_count": int((pool_depths == 2).sum()),
                    "candidate_depth_3_count": int((pool_depths == 3).sum()),
                    "candidate_depth_4_count": int((pool_depths == 4).sum()),
                    "selected_depth_1_count": int((selected_depths == 1).sum()),
                    "selected_depth_2_count": int((selected_depths == 2).sum()),
                    "selected_depth_3_count": int((selected_depths == 3).sum()),
                    "selected_depth_4_count": int((selected_depths == 4).sum()),
                    "selected_mean_mutation_count": float(selected_depths.mean()),
                    "selected_min_mutation_count": int(selected_depths.min()),
                    "selected_max_mutation_count": int(selected_depths.max()),
                    "new_complete_variant_count_vs_pre_round_visible": len(new_ids_vs_visible),
                    "new_mutated_position_count_vs_pre_round_visible": len(new_positions_vs_visible),
                    "new_position_residue_pair_count_vs_pre_round_visible": len(new_pairs_vs_visible),
                    "new_mutated_position_count_vs_campaign_selected": len(new_positions_vs_campaign),
                    "new_position_residue_pair_count_vs_campaign_selected": len(new_pairs_vs_campaign),
                    "hamming_one_from_previous_round_count": hamming_one_count,
                    "strict_add_one_from_previous_round_count": strict_add_one_count,
                    "candidate_pool_receipt_path": _repo_relative(
                        round_dir / "candidate_pool_receipt.json"
                    ),
                    "scientist_conversation_path": (
                        _repo_relative(conversation_path) if conversation_path else ""
                    ),
                }
            )

            selected_order = {
                str(item.variant_id): int(item.selection_order)
                for item in selected.itertuples(index=False)
            }
            for item in pool.itertuples(index=False):
                variant = str(item.variant)
                matched_positions = sum(
                    variant[index] in preferred.get(position, [])
                    for index, position in enumerate(MUTABLE_POSITIONS)
                )
                pool_rows.append(
                    {
                        "condition": run.condition,
                        "fold": run.fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "pool_order": int(item.pool_order),
                        "variant_id": str(item.variant_id),
                        "variant": variant,
                        "mutation_notation": _mutation_notation(variant),
                        "mutation_count": int(item.mutation_count),
                        "mutated_positions_json": _json(sorted(_variant_positions(variant))),
                        "preferred_position_matches": matched_positions if preferred else np.nan,
                        "matches_all_scientist_preferences": (
                            matched_positions == len(MUTABLE_POSITIONS) if preferred else False
                        ),
                        "selected": str(item.variant_id) in selected_ids,
                        "selection_order": selected_order.get(str(item.variant_id), np.nan),
                    }
                )

            for index, position in enumerate(MUTABLE_POSITIONS):
                pool_residues = sorted({variant[index] for variant in pool_variants})
                selected_residues = sorted({variant[index] for variant in selected_variants})
                position_rows.append(
                    {
                        "condition": run.condition,
                        "fold": run.fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "position": position,
                        "wild_type_residue": WILD_TYPE_SITES[index],
                        "scientist_preferred_residues_json": _json(preferred.get(position, [])),
                        "candidate_pool_residues_json": _json(pool_residues),
                        "selected_batch_residues_json": _json(selected_residues),
                        "candidate_pool_mutant_residues_json": _json(
                            [value for value in pool_residues if value != WILD_TYPE_SITES[index]]
                        ),
                        "selected_batch_mutant_residues_json": _json(
                            [value for value in selected_residues if value != WILD_TYPE_SITES[index]]
                        ),
                        "preferred_residues_present_in_pool_json": _json(
                            sorted(set(preferred.get(position, [])) & set(pool_residues))
                        ),
                        "preferred_residues_present_in_selected_json": _json(
                            sorted(set(preferred.get(position, [])) & set(selected_residues))
                        ),
                    }
                )

            for item in selected.itertuples(index=False):
                child = str(item.variant)
                distances = (
                    [_hamming(child, parent) for parent in previous_round_variants]
                    if previous_round_variants
                    else []
                )
                lineage_rows.append(
                    {
                        "condition": run.condition,
                        "fold": run.fold,
                        "round_id": round_id,
                        "run_id": run.run_id,
                        "selection_order": int(item.selection_order),
                        "variant_id": str(item.variant_id),
                        "variant": child,
                        "mutation_notation": str(item.mutation_notation),
                        "mutation_count": int(item.mutation_count),
                        "mutated_positions_json": _json(sorted(_variant_positions(child))),
                        "new_complete_variant_vs_pre_round_visible": (
                            str(item.variant_id) in new_ids_vs_visible
                        ),
                        "min_hamming_to_previous_round": min(distances) if distances else np.nan,
                        "hamming_one_from_any_previous_round_variant": (
                            any(value == 1 for value in distances) if distances else False
                        ),
                        "strict_add_one_from_any_previous_round_variant": (
                            any(
                                _is_strict_add_one_child(child, parent)
                                for parent in previous_round_variants
                            )
                            if previous_round_variants
                            else False
                        ),
                    }
                )

            historical_selected_ids.update(selected_ids)
            historical_selected_positions.update(selected_positions)
            historical_selected_pairs.update(selected_pairs)
            previous_round_variants = selected_variants

    round_frame = pd.DataFrame(round_rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)
    position_frame = pd.DataFrame(position_rows).sort_values(
        ["condition", "fold", "round_id", "position"]
    ).reset_index(drop=True)
    pool_frame = pd.DataFrame(pool_rows).sort_values(
        ["condition", "fold", "round_id", "pool_order"]
    ).reset_index(drop=True)
    lineage_frame = pd.DataFrame(lineage_rows).sort_values(
        ["condition", "fold", "round_id", "selection_order"]
    ).reset_index(drop=True)
    return round_frame, position_frame, pool_frame, lineage_frame


def aggregate_round_behavior(round_frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "catalog_remaining",
        "candidate_pool_size",
        "selected_batch_size",
        "candidate_pool_unique_mutated_position_count",
        "selected_unique_mutated_position_count",
        "scientist_preferred_position_count",
        "candidate_depth_1_count",
        "candidate_depth_2_count",
        "candidate_depth_3_count",
        "candidate_depth_4_count",
        "selected_depth_1_count",
        "selected_depth_2_count",
        "selected_depth_3_count",
        "selected_depth_4_count",
        "selected_mean_mutation_count",
        "new_complete_variant_count_vs_pre_round_visible",
        "new_mutated_position_count_vs_pre_round_visible",
        "new_position_residue_pair_count_vs_pre_round_visible",
        "new_mutated_position_count_vs_campaign_selected",
        "new_position_residue_pair_count_vs_campaign_selected",
        "hamming_one_from_previous_round_count",
        "strict_add_one_from_previous_round_count",
    ]
    rows = []
    for (condition, round_id), group in round_frame.groupby(["condition", "round_id"]):
        for metric in numeric:
            values = group[metric].astype(float)
            rows.append(
                {
                    "condition": condition,
                    "round_id": int(round_id),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_folds": len(values),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition", "round_id", "metric"]
    ).reset_index(drop=True)


def widen_kg_position_sets(position_frame: pd.DataFrame) -> pd.DataFrame:
    """Return one auditable row per KG condition/fold/round."""

    kg = position_frame[position_frame["condition"].isin(KG_CONDITIONS)]
    rows: list[dict[str, Any]] = []
    for (condition, fold, round_id, run_id), group in kg.groupby(
        ["condition", "fold", "round_id", "run_id"], sort=True
    ):
        record: dict[str, Any] = {
            "condition": condition,
            "fold": int(fold),
            "round_id": int(round_id),
            "run_id": run_id,
        }
        for item in group.itertuples(index=False):
            position = int(item.position)
            record[f"scientist_preferred_{position}"] = item.scientist_preferred_residues_json
            record[f"candidate_pool_residues_{position}"] = item.candidate_pool_residues_json
            record[f"selected_batch_residues_{position}"] = item.selected_batch_residues_json
            record[f"candidate_pool_mutant_residues_{position}"] = (
                item.candidate_pool_mutant_residues_json
            )
            record[f"selected_batch_mutant_residues_{position}"] = (
                item.selected_batch_mutant_residues_json
            )
        rows.append(record)
    return pd.DataFrame(rows).sort_values(
        ["condition", "fold", "round_id"]
    ).reset_index(drop=True)


def build_mutation_behavior_report(
    round_frame: pd.DataFrame,
    position_frame: pd.DataFrame,
    lineage_frame: pd.DataFrame,
) -> str:
    complete = round_frame.groupby(["condition", "round_id"]).agg(
        batch=("selected_batch_size", "mean"),
        pool=("candidate_pool_size", "mean"),
        mean_depth=("selected_mean_mutation_count", "mean"),
        depth_sd=("selected_mean_mutation_count", "std"),
        depth2=("selected_depth_2_count", "sum"),
        depth3=("selected_depth_3_count", "sum"),
        depth4=("selected_depth_4_count", "sum"),
        new_pairs=("new_position_residue_pair_count_vs_campaign_selected", "mean"),
        add_one=("strict_add_one_from_previous_round_count", "mean"),
    ).reset_index()
    all_batch_fixed = bool((round_frame["selected_batch_size"] == 16).all())
    all_pool_fixed = bool((round_frame["candidate_pool_size"] == 32).all())
    all_new_variants = bool(
        (round_frame["new_complete_variant_count_vs_pre_round_visible"] == 16).all()
    )
    no_new_coordinates = bool(
        (round_frame["new_mutated_position_count_vs_pre_round_visible"] == 0).all()
    )
    no_new_pairs = bool(
        (round_frame["new_position_residue_pair_count_vs_pre_round_visible"] == 0).all()
    )
    kg = round_frame[round_frame["condition"].isin(KG_CONDITIONS)]
    all_kg_preferences = bool(kg["scientist_covers_all_configured_positions"].all())
    selected_cover_all_count = int(
        (round_frame["selected_unique_mutated_position_count"] == 4).sum()
    )
    pool_cover_all_count = int(
        (round_frame["candidate_pool_unique_mutated_position_count"] == 4).sum()
    )
    invariant_checks = {
        "candidate_pool_size_is_32": all_pool_fixed,
        "selected_batch_size_is_16": all_batch_fixed,
        "all_selected_variants_are_new_ids": all_new_variants,
        "no_new_mutable_coordinates_vs_al96": no_new_coordinates,
        "no_new_position_residue_pairs_vs_al96": no_new_pairs,
        "scientist_preferences_cover_all_four_positions": all_kg_preferences,
    }
    failed_invariants = [name for name, passed in invariant_checks.items() if not passed]
    if failed_invariants:
        raise ValueError(f"Mutation-behavior invariants failed: {failed_invariants}")
    lines = [
        "# GB1 闭池突变推荐行为审计",
        "",
        "## 结论",
        "",
        "当前三轮实验不是亲本—子代式逐轮加点进化。系统始终在预先给定的 GB1 四位点组合库中选择新的完整变体；上一轮 wet 结果会更新 Scientist/KG/模型，但不会把上一轮变体指定为下一轮亲本。",
        "",
        "所有正式 run 的每轮候选池均固定为 32 个完整变体，每轮实际入选批次均固定为 16 个完整变体，且这 16 个 variant IDs 均未在 pre-round visible 集合中出现。AL96 初始集已覆盖四个坐标和全部 76 个非野生型 position–residue 对，因此相对 pre-round visible 数据，每轮新增突变坐标和新增 position–residue 对均为 0。",
        "",
        f"三条 KG 路线的 27 个 fold–round Scientist 输出均为 39/40/41/54 四个位点给出偏好。45 个正式 condition–fold–round 中，32-candidate pool 有 {pool_cover_all_count}/45 在批次并集上出现四个位点的非野生型残基，最终 16 个入选批次有 {selected_cover_all_count}/45；例外集中在 `kg_base_al` 第一轮对野生型 G41 的保留。这里的‘Scientist 为某个位点给出偏好’不等于要求突变该位点：例如 `41=[G]` 的含义是偏好保留 G41。",
        "",
        "## 每轮实际选择的突变深度",
        "",
        "| 条件 | Round | Pool | Batch | 单变体平均突变数 | depth 1/2/3/4（3 folds 合计） | 新 position–residue 对（仅相对此前 campaign 批次） | 严格由上一轮变体加一个突变（均值/16） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in complete.itertuples(index=False):
        lines.append(
            f"| `{row.condition}` | {int(row.round_id)} | {int(row.pool)} | {int(row.batch)} | "
            f"{row.mean_depth:.3f} ± {row.depth_sd:.3f} | "
            f"0/{int(row.depth2)}/{int(row.depth3)}/{int(row.depth4)} | "
            f"{row.new_pairs:.1f} | {row.add_one:.1f}/16 |"
        )
    lines.extend(
        [
            "",
            "`mutation_count` 是完整四位点序列相对 `VDGV` 的 Hamming 距离。AL96 初始集已经包含 WT、76 个单点和 19 个双点，因此正式候选批次中没有新的单点类别；系统可以在同一批中同时选择双点、三点和四点变体。表中的后轮‘新 position–residue 对’减少只表示早期批次已经覆盖了较多残基选择，不是程序逐轮减少可突变坐标。",
            "",
            "## 三层对象",
            "",
            "1. `Scientist preferred_residues`：39/40/41/54 上的软残基集合；它们用于计算每个完整变体命中几个偏好位点，不是新坐标提案，也不是硬约束。",
            "2. `candidate_pool`：从当前尚未揭示的约 119k 完整变体中，先按硬约束，再按偏好命中数、可选择 KG evidence 和确定性 tie-break 截取 32 个；每个对象都是一个完整四字符组合。",
            "3. `selected_batch`：Agent-UQ、predictor 或 random 在 32 个候选内选出 16 个并提交 oracle；本轮结束后只删除这 16 个完整 variant IDs，其他组合仍可在后续轮次被考虑。",
            "",
            "即使某个后轮候选与上一轮候选相差一个位点，工件中也没有 parent ID 或 lineage edge，且严格‘保留上一轮全部突变并新增一个 WT→mutant 编辑’的变体只占部分入选。这些相似关系是组合库中偶然存在的邻接，而不是系统的生成规则。",
            "",
            "## 证据路径",
            "",
            "- `round_behavior_by_fold.csv`：每个 condition/fold/round 的批量、突变深度、新坐标/残基对和邻接统计。",
            "- `position_sets_by_fold_round.csv`：Scientist 偏好、32-candidate pool 与 16-selected batch 在每个位点上的 residue sets。",
            "- `candidate_pool_variants.csv`：每轮 32 个完整候选及是否入选。",
            "- `selected_variant_lineage_audit.csv`：每个入选完整变体与上一轮批次的 Hamming/严格加点关系。",
            "",
            f"审计包含 {len(round_frame)} 个 condition–fold–round、{len(position_frame)} 个位置集合记录和 {len(lineage_frame)} 个入选变体记录。",
        ]
    )
    return "\n".join(lines) + "\n"
