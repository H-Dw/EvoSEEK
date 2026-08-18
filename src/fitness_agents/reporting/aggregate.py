from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REFERENCE_CONDITIONS = ("fitness_direct", "llm_agent", "knowledge_agent")
RAG_CONDITION = "knowledge_agent_rag"
NO_RAG_CONDITION = "knowledge_agent"


def infer_condition(summary: dict[str, Any], *, job_mode: str | None = None) -> str:
    if summary.get("condition"):
        return str(summary["condition"])
    if job_mode:
        return str(job_mode)
    blob = f"{summary.get('run_id', '')} {summary.get('run_label', '')}".casefold()
    if "knowledge_agent_rag" in blob or "al96-5cv-rag" in blob:
        return RAG_CONDITION
    return str(summary.get("mode") or "unknown")


def _pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["seed"],
        row["queries_used"],
        row["split_strategy"],
        row["fold_index"],
        row["assignment_sha256"],
    )


def _index_by_condition(
    rows: Sequence[dict[str, Any]], condition: str
) -> dict[tuple[Any, ...], dict[str, Any]]:
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["condition"] != condition:
            continue
        index[_pair_key(row)] = row
    return index


def _attach_reference_deltas(rows: list[dict[str, Any]]) -> None:
    references = {
        name: _index_by_condition(rows, name) for name in REFERENCE_CONDITIONS
    }
    for row in rows:
        key = _pair_key(row)
        for name, index in references.items():
            reference = index.get(key)
            row[f"same_fold_{name}_available"] = reference is not None
            row[f"same_fold_{name}_run_id"] = (
                reference["run_id"] if reference else None
            )
            row[f"delta_best_seen_vs_{name}"] = (
                row["best_seen_fitness"] - reference["best_seen_fitness"]
                if reference
                else None
            )
            row[f"delta_last_batch_mean_vs_{name}"] = (
                row["last_batch_mean_fitness"] - reference["last_batch_mean_fitness"]
                if reference
                else None
            )
        # Keep the historical fitness_direct aliases used by existing reports.
        row["same_fold_baseline_available"] = row["same_fold_fitness_direct_available"]
        row["same_fold_baseline_run_id"] = row["same_fold_fitness_direct_run_id"]
        row["delta_best_seen_vs_fitness_direct"] = row[
            "delta_best_seen_vs_fitness_direct"
        ]
        row["delta_last_batch_mean_vs_fitness_direct"] = row[
            "delta_last_batch_mean_vs_fitness_direct"
        ]


def write_paired_contrast(
    rows: Sequence[dict[str, Any]],
    output_dir: str | Path,
    *,
    left: str = NO_RAG_CONDITION,
    right: str = RAG_CONDITION,
) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    left_index = _index_by_condition(rows, left)
    pairs: list[dict[str, Any]] = []
    for right_row in rows:
        if right_row["condition"] != right:
            continue
        left_row = left_index.get(_pair_key(right_row))
        if left_row is None:
            continue
        pairs.append(
            {
                "seed": right_row["seed"],
                "fold_index": right_row["fold_index"],
                "queries_used": right_row["queries_used"],
                "assignment_sha256": right_row["assignment_sha256"],
                "no_rag_run_id": left_row["run_id"],
                "rag_run_id": right_row["run_id"],
                "no_rag_best_seen": left_row["best_seen_fitness"],
                "rag_best_seen": right_row["best_seen_fitness"],
                "delta_best_seen_rag_minus_no_rag": (
                    right_row["best_seen_fitness"] - left_row["best_seen_fitness"]
                ),
                "no_rag_last_batch_mean": left_row["last_batch_mean_fitness"],
                "rag_last_batch_mean": right_row["last_batch_mean_fitness"],
                "delta_last_batch_mean_rag_minus_no_rag": (
                    right_row["last_batch_mean_fitness"]
                    - left_row["last_batch_mean_fitness"]
                ),
                "no_rag_allow_remote_context": left_row.get("allow_remote_context"),
                "rag_allow_remote_context": right_row.get("allow_remote_context"),
            }
        )
    json_path = target / "rag_contrast.json"
    md_path = target / "rag_contrast.md"
    json_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    lines = [
        "# RAG vs no-RAG paired contrast",
        "",
        f"No-RAG condition: `{left}`. RAG condition: `{right}`.",
        "Positive delta means RAG found a higher fitness on the same fold/seed.",
        "",
        "| Fold | Seed | No-RAG best | RAG best | Δ best | No-RAG batch mean | RAG batch mean | Δ mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in pairs:
        lines.append(
            "| {fold_index} | {seed} | {no_rag_best_seen:.4f} | {rag_best_seen:.4f} | "
            "{delta_best_seen_rag_minus_no_rag:+.4f} | {no_rag_last_batch_mean:.4f} | "
            "{rag_last_batch_mean:.4f} | {delta_last_batch_mean_rag_minus_no_rag:+.4f} |".format(
                **item
            )
        )
    if not pairs:
        lines.extend(["", "No paired fold/seed rows were found for both conditions."])
    else:
        mean_best = sum(item["delta_best_seen_rag_minus_no_rag"] for item in pairs) / len(
            pairs
        )
        mean_batch = sum(
            item["delta_last_batch_mean_rag_minus_no_rag"] for item in pairs
        ) / len(pairs)
        lines.extend(
            [
                "",
                f"Paired folds: {len(pairs)}. Mean Δ best seen: {mean_best:+.4f}. "
                f"Mean Δ last-batch mean: {mean_batch:+.4f}.",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"rag_json": json_path, "rag_markdown": md_path}


def aggregate_runs(summaries: Sequence[dict[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in summaries:
        round_metrics = summary.get("round_metrics") or []
        if not round_metrics:
            continue
        last_round = round_metrics[-1]
        data_source = summary.get("data_source", {})
        final_metrics = summary.get("final_prediction_metrics") or {}
        rows.append(
            {
                "run_id": summary["run_id"],
                "mode": summary["mode"],
                "condition": infer_condition(summary),
                "run_label": summary.get("run_label"),
                "local_knowledge_enabled": summary.get("local_knowledge_enabled"),
                "allow_remote_context": summary.get("allow_remote_context"),
                "seed": summary["seed"],
                "queries_used": summary["queries_used"],
                "split_strategy": data_source.get("strategy"),
                "protocol_version": data_source.get("protocol_version"),
                "fold_index": data_source.get("fold_index"),
                "manifest_sha256": data_source.get("manifest_sha256"),
                "assignment_sha256": data_source.get("assignment_sha256"),
                "best_seen_fitness": last_round["best_seen_fitness"],
                "last_batch_mean_fitness": last_round["batch_mean_fitness"],
                "mean_selected_model_rank_fraction": last_round[
                    "mean_selected_model_rank_fraction"
                ],
                **{f"final_{key}": value for key, value in final_metrics.items()},
            }
        )
    _attach_reference_deltas(rows)
    frame = pd.DataFrame(rows)
    csv_path = target / "run_comparison.csv"
    json_path = target / "run_comparison.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    paths = {"csv": csv_path, "json": json_path}
    conditions = {row["condition"] for row in rows}
    if NO_RAG_CONDITION in conditions and RAG_CONDITION in conditions:
        paths.update(write_paired_contrast(rows, target))
    return paths


def write_science_markdown(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Scientific-thinking intervention report",
        "",
        f"Verdict: `{report['verdict']}`",
        "",
        report["interpretation"],
        "",
        "## Behavioral metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value:.4f} |" for key, value in report["metrics"].items())
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- [{'x' if passed else ' '}] {name}" for name, passed in report["gates"].items())
    lines.extend(["", "## Run directories", ""])
    lines.extend(f"- `{name}`: `{path}`" for name, path in report["run_dirs"].items())
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
