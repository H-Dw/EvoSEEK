from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


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
    baseline_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row["mode"] != "fitness_direct":
            continue
        key = (
            row["seed"],
            row["queries_used"],
            row["split_strategy"],
            row["fold_index"],
            row["assignment_sha256"],
        )
        baseline_by_key[key] = row
    for row in rows:
        key = (
            row["seed"],
            row["queries_used"],
            row["split_strategy"],
            row["fold_index"],
            row["assignment_sha256"],
        )
        baseline = baseline_by_key.get(key)
        row["same_fold_baseline_available"] = baseline is not None
        row["same_fold_baseline_run_id"] = baseline["run_id"] if baseline else None
        row["delta_best_seen_vs_fitness_direct"] = (
            row["best_seen_fitness"] - baseline["best_seen_fitness"] if baseline else None
        )
        row["delta_last_batch_mean_vs_fitness_direct"] = (
            row["last_batch_mean_fitness"] - baseline["last_batch_mean_fitness"]
            if baseline
            else None
        )
    frame = pd.DataFrame(rows)
    csv_path = target / "run_comparison.csv"
    json_path = target / "run_comparison.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}


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
