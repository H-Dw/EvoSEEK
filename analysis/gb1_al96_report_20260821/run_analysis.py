"""Single reproducible entry point for the GB1 AL96 manuscript analysis."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from cases import (
    build_candidate_table,
    build_case_studies,
    build_condition_shortlist,
    cases_to_markdown,
    shortlist_to_markdown,
)
from config import (
    ARTIFACT_ROOTS,
    CASE_STUDY_DIR,
    FIGURE_DIR,
    OUTPUT_ROOT,
    REPO_ROOT,
    SOURCE_DATA_DIR,
    TABLE_DIR,
)
from io_artifacts import (
    discover_runs,
    input_fingerprints,
    run_status_frame,
    sha256_file,
    validate_analysis_matrix,
)
from metrics import (
    aggregate_candidate_pool_overlap,
    aggregate_feature_rag_interaction_deltas,
    aggregate_final_metrics,
    aggregate_fold_deltas,
    aggregate_round_metrics,
    build_active_learning_audit,
    build_candidate_pool_overlap,
    build_feature_channel_audit,
    build_feature_rag_interaction_deltas,
    build_final_metrics,
    build_fold_deltas,
    build_new_condition_runtime_audit,
    build_round_metrics,
    compact_metric_summary,
)
from plots import generate_all_figures
from tables import build_ablation_tables, build_performance_tables, build_status_table


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", float_format="%.12g")


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _output_hashes(output_root: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(item for item in output_root.rglob("*") if item.is_file()):
        if path.name == "analysis_summary.json":
            continue
        records.append(
            {
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    return records


def main() -> int:
    for directory in (OUTPUT_ROOT, SOURCE_DATA_DIR, CASE_STUDY_DIR, TABLE_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    runs = discover_runs()
    validate_analysis_matrix(runs)
    status = run_status_frame(runs)
    round_metrics = build_round_metrics(runs)
    final_metrics = build_final_metrics(runs, round_metrics)
    aggregate = aggregate_final_metrics(final_metrics)
    round_summary = aggregate_round_metrics(round_metrics)
    fold_deltas = build_fold_deltas(final_metrics)
    fold_delta_summary = aggregate_fold_deltas(fold_deltas)
    pool_overlap = build_candidate_pool_overlap(runs)
    pool_overlap_summary = aggregate_candidate_pool_overlap(pool_overlap)
    active_learning = build_active_learning_audit(runs)
    feature_channels = build_feature_channel_audit(runs)
    new_condition_runtime = build_new_condition_runtime_audit(runs)
    feature_rag_interaction = build_feature_rag_interaction_deltas(final_metrics)
    feature_rag_interaction_summary = aggregate_feature_rag_interaction_deltas(
        feature_rag_interaction
    )
    candidates = build_candidate_table(runs)
    cases, case_audit = build_case_studies(runs, candidates)
    condition_shortlist = build_condition_shortlist(case_audit)

    csv_outputs = {
        "run_status.csv": status,
        "round_metrics_by_fold.csv": round_metrics,
        "round_metrics_mean_sd.csv": round_summary,
        "final_metrics_by_fold.csv": final_metrics,
        "final_metrics_mean_sd.csv": aggregate,
        "kg_module_fold_deltas.csv": fold_deltas,
        "kg_module_fold_deltas_mean_sd.csv": fold_delta_summary,
        "candidate_pool_overlap.csv": pool_overlap,
        "candidate_pool_overlap_summary.csv": pool_overlap_summary,
        "active_learning_audit.csv": active_learning,
        "feature_channel_audit.csv": feature_channels,
        "new_condition_runtime_audit.csv": new_condition_runtime,
        "feature_rag_interaction_deltas.csv": feature_rag_interaction,
        "feature_rag_interaction_deltas_mean_sd.csv": feature_rag_interaction_summary,
        "selected_candidates.csv": candidates,
    }
    for name, frame in csv_outputs.items():
        _write_csv(frame, SOURCE_DATA_DIR / name)
    _write_csv(case_audit, CASE_STUDY_DIR / "case_selection_audit.csv")
    _write_csv(
        condition_shortlist, CASE_STUDY_DIR / "condition_case_shortlist.csv"
    )
    (CASE_STUDY_DIR / "condition_case_shortlist.md").write_text(
        shortlist_to_markdown(condition_shortlist), encoding="utf-8"
    )
    _write_json(cases, CASE_STUDY_DIR / "selected_cases.json")
    (CASE_STUDY_DIR / "selected_cases.md").write_text(
        cases_to_markdown(cases), encoding="utf-8"
    )
    (TABLE_DIR / "performance_tables.md").write_text(
        build_performance_tables(aggregate), encoding="utf-8"
    )
    (TABLE_DIR / "completion_matrix.md").write_text(
        build_status_table(status), encoding="utf-8"
    )
    (TABLE_DIR / "ablation_tables.md").write_text(
        build_ablation_tables(
            fold_delta_summary,
            feature_rag_interaction_summary,
            feature_channels,
            new_condition_runtime,
        ),
        encoding="utf-8",
    )
    figure_paths = generate_all_figures(SOURCE_DATA_DIR, FIGURE_DIR)

    summary = {
        "analysis_id": "gb1_al96_report_20260821",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "protocol": {
            "folds": 3,
            "rounds": 3,
            "batch_size": 16,
            "initial_observations": 96,
            "final_observations": 144,
            "sample_sd_ddof": 1,
            "inferential_tests": False,
        },
        "eligible_runs": int(status["eligible"].sum()),
        "excluded_runs": int((~status["eligible"]).sum()),
        "input_roots": [
            path.relative_to(REPO_ROOT).as_posix() for path in ARTIFACT_ROOTS
        ],
        "run_status": [
            {
                "condition": run.condition,
                "fold": run.fold,
                "run_id": run.run_id,
                "eligible": run.eligible,
                "exclusion_reason": run.exclusion_reason,
            }
            for run in runs
        ],
        "input_fingerprints": input_fingerprints(runs),
        "metrics": compact_metric_summary(aggregate),
        "selected_cases": [
            {
                "case_id": case["case_id"],
                "condition": case["condition"],
                "fold": case["fold"],
                "round_id": case["round_id"],
                "variant_id": case["variant_id"],
            }
            for case in cases
        ],
        "figure_paths": [path.relative_to(OUTPUT_ROOT).as_posix() for path in figure_paths],
        "output_hashes": _output_hashes(OUTPUT_ROOT),
    }
    _write_json(summary, OUTPUT_ROOT / "analysis_summary.json")
    print(
        f"Analysis complete: {summary['eligible_runs']} eligible runs; "
        f"outputs written to {OUTPUT_ROOT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
