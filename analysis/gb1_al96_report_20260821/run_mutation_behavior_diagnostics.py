"""Run the reproducible GB1 closed-pool mutation-behavior audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import MUTATION_BEHAVIOR_DIR, MUTATION_BEHAVIOR_SOURCE_DIR
from io_artifacts import discover_runs, sha256_file, validate_analysis_matrix
from mutation_behavior_diagnostics import (
    aggregate_round_behavior,
    build_mutation_behavior_report,
    build_mutation_behavior_tables,
    widen_kg_position_sets,
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


def main() -> int:
    MUTATION_BEHAVIOR_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    validate_analysis_matrix(runs)
    round_frame, position_frame, pool_frame, lineage_frame = (
        build_mutation_behavior_tables(runs)
    )
    aggregate_frame = aggregate_round_behavior(round_frame)
    wide_position_frame = widen_kg_position_sets(position_frame)

    outputs = {
        "round_behavior_by_fold.csv": round_frame,
        "round_behavior_mean_sd.csv": aggregate_frame,
        "position_sets_by_fold_round.csv": position_frame,
        "kg_position_sets_wide.csv": wide_position_frame,
        "candidate_pool_variants.csv": pool_frame,
        "selected_variant_lineage_audit.csv": lineage_frame,
    }
    for name, frame in outputs.items():
        _write_csv(frame, MUTATION_BEHAVIOR_SOURCE_DIR / name)

    report_path = MUTATION_BEHAVIOR_DIR / "mutation_behavior_analysis.md"
    report_path.write_text(
        build_mutation_behavior_report(round_frame, position_frame, lineage_frame),
        encoding="utf-8",
    )
    declared = [
        report_path,
        *(MUTATION_BEHAVIOR_SOURCE_DIR / name for name in outputs),
    ]
    manifest = {
        "analysis_id": "mutation_behavior_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conditions": sorted(round_frame["condition"].unique().tolist()),
        "folds": int(round_frame["fold"].nunique()),
        "rounds": int(round_frame["round_id"].nunique()),
        "round_rows": len(round_frame),
        "position_rows": len(position_frame),
        "candidate_pool_rows": len(pool_frame),
        "selected_lineage_rows": len(lineage_frame),
        "outputs": [
            {
                "path": path.relative_to(MUTATION_BEHAVIOR_DIR).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(declared)
        ],
    }
    _write_json(manifest, MUTATION_BEHAVIOR_DIR / "mutation_behavior_manifest.json")
    print(
        "Mutation behavior diagnostics complete: "
        f"{len(round_frame)} rounds, {len(pool_frame)} pool variants, "
        f"{len(lineage_frame)} selected variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
