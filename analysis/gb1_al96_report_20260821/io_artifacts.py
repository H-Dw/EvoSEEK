"""Read-only discovery and validation of campaign artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from config import (
    ARTIFACT_ROOTS,
    CONDITION_ORDER,
    EXPECTED_BATCH_SIZES,
    EXPECTED_FOLDS,
    EXPECTED_ROUNDS,
    HIERARCHICAL_ARTIFACT_ROOT,
    SUPERSEDED_FAILED_CONDITION,
)


@dataclass(frozen=True)
class RunArtifact:
    """A discovered run plus the metadata needed by downstream modules."""

    path: Path
    run_id: str
    condition: str
    fold: int
    seed: int | None
    assignment_sha256: str | None
    summary: dict[str, Any]
    manifest: dict[str, Any]
    eligible: bool
    exclusion_reason: str | None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _infer_fold(run_dir: Path, summary: dict[str, Any]) -> int:
    data_source = summary.get("data_source") or {}
    if data_source.get("fold_index") is not None:
        return int(data_source["fold_index"])
    match = re.search(r"(?:^|-)f(\d{2})(?:-|$)", run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer fold from {run_dir}")
    return int(match.group(1))


def _validate_run(
    artifact_root: Path,
    run_dir: Path,
    condition: str,
    fold: int,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> str | None:
    if condition not in CONDITION_ORDER:
        return "condition_not_in_analysis_scope"
    if (
        artifact_root == HIERARCHICAL_ARTIFACT_ROOT
        and condition == SUPERSEDED_FAILED_CONDITION
    ):
        return "superseded_failed_run"
    if fold not in EXPECTED_FOLDS:
        return "unexpected_fold"
    if not bool(manifest.get("pass_eligible")):
        return "manifest_not_pass_eligible"
    if not bool(summary.get("pass_eligible")):
        return "summary_not_pass_eligible"
    if int(manifest.get("completed_rounds", -1)) != len(EXPECTED_ROUNDS):
        return "incomplete_rounds"
    if int(manifest.get("aborted_rounds", -1)) != 0:
        return "aborted_rounds_present"
    planned = tuple(int(value) for value in manifest.get("planned_batch_sizes", []))
    actual = tuple(int(value) for value in manifest.get("actual_batch_sizes", []))
    if planned != EXPECTED_BATCH_SIZES or actual != EXPECTED_BATCH_SIZES:
        return "batch_size_mismatch"
    required = (
        "state.json",
        "summary.json",
        "round_metrics.csv",
        "completion_manifest.json",
        "top_k_all_rounds.csv",
    )
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        return "missing_required_files:" + ",".join(missing)
    return None


def discover_runs(artifact_roots: Iterable[Path] = ARTIFACT_ROOTS) -> list[RunArtifact]:
    records: list[RunArtifact] = []
    for root in artifact_roots:
        runs_dir = root / "runs"
        if not runs_dir.exists():
            raise FileNotFoundError(f"Missing runs directory: {runs_dir}")
        for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
            summary_path = run_dir / "summary.json"
            manifest_path = run_dir / "completion_manifest.json"
            if not summary_path.exists() or not manifest_path.exists():
                continue
            summary = read_json(summary_path)
            manifest = read_json(manifest_path)
            condition = str(summary.get("condition") or summary.get("mode") or "unknown")
            fold = _infer_fold(run_dir, summary)
            data_source = summary.get("data_source") or {}
            exclusion_reason = _validate_run(
                root, run_dir, condition, fold, summary, manifest
            )
            records.append(
                RunArtifact(
                    path=run_dir,
                    run_id=str(summary.get("run_id") or run_dir.name),
                    condition=condition,
                    fold=fold,
                    seed=(int(summary["seed"]) if summary.get("seed") is not None else None),
                    assignment_sha256=data_source.get("assignment_sha256"),
                    summary=summary,
                    manifest=manifest,
                    eligible=exclusion_reason is None,
                    exclusion_reason=exclusion_reason,
                )
            )
    return records


def validate_analysis_matrix(runs: list[RunArtifact]) -> None:
    eligible = [run for run in runs if run.eligible]
    observed = {(run.condition, run.fold) for run in eligible}
    expected = {
        (condition, fold) for condition in CONDITION_ORDER for fold in EXPECTED_FOLDS
    }
    missing = sorted(expected - observed)
    duplicates = sorted(
        key
        for key in observed
        if sum(
            run.condition == key[0] and run.fold == key[1] and run.eligible
            for run in runs
        )
        != 1
    )
    if missing or duplicates:
        raise ValueError(
            f"Invalid analysis matrix; missing={missing}, duplicates={duplicates}"
        )


def run_status_frame(runs: list[RunArtifact]) -> pd.DataFrame:
    rows = []
    for run in runs:
        rows.append(
            {
                "condition": run.condition,
                "fold": run.fold,
                "seed": run.seed,
                "assignment_sha256": run.assignment_sha256,
                "run_id": run.run_id,
                "eligible": run.eligible,
                "exclusion_reason": run.exclusion_reason,
                "experiment_status": run.manifest.get("experiment_status"),
                "pass_eligible": run.manifest.get("pass_eligible"),
                "completed_rounds": run.manifest.get("completed_rounds"),
                "aborted_rounds": run.manifest.get("aborted_rounds"),
                "planned_batch_sizes": json.dumps(
                    run.manifest.get("planned_batch_sizes", []), separators=(",", ":")
                ),
                "actual_batch_sizes": json.dumps(
                    run.manifest.get("actual_batch_sizes", []), separators=(",", ":")
                ),
                "run_path": str(run.path.relative_to(run.path.parents[3])),
            }
        )
    return pd.DataFrame(rows).sort_values(["condition", "fold"]).reset_index(drop=True)


def input_fingerprints(runs: list[RunArtifact]) -> list[dict[str, str]]:
    records = []
    for run in runs:
        for name in (
            "completion_manifest.json",
            "summary.json",
            "round_metrics.csv",
            "state.json",
            "top_k_all_rounds.csv",
        ):
            path = run.path / name
            if path.exists():
                records.append(
                    {
                        "run_id": run.run_id,
                        "file": name,
                        "sha256": sha256_file(path),
                    }
                )
    return records


def query_structured_kg(
    db_path: Path, variant_id: str, limit: int = 12
) -> dict[str, list[dict[str, Any]]]:
    """Return a compact, read-only subgraph whose records mention a variant ID."""

    uri = f"file:{db_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        connection.row_factory = sqlite3.Row
        like = f"%{variant_id}%"
        entity_rows = connection.execute(
            """
            SELECT entity_id, entity_type, layer, modalities_json, properties_json,
                   source_ids_json, source_group, confidence,
                   valid_from_round, valid_to_round
            FROM entities
            WHERE entity_id LIKE ? OR properties_json LIKE ? OR source_ids_json LIKE ?
            LIMIT ?
            """,
            (like, like, like, limit),
        ).fetchall()
        entity_ids = [row["entity_id"] for row in entity_rows]
        relation_rows: list[sqlite3.Row] = []
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            relation_rows = connection.execute(
                f"""
                SELECT relation_id, subject_id, predicate, object_id, layer,
                       modalities_json, properties_json, source_ids_json,
                       evidence_ids_json, source_group, confidence, context_id,
                       valid_from_round, valid_to_round
                FROM relations
                WHERE subject_id IN ({placeholders}) OR object_id IN ({placeholders})
                   OR properties_json LIKE ? OR evidence_ids_json LIKE ?
                LIMIT ?
                """,
                (*entity_ids, *entity_ids, like, like, limit),
            ).fetchall()
        else:
            relation_rows = connection.execute(
                """
                SELECT relation_id, subject_id, predicate, object_id, layer,
                       modalities_json, properties_json, source_ids_json,
                       evidence_ids_json, source_group, confidence, context_id,
                       valid_from_round, valid_to_round
                FROM relations
                WHERE properties_json LIKE ? OR evidence_ids_json LIKE ?
                LIMIT ?
                """,
                (like, like, limit),
            ).fetchall()
    return {
        "entities": [dict(row) for row in entity_rows],
        "relations": [dict(row) for row in relation_rows],
    }
