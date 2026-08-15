from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fitness_agents.data.canonical import TARGET_PROXY_COLUMNS, CanonicalDataset

from .audit import assert_audit_passed, audit_split
from .contracts import SplitRequest, SplitResult
from .hashing import effective_salt, salt_commitment


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _csv_gzip_bytes(frame: pd.DataFrame) -> bytes:
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as archive:
        archive.write(csv_bytes)
    return buffer.getvalue()


def _write_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _csv_gzip_bytes(frame)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def _code_hash() -> str:
    roots = [Path(__file__).parent, Path(__file__).parents[1] / "adapters"]
    digest = hashlib.sha256()
    for path in sorted(file for root in roots for file in root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _public_features(frame: pd.DataFrame) -> pd.DataFrame:
    banned = TARGET_PROXY_COLUMNS | {"source_row_id", "source_sha256"}
    keep = [column for column in frame.columns if column.lower() not in banned]
    return frame.loc[:, keep].copy()


def _configuration_record(dataset: CanonicalDataset, request: SplitRequest) -> dict[str, Any]:
    salt = effective_salt(request.public_salt, request.seed)
    return {
        "dataset_id": dataset.spec.dataset_id,
        "assay_id": dataset.spec.assay_id,
        "dataset_scope": dataset.spec.dataset_scope,
        "strategy": request.strategy,
        "protocol_version": request.protocol_version,
        "n_folds": request.n_folds,
        "seed": request.seed,
        "salt_commitment": salt_commitment(salt),
        "options": request.options,
        "allow_label_dependent_membership": request.allow_label_dependent_membership,
    }


def _validate_existing_output(root: Path, manifest: dict[str, Any]) -> None:
    for fold_record in manifest.get("fold_manifests", []):
        fold_path = root / str(fold_record["path"])
        if not fold_path.is_file() or _sha256_bytes(fold_path.read_bytes()) != fold_record["sha256"]:
            raise ValueError(f"Existing fold manifest failed hash verification: {fold_path}")
        fold_manifest = json.loads(fold_path.read_text(encoding="utf-8"))
        for file_record in fold_manifest["files"]:
            output = fold_path.parent / str(file_record["path"])
            if not output.is_file() or _sha256_bytes(output.read_bytes()) != file_record["sha256"]:
                raise ValueError(f"Existing split output failed hash verification: {output}")


def write_split(
    dataset: CanonicalDataset,
    request: SplitRequest,
    result: SplitResult,
    output_root: str | Path,
) -> Path:
    audit = audit_split(dataset, result)
    assert_audit_passed(audit)
    root = (
        Path(output_root)
        / dataset.spec.dataset_id
        / request.strategy
        / request.protocol_version
    )
    config_record = _configuration_record(dataset, request)
    config_hash = _sha256_bytes(_json_bytes(config_record))
    current_code_hash = _code_hash()
    existing_manifest = root / "manifest.public.json"
    if root.exists() and any(root.iterdir()):
        if existing_manifest.exists():
            existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
            if (
                existing.get("config_sha256") == config_hash
                and existing.get("source_sha256") == dataset.source_sha256
                and existing.get("code_sha256") == current_code_hash
            ):
                _validate_existing_output(root, existing)
                return root
        raise FileExistsError(
            f"Output {root} already exists with a different manifest; use a new protocol version"
        )
    root.mkdir(parents=True, exist_ok=True)
    feature_by_id = dataset.features.set_index("variant_id", drop=False)
    label_by_id = dataset.labels.set_index("variant_id", drop=False)
    root_files: list[dict[str, Any]] = []
    for fold in result.folds:
        fold_root = root / f"fold_{fold.fold_index:02d}"
        assignments = fold.assignments.set_index("variant_id", drop=False)

        def ids_for(role: str, _assignments: pd.DataFrame = assignments) -> list[str]:
            return sorted(
                _assignments.loc[
                    _assignments["split_role"] == role, "variant_id"
                ].astype(str)
            )

        train_ids = ids_for("initial_observed") + ids_for("train_observed")
        validation_ids = ids_for("benchmark_validation")
        candidate_ids = ids_for("candidate_pool")
        final_ids = ids_for("final_test")
        quarantine_ids = ids_for("quarantine")

        def features_for(ids: list[str]) -> pd.DataFrame:
            if not ids:
                return _public_features(dataset.features.iloc[0:0])
            return _public_features(feature_by_id.loc[ids].reset_index(drop=True))

        def labeled_for(ids: list[str]) -> pd.DataFrame:
            features = features_for(ids)
            labels = label_by_id.loc[ids, ["target"]].reset_index(drop=True)
            return features.assign(target=labels["target"].to_numpy())

        outputs: list[tuple[str, pd.DataFrame]] = [
            ("agent/initial_or_train_observed.csv.gz", labeled_for(train_ids)),
            ("agent/candidate_pool.csv.gz", features_for(candidate_ids)),
            ("controller/benchmark_validation.csv.gz", labeled_for(validation_ids)),
            (
                "oracle/queryable_labels.csv.gz",
                label_by_id.loc[candidate_ids, ["variant_id", "target"]].reset_index(drop=True)
                if candidate_ids
                else dataset.labels.iloc[0:0].copy(),
            ),
            ("evaluator/final_test_inputs.csv.gz", features_for(final_ids)),
            (
                "evaluator/final_test_labels.csv.gz",
                label_by_id.loc[final_ids, ["variant_id", "target"]].reset_index(drop=True),
            ),
            ("quarantine/excluded_variants.csv.gz", features_for(quarantine_ids)),
        ]
        if request.strategy == "flip_static_ood":
            compat_ids = train_ids + validation_ids + final_ids
            compat = labeled_for(compat_ids).loc[:, ["sequence", "target", "variant_id"]]
            role_by_id = assignments["split_role"].to_dict()
            compat["set"] = compat["variant_id"].map(
                lambda value, mapping=role_by_id: (
                    "test" if mapping[value] == "final_test" else "train"
                )
            )
            compat["validation"] = compat["variant_id"].map(
                lambda value, mapping=role_by_id: mapping[value] == "benchmark_validation"
            )
            compat = compat.loc[:, ["sequence", "target", "set", "validation"]]
            outputs.append(("compat/flip.csv.gz", compat))
        file_records: list[dict[str, Any]] = []
        for relative, frame in outputs:
            record = _write_frame(fold_root / relative, frame)
            record["path"] = relative
            file_records.append(record)
        assignment_payload = _csv_gzip_bytes(
            fold.assignments.sort_values("variant_id", kind="stable").reset_index(drop=True)
        )
        fold_manifest = {
            "fold_index": fold.fold_index,
            "strategy": result.strategy,
            "protocol_version": request.protocol_version,
            "assignment_sha256": _sha256_bytes(assignment_payload),
            "role_counts": {
                str(key): int(value)
                for key, value in fold.assignments["split_role"].value_counts().sort_index().items()
            },
            "metadata": fold.metadata,
            "files": file_records,
        }
        fold_manifest_path = fold_root / "fold_manifest.json"
        _write_json(fold_manifest_path, fold_manifest)
        root_files.append(
            {
                "path": fold_manifest_path.relative_to(root).as_posix(),
                "sha256": _sha256_bytes(fold_manifest_path.read_bytes()),
            }
        )
    _write_json(root / "audit_summary.json", audit)
    manifest = {
        **config_record,
        "config_sha256": config_hash,
        "source": str(dataset.spec.source),
        "source_url": dataset.spec.source_url,
        "source_sha256": dataset.source_sha256,
        "canonical_row_count": len(dataset.features),
        "code_sha256": current_code_hash,
        "strategy_metadata": {
            key: value for key, value in result.metadata.items() if key != "identity_shards"
        },
        "audit_sha256": _sha256_bytes((root / "audit_summary.json").read_bytes()),
        "fold_manifests": root_files,
    }
    _write_json(existing_manifest, manifest)
    return root
