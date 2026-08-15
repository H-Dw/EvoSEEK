from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import DatasetSpec

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mutation_tokens(
    component_sequences: tuple[str, ...], spec: DatasetSpec, backbone_id: str
) -> tuple[str, ...]:
    tokens: list[str] = []
    for sequence, component in zip(component_sequences, spec.components, strict=True):
        if len(sequence) != len(component.reference):
            raise ValueError(
                f"Component {component.component_id!r} expected length "
                f"{len(component.reference)}, received {len(sequence)}"
            )
        if set(sequence).difference(AMINO_ACIDS):
            raise ValueError(f"Component {component.component_id!r} contains invalid residues")
        positions = component.positions or tuple(range(1, len(component.reference) + 1))
        for position, wild_type, mutant in zip(
            positions, component.reference, sequence, strict=True
        ):
            if mutant != wild_type:
                tokens.append(
                    json.dumps(
                        [backbone_id, component.component_id, position, wild_type, mutant],
                        separators=(",", ":"),
                    )
                )
    return tuple(sorted(tokens))


def canonical_variant_id(
    spec: DatasetSpec, backbone_id: str, component_sequences: tuple[str, ...]
) -> str:
    payload = json.dumps(
        [spec.assay_id, backbone_id, list(component_sequences)],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def assemble_canonical(
    records: list[dict[str, object]], spec: DatasetSpec, source_sha256: str
) -> CanonicalDataset:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No valid records were produced by the adapter")
    frame["target"] = pd.to_numeric(frame["target"], errors="raise")
    if not np.isfinite(frame["target"].to_numpy(dtype=float)).all():
        raise ValueError("Targets must be finite numeric values")
    rows: list[dict[str, object]] = []
    conflicts: list[str] = []
    for variant_id, group in frame.groupby("variant_id", sort=True):
        targets = group["target"].to_numpy(dtype=float)
        spread = float(np.max(targets) - np.min(targets))
        if spread > spec.target_conflict_tolerance:
            conflicts.append(str(variant_id))
            continue
        if spec.replicate_policy == "mean":
            target = float(np.mean(targets))
        elif spec.replicate_policy == "first":
            target = float(targets[0])
        else:
            raise ValueError(f"Unsupported replicate_policy: {spec.replicate_policy}")
        row = group.sort_values("source_row_id", kind="stable").iloc[0].to_dict()
        row["target"] = target
        rows.append(row)
    if conflicts:
        preview = ", ".join(conflicts[:5])
        raise ValueError(
            f"Conflicting replicate targets exceed tolerance for {len(conflicts)} variants: {preview}"
        )
    canonical = pd.DataFrame(rows).sort_values("variant_id", kind="stable").reset_index(drop=True)
    label_frame = canonical.loc[:, ["variant_id", "target"]].copy()
    feature_frame = canonical.drop(columns="target")
    return CanonicalDataset(feature_frame, label_frame, spec, source_sha256)
