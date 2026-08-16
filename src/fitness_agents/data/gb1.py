from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_agents.mutation.notation import edits_from_site_code, format_canonical

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)
GB1_WT_SITES = "VDGV"
GB1_POSITIONS = (39, 40, 41, 54)
SOURCE_URL = "https://github.com/J-SNACKKB/FLIP/tree/main/splits/gb1"


def canonical_mutation_notation(
    variant: str,
    wild_type: str = GB1_WT_SITES,
    positions: Iterable[int] = GB1_POSITIONS,
) -> str:
    return format_canonical(
        edits_from_site_code(variant, wild_type=wild_type, positions=positions)
    )


def variant_id(variant: str, assay_id: str = "GB1_IgG_binding_Wu2016") -> str:
    notation = canonical_mutation_notation(variant)
    digest = hashlib.sha256(f"GB1|{assay_id}|{notation}".encode()).hexdigest()
    return f"sha256:{digest}"


def _read_source(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(f"Expected one CSV in {path}; found {members}")
            with archive.open(members[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path, low_memory=False)


def _clean_source(source: Path) -> pd.DataFrame:
    raw = _read_source(source)
    required = {"Variants", "HD", "Fitness", "sequence"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"GB1 source is missing columns: {sorted(missing)}")
    frame = raw.loc[:, ["Variants", "HD", "Fitness", "sequence"]].rename(
        columns={"Variants": "variant", "HD": "mutation_count", "Fitness": "fitness"}
    )
    frame["variant"] = frame["variant"].astype(str).str.upper()
    valid = frame["variant"].map(lambda value: len(value) == 4 and set(value) <= AA_SET)
    frame = frame.loc[valid].dropna(subset=["fitness"]).drop_duplicates("variant").copy()
    frame["fitness"] = pd.to_numeric(frame["fitness"], errors="raise")
    frame["mutation_count"] = frame["variant"].map(
        lambda value: sum(a != b for a, b in zip(value, GB1_WT_SITES, strict=True))
    )
    frame["variant_id"] = frame["variant"].map(variant_id)
    frame["mutation_notation"] = frame["variant"].map(canonical_mutation_notation)
    return frame.sort_values("variant").reset_index(drop=True)


def _stratified_take(
    frame: pd.DataFrame,
    size: int,
    rng: np.random.Generator,
    *,
    guarantee_wt: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if size > len(frame):
        raise ValueError(f"Requested {size} rows from a pool of {len(frame)}")
    chosen: list[int] = []
    if guarantee_wt and (frame["variant"] == GB1_WT_SITES).any():
        chosen.append(int(frame.index[frame["variant"] == GB1_WT_SITES][0]))

    remaining_size = size - len(chosen)
    available = frame.drop(index=chosen)
    counts = available["mutation_count"].value_counts().sort_index()
    ideal = counts / counts.sum() * remaining_size
    allocation = np.floor(ideal).astype(int)
    for depth in counts.index:
        if counts.loc[depth] > 0 and allocation.loc[depth] == 0 and allocation.sum() < remaining_size:
            allocation.loc[depth] = 1
    while allocation.sum() < remaining_size:
        depth = (ideal - allocation).idxmax()
        allocation.loc[depth] += 1
    while allocation.sum() > remaining_size:
        eligible = allocation[allocation > 1]
        depth = (allocation - ideal).loc[eligible.index].idxmax()
        allocation.loc[depth] -= 1

    for depth, count in allocation.items():
        indices = available.index[available["mutation_count"] == depth].to_numpy()
        chosen.extend(rng.choice(indices, size=int(count), replace=False).tolist())
    selected = frame.loc[chosen].copy()
    rest = frame.drop(index=chosen).copy()
    return selected.reset_index(drop=True), rest.reset_index(drop=True)


def _assign_splits(frame: pd.DataFrame, sizes: dict[str, int], seed: int) -> pd.DataFrame:
    if sum(sizes.values()) > len(frame):
        raise ValueError("Requested split sizes exceed the source dataset")
    rng = np.random.default_rng(seed)
    pool = frame.copy()
    parts: list[pd.DataFrame] = []
    for role in ("initial_observed", "validation", "final_test", "oracle_pool"):
        count = sizes[role]
        selected, pool = _stratified_take(
            pool, count, rng, guarantee_wt=(role == "initial_observed")
        )
        selected["split_role"] = role
        parts.append(selected)
    return pd.concat(parts, ignore_index=True)


def _write_benchmark(frame: pd.DataFrame, output_dir: Path, prefix: str) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    public_columns = [
        "variant_id",
        "variant",
        "sequence",
        "mutation_notation",
        "mutation_count",
        "split_role",
    ]
    label_columns = ["variant_id", "fitness", "split_role"]
    public_path = output_dir / f"{prefix}_public.csv"
    oracle_path = output_dir / f"{prefix}_oracle.csv"
    frame.loc[:, public_columns].to_csv(public_path, index=False)
    frame.loc[:, label_columns].to_csv(oracle_path, index=False)
    return {"public": public_path, "oracle": oracle_path}


def build_gb1_benchmark(
    source: str | Path,
    *,
    processed_dir: str | Path,
    demo_dir: str | Path,
    seed: int = 20260814,
) -> dict[str, object]:
    source_path = Path(source)
    frame = _clean_source(source_path)
    full_sizes = {
        "initial_observed": 96,
        "validation": 96,
        "final_test": 2048,
        "oracle_pool": len(frame) - 2240,
    }
    full = _assign_splits(frame, full_sizes, seed)
    full_paths = _write_benchmark(full, Path(processed_dir), "gb1_full")

    demo_sample, _ = _stratified_take(frame, 512, np.random.default_rng(seed + 1), guarantee_wt=True)
    demo_sizes = {
        "initial_observed": 64,
        "validation": 32,
        "final_test": 64,
        "oracle_pool": 352,
    }
    demo = _assign_splits(demo_sample, demo_sizes, seed + 2)
    demo_paths = _write_benchmark(demo, Path(demo_dir), "gb1_demo")

    manifest = {
        "source": str(source_path),
        "source_url": SOURCE_URL,
        "raw_license": "CC BY 4.0",
        "derived_license": "AFL-3.0",
        "rows_clean": len(frame),
        "fitness_min": float(frame["fitness"].min()),
        "fitness_max": float(frame["fitness"].max()),
        "hamming_depth_counts": {
            str(key): int(value) for key, value in frame["mutation_count"].value_counts().items()
        },
        "demo_split_counts": {
            key: int(value) for key, value in demo["split_role"].value_counts().items()
        },
        "full_split_counts": {
            key: int(value) for key, value in full["split_role"].value_counts().items()
        },
    }
    manifest_path = Path(demo_dir) / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"manifest": manifest, "manifest_path": manifest_path, **demo_paths, **{
        f"full_{key}": value for key, value in full_paths.items()
    }}
