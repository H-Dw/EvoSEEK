"""Build an offline ESM-2 feature library for Kermut lookup.

The NPZ is keyed by variant_id and also stores sequences so the same file can
serve every AL96 fold. Campaigns with feature_mode: live_esm2 read this path
from options.precomputed_features_path and only run live ESM-2 on misses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_agents.config import project_root
from fitness_agents.contracts.schemas import Variant
from fitness_agents.models.backends.kermut_features import create_kermut_feature_source


def _collect_split_rows(split_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(split_root.rglob("*.csv.gz")):
        frame = pd.read_csv(path)
        if not {"variant_id", "sequence"}.issubset(frame.columns):
            continue
        columns = ["variant_id", "sequence"]
        if "variant" in frame.columns:
            columns.append("variant")
        if "mutation_notation" in frame.columns:
            columns.append("mutation_notation")
        if "mutation_count" in frame.columns:
            columns.append("mutation_count")
        if "split_role" in frame.columns:
            columns.append("split_role")
        frames.append(frame.loc[:, columns])
    if not frames:
        raise FileNotFoundError(f"No variant tables with sequence columns under {split_root}")
    return pd.concat(frames, ignore_index=True).drop_duplicates("variant_id")


def _row_to_variant(row: object) -> Variant:
    sequence = str(row.sequence)
    code = str(getattr(row, "variant", sequence))
    return Variant(
        variant_id=str(row.variant_id),
        variant=code,
        sequence=sequence,
        mutation_notation=str(getattr(row, "mutation_notation", code)),
        mutation_count=int(getattr(row, "mutation_count", 0)),
        split_role=str(getattr(row, "split_role", "oracle_pool")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="Versioned split directory whose CSV tables define the closed search library",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination NPZ (embeddings, zero_shot, variant_ids, sequences)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/model_cache/kermut_esm2"),
        help="Per-sequence npy cache used while computing and by later live fallback",
    )
    parser.add_argument(
        "--checkpoint",
        default="~/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--wild-type-sequence", default=None)
    args = parser.parse_args()

    root = project_root()
    split_root = args.split_root if args.split_root.is_absolute() else root / args.split_root
    output = args.output if args.output.is_absolute() else root / args.output
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else root / args.cache_dir
    table = _collect_split_rows(split_root)
    variants = [_row_to_variant(row) for row in table.itertuples(index=False)]
    wild_type = args.wild_type_sequence
    if not wild_type:
        wt_rows = [item for item in variants if item.mutation_count == 0]
        if not wt_rows:
            raise ValueError("Pass --wild-type-sequence; no mutation_count=0 row was found")
        wild_type = wt_rows[0].sequence

    source = create_kermut_feature_source(
        device=args.device,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        options={
            "feature_mode": "live_esm2",
            "cache_dir": str(cache_dir),
            "precomputed_features_path": None,
        },
    )
    embeddings, zero_shot = source.encode(variants, wild_type)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        variant_ids=np.asarray([item.variant_id for item in variants]),
        sequences=np.asarray([item.sequence for item in variants]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        zero_shot=np.asarray(zero_shot, dtype=np.float32),
    )
    print(
        f"Wrote {len(variants)} ESM-2 rows to {output} "
        f"(embedding dim={embeddings.shape[1]}, cache_dir={cache_dir})"
    )


if __name__ == "__main__":
    main()
