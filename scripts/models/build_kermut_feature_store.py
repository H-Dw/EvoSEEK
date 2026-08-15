"""Precompute the ESM-2 inputs consumed by the built-in Kermut backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fitness_agents.contracts.schemas import Variant
from fitness_agents.models.backends.kermut_features import LiveESM2KermutFeatures

REQUIRED_COLUMNS = {
    "variant_id",
    "variant",
    "sequence",
    "mutation_notation",
    "mutation_count",
    "split_role",
}


def _load_variants(path: Path) -> list[Variant]:
    frame = pd.read_csv(path)
    if missing := REQUIRED_COLUMNS.difference(frame.columns):
        raise ValueError(f"Public candidate table is missing columns: {sorted(missing)}")
    if frame["variant_id"].duplicated().any():
        raise ValueError("Public candidate table contains duplicate variant_id values")
    return [
        Variant(
            variant_id=str(row.variant_id),
            variant=str(row.variant),
            sequence=str(row.sequence),
            mutation_notation=str(row.mutation_notation),
            mutation_count=int(row.mutation_count),
            split_role=str(row.split_role),
        )
        for row in frame.itertuples(index=False)
    ]


def _wild_type(variants: list[Variant], configured: str | None) -> str:
    observed = {variant.sequence for variant in variants if variant.mutation_count == 0}
    if configured:
        if observed and observed != {configured}:
            raise ValueError("--wild-type-sequence disagrees with the public candidate table")
        return configured
    if len(observed) != 1:
        raise ValueError(
            "The public candidate table must contain one unique mutation_count=0 sequence, "
            "or --wild-type-sequence must be provided"
        )
    return observed.pop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wild-type-sequence")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-dir", type=Path, default=Path("artifacts/model_cache/kermut_esm2"))
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--esm-representation-layer", type=int, default=33)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    variants = _load_variants(args.public_csv)
    if not variants:
        raise ValueError("Public candidate table is empty")
    wild_type = _wild_type(variants, args.wild_type_sequence)
    feature_source = LiveESM2KermutFeatures(
        device=args.device,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint,
        options={
            "cache_dir": str(args.cache_dir),
            "esm_model": args.esm_model,
            "esm_representation_layer": args.esm_representation_layer,
        },
    )
    embeddings, zero_shot = feature_source.encode(variants, wild_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        variant_ids=np.asarray([variant.variant_id for variant in variants]),
        embeddings=np.asarray(embeddings, dtype=np.float32),
        zero_shot=np.asarray(zero_shot, dtype=np.float32),
    )
    print(f"Wrote {len(variants)} Kermut feature rows to {args.output}")


if __name__ == "__main__":
    main()
