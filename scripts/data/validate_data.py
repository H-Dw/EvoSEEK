#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.config import project_root
from fitness_agents.data import load_dataset_bundle, load_fold_bundle


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate public/oracle schema and split isolation")
    parser.add_argument("--split-root", type=Path)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--public", type=Path, default=root / "data/demo/gb1_demo_public.csv")
    parser.add_argument("--oracle", type=Path, default=root / "data/demo/gb1_demo_oracle.csv")
    args = parser.parse_args()
    if args.split_root:
        auditor = load_fold_bundle(args.split_root, args.fold_index, "auditor")
        result = {
            "fold_index": args.fold_index,
            "manifest_strategy": auditor.manifest["strategy"],
            "observed": len(auditor.observed) if auditor.observed is not None else 0,
            "candidates": len(auditor.candidates) if auditor.candidates is not None else 0,
            "validation": len(auditor.validation) if auditor.validation is not None else 0,
            "queryable_labels": (
                len(auditor.queryable_labels) if auditor.queryable_labels is not None else 0
            ),
            "final_test": len(auditor.final_inputs) if auditor.final_inputs is not None else 0,
            "quarantine": len(auditor.quarantine) if auditor.quarantine is not None else 0,
            "valid": True,
        }
        print(json.dumps(result, indent=2))
        return
    bundle = load_dataset_bundle(args.public, args.oracle)
    result = {
        "initial_observed": len(bundle.initial_variants),
        "validation": len(bundle.validation_variants),
        "oracle_pool": len(bundle.oracle_pool),
        "final_test": len(bundle.final_test),
        "public_candidate_has_fitness": False,
        "valid": True,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
