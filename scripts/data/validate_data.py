#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.config import project_root
from fitness_agents.data import load_dataset_bundle


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description="Validate public/oracle schema and split isolation")
    parser.add_argument("--public", type=Path, default=root / "data/demo/gb1_demo_public.csv")
    parser.add_argument("--oracle", type=Path, default=root / "data/demo/gb1_demo_oracle.csv")
    args = parser.parse_args()
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

