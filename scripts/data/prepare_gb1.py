#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.config import project_root
from fitness_agents.data.gb1 import build_gb1_benchmark


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Build leakage-safe full and demo GB1 datasets")
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data/raw/flip/gb1/four_mutations_full_data.csv",
    )
    parser.add_argument("--processed-dir", type=Path, default=root / "data/processed")
    parser.add_argument("--demo-dir", type=Path, default=root / "data/demo")
    parser.add_argument("--seed", type=int, default=20260814)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_gb1_benchmark(
        args.source,
        processed_dir=args.processed_dir,
        demo_dir=args.demo_dir,
        seed=args.seed,
    )
    print(json.dumps({key: str(value) for key, value in result.items() if key != "manifest"}, indent=2))
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

