#!/usr/bin/env python3
"""Run the four-condition hierarchical matrix as a 1-fold x 1-round canary."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fitness_agents.config import project_root

CONDITIONS = "kg_base,kg_base_rag,kg_base_al,kg_3features_rag"


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs/experiments/hierarchical_scientist.deepseek.yaml",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    root = project_root()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or root / "artifacts" / f"hierarchical-canary-{stamp}"
    command = [
        sys.executable,
        str(root / "scripts/run_hierarchical_scientist.py"),
        "--config",
        str(args.config),
        "--folds",
        str(args.fold),
        "--conditions",
        CONDITIONS,
        "--rounds",
        "1",
        "--max-parallel",
        str(args.max_parallel),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--output-dir",
        str(output_dir),
        "--placeholder-predictor",
    ]
    if args.seed is not None:
        command.extend(("--seed", str(args.seed)))
    if args.dry_run:
        command.append("--dry-run")
    return command


def main() -> int:
    args = parse_args()
    if args.fold < 0:
        raise SystemExit("--fold must be non-negative")
    if args.max_parallel < 1 or args.max_parallel > 4:
        raise SystemExit("--max-parallel must be between 1 and 4 for the canary")
    completed = subprocess.run(
        build_command(args),
        cwd=project_root(),
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
