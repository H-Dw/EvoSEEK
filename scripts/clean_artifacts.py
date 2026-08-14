#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from fitness_agents.config import project_root


def main() -> None:
    root = project_root().resolve()
    parser = argparse.ArgumentParser(description="Remove generated run artifacts inside this project")
    parser.add_argument("--path", type=Path, default=root / "artifacts/runs")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion")
    args = parser.parse_args()
    target = args.path.resolve()
    allowed_root = (root / "artifacts").resolve()
    if allowed_root not in target.parents or target == allowed_root:
        raise SystemExit(f"Refusing to delete outside an artifacts subdirectory: {target}")
    if not args.yes:
        raise SystemExit("Pass --yes after checking the printed target")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    (target / ".gitkeep").touch()
    print(f"Removed generated artifacts under {target}")

