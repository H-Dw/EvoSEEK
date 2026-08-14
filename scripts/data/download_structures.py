#!/usr/bin/env python3
"""Download ProteinGym AF2 structures (~84 MB) for structure-evidence modules."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from fitness_agents.config import project_root
from fitness_agents.data.download import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    results = run(["proteingym_structures"], project_root(), force=args.force,
                  offline=args.offline, verify_only=args.verify_only)
    for result in results:
        print(f"{result.dataset_id}: {result.status} ({len(result.files)} files) -> {result.dest}")
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
