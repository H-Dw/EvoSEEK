from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "src"
    sys.path.insert(0, str(source_root))
    from fitness_agents.deep_research.cli import main as deep_research_main

    return deep_research_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
