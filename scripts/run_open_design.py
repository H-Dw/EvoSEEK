"""Prompt/FASTA entry point for the same service used by the local UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.interaction import EvolutionApplicationService


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview and run one open-design request")
    parser.add_argument("--experiment", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    sequence_text = args.fasta.read_text(encoding="utf-8") if args.fasta else None
    service = EvolutionApplicationService(args.experiment)
    preview = service.preview(args.prompt, sequence_text=sequence_text)
    print(json.dumps(preview.model_dump(mode="json"), ensure_ascii=False, indent=2))
    if args.confirm:
        result = service.run(preview.preview_id, confirmed=True)
        print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

