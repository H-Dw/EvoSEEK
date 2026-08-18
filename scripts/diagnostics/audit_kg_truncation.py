from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.kg_interaction import KGKeywordTruncationAuditor
from fitness_agents.kg_knowledge import SQLiteGraphSink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit literal keyword items against visible structured-KG rows."
    )
    parser.add_argument("structured_kg", type=Path, help="Path to structured_kg.sqlite")
    parser.add_argument("--round-id", type=int, required=True)
    parser.add_argument("--max-rows", type=int, default=12)
    parser.add_argument("--sample-rows", type=int, default=3)
    parser.add_argument(
        "--item",
        action="append",
        required=True,
        help="Literal keyword item to count; repeat for multiple items.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.structured_kg.is_file():
        raise FileNotFoundError(f"Structured KG does not exist: {args.structured_kg}")
    sink = SQLiteGraphSink(args.structured_kg)
    try:
        report = KGKeywordTruncationAuditor(sink).audit(
            args.item,
            round_id=args.round_id,
            max_rows=args.max_rows,
            sample_rows=args.sample_rows,
        )
        payload = report.as_dict()
    finally:
        sink.close()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
