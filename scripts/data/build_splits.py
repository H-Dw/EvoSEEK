#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fitness_agents.config import project_root
from fitness_agents.data.adapters import create_adapter
from fitness_agents.data.specs import load_dataset_spec
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.writer import write_split

STRATEGIES = ("al96_closed_loop", "flip_static_ood", "mutation_identity_ood")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Build auditable five-fold closed-loop and OOD protein splits"
    )
    parser.add_argument("--dataset-spec", type=Path, required=True)
    parser.add_argument("--strategy", choices=(*STRATEGIES, "all"), required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--public-salt")
    parser.add_argument("--protocol-version", default="v1")
    parser.add_argument(
        "--output-root", type=Path, default=root / "data/processed/splits"
    )
    parser.add_argument("--allow-label-dependent-membership", action="store_true")

    parser.add_argument("--initial-budget", type=int, default=96)
    parser.add_argument("--test-depth-min", type=int, default=3)
    parser.add_argument("--validation-size", type=int, default=384)
    parser.add_argument(
        "--validation-strata", default="mutation_count,backbone_id"
    )

    parser.add_argument(
        "--ood-rule",
        choices=("one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high"),
        default="two_vs_rest",
    )
    parser.add_argument("--population", choices=("full", "flip_keep"), default="full")
    parser.add_argument(
        "--mutation-row-policy",
        choices=("contains_unseen", "pure_group_only"),
        default="contains_unseen",
    )
    parser.add_argument("--mixed-policy", choices=("quarantine",), default="quarantine")
    return parser.parse_args()


def _options(args: argparse.Namespace, strategy: str) -> dict[str, object]:
    if strategy == "al96_closed_loop":
        return {
            "initial_budget": args.initial_budget,
            "test_depth_min": args.test_depth_min,
            "validation_size": args.validation_size,
            "validation_strata": tuple(
                value.strip() for value in args.validation_strata.split(",") if value.strip()
            ),
            "initial_policy": "low_order_coverage",
        }
    if strategy == "flip_static_ood":
        return {"ood_rule": args.ood_rule, "population": args.population}
    return {
        "mutation_row_policy": args.mutation_row_policy,
        "mixed_policy": args.mixed_policy,
    }


def main() -> None:
    args = parse_args()
    if args.n_folds != 5:
        print("WARNING: formal benchmark configuration uses exactly five folds")
    spec = load_dataset_spec(args.dataset_spec)
    dataset = create_adapter(spec).canonicalize()
    selected = STRATEGIES if args.strategy == "all" else (args.strategy,)
    outputs: dict[str, object] = {}
    for strategy in selected:
        request = SplitRequest(
            strategy=strategy,
            n_folds=args.n_folds,
            seed=args.seed,
            public_salt=args.public_salt,
            protocol_version=args.protocol_version,
            options=_options(args, strategy),
            allow_label_dependent_membership=args.allow_label_dependent_membership,
        )
        result = build_split(dataset, request)
        output = write_split(dataset, request, result, args.output_root)
        outputs[strategy] = {
            "output": str(output),
            "folds": len(result.folds),
            "rows": len(dataset.features),
        }
    print(json.dumps(outputs, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

