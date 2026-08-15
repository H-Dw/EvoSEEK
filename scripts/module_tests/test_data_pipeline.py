from __future__ import annotations

from itertools import product

import pandas as pd
from common import ensure, load_config, parse_args, resolve_output, write_result

from fitness_agents.data.adapters import create_adapter
from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.loader import load_fold_bundle
from fitness_agents.data.specs import ComponentSpec, DatasetSpec
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.audit import audit_split
from fitness_agents.data.splitting.writer import write_split


def main() -> None:
    args = parse_args("configs/module_tests/data_pipeline.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    sample = config["sample"]
    reference = str(sample["reference"])
    residues = tuple(tuple(str(item) for item in site) for site in sample["residues_by_site"])
    raw_path = output / "synthetic_raw.csv"
    raw = pd.DataFrame(
        {
            "sequence": ["".join(code) for code in product(*residues)],
        }
    )
    raw["target"] = raw["sequence"].map(
        lambda code: sum((index + 1) * (ord(residue) % 11) for index, residue in enumerate(code))
        / 100.0
    )
    raw.to_csv(raw_path, index=False)

    spec = DatasetSpec(
        dataset_id="module_test_grid",
        assay_id="module_test_assay",
        adapter="generic_sequence",
        source=raw_path,
        sequence_column="sequence",
        target_column="target",
        components=(
            ComponentSpec(
                "GB1",
                reference,
                tuple(int(item) for item in sample["positions"]),
            ),
        ),
    )
    dataset = create_adapter(spec).canonicalize()
    ensure(len(dataset.features) == len(raw), "Canonical adapter changed the unique row count")
    ensure("target" not in dataset.features, "Canonical features leaked the target")
    ensure(set(dataset.labels.columns) == {"variant_id", "target"}, "Label schema is wrong")

    leakage_guard = False
    try:
        CanonicalDataset(
            dataset.features.assign(raw_fitness=dataset.labels["target"].to_numpy()),
            dataset.labels,
            dataset.spec,
            dataset.source_sha256,
        )
    except ValueError as error:
        leakage_guard = "target/proxy" in str(error)
    ensure(leakage_guard, "Canonical target-proxy leakage guard did not fire")

    split_cfg = config["split"]
    n_folds = int(split_cfg["n_folds"])
    strategy_results: dict[str, object] = {}
    for strategy in (
        "al96_closed_loop",
        "flip_static_ood",
        "mutation_identity_ood",
    ):
        values = dict(split_cfg[strategy])
        protocol = str(values.pop("protocol_version"))
        request = SplitRequest(
            strategy=strategy,
            n_folds=n_folds,
            seed=int(config["seed"]),
            protocol_version=protocol,
            options=values,
        )
        split = build_split(dataset, request)
        audit = audit_split(dataset, split)
        ensure(audit["valid"], f"{strategy} leakage audit failed: {audit['hard_failures']}")
        split_root = write_split(dataset, request, split, output / "splits")
        ensure(
            write_split(dataset, request, split, output / "splits") == split_root,
            f"{strategy} idempotent manifest write failed",
        )

        views = {
            role: load_fold_bundle(split_root, 0, role)
            for role in ("agent", "controller", "oracle", "evaluator_inputs", "evaluator", "auditor")
        }
        ensure(views["agent"].queryable_labels is None, "Agent received oracle labels")
        ensure(views["agent"].final_labels is None, "Agent received final-test labels")
        if views["agent"].candidates is not None:
            ensure("target" not in views["agent"].candidates, "Candidate table leaked target")
        ensure(views["oracle"].final_labels is None, "Oracle received evaluator labels")
        ensure(views["evaluator"].queryable_labels is None, "Evaluator received oracle labels")
        ensure(views["auditor"].final_labels is not None, "Auditor lacks final labels")
        strategy_results[strategy] = {
            "split_root": split_root,
            "folds": audit["folds"],
            "manifest_strategy": views["agent"].manifest["strategy"],
            "agent_candidate_rows": (
                len(views["agent"].candidates) if views["agent"].candidates is not None else 0
            ),
            "final_test_rows": (
                len(views["evaluator"].final_labels)
                if views["evaluator"].final_labels is not None
                else 0
            ),
        }

    write_result(
        output,
        "data_pipeline",
        {
            "config": config["_config_path"],
            "raw_rows": len(raw),
            "canonical_rows": len(dataset.features),
            "target_proxy_guard": leakage_guard,
            "strategies": strategy_results,
        },
    )


if __name__ == "__main__":
    main()

