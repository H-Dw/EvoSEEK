from __future__ import annotations

import json

import pandas as pd
import pytest

from fitness_agents.data.adapters import create_adapter
from fitness_agents.data.loader import load_fold_bundle
from fitness_agents.data.specs import ComponentSpec, DatasetSpec
from fitness_agents.data.splitting import SplitRequest, build_split
from fitness_agents.data.splitting.writer import write_split


def test_paired_sequence_adapter_creates_component_aware_tokens(tmp_path):
    source = tmp_path / "paired.csv"
    pd.DataFrame(
        {"sequence": ["AD:CE", ":CE"], "target": [1.5, 0.8]}
    ).to_csv(source, index=False)
    spec = DatasetSpec(
        dataset_id="paired",
        assay_id="paired_assay",
        adapter="paired_sequence",
        source=source,
        sequence_column="sequence",
        target_column="target",
        components=(
            ComponentSpec("PDZ3", "AA", (10, 11)),
            ComponentSpec("CRIPT", "CC", (1, 2)),
        ),
    )
    dataset = create_adapter(spec).canonicalize()
    paired = dataset.features.loc[dataset.features["sequence"] == "AD:CE"].iloc[0]
    tokens = [json.loads(value) for value in json.loads(paired.mutation_tokens)]
    assert ["WT", "PDZ3", 11, "A", "D"] in tokens
    assert ["WT", "CRIPT", 2, "C", "E"] in tokens
    assert paired.mutation_count == 2
    blank_side = dataset.features.loc[dataset.features["sequence"] == "AA:CE"].iloc[0]
    blank_tokens = [json.loads(value) for value in json.loads(blank_side.mutation_tokens)]
    assert all(token[1] == "CRIPT" for token in blank_tokens)


def test_conflicting_duplicate_targets_fail_closed(tmp_path):
    source = tmp_path / "duplicates.csv"
    pd.DataFrame(
        {"sequence": ["AA", "AA"], "target": [0.0, 1.0]}
    ).to_csv(source, index=False)
    spec = DatasetSpec(
        dataset_id="duplicates",
        assay_id="duplicates_assay",
        adapter="generic_sequence",
        source=source,
        sequence_column="sequence",
        target_column="target",
        components=(ComponentSpec("protein", "AA"),),
        target_conflict_tolerance=0.0,
    )
    with pytest.raises(ValueError, match="Conflicting replicate targets"):
        create_adapter(spec).canonicalize()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"ood_rule": "low_vs_high", "population": "full"}, "low_vs_high"),
        ({"ood_rule": "two_vs_rest", "population": "flip_keep"}, "flip_keep"),
    ],
)
def test_label_dependent_membership_requires_explicit_opt_in(
    synthetic_landscape, options, message
):
    dataset = synthetic_landscape
    with pytest.raises(PermissionError, match=message):
        build_split(dataset, SplitRequest("flip_static_ood", options=options))


def test_loader_detects_tampered_capability_file(tmp_path, synthetic_landscape):
    dataset = synthetic_landscape
    request = SplitRequest(
        "al96_closed_loop",
        protocol_version="tamper-test",
        options={
            "initial_budget": 26,
            "test_depth_min": 3,
            "validation_size": 20,
            "validation_strata": ("mutation_count", "backbone_id"),
        },
    )
    root = write_split(dataset, request, build_split(dataset, request), tmp_path / "outputs")
    candidate_path = root / "fold_00/agent/candidate_pool.csv.gz"
    candidate_path.write_bytes(candidate_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_fold_bundle(root, 0, "agent")
