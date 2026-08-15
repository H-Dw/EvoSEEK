from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import ComponentSpec, DatasetSpec


@pytest.fixture
def synthetic_landscape(tmp_path: Path) -> CanonicalDataset:
    reference = "AAAA"
    alphabet = "ACDEFG"
    component = ComponentSpec("protein", reference, (1, 2, 3, 4))
    spec = DatasetSpec(
        dataset_id="synthetic",
        assay_id="synthetic_assay",
        adapter="generic_sequence",
        source=tmp_path / "source.csv",
        sequence_column="sequence",
        target_column="target",
        components=(component,),
    )
    rows = []
    for index, residues in enumerate(product(alphabet, repeat=4)):
        sequence = "".join(residues)
        tokens = []
        for position, (wild_type, mutant) in enumerate(
            zip(reference, sequence, strict=True), start=1
        ):
            if wild_type != mutant:
                tokens.append(
                    json.dumps(
                        ["WT", "protein", position, wild_type, mutant], separators=(",", ":")
                    )
                )
        rows.append(
            {
                "dataset_id": "synthetic",
                "assay_id": "synthetic_assay",
                "variant_id": f"v{index:04d}",
                "backbone_id": "WT",
                "variant": sequence,
                "sequence": sequence,
                "component_sequences": json.dumps([sequence]),
                "mutation_tokens": json.dumps(sorted(tokens), separators=(",", ":")),
                "mutation_count": len(tokens),
                "mutated_positions": json.dumps([]),
                "source_row_id": index,
                "source_sha256": "synthetic",
            }
        )
    features = pd.DataFrame(rows)
    labels = pd.DataFrame(
        {
            "variant_id": features["variant_id"],
            "target": [float(value) for value in range(len(features))],
        }
    )
    return CanonicalDataset(features, labels, spec, "synthetic")

