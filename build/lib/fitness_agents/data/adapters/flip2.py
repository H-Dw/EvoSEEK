from __future__ import annotations

import json

import pandas as pd

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import DatasetSpec

from .common import assemble_canonical, canonical_variant_id, mutation_tokens, sha256_file
from .paired_sequence import split_component_sequences


class Flip2Adapter:
    """Config-driven adapter for single- and multi-component FLIP-2 style CSVs."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec

    def canonicalize(self) -> CanonicalDataset:
        raw = pd.read_csv(self.spec.source, low_memory=False)
        required = {self.spec.sequence_column, self.spec.target_column}
        if missing := required.difference(raw.columns):
            raise ValueError(f"Source is missing columns: {sorted(missing)}")
        source_hash = sha256_file(self.spec.source)
        records: list[dict[str, object]] = []
        for source_row_id, row in raw.iterrows():
            components = split_component_sequences(row[self.spec.sequence_column], self.spec)
            backbone_id = (
                str(row[self.spec.backbone_column])
                if self.spec.backbone_column
                else self.spec.default_backbone_id
            )
            tokens = mutation_tokens(components, self.spec, backbone_id)
            sequence = self.spec.component_separator.join(components)
            records.append(
                {
                    "dataset_id": self.spec.dataset_id,
                    "assay_id": self.spec.assay_id,
                    "variant_id": canonical_variant_id(self.spec, backbone_id, components),
                    "backbone_id": backbone_id,
                    "variant": sequence,
                    "sequence": sequence,
                    "component_sequences": json.dumps(list(components), separators=(",", ":")),
                    "mutation_tokens": json.dumps(list(tokens), separators=(",", ":")),
                    "mutation_count": len(tokens),
                    "mutated_positions": json.dumps(
                        [json.loads(token)[1:3] for token in tokens], separators=(",", ":")
                    ),
                    "source_row_id": int(source_row_id),
                    "source_sha256": source_hash,
                    "target": row[self.spec.target_column],
                }
            )
        return assemble_canonical(records, self.spec, source_hash)

