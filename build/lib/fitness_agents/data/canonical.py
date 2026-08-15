from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .specs import DatasetSpec

TARGET_PROXY_COLUMNS = frozenset(
    {
        "target",
        "fitness",
        "dms_score",
        "raw_fitness",
        "normalized_fitness",
        "input_count",
        "selected_count",
        "enrichment",
        "keep",
        "low_vs_high",
        "top_k",
        "percentile",
        "source_set",
        "source_validation",
    }
)


@dataclass(frozen=True)
class CanonicalDataset:
    features: pd.DataFrame
    labels: pd.DataFrame
    spec: DatasetSpec
    source_sha256: str

    def __post_init__(self) -> None:
        if "variant_id" not in self.features or "variant_id" not in self.labels:
            raise ValueError("Canonical features and labels both require variant_id")
        if self.features["variant_id"].duplicated().any():
            raise ValueError("Canonical features contain duplicate variant IDs")
        if self.labels["variant_id"].duplicated().any():
            raise ValueError("Canonical labels contain duplicate variant IDs")
        if set(self.features["variant_id"]) != set(self.labels["variant_id"]):
            raise ValueError("Canonical features and labels have different variant IDs")
        hidden = TARGET_PROXY_COLUMNS.intersection(column.lower() for column in self.features)
        if hidden:
            raise ValueError(f"Canonical features contain target/proxy columns: {sorted(hidden)}")

