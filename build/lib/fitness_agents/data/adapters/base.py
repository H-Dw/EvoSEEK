from __future__ import annotations

from typing import Protocol

from fitness_agents.data.canonical import CanonicalDataset
from fitness_agents.data.specs import DatasetSpec


class DatasetAdapter(Protocol):
    def canonicalize(self) -> CanonicalDataset: ...


def create_adapter(spec: DatasetSpec) -> DatasetAdapter:
    if spec.adapter == "flip_gb1":
        from .flip_gb1 import FlipGB1Adapter

        return FlipGB1Adapter(spec)
    if spec.adapter in {"flip2", "paired_sequence", "generic_sequence"}:
        from .flip2 import Flip2Adapter

        return Flip2Adapter(spec)
    raise ValueError(f"Unknown dataset adapter: {spec.adapter}")

