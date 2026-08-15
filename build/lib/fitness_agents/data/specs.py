from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ComponentSpec:
    component_id: str
    reference: str
    positions: tuple[int, ...] | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ComponentSpec:
        reference = str(raw["reference"]).strip().upper()
        positions_raw = raw.get("positions")
        positions = tuple(int(value) for value in positions_raw) if positions_raw else None
        if positions is not None and len(positions) != len(reference):
            raise ValueError("Component positions must have the same length as its reference")
        return cls(
            component_id=str(raw["component_id"]),
            reference=reference,
            positions=positions,
        )


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    assay_id: str
    adapter: str
    source: Path
    sequence_column: str
    target_column: str
    components: tuple[ComponentSpec, ...]
    variant_column: str | None = None
    model_sequence_column: str | None = None
    mutation_count_column: str | None = None
    backbone_column: str | None = None
    default_backbone_id: str = "WT"
    component_separator: str = ":"
    dataset_scope: str = "source_file"
    target_conflict_tolerance: float = 0.0
    replicate_policy: str = "mean"
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: Path) -> DatasetSpec:
        source = Path(str(raw["source"]))
        if not source.is_absolute():
            candidates = [base_dir / source, *[parent / source for parent in base_dir.parents]]
            source = next((path for path in candidates if path.exists()), candidates[0])
        components = tuple(ComponentSpec.from_mapping(item) for item in raw["components"])
        if not components:
            raise ValueError("Dataset spec requires at least one component")
        return cls(
            dataset_id=str(raw["dataset_id"]),
            assay_id=str(raw["assay_id"]),
            adapter=str(raw.get("adapter", "flip2")),
            source=source.resolve(),
            sequence_column=str(raw["sequence_column"]),
            target_column=str(raw["target_column"]),
            components=components,
            variant_column=(str(raw["variant_column"]) if raw.get("variant_column") else None),
            model_sequence_column=(
                str(raw["model_sequence_column"])
                if raw.get("model_sequence_column")
                else None
            ),
            mutation_count_column=(
                str(raw["mutation_count_column"])
                if raw.get("mutation_count_column")
                else None
            ),
            backbone_column=(
                str(raw["backbone_column"]) if raw.get("backbone_column") else None
            ),
            default_backbone_id=str(raw.get("default_backbone_id", "WT")),
            component_separator=str(raw.get("component_separator", ":")),
            dataset_scope=str(raw.get("dataset_scope", "source_file")),
            target_conflict_tolerance=float(raw.get("target_conflict_tolerance", 0.0)),
            replicate_policy=str(raw.get("replicate_policy", "mean")),
            source_url=(str(raw["source_url"]) if raw.get("source_url") else None),
            metadata=dict(raw.get("metadata", {})),
        )


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    spec_path = Path(path).resolve()
    with spec_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("Dataset spec must be a YAML mapping")
    return DatasetSpec.from_mapping(raw, base_dir=spec_path.parent)
