from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import KnowledgeLayer, Modality


@dataclass(frozen=True)
class KnowledgeAblationConfig:
    """Independent switches for adapter, layer, modality and schema ablations."""

    enabled_adapters: frozenset[str] | None = None
    enabled_layers: frozenset[KnowledgeLayer] | None = None
    enabled_modalities: frozenset[Modality] | None = None
    enabled_entity_types: frozenset[str] | None = None
    enabled_relation_types: frozenset[str] | None = None

    def adapter_enabled(self, name: str) -> bool:
        return self.enabled_adapters is None or name in self.enabled_adapters

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> KnowledgeAblationConfig:
        layers = value.get("enabled_layers")
        modalities = value.get("enabled_modalities")
        return cls(
            enabled_adapters=(
                frozenset(str(item) for item in value["enabled_adapters"])
                if value.get("enabled_adapters") is not None
                else None
            ),
            enabled_layers=(
                frozenset(KnowledgeLayer(item) for item in layers) if layers is not None else None
            ),
            enabled_modalities=(
                frozenset(Modality(item) for item in modalities) if modalities is not None else None
            ),
            enabled_entity_types=(
                frozenset(str(item) for item in value["enabled_entity_types"])
                if value.get("enabled_entity_types") is not None
                else None
            ),
            enabled_relation_types=(
                frozenset(str(item) for item in value["enabled_relation_types"])
                if value.get("enabled_relation_types") is not None
                else None
            ),
        )
