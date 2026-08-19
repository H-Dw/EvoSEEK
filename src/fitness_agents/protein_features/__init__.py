"""Deterministic, versioned protein feature providers used by KnowledgeEngine."""

from .context import AssayConditions, ProteinTaskContext, StructureResource
from .msa import MSAProfile, MSAProfileProvider
from .physchem import PhyschemDescriptorProvider
from .structure import StaticStructureProvider
from .substitution_store import (
    CANONICAL_RESIDUES,
    STATIC_FEATURE_CHANNELS,
    SubstitutionFeatureStore,
    compact_static_evidence_id,
    compact_structure_site,
)

__all__ = [
    "CANONICAL_RESIDUES",
    "STATIC_FEATURE_CHANNELS",
    "AssayConditions",
    "MSAProfile",
    "MSAProfileProvider",
    "PhyschemDescriptorProvider",
    "ProteinTaskContext",
    "StaticStructureProvider",
    "StructureResource",
    "SubstitutionFeatureStore",
    "compact_static_evidence_id",
    "compact_structure_site",
]
