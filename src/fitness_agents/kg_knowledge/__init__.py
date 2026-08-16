"""Schema-first construction pipeline for the campaign's external scientific KG."""

from .ablation import KnowledgeAblationConfig
from .adapters import (
    CallableKnowledgeAdapter,
    CampaignObservationAdapter,
    InferenceKnowledgeAdapter,
    KnowledgeAdapter,
    StaticKnowledgeAdapter,
    ValidationKnowledgeAdapter,
)
from .builder import BuildReport, BuildResult, KnowledgeGraphBuilder
from .catalog import DEFAULT_ENTITY_SPECS, DEFAULT_RELATION_SPECS, EntitySpec, RelationSpec
from .fusion import FusionPolicy, ProvenanceAwareFusion
from .normalization import AliasNormalizer, IdentityNormalizer, KnowledgeNormalizer
from .schema import (
    BuildContext,
    EntityRecord,
    KnowledgeBatch,
    KnowledgeGraphSnapshot,
    KnowledgeLayer,
    Modality,
    RelationRecord,
    stable_record_id,
)
from .sinks import InMemoryGraphSink, KnowledgeGraphSink, SQLiteGraphSink
from .validation import CoreSchemaValidator, KnowledgeValidator, ValidationIssue

__all__ = [
    "DEFAULT_ENTITY_SPECS",
    "DEFAULT_RELATION_SPECS",
    "AliasNormalizer",
    "BuildContext",
    "BuildReport",
    "BuildResult",
    "CallableKnowledgeAdapter",
    "CampaignObservationAdapter",
    "CoreSchemaValidator",
    "EntityRecord",
    "EntitySpec",
    "FusionPolicy",
    "IdentityNormalizer",
    "InMemoryGraphSink",
    "InferenceKnowledgeAdapter",
    "KnowledgeAblationConfig",
    "KnowledgeAdapter",
    "KnowledgeBatch",
    "KnowledgeGraphBuilder",
    "KnowledgeGraphSink",
    "KnowledgeGraphSnapshot",
    "KnowledgeLayer",
    "KnowledgeNormalizer",
    "KnowledgeValidator",
    "Modality",
    "ProvenanceAwareFusion",
    "RelationRecord",
    "RelationSpec",
    "SQLiteGraphSink",
    "StaticKnowledgeAdapter",
    "ValidationIssue",
    "ValidationKnowledgeAdapter",
    "stable_record_id",
]
