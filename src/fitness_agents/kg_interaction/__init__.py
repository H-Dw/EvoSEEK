"""Bounded LLM-KG interaction components used by the knowledge-agent campaign path."""

from .ablation import InteractionAblationConfig
from .contracts import (
    ChangeOperation,
    EvidencePack,
    InteractionResult,
    KGChangeProposal,
    KGQueryContext,
    KGQueryPlan,
    KGQueryStep,
    KGUpdateResult,
    QueryIntent,
)
from .controller import EvidenceSufficiencyPolicy, KGInteractionController
from .operators import (
    CallableQueryOperator,
    CompareVariantsOperator,
    EvidenceProvenanceOperator,
    ExplainVariantOperator,
    FeatureEvidenceOperator,
    HypothesisContextOperator,
    LocalKnowledgeQueryOperator,
    QueryOperator,
    StructuredClaimQueryOperator,
)
from .tool_runtime import RoundScopedToolExecutor
from .writeback import (
    InMemoryChangeWriter,
    ProposalGateway,
    TrustBoundaryValidator,
)

__all__ = [
    "CallableQueryOperator",
    "ChangeOperation",
    "CompareVariantsOperator",
    "EvidencePack",
    "EvidenceProvenanceOperator",
    "EvidenceSufficiencyPolicy",
    "ExplainVariantOperator",
    "FeatureEvidenceOperator",
    "HypothesisContextOperator",
    "InMemoryChangeWriter",
    "InteractionAblationConfig",
    "InteractionResult",
    "KGChangeProposal",
    "KGInteractionController",
    "KGQueryContext",
    "KGQueryPlan",
    "KGQueryStep",
    "KGUpdateResult",
    "LocalKnowledgeQueryOperator",
    "ProposalGateway",
    "QueryIntent",
    "QueryOperator",
    "RoundScopedToolExecutor",
    "StructuredClaimQueryOperator",
    "TrustBoundaryValidator",
]
