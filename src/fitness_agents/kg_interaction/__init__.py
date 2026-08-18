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
    FeatureBundleOperator,
    FeatureEvidenceOperator,
    HypothesisContextOperator,
    KGTruncationAuditOperator,
    LocalKnowledgeQueryOperator,
    QueryOperator,
    StructuredClaimQueryOperator,
)
from .tool_runtime import RoundScopedToolExecutor
from .truncation_audit import (
    KeywordTruncationEntry,
    KGKeywordTruncationAuditor,
    KGTruncationAuditReport,
    interaction_item_presence,
    runtime_truncation_audit_payload,
)
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
    "FeatureBundleOperator",
    "FeatureEvidenceOperator",
    "HypothesisContextOperator",
    "InMemoryChangeWriter",
    "InteractionAblationConfig",
    "InteractionResult",
    "KGChangeProposal",
    "KGInteractionController",
    "KGKeywordTruncationAuditor",
    "KGQueryContext",
    "KGQueryPlan",
    "KGQueryStep",
    "KGTruncationAuditOperator",
    "KGTruncationAuditReport",
    "KGUpdateResult",
    "KeywordTruncationEntry",
    "LocalKnowledgeQueryOperator",
    "ProposalGateway",
    "QueryIntent",
    "QueryOperator",
    "RoundScopedToolExecutor",
    "StructuredClaimQueryOperator",
    "TrustBoundaryValidator",
    "interaction_item_presence",
    "runtime_truncation_audit_payload",
]
