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
    ExplainVariantOperator,
    HypothesisContextOperator,
    QueryOperator,
)
from .sdk_tools import KGToolSession
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
    "EvidenceSufficiencyPolicy",
    "ExplainVariantOperator",
    "HypothesisContextOperator",
    "InMemoryChangeWriter",
    "InteractionAblationConfig",
    "InteractionResult",
    "KGChangeProposal",
    "KGInteractionController",
    "KGQueryContext",
    "KGQueryPlan",
    "KGQueryStep",
    "KGToolSession",
    "KGUpdateResult",
    "ProposalGateway",
    "QueryIntent",
    "QueryOperator",
    "TrustBoundaryValidator",
]
