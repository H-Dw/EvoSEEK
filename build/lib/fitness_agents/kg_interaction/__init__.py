"""Optional, bounded LLM-KG interaction components.

The package is intentionally not wired into the campaign orchestrator by default.
Experiments can opt in one component at a time and keep the current baseline intact.
"""

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
    "KGUpdateResult",
    "ProposalGateway",
    "QueryIntent",
    "QueryOperator",
    "TrustBoundaryValidator",
]
