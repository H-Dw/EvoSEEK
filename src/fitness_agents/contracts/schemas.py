from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class CampaignPhase(str, Enum):
    INITIALIZED = "initialized"
    MODEL_FIT = "model_fit"
    PROPOSED = "proposed"
    HARD_VALIDATED = "hard_validated"
    CRITIQUE_REQUESTED = "critique_requested"
    REVISION_REQUESTED = "revision_requested"
    APPROVED = "approved"
    SELECTED = "selected"
    SUBMITTED = "submitted"
    MEASURED = "measured"
    HYPOTHESIS_EVALUATED = "hypothesis_evaluated"
    ROUND_ABORTED = "round_aborted"
    FINALIZED = "finalized"


class ReviewVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    REJECT = "REJECT"


class IssueScope(str, Enum):
    RESIDUE = "residue"
    INTERACTION = "interaction"
    SEQUENCE = "sequence"
    BATCH = "batch"
    EVIDENCE = "evidence"
    HYPOTHESIS = "hypothesis"
    SYSTEM = "system"


class IssueSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


class FalsificationReadiness(str, Enum):
    READY = "ready"
    NEEDS_REVISION = "needs_revision"
    UNTESTABLE = "untestable"


class RequiredChangeAction(str, Enum):
    EXCLUDE_CANDIDATE = "EXCLUDE_CANDIDATE"
    REPLACE_CANDIDATE = "REPLACE_CANDIDATE"
    REQUEST_EVIDENCE = "REQUEST_EVIDENCE"
    ADD_COUNTEREVIDENCE_SEARCH = "ADD_COUNTEREVIDENCE_SEARCH"
    ADD_CONTROL = "ADD_CONTROL"
    INCREASE_DIVERSITY = "INCREASE_DIVERSITY"
    ADD_EXPLORATION_QUOTA = "ADD_EXPLORATION_QUOTA"
    REDUCE_MUTATION_DEPTH = "REDUCE_MUTATION_DEPTH"
    RELAX_SOFT_PRIOR = "RELAX_SOFT_PRIOR"
    REGENERATE_WITH_CONSTRAINTS = "REGENERATE_WITH_CONSTRAINTS"
    MAKE_FALSIFICATION_EXECUTABLE = "MAKE_FALSIFICATION_EXECUTABLE"
    ABORT_ROUND = "ABORT_ROUND"


class HypothesisStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INCONCLUSIVE = "INCONCLUSIVE"


class CriterionSignal(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICT = "CONTRADICT"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class Variant:
    variant_id: str
    variant: str
    sequence: str
    mutation_notation: str
    mutation_count: int
    split_role: str

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitnessObservation:
    variant_id: str
    fitness: float
    split_role: str
    round_revealed: int
    source: str = "experiment"


@dataclass(frozen=True)
class Prediction:
    variant_id: str
    fitness_mean: float
    fitness_std: float
    interval_90: tuple[float, float]
    ood_score: float
    component_scores: dict[str, float]
    model_version: str
    is_measured: bool = False


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    variant_id: str
    channel: str
    statement: str
    score: float
    source_id: str
    confidence: float
    round_id: int
    evidence_type: str = "computed"


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    statement: str
    preferred_residues: dict[int, tuple[str, ...]]
    evidence_ids: tuple[str, ...]
    expected_outcome: str
    falsification_criterion: str
    parent_hypothesis_id: str | None = None


@dataclass(frozen=True)
class DesignRationale:
    candidate_id: str
    hypothesis_id: str | None
    claim: str
    evidence_ids: tuple[str, ...] = ()
    intended_test: str = ""


@dataclass(frozen=True)
class MutationConflict:
    conflict_id: str
    code: str
    scope: IssueScope
    severity: IssueSeverity
    message: str
    candidate_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    hard: bool = False
    detector: str = ""


@dataclass(frozen=True)
class ConflictReport:
    report_id: str
    round_id: int
    conflicts: tuple[MutationConflict, ...]
    validator_version: str
    input_hash: str

    @property
    def hard_conflicts(self) -> tuple[MutationConflict, ...]:
        return tuple(item for item in self.conflicts if item.hard)


@dataclass(frozen=True)
class FalsificationCriterion:
    criterion_id: str
    detector_name: str
    detector_version: str
    target_variant_ids: tuple[str, ...]
    comparator_variant_ids: tuple[str, ...]
    metric: str
    expected_direction: str
    support_threshold: float
    contradiction_threshold: float
    min_observations: int = 1
    min_replicates: int = 1
    primary: bool = True
    missing_data_policy: str = "INCONCLUSIVE"


@dataclass(frozen=True)
class FalsificationSpec:
    spec_id: str
    hypothesis_id: str
    version: str
    registered_at_round: int
    criteria: tuple[FalsificationCriterion, ...]
    reduction_policy: str
    human_readable_description: str
    pre_registration_hash: str


@dataclass(frozen=True)
class DraftBatch:
    draft_batch_id: str
    parent_draft_batch_id: str | None
    round_id: int
    review_attempt: int
    candidate_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    prediction_snapshot_id: str
    evidence_snapshot_id: str
    acquisition_snapshot_id: str
    design_rationales: tuple[DesignRationale, ...]
    falsification_spec: FalsificationSpec | None
    batch_hash: str


@dataclass(frozen=True)
class CandidateIssue:
    issue_id: str
    candidate_id: str
    scope: IssueScope
    severity: IssueSeverity
    code: str
    claim: str
    evidence_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    suggested_action: RequiredChangeAction | None = None


@dataclass(frozen=True)
class BatchRisk:
    risk_id: str
    code: str
    severity: IssueSeverity
    statement: str
    candidate_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceConflict:
    conflict_id: str
    topic: str
    supporting_ids: tuple[str, ...]
    opposing_ids: tuple[str, ...]
    source_independence: str
    unresolved_reason: str
    impact: str


@dataclass(frozen=True)
class UnsupportedClaim:
    claim_id: str
    claim: str
    reason: str
    missing_evidence_type: str
    required_action: RequiredChangeAction


@dataclass(frozen=True)
class RequiredChange:
    action: RequiredChangeAction
    target_ids: tuple[str, ...]
    parameters: dict[str, Any]
    rationale: str
    evidence_ids: tuple[str, ...] = ()
    priority: int = 1


@dataclass(frozen=True)
class CritiqueDecision:
    decision_id: str
    draft_batch_id: str
    round_id: int
    review_attempt: int
    verdict: ReviewVerdict
    falsification_readiness: FalsificationReadiness
    candidate_issues: tuple[CandidateIssue, ...] = ()
    batch_level_risks: tuple[BatchRisk, ...] = ()
    evidence_conflicts: tuple[EvidenceConflict, ...] = ()
    unsupported_claims: tuple[UnsupportedClaim, ...] = ()
    required_changes: tuple[RequiredChange, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    summary: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("CritiqueDecision confidence must be in [0, 1]")


@dataclass(frozen=True)
class ApprovedBatch:
    draft_batch_id: str
    round_id: int
    candidate_ids: tuple[str, ...]
    batch_hash: str
    hard_validation_report_id: str
    critique_decision_id: str
    approval_policy_version: str
    approval_receipt_hash: str


@dataclass(frozen=True)
class CriterionResult:
    criterion_id: str
    signal: CriterionSignal
    metric_value: float | None
    comparator_value: float | None
    effect_size: float | None
    observation_ids: tuple[str, ...]
    qc_status: str
    detector_name: str
    detector_version: str
    reason_code: str


@dataclass(frozen=True)
class HypothesisAssessment:
    assessment_id: str
    hypothesis_id: str
    falsification_spec_id: str
    round_id: int
    status: HypothesisStatus
    criterion_results: tuple[CriterionResult, ...]
    observation_ids: tuple[str, ...]
    decisive_criterion_ids: tuple[str, ...]
    unresolved_criterion_ids: tuple[str, ...]
    evaluator_version: str
    assessment_hash: str


@dataclass(frozen=True)
class SelectionRecord:
    variant_id: str
    round_id: int
    selection_order: int
    model_rank_all: int
    acquisition_rank_all: int
    eligible_rank: int
    total_candidates: int
    eligible_candidates: int
    fitness_mean: float
    fitness_std: float
    acquisition_score: float
    knowledge_score: float
    evidence_ids: tuple[str, ...]
    hypothesis_id: str | None
    reason: str
    intervention_tags: tuple[str, ...] = ()


@dataclass
class CampaignState:
    run_id: str
    mode: str
    seed: int
    round_id: int = 0
    phase: CampaignPhase = CampaignPhase.INITIALIZED
    observed: list[FitnessObservation] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    selections: list[SelectionRecord] = field(default_factory=list)
    critique_decisions: list[CritiqueDecision] = field(default_factory=list)
    hypothesis_assessments: list[HypothesisAssessment] = field(default_factory=list)
    approved_batch_ids: list[str] = field(default_factory=list)
    revealed_variant_ids: set[str] = field(default_factory=set)
    final_test_opened: bool = False

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        def convert(item: Any) -> Any:
            if isinstance(item, Enum):
                return item.value
            if isinstance(item, dict):
                return {key: convert(entry) for key, entry in item.items()}
            if isinstance(item, (list, tuple, set)):
                return [convert(entry) for entry in item]
            return item

        value = convert(value)
        value["revealed_variant_ids"] = sorted(self.revealed_variant_ids)
        return value
