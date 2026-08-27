from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .attestation import HMACAttestation
from .canonical import content_sha256


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _require_timezone(value: datetime | None) -> datetime | None:
    if value is not None and (
        value.tzinfo is None or value.utcoffset() is None
    ):
        raise ValueError("Deep Research timestamps must include a timezone")
    return value


class SearchRoute(str, Enum):
    LANDSCAPE = "landscape"
    PRIMARY = "primary_evidence"
    COUNTEREVIDENCE = "counterevidence"
    BOUNDARY = "scope_boundary"
    REPLICATION = "replication"
    METADATA_VERIFY = "metadata_verify"


class DecisionPermission(str, Enum):
    EXPLANATION_ONLY = "explanation_only"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    CANDIDATE_RERANKING = "candidate_reranking"
    HARD_GATE = "hard_gate"


class SubjectScope(str, Enum):
    GENERIC_PROTEIN = "generic_protein"
    NONVIRAL_PROTEIN = "nonviral_protein"
    VIRAL_PROTEIN = "viral_protein"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SubjectRole(str, Enum):
    PRIMARY_SUBJECT = "primary_subject"
    EXPERIMENTAL_SYSTEM = "experimental_system"
    OPERATIONAL_METHOD = "operational_method"
    INCIDENTAL_MENTION = "incidental_mention"


class ScopeAssertion(StrictModel):
    schema_version: Literal["external-scope-assertion:v1"] = (
        "external-scope-assertion:v1"
    )
    scope_assertion_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    subject_scope: SubjectScope
    excluded_subject_present: Literal[False]
    roles: tuple[SubjectRole, ...] = Field(min_length=1)
    assertion_status: Literal["verified", "unverified", "conflict"]
    issuer: str = Field(min_length=1)
    issuer_kind: Literal[
        "human_review",
        "curated_registry",
        "authoritative_taxonomy",
    ]
    source_record_id: str = Field(min_length=1)
    verification_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewed_at: datetime
    expires_at: datetime | None = None
    review_receipt_ids: tuple[str, ...] = Field(min_length=1)

    _validate_times = field_validator("reviewed_at", "expires_at")(
        _require_timezone
    )

    @model_validator(mode="after")
    def require_primary_subject_for_allowed_scope(self) -> ScopeAssertion:
        if self.expires_at is not None and self.expires_at <= self.reviewed_at:
            raise ValueError("ScopeAssertion expires_at must follow reviewed_at")
        if self.subject_scope in {
            SubjectScope.GENERIC_PROTEIN,
            SubjectScope.NONVIRAL_PROTEIN,
        } and SubjectRole.PRIMARY_SUBJECT not in self.roles:
            raise ValueError(
                "Generic/nonviral allow assertions require primary_subject scope"
            )
        return self

    @property
    def assertion_hash(self) -> str:
        return content_sha256(self)


class PolicyReceipt(StrictModel):
    schema_version: Literal["external-policy-receipt:v2"] = (
        "external-policy-receipt:v2"
    )
    policy_version: str
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["allowed", "denied", "quarantined"]
    matched_categories: tuple[str, ...] = ()
    stage: str
    subject_id: str
    subject_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    issuer: Literal["external_evidence_scope_policy"] = (
        "external_evidence_scope_policy"
    )
    attestation: HMACAttestation

    _validate_issued_at = field_validator("issued_at")(_require_timezone)


class ReviewReceipt(StrictModel):
    schema_version: Literal["evidence-review-receipt:v2"] = (
        "evidence-review-receipt:v2"
    )
    review_receipt_id: str
    review_type: Literal[
        "metadata_identity",
        "full_text_scope",
        "source_span_resolution",
        "independence_grouping",
        "claim_entailment",
        "task_applicability",
        "decision_permission",
        "scope_assertion",
    ]
    record_id: str
    reviewer_id: str
    reviewer_kind: Literal["human", "model_assisted", "deterministic_rule"]
    method_version: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["passed", "failed", "escalated"]
    reviewed_at: datetime
    expires_at: datetime
    model_fingerprint: str | None = None
    prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    attestation: HMACAttestation

    _validate_times = field_validator("reviewed_at", "expires_at")(
        _require_timezone
    )

    @model_validator(mode="after")
    def require_positive_validity_window(self) -> ReviewReceipt:
        if self.expires_at <= self.reviewed_at:
            raise ValueError("ReviewReceipt expires_at must follow reviewed_at")
        return self


class ReleaseApprovalReceipt(StrictModel):
    schema_version: Literal["evidence-release-approval:v2"] = (
        "evidence-release-approval:v2"
    )
    approval_id: str
    reviewer_id: str
    reviewer_kind: Literal["human"] = "human"
    method_version: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_artifact_id: str = Field(min_length=1)
    approval_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["approved", "rejected"]
    approved_at: datetime
    target_release_version: str = Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$"
    )
    target_status: Literal["released"] = "released"
    target_created_at: datetime
    target_parent_release_id: str | None = None
    attestation: HMACAttestation

    _validate_times = field_validator("approved_at", "target_created_at")(
        _require_timezone
    )


class SearchBudget(StrictModel):
    max_queries: int = Field(default=12, ge=1, le=100)
    max_results_per_query: int = Field(default=20, ge=1, le=100)
    max_publications: int = Field(default=100, ge=1, le=1000)


class QuestionLeaf(StrictModel):
    question_leaf_id: str
    decision_slot: Literal[
        "mechanism",
        "transfer",
        "measurement",
        "interaction",
        "uncertainty",
        "failure_mode",
    ]
    question: str
    context_signature: dict[str, str] = Field(default_factory=dict)
    required_evidence_types: tuple[str, ...] = ("primary_research",)
    must_find: tuple[str, ...] = ()
    counterevidence_question: str
    closure_rule: str
    priority: int = Field(default=1, ge=1, le=5)


class ResearchBrief(StrictModel):
    schema_version: Literal["deep-research-brief:v1"] = "deep-research-brief:v1"
    brief_id: str
    research_question: str
    decision_use: Literal[
        "mechanism_explanation",
        "candidate_ranking",
        "campaign_strategy",
    ]
    question_tree: tuple[QuestionLeaf, ...]
    inclusion_criteria: tuple[str, ...]
    exclusion_criteria: tuple[str, ...]
    required_source_types: tuple[str, ...] = ("primary_research",)
    required_search_routes: tuple[SearchRoute, ...] = (
        SearchRoute.LANDSCAPE,
        SearchRoute.PRIMARY,
        SearchRoute.COUNTEREVIDENCE,
        SearchRoute.BOUNDARY,
        SearchRoute.REPLICATION,
    )
    non_viral_only: Literal[True] = True
    excluded_source_scopes: tuple[Literal["viral_protein"], ...] = (
        "viral_protein",
    )
    publication_cutoff: str | None = None
    source_quarantine_ids: tuple[str, ...] = ()
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget: SearchBudget = Field(default_factory=SearchBudget)
    stop_conditions: tuple[str, ...]

    @model_validator(mode="after")
    def require_counterevidence_route(self) -> ResearchBrief:
        if SearchRoute.COUNTEREVIDENCE not in self.required_search_routes:
            raise ValueError("Deep Research requires an explicit counterevidence route")
        if not self.question_tree:
            raise ValueError("ResearchBrief requires at least one QuestionLeaf")
        if "viral_protein" not in self.excluded_source_scopes:
            raise ValueError(
                "ResearchBrief must explicitly exclude the viral_protein scope"
            )
        required_query_count = len(self.question_tree) * len(
            set(self.required_search_routes)
        )
        if self.budget.max_queries < required_query_count:
            raise ValueError(
                "Search budget cannot cover every QuestionLeaf and required route"
            )
        return self


class PlannedQuery(StrictModel):
    query_id: str
    question_leaf_id: str
    route: SearchRoute
    query: str
    filters: dict[str, str] = Field(default_factory=dict)
    policy_receipt: PolicyReceipt


class SearchRun(StrictModel):
    schema_version: Literal["deep-search-run:v1"] = "deep-search-run:v1"
    search_run_id: str
    brief_id: str
    planned_query_id: str
    question_leaf_id: str
    provider: str
    route: SearchRoute
    exact_query: str
    exact_query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filters: dict[str, str] = Field(default_factory=dict)
    executed_at: datetime
    result_ids: tuple[str, ...]
    accepted_result_ids: tuple[str, ...]
    excluded_result_count: int = Field(ge=0)
    duplicate_result_count: int = Field(ge=0)
    # Zero records a provider preflight failure (for example, a missing API
    # credential) where no network request was made.
    attempt_count: int = Field(default=1, ge=0, le=10)
    error_code: str | None = None
    stop_reason: Literal[
        "completed",
        "budget_exhausted",
        "no_results",
        "policy_denied",
        "provider_error",
    ]
    snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_receipt: PolicyReceipt

    _validate_executed_at = field_validator("executed_at")(_require_timezone)


class AllowedSearchHit(StrictModel):
    """Allowed metadata observation retained as a first-class audit record."""

    schema_version: Literal["allowed-search-hit:v1"] = "allowed-search-hit:v1"
    search_hit_id: str
    search_run_id: str
    result_id: str
    disposition: Literal["accepted", "duplicate_publication"]
    artifact_id: str
    provider: str
    provider_record_id: str
    title: str
    abstract: str | None = None
    authors: tuple[str, ...]
    year: int | None = Field(default=None, ge=1600, le=2200)
    venue: str | None = None
    doi: str | None = None
    url: str
    publication_type: str
    subjects: tuple[str, ...] = ()
    provider_score: float | None = None
    retrieval_score: float = Field(ge=0.0, le=1.0)
    retrieval_components: dict[str, float] = Field(default_factory=dict)
    scope_assertion_id: str
    policy_receipt: PolicyReceipt


class PublicationAcquisition(StrictModel):
    """Auditable link to one retained allowed hit, including duplicate observations."""

    search_hit_id: str


class Publication(StrictModel):
    schema_version: Literal["scientific-publication:v3"] = "scientific-publication:v3"
    publication_id: str
    identifier_aliases: tuple[str, ...] = ()
    title: str
    authors: tuple[str, ...]
    year: int = Field(ge=1600, le=2200)
    venue: str
    doi: str | None = None
    canonical_url: str
    publication_type: str
    study_family_id: str
    version_status: Literal[
        "active",
        "corrected",
        "retracted",
        "expression_of_concern",
    ] = "active"
    metadata_verified: bool = False
    full_text_status: Literal["verified", "unavailable", "metadata_only"] = "metadata_only"
    canonical_search_hit_id: str
    acquisitions: tuple[PublicationAcquisition, ...] = Field(min_length=1)
    source_scope: Literal["generic_protein", "nonviral_protein"]
    scope_assertion_id: str
    review_receipt_ids: tuple[str, ...] = ()


class SourceSpan(StrictModel):
    schema_version: Literal["scientific-source-span:v1"] = "scientific-source-span:v1"
    source_span_id: str
    publication_id: str
    artifact_id: str
    artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str
    normalized_span_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    span_text: str | None = None
    support_paraphrase: str
    evidence_role: Literal[
        "result",
        "method",
        "definition",
        "limitation",
        "negative_result",
    ]
    language: Literal["en"] = "en"
    extraction_method: Literal["native_text", "ocr", "manual"]
    resolved_against_artifact: bool
    independently_checked: bool
    instruction_markers: tuple[str, ...] = ()
    scope_assertion_id: str
    review_receipt_ids: tuple[str, ...] = ()


class EvidenceGroup(StrictModel):
    schema_version: Literal["scientific-evidence-group:v1"] = "scientific-evidence-group:v1"
    evidence_group_id: str
    source_span_ids: tuple[str, ...]
    stance: Literal["supports", "refutes", "limits", "unknown"]
    completeness: Literal["complete", "partial", "ambiguous"]
    independence_group: str
    grouping_rationale: str
    verified_by: tuple[str, ...]
    review_receipt_ids: tuple[str, ...] = ()


class AtomicClaim(StrictModel):
    schema_version: Literal["scientific-atomic-claim:v2"] = "scientific-atomic-claim:v2"
    claim_id: str
    statement: str
    subject: str
    predicate: str
    object: str
    claim_kind: Literal[
        "scientific_prior",
        "evidence_informed_policy",
        "operational_guideline",
    ]
    knowledge_type: str = Field(pattern=r"^[a-z][a-z0-9_]{1,63}$")
    applicability: dict[str, Any]
    evidence_group_ids: tuple[str, ...]
    claim_status: Literal["supported", "contested", "insufficient"]
    language: Literal["en"] = "en"


class ScientificQuality(StrictModel):
    identity_verified: bool
    span_verified: bool
    entailment_status: Literal["verified", "failed", "unverified"]
    source_credibility: float = Field(ge=0.0, le=1.0)
    independent_support_count: int = Field(ge=0)
    counterevidence_status: Literal["searched_found", "searched_none", "not_searched"]
    conflict_status: Literal["none", "resolved", "unresolved"]
    uncertainty: float = Field(ge=0.0, le=1.0)


class TaskApplicability(StrictModel):
    directness: Literal["direct", "analogical", "out_of_scope", "unknown"]
    context_match: float = Field(ge=0.0, le=1.0)
    candidate_discriminative_value: float = Field(ge=0.0, le=1.0)
    matched_dimensions: tuple[str, ...] = ()
    unmatched_dimensions: tuple[str, ...] = ()
    boundary_conditions: tuple[str, ...] = ()


class LogicUnit(StrictModel):
    schema_version: Literal["scientific-logic-unit:v1"] = "scientific-logic-unit:v1"
    logic_unit_id: str
    question_leaf_id: str
    task_route: Literal[
        "mechanism_explanation",
        "candidate_ranking",
        "campaign_strategy",
    ]
    subquestion: str
    premise_claim_ids: tuple[str, ...]
    counterclaim_ids: tuple[str, ...] = ()
    search_coverage_run_ids: tuple[str, ...] = Field(min_length=1)
    operator: Literal[
        "qualify",
        "compare",
        "constraint",
        "causal_chain",
        "conflict_summary",
        "abstain",
    ]
    conclusion: str
    applicability_tests: tuple[str, ...]
    falsifiers: tuple[str, ...]
    abstain_if: tuple[str, ...]
    retrieval_text: str
    scientific_quality: ScientificQuality
    task_applicability: TaskApplicability
    review_receipt_ids: tuple[str, ...] = ()


class KnowledgeDecisionCard(StrictModel):
    schema_version: Literal["knowledge-decision-card:v1"] = "knowledge-decision-card:v1"
    decision_card_id: str
    question_leaf_id: str
    task_route: Literal[
        "mechanism_explanation",
        "candidate_ranking",
        "campaign_strategy",
    ]
    logic_unit_ids: tuple[str, ...]
    required_inputs: tuple[str, ...]
    candidate_feature: str | None = None
    expected_direction: Literal[
        "positive",
        "negative",
        "non_monotonic",
        "unknown",
    ] = "unknown"
    interaction_order: int = Field(default=1, ge=1)
    boundary_conditions: tuple[str, ...]
    uncertainty: float = Field(ge=0.0, le=1.0)
    permission: DecisionPermission = DecisionPermission.EXPLANATION_ONLY
    calibration_id: str | None = None
    calibration_status: Literal["none", "draft", "validated"] = "none"
    benchmark_overlap_status: Literal["clear", "quarantined", "unknown"] = "unknown"
    abstain_if: tuple[str, ...]
    human_approval_ids: tuple[str, ...] = ()
    review_receipt_ids: tuple[str, ...] = ()


class ReleaseRecord(StrictModel):
    record_id: str
    record_type: Literal[
        "research_brief",
        "search_run",
        "scope_assertion",
        "allowed_search_hit",
        "publication",
        "source_span",
        "evidence_group",
        "atomic_claim",
        "logic_unit",
        "knowledge_decision_card",
        "review_receipt",
        "release_approval",
    ]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_ids: tuple[str, ...] = ()


class ReleaseManifest(StrictModel):
    schema_version: Literal["scientific-knowledge-release:v2"] = (
        "scientific-knowledge-release:v2"
    )
    release_id: str
    release_version: str = Field(
        pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$"
    )
    status: Literal["candidate", "released", "superseded", "revoked"]
    created_at: datetime
    parent_release_id: str | None = None
    policy_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validator_version: str
    records: tuple[ReleaseRecord, ...]
    dependency_graph_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_record_types: tuple[str, ...] = (
        "atomic_claim",
        "logic_unit",
        "knowledge_decision_card",
    )
    denied_path_operations: Literal[0] = 0
    excluded_result_count: int = Field(ge=0)
    release_approval_ids: tuple[str, ...] = ()

    _validate_created_at = field_validator("created_at")(_require_timezone)


class EvidenceProductBundle(StrictModel):
    schema_version: Literal["scientific-evidence-product:v2"] = (
        "scientific-evidence-product:v2"
    )
    research_brief: ResearchBrief
    search_runs: tuple[SearchRun, ...]
    scope_assertions: tuple[ScopeAssertion, ...]
    allowed_search_hits: tuple[AllowedSearchHit, ...]
    publications: tuple[Publication, ...]
    source_spans: tuple[SourceSpan, ...]
    evidence_groups: tuple[EvidenceGroup, ...]
    atomic_claims: tuple[AtomicClaim, ...]
    logic_units: tuple[LogicUnit, ...]
    decision_cards: tuple[KnowledgeDecisionCard, ...]
    review_receipts: tuple[ReviewReceipt, ...] = ()
    release_approvals: tuple[ReleaseApprovalReceipt, ...] = ()
    release_manifest: ReleaseManifest | None = None
