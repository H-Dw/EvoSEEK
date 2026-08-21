"""Typed contracts for the isolated, hierarchical hypothesis review graph."""

from __future__ import annotations

from typing import Annotated, Any, Literal, get_args

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from fitness_agents.contracts.evidence_universe import RoleVisibleEvidenceUniverse

ChannelName = Literal["physchem", "conservation", "structure"]
ReviewVerdictName = Literal["APPROVE", "REVISE", "REJECT"]


def verdict_for_rating(score: int) -> ReviewVerdictName:
    """Map the model's fixed Rating region to the only legal downstream action."""

    if score < 2:
        return "REJECT"
    if score < 4:
        return "REVISE"
    return "APPROVE"


COUPLED_REVIEW_CONTRACT = (
    "rating.score, verdict, and required_changes are one legal object. "
    "Score 0-1 maps to REJECT with required_changes=[]. "
    "Score 2-3 maps to REVISE with at least one allow-listed required_changes item "
    "and at least one rating.suggestions entry. "
    "Score 4-5 maps to APPROVE with required_changes=[], no blocker issues, and empty "
    "text_errors. Non-empty text_errors caps the score at 3. "
    "rating.suggestions is free-text repair advice and is not a substitute for "
    "required_changes. On a schema retry, keep existing suggestions and emit matching "
    "allow-listed actions; repair verdict, rating, and required_changes together."
)

# LLM prose caps. Sized from hierarchical Main Critic production output
# (explanation 674, rationale 558, hypothesis statement ~380) with headroom
# for three-channel synthesis reviews. Identifiers stay on their own limits.
CRITIC_EXPLANATION_MAX = 2000
CRITIC_RATIONALE_MAX = 1200
CRITIC_SUGGESTION_MAX = 600
CRITIC_TEXT_ERROR_MAX = 400
CRITIC_ISSUE_MESSAGE_MAX = 800
CRITIC_SUMMARY_MAX = 800
CRITIC_NESTED_TEXT_MAX = 400
SAMPLE_REVIEW_PROSE_MAX = 600
CHANNEL_FINDING_STATEMENT_MAX = 600
CHANNEL_ANALYSIS_PROSE_MAX = 800
PHYSCHEM_SUMMARY_MAX = 600
PHYSCHEM_INTERPRETATION_PROSE_MAX = 480
CANDIDATE_PROSE_MAX = 800
SYNTHESIS_CARD_STATEMENT_MAX = 1200


class CriticRatingRegion(BaseModel):
    """Shared, model-visible Rating region used by every semantic Critic."""

    model_config = ConfigDict(extra="forbid", strict=True)

    score: int = Field(ge=0, le=5)
    rationale: str = Field(min_length=1, max_length=CRITIC_RATIONALE_MAX)
    suggestions: list[Annotated[str, Field(min_length=1, max_length=CRITIC_SUGGESTION_MAX)]] = Field(
        default_factory=list, max_length=12
    )
    text_errors: list[Annotated[str, Field(min_length=1, max_length=CRITIC_TEXT_ERROR_MAX)]] = Field(
        default_factory=list, max_length=8
    )

    @model_validator(mode="after")
    def validate_rating_band(self) -> CriticRatingRegion:
        if 2 <= self.score <= 3 and not self.suggestions:
            raise ValueError("Rating 2-3 requires at least one actionable suggestion")
        if self.score >= 4 and self.text_errors:
            raise ValueError("Rating 4-5 is forbidden while declared text errors remain")
        return self

PhyschemIssueCode = Literal[
    "ANALYSIS_SCOPE_OVERREACH",
    "FINDING_UNSUPPORTED",
    "OBSERVATION_HYPOTHESIS_CONFLATED",
    "COUNTEREVIDENCE_IGNORED",
    "OVERCONFIDENT",
    "UNTESTABLE_CANDIDATE",
    "RESIDUE_DIRECTION_UNSUPPORTED",
]
ConservationIssueCode = Literal[
    "ANALYSIS_SCOPE_OVERREACH",
    "FINDING_UNSUPPORTED",
    "OBSERVATION_HYPOTHESIS_CONFLATED",
    "COUNTEREVIDENCE_IGNORED",
    "OVERCONFIDENT",
    "UNTESTABLE_CANDIDATE",
    "COVERAGE_INSUFFICIENT",
    "NEFF_INSUFFICIENT",
    "PAIRWISE_INELIGIBLE",
]
StructureIssueCode = Literal[
    "ANALYSIS_SCOPE_OVERREACH",
    "FINDING_UNSUPPORTED",
    "OBSERVATION_HYPOTHESIS_CONFLATED",
    "COUNTEREVIDENCE_IGNORED",
    "OVERCONFIDENT",
    "UNTESTABLE_CANDIDATE",
    "COORDINATES_MISSING",
    "STATIC_STRUCTURE_LIMIT",
    "DYNAMICS_OVERREACH",
]
MainIssueCode = Literal[
    "EXPLANATION_MISSING",
    "CROSS_CHANNEL_CONFLICT",
    "UNSUPPORTED_SYNTHESIS",
    "COUNTEREVIDENCE_IGNORED",
    "OVERCONFIDENT",
    "UNTESTABLE",
]

PhyschemRequiredAction = Literal[
    "NARROW_ANALYSIS",
    "ADD_EVIDENCE_LINK",
    "SEPARATE_OBSERVATION_FROM_HYPOTHESIS",
    "ADD_COUNTEREVIDENCE",
    "LOWER_CONFIDENCE",
    "MAKE_CANDIDATE_FALSIFIABLE",
    "REMOVE_FITNESS_INFERENCE",
]
ConservationRequiredAction = Literal[
    "NARROW_ANALYSIS",
    "ADD_EVIDENCE_LINK",
    "SEPARATE_OBSERVATION_FROM_HYPOTHESIS",
    "ADD_COUNTEREVIDENCE",
    "LOWER_CONFIDENCE",
    "MAKE_CANDIDATE_FALSIFIABLE",
    "REPORT_COVERAGE",
    "REPORT_NEFF",
    "REMOVE_PAIRWISE_CLAIM",
]
StructureRequiredAction = Literal[
    "NARROW_ANALYSIS",
    "ADD_EVIDENCE_LINK",
    "SEPARATE_OBSERVATION_FROM_HYPOTHESIS",
    "ADD_COUNTEREVIDENCE",
    "LOWER_CONFIDENCE",
    "MAKE_CANDIDATE_FALSIFIABLE",
    "ACKNOWLEDGE_MISSING_COORDINATES",
    "LIMIT_TO_STATIC_STRUCTURE",
    "REMOVE_DYNAMICS_CLAIM",
]
MainRequiredAction = Literal[
    "NARROW_CLAIM",
    "ADD_COUNTEREVIDENCE",
    "LOWER_CONFIDENCE",
    "MAKE_FALSIFIABLE",
    "ADD_EXPLANATION",
    "RESOLVE_CHANNEL_CONFLICT",
]


def required_actions_for_review(
    scope: Literal["main", "physchem", "conservation", "structure"],
) -> tuple[str, ...]:
    """Return the allow-listed required_changes enums for one Critic body."""

    mapping = {
        "main": MainRequiredAction,
        "physchem": PhyschemRequiredAction,
        "conservation": ConservationRequiredAction,
        "structure": StructureRequiredAction,
    }
    return get_args(mapping[scope])


class DescriptorObservationFact(BaseModel):
    """One sample-local descriptor delta with an explicit mutation identity."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    fact_id: str = Field(min_length=1, max_length=320)
    evidence_id: str = Field(min_length=1, max_length=320)
    sample_id: str = Field(min_length=1, max_length=320)
    position: int = Field(gt=0)
    from_residue: str = Field(
        min_length=1,
        max_length=1,
        pattern=r"^[ACDEFGHIKLMNPQRSTVWY]$",
    )
    to_residue: str = Field(
        min_length=1,
        max_length=1,
        pattern=r"^[ACDEFGHIKLMNPQRSTVWY]$",
    )
    descriptor: str = Field(min_length=1, max_length=120)
    delta: float


class ChildSampleCard(BaseModel):
    """Fitness-blind sample view supplied to one feature child."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    sample_id: str = Field(min_length=1, max_length=320)
    mutation_notation: str = Field(min_length=1, max_length=240)
    residues_by_position: dict[str, str]
    evidence_ids: tuple[str, ...] = Field(max_length=16)
    feature_values: dict[str, dict[str, Any]]
    descriptor_facts: tuple[DescriptorObservationFact, ...] = Field(
        default=(), max_length=64
    )


class ChannelEvidenceInput(BaseModel):
    """The complete and exclusive prompt payload for one child Scientist."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_id: str
    round_id: int = Field(ge=0)
    channel: ChannelName
    task: str
    mutable_positions: tuple[int, ...]
    wild_type_sites: str
    sample_map: dict[str, str] = Field(default_factory=dict)
    visible_observations: tuple[ChildSampleCard, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    kg_packs: tuple[dict[str, Any], ...] = ()
    retry_control: dict[str, Any] | None = None

    @field_validator(
        "mutable_positions", "visible_observations", "evidence", "kg_packs", mode="before"
    )
    @classmethod
    def normalize_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def enforce_channel_isolation(self) -> ChannelEvidenceInput:
        if len(self.mutable_positions) != len(self.wild_type_sites):
            raise ValueError("wild_type_sites must align with mutable_positions")
        foreign = {
            str(item.get("channel"))
            for item in self.evidence
            if item.get("channel") not in {None, self.channel}
        }
        for pack in self.kg_packs:
            for item in pack.get("evidence", ()):
                if isinstance(item, dict) and item.get("channel") not in {None, self.channel}:
                    foreign.add(str(item.get("channel")))
        if foreign:
            raise ValueError(
                f"{self.channel} child context contains foreign channels: {sorted(foreign)}"
            )
        return self

    @property
    def visible_evidence_ids(self) -> frozenset[str]:
        from fitness_agents.contracts.evidence_universe import (
            RoleVisibleEvidenceUniverse,
        )

        interaction = {"packs": self.kg_packs}
        return RoleVisibleEvidenceUniverse.from_role_sources(
            role=f"subscientist:{self.channel}",
            evidence=self.evidence,
            interaction=interaction,
        ).ids

    @property
    def descriptor_fact_by_id(self) -> dict[str, DescriptorObservationFact]:
        facts = {
            fact.fact_id: fact
            for sample in self.visible_observations
            for fact in sample.descriptor_facts
        }
        if len(facts) != sum(
            len(sample.descriptor_facts) for sample in self.visible_observations
        ):
            raise ValueError("descriptor fact IDs must be unique in one child context")
        return facts


class ChannelFinding(BaseModel):
    """One tool-grounded observation or bounded interpretation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    finding_id: str = Field(min_length=1, max_length=160)
    kind: Literal["OBSERVATION", "INTERPRETATION", "LIMITATION"]
    statement: str = Field(min_length=1, max_length=CHANNEL_FINDING_STATEMENT_MAX)
    evidence_ids: list[str] = Field(max_length=8)
    fact_ids: list[str] = Field(default_factory=list, max_length=8)
    confidence: Literal["low", "medium", "high"]


class PhyschemInterpretationOutput(BaseModel):
    """Bounded interpretation plus request-local sample/evidence/fact references."""

    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_summary: str = Field(min_length=1, max_length=PHYSCHEM_SUMMARY_MAX)
    interpretations: list[Annotated[str, Field(min_length=1, max_length=PHYSCHEM_INTERPRETATION_PROSE_MAX)]] = Field(
        default_factory=list, max_length=8
    )
    counterevidence: list[Annotated[str, Field(min_length=1, max_length=PHYSCHEM_INTERPRETATION_PROSE_MAX)]] = Field(
        default_factory=list, max_length=4
    )
    uncertainty: str = Field(min_length=1, max_length=PHYSCHEM_SUMMARY_MAX)
    sample_ids: list[str] = Field(default_factory=list, max_length=16)
    evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    fact_ids: list[str] = Field(default_factory=list, max_length=32)


class ChannelCandidateHypothesis(BaseModel):
    """Optional downstream hypothesis; an analysis may legitimately contain none."""

    model_config = ConfigDict(extra="forbid", strict=True)

    hypothesis_id: str = Field(min_length=1, max_length=160)
    statement: str = Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)
    proposed_residues: dict[str, list[str]] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(max_length=8)
    expected_observation: str = Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)
    falsification_criterion: str = Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)

    @model_validator(mode="after")
    def validate_residues(self) -> ChannelCandidateHypothesis:
        canonical = set("ACDEFGHIKLMNPQRSTVWY")
        for position, residues in self.proposed_residues.items():
            int(position)
            if not residues or any(item not in canonical for item in residues):
                raise ValueError("proposed_residues must use canonical one-letter residues")
        return self


class ChannelAnalysisOutput(BaseModel):
    """Analysis-first child output for one isolated tool channel."""

    model_config = ConfigDict(extra="forbid", strict=True)

    analysis_id: str = Field(min_length=1, max_length=160)
    channel: ChannelName
    analysis_summary: str = Field(min_length=1, max_length=CHANNEL_ANALYSIS_PROSE_MAX)
    findings: list[ChannelFinding] = Field(min_length=1, max_length=8)
    candidate_hypotheses: list[ChannelCandidateHypothesis] = Field(
        default_factory=list, max_length=4
    )
    evidence_ids: list[str] = Field(max_length=12)
    fact_ids: list[str] = Field(default_factory=list, max_length=24)
    counterevidence: list[Annotated[str, Field(min_length=1, max_length=CHANNEL_ANALYSIS_PROSE_MAX)]] = Field(
        max_length=8
    )
    uncertainty: str = Field(min_length=1, max_length=CHANNEL_ANALYSIS_PROSE_MAX)

    @model_validator(mode="after")
    def validate_citations(self) -> ChannelAnalysisOutput:
        declared = set(self.evidence_ids)
        nested = {
            evidence_id
            for item in [*self.findings, *self.candidate_hypotheses]
            for evidence_id in item.evidence_ids
        }
        if nested != declared:
            raise ValueError(
                "top-level evidence_ids must exactly equal the sorted unique nested ID union"
            )
        if self.evidence_ids != sorted(declared):
            raise ValueError("top-level evidence_ids must be sorted and unique")
        declared_facts = set(self.fact_ids)
        nested_facts = {
            fact_id for finding in self.findings for fact_id in finding.fact_ids
        }
        if nested_facts != declared_facts:
            raise ValueError(
                "top-level fact_ids must exactly equal the sorted unique finding fact ID union"
            )
        if self.fact_ids != sorted(declared_facts):
            raise ValueError("top-level fact_ids must be sorted and unique")
        return self


class ChannelAnalysisBatchArtifact(BaseModel):
    """Runtime-owned coverage receipt for one child-Scientist sample batch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    batch_id: str = Field(min_length=1, max_length=160)
    split_depth: int = Field(ge=0)
    sample_ids: tuple[str, ...] = Field(min_length=1)
    input_receipt_id: str = Field(min_length=1, max_length=160)
    output_receipt_id: str = Field(min_length=1, max_length=160)
    evidence_universe: RoleVisibleEvidenceUniverse
    analysis: ChannelAnalysisOutput
    input_chars: int | None = Field(default=None, ge=0)
    request_started: bool = False

    @model_validator(mode="after")
    def validate_batch_coverage(self) -> ChannelAnalysisBatchArtifact:
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("sample_ids must be unique within a child batch")
        return self


class BatchedChannelAnalysisResult(BaseModel):
    """Bounded child analysis for review plus complete typed batch outputs."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    analysis: ChannelAnalysisOutput
    batches: tuple[ChannelAnalysisBatchArtifact, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_batch_set(self) -> BatchedChannelAnalysisResult:
        if any(item.analysis.channel != self.analysis.channel for item in self.batches):
            raise ValueError("all child batch analyses must match the aggregate channel")
        sample_ids = [sample_id for item in self.batches for sample_id in item.sample_ids]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("child batch artifacts must cover disjoint sample IDs")
        return self


# Backward import compatibility for callers compiled against the pre-analysis name.
ChannelHypothesisOutput = ChannelAnalysisOutput


class _ReviewIssueBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    severity: Literal["warning", "error", "blocker"]
    message: str = Field(min_length=1, max_length=CRITIC_ISSUE_MESSAGE_MAX)
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)


class PhyschemReviewIssue(_ReviewIssueBase):
    code: PhyschemIssueCode


class ConservationReviewIssue(_ReviewIssueBase):
    code: ConservationIssueCode


class StructureReviewIssue(_ReviewIssueBase):
    code: StructureIssueCode


class MainReviewIssue(_ReviewIssueBase):
    code: MainIssueCode


class SampleCriticExplanation(BaseModel):
    """One sample's feature interpretation and Critic explanation."""

    model_config = ConfigDict(extra="forbid", strict=True)
    sample_id: str = Field(min_length=1, max_length=160)
    feature_analysis: str = Field(min_length=1, max_length=SAMPLE_REVIEW_PROSE_MAX)
    critic_explanation: str = Field(min_length=1, max_length=SAMPLE_REVIEW_PROSE_MAX)


class _ReviewBodyBase(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: ReviewVerdictName
    rating: CriticRatingRegion
    cited_evidence_ids: list[str] = Field(max_length=16)
    summary: str = Field(min_length=1, max_length=CRITIC_SUMMARY_MAX)
    sample_reviews: list[SampleCriticExplanation] = Field(
        default_factory=list, max_length=128
    )

    @model_validator(mode="after")
    def consistent_verdict(self) -> _ReviewBodyBase:
        expected = verdict_for_rating(self.rating.score)
        if self.verdict != expected:
            raise ValueError(
                f"verdict must be {expected} for Rating score {self.rating.score}"
            )
        blockers = [item for item in self.issues if item.severity == "blocker"]
        if self.verdict == "APPROVE" and (blockers or self.required_changes):
            raise ValueError("APPROVE cannot contain blockers or required changes")
        if self.verdict == "REVISE" and not self.required_changes:
            raise ValueError("REVISE requires at least one allow-listed change")
        return self


class PhyschemReviewBody(_ReviewBodyBase):
    review_scope: Literal["physchem"] = "physchem"
    issues: list[PhyschemReviewIssue] = Field(max_length=12)
    required_changes: list[PhyschemRequiredAction] = Field(max_length=12)


class ConservationReviewBody(_ReviewBodyBase):
    review_scope: Literal["conservation"] = "conservation"
    issues: list[ConservationReviewIssue] = Field(max_length=12)
    required_changes: list[ConservationRequiredAction] = Field(max_length=12)


class StructureReviewBody(_ReviewBodyBase):
    review_scope: Literal["structure"] = "structure"
    issues: list[StructureReviewIssue] = Field(max_length=12)
    required_changes: list[StructureRequiredAction] = Field(max_length=12)


class MainReviewBody(BaseModel):
    """Main Critic decision plus its explanation; it never returns a hypothesis."""

    model_config = ConfigDict(extra="forbid", strict=True)

    review_scope: Literal["main"] = "main"
    verdict: ReviewVerdictName
    rating: CriticRatingRegion
    issues: list[MainReviewIssue] = Field(max_length=12)
    required_changes: list[MainRequiredAction] = Field(max_length=12)
    cited_evidence_ids: list[str] = Field(max_length=16)
    explanation: str = Field(
        min_length=1,
        max_length=CRITIC_EXPLANATION_MAX,
        validation_alias=AliasChoices("explanation", "summary"),
    )

    @model_validator(mode="after")
    def consistent_verdict(self) -> MainReviewBody:
        expected = verdict_for_rating(self.rating.score)
        if self.verdict != expected:
            raise ValueError(
                f"verdict must be {expected} for Rating score {self.rating.score}"
            )
        blockers = [item for item in self.issues if item.severity == "blocker"]
        if self.verdict == "APPROVE" and (blockers or self.required_changes):
            raise ValueError("APPROVE cannot contain blockers or required changes")
        if self.verdict == "REVISE" and not self.required_changes:
            raise ValueError("REVISE requires at least one allow-listed change")
        return self


class PhyschemReviewOutput(PhyschemReviewBody):
    decision_id: str = Field(min_length=1, max_length=160)


class ConservationReviewOutput(ConservationReviewBody):
    decision_id: str = Field(min_length=1, max_length=160)


class StructureReviewOutput(StructureReviewBody):
    decision_id: str = Field(min_length=1, max_length=160)


class MainReviewOutput(MainReviewBody):
    decision_id: str = Field(min_length=1, max_length=160)


ChannelReviewBody = Annotated[
    PhyschemReviewBody | ConservationReviewBody | StructureReviewBody,
    Field(discriminator="review_scope"),
]
ChannelReviewOutput = Annotated[
    PhyschemReviewOutput | ConservationReviewOutput | StructureReviewOutput,
    Field(discriminator="review_scope"),
]


def review_body_type(channel: ChannelName) -> type[_ReviewBodyBase]:
    return {
        "physchem": PhyschemReviewBody,
        "conservation": ConservationReviewBody,
        "structure": StructureReviewBody,
    }[channel]


def review_output_type(channel: ChannelName) -> type[_ReviewBodyBase]:
    return {
        "physchem": PhyschemReviewOutput,
        "conservation": ConservationReviewOutput,
        "structure": StructureReviewOutput,
    }[channel]


class ApprovedChannelAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    channel: ChannelName
    analysis: ChannelAnalysisOutput = Field(
        validation_alias=AliasChoices("analysis", "hypothesis")
    )
    review: ChannelReviewOutput
    attempt: int = Field(ge=0)
    input_receipt_id: str
    output_receipt_id: str

    @model_validator(mode="after")
    def must_be_approved(self) -> ApprovedChannelAnalysis:
        if self.analysis.channel != self.channel or self.review.verdict != "APPROVE":
            raise ValueError("approved channel analysis must match channel and APPROVE verdict")
        return self

    @property
    def hypothesis(self) -> ChannelAnalysisOutput:
        """Compatibility accessor; new artifacts serialize the field as ``analysis``."""

        return self.analysis


ApprovedSubHypothesis = ApprovedChannelAnalysis


class ChildReviewAttemptArtifact(BaseModel):
    """Typed, append-only receipt for every attempted child semantic review."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    channel: ChannelName
    attempt: int = Field(ge=0)
    disposition: Literal["APPROVED", "REVISE", "REJECTED", "FAILED"]
    input_receipt_id: str
    evidence_universe: RoleVisibleEvidenceUniverse
    output_receipt_id: str | None = None
    analysis: ChannelAnalysisOutput | None = None
    analysis_batches: tuple[ChannelAnalysisBatchArtifact, ...] = ()
    review: ChannelReviewOutput | None = None
    error_code: str | None = None
    input_chars: int | None = Field(default=None, ge=0)
    request_started: bool = False
    failure_stage: str | None = None
    failed_batch_id: str | None = None
    failed_sample_ids: tuple[str, ...] = ()
    validation_paths: tuple[str, ...] = ()


class CrossChannelConflict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    position: int
    channels: tuple[ChannelName, ...]
    residue_sets: dict[str, tuple[str, ...]]
    resolution: Literal["main_scientist_must_resolve"] = "main_scientist_must_resolve"


class MainSynthesisEvidenceCard(BaseModel):
    """Bounded atomic evidence visible to the main synthesis Critic."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    evidence_id: str = Field(min_length=1, max_length=320)
    atomic_statement: str = Field(min_length=1, max_length=SYNTHESIS_CARD_STATEMENT_MAX)
    channel: str = Field(min_length=1, max_length=120)
    contribution: Literal["support", "constraint_counterevidence", "analysis_only"]
    polarity: Literal["support", "contradict", "neutral", "unknown"]
    applicability: str = Field(min_length=1, max_length=240)
    confidence: float = Field(ge=0, le=1)
    quality_status: str = Field(min_length=1, max_length=120)
    warnings: tuple[str, ...] = Field(default=(), max_length=8)
    source_uri: str | None = Field(default=None, max_length=1200)
    source_span: tuple[int, int] | None = None


class SynthesisAbstention(BaseModel):
    """Typed no-hypothesis outcome; absence of a child candidate is not a failure."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: Literal["NO_SUPPORTED_HYPOTHESIS"] = "NO_SUPPORTED_HYPOTHESIS"
    abstention_id: str = Field(min_length=1, max_length=160)
    reason: str = Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)
    evidence_ids: tuple[str, ...] = Field(default=(), max_length=12)
    unresolved_constraints: tuple[Annotated[str, Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)], ...]
    recommended_next_evidence: tuple[Annotated[str, Field(min_length=1, max_length=CANDIDATE_PROSE_MAX)], ...]


class MainReviewAttemptArtifact(BaseModel):
    """Append-only typed receipt for each main hypothesis/review attempt."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    hypothesis_attempt: int = Field(ge=0)
    disposition: Literal[
        "APPROVED", "REVISE", "REJECTED", "FAILED", "ABSTAINED"
    ]
    input_receipt_id: str = Field(min_length=1, max_length=160)
    output_receipt_id: str | None = Field(default=None, min_length=1, max_length=160)
    evidence_universe_id: str = Field(min_length=1, max_length=160)
    evidence_cards: tuple[MainSynthesisEvidenceCard, ...] = Field(max_length=12)
    hypothesis: dict[str, Any] | None = None
    abstention: SynthesisAbstention | None = None
    review: MainReviewOutput | None = None
    error_code: str | None = None


class BranchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    channel: ChannelName
    status: Literal["SUCCEEDED", "FAILED", "SKIPPED_UNAVAILABLE"]
    attempts: int = Field(ge=0)
    error_code: str | None = None
    input_chars: int | None = Field(default=None, ge=0)
    failure_category: str | None = None
    request_started: bool = False
    failure_stage: str | None = None
    batch_id: str | None = None
    sample_ids: tuple[str, ...] = ()
    validation_paths: tuple[str, ...] = ()
    completed_artifacts: tuple[ChannelAnalysisBatchArtifact, ...] = ()
    review_attempts: tuple[ChildReviewAttemptArtifact, ...] = ()
    approved: ApprovedChannelAnalysis | None = None


class HypothesisPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    status: Literal["SUCCEEDED", "FAILED"]
    branches: tuple[BranchReceipt, ...]
    conflicts: tuple[CrossChannelConflict, ...]
    evidence_universe: RoleVisibleEvidenceUniverse | None = None
    main_hypothesis: dict[str, Any] | None = None
    main_review: MainReviewOutput | None = None
    main_abstention: SynthesisAbstention | None = None
    main_review_attempts: tuple[MainReviewAttemptArtifact, ...] = ()
    main_attempts: int = Field(default=0, ge=0)
    degraded: bool = False
    failure_code: str | None = None


class CompletionManifest(BaseModel):
    """Run closure is separate from scientific/evaluation success."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    artifact_finalized: bool
    run_status: Literal["completed", "failed", "partial"]
    experiment_status: Literal["completed", "failed", "partial"]
    evaluation_status: Literal["not_evaluated", "eligible", "passed", "failed"]
    pass_eligible: bool
    expected_rounds: int = Field(ge=0)
    completed_rounds: int = Field(ge=0)
    aborted_rounds: int = Field(ge=0)
    planned_batch_sizes: tuple[int, ...]
    actual_batch_sizes: tuple[int, ...]
    required_node_failures: tuple[str, ...] = ()
    fallback_nodes: tuple[str, ...] = ()

    @field_validator(
        "planned_batch_sizes",
        "actual_batch_sizes",
        "required_node_failures",
        "fallback_nodes",
        mode="before",
    )
    @classmethod
    def normalize_json_tuples(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def prevent_false_pass(self) -> CompletionManifest:
        if len(self.planned_batch_sizes) != len(self.actual_batch_sizes):
            raise ValueError("planned and actual batch-size receipts must align")
        if len(self.planned_batch_sizes) != self.completed_rounds + self.aborted_rounds:
            raise ValueError(
                "batch-size receipts must cover every completed or aborted round"
            )
        if any(
            planned < 0 or actual < 0 or actual > planned
            for planned, actual in zip(
                self.planned_batch_sizes,
                self.actual_batch_sizes,
                strict=True,
            )
        ):
            raise ValueError("actual batch size must be between zero and planned size")
        complete = (
            self.artifact_finalized
            and self.run_status == "completed"
            and self.experiment_status == "completed"
            and self.completed_rounds == self.expected_rounds
            and self.aborted_rounds == 0
            and not self.required_node_failures
            and not self.fallback_nodes
            and self.actual_batch_sizes == self.planned_batch_sizes
        )
        if self.pass_eligible != complete:
            raise ValueError("pass_eligible does not match completion gate")
        if self.evaluation_status == "passed" and not self.pass_eligible:
            raise ValueError("an incomplete or degraded experiment cannot be passed")
        return self
