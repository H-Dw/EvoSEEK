from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from pydantic import model_validator

from fitness_agents.local_knowledge.prompt_safety import instruction_like_markers

from .attestation import (
    AttestationError,
    sign_canonical_payload,
    verify_canonical_payload,
)
from .canonical import content_sha256, stable_id, text_sha256
from .contracts import (
    AllowedSearchHit,
    DecisionPermission,
    EvidenceProductBundle,
    PlannedQuery,
    PolicyReceipt,
    Publication,
    PublicationAcquisition,
    ReleaseApprovalReceipt,
    ReleaseManifest,
    ReleaseRecord,
    ResearchBrief,
    ReviewReceipt,
    ScopeAssertion,
    SearchRoute,
    SearchRun,
    StrictModel,
    SubjectScope,
)
from .policy import (
    ExternalEvidenceScopePolicy,
    SourceAccessPermit,
)
from .providers import ExternalSearchResult, SearchProvider

VALIDATOR_VERSION = "deep-research-evidence-validator:v2"


class ScreenedSearchResult(StrictModel):
    result: ExternalSearchResult
    disposition: Literal["allowed"]
    policy_receipt: PolicyReceipt


class SearchPolicyEvent(StrictModel):
    result_id: str
    disposition: Literal[
        "denied",
        "quarantined",
        "duplicate",
        "budget_excluded",
    ]
    reason_code: str
    policy_hash: str


class DeepSearchDiscovery(StrictModel):
    schema_version: Literal["deep-search-discovery:v2"] = "deep-search-discovery:v2"
    research_brief: ResearchBrief
    planned_queries: tuple[PlannedQuery, ...]
    search_runs: tuple[SearchRun, ...]
    scope_assertions: tuple[ScopeAssertion, ...]
    allowed_search_hits: tuple[AllowedSearchHit, ...]
    screened_results: tuple[ScreenedSearchResult, ...]
    publications: tuple[Publication, ...]
    policy_events: tuple[SearchPolicyEvent, ...]
    discovery_hash: str

    @model_validator(mode="after")
    def verify_discovery_hash(self) -> DeepSearchDiscovery:
        expected = content_sha256(
            {
                "research_brief": self.research_brief,
                "planned_queries": self.planned_queries,
                "search_runs": self.search_runs,
                "scope_assertions": self.scope_assertions,
                "allowed_search_hits": self.allowed_search_hits,
                "screened_results": self.screened_results,
                "publications": self.publications,
                "policy_events": self.policy_events,
            }
        )
        if self.discovery_hash != expected:
            raise ValueError("DeepSearchDiscovery content hash mismatch")
        return self


class ValidationIssue(StrictModel):
    severity: Literal["error", "warning"]
    code: str
    record_id: str | None = None
    message: str


class EvidenceValidationReport(StrictModel):
    validator_version: str = VALIDATOR_VERSION
    valid: bool
    release_ready: bool
    issues: tuple[ValidationIssue, ...]
    counts: dict[str, int]
    bundle_hash: str


class DeepSearchPlanner:
    _ROUTE_SUFFIX: ClassVar[dict[SearchRoute, str]] = {
        SearchRoute.LANDSCAPE: "review terminology landscape",
        SearchRoute.PRIMARY: "primary research empirical evidence",
        SearchRoute.COUNTEREVIDENCE: "negative result limitation no effect",
        SearchRoute.BOUNDARY: "boundary condition context dependence",
        SearchRoute.REPLICATION: "independent replication validation",
        SearchRoute.METADATA_VERIFY: "publication metadata DOI",
    }

    def __init__(self, policy: ExternalEvidenceScopePolicy) -> None:
        self.policy = policy

    def plan(self, brief: ResearchBrief) -> tuple[PlannedQuery, ...]:
        if brief.policy_hash != self.policy.policy_hash:
            raise ValueError("ResearchBrief policy hash does not match the active scope policy")
        root_decision = self.policy.inspect_query(brief.research_question)
        if root_decision.decision != "allowed":
            raise PermissionError(root_decision.reason)
        planned: list[PlannedQuery] = []
        for leaf in sorted(brief.question_tree, key=lambda item: (-item.priority, item.question_leaf_id)):
            leaf_decision = self.policy.inspect_query(leaf.question)
            counter_decision = self.policy.inspect_query(leaf.counterevidence_question)
            if leaf_decision.decision != "allowed" or counter_decision.decision != "allowed":
                raise PermissionError("QuestionLeaf enters an excluded external-evidence scope")
            for route in brief.required_search_routes:
                route_decision = (
                    counter_decision
                    if route == SearchRoute.COUNTEREVIDENCE
                    else leaf_decision
                )
                base = (
                    leaf.counterevidence_question
                    if route == SearchRoute.COUNTEREVIDENCE
                    else leaf.question
                )
                query = " ".join(
                    (
                        base.strip(),
                        self._ROUTE_SUFFIX[route],
                        self.policy.negative_query_suffix,
                    )
                )
                filters = (
                    {"until_date": brief.publication_cutoff}
                    if brief.publication_cutoff
                    else {}
                )
                query_id = stable_id(
                    brief.brief_id,
                    leaf.question_leaf_id,
                    route.value,
                )
                planned.append(
                    PlannedQuery(
                        query_id=query_id,
                        question_leaf_id=leaf.question_leaf_id,
                        route=route,
                        query=query,
                        filters=filters,
                        policy_receipt=self.policy.issue_receipt(
                            route_decision,
                            stage="discover_query",
                            subject_id=query_id,
                            subject_sha256=text_sha256(query),
                        ),
                    )
                )
                if len(planned) >= brief.budget.max_queries:
                    return tuple(planned)
        return tuple(planned)


class DeepSearchEngine:
    def __init__(
        self,
        providers: Iterable[SearchProvider],
        *,
        policy: ExternalEvidenceScopePolicy,
        scope_assertions: Mapping[str, ScopeAssertion] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.providers = tuple(providers)
        if not self.providers:
            raise ValueError("DeepSearchEngine requires at least one provider")
        self.policy = policy
        self.scope_assertions = dict(scope_assertions or {})
        self.clock = clock or (lambda: datetime.now(timezone.utc))

    def discover(self, brief: ResearchBrief) -> DeepSearchDiscovery:
        planned = DeepSearchPlanner(self.policy).plan(brief)
        runs: list[SearchRun] = []
        screened: list[ScreenedSearchResult] = []
        allowed_hits: list[AllowedSearchHit] = []
        used_scope_assertions: dict[str, ScopeAssertion] = {}
        events: list[SearchPolicyEvent] = []
        accepted_by_key: dict[str, ExternalSearchResult] = {}
        acquisition_hits: defaultdict[
            str, list[AllowedSearchHit]
        ] = defaultdict(list)
        quarantined_identities = {
            value.casefold() for value in brief.source_quarantine_ids
        }

        for query in planned:
            for provider in self.providers:
                executed_at = self.clock()
                search_run_id = stable_id(
                    "search-run",
                    brief.brief_id,
                    query.query_id,
                    provider.name,
                    executed_at.isoformat(),
                )
                provider_error_code: str | None = None
                attempt_count = 1
                try:
                    raw_results = provider.search(
                        query.query,
                        limit=brief.budget.max_results_per_query,
                        filters=query.filters,
                    )
                    attempt_count = int(
                        getattr(provider, "last_attempt_count", 1)
                    )
                except Exception as provider_error:  # noqa: BLE001 - audited provider boundary
                    raw_results = ()
                    provider_error_code = type(provider_error).__name__
                    attempt_count = int(
                        getattr(
                            provider,
                            "last_attempt_count",
                            getattr(provider_error, "attempt_count", 1),
                        )
                    )
                opaque_result_ids: list[str] = []
                accepted_result_ids: list[str] = []
                excluded_count = 0
                duplicate_count = 0
                publication_budget_exhausted = False
                for raw in raw_results:
                    result = raw
                    assertion = self.scope_assertions.get(result.artifact_id)
                    if assertion is not None:
                        result = result.model_copy(update={"scope_assertion": assertion})
                    opaque_result_ids.append(result.result_id)
                    result_identities = {
                        result.result_id.casefold(),
                        result.artifact_id.casefold(),
                        result.provider_record_id.casefold(),
                    }
                    if result.doi:
                        result_identities.update(
                            {result.doi.casefold(), f"doi:{result.doi.casefold()}"}
                        )
                    if quarantined_identities.intersection(result_identities):
                        excluded_count += 1
                        events.append(
                            SearchPolicyEvent(
                                result_id=result.result_id,
                                disposition="denied",
                                reason_code="benchmark_or_source_quarantine",
                                policy_hash=self.policy.policy_hash,
                            )
                        )
                        continue
                    decision = self.policy.decide_metadata(
                        artifact_id=result.artifact_id,
                        title=result.title,
                        abstract=result.abstract,
                        subjects=result.subjects,
                        assertion=result.scope_assertion,
                    )
                    if decision.decision == "denied":
                        excluded_count += 1
                        events.append(
                            SearchPolicyEvent(
                                result_id=result.result_id,
                                disposition="denied",
                                reason_code=decision.reason,
                                policy_hash=decision.policy_hash,
                            )
                        )
                        continue
                    if decision.decision == "quarantined":
                        excluded_count += 1
                        events.append(
                            SearchPolicyEvent(
                                result_id=result.result_id,
                                disposition="quarantined",
                                reason_code=decision.reason,
                                policy_hash=decision.policy_hash,
                            )
                        )
                        continue
                    if result.scope_assertion is None:
                        raise RuntimeError(
                            "Allowed metadata requires a retained ScopeAssertion"
                        )
                    key = _publication_key(result)
                    if key in accepted_by_key:
                        duplicate_count += 1
                        search_hit = _allowed_search_hit(
                            search_run_id=search_run_id,
                            result=result,
                            disposition="duplicate_publication",
                            decision=decision,
                            policy=self.policy,
                            issued_at=executed_at,
                        )
                        allowed_hits.append(search_hit)
                        acquisition_hits[key].append(search_hit)
                        used_scope_assertions[
                            result.scope_assertion.scope_assertion_id
                        ] = result.scope_assertion
                        screened.append(
                            ScreenedSearchResult(
                                result=result,
                                disposition="allowed",
                                policy_receipt=search_hit.policy_receipt,
                            )
                        )
                        events.append(
                            SearchPolicyEvent(
                                result_id=result.result_id,
                                disposition="duplicate",
                                reason_code="canonical_publication_duplicate",
                                policy_hash=self.policy.policy_hash,
                            )
                        )
                        continue
                    if len(accepted_by_key) >= brief.budget.max_publications:
                        excluded_count += 1
                        publication_budget_exhausted = True
                        events.append(
                            SearchPolicyEvent(
                                result_id=result.result_id,
                                disposition="budget_excluded",
                                reason_code="maximum_unique_publications_reached",
                                policy_hash=self.policy.policy_hash,
                            )
                        )
                        continue
                    accepted_by_key[key] = result
                    accepted_result_ids.append(result.result_id)
                    search_hit = _allowed_search_hit(
                        search_run_id=search_run_id,
                        result=result,
                        disposition="accepted",
                        decision=decision,
                        policy=self.policy,
                        issued_at=executed_at,
                    )
                    allowed_hits.append(search_hit)
                    acquisition_hits[key].append(search_hit)
                    used_scope_assertions[
                        result.scope_assertion.scope_assertion_id
                    ] = result.scope_assertion
                    screened.append(
                        ScreenedSearchResult(
                            result=result,
                            disposition="allowed",
                            policy_receipt=search_hit.policy_receipt,
                        )
                    )
                stop_reason = (
                    "provider_error"
                    if provider_error_code is not None
                    else (
                        "budget_exhausted"
                        if publication_budget_exhausted
                        else ("completed" if raw_results else "no_results")
                    )
                )
                run_payload = {
                    "search_run_id": search_run_id,
                    "brief_id": brief.brief_id,
                    "provider": provider.name,
                    "planned_query_id": query.query_id,
                    "question_leaf_id": query.question_leaf_id,
                    "route": query.route.value,
                    "filters": query.filters,
                    "executed_at": executed_at,
                    "exact_query_sha256": text_sha256(query.query),
                    "result_ids": opaque_result_ids,
                    "accepted_result_ids": accepted_result_ids,
                    "excluded_count": excluded_count,
                    "duplicate_count": duplicate_count,
                    "attempt_count": attempt_count,
                    "error_code": provider_error_code,
                    "stop_reason": stop_reason,
                }
                runs.append(
                    SearchRun(
                        search_run_id=search_run_id,
                        brief_id=brief.brief_id,
                        planned_query_id=query.query_id,
                        question_leaf_id=query.question_leaf_id,
                        provider=provider.name,
                        route=query.route,
                        exact_query=query.query,
                        exact_query_sha256=text_sha256(query.query),
                        filters=query.filters,
                        executed_at=executed_at,
                        result_ids=tuple(opaque_result_ids),
                        accepted_result_ids=tuple(accepted_result_ids),
                        excluded_result_count=excluded_count,
                        duplicate_result_count=duplicate_count,
                        attempt_count=attempt_count,
                        error_code=provider_error_code,
                        stop_reason=stop_reason,
                        snapshot_hash=content_sha256(run_payload),
                        policy_receipt=self.policy.issue_receipt(
                            self.policy.inspect_query(query.query),
                            stage="search_run",
                            subject_id=query.query_id,
                            subject_sha256=text_sha256(query.query),
                            issued_at=executed_at,
                        ),
                    )
                )

        publications = tuple(
            publication
            for key, result in sorted(accepted_by_key.items())
            if (
                publication := _publication_from_result(
                    result,
                    acquisition_hits=tuple(acquisition_hits[key]),
                    policy=self.policy,
                )
            )
            is not None
        )
        discovery_payload = {
            "research_brief": brief,
            "planned_queries": planned,
            "search_runs": runs,
            "scope_assertions": [
                used_scope_assertions[item]
                for item in sorted(used_scope_assertions)
            ],
            "allowed_search_hits": allowed_hits,
            "screened_results": screened,
            "publications": publications,
            "policy_events": events,
        }
        return DeepSearchDiscovery(
            research_brief=brief,
            planned_queries=planned,
            search_runs=tuple(runs),
            scope_assertions=tuple(
                used_scope_assertions[item]
                for item in sorted(used_scope_assertions)
            ),
            allowed_search_hits=tuple(allowed_hits),
            screened_results=tuple(screened),
            publications=publications,
            policy_events=tuple(events),
            discovery_hash=content_sha256(discovery_payload),
        )

    def fetch_full_text(
        self,
        result: ExternalSearchResult,
        *,
        permit: SourceAccessPermit,
        fetcher: Callable[[str], str],
    ) -> str:
        assertion = result.scope_assertion
        if assertion is None:
            raise PermissionError("Full-text fetch requires a verified scope assertion")
        self.policy.verify_permit(
            permit,
            artifact_id=result.artifact_id,
            operation="full_text_fetch",
            assertion=assertion,
            resource_locator=result.url,
        )
        return fetcher(result.url)


def _publication_key(result: ExternalSearchResult) -> str:
    if result.doi:
        return f"doi:{result.doi.casefold()}"
    normalized_title = re.sub(r"\W+", " ", result.title.casefold()).strip()
    first_author = (
        re.sub(r"\W+", " ", result.authors[0].casefold()).strip()
        if result.authors
        else "unknown-author"
    )
    return stable_id(
        "metadata-publication",
        normalized_title,
        first_author,
        str(result.year or "unknown-year"),
    )


def _allowed_hit_publication_key(hit: AllowedSearchHit) -> str:
    if hit.doi:
        return f"doi:{hit.doi.casefold()}"
    normalized_title = re.sub(r"\W+", " ", hit.title.casefold()).strip()
    first_author = (
        re.sub(r"\W+", " ", hit.authors[0].casefold()).strip()
        if hit.authors
        else "unknown-author"
    )
    return stable_id(
        "metadata-publication",
        normalized_title,
        first_author,
        str(hit.year or "unknown-year"),
    )


def _scope_metadata_sha256(result: ExternalSearchResult) -> str:
    assertion_hash = (
        result.scope_assertion.assertion_hash
        if result.scope_assertion is not None
        else None
    )
    return content_sha256(
        {
            "artifact_id": result.artifact_id,
            "title": result.title,
            "abstract": result.abstract,
            "subjects": result.subjects,
            "scope_assertion_sha256": assertion_hash,
        }
    )


def _allowed_search_hit(
    *,
    search_run_id: str,
    result: ExternalSearchResult,
    disposition: Literal["accepted", "duplicate_publication"],
    decision: Any,
    policy: ExternalEvidenceScopePolicy,
    issued_at: datetime,
) -> AllowedSearchHit:
    assertion = result.scope_assertion
    if assertion is None:
        raise ValueError("AllowedSearchHit requires a retained ScopeAssertion")
    search_hit_id = stable_id(
        "search-hit",
        search_run_id,
        result.result_id,
        assertion.scope_assertion_id,
    )
    return AllowedSearchHit(
        search_hit_id=search_hit_id,
        search_run_id=search_run_id,
        result_id=result.result_id,
        disposition=disposition,
        artifact_id=result.artifact_id,
        provider=result.provider,
        provider_record_id=result.provider_record_id,
        title=result.title,
        abstract=result.abstract,
        authors=result.authors,
        year=result.year,
        venue=result.venue or None,
        doi=result.doi,
        url=result.url,
        publication_type=result.publication_type,
        subjects=result.subjects,
        provider_score=result.retrieval.provider_score,
        retrieval_score=result.retrieval.retrieval_score,
        retrieval_components=result.retrieval.components,
        scope_assertion_id=assertion.scope_assertion_id,
        policy_receipt=policy.issue_receipt(
            decision,
            stage="search_hit",
            subject_id=search_hit_id,
            subject_sha256=_scope_metadata_sha256(result),
            issued_at=issued_at,
        ),
    )
def _search_hit_scope_metadata_sha256(
    hit: AllowedSearchHit,
    assertion: ScopeAssertion,
) -> str:
    return content_sha256(
        {
            "artifact_id": hit.artifact_id,
            "title": hit.title,
            "abstract": hit.abstract,
            "subjects": hit.subjects,
            "scope_assertion_sha256": assertion.assertion_hash,
        }
    )


def _publication_from_result(
    result: ExternalSearchResult,
    *,
    acquisition_hits: tuple[AllowedSearchHit, ...],
    policy: ExternalEvidenceScopePolicy,
) -> Publication | None:
    assertion = result.scope_assertion
    if assertion is None or result.year is None or not result.authors or not result.venue:
        return None
    decision = policy.decide_metadata(
        artifact_id=result.artifact_id,
        title=result.title,
        abstract=result.abstract,
        subjects=result.subjects,
        assertion=assertion,
    )
    if decision.decision != "allowed":
        return None
    if assertion.subject_scope not in {
        SubjectScope.GENERIC_PROTEIN,
        SubjectScope.NONVIRAL_PROTEIN,
    }:
        return None
    publication_id = f"doi:{result.doi}" if result.doi else result.artifact_id
    return Publication(
        publication_id=publication_id,
        identifier_aliases=(result.artifact_id,),
        title=result.title,
        authors=result.authors,
        year=result.year,
        venue=result.venue,
        doi=result.doi,
        canonical_url=result.url,
        publication_type=result.publication_type,
        study_family_id=stable_id("study", publication_id),
        metadata_verified=False,
        full_text_status="metadata_only",
        canonical_search_hit_id=acquisition_hits[0].search_hit_id,
        acquisitions=tuple(
            PublicationAcquisition(
                search_hit_id=hit.search_hit_id,
            )
            for hit in acquisition_hits
        ),
        source_scope=assertion.subject_scope.value,
        scope_assertion_id=assertion.scope_assertion_id,
    )


def _record_entries(bundle: EvidenceProductBundle) -> tuple[ReleaseRecord, ...]:
    records: list[tuple[str, str, Any, tuple[str, ...]]] = [
        (
            bundle.research_brief.brief_id,
            "research_brief",
            bundle.research_brief,
            (),
        )
    ]
    records.extend(
        (item.search_run_id, "search_run", item, (item.brief_id,))
        for item in bundle.search_runs
    )
    records.extend(
        (
            item.scope_assertion_id,
            "scope_assertion",
            item,
            (),
        )
        for item in bundle.scope_assertions
    )
    records.extend(
        (
            item.search_hit_id,
            "allowed_search_hit",
            item,
            (item.search_run_id, item.scope_assertion_id),
        )
        for item in bundle.allowed_search_hits
    )
    records.extend(
        (
            item.publication_id,
            "publication",
            item,
            (
                item.scope_assertion_id,
                *(acquisition.search_hit_id for acquisition in item.acquisitions),
            ),
        )
        for item in bundle.publications
    )
    records.extend(
        (
            item.source_span_id,
            "source_span",
            item,
            (item.publication_id, item.scope_assertion_id),
        )
        for item in bundle.source_spans
    )
    records.extend(
        (
            item.evidence_group_id,
            "evidence_group",
            item,
            item.source_span_ids,
        )
        for item in bundle.evidence_groups
    )
    records.extend(
        (
            item.claim_id,
            "atomic_claim",
            item,
            item.evidence_group_ids,
        )
        for item in bundle.atomic_claims
    )
    records.extend(
        (
            item.logic_unit_id,
            "logic_unit",
            item,
            (
                *item.premise_claim_ids,
                *item.counterclaim_ids,
                *item.search_coverage_run_ids,
            ),
        )
        for item in bundle.logic_units
    )
    records.extend(
        (
            item.decision_card_id,
            "knowledge_decision_card",
            item,
            item.logic_unit_ids,
        )
        for item in bundle.decision_cards
    )
    records.extend(
        (
            item.review_receipt_id,
            "review_receipt",
            item,
            (item.record_id,),
        )
        for item in bundle.review_receipts
    )
    approval_dependencies = tuple(record_id for record_id, _, _, _ in records)
    records.extend(
        (
            item.approval_id,
            "release_approval",
            item,
            approval_dependencies,
        )
        for item in bundle.release_approvals
    )
    return tuple(
        ReleaseRecord(
            record_id=record_id,
            record_type=record_type,
            content_sha256=content_sha256(record),
            dependency_ids=tuple(sorted(set(dependencies))),
        )
        for record_id, record_type, record, dependencies in sorted(
            records, key=lambda value: (value[1], value[0])
        )
    )


def review_input_sha256(
    bundle: EvidenceProductBundle,
    *,
    record_id: str,
    review_type: str,
    method_version: str,
) -> str:
    """Hash a reviewed record's transitive evidence closure and review method."""

    records = {
        item.record_id: item
        for item in _record_entries(
            bundle.model_copy(update={"release_manifest": None})
        )
        if item.record_type != "review_receipt"
    }
    if record_id not in records:
        raise KeyError(f"Unknown reviewed record: {record_id}")
    closure: dict[str, str] = {}
    visiting: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in closure:
            return
        if identifier in visiting:
            raise ValueError("Review dependency graph contains a cycle")
        record = records.get(identifier)
        if record is None:
            raise KeyError(f"Review dependency is missing: {identifier}")
        visiting.add(identifier)
        for dependency_id in record.dependency_ids:
            visit(dependency_id)
        visiting.remove(identifier)
        closure[identifier] = record.content_sha256

    visit(record_id)
    return content_sha256(
        {
            "reviewed_record_id": record_id,
            "dependency_closure": sorted(closure.items()),
            "policy_hash": bundle.research_brief.policy_hash,
            "review_type": review_type,
            "method_version": method_version,
        }
    )


def issue_review_receipt(
    bundle: EvidenceProductBundle,
    *,
    review_receipt_id: str,
    review_type: Literal[
        "metadata_identity",
        "full_text_scope",
        "source_span_resolution",
        "independence_grouping",
        "claim_entailment",
        "task_applicability",
        "decision_permission",
        "scope_assertion",
    ],
    record_id: str,
    reviewer_id: str,
    reviewer_kind: Literal["human", "model_assisted", "deterministic_rule"],
    method_version: str,
    decision: Literal["passed", "failed", "escalated"],
    reviewed_at: datetime,
    expires_at: datetime,
    signing_key_id: str,
    signing_key: bytes,
    model_fingerprint: str | None = None,
    prompt_sha256: str | None = None,
) -> ReviewReceipt:
    """Create a signed review receipt over the record's evidence closure."""

    payload = {
        "schema_version": "evidence-review-receipt:v2",
        "review_receipt_id": review_receipt_id,
        "review_type": review_type,
        "record_id": record_id,
        "reviewer_id": reviewer_id,
        "reviewer_kind": reviewer_kind,
        "method_version": method_version,
        "input_sha256": review_input_sha256(
            bundle,
            record_id=record_id,
            review_type=review_type,
            method_version=method_version,
        ),
        "decision": decision,
        "reviewed_at": reviewed_at,
        "expires_at": expires_at,
        "model_fingerprint": model_fingerprint,
        "prompt_sha256": prompt_sha256,
    }
    attestation = sign_canonical_payload(
        payload,
        key_id=signing_key_id,
        key=signing_key,
        purpose="evidence-review:v1",
    )
    return ReviewReceipt(**payload, attestation=attestation)


def release_approval_input_sha256(
    bundle: EvidenceProductBundle,
    *,
    release_version: str,
    status: Literal["released"],
    created_at: datetime,
    parent_release_id: str | None = None,
) -> str:
    """Hash the complete pre-approval product and exact intended release core."""

    unsigned = bundle.model_copy(
        update={"release_approvals": (), "release_manifest": None}
    )
    records = _record_entries(unsigned)
    dependency_graph = {
        item.record_id: list(item.dependency_ids) for item in records
    }
    return content_sha256(
        {
            "schema_version": "release-approval-input:v2",
            "release_version": release_version,
            "status": status,
            "created_at": created_at,
            "parent_release_id": parent_release_id,
            "validator_version": VALIDATOR_VERSION,
            "policy_hash": unsigned.research_brief.policy_hash,
            "records": records,
            "dependency_graph_sha256": content_sha256(dependency_graph),
            "retrieval_record_types": (
                "atomic_claim",
                "logic_unit",
                "knowledge_decision_card",
            ),
            "denied_path_operations": 0,
            "excluded_result_count": sum(
                item.excluded_result_count for item in unsigned.search_runs
            ),
        }
    )


def issue_release_approval(
    bundle: EvidenceProductBundle,
    *,
    approval_id: str,
    reviewer_id: str,
    method_version: str,
    approval_artifact_id: str,
    approval_artifact_sha256: str,
    approved_at: datetime,
    release_version: str,
    created_at: datetime,
    parent_release_id: str | None,
    signing_key_id: str,
    signing_key: bytes,
) -> ReleaseApprovalReceipt:
    """Create a signed human approval that cannot replay across release intent."""

    input_sha256 = release_approval_input_sha256(
        bundle,
        release_version=release_version,
        status="released",
        created_at=created_at,
        parent_release_id=parent_release_id,
    )
    payload = {
        "schema_version": "evidence-release-approval:v2",
        "approval_id": approval_id,
        "reviewer_id": reviewer_id,
        "reviewer_kind": "human",
        "method_version": method_version,
        "input_sha256": input_sha256,
        "approval_artifact_id": approval_artifact_id,
        "approval_artifact_sha256": approval_artifact_sha256,
        "decision": "approved",
        "approved_at": approved_at,
        "target_release_version": release_version,
        "target_status": "released",
        "target_created_at": created_at,
        "target_parent_release_id": parent_release_id,
    }
    attestation = sign_canonical_payload(
        payload,
        key_id=signing_key_id,
        key=signing_key,
        purpose="release-approval:v1",
    )
    return ReleaseApprovalReceipt(**payload, attestation=attestation)


def build_release_manifest(
    bundle: EvidenceProductBundle,
    *,
    release_version: str,
    status: Literal["candidate", "released"] = "candidate",
    created_at: datetime | None = None,
    parent_release_id: str | None = None,
) -> ReleaseManifest:
    records = _record_entries(bundle)
    dependency_graph = {
        item.record_id: list(item.dependency_ids) for item in records
    }
    created = created_at or datetime.now(timezone.utc)
    if status == "released":
        for approval in bundle.release_approvals:
            if (
                approval.target_release_version != release_version
                or approval.target_status != status
                or approval.target_created_at != created
                or approval.target_parent_release_id != parent_release_id
            ):
                raise ValueError(
                    "Release approval target does not match the requested release core"
                )
    excluded_count = sum(
        item.excluded_result_count for item in bundle.search_runs
    )
    release_id = _release_id(
        release_version=release_version,
        status=status,
        created_at=created,
        parent_release_id=parent_release_id,
        policy_hash=bundle.research_brief.policy_hash,
        records=records,
        dependency_graph_sha256=content_sha256(dependency_graph),
        excluded_result_count=excluded_count,
    )
    return ReleaseManifest(
        release_id=release_id,
        release_version=release_version,
        status=status,
        created_at=created,
        parent_release_id=parent_release_id,
        policy_hash=bundle.research_brief.policy_hash,
        validator_version=VALIDATOR_VERSION,
        records=records,
        dependency_graph_sha256=content_sha256(dependency_graph),
        denied_path_operations=0,
        excluded_result_count=excluded_count,
        release_approval_ids=tuple(
            item.approval_id
            for item in bundle.release_approvals
            if item.decision == "approved"
        ),
    )


def _release_id(
    *,
    release_version: str,
    status: str,
    created_at: datetime,
    parent_release_id: str | None,
    policy_hash: str,
    records: tuple[ReleaseRecord, ...],
    dependency_graph_sha256: str,
    excluded_result_count: int,
) -> str:
    core = {
        "release_version": release_version,
        "status": status,
        "created_at": created_at,
        "parent_release_id": parent_release_id,
        "policy_hash": policy_hash,
        "validator_version": VALIDATOR_VERSION,
        "records": records,
        "dependency_graph_sha256": dependency_graph_sha256,
        "retrieval_record_types": (
            "atomic_claim",
            "logic_unit",
            "knowledge_decision_card",
        ),
        "denied_path_operations": 0,
        "excluded_result_count": excluded_result_count,
    }
    return stable_id("release-core", content_sha256(core))


def validate_evidence_product(
    bundle: EvidenceProductBundle,
    *,
    active_policy: ExternalEvidenceScopePolicy | None = None,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]] | None = None,
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]] | None = None,
) -> EvidenceValidationReport:
    issues: list[ValidationIssue] = []

    def error(code: str, message: str, record_id: str | None = None) -> None:
        issues.append(
            ValidationIssue(
                severity="error",
                code=code,
                record_id=record_id,
                message=message,
            )
        )

    def warning(code: str, message: str, record_id: str | None = None) -> None:
        issues.append(
            ValidationIssue(
                severity="warning",
                code=code,
                record_id=record_id,
                message=message,
            )
        )

    if not bundle.research_brief.non_viral_only:
        error("scope.nonviral_required", "ResearchBrief must remain non-viral-only")
    if active_policy is None:
        error(
            "policy.trust_anchor_missing",
            "Validation requires an explicitly keyed active scope policy",
        )
    elif active_policy.signing_key_id is None:
        error(
            "policy.trust_anchor_missing",
            "Active scope policy has no explicit receipt-verification key",
        )
    elif bundle.research_brief.policy_hash != active_policy.policy_hash:
        error(
            "policy.active_policy_mismatch",
            "ResearchBrief does not match the active scope policy",
        )

    search_runs = _unique_map(
        bundle.search_runs, "search_run_id", error=error
    )
    scope_assertions = _unique_map(
        bundle.scope_assertions, "scope_assertion_id", error=error
    )
    search_hits = _unique_map(
        bundle.allowed_search_hits, "search_hit_id", error=error
    )
    publications = _unique_map(
        bundle.publications, "publication_id", error=error
    )
    spans = _unique_map(bundle.source_spans, "source_span_id", error=error)
    groups = _unique_map(
        bundle.evidence_groups, "evidence_group_id", error=error
    )
    claims = _unique_map(bundle.atomic_claims, "claim_id", error=error)
    logic_units = _unique_map(
        bundle.logic_units, "logic_unit_id", error=error
    )
    cards = _unique_map(
        bundle.decision_cards, "decision_card_id", error=error
    )
    review_receipts = _unique_map(
        bundle.review_receipts, "review_receipt_id", error=error
    )
    release_approvals = _unique_map(
        bundle.release_approvals, "approval_id", error=error
    )
    review_reference_time = (
        bundle.release_manifest.created_at
        if bundle.release_manifest is not None
        else (
            max(
                item.target_created_at for item in bundle.release_approvals
            )
            if bundle.release_approvals
            else datetime.now(timezone.utc)
        )
    )

    global_record_ids = [
        bundle.research_brief.brief_id,
        *search_runs,
        *scope_assertions,
        *search_hits,
        *publications,
        *spans,
        *groups,
        *claims,
        *logic_units,
        *cards,
        *review_receipts,
        *release_approvals,
    ]
    if len(global_record_ids) != len(set(global_record_ids)):
        error(
            "record.cross_type_duplicate_id",
            "Record IDs must be unique across all evidence-product record types",
        )

    records_by_id: dict[str, Any] = {
        bundle.research_brief.brief_id: bundle.research_brief,
        **search_runs,
        **scope_assertions,
        **search_hits,
        **publications,
        **spans,
        **groups,
        **claims,
        **logic_units,
        **cards,
    }
    valid_receipt_ids: set[str] = set()
    allowed_reviewer_kinds = {
        "metadata_identity": {"human", "deterministic_rule"},
        "full_text_scope": {"human"},
        "source_span_resolution": {"human", "model_assisted"},
        "independence_grouping": {"human", "model_assisted"},
        "claim_entailment": {"human", "model_assisted"},
        "task_applicability": {"human", "model_assisted"},
        "decision_permission": {"human"},
        "scope_assertion": {"human"},
    }
    for receipt in bundle.review_receipts:
        receipt_valid = True
        if (
            receipt.reviewed_at > review_reference_time
            or receipt.expires_at < review_reference_time
        ):
            receipt_valid = False
            error(
                "review.validity_window_mismatch",
                "ReviewReceipt is not valid at the release/reference time",
                receipt.review_receipt_id,
            )
        reviewed_record = records_by_id.get(receipt.record_id)
        if reviewed_record is None:
            receipt_valid = False
            error(
                "review.record_dangling",
                f"ReviewReceipt references unknown record {receipt.record_id}",
                receipt.review_receipt_id,
            )
        else:
            try:
                expected_review_hash = review_input_sha256(
                    bundle,
                    record_id=receipt.record_id,
                    review_type=receipt.review_type,
                    method_version=receipt.method_version,
                )
            except (KeyError, ValueError) as review_error:
                receipt_valid = False
                error(
                    "review.dependency_closure_invalid",
                    str(review_error),
                    receipt.review_receipt_id,
                )
            else:
                if receipt.input_sha256 != expected_review_hash:
                    receipt_valid = False
                    error(
                        "review.input_hash_mismatch",
                        "ReviewReceipt does not bind the current dependency closure",
                        receipt.review_receipt_id,
                    )
        if receipt.decision != "passed":
            receipt_valid = False
            error(
                "review.not_passed",
                "Only passed review receipts can support a release",
                receipt.review_receipt_id,
            )
        if receipt.reviewer_kind == "model_assisted" and (
            not receipt.model_fingerprint or not receipt.prompt_sha256
        ):
            receipt_valid = False
            error(
                "review.model_provenance_missing",
                "Model-assisted reviews require model and prompt fingerprints",
                receipt.review_receipt_id,
            )
        if receipt.reviewer_kind not in allowed_reviewer_kinds[receipt.review_type]:
            receipt_valid = False
            error(
                "review.reviewer_kind_unauthorized",
                "Reviewer kind is not authorized for this review type",
                receipt.review_receipt_id,
            )
        reviewer_trust = (trusted_reviewer_keys or {}).get(receipt.reviewer_id)
        if reviewer_trust is None:
            receipt_valid = False
            error(
                "review.trust_anchor_missing",
                "ReviewReceipt reviewer has no explicit trusted key",
                receipt.review_receipt_id,
            )
        else:
            reviewer_key_id, reviewer_key = reviewer_trust
            if receipt.attestation.key_id != reviewer_key_id:
                receipt_valid = False
                error(
                    "review.signer_key_id_mismatch",
                    "ReviewReceipt key ID differs from the reviewer trust registry",
                    receipt.review_receipt_id,
                )
            try:
                verify_canonical_payload(
                    receipt.model_dump(mode="python", exclude={"attestation"}),
                    receipt.attestation,
                    trusted_keyring={reviewer_key_id: reviewer_key},
                    expected_purpose="evidence-review:v1",
                )
            except AttestationError as attestation_error:
                receipt_valid = False
                error(
                    "review.signature_invalid",
                    str(attestation_error),
                    receipt.review_receipt_id,
                )
        if receipt_valid:
            valid_receipt_ids.add(receipt.review_receipt_id)

    valid_approval_ids: set[str] = set()
    approved_reviewers: set[str] = set()
    for approval in bundle.release_approvals:
        approval_valid = True
        if approval.approval_artifact_sha256 == "0" * 64:
            approval_valid = False
            error(
                "release_approval.artifact_hash_placeholder",
                "Release approval artifact hash cannot be a placeholder",
                approval.approval_id,
            )
        if approval.approved_at > approval.target_created_at:
            approval_valid = False
            error(
                "release_approval.chronology_invalid",
                "Release approval cannot postdate its target release timestamp",
                approval.approval_id,
            )
        expected_approval_input = release_approval_input_sha256(
            bundle,
            release_version=approval.target_release_version,
            status=approval.target_status,
            created_at=approval.target_created_at,
            parent_release_id=approval.target_parent_release_id,
        )
        if approval.input_sha256 != expected_approval_input:
            approval_valid = False
            error(
                "release_approval.input_hash_mismatch",
                "Release approval does not bind the complete current evidence product",
                approval.approval_id,
            )
        if approval.decision != "approved":
            approval_valid = False
            error(
                "release_approval.not_approved",
                "Only approved release receipts can authorize release",
                approval.approval_id,
            )
        if approval.reviewer_id in approved_reviewers:
            approval_valid = False
            error(
                "release_approval.reviewer_duplicate",
                "Release approvals require distinct human reviewer identities",
                approval.approval_id,
            )
        approved_reviewers.add(approval.reviewer_id)
        approval_trust = (trusted_release_approval_keys or {}).get(
            approval.reviewer_id
        )
        if approval_trust is None:
            approval_valid = False
            error(
                "release_approval.trust_anchor_missing",
                "Release approver has no explicit trusted key",
                approval.approval_id,
            )
        else:
            approval_key_id, approval_key = approval_trust
            if approval.attestation.key_id != approval_key_id:
                approval_valid = False
                error(
                    "release_approval.signer_key_id_mismatch",
                    "ReleaseApproval key ID differs from the approver trust registry",
                    approval.approval_id,
                )
            try:
                verify_canonical_payload(
                    approval.model_dump(mode="python", exclude={"attestation"}),
                    approval.attestation,
                    trusted_keyring={approval_key_id: approval_key},
                    expected_purpose="release-approval:v1",
                )
            except AttestationError as attestation_error:
                approval_valid = False
                error(
                    "release_approval.signature_invalid",
                    str(attestation_error),
                    approval.approval_id,
                )
        if approval_valid:
            valid_approval_ids.add(approval.approval_id)

    def require_reviews(
        record_id: str,
        receipt_ids: tuple[str, ...],
        required_types: set[str],
    ) -> None:
        present: set[str] = set()
        human_present: set[str] = set()
        human_reviewers: defaultdict[str, set[str]] = defaultdict(set)
        model_reviewers: defaultdict[str, set[str]] = defaultdict(set)
        for receipt_id in receipt_ids:
            receipt = review_receipts.get(receipt_id)
            if receipt is None:
                error(
                    "review.receipt_dangling",
                    f"Unknown ReviewReceipt {receipt_id}",
                    record_id,
                )
                continue
            if receipt.record_id != record_id:
                error(
                    "review.receipt_record_mismatch",
                    "ReviewReceipt is bound to a different record",
                    record_id,
                )
                continue
            if (
                receipt.decision == "passed"
                and receipt.review_receipt_id in valid_receipt_ids
            ):
                present.add(receipt.review_type)
                if receipt.reviewer_kind == "human":
                    human_present.add(receipt.review_type)
                    human_reviewers[receipt.review_type].add(receipt.reviewer_id)
                elif receipt.reviewer_kind == "model_assisted":
                    model_reviewers[receipt.review_type].add(receipt.reviewer_id)
        missing = sorted(required_types.difference(present))
        if missing:
            error(
                "review.required_receipt_missing",
                f"Missing passed review types: {', '.join(missing)}",
                record_id,
            )
        human_required = required_types.intersection(
            {
                "full_text_scope",
                "source_span_resolution",
                "independence_grouping",
                "claim_entailment",
                "task_applicability",
                "decision_permission",
                "scope_assertion",
            }
        )
        missing_human = sorted(human_required.difference(human_present))
        if missing_human:
            error(
                "review.human_cosign_missing",
                "Missing human co-sign for review types: "
                + ", ".join(missing_human),
                record_id,
            )
        for review_type in required_types:
            if model_reviewers[review_type] and not all(
                reviewer_id not in model_reviewers[review_type]
                for reviewer_id in human_reviewers[review_type]
            ):
                error(
                    "review.human_cosigner_not_distinct",
                    "Human co-signer identity must differ from model-assisted reviewer",
                    record_id,
                )

    leaf_ids = {
        leaf.question_leaf_id for leaf in bundle.research_brief.question_tree
    }
    expected_queries: dict[str, PlannedQuery] = {}
    try:
        if active_policy is None:
            raise ValueError("active keyed policy is missing")
        planned_queries = DeepSearchPlanner(active_policy).plan(bundle.research_brief)
    except (PermissionError, RuntimeError, ValueError) as planning_error:
        error(
            "search.plan_invalid",
            f"ResearchBrief cannot be reproduced by the active planner: {planning_error}",
        )
    else:
        expected_queries = {item.query_id: item for item in planned_queries}
    successful_route_matrix = {
        (run.question_leaf_id, run.route)
        for run in bundle.search_runs
        if run.brief_id == bundle.research_brief.brief_id
        and run.stop_reason in {"completed", "no_results"}
    }
    for leaf_id in leaf_ids:
        for required_route in set(bundle.research_brief.required_search_routes):
            if (leaf_id, required_route) not in successful_route_matrix:
                error(
                    "search.required_route_missing",
                    f"QuestionLeaf {leaf_id} lacks completed route {required_route.value}",
                    leaf_id,
                )

    for run in bundle.search_runs:
        if run.brief_id != bundle.research_brief.brief_id:
            error("search.brief_dangling", "SearchRun references another brief", run.search_run_id)
        if active_policy is not None:
            try:
                active_policy.verify_receipt(run.policy_receipt)
            except (AttestationError, RuntimeError) as receipt_error:
                error(
                    "search.policy_signature_invalid",
                    str(receipt_error),
                    run.search_run_id,
                )
            recomputed_query_decision = active_policy.inspect_query(run.exact_query)
            if (
                run.policy_receipt.decision != recomputed_query_decision.decision
                or run.policy_receipt.matched_categories
                != recomputed_query_decision.matched_categories
            ):
                error(
                    "search.policy_decision_mismatch",
                    "SearchRun policy outcome differs from active-policy recomputation",
                    run.search_run_id,
                )
        if run.policy_receipt.decision != "allowed":
            error("search.policy_not_allowed", "SearchRun was not policy-allowed", run.search_run_id)
        if run.policy_receipt.policy_hash != bundle.research_brief.policy_hash:
            error(
                "search.policy_hash_mismatch",
                "SearchRun policy receipt does not match the ResearchBrief",
                run.search_run_id,
            )
        if run.policy_receipt.policy_version != "external-evidence-scope:v1":
            error(
                "search.policy_version_mismatch",
                "SearchRun policy receipt version is unsupported",
                run.search_run_id,
            )
        if (
            run.policy_receipt.subject_id != run.planned_query_id
            or run.policy_receipt.subject_sha256 != run.exact_query_sha256
        ):
            error(
                "search.policy_subject_mismatch",
                "SearchRun policy receipt is not bound to its exact planned query",
                run.search_run_id,
            )
        if run.question_leaf_id not in leaf_ids:
            error(
                "search.question_leaf_dangling",
                "SearchRun references an unknown QuestionLeaf",
                run.search_run_id,
            )
        expected_query = expected_queries.get(run.planned_query_id)
        if expected_query is None:
            error(
                "search.planned_query_unknown",
                "SearchRun does not reference a query reproduced from the ResearchBrief",
                run.search_run_id,
            )
        elif (
            run.question_leaf_id != expected_query.question_leaf_id
            or run.route != expected_query.route
            or run.exact_query != expected_query.query
            or run.filters != expected_query.filters
        ):
            error(
                "search.plan_binding_mismatch",
                "SearchRun leaf, route, query, or filters differ from the reproduced plan",
                run.search_run_id,
            )
        if run.policy_receipt.stage != "search_run":
            error(
                "search.policy_stage_mismatch",
                "SearchRun policy receipt must be issued for the search_run stage",
                run.search_run_id,
            )
        if run.policy_receipt.issued_at != run.executed_at:
            error(
                "search.policy_time_mismatch",
                "SearchRun receipt time must equal the recorded execution time",
                run.search_run_id,
            )
        if run.exact_query_sha256 != text_sha256(run.exact_query):
            error(
                "search.query_hash_mismatch",
                "SearchRun exact-query hash mismatch",
                run.search_run_id,
            )
        if not set(run.accepted_result_ids).issubset(run.result_ids):
            error(
                "search.accepted_result_not_in_ledger",
                "Accepted SearchRun results must be present in result_ids",
                run.search_run_id,
            )
        if len(run.result_ids) != len(set(run.result_ids)):
            error(
                "search.duplicate_result_id",
                "SearchRun result IDs must be unique",
                run.search_run_id,
            )
        if len(run.result_ids) != (
            len(run.accepted_result_ids)
            + run.excluded_result_count
            + run.duplicate_result_count
        ):
            error(
                "search.result_count_mismatch",
                "SearchRun accepted, excluded, and duplicate counts must conserve the result ledger",
                run.search_run_id,
            )
        expected_snapshot = content_sha256(
            {
                "search_run_id": run.search_run_id,
                "brief_id": run.brief_id,
                "provider": run.provider,
                "planned_query_id": run.planned_query_id,
                "question_leaf_id": run.question_leaf_id,
                "route": run.route.value,
                "filters": run.filters,
                "executed_at": run.executed_at,
                "exact_query_sha256": run.exact_query_sha256,
                "result_ids": list(run.result_ids),
                "accepted_result_ids": list(run.accepted_result_ids),
                "excluded_count": run.excluded_result_count,
                "duplicate_count": run.duplicate_result_count,
                "attempt_count": run.attempt_count,
                "error_code": run.error_code,
                "stop_reason": run.stop_reason,
            }
        )
        if run.snapshot_hash != expected_snapshot:
            error(
                "search.snapshot_hash_mismatch",
                "SearchRun snapshot hash does not match its result ledger",
                run.search_run_id,
            )
        if run.stop_reason == "provider_error":
            if not run.error_code or run.result_ids or run.accepted_result_ids:
                error(
                    "search.provider_error_ledger_invalid",
                    "Provider-error SearchRun requires an opaque error code and no results",
                    run.search_run_id,
                )
        elif run.error_code is not None:
            error(
                "search.unexpected_error_code",
                "Successful SearchRun must not carry a provider error code",
                run.search_run_id,
            )

    referenced_scope_assertion_ids: set[str] = set()
    for assertion in bundle.scope_assertions:
        require_reviews(
            assertion.scope_assertion_id,
            assertion.review_receipt_ids,
            {"scope_assertion"},
        )
        if assertion.assertion_status != "verified":
            error(
                "scope_assertion.not_verified",
                "Only verified ScopeAssertions can enter a release",
                assertion.scope_assertion_id,
            )
        if (
            assertion.reviewed_at > review_reference_time
            or (
                assertion.expires_at is not None
                and assertion.expires_at < review_reference_time
            )
        ):
            error(
                "scope_assertion.validity_window_mismatch",
                "ScopeAssertion is not valid at the release/reference time",
                assertion.scope_assertion_id,
            )
        if assertion.verification_receipt_sha256 == "0" * 64:
            error(
                "scope_assertion.verification_artifact_placeholder",
                "ScopeAssertion verification artifact hash cannot be a placeholder",
                assertion.scope_assertion_id,
            )
        if assertion.excluded_subject_present is not False:
            error(
                "scope_assertion.excluded_subject_not_cleared",
                "ScopeAssertion must explicitly clear excluded subject presence",
                assertion.scope_assertion_id,
            )
        if assertion.subject_scope not in {
            SubjectScope.GENERIC_PROTEIN,
            SubjectScope.NONVIRAL_PROTEIN,
        }:
            error(
                "scope_assertion.scope_not_releasable",
                "Mixed, unknown, or excluded source scopes cannot be released",
                assertion.scope_assertion_id,
            )

    quarantined_identities = {
        value.strip().casefold()
        for value in bundle.research_brief.source_quarantine_ids
        if value.strip()
    }
    hit_pairs: set[tuple[str, str]] = set()
    hit_ids_by_run: defaultdict[str, set[str]] = defaultdict(set)
    accepted_result_ids_by_run: defaultdict[str, set[str]] = defaultdict(set)
    duplicate_result_ids_by_run: defaultdict[str, set[str]] = defaultdict(set)
    for hit in bundle.allowed_search_hits:
        hit_identities = {
            hit.search_hit_id.casefold(),
            hit.result_id.casefold(),
            hit.artifact_id.casefold(),
            hit.provider_record_id.casefold(),
            hit.url.casefold(),
            f"{hit.provider}:{hit.provider_record_id}".casefold(),
        }
        if hit.doi:
            hit_identities.update(
                {
                    hit.doi.casefold(),
                    f"doi:{hit.doi}".casefold(),
                }
            )
        if quarantined_identities.intersection(hit_identities):
            error(
                "search_hit.source_quarantined",
                "SearchHit identity intersects the benchmark/source quarantine",
                hit.search_hit_id,
            )
        pair = (hit.search_run_id, hit.result_id)
        if pair in hit_pairs:
            error(
                "search_hit.run_result_duplicate",
                "Each SearchRun result can produce at most one AllowedSearchHit",
                hit.search_hit_id,
            )
        hit_pairs.add(pair)
        hit_ids_by_run[hit.search_run_id].add(hit.result_id)
        search_run = search_runs.get(hit.search_run_id)
        if search_run is None:
            error(
                "search_hit.search_run_dangling",
                "AllowedSearchHit references an unknown SearchRun",
                hit.search_hit_id,
            )
        else:
            if hit.provider != search_run.provider:
                error(
                    "search_hit.provider_mismatch",
                    "AllowedSearchHit provider differs from its SearchRun",
                    hit.search_hit_id,
                )
            if hit.result_id not in search_run.result_ids:
                error(
                    "search_hit.result_not_in_ledger",
                    "AllowedSearchHit result is absent from its SearchRun ledger",
                    hit.search_hit_id,
                )
            if hit.disposition == "accepted":
                accepted_result_ids_by_run[hit.search_run_id].add(hit.result_id)
                if hit.result_id not in search_run.accepted_result_ids:
                    error(
                        "search_hit.accepted_disposition_mismatch",
                        "Accepted hit is not marked accepted by its SearchRun",
                        hit.search_hit_id,
                    )
            else:
                duplicate_result_ids_by_run[hit.search_run_id].add(hit.result_id)
                if hit.result_id in search_run.accepted_result_ids:
                    error(
                        "search_hit.duplicate_disposition_mismatch",
                        "Duplicate-publication hit cannot also be accepted",
                        hit.search_hit_id,
                    )
        assertion = scope_assertions.get(hit.scope_assertion_id)
        if assertion is None:
            error(
                "search_hit.scope_assertion_dangling",
                "AllowedSearchHit references an unknown ScopeAssertion",
                hit.search_hit_id,
            )
            continue
        referenced_scope_assertion_ids.add(assertion.scope_assertion_id)
        if assertion.artifact_id != hit.artifact_id:
            error(
                "search_hit.scope_artifact_mismatch",
                "AllowedSearchHit and ScopeAssertion artifact identities differ",
                hit.search_hit_id,
            )
        expected_hit_id = stable_id(
            "search-hit",
            hit.search_run_id,
            hit.result_id,
            hit.scope_assertion_id,
        )
        if hit.search_hit_id != expected_hit_id:
            error(
                "search_hit.id_mismatch",
                "AllowedSearchHit ID does not match its canonical provenance",
                hit.search_hit_id,
            )
        if active_policy is not None:
            decision = active_policy.decide_metadata(
                artifact_id=hit.artifact_id,
                title=hit.title,
                abstract=hit.abstract,
                subjects=hit.subjects,
                assertion=assertion,
                as_of=hit.policy_receipt.issued_at,
            )
            try:
                active_policy.verify_receipt(hit.policy_receipt)
            except (AttestationError, RuntimeError) as receipt_error:
                error(
                    "search_hit.policy_signature_invalid",
                    str(receipt_error),
                    hit.search_hit_id,
                )
            if (
                decision.decision != "allowed"
                or hit.policy_receipt.decision != decision.decision
                or hit.policy_receipt.matched_categories
                != decision.matched_categories
            ):
                error(
                    "search_hit.policy_decision_mismatch",
                    "AllowedSearchHit differs from active-policy recomputation",
                    hit.search_hit_id,
                )
        if (
            hit.policy_receipt.stage != "search_hit"
            or hit.policy_receipt.subject_id != hit.search_hit_id
            or hit.policy_receipt.subject_sha256
            != _search_hit_scope_metadata_sha256(hit, assertion)
        ):
            error(
                "search_hit.policy_subject_mismatch",
                "AllowedSearchHit policy receipt does not bind its metadata and assertion",
                hit.search_hit_id,
            )
        if search_run is not None and (
            hit.policy_receipt.issued_at != search_run.executed_at
        ):
            error(
                "search_hit.policy_time_mismatch",
                "SearchHit receipt time must equal its SearchRun execution time",
                hit.search_hit_id,
            )
        if assertion.reviewed_at > hit.policy_receipt.issued_at:
            error(
                "search_hit.scope_assertion_postdates_use",
                "ScopeAssertion must be reviewed before it authorizes a SearchHit",
                hit.search_hit_id,
            )

    for run in bundle.search_runs:
        observed_hit_ids = hit_ids_by_run[run.search_run_id]
        if accepted_result_ids_by_run[run.search_run_id] != set(
            run.accepted_result_ids
        ):
            error(
                "search_hit.accepted_set_mismatch",
                "Allowed accepted-hit set must exactly match SearchRun accepted results",
                run.search_run_id,
            )
        if len(duplicate_result_ids_by_run[run.search_run_id]) != (
            run.duplicate_result_count
        ):
            error(
                "search_hit.duplicate_count_mismatch",
                "Allowed duplicate-hit records must match SearchRun duplicate count",
                run.search_run_id,
            )
        if len(set(run.result_ids).difference(observed_hit_ids)) != (
            run.excluded_result_count
        ):
            error(
                "search_hit.excluded_count_mismatch",
                "SearchRun excluded count must equal results without retained allowed hits",
                run.search_run_id,
            )

    acquired_hit_owners: dict[str, str] = {}
    for publication in bundle.publications:
        require_reviews(
            publication.publication_id,
            publication.review_receipt_ids,
            {"metadata_identity", "full_text_scope"},
        )
        if publication.source_scope not in {"generic_protein", "nonviral_protein"}:
            error(
                "publication.scope_invalid",
                "Only verified generic/nonviral publications can enter this evidence product",
                publication.publication_id,
            )
        publication_identities = {publication.publication_id.casefold()}
        publication_identities.update(
            value.casefold() for value in publication.identifier_aliases
        )
        if publication.doi:
            publication_identities.update(
                {
                    publication.doi.casefold(),
                    f"doi:{publication.doi.casefold()}",
                }
            )
        if quarantined_identities.intersection(publication_identities):
            error(
                "publication.source_quarantined",
                "Publication identity intersects the benchmark/source quarantine",
                publication.publication_id,
            )
        publication_assertion = scope_assertions.get(publication.scope_assertion_id)
        if publication_assertion is None:
            error(
                "publication.scope_assertion_dangling",
                "Publication references an unknown ScopeAssertion",
                publication.publication_id,
            )
        else:
            referenced_scope_assertion_ids.add(
                publication_assertion.scope_assertion_id
            )
            if publication.source_scope != publication_assertion.subject_scope.value:
                error(
                    "publication.scope_assertion_mismatch",
                    "Publication source scope differs from its ScopeAssertion",
                    publication.publication_id,
                )
        if not publication.metadata_verified or publication.full_text_status != "verified":
            error(
                "publication.not_source_verified",
                "Released publications require verified metadata and full text",
                publication.publication_id,
            )
        if publication.version_status in {"retracted", "expression_of_concern"}:
            error(
                "publication.status_blocked",
                "Blocked publication status cannot support a release",
                publication.publication_id,
            )
        seen_acquisition_ids: set[str] = set()
        for acquisition in publication.acquisitions:
            acquisition_id = acquisition.search_hit_id
            if acquisition_id in seen_acquisition_ids:
                error(
                    "publication.acquisition_duplicate",
                    "Publication acquisition links must be unique",
                    publication.publication_id,
                )
            seen_acquisition_ids.add(acquisition_id)
            hit = search_hits.get(acquisition.search_hit_id)
            if hit is None:
                error(
                    "publication.search_hit_dangling",
                    f"Unknown acquisition SearchHit {acquisition.search_hit_id}",
                    publication.publication_id,
                )
                continue
            previous_owner = acquired_hit_owners.get(hit.search_hit_id)
            if previous_owner is not None and previous_owner != publication.publication_id:
                error(
                    "publication.search_hit_reused",
                    "One AllowedSearchHit cannot belong to multiple publications",
                    publication.publication_id,
                )
            acquired_hit_owners[hit.search_hit_id] = publication.publication_id
            canonical_hit = search_hits.get(publication.canonical_search_hit_id)
            if canonical_hit is not None and (
                _allowed_hit_publication_key(hit)
                != _allowed_hit_publication_key(canonical_hit)
            ):
                error(
                    "publication.acquisition_identity_mismatch",
                    "Acquisition hits do not resolve to one canonical publication",
                    publication.publication_id,
                )
        canonical_hit = search_hits.get(publication.canonical_search_hit_id)
        if canonical_hit is None:
            error(
                "publication.canonical_search_hit_dangling",
                "Publication canonical SearchHit is missing",
                publication.publication_id,
            )
        elif publication.canonical_search_hit_id not in seen_acquisition_ids:
            error(
                "publication.canonical_search_hit_not_acquired",
                "Canonical SearchHit must appear in publication acquisitions",
                publication.publication_id,
            )
        else:
            if canonical_hit.disposition != "accepted":
                error(
                    "publication.canonical_search_hit_not_accepted",
                    "Canonical publication metadata must come from an accepted hit",
                    publication.publication_id,
                )
            if canonical_hit.scope_assertion_id != publication.scope_assertion_id:
                error(
                    "publication.canonical_scope_mismatch",
                    "Publication scope must derive from its canonical SearchHit",
                    publication.publication_id,
                )
            publication_aliases = {
                publication.publication_id.casefold(),
                *(item.casefold() for item in publication.identifier_aliases),
            }
            if canonical_hit.artifact_id.casefold() not in publication_aliases:
                error(
                    "publication.canonical_identity_mismatch",
                    "Publication identity does not include its canonical SearchHit artifact",
                    publication.publication_id,
                )
            if (publication.doi or "").casefold() != (
                canonical_hit.doi or ""
            ).casefold():
                error(
                    "publication.canonical_doi_mismatch",
                    "Publication DOI differs from its canonical SearchHit",
                    publication.publication_id,
                )

    if set(acquired_hit_owners) != set(search_hits):
        error(
            "publication.search_hit_coverage_mismatch",
            "Every retained AllowedSearchHit must be acquired by exactly one Publication",
        )

    for span in bundle.source_spans:
        require_reviews(
            span.source_span_id,
            span.review_receipt_ids,
            {"source_span_resolution"},
        )
        publication = publications.get(span.publication_id)
        if publication is None:
            error(
                "span.publication_dangling",
                f"Unknown publication {span.publication_id}",
                span.source_span_id,
            )
        if span.span_text is not None and text_sha256(span.span_text) != span.normalized_span_sha256:
            error("span.hash_mismatch", "SourceSpan text hash mismatch", span.source_span_id)
        if not span.resolved_against_artifact or not span.independently_checked:
            error(
                "span.not_verified",
                "Released SourceSpan must be resolved and independently checked",
                span.source_span_id,
            )
        span_assertion = scope_assertions.get(span.scope_assertion_id)
        if span_assertion is None:
            error(
                "span.scope_assertion_dangling",
                "SourceSpan references an unknown ScopeAssertion",
                span.source_span_id,
            )
        else:
            referenced_scope_assertion_ids.add(span_assertion.scope_assertion_id)
            if span_assertion.artifact_id != span.artifact_id:
                error(
                    "span.scope_artifact_mismatch",
                    "SourceSpan and ScopeAssertion artifact identities differ",
                    span.source_span_id,
                )
            if publication is not None and (
                span.scope_assertion_id != publication.scope_assertion_id
            ):
                error(
                    "span.publication_scope_mismatch",
                    "SourceSpan and Publication use different ScopeAssertions",
                    span.source_span_id,
                )
        if span.instruction_markers or instruction_like_markers(span.support_paraphrase):
            error(
                "span.untrusted_instruction",
                "Instruction-like source content cannot enter evidence synthesis",
                span.source_span_id,
            )

    if referenced_scope_assertion_ids != set(scope_assertions):
        error(
            "scope_assertion.coverage_mismatch",
            "ScopeAssertion set must exactly cover retained hits/publications/spans",
        )

    for group in bundle.evidence_groups:
        require_reviews(
            group.evidence_group_id,
            group.review_receipt_ids,
            {"independence_grouping"},
        )
        if not group.source_span_ids:
            error("group.empty", "EvidenceGroup requires SourceSpan links", group.evidence_group_id)
            continue
        source_publications: set[str] = set()
        source_families: set[str] = set()
        for span_id in group.source_span_ids:
            span = spans.get(span_id)
            if span is None:
                error(
                    "group.span_dangling",
                    f"Unknown SourceSpan {span_id}",
                    group.evidence_group_id,
                )
            else:
                source_publications.add(span.publication_id)
                publication = publications.get(span.publication_id)
                if publication is not None:
                    source_families.add(publication.study_family_id)
        if len(source_publications) > 1:
            error(
                "group.mixed_publications",
                "One EvidenceGroup must remain within one publication",
                group.evidence_group_id,
            )
        if len(source_families) == 1 and group.independence_group not in source_families:
            error(
                "group.independence_family_mismatch",
                "EvidenceGroup independence identity must match its publication study family",
                group.evidence_group_id,
            )
        if not group.verified_by:
            error(
                "group.unverified",
                "EvidenceGroup requires an independent verifier",
                group.evidence_group_id,
            )

    for claim in bundle.atomic_claims:
        if not claim.evidence_group_ids:
            error("claim.no_evidence", "AtomicClaim requires EvidenceGroup links", claim.claim_id)
        for group_id in claim.evidence_group_ids:
            group = groups.get(group_id)
            if group is None:
                error(
                    "claim.group_dangling",
                    f"Unknown EvidenceGroup {group_id}",
                    claim.claim_id,
                )
            elif claim.claim_status == "supported" and group.completeness != "complete":
                error(
                    "claim.partial_evidence",
                    "Supported claims require complete EvidenceGroups",
                    claim.claim_id,
                )
        claim_stances = {
            groups[group_id].stance
            for group_id in claim.evidence_group_ids
            if group_id in groups
        }
        if claim.claim_status == "supported" and "supports" not in claim_stances:
            error(
                "claim.support_missing",
                "A supported claim requires at least one supporting EvidenceGroup",
                claim.claim_id,
            )
        if claim.claim_status == "supported" and {
            "refutes",
            "limits",
        }.intersection(claim_stances):
            error(
                "claim.conflict_hidden",
                "A claim with limiting or refuting evidence cannot be labeled supported",
                claim.claim_id,
            )
        if claim.claim_status == "contested" and not (
            "supports" in claim_stances
            and bool({"refutes", "limits"}.intersection(claim_stances))
        ):
            error(
                "claim.contested_evidence_incomplete",
                "A contested claim requires both support and limiting/refuting evidence",
                claim.claim_id,
            )
        if instruction_like_markers(claim.statement):
            error(
                "claim.untrusted_instruction",
                "Instruction-like claim content is rejected",
                claim.claim_id,
            )

    for logic in bundle.logic_units:
        require_reviews(
            logic.logic_unit_id,
            logic.review_receipt_ids,
            {"claim_entailment", "task_applicability"},
        )
        for claim_id in (*logic.premise_claim_ids, *logic.counterclaim_ids):
            if claim_id not in claims:
                error(
                    "logic.claim_dangling",
                    f"Unknown AtomicClaim {claim_id}",
                    logic.logic_unit_id,
                )
        if logic.question_leaf_id not in leaf_ids:
            error(
                "logic.question_leaf_dangling",
                "LogicUnit references an unknown QuestionLeaf",
                logic.logic_unit_id,
            )
        coverage_routes: set[SearchRoute] = set()
        for search_run_id in logic.search_coverage_run_ids:
            search_run = search_runs.get(search_run_id)
            if search_run is None:
                error(
                    "logic.search_coverage_dangling",
                    f"Unknown search-coverage run {search_run_id}",
                    logic.logic_unit_id,
                )
                continue
            if (
                search_run.question_leaf_id != logic.question_leaf_id
                or search_run.stop_reason not in {"completed", "no_results"}
            ):
                error(
                    "logic.search_coverage_invalid",
                    "LogicUnit search coverage must be successful and leaf-specific",
                    logic.logic_unit_id,
                )
                continue
            coverage_routes.add(search_run.route)
        missing_logic_routes = set(
            bundle.research_brief.required_search_routes
        ).difference(coverage_routes)
        if missing_logic_routes:
            error(
                "logic.search_coverage_incomplete",
                "LogicUnit review closure lacks routes: "
                + ", ".join(sorted(item.value for item in missing_logic_routes)),
                logic.logic_unit_id,
            )
        if not logic.premise_claim_ids:
            error(
                "logic.premise_missing",
                "LogicUnit requires at least one premise claim",
                logic.logic_unit_id,
            )
        quality = logic.scientific_quality
        support_families: set[str] = set()
        counterevidence_present = bool(logic.counterclaim_ids)
        for claim_id in logic.premise_claim_ids:
            claim = claims.get(claim_id)
            if claim is None:
                continue
            if claim.claim_status == "insufficient" and logic.operator != "abstain":
                error(
                    "logic.insufficient_premise_not_abstained",
                    "Insufficient claims may only support an explicit abstention LogicUnit",
                    logic.logic_unit_id,
                )
            if claim.claim_status == "contested" and logic.operator not in {
                "qualify",
                "conflict_summary",
                "abstain",
            }:
                error(
                    "logic.contested_premise_unqualified",
                    "Contested claims require qualification, conflict summary, or abstention",
                    logic.logic_unit_id,
                )
            for group_id in claim.evidence_group_ids:
                group = groups.get(group_id)
                if group is None:
                    continue
                if group.stance in {"refutes", "limits"}:
                    counterevidence_present = True
                if group.stance != "supports":
                    continue
                for span_id in group.source_span_ids:
                    span = spans.get(span_id)
                    publication = (
                        publications.get(span.publication_id)
                        if span is not None
                        else None
                    )
                    if publication is not None:
                        support_families.add(publication.study_family_id)
        if quality.independent_support_count != len(support_families):
            error(
                "logic.independence_count_mismatch",
                "ScientificQuality independent-support count must be derived from study families",
                logic.logic_unit_id,
            )
        if (
            quality.counterevidence_status == "searched_found"
            and not counterevidence_present
        ):
            error(
                "logic.counterevidence_evidence_missing",
                "Counterevidence marked found must link a limiting/refuting group or counterclaim",
                logic.logic_unit_id,
            )
        if not quality.identity_verified or not quality.span_verified:
            error(
                "logic.identity_or_span_unverified",
                "LogicUnit requires verified identity and spans",
                logic.logic_unit_id,
            )
        if quality.entailment_status != "verified":
            error(
                "logic.entailment_unverified",
                "LogicUnit claim entailment must be verified",
                logic.logic_unit_id,
            )
        if quality.counterevidence_status == "not_searched":
            error(
                "logic.counterevidence_not_searched",
                "LogicUnit requires an explicit counterevidence search",
                logic.logic_unit_id,
            )
        if instruction_like_markers(logic.retrieval_text):
            error(
                "logic.untrusted_instruction",
                "Instruction-like retrieval text is rejected",
                logic.logic_unit_id,
            )

    entailed_premise_claim_ids = {
        claim_id
        for logic in bundle.logic_units
        for claim_id in logic.premise_claim_ids
    }
    for claim in bundle.atomic_claims:
        if (
            claim.claim_status == "supported"
            and claim.claim_id not in entailed_premise_claim_ids
        ):
            error(
                "claim.logic_entailment_missing",
                "Supported AtomicClaim must be covered by a reviewed LogicUnit premise",
                claim.claim_id,
            )

    for card in bundle.decision_cards:
        if not card.logic_unit_ids:
            error(
                "card.logic_missing",
                "KnowledgeDecisionCard requires at least one LogicUnit",
                card.decision_card_id,
            )
        for logic_id in card.logic_unit_ids:
            logic = logic_units.get(logic_id)
            if logic is None:
                error(
                    "card.logic_dangling",
                    f"Unknown LogicUnit {logic_id}",
                    card.decision_card_id,
                )
            elif logic.task_route != card.task_route:
                error(
                    "card.logic_route_mismatch",
                    "DecisionCard and LogicUnit task routes must match",
                    card.decision_card_id,
                )
            elif logic.question_leaf_id != card.question_leaf_id:
                error(
                    "card.logic_leaf_mismatch",
                    "DecisionCard and LogicUnit question leaves must match",
                    card.decision_card_id,
                )
        if card.question_leaf_id not in {
            leaf.question_leaf_id for leaf in bundle.research_brief.question_tree
        }:
            error(
                "card.question_leaf_dangling",
                "KnowledgeDecisionCard references an unknown QuestionLeaf",
                card.decision_card_id,
            )
        if card.permission != DecisionPermission.EXPLANATION_ONLY:
            require_reviews(
                card.decision_card_id,
                card.review_receipt_ids,
                {"decision_permission"},
            )
            for logic_id in card.logic_unit_ids:
                logic = logic_units.get(logic_id)
                if logic is None:
                    continue
                for claim_id in logic.premise_claim_ids:
                    claim = claims.get(claim_id)
                    if claim is not None and claim.claim_status != "supported":
                        error(
                            "card.unsupported_premise",
                            "Elevated permissions require supported premise claims",
                            card.decision_card_id,
                        )
        if card.permission in {
            DecisionPermission.CANDIDATE_RERANKING,
            DecisionPermission.HARD_GATE,
        }:
            error(
                "card.selection_permission_not_releasable",
                "Selection permissions require a manifest-addressed calibration record, which is not yet supported",
                card.decision_card_id,
            )
            if card.task_route != "candidate_ranking" or not card.candidate_feature:
                error(
                    "card.selection_route_or_feature",
                    "Selection permissions require candidate_ranking and a candidate feature",
                    card.decision_card_id,
                )
            if not card.calibration_id or card.calibration_status != "validated":
                error(
                    "card.calibration_required",
                    "Selection permissions require validated task calibration",
                    card.decision_card_id,
                )
            if card.benchmark_overlap_status != "clear":
                error(
                    "card.benchmark_overlap",
                    "Selection permissions require clear benchmark overlap status",
                    card.decision_card_id,
                )
            for logic_id in card.logic_unit_ids:
                logic = logic_units.get(logic_id)
                if logic is None:
                    continue
                if logic.scientific_quality.conflict_status == "unresolved":
                    error(
                        "card.unresolved_conflict",
                        "Selection permissions cannot use unresolved conflicts",
                        card.decision_card_id,
                    )
                if logic.task_applicability.directness != "direct":
                    error(
                        "card.indirect_applicability",
                        "Selection permissions require direct task applicability",
                        card.decision_card_id,
                    )
        if (
            card.permission == DecisionPermission.HARD_GATE
            and len(set(card.human_approval_ids)) < 2
        ):
            error(
                "card.hard_gate_approvals",
                "Hard-gate permission requires two distinct human approvals",
                card.decision_card_id,
            )

    manifest = bundle.release_manifest
    if manifest is None:
        warning("release.manifest_missing", "Bundle is validatable but not releasable")
    else:
        expected_records = _record_entries(
            bundle.model_copy(update={"release_manifest": None})
        )
        if manifest.records != expected_records:
            error("release.record_hash_mismatch", "ReleaseManifest records do not match bundle")
        dependency_graph = {
            item.record_id: list(item.dependency_ids) for item in expected_records
        }
        expected_dependency_hash = content_sha256(dependency_graph)
        if manifest.dependency_graph_sha256 != expected_dependency_hash:
            error("release.dependency_hash_mismatch", "Dependency graph hash mismatch")
        if manifest.policy_hash != bundle.research_brief.policy_hash:
            error("release.policy_hash_mismatch", "Release policy hash mismatch")
        if manifest.denied_path_operations != 0:
            error("release.denied_path_access", "Release reports denied path operations")
        expected_release_id = _release_id(
            release_version=manifest.release_version,
            status=manifest.status,
            created_at=manifest.created_at,
            parent_release_id=manifest.parent_release_id,
            policy_hash=manifest.policy_hash,
            records=expected_records,
            dependency_graph_sha256=expected_dependency_hash,
            excluded_result_count=manifest.excluded_result_count,
        )
        if manifest.release_id != expected_release_id:
            error(
                "release.id_mismatch",
                "Release ID does not match the unsigned release core",
            )
        if manifest.validator_version != VALIDATOR_VERSION:
            error(
                "release.validator_version_mismatch",
                "Release manifest uses an unsupported validator version",
            )
        if manifest.retrieval_record_types != (
            "atomic_claim",
            "logic_unit",
            "knowledge_decision_card",
        ):
            error(
                "release.runtime_types_invalid",
                "Release runtime record types are not the fixed safe allowlist",
            )
        if manifest.status == "released" and not manifest.release_approval_ids:
            error("release.approval_missing", "Released manifest requires approval receipts")
        if len(manifest.release_approval_ids) != len(
            set(manifest.release_approval_ids)
        ):
            error(
                "release.approval_duplicate",
                "Release approval IDs must be distinct",
            )
        if set(manifest.release_approval_ids) != valid_approval_ids:
            error(
                "release.approval_receipt_mismatch",
                "Release approval IDs must exactly match valid bundle-bound human approvals",
            )
        for approval_id in manifest.release_approval_ids:
            approval = release_approvals.get(approval_id)
            if approval is None:
                continue
            if (
                approval.target_release_version != manifest.release_version
                or approval.target_status != manifest.status
                or approval.target_created_at != manifest.created_at
                or approval.target_parent_release_id != manifest.parent_release_id
            ):
                error(
                    "release.approval_target_mismatch",
                    "Release approval cannot replay across release core fields",
                    approval_id,
                )
        expected_excluded_count = sum(
            item.excluded_result_count for item in bundle.search_runs
        )
        if manifest.excluded_result_count != expected_excluded_count:
            error(
                "release.excluded_count_mismatch",
                "Release excluded-result count does not match SearchRun ledgers",
            )

    counts = {
        "search_runs": len(search_runs),
        "scope_assertions": len(scope_assertions),
        "allowed_search_hits": len(search_hits),
        "publications": len(publications),
        "source_spans": len(spans),
        "evidence_groups": len(groups),
        "atomic_claims": len(claims),
        "logic_units": len(logic_units),
        "decision_cards": len(cards),
        "review_receipts": len(review_receipts),
        "release_approvals": len(release_approvals),
        "errors": sum(item.severity == "error" for item in issues),
        "warnings": sum(item.severity == "warning" for item in issues),
    }
    valid = counts["errors"] == 0
    release_ready = bool(
        valid
        and manifest is not None
        and manifest.status == "released"
        and manifest.release_approval_ids
        and set(manifest.release_approval_ids) == valid_approval_ids
    )
    return EvidenceValidationReport(
        valid=valid,
        release_ready=release_ready,
        issues=tuple(issues),
        counts=counts,
        bundle_hash=content_sha256(
            bundle.model_copy(update={"release_manifest": None})
        ),
    )


def _unique_map(
    values: Iterable[Any],
    key: str,
    *,
    error: Callable[[str, str, str | None], None],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value in values:
        identifier = str(getattr(value, key))
        if identifier in output:
            error("record.duplicate_id", f"Duplicate record ID {identifier}", identifier)
        output[identifier] = value
    return output
