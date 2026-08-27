from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from fitness_agents.config import LocalKnowledgeRootConfig
from fitness_agents.deep_research.canonical import content_sha256, stable_id, text_sha256
from fitness_agents.deep_research.cli import main as deep_research_cli
from fitness_agents.deep_research.contracts import (
    AllowedSearchHit,
    AtomicClaim,
    DecisionPermission,
    EvidenceGroup,
    EvidenceProductBundle,
    KnowledgeDecisionCard,
    LogicUnit,
    Publication,
    PublicationAcquisition,
    QuestionLeaf,
    ResearchBrief,
    ReviewReceipt,
    ScientificQuality,
    SearchBudget,
    SearchRoute,
    SearchRun,
    SourceSpan,
    TaskApplicability,
)
from fitness_agents.deep_research.export import (
    export_legacy_local_rag_bundle,
    export_native_local_rag_bundle,
)
from fitness_agents.deep_research.legacy_validator import (
    validate_legacy_runtime_bundle,
)
from fitness_agents.deep_research.pipeline import (
    DeepSearchEngine,
    DeepSearchPlanner,
    build_release_manifest,
    issue_release_approval,
    issue_review_receipt,
    release_approval_input_sha256,
    validate_evidence_product,
)
from fitness_agents.deep_research.policy import (
    ExternalEvidenceScopePolicy,
    ScopeAssertion,
    SubjectRole,
    SubjectScope,
)
from fitness_agents.deep_research.providers import (
    CrossrefSearchProvider,
    ExternalSearchResult,
    OpenAlexSearchProvider,
    ProviderCredentialMissingError,
    RetrievalAssessment,
)
from fitness_agents.local_knowledge.parsers import (
    AutoLocalParser,
    discover_local_files,
)

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
POLICY_KEY_ID = "external-policy:v1"
POLICY_KEY = b"synthetic-policy-key-material-0001"
REVIEW_KEY_ID = "evidence-reviewer:v1"
REVIEW_KEY = b"synthetic-review-key-material-0001"
RELEASE_KEY_ID = "release-reviewer:v1"
RELEASE_KEY = b"synthetic-release-key-material-001"


@pytest.fixture(autouse=True)
def _synthetic_trust_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reviewer_ids = {
        f"reviewer:{review_type}"
        for review_type in (
            "metadata_identity",
            "full_text_scope",
            "source_span_resolution",
            "independence_grouping",
            "claim_entailment",
            "task_applicability",
            "decision_permission",
            "scope_assertion",
        )
    }
    monkeypatch.setenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_ID", POLICY_KEY_ID)
    monkeypatch.setenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX", POLICY_KEY.hex())
    monkeypatch.setenv(
        "FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON",
        json.dumps(
            {
                reviewer_id: {
                    "key_id": REVIEW_KEY_ID,
                    "key_hex": REVIEW_KEY.hex(),
                }
                for reviewer_id in reviewer_ids
            }
        ),
    )
    monkeypatch.setenv(
        "FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON",
        json.dumps(
            {
                "release-reviewer:one": {
                    "key_id": RELEASE_KEY_ID,
                    "key_hex": RELEASE_KEY.hex(),
                }
            }
        ),
    )


def _policy() -> ExternalEvidenceScopePolicy:
    return ExternalEvidenceScopePolicy(
        signing_key_id=POLICY_KEY_ID,
        signing_key=POLICY_KEY,
    )


def _reviewer_keys(
    bundle: EvidenceProductBundle,
) -> dict[str, tuple[str, bytes]]:
    return {
        item.reviewer_id: (REVIEW_KEY_ID, REVIEW_KEY)
        for item in bundle.review_receipts
    }


def _release_keys(
    bundle: EvidenceProductBundle,
) -> dict[str, tuple[str, bytes]]:
    return {
        item.reviewer_id: (RELEASE_KEY_ID, RELEASE_KEY)
        for item in bundle.release_approvals
    }


def _validate(
    bundle: EvidenceProductBundle,
    policy: ExternalEvidenceScopePolicy,
):
    return validate_evidence_product(
        bundle,
        active_policy=policy,
        trusted_reviewer_keys=_reviewer_keys(bundle),
        trusted_release_approval_keys=_release_keys(bundle),
    )


def _export(bundle: EvidenceProductBundle, output_root: Path):
    return export_legacy_local_rag_bundle(
        bundle,
        output_root,
        active_policy=_policy(),
        trusted_reviewer_keys=_reviewer_keys(bundle),
        trusted_release_approval_keys=_release_keys(bundle),
    )


def _review(
    receipt_id: str,
    review_type: str,
    record,
    bundle: EvidenceProductBundle,
) -> ReviewReceipt:
    method_version = "synthetic-review:v1"
    record_id = (
        getattr(record, "source_span_id", None)
        or getattr(record, "evidence_group_id", None)
        or getattr(record, "logic_unit_id", None)
        or getattr(record, "decision_card_id", None)
        or getattr(record, "publication_id", None)
        or getattr(record, "scope_assertion_id", None)
    )
    return issue_review_receipt(
        bundle,
        review_receipt_id=receipt_id,
        review_type=review_type,
        record_id=record_id,
        reviewer_id=f"reviewer:{review_type}",
        reviewer_kind="human",
        method_version=method_version,
        decision="passed",
        reviewed_at=NOW,
        expires_at=NOW + timedelta(days=3650),
        signing_key_id=REVIEW_KEY_ID,
        signing_key=REVIEW_KEY,
    )


class _FakeProvider:
    name = "fake"

    def __init__(self, results: tuple[ExternalSearchResult, ...]) -> None:
        self.results = results
        self.calls: list[str] = []

    def search(self, query: str, *, limit: int, filters):
        self.calls.append(query)
        return self.results[:limit]


class _FailingProvider:
    name = "failing"

    def search(self, query: str, *, limit: int, filters):
        raise ConnectionError("synthetic provider outage")


class _SyntheticResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


class _SyntheticClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    def get(self, endpoint: str, *, params: dict) -> _SyntheticResponse:
        self.calls.append((endpoint, params))
        return _SyntheticResponse(self.payload)


class _FlakySyntheticClient(_SyntheticClient):
    def get(self, endpoint: str, *, params: dict) -> _SyntheticResponse:
        self.calls.append((endpoint, params))
        if len(self.calls) == 1:
            raise ConnectionError("synthetic transient provider failure")
        return _SyntheticResponse(self.payload)


def _assertion(artifact_id: str, scope: SubjectScope) -> ScopeAssertion:
    return ScopeAssertion(
        scope_assertion_id=stable_id("scope-assertion", artifact_id),
        artifact_id=artifact_id,
        subject_scope=scope,
        excluded_subject_present=False,
        roles=(SubjectRole.PRIMARY_SUBJECT,),
        assertion_status="verified",
        issuer="synthetic-reviewer",
        issuer_kind="human_review",
        source_record_id="scope-review:test",
        verification_receipt_sha256="d" * 64,
        reviewed_at=NOW,
        expires_at=NOW + timedelta(days=3650),
        review_receipt_ids=(f"review:scope:{artifact_id}",),
    )


def _brief(policy: ExternalEvidenceScopePolicy) -> ResearchBrief:
    return ResearchBrief(
        brief_id="brief:synthetic",
        research_question="How does sequence context affect nonviral enzyme stability?",
        decision_use="mechanism_explanation",
        question_tree=(
            QuestionLeaf(
                question_leaf_id="leaf:stability",
                decision_slot="mechanism",
                question="How does sequence context affect enzyme stability?",
                counterevidence_question=(
                    "When does sequence context fail to explain enzyme stability?"
                ),
                closure_rule="One verified primary source or an explicit evidence gap.",
            ),
        ),
        inclusion_criteria=("Primary studies of nonviral proteins.",),
        exclusion_criteria=("Excluded source scopes.",),
        required_search_routes=(SearchRoute.PRIMARY, SearchRoute.COUNTEREVIDENCE),
        policy_hash=policy.policy_hash,
        budget=SearchBudget(max_queries=2, max_results_per_query=10, max_publications=20),
        stop_conditions=("Both planned routes completed.",),
    )


def _search_snapshot(
    *,
    search_run_id: str,
    brief_id: str,
    provider: str,
    planned_query_id: str,
    question_leaf_id: str,
    route: SearchRoute,
    exact_query: str,
    result_ids: tuple[str, ...],
    accepted_result_ids: tuple[str, ...],
    filters: dict[str, str] | None = None,
    executed_at: datetime = NOW,
    excluded_count: int = 0,
    duplicate_count: int = 0,
    attempt_count: int = 1,
    error_code: str | None = None,
    stop_reason: str = "completed",
) -> str:
    return content_sha256(
        {
            "search_run_id": search_run_id,
            "brief_id": brief_id,
            "provider": provider,
            "planned_query_id": planned_query_id,
            "question_leaf_id": question_leaf_id,
            "route": route.value,
            "filters": filters or {},
            "executed_at": executed_at,
            "exact_query_sha256": text_sha256(exact_query),
            "result_ids": list(result_ids),
            "accepted_result_ids": list(accepted_result_ids),
            "excluded_count": excluded_count,
            "duplicate_count": duplicate_count,
            "attempt_count": attempt_count,
            "error_code": error_code,
            "stop_reason": stop_reason,
        }
    )


def _result(
    result_id: str,
    artifact_id: str,
    title: str,
    *,
    assertion: ScopeAssertion | None = None,
) -> ExternalSearchResult:
    return ExternalSearchResult(
        result_id=result_id,
        artifact_id=artifact_id,
        provider="fake",
        provider_record_id=result_id,
        title=title,
        abstract="Synthetic metadata for a policy test.",
        authors=("Researcher A",),
        year=2025,
        venue="Synthetic Journal",
        doi=f"10.0000/{result_id}",
        url=f"https://example.test/{result_id}",
        publication_type="journal-article",
        subjects=("protein engineering",),
        retrieval=RetrievalAssessment(
            provider_score=10.0,
            retrieval_score=0.8,
            components={"provider_rank": 1.0},
        ),
        scope_assertion=assertion,
    )


def test_search_filters_scope_before_publication_use_and_deduplicates() -> None:
    policy = _policy()
    allowed_assertion = _assertion("artifact:allowed", SubjectScope.NONVIRAL_PROTEIN)
    provider = _FakeProvider(
        (
            _result(
                "allowed",
                "artifact:allowed",
                "Sequence context and enzyme stability",
            ),
            _result(
                "blocked",
                "artifact:blocked",
                "Virus protein structure",
            ),
            _result(
                "unknown",
                "artifact:unknown",
                "Unreviewed protein record",
            ),
        )
    )
    discovery = DeepSearchEngine(
        (provider,),
        policy=policy,
        scope_assertions={"artifact:allowed": allowed_assertion},
        clock=lambda: NOW,
    ).discover(_brief(policy))

    assert len(discovery.search_runs) == 2
    assert len(discovery.publications) == 1
    assert discovery.publications[0].publication_id == "doi:10.0000/allowed"
    assert len(discovery.allowed_search_hits) == 2
    assert {
        item.disposition for item in discovery.allowed_search_hits
    } == {"accepted", "duplicate_publication"}
    assert len(discovery.publications[0].acquisitions) == 2
    assert any(event.disposition == "denied" for event in discovery.policy_events)
    assert any(event.disposition == "duplicate" for event in discovery.policy_events)
    assert all(
        item.result.result_id != "blocked" for item in discovery.screened_results
    )
    assert any(
        event.disposition == "quarantined" for event in discovery.policy_events
    )
    serialized = discovery.model_dump(mode="json")
    quarantine_events = [
        event
        for event in serialized["policy_events"]
        if event["disposition"] == "quarantined"
    ]
    assert quarantine_events
    assert not {
        "title",
        "abstract",
        "authors",
        "url",
        "subjects",
    }.intersection(quarantine_events[0])
    assert all("-virus" in query for query in provider.calls)


def test_provider_adapters_preserve_metadata_and_retrieval_scores_without_network() -> None:
    crossref_client = _SyntheticClient(
        {
            "message": {
                "items": [
                    {
                        "DOI": "10.0000/SYNTHETIC",
                        "title": ["Sequence context and ordinary enzyme stability"],
                        "author": [{"given": "A", "family": "Researcher"}],
                        "published": {"date-parts": [[2025, 1, 1]]},
                        "container-title": ["Synthetic Journal"],
                        "URL": "https://example.test/crossref",
                        "type": "journal-article",
                        "abstract": "<jats:p>Synthetic abstract.</jats:p>",
                        "subject": ["protein engineering"],
                        "score": 12.0,
                    }
                ]
            }
        }
    )
    crossref = CrossrefSearchProvider(client=crossref_client)
    crossref_results = crossref.search(
        "ordinary enzyme stability",
        limit=3,
        filters={"until_date": "2025-12-31"},
    )

    assert crossref_results[0].artifact_id == "doi:10.0000/synthetic"
    assert crossref_results[0].abstract == "Synthetic abstract."
    assert 0.0 <= crossref_results[0].retrieval.retrieval_score <= 1.0
    assert crossref_client.calls[0][1]["filter"] == "until-pub-date:2025-12-31"
    ranked_again = crossref._convert(
        crossref_client.payload["message"]["items"][0],
        rank=9,
    )
    assert ranked_again is not None
    assert ranked_again.result_id == crossref_results[0].result_id

    openalex_client = _SyntheticClient(
        {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "title": "Synthetic enzyme context study",
                    "doi": "https://doi.org/10.0000/SYNTHETIC-TWO",
                    "authorships": [
                        {"author": {"display_name": "B Researcher"}}
                    ],
                    "publication_year": 2024,
                    "primary_location": {
                        "landing_page_url": "https://example.test/openalex",
                        "source": {"display_name": "Synthetic Journal"},
                    },
                    "type": "article",
                    "concepts": [{"display_name": "Protein engineering"}],
                    "abstract_inverted_index": {
                        "Synthetic": [0],
                        "abstract": [1],
                    },
                    "relevance_score": 0.9,
                }
            ]
        }
    )
    openalex_results = OpenAlexSearchProvider(
        client=openalex_client,
        api_key="unit-test-openalex-key",
    ).search(
        "ordinary enzyme context",
        limit=2,
        filters={},
    )

    assert openalex_results[0].artifact_id == "doi:10.0000/synthetic-two"
    assert openalex_results[0].abstract == "Synthetic abstract"
    assert openalex_client.calls[0][1]["api_key"] == "unit-test-openalex-key"


def test_openalex_missing_key_fails_before_network_request() -> None:
    client = _SyntheticClient({"results": []})
    provider = OpenAlexSearchProvider(client=client)

    with pytest.raises(ProviderCredentialMissingError, match="OPENALEX_API_KEY"):
        provider.search("ordinary enzyme context", limit=2, filters={})

    assert client.calls == []


def test_openalex_missing_key_records_zero_network_attempts() -> None:
    policy = _policy()
    discovery = DeepSearchEngine(
        (OpenAlexSearchProvider(client=_SyntheticClient({"results": []})),),
        policy=policy,
        clock=lambda: NOW,
    ).discover(_brief(policy))

    assert discovery.search_runs
    assert all(
        run.error_code == "ProviderCredentialMissingError"
        and run.stop_reason == "provider_error"
        and run.attempt_count == 0
        for run in discovery.search_runs
    )


def test_crossref_retries_transient_transport_failure() -> None:
    client = _FlakySyntheticClient({"message": {"items": []}})
    provider = CrossrefSearchProvider(
        client=client,
        max_retries=1,
        retry_backoff_seconds=0.0,
    )

    assert provider.search("ordinary enzyme context", limit=2, filters={}) == ()
    assert len(client.calls) == 2
    assert provider.last_attempt_count == 2


def test_discovery_enforces_unique_publication_budget() -> None:
    policy = _policy()
    first = _result(
        "first",
        "artifact:first",
        "First ordinary enzyme study",
    )
    second = _result(
        "second",
        "artifact:second",
        "Second ordinary enzyme study",
    )
    brief = _brief(policy).model_copy(
        update={
            "budget": SearchBudget(
                max_queries=2,
                max_results_per_query=10,
                max_publications=1,
            )
        }
    )
    discovery = DeepSearchEngine(
        (_FakeProvider((first, second)),),
        policy=policy,
        scope_assertions={
            "artifact:first": _assertion(
                "artifact:first", SubjectScope.NONVIRAL_PROTEIN
            ),
            "artifact:second": _assertion(
                "artifact:second", SubjectScope.NONVIRAL_PROTEIN
            ),
        },
        clock=lambda: NOW,
    ).discover(brief)

    assert len(discovery.publications) == 1
    assert any(
        event.disposition == "budget_excluded"
        for event in discovery.policy_events
    )
    assert any(run.stop_reason == "budget_exhausted" for run in discovery.search_runs)


def test_provider_failure_is_audited_without_stopping_other_providers() -> None:
    policy = _policy()
    result = _result(
        "allowed",
        "artifact:allowed",
        "Synthetic ordinary enzyme evidence",
    )

    discovery = DeepSearchEngine(
        (_FailingProvider(), _FakeProvider((result,))),
        policy=policy,
        scope_assertions={
            "artifact:allowed": _assertion(
                "artifact:allowed",
                SubjectScope.NONVIRAL_PROTEIN,
            )
        },
        clock=lambda: NOW,
    ).discover(_brief(policy))

    failed_runs = [
        run for run in discovery.search_runs if run.stop_reason == "provider_error"
    ]
    assert len(failed_runs) == 2
    assert all(run.error_code == "ConnectionError" for run in failed_runs)
    assert len(discovery.publications) == 1


def test_discovery_excludes_benchmark_quarantine_before_source_use() -> None:
    policy = _policy()
    result = _result(
        "quarantined",
        "artifact:quarantined",
        "Synthetic ordinary enzyme study",
    )
    brief = _brief(policy).model_copy(
        update={"source_quarantine_ids": ("artifact:quarantined",)}
    )

    discovery = DeepSearchEngine(
        (_FakeProvider((result,)),),
        policy=policy,
        scope_assertions={
            "artifact:quarantined": _assertion(
                "artifact:quarantined",
                SubjectScope.NONVIRAL_PROTEIN,
            )
        },
        clock=lambda: NOW,
    ).discover(brief)

    assert discovery.publications == ()
    assert any(
        event.reason_code == "benchmark_or_source_quarantine"
        for event in discovery.policy_events
    )


def test_full_text_fetch_requires_scope_bound_permit() -> None:
    policy = _policy()
    assertion = _assertion("artifact:allowed", SubjectScope.NONVIRAL_PROTEIN)
    allowed = _result(
        "allowed",
        "artifact:allowed",
        "Sequence context and enzyme stability",
        assertion=assertion,
    )
    engine = DeepSearchEngine((_FakeProvider((allowed,)),), policy=policy)
    called = 0

    def fetcher(url: str) -> str:
        nonlocal called
        called += 1
        return f"Fetched {url}"

    with pytest.raises(PermissionError):
        engine.fetch_full_text(
            allowed.model_copy(update={"scope_assertion": None}),
            permit=policy.issue_permit(
                artifact_id=allowed.artifact_id,
                operation="full_text_fetch",
                assertion=assertion,
                resource_locator=allowed.url,
            ),
            fetcher=fetcher,
        )
    assert called == 0

    permit = policy.issue_permit(
        artifact_id=allowed.artifact_id,
        operation="full_text_fetch",
        assertion=assertion,
        resource_locator=allowed.url,
    )
    assert engine.fetch_full_text(allowed, permit=permit, fetcher=fetcher)
    assert called == 1

    with pytest.raises(PermissionError):
        engine.fetch_full_text(
            allowed.model_copy(update={"url": "https://example.test/other"}),
            permit=permit,
            fetcher=fetcher,
        )
    assert called == 1


def test_scope_assertion_requires_explicit_absence_of_excluded_subject() -> None:
    payload = _assertion(
        "artifact:allowed",
        SubjectScope.NONVIRAL_PROTEIN,
    ).model_dump(mode="python")
    payload.pop("excluded_subject_present")

    with pytest.raises(ValidationError):
        ScopeAssertion.model_validate(payload)


def _valid_bundle(policy: ExternalEvidenceScopePolicy) -> EvidenceProductBundle:
    brief = _brief(policy)
    planned = DeepSearchPlanner(policy).plan(brief)
    primary_query = next(
        item for item in planned if item.route == SearchRoute.PRIMARY
    )
    counter_query = next(
        item for item in planned if item.route == SearchRoute.COUNTEREVIDENCE
    )
    primary_receipt = policy.issue_receipt(
        policy.inspect_query(primary_query.query),
        stage="search_run",
        subject_id=primary_query.query_id,
        subject_sha256=text_sha256(primary_query.query),
        issued_at=NOW,
    )
    counter_receipt = policy.issue_receipt(
        policy.inspect_query(counter_query.query),
        stage="search_run",
        subject_id=counter_query.query_id,
        subject_sha256=text_sha256(counter_query.query),
        issued_at=NOW,
    )
    primary_run = SearchRun(
        search_run_id="search:primary",
        brief_id=brief.brief_id,
        planned_query_id=primary_query.query_id,
        question_leaf_id="leaf:stability",
        provider="synthetic",
        route=SearchRoute.PRIMARY,
        exact_query=primary_query.query,
        exact_query_sha256=text_sha256(primary_query.query),
        filters=primary_query.filters,
        executed_at=NOW,
        result_ids=("result:one",),
        accepted_result_ids=("result:one",),
        excluded_result_count=0,
        duplicate_result_count=0,
        stop_reason="completed",
        snapshot_hash=_search_snapshot(
            search_run_id="search:primary",
            brief_id=brief.brief_id,
            provider="synthetic",
            planned_query_id=primary_query.query_id,
            question_leaf_id="leaf:stability",
            route=SearchRoute.PRIMARY,
            exact_query=primary_query.query,
            result_ids=("result:one",),
            accepted_result_ids=("result:one",),
            filters=primary_query.filters,
        ),
        policy_receipt=primary_receipt,
    )
    counter_run = SearchRun(
        search_run_id="search:counter",
        brief_id=brief.brief_id,
        planned_query_id=counter_query.query_id,
        question_leaf_id="leaf:stability",
        provider="synthetic",
        route=SearchRoute.COUNTEREVIDENCE,
        exact_query=counter_query.query,
        exact_query_sha256=text_sha256(counter_query.query),
        filters=counter_query.filters,
        executed_at=NOW,
        result_ids=(),
        accepted_result_ids=(),
        excluded_result_count=0,
        duplicate_result_count=0,
        stop_reason="no_results",
        snapshot_hash=_search_snapshot(
            search_run_id="search:counter",
            brief_id=brief.brief_id,
            provider="synthetic",
            planned_query_id=counter_query.query_id,
            question_leaf_id="leaf:stability",
            route=SearchRoute.COUNTEREVIDENCE,
            exact_query=counter_query.query,
            result_ids=(),
            accepted_result_ids=(),
            filters=counter_query.filters,
            stop_reason="no_results",
        ),
        policy_receipt=counter_receipt,
    )
    assertion = _assertion(
        "artifact:synthetic",
        SubjectScope.NONVIRAL_PROTEIN,
    )
    search_hit_id = stable_id(
        "search-hit",
        primary_run.search_run_id,
        "result:one",
        assertion.scope_assertion_id,
    )
    hit_title = "Synthetic nonviral enzyme study"
    hit_abstract = "Synthetic metadata supporting a nonviral enzyme claim."
    hit_subjects = ("protein engineering",)
    hit_scope_hash = content_sha256(
        {
            "artifact_id": assertion.artifact_id,
            "title": hit_title,
            "abstract": hit_abstract,
            "subjects": hit_subjects,
            "scope_assertion_sha256": assertion.assertion_hash,
        }
    )
    hit_decision = policy.decide_metadata(
        artifact_id=assertion.artifact_id,
        title=hit_title,
        abstract=hit_abstract,
        subjects=hit_subjects,
        assertion=assertion,
    )
    search_hit = AllowedSearchHit(
        search_hit_id=search_hit_id,
        search_run_id=primary_run.search_run_id,
        result_id="result:one",
        disposition="accepted",
        artifact_id=assertion.artifact_id,
        provider="synthetic",
        provider_record_id="synthetic:one",
        title=hit_title,
        abstract=hit_abstract,
        authors=("Researcher A",),
        year=2025,
        venue="Synthetic Journal",
        doi="10.0000/synthetic",
        url="https://example.test/synthetic",
        publication_type="journal-article",
        subjects=hit_subjects,
        retrieval_score=0.5,
        retrieval_components={"provider_rank": 1.0},
        scope_assertion_id=assertion.scope_assertion_id,
        policy_receipt=policy.issue_receipt(
            hit_decision,
            stage="search_hit",
            subject_id=search_hit_id,
            subject_sha256=hit_scope_hash,
            issued_at=NOW,
        ),
    )
    publication = Publication(
        publication_id="doi:10.0000/synthetic",
        identifier_aliases=("artifact:synthetic",),
        title="Synthetic nonviral enzyme study",
        authors=("Researcher A",),
        year=2025,
        venue="Synthetic Journal",
        doi="10.0000/synthetic",
        canonical_url="https://example.test/synthetic",
        publication_type="journal-article",
        study_family_id="study:synthetic",
        metadata_verified=True,
        full_text_status="verified",
        canonical_search_hit_id=search_hit.search_hit_id,
        acquisitions=(
            PublicationAcquisition(
                search_hit_id=search_hit.search_hit_id,
            ),
        ),
        source_scope="nonviral_protein",
        scope_assertion_id=assertion.scope_assertion_id,
        review_receipt_ids=("review:publication:identity", "review:publication:scope"),
    )
    span_text = "Sequence context changed the measured stability outcome."
    span = SourceSpan(
        source_span_id="span:synthetic",
        publication_id=publication.publication_id,
        artifact_id="artifact:synthetic",
        artifact_sha256="c" * 64,
        locator="results:paragraph-1",
        normalized_span_sha256=text_sha256(span_text),
        span_text=span_text,
        support_paraphrase=span_text,
        evidence_role="result",
        extraction_method="manual",
        resolved_against_artifact=True,
        independently_checked=True,
        scope_assertion_id=assertion.scope_assertion_id,
        review_receipt_ids=("review:span:resolution",),
    )
    group = EvidenceGroup(
        evidence_group_id="eg:synthetic",
        source_span_ids=(span.source_span_id,),
        stance="supports",
        completeness="complete",
        independence_group="study:synthetic",
        grouping_rationale="One complete synthetic result span.",
        verified_by=("reviewer:independent",),
        review_receipt_ids=("review:group:independence",),
    )
    claim = AtomicClaim(
        claim_id="claim:synthetic",
        statement="Sequence context can change a measured protein-stability outcome.",
        subject="sequence context",
        predicate="can_change",
        object="measured protein-stability outcome",
        claim_kind="scientific_prior",
        knowledge_type="sequence_context",
        applicability={"scope": "nonviral proteins"},
        evidence_group_ids=(group.evidence_group_id,),
        claim_status="supported",
    )
    logic = LogicUnit(
        logic_unit_id="logic:synthetic",
        question_leaf_id="leaf:stability",
        task_route="mechanism_explanation",
        subquestion="Does sequence context matter?",
        premise_claim_ids=(claim.claim_id,),
        search_coverage_run_ids=(
            primary_run.search_run_id,
            counter_run.search_run_id,
        ),
        operator="qualify",
        conclusion=claim.statement,
        applicability_tests=("Check the study context.",),
        falsifiers=("A matched context with no effect.",),
        abstain_if=("Context is not comparable.",),
        retrieval_text=claim.statement,
        scientific_quality=ScientificQuality(
            identity_verified=True,
            span_verified=True,
            entailment_status="verified",
            source_credibility=0.8,
            independent_support_count=1,
            counterevidence_status="searched_none",
            conflict_status="none",
            uncertainty=0.2,
        ),
        task_applicability=TaskApplicability(
            directness="direct",
            context_match=0.8,
            candidate_discriminative_value=0.4,
            matched_dimensions=("objective",),
            boundary_conditions=("Comparable measurement context.",),
        ),
        review_receipt_ids=("review:logic:entailment", "review:logic:applicability"),
    )
    card = KnowledgeDecisionCard(
        decision_card_id="card:synthetic",
        question_leaf_id="leaf:stability",
        task_route="mechanism_explanation",
        logic_unit_ids=(logic.logic_unit_id,),
        required_inputs=("measurement context",),
        boundary_conditions=("Comparable measurement context.",),
        uncertainty=0.2,
        permission=DecisionPermission.EXPLANATION_ONLY,
        abstain_if=("Context mismatch.",),
    )
    base_bundle = EvidenceProductBundle(
        research_brief=brief,
        search_runs=(primary_run, counter_run),
        scope_assertions=(assertion,),
        allowed_search_hits=(search_hit,),
        publications=(publication,),
        source_spans=(span,),
        evidence_groups=(group,),
        atomic_claims=(claim,),
        logic_units=(logic,),
        decision_cards=(card,),
        review_receipts=(),
    )
    review_receipts = (
        _review(
            assertion.review_receipt_ids[0],
            "scope_assertion",
            assertion,
            base_bundle,
        ),
        _review(
            "review:publication:identity",
            "metadata_identity",
            publication,
            base_bundle,
        ),
        _review(
            "review:publication:scope",
            "full_text_scope",
            publication,
            base_bundle,
        ),
        _review(
            "review:span:resolution",
            "source_span_resolution",
            span,
            base_bundle,
        ),
        _review(
            "review:group:independence",
            "independence_grouping",
            group,
            base_bundle,
        ),
        _review(
            "review:logic:entailment",
            "claim_entailment",
            logic,
            base_bundle,
        ),
        _review(
            "review:logic:applicability",
            "task_applicability",
            logic,
            base_bundle,
        ),
    )
    return base_bundle.model_copy(update={"review_receipts": review_receipts})


def test_release_manifest_closes_full_evidence_graph() -> None:
    policy = _policy()
    bundle = _approved_bundle(policy)
    manifest = build_release_manifest(
        bundle,
        release_version="1.0.0",
        status="released",
        created_at=NOW,
    )
    released = bundle.model_copy(update={"release_manifest": manifest})

    report = _validate(released, policy)

    assert report.valid is True
    assert report.release_ready is True
    assert report.counts["errors"] == 0


def test_cli_records_a_bundle_bound_human_release_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _policy()
    source = tmp_path / "bundle.json"
    output = tmp_path / "approved.json"
    artifact = tmp_path / "human-review.txt"
    source.write_text(
        _valid_bundle(policy).model_dump_json(indent=2),
        encoding="utf-8",
    )
    artifact.write_text("Synthetic human review record.", encoding="utf-8")
    monkeypatch.setenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_ID", POLICY_KEY_ID)
    monkeypatch.setenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX", POLICY_KEY.hex())
    monkeypatch.setenv(
        "FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON",
        json.dumps(
            {
                reviewer_id: {
                    "key_id": REVIEW_KEY_ID,
                    "key_hex": REVIEW_KEY.hex(),
                }
                for reviewer_id in _reviewer_keys(_valid_bundle(policy))
            }
        ),
    )
    monkeypatch.setenv(
        "FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON",
        json.dumps(
            {
                "reviewer:cli": {
                    "key_id": RELEASE_KEY_ID,
                    "key_hex": RELEASE_KEY.hex(),
                }
            }
        ),
    )

    exit_code = deep_research_cli(
        [
            "approve",
            str(source),
            "--output",
            str(output),
            "--approval-id",
            "approval:cli",
            "--reviewer-id",
            "reviewer:cli",
            "--method-version",
            "human-release-review:v1",
            "--approval-artifact",
            str(artifact),
            "--approval-artifact-id",
            "human-review:cli",
            "--release-version",
            "1.0.0",
        ]
    )
    approved = EvidenceProductBundle.model_validate_json(
        output.read_text(encoding="utf-8")
    )

    assert exit_code == 0
    assert approved.release_approvals[0].approval_id == "approval:cli"
    assert approved.release_approvals[0].input_sha256 == (
        release_approval_input_sha256(
            approved,
            release_version="1.0.0",
            status="released",
            created_at=approved.release_approvals[0].target_created_at,
        )
    )


def test_cli_rejects_keyring_records_with_unknown_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON",
        json.dumps(
            {
                "reviewer:synthetic": {
                    "key_id": REVIEW_KEY_ID,
                    "key_hex": REVIEW_KEY.hex(),
                    "unexpected": "must-fail-closed",
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="exactly key_id and key_hex"):
        deep_research_cli(["policy"])


def test_selection_permission_requires_calibration_and_clear_overlap() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    card = bundle.decision_cards[0].model_copy(
        update={
            "permission": DecisionPermission.CANDIDATE_RERANKING,
            "task_route": "candidate_ranking",
        }
    )
    invalid = bundle.model_copy(update={"decision_cards": (card,)})
    report = _validate(invalid, policy)

    assert report.valid is False
    assert {issue.code for issue in report.issues} >= {
        "card.calibration_required",
        "card.benchmark_overlap",
    }


def test_manifest_detects_semantic_tampering() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    manifest = build_release_manifest(
        bundle,
        release_version="1.0.0",
        created_at=NOW,
    )
    changed_claim = bundle.atomic_claims[0].model_copy(
        update={"statement": "A semantically different synthetic statement."}
    )
    tampered = bundle.model_copy(
        update={
            "atomic_claims": (changed_claim,),
            "release_manifest": manifest,
        }
    )
    report = _validate(tampered, policy)

    assert report.valid is False
    assert "release.record_hash_mismatch" in {issue.code for issue in report.issues}


def test_review_receipt_is_invalidated_when_dependency_changes() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    changed_claim = bundle.atomic_claims[0].model_copy(
        update={"statement": "A changed statement with the same record identity."}
    )
    tampered = bundle.model_copy(update={"atomic_claims": (changed_claim,)})

    report = _validate(tampered, policy)

    assert report.valid is False
    assert "review.input_hash_mismatch" in {issue.code for issue in report.issues}


def test_release_approval_is_invalidated_when_evidence_product_changes() -> None:
    policy = _policy()
    bundle = _approved_bundle(policy)
    changed_claim = bundle.atomic_claims[0].model_copy(
        update={"statement": "Evidence changed after the human release approval."}
    )
    tampered = bundle.model_copy(update={"atomic_claims": (changed_claim,)})

    report = _validate(tampered, policy)

    assert report.valid is False
    assert "release_approval.input_hash_mismatch" in {
        issue.code for issue in report.issues
    }


def test_policy_receipt_requires_the_trusted_policy_signature() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    forged_receipt = bundle.search_runs[0].policy_receipt.model_copy(
        update={"decision": "quarantined"}
    )
    forged_run = bundle.search_runs[0].model_copy(
        update={"policy_receipt": forged_receipt}
    )
    report = _validate(
        bundle.model_copy(
            update={"search_runs": (forged_run, bundle.search_runs[1])}
        ),
        policy,
    )

    assert report.valid is False
    assert "search.policy_signature_invalid" in {
        issue.code for issue in report.issues
    }


def test_release_approval_requires_a_trusted_reviewer_signature() -> None:
    policy = _policy()
    released = _released_bundle(policy)
    report = validate_evidence_product(
        released,
        active_policy=policy,
        trusted_reviewer_keys=_reviewer_keys(released),
        trusted_release_approval_keys={
            "release-reviewer:one": (
                RELEASE_KEY_ID,
                b"incorrect-release-trust-key-0001",
            )
        },
    )

    assert report.valid is False
    assert "release_approval.signature_invalid" in {
        issue.code for issue in report.issues
    }


def test_release_approval_cannot_replay_across_release_version() -> None:
    policy = _policy()
    approved = _approved_bundle(policy)

    with pytest.raises(ValueError, match="approval target"):
        build_release_manifest(
            approved,
            release_version="9.0.0",
            status="released",
            created_at=NOW,
        )


def test_search_run_route_is_bound_to_the_reproduced_plan() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    primary = bundle.search_runs[0]
    changed = primary.model_copy(
        update={
            "route": SearchRoute.BOUNDARY,
            "snapshot_hash": _search_snapshot(
                search_run_id=primary.search_run_id,
                brief_id=primary.brief_id,
                provider=primary.provider,
                planned_query_id=primary.planned_query_id,
                question_leaf_id=primary.question_leaf_id,
                route=SearchRoute.BOUNDARY,
                exact_query=primary.exact_query,
                result_ids=primary.result_ids,
                accepted_result_ids=primary.accepted_result_ids,
                filters=primary.filters,
            ),
        }
    )
    report = _validate(
        bundle.model_copy(update={"search_runs": (changed, bundle.search_runs[1])}),
        policy,
    )

    assert report.valid is False
    assert "search.plan_binding_mismatch" in {
        issue.code for issue in report.issues
    }


def test_counter_search_change_invalidates_logic_review_closure() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    counter = bundle.search_runs[1]
    changed_time = NOW + timedelta(minutes=1)
    changed_counter = counter.model_copy(
        update={
            "executed_at": changed_time,
            "snapshot_hash": _search_snapshot(
                search_run_id=counter.search_run_id,
                brief_id=counter.brief_id,
                provider=counter.provider,
                planned_query_id=counter.planned_query_id,
                question_leaf_id=counter.question_leaf_id,
                route=counter.route,
                exact_query=counter.exact_query,
                result_ids=counter.result_ids,
                accepted_result_ids=counter.accepted_result_ids,
                filters=counter.filters,
                executed_at=changed_time,
                stop_reason=counter.stop_reason,
            ),
        }
    )
    report = _validate(
        bundle.model_copy(
            update={"search_runs": (bundle.search_runs[0], changed_counter)}
        ),
        policy,
    )

    assert report.valid is False
    assert "review.input_hash_mismatch" in {
        issue.code for issue in report.issues
    }


def test_model_assisted_reviews_cannot_approve_without_human_cosign() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    logic = bundle.logic_units[0]
    replacements = []
    for review_type, receipt_id in (
        ("claim_entailment", "review:logic:entailment"),
        ("task_applicability", "review:logic:applicability"),
    ):
        replacements.append(
            issue_review_receipt(
                bundle,
                review_receipt_id=receipt_id,
                review_type=review_type,
                record_id=logic.logic_unit_id,
                reviewer_id=f"model:{review_type}",
                reviewer_kind="model_assisted",
                method_version="synthetic-review:v1",
                decision="passed",
                reviewed_at=NOW,
                expires_at=NOW + timedelta(days=3650),
                signing_key_id=REVIEW_KEY_ID,
                signing_key=REVIEW_KEY,
                model_fingerprint="synthetic-model:v1",
                prompt_sha256="a" * 64,
            )
        )
    retained = tuple(
        item
        for item in bundle.review_receipts
        if item.review_receipt_id
        not in {"review:logic:entailment", "review:logic:applicability"}
    )
    invalid = bundle.model_copy(
        update={"review_receipts": (*retained, *replacements)}
    )
    report = _validate(invalid, policy)

    assert report.valid is False
    assert "review.human_cosign_missing" in {
        issue.code for issue in report.issues
    }


def test_supported_orphan_claim_is_rejected() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    orphan = bundle.atomic_claims[0].model_copy(
        update={
            "claim_id": "claim:orphan",
            "statement": "An unreviewed inference with reused evidence.",
        }
    )
    report = _validate(
        bundle.model_copy(update={"atomic_claims": (*bundle.atomic_claims, orphan)}),
        policy,
    )

    assert report.valid is False
    assert "claim.logic_entailment_missing" in {
        issue.code for issue in report.issues
    }


def test_allowed_search_hit_metadata_is_a_signed_replayable_preimage() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    tampered_hit = bundle.allowed_search_hits[0].model_copy(
        update={"retrieval_score": 0.99, "title": "Tampered metadata"}
    )
    report = _validate(
        bundle.model_copy(update={"allowed_search_hits": (tampered_hit,)}),
        policy,
    )

    assert report.valid is False
    assert {
        "search_hit.policy_subject_mismatch",
        "publication.canonical_metadata_mismatch",
    }.intersection(issue.code for issue in report.issues)


def test_search_run_rejects_foreign_policy_receipt() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    foreign_receipt = bundle.search_runs[0].policy_receipt.model_copy(
        update={"policy_hash": "e" * 64}
    )
    changed_run = bundle.search_runs[0].model_copy(
        update={"policy_receipt": foreign_receipt}
    )
    tampered = bundle.model_copy(
        update={"search_runs": (changed_run, bundle.search_runs[1])}
    )

    report = _validate(tampered, policy)

    assert report.valid is False
    assert "search.policy_hash_mismatch" in {issue.code for issue in report.issues}


def test_search_policy_receipt_binds_the_exact_query() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    replayed_receipt = bundle.search_runs[0].policy_receipt.model_copy(
        update={"subject_sha256": text_sha256("another query")}
    )
    changed_run = bundle.search_runs[0].model_copy(
        update={"policy_receipt": replayed_receipt}
    )
    invalid = bundle.model_copy(
        update={"search_runs": (changed_run, bundle.search_runs[1])}
    )

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "search.policy_subject_mismatch" in {
        issue.code for issue in report.issues
    }


def test_publication_must_link_an_accepted_search_result() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    acquisition = bundle.publications[0].acquisitions[0].model_copy(
        update={"search_hit_id": "search-hit:not-retained"}
    )
    changed_publication = bundle.publications[0].model_copy(
        update={"acquisitions": (acquisition,)}
    )
    invalid = bundle.model_copy(update={"publications": (changed_publication,)})

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "publication.search_hit_dangling" in {
        issue.code for issue in report.issues
    }


def test_release_rejects_publication_on_source_quarantine() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    quarantined_brief = bundle.research_brief.model_copy(
        update={"source_quarantine_ids": (bundle.publications[0].publication_id,)}
    )
    invalid = bundle.model_copy(update={"research_brief": quarantined_brief})

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "publication.source_quarantined" in {
        issue.code for issue in report.issues
    }


def test_release_rejects_search_hit_identity_on_source_quarantine() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    quarantined_brief = bundle.research_brief.model_copy(
        update={
            "source_quarantine_ids": (
                bundle.allowed_search_hits[0].provider_record_id,
            )
        }
    )
    invalid = bundle.model_copy(update={"research_brief": quarantined_brief})

    report = _validate(invalid, policy)

    assert "search_hit.source_quarantined" in {
        issue.code for issue in report.issues
    }


def test_release_rejects_expired_scope_assertion_at_reference_time() -> None:
    policy = _policy()
    bundle = _released_bundle(policy)
    expired_assertion = bundle.scope_assertions[0].model_copy(
        update={
            "reviewed_at": NOW - timedelta(days=2),
            "expires_at": NOW - timedelta(days=1),
        }
    )
    invalid = bundle.model_copy(
        update={
            "scope_assertions": (
                expired_assertion,
                *bundle.scope_assertions[1:],
            )
        }
    )

    report = _validate(invalid, policy)

    assert "scope_assertion.validity_window_mismatch" in {
        issue.code for issue in report.issues
    }


def test_every_question_leaf_requires_every_declared_search_route() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    expanded_brief = bundle.research_brief.model_copy(
        update={
            "required_search_routes": (
                SearchRoute.PRIMARY,
                SearchRoute.COUNTEREVIDENCE,
                SearchRoute.BOUNDARY,
            ),
            "budget": SearchBudget(
                max_queries=3,
                max_results_per_query=10,
                max_publications=20,
            ),
        }
    )
    incomplete = bundle.model_copy(update={"research_brief": expanded_brief})

    report = _validate(incomplete, policy)

    assert report.valid is False
    assert "search.required_route_missing" in {issue.code for issue in report.issues}


def test_release_identity_and_validator_version_are_recomputed() -> None:
    policy = _policy()
    released = _released_bundle(policy)
    assert released.release_manifest is not None
    tampered_manifest = released.release_manifest.model_copy(
        update={
            "release_id": "release:forged",
            "validator_version": "unknown-validator:v999",
        }
    )
    tampered = released.model_copy(update={"release_manifest": tampered_manifest})

    report = _validate(tampered, policy)
    codes = {issue.code for issue in report.issues}

    assert report.valid is False
    assert {"release.id_mismatch", "release.validator_version_mismatch"} <= codes


def test_decision_card_requires_nonempty_route_matched_logic() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    empty_card = bundle.decision_cards[0].model_copy(update={"logic_unit_ids": ()})
    invalid = bundle.model_copy(update={"decision_cards": (empty_card,)})

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "card.logic_missing" in {issue.code for issue in report.issues}


def test_insufficient_claim_can_only_drive_an_explicit_abstention() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    insufficient = bundle.atomic_claims[0].model_copy(
        update={"claim_status": "insufficient"}
    )
    invalid = bundle.model_copy(update={"atomic_claims": (insufficient,)})

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "logic.insufficient_premise_not_abstained" in {
        issue.code for issue in report.issues
    }


def test_evidence_group_identity_must_match_publication_study_family() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    changed_group = bundle.evidence_groups[0].model_copy(
        update={"independence_group": "study:unrelated"}
    )
    invalid = bundle.model_copy(update={"evidence_groups": (changed_group,)})

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "group.independence_family_mismatch" in {
        issue.code for issue in report.issues
    }


def test_selection_permissions_fail_closed_until_calibration_is_a_release_record() -> None:
    policy = _policy()
    bundle = _valid_bundle(policy)
    candidate_logic = bundle.logic_units[0].model_copy(
        update={"task_route": "candidate_ranking"}
    )
    candidate_card = bundle.decision_cards[0].model_copy(
        update={
            "task_route": "candidate_ranking",
            "logic_unit_ids": (candidate_logic.logic_unit_id,),
            "permission": DecisionPermission.CANDIDATE_RERANKING,
            "candidate_feature": "synthetic_feature",
            "calibration_id": "calibration:self-reported",
            "calibration_status": "validated",
            "benchmark_overlap_status": "clear",
        }
    )
    invalid = bundle.model_copy(
        update={
            "logic_units": (candidate_logic,),
            "decision_cards": (candidate_card,),
        }
    )

    report = _validate(invalid, policy)

    assert report.valid is False
    assert "card.selection_permission_not_releasable" in {
        issue.code for issue in report.issues
    }


def test_strict_schema_rejects_string_boolean() -> None:
    policy = _policy()
    payload = _brief(policy).model_dump(mode="json")
    payload["non_viral_only"] = "true"

    with pytest.raises(ValidationError):
        ResearchBrief.model_validate(payload)


def test_v1_evidence_product_requires_explicit_rebuild() -> None:
    payload = _valid_bundle(_policy()).model_dump(mode="json")
    payload["schema_version"] = "scientific-evidence-product:v1"

    with pytest.raises(ValidationError):
        EvidenceProductBundle.model_validate(payload)


def test_cli_validate_writes_structured_schema_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_ID", raising=False)
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX", raising=False)
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON", raising=False)
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON", raising=False)
    source = tmp_path / "invalid.json"
    output = tmp_path / "validation.json"
    source.write_text("{}", encoding="utf-8")

    exit_code = deep_research_cli(
        ["validate", str(source), "--output", str(output)]
    )
    report = json.loads(output.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert report["valid"] is False
    assert report["issues"][0]["code"] == "bundle.schema_invalid"


def _approved_bundle(policy: ExternalEvidenceScopePolicy) -> EvidenceProductBundle:
    bundle = _valid_bundle(policy)
    approval = issue_release_approval(
        bundle,
        approval_id="release-review:one",
        reviewer_id="release-reviewer:one",
        method_version="synthetic-release-review:v1",
        approval_artifact_id="human-review:synthetic",
        approval_artifact_sha256="f" * 64,
        approved_at=NOW,
        release_version="1.0.0",
        created_at=NOW,
        parent_release_id=None,
        signing_key_id=RELEASE_KEY_ID,
        signing_key=RELEASE_KEY,
    )
    return bundle.model_copy(update={"release_approvals": (approval,)})


def _released_bundle(policy: ExternalEvidenceScopePolicy) -> EvidenceProductBundle:
    bundle = _approved_bundle(policy)
    manifest = build_release_manifest(
        bundle,
        release_version="1.0.0",
        status="released",
        created_at=NOW,
    )
    return bundle.model_copy(update={"release_manifest": manifest})


def test_released_product_exports_and_validates_exact_legacy_view(tmp_path: Path) -> None:
    policy = _policy()
    output_root = tmp_path / "runtime"

    runtime_manifest = _export(
        _released_bundle(policy),
        output_root,
    )
    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is True
    assert runtime_manifest["schema_version"] == "local-rag-runtime-files:v1"
    evidence_payload = json.loads(
        (output_root / "evidence-release.json").read_text(encoding="utf-8")
    )
    assert evidence_payload["schema_version"] == "scientific-evidence-product:v2"
    assert evidence_payload["review_receipts"]
    assert evidence_payload["release_approvals"]
    assert evidence_payload["release_manifest"]["status"] == "released"
    claim_entry = next(
        item
        for item in runtime_manifest["files"]
        if item["record_type"] == "atomic_claim"
    )
    discovered = discover_local_files(
        (
            LocalKnowledgeRootConfig(
                path=output_root,
                root_id="SYNTHETIC",
                access_policy_mode="synthetic_test",
                include=("**/*.md",),
            ),
        ),
        follow_symlinks=False,
    )
    assert discovered[0].relative_path == claim_entry["relative_path"]
    parsed = AutoLocalParser().parse(discovered[0])
    assert parsed.metadata["confidence"] == pytest.approx(0.64)
    assert parsed.metadata["selection_eligible"] is False


def test_released_product_exports_three_native_record_types(tmp_path: Path) -> None:
    policy = _policy()
    bundle = _released_bundle(policy)
    output_root = tmp_path / "native-runtime"

    runtime_manifest = export_native_local_rag_bundle(
        bundle,
        output_root,
        active_policy=policy,
        trusted_reviewer_keys=_reviewer_keys(bundle),
        trusted_release_approval_keys=_release_keys(bundle),
    )

    assert runtime_manifest["schema_version"] == "local-rag-runtime-files:v2"
    assert runtime_manifest["projection"] == "native_evidence_records_v1"
    assert {
        item["record_type"]
        for item in runtime_manifest["files"]
        if item["record_type"] != "evidence_release"
    } == {"atomic_claim", "logic_unit", "knowledge_decision_card"}
    discovered = discover_local_files(
        (
            LocalKnowledgeRootConfig(
                path=output_root,
                root_id="SYNTHETIC_NATIVE",
                access_policy_mode="synthetic_test",
                include=("records/**/*.md",),
            ),
        ),
        follow_symlinks=False,
    )
    assert {
        AutoLocalParser().parse(item).metadata["record_type"] for item in discovered
    } == {"atomic_claim", "logic_unit", "knowledge_decision_card"}


def test_failed_export_never_publishes_a_partial_runtime_directory(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "runtime"

    with pytest.raises(ValueError, match="valid released evidence product"):
        _export(
            _valid_bundle(_policy()),
            output_root,
        )

    assert output_root.exists() is False


def test_manifest_aware_discovery_never_touches_denied_unlisted_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_file = tmp_path / "AvoidRead.txt"
    policy_file.write_text("runtime/blocked.md\n", encoding="utf-8")
    output_root = tmp_path / "runtime"
    _export(_released_bundle(_policy()), output_root)
    blocked = output_root / "blocked.md"
    blocked.write_text("synthetic blocked sentinel", encoding="utf-8")

    original_is_file = Path.is_file

    def guarded_is_file(path: Path) -> bool:
        if path.absolute() == blocked.absolute():
            raise AssertionError("denied unlisted file was probed")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    discovered = discover_local_files(
        (
            LocalKnowledgeRootConfig(
                path=output_root,
                root_id="SYNTHETIC",
                include=("**/*.md",),
            ),
        ),
        follow_symlinks=False,
    )

    assert len(discovered) == 1
    assert discovered[0].expected_sha256 is not None
    assert discovered[0].path != blocked


def test_legacy_validator_rejects_missing_or_tampered_runtime_manifest(
    tmp_path: Path,
) -> None:
    unmanifested = tmp_path / "unmanifested"
    unmanifested.mkdir()
    (unmanifested / "claim.md").write_text("synthetic", encoding="utf-8")
    assert validate_legacy_runtime_bundle(unmanifested)["valid"] is False

    output_root = tmp_path / "tampered"
    _export(_released_bundle(_policy()), output_root)
    manifest_path = output_root / "runtime-files.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source_release_id"] = "release:tampered"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    report = validate_legacy_runtime_bundle(output_root)
    assert report["valid"] is False
    assert "runtime-files.json manifest hash mismatch" in report["errors"]


def test_runtime_manifest_record_identity_is_checked_against_release(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "record-id-tamper"
    _export(
        _released_bundle(_policy()),
        output_root,
    )
    manifest_path = output_root / "runtime-files.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    claim_entry = next(
        item for item in payload["files"] if item["record_type"] == "atomic_claim"
    )
    claim_entry["record_id"] = "claim:forged"
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    payload["manifest_sha256"] = content_sha256(unsigned)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any(
        "matching canonical release record" in error
        for error in report["errors"]
    )


def test_runtime_manifest_rejects_recomputed_foreign_external_policy(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "foreign-policy"
    _export(
        _released_bundle(_policy()),
        output_root,
    )
    manifest_path = output_root / "runtime-files.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["policy_hash"] = "e" * 64
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    payload["manifest_sha256"] = content_sha256(unsigned)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any(
        "external evidence-policy hash mismatch" in error
        for error in report["errors"]
    )


def test_runtime_bundle_cannot_self_attest_after_release_signature_tamper(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "forged-release-signature"
    _export(_released_bundle(_policy()), output_root)
    evidence_path = output_root / "evidence-release.json"
    evidence_payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    signature = evidence_payload["release_approvals"][0]["attestation"][
        "signature"
    ]
    evidence_payload["release_approvals"][0]["attestation"]["signature"] = (
        ("0" if signature[0] != "0" else "1") + signature[1:]
    )
    evidence_path.write_text(
        json.dumps(evidence_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    manifest_path = output_root / "runtime-files.json"
    runtime_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    release_entry = next(
        item
        for item in runtime_payload["files"]
        if item["record_type"] == "evidence_release"
    )
    raw_evidence = evidence_path.read_bytes()
    release_entry["sha256"] = hashlib.sha256(raw_evidence).hexdigest()
    release_entry["bytes"] = len(raw_evidence)
    unsigned = dict(runtime_payload)
    unsigned.pop("manifest_sha256", None)
    runtime_payload["manifest_sha256"] = content_sha256(unsigned)
    manifest_path.write_text(
        json.dumps(runtime_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any(
        "release_approval.signature_invalid" in error
        for error in report["errors"]
    )


def test_runtime_projection_cannot_be_rewritten_with_recomputed_file_hashes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "rewritten-projection"
    runtime_payload = _export(_released_bundle(_policy()), output_root)
    claim_entry = next(
        item
        for item in runtime_payload["files"]
        if item["record_type"] == "atomic_claim"
    )
    claim_path = output_root / claim_entry["relative_path"]
    original_claim = claim_path.read_bytes()
    raw_claim = original_claim.replace(
        b"Sequence context can change",
        b"Rewritten context can change",
        1,
    )
    assert raw_claim != original_claim
    claim_path.write_bytes(raw_claim)

    claim_entry["sha256"] = hashlib.sha256(raw_claim).hexdigest()
    claim_entry["bytes"] = len(raw_claim)
    unsigned = dict(runtime_payload)
    unsigned.pop("manifest_sha256", None)
    runtime_payload["manifest_sha256"] = content_sha256(unsigned)
    (output_root / "runtime-files.json").write_text(
        json.dumps(runtime_payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any(
        "deterministic projection" in error for error in report["errors"]
    )


def test_runtime_release_validation_requires_explicit_trust_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "missing-trust"
    _export(_released_bundle(_policy()), output_root)
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_ID")
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX")
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON")
    monkeypatch.delenv("FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON")

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any("FITNESS_DEEP_RESEARCH_POLICY" in error for error in report["errors"])


def test_legacy_validator_reports_malformed_manifest_without_crashing(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "malformed-manifest"
    output_root.mkdir()
    (output_root / "runtime-files.json").write_text("{", encoding="utf-8")

    report = validate_legacy_runtime_bundle(output_root)

    assert report["valid"] is False
    assert any("runtime-files.json" in error for error in report["errors"])
