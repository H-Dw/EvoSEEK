from __future__ import annotations

import os
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import httpx
from pydantic import Field

from .canonical import stable_id
from .contracts import StrictModel
from .policy import ScopeAssertion


class ProviderCredentialMissingError(RuntimeError):
    """Provider is configured but cannot make an authenticated network call."""

    attempt_count = 0


def _get_with_retries(
    *,
    client: httpx.Client | None,
    endpoint: str,
    parameters: Mapping[str, str | int],
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
    on_attempt: Any,
) -> Any:
    for attempt in range(1, max_retries + 2):
        on_attempt(attempt)
        try:
            response = (
                client.get(endpoint, params=dict(parameters))
                if client is not None
                else httpx.get(
                    endpoint,
                    params=dict(parameters),
                    timeout=timeout_seconds,
                )
            )
            response.raise_for_status()
            return response
        except (httpx.HTTPError, TimeoutError, ConnectionError) as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            retryable = status is None or int(status) == 429 or int(status) >= 500
            if not retryable or attempt > max_retries:
                raise
            if retry_backoff_seconds:
                time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
    raise RuntimeError("Provider retry loop exited unexpectedly")


class RetrievalAssessment(StrictModel):
    provider_score: float | None = None
    retrieval_score: float = Field(ge=0.0, le=1.0)
    components: dict[str, float] = Field(default_factory=dict)


class ExternalSearchResult(StrictModel):
    result_id: str
    artifact_id: str
    provider: str
    provider_record_id: str
    title: str
    abstract: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = Field(default=None, ge=1600, le=2200)
    venue: str = ""
    doi: str | None = None
    url: str
    publication_type: str = "unknown"
    subjects: tuple[str, ...] = ()
    retrieval: RetrievalAssessment
    scope_assertion: ScopeAssertion | None = None


class SearchProvider(Protocol):
    name: str

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, str],
    ) -> tuple[ExternalSearchResult, ...]: ...


def _strip_markup(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(re.sub(r"<[^>]+>", " ", value).split())


def _normalized_doi(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.casefold().startswith("https://doi.org/"):
        normalized = normalized[16:]
    return normalized.casefold() or None


def _retrieval_score(rank: int, provider_score: float | None) -> RetrievalAssessment:
    rank_component = 1.0 / max(1, rank)
    provider_component = 0.0
    if provider_score is not None and provider_score > 0:
        provider_component = provider_score / (provider_score + 10.0)
    score = min(1.0, 0.75 * rank_component + 0.25 * provider_component)
    return RetrievalAssessment(
        provider_score=provider_score,
        retrieval_score=score,
        components={
            "provider_rank": rank_component,
            "provider_score_normalized": provider_component,
        },
    )


class CrossrefSearchProvider:
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.client = client
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.last_attempt_count = 0

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, str],
    ) -> tuple[ExternalSearchResult, ...]:
        parameters: dict[str, str | int] = {
            "query.bibliographic": query,
            "rows": limit,
            "select": (
                "DOI,title,author,published,container-title,URL,type,abstract,"
                "subject,score"
            ),
        }
        if self.mailto:
            parameters["mailto"] = self.mailto
        crossref_filters: list[str] = []
        if filters.get("from_date"):
            crossref_filters.append(f"from-pub-date:{filters['from_date']}")
        if filters.get("until_date"):
            crossref_filters.append(f"until-pub-date:{filters['until_date']}")
        if crossref_filters:
            parameters["filter"] = ",".join(crossref_filters)
        response = _get_with_retries(
            client=self.client,
            endpoint=self.endpoint,
            parameters=parameters,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            on_attempt=lambda attempt: setattr(self, "last_attempt_count", attempt),
        )
        items = response.json().get("message", {}).get("items", [])
        return tuple(
            result
            for rank, item in enumerate(items, start=1)
            if (result := self._convert(item, rank=rank)) is not None
        )

    def _convert(self, item: Mapping[str, Any], *, rank: int) -> ExternalSearchResult | None:
        titles = item.get("title") or []
        title = str(titles[0]).strip() if titles else ""
        if not title:
            return None
        doi = _normalized_doi(str(item.get("DOI", "")))
        provider_record_id = doi or stable_id("crossref-record", title)
        artifact_id = f"doi:{doi}" if doi else f"crossref:{provider_record_id}"
        authors = tuple(
            " ".join(
                part
                for part in (str(raw.get("given", "")).strip(), str(raw.get("family", "")).strip())
                if part
            )
            for raw in item.get("author", [])
            if isinstance(raw, Mapping)
        )
        date_parts = (item.get("published") or {}).get("date-parts") or []
        year = int(date_parts[0][0]) if date_parts and date_parts[0] else None
        venues = item.get("container-title") or []
        venue = str(venues[0]).strip() if venues else ""
        raw_score = item.get("score")
        provider_score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        url = str(item.get("URL") or (f"https://doi.org/{doi}" if doi else "")).strip()
        if not url:
            return None
        return ExternalSearchResult(
            # Provider-record identity must not change when result rank changes.
            result_id=stable_id(self.name, provider_record_id),
            artifact_id=artifact_id,
            provider=self.name,
            provider_record_id=provider_record_id,
            title=title,
            abstract=_strip_markup(str(item.get("abstract") or "")),
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            url=url,
            publication_type=str(item.get("type") or "unknown"),
            subjects=tuple(str(value) for value in item.get("subject", []) if value),
            retrieval=_retrieval_score(rank, provider_score),
        )


def _openalex_abstract(inverted: Mapping[str, Sequence[int]] | None) -> str | None:
    if not inverted:
        return None
    positioned = sorted(
        (int(position), str(token))
        for token, positions in inverted.items()
        for position in positions
    )
    return " ".join(token for _, token in positioned) or None


class OpenAlexSearchProvider:
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        mailto: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.client = client
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.mailto = mailto
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.last_attempt_count = 0

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: Mapping[str, str],
    ) -> tuple[ExternalSearchResult, ...]:
        if not self.api_key:
            self.last_attempt_count = 0
            raise ProviderCredentialMissingError(
                "Set OPENALEX_API_KEY before using the OpenAlex provider"
            )
        parameters: dict[str, str | int] = {
            "search": query,
            "per-page": limit,
            "api_key": self.api_key,
        }
        if self.mailto:
            parameters["mailto"] = self.mailto
        openalex_filters: list[str] = []
        if filters.get("from_date"):
            openalex_filters.append(f"from_publication_date:{filters['from_date']}")
        if filters.get("until_date"):
            openalex_filters.append(f"to_publication_date:{filters['until_date']}")
        if openalex_filters:
            parameters["filter"] = ",".join(openalex_filters)
        response = _get_with_retries(
            client=self.client,
            endpoint=self.endpoint,
            parameters=parameters,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
            on_attempt=lambda attempt: setattr(self, "last_attempt_count", attempt),
        )
        items = response.json().get("results", [])
        return tuple(
            result
            for rank, item in enumerate(items, start=1)
            if (result := self._convert(item, rank=rank)) is not None
        )

    def _convert(self, item: Mapping[str, Any], *, rank: int) -> ExternalSearchResult | None:
        title = str(item.get("title") or "").strip()
        provider_record_id = str(item.get("id") or "").rsplit("/", 1)[-1]
        if not title or not provider_record_id:
            return None
        doi = _normalized_doi(str(item.get("doi") or ""))
        artifact_id = f"doi:{doi}" if doi else f"openalex:{provider_record_id}"
        authors = tuple(
            str((authorship.get("author") or {}).get("display_name") or "").strip()
            for authorship in item.get("authorships", [])
            if isinstance(authorship, Mapping)
            and str((authorship.get("author") or {}).get("display_name") or "").strip()
        )
        location = item.get("primary_location") or {}
        source = location.get("source") or {}
        url = str(location.get("landing_page_url") or item.get("id") or "").strip()
        if not url:
            return None
        concepts = item.get("concepts") or []
        subjects = tuple(
            str(value.get("display_name"))
            for value in concepts
            if isinstance(value, Mapping) and value.get("display_name")
        )
        raw_score = item.get("relevance_score")
        provider_score = float(raw_score) if isinstance(raw_score, (int, float)) else None
        return ExternalSearchResult(
            # Provider-record identity must not change when result rank changes.
            result_id=stable_id(self.name, provider_record_id),
            artifact_id=artifact_id,
            provider=self.name,
            provider_record_id=provider_record_id,
            title=title,
            abstract=_openalex_abstract(item.get("abstract_inverted_index")),
            authors=authors,
            year=int(item["publication_year"]) if item.get("publication_year") else None,
            venue=str(source.get("display_name") or ""),
            doi=doi,
            url=url,
            publication_type=str(item.get("type") or "unknown"),
            subjects=subjects,
            retrieval=_retrieval_score(rank, provider_score),
        )
