from __future__ import annotations

import hashlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from fitness_agents.config import (
    EmbeddingAPIConfig,
    LocalKnowledgeRetrievalConfig,
    RerankerAPIConfig,
)

from .embeddings import (
    SentenceTransformerEmbeddingBackend,
    SentenceTransformerRerankerBackend,
)
from .protocols import EmbeddingBackend, RerankerBackend

_ENV_REFERENCE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")
_PLACEHOLDER_MARKERS = ("<YOUR_", "{WORKSPACE_ID}", "YOUR_WORKSPACE_ID")


@dataclass(frozen=True)
class HTTPJSONResponse:
    status_code: int
    headers: Mapping[str, str]
    payload: Any


class JSONHTTPTransport(Protocol):
    def post_json(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse: ...


class UrllibJSONTransport:
    """Small dependency-free JSON transport used by remote RAG backends."""

    def post_json(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        request = Request(
            endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=dict(headers),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
                return HTTPJSONResponse(
                    status_code=int(response.status),
                    headers=dict(response.headers.items()),
                    payload=parsed,
                )
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body) if body else {}
            except json.JSONDecodeError:
                parsed = {}
            return HTTPJSONResponse(
                status_code=int(error.code),
                headers=dict(error.headers.items()) if error.headers else {},
                payload=parsed,
            )


def _resolve_api_key(reference: str) -> tuple[str | None, str]:
    from fitness_agents.agents.remote_llm import load_project_env

    load_project_env()
    if reference.casefold() == "none":
        return None, "none"
    if reference.startswith("env:"):
        variable = reference[4:].strip()
        if not variable:
            raise RuntimeError("Malformed API key environment reference")
        value = os.getenv(variable)
        if not value:
            raise RuntimeError(
                f"API key environment variable {variable!r} is not set; "
                "the checked-in config contains only a reference"
            )
        return value, f"env:{variable}"
    match = _ENV_REFERENCE.fullmatch(reference.strip())
    if match is None:
        if reference.startswith("${") or "PLACEHOLDER" in reference.upper():
            raise RuntimeError("Malformed or unresolved API key placeholder")
        return reference, "literal:redacted"
    variable = match.group(1)
    value = os.getenv(variable)
    if not value:
        raise RuntimeError(
            f"API key environment variable {variable!r} is not set; "
            "the checked-in config contains only a placeholder"
        )
    return value, f"env:{variable}"


def _assert_runtime_endpoint(endpoint: str) -> None:
    upper = endpoint.upper()
    if any(marker in upper for marker in _PLACEHOLDER_MARKERS):
        raise RuntimeError("API endpoint still contains a workspace/deployment placeholder")


def _assert_runtime_revision(revision: str) -> None:
    upper = revision.upper()
    if "REPLACE_WITH" in upper or upper in {"LATEST", "MAIN"}:
        raise RuntimeError(
            "Remote model_revision is not pinned; replace the example value with a deployment "
            "revision or an explicit provider-managed release label"
        )


def _safe_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tree_hash(path: Path) -> str:
    candidates = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.casefold() in {".json", ".txt", ".model", ".tiktoken"}
    )
    if not candidates:
        raise RuntimeError(f"No tokenizer files were found under {path}")
    digest = hashlib.sha256()
    for item in candidates:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


class _TokenCounter(Protocol):
    fingerprint: Mapping[str, object]

    def count(self, text: str) -> int: ...


class _ConservativeUTF8TokenCounter:
    """Provider-independent upper bound for ordinary byte-backed tokenizers.

    It intentionally over-counts English scientific prose so an API is never asked to
    truncate silently. Exact provider token accounting should use a pinned local tokenizer.
    """

    fingerprint: ClassVar[Mapping[str, object]] = {
        "strategy": "conservative_utf8_bytes",
        "version": "v1",
        "special_token_margin": 8,
        "exact": False,
    }

    def count(self, text: str) -> int:
        return max(1, len(text.encode("utf-8")) + 8)


class _HuggingFaceTokenCounter:
    def __init__(
        self,
        path: Path,
        *,
        model_id: str | None,
        revision: str | None,
    ) -> None:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Configured tokenizer does not exist: {resolved}")
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for exact API token counting; install the rag extra"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(str(resolved), local_files_only=True)
        self.fingerprint = {
            "strategy": "huggingface_tokenizer",
            "model_id": model_id or resolved.name,
            "revision": revision or "local",
            "tokenizer_hash": _tree_hash(resolved),
            "exact": True,
        }

    def count(self, text: str) -> int:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids", ())
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return len(input_ids)


def _token_counter(
    path: Path | None,
    *,
    model_id: str | None,
    revision: str | None,
) -> _TokenCounter:
    if path is None:
        return _ConservativeUTF8TokenCounter()
    return _HuggingFaceTokenCounter(path, model_id=model_id, revision=revision)


class _RemoteAPIBackend(ABC):
    def __init__(
        self,
        *,
        endpoint: str,
        api_key_reference: str,
        timeout_seconds: float,
        max_retries: int,
        transport: JSONHTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _assert_runtime_endpoint(endpoint)
        self.endpoint = endpoint
        self._api_key, self.api_key_source = _resolve_api_key(api_key_reference)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.transport = transport or UrllibJSONTransport()
        self._sleep = sleep

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key is not None:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _post(self, payload: Mapping[str, Any]) -> Any:
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.transport.post_json(
                    self.endpoint,
                    headers=self._headers(),
                    payload=payload,
                    timeout_seconds=self.timeout_seconds,
                )
            except (OSError, TimeoutError, URLError) as error:
                if attempt >= self.max_retries:
                    raise RuntimeError("Remote inference request failed after retries") from error
                self._sleep(min(8.0, 0.25 * (2**attempt)))
                continue
            last_status = response.status_code
            if 200 <= response.status_code < 300:
                return response.payload
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt >= self.max_retries:
                request_id = None
                error_code = None
                error_message = None
                if isinstance(response.payload, dict):
                    request_id = response.payload.get("request_id") or response.payload.get("id")
                    error_code = response.payload.get("code")
                    error_message = response.payload.get("message")
                suffix = f" request_id={request_id}" if request_id else ""
                if error_code:
                    suffix += f" code={error_code}"
                if error_message:
                    safe_message = str(error_message).replace("\r", " ").replace("\n", " ")[:300]
                    suffix += f" message={safe_message}"
                raise RuntimeError(
                    f"Remote inference request returned HTTP {response.status_code}.{suffix}"
                )
            raw_retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(30.0, max(0.0, float(raw_retry_after)))
            except ValueError:
                delay = min(8.0, 0.25 * (2**attempt))
            self._sleep(delay)
        raise RuntimeError(f"Remote inference request failed with HTTP {last_status}")


class APIEmbeddingBackend(_RemoteAPIBackend, ABC):
    def __init__(
        self,
        config: EmbeddingAPIConfig,
        *,
        transport: JSONHTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _assert_runtime_revision(config.model_revision)
        super().__init__(
            endpoint=config.endpoint,
            api_key_reference=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            transport=transport,
            sleep=sleep,
        )
        self.config = config
        self.dimension = config.dimension
        self.max_input_tokens = config.max_input_tokens
        self._counter = _token_counter(
            config.tokenizer_model_path,
            model_id=config.tokenizer_model_id,
            revision=config.tokenizer_revision,
        )
        identity = {
            "backend": "remote-api",
            "provider": config.provider,
            "model_family": config.model_family,
            "model_id": config.model,
            "model_revision": config.model_revision,
            "endpoint_sha256": hashlib.sha256(config.endpoint.encode("utf-8")).hexdigest(),
            "dimension": config.dimension,
            "max_input_tokens": config.max_input_tokens,
            "batch_size": config.batch_size,
            "query_task": config.query_task,
            "document_task": config.document_task,
            "query_instruction": config.query_instruction,
            "document_instruction": config.document_instruction,
            "query_prefix": config.query_prefix,
            "document_prefix": config.document_prefix,
            "token_counter": dict(self._counter.fingerprint),
            "normalize_embeddings": True,
            "api_key_source": self.api_key_source,
        }
        self.fingerprint: dict[str, object] = identity
        self.name = (
            f"remote-api:{config.provider}:{config.model}@{config.model_revision}:"
            f"{_safe_hash(identity)[:16]}"
        )

    def _instruction(self, *, query: bool) -> str | None:
        return self.config.query_instruction if query else self.config.document_instruction

    def _task(self, *, query: bool) -> str | None:
        return self.config.query_task if query else self.config.document_task

    def _prepare(self, text: str, *, query: bool) -> str:
        prefix = self.config.query_prefix if query else self.config.document_prefix
        return f"{prefix}{text}"

    def count_tokens(self, text: str, *, query: bool = False) -> int:
        prepared = self._prepare(text, query=query)
        instruction = self._instruction(query=query)
        if instruction:
            prepared = f"{instruction}\n{prepared}"
        return self._counter.count(prepared)

    def _validate_lengths(self, texts: Sequence[str], *, query: bool) -> None:
        for index, text in enumerate(texts):
            count = self.count_tokens(str(text), query=query)
            if count > self.max_input_tokens:
                kind = "query" if query else "document"
                raise ValueError(
                    f"Remote embedding {kind} {index} uses {count} tokens, exceeding "
                    f"max_input_tokens={self.max_input_tokens}; silent truncation is disabled"
                )

    @abstractmethod
    def _request_payload(self, texts: Sequence[str], *, query: bool) -> Mapping[str, Any]: ...

    @abstractmethod
    def _extract_vectors(self, payload: Any, *, expected: int) -> Sequence[Sequence[float]]: ...

    def _encode(self, texts: Sequence[str], *, query: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        raw_texts = [str(item) for item in texts]
        self._validate_lengths(raw_texts, query=query)
        matrices: list[np.ndarray] = []
        for start in range(0, len(raw_texts), self.config.batch_size):
            batch = raw_texts[start : start + self.config.batch_size]
            response = self._post(self._request_payload(batch, query=query))
            raw_vectors = self._extract_vectors(response, expected=len(batch))
            matrix = np.asarray(raw_vectors, dtype=np.float32)
            if matrix.shape != (len(batch), self.dimension):
                raise RuntimeError(
                    "Remote embedding API returned shape "
                    f"{matrix.shape}, expected {(len(batch), self.dimension)}"
                )
            if not np.isfinite(matrix).all():
                raise RuntimeError("Remote embedding API returned non-finite values")
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise RuntimeError("Remote embedding API returned a zero vector")
            matrices.append(matrix / norms)
        return np.vstack(matrices).astype(np.float32, copy=False)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=False)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, query=True)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode_documents(texts)


def _indexed_vectors(
    entries: Any,
    *,
    index_field: str,
    vector_field: str,
    expected: int,
) -> list[Sequence[float]]:
    if not isinstance(entries, list):
        raise TypeError("Remote embedding response does not contain an embedding list")
    ordered: list[Sequence[float] | None] = [None] * expected
    for fallback_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError("Remote embedding response contains a malformed item")
        index = int(entry.get(index_field, fallback_index))
        if not 0 <= index < expected or ordered[index] is not None:
            raise RuntimeError("Remote embedding response contains invalid or duplicate indices")
        vector = entry.get(vector_field)
        if not isinstance(vector, list):
            raise TypeError("Remote embedding response item has no float vector")
        ordered[index] = vector
    if any(item is None for item in ordered):
        raise RuntimeError("Remote embedding response omitted one or more inputs")
    return [item for item in ordered if item is not None]


class DashScopeEmbeddingBackend(APIEmbeddingBackend):
    def _request_payload(self, texts: Sequence[str], *, query: bool) -> Mapping[str, Any]:
        parameters: dict[str, Any] = {
            "dimension": self.dimension,
            "output_type": "dense",
            "text_type": self._task(query=query) or ("query" if query else "document"),
        }
        instruction = self._instruction(query=query)
        if instruction:
            parameters["instruct"] = instruction
        return {
            "model": self.config.model,
            "input": {"texts": [self._prepare(text, query=query) for text in texts]},
            "parameters": parameters,
        }

    def _extract_vectors(self, payload: Any, *, expected: int) -> Sequence[Sequence[float]]:
        if not isinstance(payload, dict) or not isinstance(payload.get("output"), dict):
            raise TypeError("Malformed DashScope embedding response")
        return _indexed_vectors(
            payload["output"].get("embeddings"),
            index_field="text_index",
            vector_field="embedding",
            expected=expected,
        )


class JinaEmbeddingBackend(APIEmbeddingBackend):
    def _request_payload(self, texts: Sequence[str], *, query: bool) -> Mapping[str, Any]:
        return {
            "model": self.config.model,
            "input": [self._prepare(text, query=query) for text in texts],
            "task": self._task(query=query)
            or ("retrieval.query" if query else "retrieval.passage"),
            "dimensions": self.dimension,
            "embedding_type": "float",
            "normalized": True,
            "truncate": False,
        }

    def _extract_vectors(self, payload: Any, *, expected: int) -> Sequence[Sequence[float]]:
        if not isinstance(payload, dict):
            raise TypeError("Malformed Jina embedding response")
        return _indexed_vectors(
            payload.get("data"),
            index_field="index",
            vector_field="embedding",
            expected=expected,
        )


class OpenAICompatibleEmbeddingBackend(APIEmbeddingBackend):
    def _request_payload(self, texts: Sequence[str], *, query: bool) -> Mapping[str, Any]:
        return {
            "model": self.config.model,
            "input": [self._prepare(text, query=query) for text in texts],
            "dimensions": self.dimension,
            "encoding_format": "float",
        }

    def _extract_vectors(self, payload: Any, *, expected: int) -> Sequence[Sequence[float]]:
        if not isinstance(payload, dict):
            raise TypeError("Malformed OpenAI-compatible embedding response")
        return _indexed_vectors(
            payload.get("data"),
            index_field="index",
            vector_field="embedding",
            expected=expected,
        )


class APIRerankerBackend(_RemoteAPIBackend, ABC):
    def __init__(
        self,
        config: RerankerAPIConfig,
        *,
        transport: JSONHTTPTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        _assert_runtime_revision(config.model_revision)
        super().__init__(
            endpoint=config.endpoint,
            api_key_reference=config.api_key,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            transport=transport,
            sleep=sleep,
        )
        self.config = config
        self.score_kind = config.score_kind
        self._counter = _token_counter(
            config.tokenizer_model_path,
            model_id=config.tokenizer_model_id,
            revision=config.tokenizer_revision,
        )
        identity = {
            "backend": "remote-api-reranker",
            "provider": config.provider,
            "model_family": config.model_family,
            "model_id": config.model,
            "model_revision": config.model_revision,
            "endpoint_sha256": hashlib.sha256(config.endpoint.encode("utf-8")).hexdigest(),
            "max_input_tokens": config.max_input_tokens,
            "max_documents": config.max_documents,
            "instruction": config.instruction,
            "score_kind": config.score_kind,
            "token_counter": dict(self._counter.fingerprint),
            "api_key_source": self.api_key_source,
        }
        self.fingerprint: dict[str, object] = identity
        self.name = (
            f"remote-api-reranker:{config.provider}:{config.model}@{config.model_revision}:"
            f"{_safe_hash(identity)[:16]}"
        )

    def _validate(self, query: str, texts: Sequence[str]) -> None:
        if len(texts) > self.config.max_documents:
            raise ValueError(
                f"Reranker received {len(texts)} documents; configured maximum is "
                f"{self.config.max_documents}"
            )
        query_tokens = self._counter.count(query)
        if query_tokens > self.config.max_input_tokens:
            raise ValueError("Reranker query exceeds max_input_tokens; truncation is disabled")
        for index, text in enumerate(texts):
            if self._counter.count(text) > self.config.max_input_tokens:
                raise ValueError(
                    f"Reranker document {index} exceeds max_input_tokens; truncation is disabled"
                )

    @abstractmethod
    def _request_payload(self, query: str, texts: Sequence[str]) -> Mapping[str, Any]: ...

    @abstractmethod
    def _response_entries(self, payload: Any) -> Any: ...

    def score(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty(0, dtype=np.float32)
        normalized_texts = [str(item) for item in texts]
        self._validate(query, normalized_texts)
        response = self._post(self._request_payload(query, normalized_texts))
        entries = self._response_entries(response)
        if not isinstance(entries, list):
            raise TypeError("Remote reranker response does not contain results")
        scores = np.full(len(normalized_texts), np.nan, dtype=np.float32)
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("Remote reranker result is malformed")
            index = int(entry.get("index", -1))
            raw_score = entry.get("relevance_score", entry.get("score"))
            if not 0 <= index < len(scores) or raw_score is None or np.isfinite(scores[index]):
                raise RuntimeError("Remote reranker returned invalid or duplicate indices")
            scores[index] = float(raw_score)
        if not np.isfinite(scores).all():
            raise RuntimeError("Remote reranker omitted a score or returned a non-finite value")
        if self.score_kind == "probability" and np.any((scores < 0.0) | (scores > 1.0)):
            raise RuntimeError("Probability reranker returned a score outside [0, 1]")
        return scores


class DashScopeRerankerBackend(APIRerankerBackend):
    def _request_payload(self, query: str, texts: Sequence[str]) -> Mapping[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "query": query,
            "documents": list(texts),
            "top_n": len(texts),
        }
        if self.config.instruction:
            payload["instruct"] = self.config.instruction
        return payload

    def _response_entries(self, payload: Any) -> Any:
        if not isinstance(payload, dict):
            raise TypeError("Malformed DashScope reranker response")
        if isinstance(payload.get("results"), list):
            return payload["results"]
        output = payload.get("output")
        return output.get("results") if isinstance(output, dict) else None


class JinaRerankerBackend(APIRerankerBackend):
    def _request_payload(self, query: str, texts: Sequence[str]) -> Mapping[str, Any]:
        return {
            "model": self.config.model,
            "query": query,
            "documents": list(texts),
            "top_n": len(texts),
            "return_documents": False,
        }

    def _response_entries(self, payload: Any) -> Any:
        return payload.get("results") if isinstance(payload, dict) else None


class TEIRerankerBackend(APIRerankerBackend):
    def _request_payload(self, query: str, texts: Sequence[str]) -> Mapping[str, Any]:
        return {"query": query, "texts": list(texts), "truncate": False}

    def _response_entries(self, payload: Any) -> Any:
        if isinstance(payload, list):
            return payload
        return payload.get("results") if isinstance(payload, dict) else None


def build_embedding_backend(
    config: LocalKnowledgeRetrievalConfig,
    *,
    transport: JSONHTTPTransport | None = None,
) -> EmbeddingBackend | None:
    if not config.dense_enabled:
        return None
    if config.embedding_backend == "local":
        return SentenceTransformerEmbeddingBackend(
            config.embedding_model_path,  # type: ignore[arg-type]
            model_id=config.embedding_model_id,
            revision=config.embedding_model_revision,
            query_prefix=config.embedding_query_prefix,
            document_prefix=config.embedding_document_prefix,
        )
    api = config.embedding_api_config
    if not isinstance(api, EmbeddingAPIConfig):
        raise TypeError("API embedding backend was selected without a parsed API config")
    if api.provider == "dashscope":
        return DashScopeEmbeddingBackend(api, transport=transport)
    if api.provider == "jina":
        return JinaEmbeddingBackend(api, transport=transport)
    return OpenAICompatibleEmbeddingBackend(api, transport=transport)


def build_reranker_backend(
    config: LocalKnowledgeRetrievalConfig,
    *,
    transport: JSONHTTPTransport | None = None,
) -> RerankerBackend | None:
    if config.reranker_backend == "none":
        return None
    if config.reranker_backend == "local":
        return SentenceTransformerRerankerBackend(
            config.reranker_model_path,  # type: ignore[arg-type]
            model_id=config.reranker_model_id,
            revision=config.reranker_model_revision,
        )
    api = config.reranker_api_config
    if not isinstance(api, RerankerAPIConfig):
        raise TypeError("API reranker backend was selected without a parsed API config")
    if api.provider == "dashscope":
        return DashScopeRerankerBackend(api, transport=transport)
    if api.provider == "jina":
        return JinaRerankerBackend(api, transport=transport)
    return TEIRerankerBackend(api, transport=transport)
