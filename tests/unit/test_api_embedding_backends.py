from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from fitness_agents.config import (
    EmbeddingAPIConfig,
    LocalKnowledgeRetrievalConfig,
    RerankerAPIConfig,
    load_embedding_api_config,
    load_experiment_config,
    load_reranker_api_config,
)
from fitness_agents.local_knowledge.api_backends import (
    DashScopeEmbeddingBackend,
    DashScopeRerankerBackend,
    HTTPJSONResponse,
    JinaEmbeddingBackend,
    JinaRerankerBackend,
    OpenAICompatibleEmbeddingBackend,
    build_embedding_backend,
)


class _FakeTransport:
    def __init__(self, *payloads: Any) -> None:
        self.responses = list(payloads)
        self.requests: list[dict[str, Any]] = []

    def post_json(
        self,
        endpoint: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> HTTPJSONResponse:
        self.requests.append(
            {
                "endpoint": endpoint,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        response = self.responses.pop(0)
        return HTTPJSONResponse(status_code=200, headers={}, payload=response)


def _embedding_config(provider: str, **overrides: Any) -> EmbeddingAPIConfig:
    values: dict[str, Any] = {
        "provider": provider,
        "endpoint": (
            "https://workspace.example.test/api/v1/services/embeddings/"
            "text-embedding/text-embedding"
            if provider == "dashscope"
            else "https://inference.example.test/v1/embeddings"
        ),
        "api_key": "unit-test-secret",
        "model": "test-embedding",
        "model_family": "custom",
        "model_revision": "sha256:test",
        "dimension": 3,
        "max_input_tokens": 512,
        "batch_size": 2,
        "max_retries": 0,
    }
    values.update(overrides)
    return EmbeddingAPIConfig(**values)


def test_dashscope_separates_query_and_document_contracts_and_normalizes() -> None:
    transport = _FakeTransport(
        {
            "output": {
                "embeddings": [
                    {"text_index": 0, "embedding": [3.0, 4.0, 0.0]},
                ]
            }
        },
        {
            "output": {
                "embeddings": [
                    {"text_index": 1, "embedding": [0.0, 0.0, 2.0]},
                    {"text_index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            }
        },
    )
    config = _embedding_config(
        "dashscope",
        model="text-embedding-v4",
        model_family="qwen",
        query_task="query",
        document_task="document",
        query_instruction="Retrieve direct scientific evidence.",
    )
    backend = DashScopeEmbeddingBackend(config, transport=transport)

    query = backend.encode_queries(["How does epistasis affect mutation outcomes?"])
    documents = backend.encode_documents(["Epistasis is context dependent.", "Tax policy."])

    assert query.shape == (1, 3)
    assert documents.shape == (2, 3)
    assert np.allclose(np.linalg.norm(query, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(documents, axis=1), 1.0)
    assert transport.requests[0]["payload"]["parameters"] == {
        "dimension": 3,
        "output_type": "dense",
        "text_type": "query",
        "instruct": "Retrieve direct scientific evidence.",
    }
    assert transport.requests[1]["payload"]["parameters"]["text_type"] == "document"
    assert transport.requests[1]["payload"]["input"]["texts"] == [
        "Epistasis is context dependent.",
        "Tax policy.",
    ]
    assert "unit-test-secret" not in str(backend.fingerprint)
    assert backend.fingerprint["api_key_source"] == "literal:redacted"


def test_jina_uses_asymmetric_tasks_and_forbids_server_truncation() -> None:
    transport = _FakeTransport(
        {"data": [{"index": 0, "embedding": [1.0, 1.0, 0.0]}]},
        {"data": [{"index": 0, "embedding": [0.0, 1.0, 1.0]}]},
    )
    config = _embedding_config(
        "jina",
        model="jina-embeddings-v5-text-small",
        model_family="jina",
        query_task="retrieval.query",
        document_task="retrieval.passage",
    )
    backend = JinaEmbeddingBackend(config, transport=transport)

    backend.encode_queries(["protein engineering evidence"])
    backend.encode_documents(["A supported atomic scientific claim."])

    assert transport.requests[0]["payload"]["task"] == "retrieval.query"
    assert transport.requests[1]["payload"]["task"] == "retrieval.passage"
    assert transport.requests[0]["payload"]["truncate"] is False
    assert transport.requests[0]["payload"]["normalized"] is True


def test_openai_compatible_backend_supports_e5_prefixes_and_batch_order() -> None:
    transport = _FakeTransport(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 2.0, 0.0]},
                {"index": 0, "embedding": [2.0, 0.0, 0.0]},
            ]
        }
    )
    config = _embedding_config(
        "tei",
        model="intfloat/e5-base-v2",
        model_family="e5",
        query_prefix="query: ",
        document_prefix="passage: ",
    )
    backend = OpenAICompatibleEmbeddingBackend(config, transport=transport)
    vectors = backend.encode_queries(["first", "second"])

    assert transport.requests[0]["payload"]["input"] == ["query: first", "query: second"]
    assert np.allclose(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_remote_backend_rejects_unset_secret_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEST_EMBEDDING_API_KEY", raising=False)
    config = _embedding_config("jina", api_key="${TEST_EMBEDDING_API_KEY}")
    with pytest.raises(RuntimeError, match="environment variable"):
        JinaEmbeddingBackend(config, transport=_FakeTransport())


def test_remote_backend_rejects_unpinned_example_revision() -> None:
    config = _embedding_config("tei", model_revision="REPLACE_WITH_PINNED_HF_COMMIT")
    with pytest.raises(RuntimeError, match="not pinned"):
        OpenAICompatibleEmbeddingBackend(config, transport=_FakeTransport())


def test_remote_backend_rejects_oversized_text_before_request() -> None:
    transport = _FakeTransport()
    config = _embedding_config("jina", max_input_tokens=12)
    backend = JinaEmbeddingBackend(config, transport=transport)
    with pytest.raises(ValueError, match="silent truncation is disabled"):
        backend.encode_documents(["This text is longer than twelve conservative byte tokens."])
    assert transport.requests == []


def test_jina_reranker_restores_original_document_order() -> None:
    transport = _FakeTransport(
        {
            "results": [
                {"index": 1, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.08},
            ]
        }
    )
    config = RerankerAPIConfig(
        provider="jina",
        endpoint="https://api.jina.ai/v1/rerank",
        api_key="unit-test-secret",
        model="jina-reranker-v3",
        model_family="jina",
        model_revision="provider-managed:test",
        max_input_tokens=512,
        max_documents=8,
        max_retries=0,
    )
    backend = JinaRerankerBackend(config, transport=transport)
    scores = backend.score("epistasis", ["tax policy", "genetic background dependence"])

    assert np.allclose(scores, [0.08, 0.91])
    assert backend.score_kind == "probability"
    assert transport.requests[0]["payload"]["top_n"] == 2
    assert transport.requests[0]["payload"]["return_documents"] is False


def test_dashscope_qwen3_reranker_uses_official_wire_contract() -> None:
    transport = _FakeTransport(
        {
            "object": "list",
            "results": [
                {"index": 1, "relevance_score": 0.92},
                {"index": 0, "relevance_score": 0.07},
            ],
            "model": "qwen3-rerank",
        }
    )
    config = RerankerAPIConfig(
        provider="dashscope",
        endpoint="https://workspace.example.test/compatible-api/v1/reranks",
        api_key="unit-test-secret",
        model="qwen3-rerank",
        model_family="qwen",
        model_revision="provider-managed:test",
        max_input_tokens=4000,
        max_documents=64,
        max_retries=0,
        instruction="Rank direct scientific evidence first.",
    )
    backend = DashScopeRerankerBackend(config, transport=transport)
    scores = backend.score("What constrains mutation effects?", ["Tax policy.", "Epistasis."])

    assert np.allclose(scores, [0.07, 0.92])
    assert transport.requests[0]["endpoint"].endswith("/compatible-api/v1/reranks")
    assert transport.requests[0]["payload"] == {
        "model": "qwen3-rerank",
        "query": "What constrains mutation effects?",
        "documents": ["Tax policy.", "Epistasis."],
        "top_n": 2,
        "instruct": "Rank direct scientific evidence first.",
    }


def test_dashscope_rejects_base_urls_instead_of_full_operation_endpoints() -> None:
    with pytest.raises(ValueError, match="full provider endpoint path"):
        _embedding_config(
            "dashscope",
            endpoint="https://workspace.example.test/compatible-mode/v1",
            model="text-embedding-v4",
            model_family="qwen",
        )
    with pytest.raises(ValueError, match="full provider endpoint path"):
        RerankerAPIConfig(
            provider="dashscope",
            endpoint="https://workspace.example.test/api/v1",
            api_key="unit-test-secret",
            model="qwen3-rerank",
            model_family="qwen",
            model_revision="provider-managed:test",
            max_input_tokens=4000,
        )


def test_checked_in_qwen_api_configs_use_official_beijing_endpoints() -> None:
    embedding = load_embedding_api_config("configs/knowledge/api/embedding.qwen-v4.yaml")
    reranker = load_reranker_api_config("configs/knowledge/api/reranker.qwen3.yaml")

    assert embedding.endpoint.endswith(
        "/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    assert embedding.api_key == "env:DASHSCOPE_API_KEY"
    assert embedding.dimension == 1024
    assert embedding.max_input_tokens == 8192
    assert reranker.endpoint.endswith("/compatible-api/v1/reranks")
    assert reranker.api_key == "env:DASHSCOPE_API_KEY"
    assert reranker.max_input_tokens == 4000


def test_qwen_api_experiment_keeps_sqlite_local_and_remote_models_explicit() -> None:
    experiment = load_experiment_config("configs/experiments/knowledge_agent_qwen_api.yaml")
    local = experiment.knowledge.local_knowledge

    assert local.corpus_index_path is not None
    assert local.corpus_index_path.suffix == ".sqlite"
    assert local.retrieval_overlay_path is not None
    assert local.retrieval_overlay_path.suffix == ".sqlite"
    assert local.retrieval.embedding_backend == "api"
    assert local.retrieval.embedding_model_path is None
    assert local.retrieval.reranker_backend == "api"
    assert local.retrieval.reranker_model_path is None


def test_unified_qwen_catalog_resolves_llm_embedding_and_reranker_items() -> None:
    experiment = load_experiment_config(
        "configs/experiments/knowledge_agent_qwen_unified_api.yaml"
    )
    embedding = experiment.knowledge.local_knowledge.retrieval.embedding_api_config
    reranker = experiment.knowledge.local_knowledge.retrieval.reranker_api_config

    assert experiment.llm.provider == "openai_compatible"
    assert experiment.llm.model == "qwen-plus"
    assert experiment.llm.base_url is not None
    assert experiment.llm.base_url.endswith("/compatible-mode/v1")
    assert experiment.llm.api_key == "env:DASHSCOPE_API_KEY"
    assert embedding is not None and embedding.api_key == experiment.llm.api_key
    assert reranker is not None and reranker.api_key == experiment.llm.api_key


def test_external_yaml_loader_and_factory_keep_model_and_provider_separate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "embedding.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "embedding-api:v1",
                "provider": "tei",
                "endpoint": "http://localhost:8080/v1/embeddings",
                "api_key": "none",
                "model": "BAAI/bge-m3",
                "model_family": "bge",
                "model_revision": "pinned-test-revision",
                "dimension": 3,
                "max_input_tokens": 8192,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    api = load_embedding_api_config(path)
    transport = _FakeTransport({"data": [{"index": 0, "embedding": [1.0, 0.0, 0.0]}]})
    retrieval = LocalKnowledgeRetrievalConfig(
        mode="dense",
        dense_enabled=True,
        embedding_backend="api",
        embedding_api_config=api,
    )
    backend = build_embedding_backend(retrieval, transport=transport)

    assert backend is not None
    assert backend.fingerprint["provider"] == "tei"
    assert backend.fingerprint["model_family"] == "bge"
    assert backend.fingerprint["model_id"] == "BAAI/bge-m3"
    assert backend.encode_documents(["atomic evidence"]).shape == (1, 3)
