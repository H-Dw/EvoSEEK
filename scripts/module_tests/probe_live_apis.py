"""Redacted live probes for configured DeepSeek and Qwen API endpoints.

Credentials are read only from process environment variables. The receipt contains
model IDs, dimensions, hashes, and rankings, never keys or response text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np
from openai import OpenAI

from fitness_agents.config import (
    LocalKnowledgeRetrievalConfig,
    load_embedding_api_config,
    load_reranker_api_config,
)
from fitness_agents.local_knowledge.api_backends import (
    build_embedding_backend,
    build_reranker_backend,
)

SAFE_QUERY = (
    "Which evidence constrains sequence-context effects on a nonviral protein "
    "benchmark?"
)
SAFE_DOCUMENTS = (
    (
        "Sequence context can alter a measured protein property in a "
        "benchmark-dependent manner."
    ),
    "A control passage unrelated to the benchmark provides no direct constraint.",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepseek-models",
        default="deepseek-v4-pro,deepseek-v4-flash",
    )
    parser.add_argument(
        "--deepseek-max-tokens",
        type=int,
        default=2048,
        help="Bounded output budget; response text is still never emitted.",
    )
    return parser.parse_args()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _failure(error: Exception) -> dict[str, Any]:
    status = getattr(error, "status_code", None)
    return {
        "status": "failed",
        "error_type": type(error).__name__,
        "status_code": int(status) if status is not None else None,
        "message_sha256": _sha256_text(str(error)),
    }


def _deepseek_probe(
    models: tuple[str, ...], *, max_tokens: int
) -> dict[str, Any]:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    client = OpenAI(
        api_key=key,
        base_url="https://api.deepseek.com",
        timeout=60.0,
        max_retries=2,
    )
    try:
        available = sorted(item.id for item in client.models.list().data)
        catalog: dict[str, Any] = {"status": "passed"}
    except Exception as error:  # noqa: BLE001 - redacted live-provider receipt
        available = []
        catalog = _failure(error)
    completions: dict[str, dict[str, Any]] = {}
    for model in models:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=(
                    {
                        "role": "system",
                        "content": "Return one compact JSON object and no prose.",
                    },
                    {
                        "role": "user",
                        "content": (
                            "Confirm the API transport with "
                            '{"status":"ok","scope":"nonviral-benchmark"}.'
                        ),
                    },
                ),
                temperature=0.0,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            completions[model] = {
                "status": "passed",
                "response_model": response.model,
                "finish_reason": response.choices[0].finish_reason,
                "content_characters": len(content),
                "content_sha256": _sha256_text(content),
            }
        except Exception as error:  # noqa: BLE001 - isolate each live model probe
            completions[model] = _failure(error)
    return {
        "catalog": catalog,
        "available_models": available,
        "completions": completions,
    }


def _qwen_probe() -> dict[str, Any]:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise RuntimeError("DASHSCOPE_API_KEY is required")
    embedding = load_embedding_api_config(
        "configs/knowledge/api/embedding.qwen-v4.yaml"
    )
    reranker = load_reranker_api_config(
        "configs/knowledge/api/reranker.qwen3.yaml"
    )
    retrieval = LocalKnowledgeRetrievalConfig(
        mode="dense",
        dense_enabled=True,
        embedding_backend="api",
        embedding_api_config=embedding,
        reranker_backend="api",
        reranker_api_config=reranker,
        strict_query_language=True,
    )
    embedding_backend = build_embedding_backend(retrieval)
    reranker_backend = build_reranker_backend(retrieval)
    if embedding_backend is None or reranker_backend is None:
        raise RuntimeError("Configured Qwen backends were not created")
    query_vector = embedding_backend.encode_queries([SAFE_QUERY])[0]
    document_vectors = embedding_backend.encode_documents(list(SAFE_DOCUMENTS))
    scores = reranker_backend.score(SAFE_QUERY, list(SAFE_DOCUMENTS))
    return {
        "embedding_backend": embedding_backend.name,
        "embedding_fingerprint": embedding_backend.fingerprint,
        "query_dimension": int(query_vector.size),
        "query_norm": float(np.linalg.norm(query_vector)),
        "query_vector_sha256": hashlib.sha256(
            np.asarray(query_vector, dtype=np.float32).tobytes()
        ).hexdigest(),
        "document_dimensions": [int(item.size) for item in document_vectors],
        "reranker_backend": reranker_backend.name,
        "reranker_score_kind": reranker_backend.score_kind,
        "ranking": [
            {"document_index": int(index), "score": float(scores[index])}
            for index in np.argsort(-scores)
        ],
    }


def main() -> None:
    args = _arguments()
    models = tuple(
        item.strip() for item in args.deepseek_models.split(",") if item.strip()
    )
    receipt = {
        "schema_version": "live-api-redacted-probe:v1",
        "deepseek": _deepseek_probe(models, max_tokens=args.deepseek_max_tokens),
        "qwen": _qwen_probe(),
        "credentials_persisted": False,
        "response_text_emitted": False,
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
