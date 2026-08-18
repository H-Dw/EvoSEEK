from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .leakage import TargetLeakageGuard

OVERLAY_SCHEMA_VERSION = "local-knowledge-overlay:v1"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class SQLiteRetrievalOverlay:
    """Task-scoped leakage decisions and retrieval audit events.

    The corpus/vector index remains reusable across targets. This overlay is the only
    database that stores target-derived policy state and per-round queries.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS overlay_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS document_policy (
                document_id TEXT PRIMARY KEY,
                corpus_manifest_hash TEXT NOT NULL,
                allowed INTEGER NOT NULL,
                reasons_json TEXT NOT NULL,
                protected_terms_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS retrieval_events (
                query_id TEXT PRIMARY KEY,
                round_id INTEGER NOT NULL,
                original_query_hash TEXT NOT NULL,
                sanitized_query TEXT NOT NULL,
                policy_json TEXT NOT NULL,
                result_chunk_ids_json TEXT NOT NULL,
                corpus_manifest_hash TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def refresh_document_policy(
        self,
        *,
        corpus_manifest_hash: str,
        documents: tuple[dict[str, str], ...],
        guard: TargetLeakageGuard,
        quarantine_target_documents: bool,
    ) -> int:
        quarantined = 0
        with self.connection:
            self.connection.execute("DELETE FROM document_policy")
            for document in documents:
                reasons = guard.matches(text=document["text"], path=document["path"])
                allowed = not (
                    guard.enabled and quarantine_target_documents and bool(reasons)
                )
                quarantined += int(not allowed)
                self.connection.execute(
                    "INSERT INTO document_policy VALUES (?, ?, ?, ?, ?)",
                    (
                        document["document_id"],
                        corpus_manifest_hash,
                        int(allowed),
                        _json(reasons),
                        guard.protected_terms_hash,
                    ),
                )
            for key, value in (
                ("schema_version", OVERLAY_SCHEMA_VERSION),
                ("corpus_manifest_hash", corpus_manifest_hash),
                ("protected_terms_hash", guard.protected_terms_hash),
            ):
                self.connection.execute(
                    "INSERT OR REPLACE INTO overlay_metadata(key, value) VALUES (?, ?)",
                    (key, value),
                )
        return quarantined

    def is_document_allowed(self, document_id: str, *, corpus_manifest_hash: str) -> bool:
        row = self.connection.execute(
            "SELECT allowed, corpus_manifest_hash FROM document_policy WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if row is None or str(row["corpus_manifest_hash"]) != corpus_manifest_hash:
            return False
        return bool(row["allowed"])

    def record_retrieval(
        self,
        *,
        query_id: str,
        round_id: int,
        original_query_hash: str,
        sanitized_query: str,
        policy: dict[str, Any],
        result_chunk_ids: tuple[str, ...],
        corpus_manifest_hash: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO retrieval_events VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    query_id,
                    round_id,
                    original_query_hash,
                    sanitized_query,
                    _json(policy),
                    _json(result_chunk_ids),
                    corpus_manifest_hash,
                ),
            )

    def stats(self) -> dict[str, int | str]:
        return {
            "documents": int(
                self.connection.execute("SELECT COUNT(*) FROM document_policy").fetchone()[0]
            ),
            "quarantined_documents": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM document_policy WHERE allowed = 0"
                ).fetchone()[0]
            ),
            "retrieval_events": int(
                self.connection.execute("SELECT COUNT(*) FROM retrieval_events").fetchone()[0]
            ),
            "corpus_manifest_hash": self._metadata("corpus_manifest_hash") or "unbuilt",
        }

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM overlay_metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def close(self) -> None:
        self.connection.close()
