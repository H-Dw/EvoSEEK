from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from fitness_agents.config import LocalKnowledgeConfig

from .chunking import CHUNKER_VERSION, chunk_document
from .contracts import DocumentChunk, IndexBuildReport
from .leakage import TargetLeakageGuard
from .parsers import AutoLocalParser, discover_local_files
from .protocols import EmbeddingBackend

INDEX_SCHEMA_VERSION = "local-knowledge-index:v2"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class SQLiteLocalKnowledgeIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL DEFAULT 'unclassified',
                    metadata_json TEXT NOT NULL,
                    quarantined INTEGER NOT NULL DEFAULT 0,
                    quarantine_reasons_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    section_path_json TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    source_group TEXT NOT NULL,
                    artifact_uri TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    knowledge_type TEXT NOT NULL DEFAULT 'unclassified',
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE INDEX IF NOT EXISTS idx_local_chunks_document
                    ON chunks(document_id);
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT PRIMARY KEY,
                    backend_name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)
                );
                CREATE TABLE IF NOT EXISTS retrieval_events (
                    query_id TEXT PRIMARY KEY,
                    round_id INTEGER NOT NULL,
                    original_query_hash TEXT NOT NULL,
                    sanitized_query TEXT NOT NULL,
                    policy_json TEXT NOT NULL,
                    result_chunk_ids_json TEXT NOT NULL,
                    manifest_hash TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    artifact_uri UNINDEXED,
                    tokenize='unicode61'
                );
                """
            )
        except sqlite3.OperationalError as error:
            raise RuntimeError("SQLite FTS5 is required for local knowledge retrieval") from error
        self._ensure_column(
            "documents",
            "knowledge_type",
            "TEXT NOT NULL DEFAULT 'unclassified'",
        )
        self._ensure_column(
            "chunks",
            "knowledge_type",
            "TEXT NOT NULL DEFAULT 'unclassified'",
        )
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        columns = {
            str(row[1]) for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if column not in columns:
            self.connection.execute(
                f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
            )

    @property
    def manifest_hash(self) -> str:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = 'manifest_hash'"
        ).fetchone()
        return str(row[0]) if row else "unbuilt"

    def build(
        self,
        config: LocalKnowledgeConfig,
        *,
        guard: TargetLeakageGuard,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> IndexBuildReport:
        parser = AutoLocalParser(
            rich_document_backend=config.ingestion.rich_document_backend
        )
        files = discover_local_files(
            config.roots, follow_symlinks=config.ingestion.follow_symlinks
        )
        existing = {
            str(row["path"]): (str(row["document_id"]), str(row["file_hash"]))
            for row in self.connection.execute("SELECT document_id, path, file_hash FROM documents")
        }
        current_paths = {str(path) for path in files}
        removed = sorted(set(existing).difference(current_paths))
        indexed_documents = 0
        indexed_chunks = 0
        unchanged_documents = 0
        quarantined_documents = 0
        warnings: list[str] = []
        manifest_entries: list[dict[str, Any]] = []

        with self.connection:
            for path_text in removed:
                document_id = existing[path_text][0]
                chunk_ids = [
                    str(row[0])
                    for row in self.connection.execute(
                        "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
                    )
                ]
                for chunk_id in chunk_ids:
                    self.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,))
                self.connection.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE document_id = ?)", (document_id,))
                self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
                self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

            for path in files:
                if path.stat().st_size > config.ingestion.max_file_mb * 1024 * 1024:
                    warnings.append(f"file_too_large:{path}")
                    continue
                if not parser.supports(path):
                    warnings.append(f"unsupported_file:{path}")
                    continue
                try:
                    document = parser.parse(path)
                except (OSError, TypeError, UnicodeError, ValueError, RuntimeError) as error:
                    warnings.append(f"parse_failed:{path}:{type(error).__name__}")
                    continue
                quarantine_reasons = guard.matches(text=document.text, path=document.path)
                quarantined = bool(
                    guard.enabled
                    and config.leakage_guard.quarantine_target_documents
                    and quarantine_reasons
                )
                manifest_entries.append(
                    {
                        "path": str(path),
                        "file_hash": document.file_hash,
                        "document_id": document.document_id,
                        "knowledge_type": document.knowledge_type,
                        "quarantined": quarantined,
                    }
                )
                previous = existing.get(str(path))
                policy_changed = (
                    self._metadata("protected_terms_hash") != guard.protected_terms_hash
                    or self._metadata("schema_version") != INDEX_SCHEMA_VERSION
                )
                if previous == (document.document_id, document.file_hash) and not policy_changed:
                    unchanged_documents += 1
                    quarantined_documents += int(quarantined)
                    continue
                if previous is not None:
                    old_document_id = previous[0]
                    old_chunk_ids = [
                        str(row[0])
                        for row in self.connection.execute(
                            "SELECT chunk_id FROM chunks WHERE document_id = ?",
                            (old_document_id,),
                        )
                    ]
                    for chunk_id in old_chunk_ids:
                        self.connection.execute(
                            "DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk_id,)
                        )
                    self.connection.execute(
                        "DELETE FROM embeddings WHERE chunk_id IN "
                        "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
                        (old_document_id,),
                    )
                    self.connection.execute(
                        "DELETE FROM chunks WHERE document_id = ?", (old_document_id,)
                    )
                    self.connection.execute(
                        "DELETE FROM documents WHERE document_id = ?", (old_document_id,)
                    )
                self.connection.execute(
                    "INSERT INTO documents("
                    "document_id, path, file_hash, mime_type, title, knowledge_type, "
                    "metadata_json, quarantined, quarantine_reasons_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.document_id,
                        str(document.path),
                        document.file_hash,
                        document.mime_type,
                        document.title,
                        document.knowledge_type,
                        _json(document.metadata),
                        int(quarantined),
                        _json(quarantine_reasons),
                    ),
                )
                indexed_documents += 1
                if quarantined:
                    quarantined_documents += 1
                    continue
                chunks = chunk_document(
                    document,
                    chunk_tokens=config.ingestion.chunk_tokens,
                    chunk_overlap=config.ingestion.chunk_overlap,
                    source_group=config.kg_update.source_group,
                )
                for chunk in chunks:
                    self._insert_chunk(chunk)
                indexed_chunks += len(chunks)
                if embedding_backend is not None and chunks:
                    vectors = embedding_backend.encode([item.text for item in chunks])
                    if vectors.shape != (len(chunks), embedding_backend.dimension):
                        raise RuntimeError("Embedding backend returned an unexpected matrix shape")
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        self.connection.execute(
                            "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                            (
                                chunk.chunk_id,
                                embedding_backend.name,
                                embedding_backend.dimension,
                                np.asarray(vector, dtype=np.float32).tobytes(),
                            ),
                        )

            manifest_payload = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "parser": parser.name,
                "embedding_backend": getattr(embedding_backend, "name", None),
                "embedding_dimension": getattr(embedding_backend, "dimension", None),
                "protected_terms_hash": guard.protected_terms_hash,
                "documents": sorted(manifest_entries, key=lambda item: item["path"]),
            }
            manifest_hash = hashlib.sha256(
                _json(manifest_payload).encode("utf-8")
            ).hexdigest()
            for key, value in (
                ("manifest_hash", manifest_hash),
                ("manifest", _json(manifest_payload)),
                ("protected_terms_hash", guard.protected_terms_hash),
                ("schema_version", INDEX_SCHEMA_VERSION),
            ):
                self.connection.execute(
                    "INSERT OR REPLACE INTO index_metadata(key, value) VALUES (?, ?)",
                    (key, value),
                )
        return IndexBuildReport(
            manifest_hash=manifest_hash,
            indexed_documents=indexed_documents,
            indexed_chunks=indexed_chunks,
            unchanged_documents=unchanged_documents,
            removed_documents=len(removed),
            quarantined_documents=quarantined_documents,
            warnings=tuple(warnings),
        )

    def _metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def _insert_chunk(self, chunk: DocumentChunk) -> None:
        self.connection.execute(
            "INSERT INTO chunks("
            "chunk_id, document_id, text, section_path_json, start_offset, end_offset, "
            "token_count, source_group, artifact_uri, file_hash, knowledge_type, metadata_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                chunk.chunk_id,
                chunk.document_id,
                chunk.text,
                _json(chunk.section_path),
                chunk.start_offset,
                chunk.end_offset,
                chunk.token_count,
                chunk.source_group,
                chunk.artifact_uri,
                chunk.file_hash,
                chunk.knowledge_type,
                _json(chunk.metadata),
            ),
        )
        self.connection.execute(
            "INSERT INTO chunks_fts(chunk_id, text, artifact_uri) VALUES (?, ?, ?)",
            (chunk.chunk_id, chunk.text, chunk.artifact_uri),
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]+", query.casefold())
        return " OR ".join(f'"{item.replace(chr(34), "")}"' for item in tokens[:32])

    def lexical_search(
        self,
        query: str,
        *,
        limit: int,
        knowledge_types: tuple[str, ...] = (),
    ) -> tuple[tuple[str, float], ...]:
        fts_query = self._fts_query(query)
        if not fts_query:
            return ()
        filter_sql = ""
        parameters: list[Any] = [fts_query]
        if knowledge_types:
            placeholders = ",".join("?" for _ in knowledge_types)
            filter_sql = f" AND chunks.knowledge_type IN ({placeholders})"
            parameters.extend(knowledge_types)
        parameters.append(limit)
        rows = self.connection.execute(
            "SELECT chunks_fts.chunk_id, bm25(chunks_fts) AS rank FROM chunks_fts "
            "JOIN chunks ON chunks.chunk_id = chunks_fts.chunk_id "
            f"WHERE chunks_fts MATCH ?{filter_sql} ORDER BY rank LIMIT ?",
            tuple(parameters),
        ).fetchall()
        return tuple((str(row["chunk_id"]), float(row["rank"])) for row in rows)

    def dense_search(
        self,
        query: str,
        *,
        limit: int,
        embedding_backend: EmbeddingBackend,
        knowledge_types: tuple[str, ...] = (),
    ) -> tuple[tuple[str, float], ...]:
        query_vector = np.asarray(embedding_backend.encode([query])[0], dtype=np.float32)
        filter_sql = ""
        parameters: list[Any] = [embedding_backend.name]
        if knowledge_types:
            placeholders = ",".join("?" for _ in knowledge_types)
            filter_sql = f" AND chunks.knowledge_type IN ({placeholders})"
            parameters.extend(knowledge_types)
        rows = self.connection.execute(
            "SELECT embeddings.chunk_id, embeddings.dimension, embeddings.vector "
            "FROM embeddings JOIN chunks ON chunks.chunk_id = embeddings.chunk_id "
            f"WHERE embeddings.backend_name = ?{filter_sql}",
            tuple(parameters),
        ).fetchall()
        scores = []
        for row in rows:
            if int(row["dimension"]) != len(query_vector):
                continue
            vector = np.frombuffer(row["vector"], dtype=np.float32)
            denominator = float(np.linalg.norm(query_vector) * np.linalg.norm(vector))
            score = float(np.dot(query_vector, vector) / denominator) if denominator else 0.0
            scores.append((str(row["chunk_id"]), score))
        return tuple(sorted(scores, key=lambda item: (-item[1], item[0]))[:limit])

    def get_chunks(self, chunk_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.connection.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", chunk_ids
        ).fetchall()
        return {
            str(row["chunk_id"]): {
                "chunk_id": str(row["chunk_id"]),
                "document_id": str(row["document_id"]),
                "text": str(row["text"]),
                "section_path": tuple(json.loads(row["section_path_json"])),
                "start_offset": int(row["start_offset"]),
                "end_offset": int(row["end_offset"]),
                "token_count": int(row["token_count"]),
                "source_group": str(row["source_group"]),
                "artifact_uri": str(row["artifact_uri"]),
                "file_hash": str(row["file_hash"]),
                "knowledge_type": str(row["knowledge_type"]),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        }

    def record_retrieval(
        self,
        *,
        query_id: str,
        round_id: int,
        original_query_hash: str,
        sanitized_query: str,
        policy: dict[str, Any],
        result_chunk_ids: tuple[str, ...],
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
                    self.manifest_hash,
                ),
            )

    def stats(self) -> dict[str, Any]:
        counts = {}
        for table in ("documents", "chunks", "embeddings", "retrieval_events"):
            counts[table] = int(
                self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
        counts["quarantined_documents"] = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM documents WHERE quarantined = 1"
            ).fetchone()[0]
        )
        counts["manifest_hash"] = self.manifest_hash
        counts["knowledge_types"] = {
            str(row["knowledge_type"]): int(row["count"])
            for row in self.connection.execute(
                "SELECT knowledge_type, COUNT(*) AS count FROM documents "
                "GROUP BY knowledge_type ORDER BY knowledge_type"
            )
        }
        return counts

    def close(self) -> None:
        self.connection.close()
