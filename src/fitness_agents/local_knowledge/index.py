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
from .prompt_safety import instruction_like_markers
from .protocols import EmbeddingBackend

INDEX_SCHEMA_VERSION = "local-knowledge-index:v5"
FTS_TABLE = "chunks_fts_en_v5"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def preflight_local_knowledge(config: LocalKnowledgeConfig) -> dict[str, Any]:
    """Parse and chunk the full corpus without opening or mutating SQLite."""

    parser = AutoLocalParser(rich_document_backend=config.ingestion.rich_document_backend)
    documents: dict[str, str] = {}
    chunks: dict[str, str] = {}
    for discovered in discover_local_files(
        config.roots, follow_symlinks=config.ingestion.follow_symlinks
    ):
        document = parser.parse(discovered)
        if document.document_id in documents:
            raise ValueError(
                f"Duplicate document_id {document.document_id!r}: "
                f"{documents[document.document_id]!r}, {str(document.path)!r}"
            )
        documents[document.document_id] = str(document.path)
        for chunk in chunk_document(
            document,
            chunk_tokens=config.ingestion.chunk_tokens,
            chunk_overlap=config.ingestion.chunk_overlap,
            source_group=config.kg_update.source_group,
        ):
            if chunk.chunk_id in chunks:
                raise ValueError(
                    f"Duplicate chunk_id {chunk.chunk_id!r}: "
                    f"{chunks[chunk.chunk_id]!r}, {chunk.document_id!r}"
                )
            chunks[chunk.chunk_id] = chunk.document_id
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "document_ids_unique": True,
        "chunk_ids_unique": True,
    }


class SQLiteLocalKnowledgeIndex:
    """Task-independent corpus, lexical index, and dense vectors."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(
                    f"Prebuilt local knowledge corpus does not exist: {self.path}"
                )
            self.connection = sqlite3.connect(
                f"file:{self.path.as_posix()}?mode=ro", uri=True
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        if self.read_only:
            schema_version = self._metadata("schema_version")
            if schema_version != INDEX_SCHEMA_VERSION:
                raise RuntimeError(
                    "Prebuilt local knowledge corpus is incompatible; "
                    f"expected={INDEX_SCHEMA_VERSION!r}, actual={schema_version!r}"
                )
            return
        try:
            self.connection.executescript(
                f"""
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
                    metadata_json TEXT NOT NULL
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
                CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE} USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    artifact_uri UNINDEXED,
                    tokenize='porter unicode61'
                );
                """
            )
        except sqlite3.OperationalError as error:
            raise RuntimeError("SQLite FTS5 is required for local knowledge retrieval") from error
        self._ensure_column(
            "documents", "knowledge_type", "TEXT NOT NULL DEFAULT 'unclassified'"
        )
        self._ensure_column(
            "chunks", "knowledge_type", "TEXT NOT NULL DEFAULT 'unclassified'"
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
        return self._metadata("manifest_hash") or "unbuilt"

    def _build_fingerprint(
        self,
        config: LocalKnowledgeConfig,
        parser: AutoLocalParser,
        embedding_backend: EmbeddingBackend | None,
        *,
        preserved_embedding_fingerprint: dict[str, Any] | None = None,
    ) -> str:
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "parser": parser.name,
            "chunk_tokens": config.ingestion.chunk_tokens,
            "chunk_overlap": config.ingestion.chunk_overlap,
            "required_language": config.ingestion.required_language,
            "instruction_content_policy": config.retrieval.instruction_content_policy,
            "embedding": (
                getattr(embedding_backend, "fingerprint", None)
                if embedding_backend is not None
                else preserved_embedding_fingerprint
            ),
        }
        return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()

    def _embedding_index_is_complete(self, backend: EmbeddingBackend | None) -> bool:
        if backend is None:
            return True
        chunks = int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        rows = int(
            self.connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE backend_name = ? AND dimension = ?",
                (backend.name, backend.dimension),
            ).fetchone()[0]
        )
        return chunks == rows

    def build(
        self,
        config: LocalKnowledgeConfig,
        *,
        guard: TargetLeakageGuard | None = None,
        embedding_backend: EmbeddingBackend | None = None,
    ) -> IndexBuildReport:
        if self.read_only:
            raise RuntimeError("Cannot build a read-only prebuilt local knowledge corpus")
        del guard  # Task policy belongs in SQLiteRetrievalOverlay, never in the corpus index.
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
        current_paths = {str(item.path) for item in files}
        removed = sorted(set(existing).difference(current_paths))
        current_manifest_raw = self._metadata("manifest")
        current_manifest = (
            json.loads(current_manifest_raw) if current_manifest_raw is not None else {}
        )
        preserved_embedding = (
            current_manifest.get("embedding")
            if embedding_backend is None and current_manifest.get("embedding")
            else None
        )
        build_fingerprint = self._build_fingerprint(
            config,
            parser,
            embedding_backend,
            preserved_embedding_fingerprint=preserved_embedding,
        )
        if preserved_embedding is not None:
            changed_paths = [
                str(item.path)
                for item in files
                if str(item.path) not in existing
                or hashlib.sha256(item.path.read_bytes()).hexdigest()
                != existing[str(item.path)][1]
            ]
            if removed or changed_paths or self._metadata("build_fingerprint") != build_fingerprint:
                raise RuntimeError(
                    "Refusing to modify a dense corpus without its embedding backend; "
                    "rebuild with the pinned model so vectors remain complete"
                )
        rebuild_all = (
            self._metadata("build_fingerprint") != build_fingerprint
            or not self._embedding_index_is_complete(embedding_backend)
        )
        indexed_documents = 0
        indexed_chunks = 0
        unchanged_documents = 0
        warnings: list[str] = []
        manifest_entries: list[dict[str, Any]] = []

        prepared: list[tuple[Any, Any, tuple[DocumentChunk, ...]]] = []
        documents_by_id: dict[str, tuple[str, str]] = {}
        chunks_by_id: dict[str, tuple[str, str]] = {}
        for discovered in files:
            path = discovered.path
            if path.stat().st_size > config.ingestion.max_file_mb * 1024 * 1024:
                raise ValueError(f"Local knowledge file exceeds max_file_mb: {path}")
            if not parser.supports(path):
                warnings.append(f"unsupported_file:{path}")
                continue
            document = parser.parse(discovered)
            required_language = config.ingestion.required_language
            if required_language is not None:
                actual_language = str(document.metadata.get("language", "")).casefold()
                if actual_language.split("-", 1)[0] != required_language:
                    raise ValueError(
                        f"Local knowledge file must declare language={required_language}: {path}"
                    )
            markers = instruction_like_markers(document.text)
            if markers and config.retrieval.instruction_content_policy == "reject":
                raise ValueError(
                    f"Instruction-like content rejected in local knowledge file {path}: {markers}"
                )
            prior_document = documents_by_id.get(document.document_id)
            document_signature = (str(document.path), document.file_hash)
            if prior_document is not None:
                raise ValueError(
                    "Duplicate local knowledge document_id detected before index mutation: "
                    f"{document.document_id!r}; first={prior_document[0]!r}, "
                    f"second={str(document.path)!r}"
                )
            documents_by_id[document.document_id] = document_signature
            chunks = chunk_document(
                document,
                chunk_tokens=config.ingestion.chunk_tokens,
                chunk_overlap=config.ingestion.chunk_overlap,
                source_group=config.kg_update.source_group,
                token_counter=(
                    (lambda text: embedding_backend.count_tokens(text, query=False))
                    if embedding_backend is not None
                    else None
                ),
                max_input_tokens=(
                    embedding_backend.max_input_tokens
                    if embedding_backend is not None
                    else None
                ),
            )
            for chunk in chunks:
                prior_chunk = chunks_by_id.get(chunk.chunk_id)
                if prior_chunk is not None:
                    raise ValueError(
                        "Duplicate local knowledge chunk_id detected before index mutation: "
                        f"{chunk.chunk_id!r}; first_document={prior_chunk[0]!r}, "
                        f"second_document={chunk.document_id!r}"
                    )
                chunks_by_id[chunk.chunk_id] = (chunk.document_id, chunk.file_hash)
            prepared.append((discovered, document, chunks))
            manifest_entries.append(
                {
                    "root_id": discovered.root_id,
                    "relative_path": discovered.relative_path,
                    "path": str(path),
                    "file_hash": document.file_hash,
                    "document_id": document.document_id,
                    "knowledge_type": document.knowledge_type,
                    "record_type": document.metadata.get("record_type", "document"),
                }
            )

        with self.connection:
            for path_text in removed:
                self._delete_document(existing[path_text][0])

            for discovered, document, chunks in prepared:
                path = discovered.path
                previous = existing.get(str(path))
                if (
                    not rebuild_all
                    and previous == (document.document_id, document.file_hash)
                ):
                    unchanged_documents += 1
                    continue
                if previous is not None:
                    self._delete_document(previous[0])
                self.connection.execute(
                    "INSERT INTO documents("
                    "document_id, path, file_hash, mime_type, title, knowledge_type, "
                    "metadata_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.document_id,
                        str(document.path),
                        document.file_hash,
                        document.mime_type,
                        document.title,
                        document.knowledge_type,
                        _json(document.metadata),
                    ),
                )
                indexed_documents += 1
                for chunk in chunks:
                    self._insert_chunk(chunk)
                indexed_chunks += len(chunks)
                if embedding_backend is not None and chunks:
                    vectors = embedding_backend.encode_documents([item.text for item in chunks])
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

            embedding_count = int(
                self.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            )
            chunk_count = int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if embedding_backend is not None and embedding_count != chunk_count:
                raise RuntimeError(
                    f"Dense index is incomplete: {embedding_count} embeddings for {chunk_count} chunks"
                )
            manifest_payload = {
                "schema_version": INDEX_SCHEMA_VERSION,
                "chunker_version": CHUNKER_VERSION,
                "parser": parser.name,
                "build_fingerprint": build_fingerprint,
                "embedding": (
                    getattr(embedding_backend, "fingerprint", None)
                    if embedding_backend is not None
                    else preserved_embedding
                ),
                "actual_chunk_count": chunk_count,
                "actual_embedding_count": embedding_count,
                "documents": sorted(manifest_entries, key=lambda item: item["path"]),
            }
            manifest_hash = hashlib.sha256(
                _json(manifest_payload).encode("utf-8")
            ).hexdigest()
            for key, value in (
                ("manifest_hash", manifest_hash),
                ("manifest", _json(manifest_payload)),
                ("build_fingerprint", build_fingerprint),
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
            quarantined_documents=0,
            warnings=tuple(warnings),
        )

    def prebuilt_report(self) -> IndexBuildReport:
        """Return a compatibility/count receipt without mutating the corpus."""

        if self._metadata("schema_version") != INDEX_SCHEMA_VERSION:
            raise RuntimeError("Local knowledge corpus schema is incompatible")
        documents = int(
            self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        )
        chunks = int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if not self.manifest_hash or self.manifest_hash == "unbuilt":
            raise RuntimeError("Prebuilt local knowledge corpus has no manifest")
        return IndexBuildReport(
            manifest_hash=self.manifest_hash,
            indexed_documents=0,
            indexed_chunks=0,
            unchanged_documents=documents,
            removed_documents=0,
            quarantined_documents=0,
            warnings=(f"read_only_prebuilt:{chunks}_chunks",),
        )

    def _delete_document(self, document_id: str) -> None:
        chunk_ids = [
            str(row[0])
            for row in self.connection.execute(
                "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
            )
        ]
        for chunk_id in chunk_ids:
            self.connection.execute(
                f"DELETE FROM {FTS_TABLE} WHERE chunk_id = ?", (chunk_id,)
            )
        self.connection.execute(
            "DELETE FROM embeddings WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE document_id = ?)",
            (document_id,),
        )
        self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self.connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

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
            f"INSERT INTO {FTS_TABLE}(chunk_id, text, artifact_uri) VALUES (?, ?, ?)",
            (chunk.chunk_id, chunk.text, chunk.artifact_uri),
        )

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9_]+", query.casefold())
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
            f"SELECT {FTS_TABLE}.chunk_id, bm25({FTS_TABLE}) AS rank FROM {FTS_TABLE} "
            f"JOIN chunks ON chunks.chunk_id = {FTS_TABLE}.chunk_id "
            f"WHERE {FTS_TABLE} MATCH ?{filter_sql} ORDER BY rank LIMIT ?",
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
        minimum_similarity: float = -1.0,
        max_exact_chunks: int = 50000,
    ) -> tuple[tuple[str, float], ...]:
        query_vector = np.asarray(
            embedding_backend.encode_queries([query])[0], dtype=np.float32
        )
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
        if len(rows) > max_exact_chunks:
            raise RuntimeError(
                f"numpy_exact dense search received {len(rows)} chunks, exceeding "
                f"max_exact_dense_chunks={max_exact_chunks}; use an ANN/vector backend"
            )
        valid_rows = [row for row in rows if int(row["dimension"]) == len(query_vector)]
        if not valid_rows:
            return ()
        matrix = np.vstack(
            [np.frombuffer(row["vector"], dtype=np.float32) for row in valid_rows]
        )
        query_norm = float(np.linalg.norm(query_vector))
        row_norms = np.linalg.norm(matrix, axis=1)
        denominators = row_norms * query_norm
        scores = np.divide(
            matrix @ query_vector,
            denominators,
            out=np.zeros(len(valid_rows), dtype=np.float32),
            where=denominators != 0,
        )
        ranked = [
            (str(row["chunk_id"]), float(score))
            for row, score in zip(valid_rows, scores, strict=True)
            if float(score) >= minimum_similarity
        ]
        return tuple(sorted(ranked, key=lambda item: (-item[1], item[0]))[:limit])

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

    def document_policy_inputs(self) -> tuple[dict[str, str], ...]:
        rows = self.connection.execute(
            "SELECT documents.document_id, documents.path, "
            "COALESCE(GROUP_CONCAT(chunks.text, '\n'), '') AS text "
            "FROM documents LEFT JOIN chunks ON chunks.document_id = documents.document_id "
            "GROUP BY documents.document_id, documents.path ORDER BY documents.path"
        ).fetchall()
        return tuple(
            {
                "document_id": str(row["document_id"]),
                "path": str(row["path"]),
                "text": str(row["text"]),
            }
            for row in rows
        )

    def stats(self) -> dict[str, Any]:
        counts = {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("documents", "chunks", "embeddings")
        }
        counts["manifest_hash"] = self.manifest_hash
        raw_manifest = self._metadata("manifest")
        manifest = json.loads(raw_manifest) if raw_manifest is not None else {}
        counts["schema_version"] = manifest.get("schema_version", "unbuilt")
        counts["chunker_version"] = manifest.get("chunker_version")
        counts["embedding_fingerprint"] = manifest.get("embedding")
        counts["manifest_counts_match"] = bool(
            manifest
            and int(manifest.get("actual_chunk_count", -1)) == counts["chunks"]
            and int(manifest.get("actual_embedding_count", -1))
            == counts["embeddings"]
        )
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
