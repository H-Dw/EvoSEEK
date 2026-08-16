from __future__ import annotations

import json
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from .schema import KnowledgeGraphSnapshot


class KnowledgeGraphSink(Protocol):
    name: str

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None: ...


class InMemoryGraphSink:
    name = "memory"

    def __init__(self) -> None:
        self.snapshot: KnowledgeGraphSnapshot | None = None

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None:
        self.snapshot = snapshot


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


class SQLiteGraphSink:
    """Persistent external-KG sink; writes versioned schema records by stable ID."""

    name = "sqlite"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                layer TEXT NOT NULL,
                modalities_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                source_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                valid_from_round INTEGER,
                valid_to_round INTEGER
            );
            CREATE TABLE IF NOT EXISTS relations (
                relation_id TEXT PRIMARY KEY,
                subject_id TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object_id TEXT NOT NULL,
                layer TEXT NOT NULL,
                modalities_json TEXT NOT NULL,
                properties_json TEXT NOT NULL,
                source_ids_json TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                source_group TEXT NOT NULL,
                confidence REAL NOT NULL,
                context_id TEXT,
                valid_from_round INTEGER,
                valid_to_round INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_relations_subject
                ON relations(subject_id, predicate);
            CREATE INDEX IF NOT EXISTS idx_relations_object
                ON relations(object_id, predicate);
            """
        )
        self.connection.commit()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None:
        with self.connection:
            for item in snapshot.entities:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.entity_id,
                        item.entity_type,
                        item.layer.value,
                        self._json(item.modalities),
                        self._json(item.properties),
                        self._json(item.source_ids),
                        item.source_group,
                        item.confidence,
                        item.valid_from_round,
                        item.valid_to_round,
                    ),
                )
            for item in snapshot.relations:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO relations VALUES
                        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.relation_id,
                        item.subject_id,
                        item.predicate,
                        item.object_id,
                        item.layer.value,
                        self._json(item.modalities),
                        self._json(item.properties),
                        self._json(item.source_ids),
                        self._json(item.evidence_ids),
                        item.source_group,
                        item.confidence,
                        item.context_id,
                        item.valid_from_round,
                        item.valid_to_round,
                    ),
                )

    def close(self) -> None:
        self.connection.close()
