from __future__ import annotations

import hashlib
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
            CREATE TABLE IF NOT EXISTS graph_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                round_id INTEGER,
                entity_count INTEGER NOT NULL,
                relation_count INTEGER NOT NULL,
                snapshot_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshot_entity_versions (
                snapshot_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, entity_id)
            );
            CREATE TABLE IF NOT EXISTS snapshot_relation_versions (
                snapshot_id TEXT NOT NULL,
                relation_id TEXT NOT NULL,
                record_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, relation_id)
            );
            """
        )
        self.connection.commit()
        self.last_snapshot_id: str | None = None

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)

    def write(self, snapshot: KnowledgeGraphSnapshot) -> None:
        with self.connection:
            for item in snapshot.entities:
                self.connection.execute(
                    """
                    INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        entity_type = excluded.entity_type,
                        layer = excluded.layer,
                        modalities_json = excluded.modalities_json,
                        properties_json = excluded.properties_json,
                        source_ids_json = excluded.source_ids_json,
                        source_group = excluded.source_group,
                        confidence = excluded.confidence,
                        valid_from_round = CASE
                            WHEN entities.valid_from_round IS NULL
                                THEN excluded.valid_from_round
                            WHEN excluded.valid_from_round IS NULL
                                THEN entities.valid_from_round
                            ELSE MIN(entities.valid_from_round, excluded.valid_from_round)
                        END,
                        valid_to_round = excluded.valid_to_round
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
            entity_payloads = {
                item.entity_id: self._json(item.__dict__) for item in snapshot.entities
            }
            relation_payloads = {
                item.relation_id: self._json(item.__dict__) for item in snapshot.relations
            }
            snapshot_payload = self._json(
                {
                    "entities": entity_payloads,
                    "relations": relation_payloads,
                }
            )
            snapshot_hash = hashlib.sha256(snapshot_payload.encode("utf-8")).hexdigest()
            snapshot_id = f"kgsnapshot:{snapshot_hash[:24]}"
            rounds = [
                item.valid_from_round
                for item in (*snapshot.entities, *snapshot.relations)
                if item.valid_from_round is not None
            ]
            round_id = max(rounds) if rounds else None
            self.connection.execute(
                "INSERT OR IGNORE INTO graph_snapshots VALUES (?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    round_id,
                    len(snapshot.entities),
                    len(snapshot.relations),
                    snapshot_hash,
                ),
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO snapshot_entity_versions VALUES (?, ?, ?)",
                (
                    (snapshot_id, entity_id, payload)
                    for entity_id, payload in entity_payloads.items()
                ),
            )
            self.connection.executemany(
                "INSERT OR IGNORE INTO snapshot_relation_versions VALUES (?, ?, ?)",
                (
                    (snapshot_id, relation_id, payload)
                    for relation_id, payload in relation_payloads.items()
                ),
            )
            self.last_snapshot_id = snapshot_id

    @staticmethod
    def _decode_entity(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "entity_id": str(row["entity_id"]),
            "entity_type": str(row["entity_type"]),
            "layer": str(row["layer"]),
            "modalities": json.loads(row["modalities_json"]),
            "properties": json.loads(row["properties_json"]),
            "source_ids": json.loads(row["source_ids_json"]),
            "source_group": str(row["source_group"]),
            "confidence": float(row["confidence"]),
            "valid_from_round": row["valid_from_round"],
            "valid_to_round": row["valid_to_round"],
        }

    @staticmethod
    def _decode_relation(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "relation_id": str(row["relation_id"]),
            "subject_id": str(row["subject_id"]),
            "predicate": str(row["predicate"]),
            "object_id": str(row["object_id"]),
            "layer": str(row["layer"]),
            "modalities": json.loads(row["modalities_json"]),
            "properties": json.loads(row["properties_json"]),
            "source_ids": json.loads(row["source_ids_json"]),
            "evidence_ids": json.loads(row["evidence_ids_json"]),
            "source_group": str(row["source_group"]),
            "confidence": float(row["confidence"]),
            "context_id": row["context_id"],
            "valid_from_round": row["valid_from_round"],
            "valid_to_round": row["valid_to_round"],
        }

    def query_entities(
        self,
        *,
        round_id: int,
        entity_type: str | None = None,
        source_group: str | None = None,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        if round_id < 0 or limit < 1:
            raise ValueError("round_id must be non-negative and limit must be positive")
        conditions = [
            "(valid_from_round IS NULL OR valid_from_round <= ?)",
            "(valid_to_round IS NULL OR valid_to_round >= ?)",
        ]
        parameters: list[Any] = [round_id, round_id]
        if entity_type is not None:
            conditions.append("entity_type = ?")
            parameters.append(entity_type)
        if source_group is not None:
            conditions.append("source_group = ?")
            parameters.append(source_group)
        parameters.append(limit)
        self.connection.row_factory = sqlite3.Row
        rows = self.connection.execute(
            "SELECT * FROM entities WHERE " + " AND ".join(conditions)
            + " ORDER BY confidence DESC, entity_id LIMIT ?",
            parameters,
        ).fetchall()
        return tuple(self._decode_entity(row) for row in rows)

    def list_snapshots(self, *, limit: int = 20) -> tuple[dict[str, Any], ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        self.connection.row_factory = sqlite3.Row
        rows = self.connection.execute(
            "SELECT * FROM graph_snapshots ORDER BY round_id DESC, snapshot_id LIMIT ?",
            (limit,),
        ).fetchall()
        return tuple(
            {
                "snapshot_id": str(row["snapshot_id"]),
                "round_id": row["round_id"],
                "entity_count": int(row["entity_count"]),
                "relation_count": int(row["relation_count"]),
                "snapshot_hash": str(row["snapshot_hash"]),
            }
            for row in rows
        )

    def query_relations(
        self,
        *,
        round_id: int,
        subject_id: str | None = None,
        predicate: str | None = None,
        object_id: str | None = None,
        limit: int = 20,
    ) -> tuple[dict[str, Any], ...]:
        if round_id < 0 or limit < 1:
            raise ValueError("round_id must be non-negative and limit must be positive")
        conditions = [
            "(valid_from_round IS NULL OR valid_from_round <= ?)",
            "(valid_to_round IS NULL OR valid_to_round >= ?)",
        ]
        parameters: list[Any] = [round_id, round_id]
        for column, value in (
            ("subject_id", subject_id),
            ("predicate", predicate),
            ("object_id", object_id),
        ):
            if value is not None:
                conditions.append(f"{column} = ?")
                parameters.append(value)
        parameters.append(limit)
        self.connection.row_factory = sqlite3.Row
        rows = self.connection.execute(
            "SELECT * FROM relations WHERE " + " AND ".join(conditions)
            + " ORDER BY confidence DESC, relation_id LIMIT ?",
            parameters,
        ).fetchall()
        return tuple(self._decode_relation(row) for row in rows)

    def query_claims(
        self, *, query: str, round_id: int, limit: int = 12
    ) -> tuple[dict[str, Any], ...]:
        terms = [item.casefold() for item in query.split() if item.strip()]
        claims = self.query_entities(round_id=round_id, entity_type="Claim", limit=max(limit * 8, 40))
        if not terms:
            selected = claims[:limit]
            return tuple(self._enrich_claim(item, round_id=round_id) for item in selected)
        ranked = []
        for claim in claims:
            properties = claim.get("properties", {})
            text = " ".join(
                str(properties.get(key, ""))
                for key in ("statement", "subject", "predicate", "object")
            ).casefold()
            overlap = sum(term in text for term in terms)
            if overlap:
                ranked.append((overlap, claim))
        ranked.sort(key=lambda item: (-item[0], -float(item[1]["confidence"]), item[1]["entity_id"]))
        return tuple(
            self._enrich_claim(item[1], round_id=round_id) for item in ranked[:limit]
        )

    def _enrich_claim(self, claim: dict[str, Any], *, round_id: int) -> dict[str, Any]:
        relations = self.query_relations(
            round_id=round_id,
            subject_id=str(claim["entity_id"]),
            predicate="SUPPORTED_BY_SOURCE",
            limit=32,
        )
        return {
            **claim,
            "evidence_ids": sorted(
                {
                    evidence_id
                    for relation in relations
                    for evidence_id in relation.get("evidence_ids", ())
                }
            ),
            "supporting_relation_ids": [
                relation["relation_id"] for relation in relations
            ],
        }

    def close(self) -> None:
        self.connection.close()
