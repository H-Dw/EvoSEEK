from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from fitness_agents.contracts.schemas import FitnessObservation, Variant


class ObservationKnowledgeGraph:
    """Small SQLite property graph with observations as first-class entities.

    Fitness is never stored as a context-free mutation property. Each measured value is tied to an
    assay, complete variant, round, and provenance record.
    """

    def __init__(self, path: str | Path, *, assay_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assay_id = assay_id
        self.connection = sqlite3.connect(self.path)
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS variants (
                variant_id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                mutation_notation TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS observations (
                observation_id TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                assay_id TEXT NOT NULL,
                fitness REAL NOT NULL,
                round_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE TABLE IF NOT EXISTS mutations (
                variant_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                wt_aa TEXT NOT NULL,
                mutant_aa TEXT NOT NULL,
                PRIMARY KEY (variant_id, position),
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                round_id INTEGER NOT NULL,
                statement TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def add_observations(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> None:
        variant_map = {variant.variant_id: variant for variant in variants}
        with self.connection:
            for observation in observations:
                variant = variant_map[observation.variant_id]
                self.connection.execute(
                    "INSERT OR IGNORE INTO variants VALUES (?, ?, ?)",
                    (variant.variant_id, variant.variant, variant.mutation_notation),
                )
                for position, (wt, mutant) in enumerate(
                    zip("VDGV", variant.variant, strict=True), start=0
                ):
                    if wt == mutant:
                        continue
                    actual_position = (39, 40, 41, 54)[position]
                    self.connection.execute(
                        "INSERT OR IGNORE INTO mutations VALUES (?, ?, ?, ?)",
                        (variant.variant_id, actual_position, wt, mutant),
                    )
                observation_id = f"obs:{self.assay_id}:{observation.round_revealed}:{variant.variant_id}"
                self.connection.execute(
                    "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        observation_id,
                        variant.variant_id,
                        self.assay_id,
                        observation.fitness,
                        observation.round_revealed,
                        observation.source,
                    ),
                )

    def add_hypothesis(
        self,
        hypothesis_id: str,
        round_id: int,
        statement: str,
        evidence_ids: Sequence[str],
        status: str = "active",
    ) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO hypotheses VALUES (?, ?, ?, ?, ?)",
                (hypothesis_id, round_id, statement, json.dumps(list(evidence_ids)), status),
            )

    def residue_statistics(self) -> dict[tuple[int, str], tuple[float, int]]:
        rows = self.connection.execute(
            """
            SELECT m.position, m.mutant_aa, AVG(o.fitness), COUNT(*)
            FROM mutations m JOIN observations o ON m.variant_id = o.variant_id
            GROUP BY m.position, m.mutant_aa
            """
        ).fetchall()
        return {
            (int(position), str(residue)): (float(mean), int(count))
            for position, residue, mean, count in rows
        }

    def export_edges(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT o.observation_id, o.variant_id, o.assay_id, o.fitness, o.round_id,
                   m.position, m.wt_aa, m.mutant_aa
            FROM observations o LEFT JOIN mutations m ON o.variant_id = m.variant_id
            ORDER BY o.round_id, o.variant_id, m.position
            """
        ).fetchall()
        return [
            {
                "observation_id": row[0],
                "variant_id": row[1],
                "assay_id": row[2],
                "fitness": row[3],
                "round_id": row[4],
                "position": row[5],
                "wt_aa": row[6],
                "mutant_aa": row[7],
                "predicate": "OBSERVED_IN_CONTEXT",
            }
            for row in rows
        ]

    def close(self) -> None:
        self.connection.close()

