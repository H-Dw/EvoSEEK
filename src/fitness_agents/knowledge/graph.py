from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Prediction,
    Variant,
)


class ObservationKnowledgeGraph:
    """Small SQLite property graph with observations as first-class entities.

    Measured fitness, model predictions, computed evidence, and LLM hypotheses are stored as
    different entity types. Measured fitness is always tied to a complete variant, assay, reveal
    round, and source; predictions and evidence retain their model/source provenance.
    """

    _WT_CODE = "VDGV"
    _POSITIONS = (39, 40, 41, 54)

    def __init__(self, path: str | Path, *, assay_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assay_id = assay_id
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS variants (
                variant_id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                sequence TEXT NOT NULL DEFAULT '',
                mutation_notation TEXT NOT NULL,
                mutation_count INTEGER NOT NULL DEFAULT 0,
                split_role TEXT NOT NULL DEFAULT ''
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
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                fitness_mean REAL NOT NULL,
                fitness_std REAL NOT NULL,
                interval_low REAL NOT NULL,
                interval_high REAL NOT NULL,
                ood_score REAL NOT NULL,
                model_version TEXT NOT NULL,
                component_scores_json TEXT NOT NULL,
                intervention_tags_json TEXT NOT NULL,
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                channel TEXT NOT NULL,
                statement TEXT NOT NULL,
                score REAL NOT NULL,
                source_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                round_id INTEGER NOT NULL,
                evidence_type TEXT NOT NULL,
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                round_id INTEGER NOT NULL,
                statement TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agent_queries (
                query_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_observations_round
                ON observations(round_id, variant_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_round
                ON predictions(round_id, fitness_mean DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_round
                ON evidence(round_id, variant_id, channel);
            """
        )
        self._migrate_legacy_variants_table()
        self.connection.commit()

    def _migrate_legacy_variants_table(self) -> None:
        """Add columns when opening a graph created by an earlier repository version."""
        existing = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(variants)").fetchall()
        }
        additions = {
            "sequence": "TEXT NOT NULL DEFAULT ''",
            "mutation_count": "INTEGER NOT NULL DEFAULT 0",
            "split_role": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in additions.items():
            if column not in existing:
                self.connection.execute(f"ALTER TABLE variants ADD COLUMN {column} {definition}")

    def add_variants(self, variants: Sequence[Variant]) -> None:
        with self.connection:
            for variant in variants:
                self.connection.execute(
                    """
                    INSERT INTO variants
                        (variant_id, code, sequence, mutation_notation, mutation_count, split_role)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(variant_id) DO UPDATE SET
                        code=excluded.code,
                        sequence=excluded.sequence,
                        mutation_notation=excluded.mutation_notation,
                        mutation_count=excluded.mutation_count,
                        split_role=excluded.split_role
                    """,
                    (
                        variant.variant_id,
                        variant.variant,
                        variant.sequence,
                        variant.mutation_notation,
                        variant.mutation_count,
                        variant.split_role,
                    ),
                )
                for index, (wt, mutant) in enumerate(
                    zip(self._WT_CODE, variant.variant, strict=True)
                ):
                    if wt == mutant:
                        continue
                    self.connection.execute(
                        """
                        INSERT OR REPLACE INTO mutations
                            (variant_id, position, wt_aa, mutant_aa)
                        VALUES (?, ?, ?, ?)
                        """,
                        (variant.variant_id, self._POSITIONS[index], wt, mutant),
                    )

    def add_observations(
        self,
        variants: Sequence[Variant],
        observations: Sequence[FitnessObservation],
    ) -> None:
        variant_map = {variant.variant_id: variant for variant in variants}
        missing = {item.variant_id for item in observations}.difference(variant_map)
        if missing:
            raise ValueError(f"Observations reference missing variants: {sorted(missing)}")
        self.add_variants(variants)
        with self.connection:
            for observation in observations:
                observation_id = (
                    f"obs:{self.assay_id}:{observation.round_revealed}:{observation.variant_id}"
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO observations
                        (observation_id, variant_id, assay_id, fitness, round_id, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation_id,
                        observation.variant_id,
                        self.assay_id,
                        observation.fitness,
                        observation.round_revealed,
                        observation.source,
                    ),
                )

    def add_predictions(
        self,
        variants: Sequence[Variant],
        predictions: Sequence[Prediction],
        *,
        round_id: int,
        intervention_tags: Sequence[str] = (),
    ) -> None:
        variant_map = {variant.variant_id: variant for variant in variants}
        missing = {item.variant_id for item in predictions}.difference(variant_map)
        if missing:
            raise ValueError(f"Predictions reference missing variants: {sorted(missing)}")
        self.add_variants(variants)
        tags_json = json.dumps(sorted(set(intervention_tags)))
        with self.connection:
            for prediction in predictions:
                prediction_id = (
                    f"pred:{round_id}:{prediction.model_version}:{prediction.variant_id}"
                )
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO predictions
                        (prediction_id, variant_id, round_id, fitness_mean, fitness_std,
                         interval_low, interval_high, ood_score, model_version,
                         component_scores_json, intervention_tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prediction_id,
                        prediction.variant_id,
                        round_id,
                        prediction.fitness_mean,
                        prediction.fitness_std,
                        prediction.interval_90[0],
                        prediction.interval_90[1],
                        prediction.ood_score,
                        prediction.model_version,
                        json.dumps(prediction.component_scores, sort_keys=True),
                        tags_json,
                    ),
                )

    def add_evidence(self, evidence: Sequence[Evidence]) -> None:
        with self.connection:
            for item in evidence:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO evidence
                        (evidence_id, variant_id, channel, statement, score, source_id,
                         confidence, round_id, evidence_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.evidence_id,
                        item.variant_id,
                        item.channel,
                        item.statement,
                        item.score,
                        item.source_id,
                        item.confidence,
                        item.round_id,
                        item.evidence_type,
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
                """
                INSERT OR REPLACE INTO hypotheses
                    (hypothesis_id, round_id, statement, evidence_ids_json, status)
                VALUES (?, ?, ?, ?, ?)
                """,
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

    def agent_hypothesis_context(self, *, round_id: int, limit: int) -> dict[str, Any]:
        """Return bounded context using only observations revealed before ``round_id``.

        Current-round predictions and computed evidence are allowed, but they remain explicitly
        typed as predictions/evidence and are never presented as measured fitness.
        """
        global_row = self.connection.execute(
            "SELECT AVG(fitness), COUNT(*) FROM observations WHERE round_id < ?",
            (round_id,),
        ).fetchone()
        global_mean = float(global_row[0]) if global_row and global_row[0] is not None else 0.0
        observation_count = int(global_row[1]) if global_row else 0
        site_rows = self.connection.execute(
            """
            SELECT m.position, m.mutant_aa, AVG(o.fitness), COUNT(*)
            FROM mutations m JOIN observations o ON m.variant_id = o.variant_id
            WHERE o.round_id < ?
            GROUP BY m.position, m.mutant_aa
            ORDER BY AVG(o.fitness) DESC, COUNT(*) DESC, m.position, m.mutant_aa
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        beneficial_sites = [
            {
                "position": int(position),
                "residue": str(residue),
                "visible_mean_fitness": float(mean),
                "visible_lift_over_global_mean": float(mean) - global_mean,
                "support": int(count),
                "source_type": "measured_aggregate",
                "caveat": "association only; complete-variant epistasis may confound residue effects",
            }
            for position, residue, mean, count in site_rows
        ]
        observed_rows = self.connection.execute(
            """
            SELECT o.observation_id, o.variant_id, v.code, v.mutation_notation,
                   o.fitness, o.round_id, o.source
            FROM observations o JOIN variants v ON v.variant_id = o.variant_id
            WHERE o.round_id < ?
            ORDER BY o.fitness DESC, o.round_id DESC, o.variant_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        top_observed = [
            {
                "observation_id": row[0],
                "variant_id": row[1],
                "variant": row[2],
                "mutation_notation": row[3],
                "measured_fitness": float(row[4]),
                "round_revealed": int(row[5]),
                "source": row[6],
                "source_type": "measurement",
            }
            for row in observed_rows
        ]
        prediction_rows = self.connection.execute(
            """
            SELECT p.prediction_id, p.variant_id, v.code, v.mutation_notation,
                   p.fitness_mean, p.fitness_std, p.interval_low, p.interval_high,
                   p.ood_score, p.model_version, p.component_scores_json,
                   p.intervention_tags_json
            FROM predictions p JOIN variants v ON v.variant_id = p.variant_id
            WHERE p.round_id = ?
            ORDER BY p.fitness_mean DESC, p.variant_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        candidates: list[dict[str, Any]] = []
        for row in prediction_rows:
            evidence_rows = self.connection.execute(
                """
                SELECT evidence_id, channel, statement, score, source_id, confidence, evidence_type
                FROM evidence
                WHERE variant_id = ? AND round_id = ?
                ORDER BY confidence * ABS(score) DESC, evidence_id
                LIMIT 6
                """,
                (row[1], round_id),
            ).fetchall()
            candidates.append(
                {
                    "prediction_id": row[0],
                    "variant_id": row[1],
                    "variant": row[2],
                    "mutation_notation": row[3],
                    "fitness_mean": float(row[4]),
                    "fitness_std": float(row[5]),
                    "interval_90": [float(row[6]), float(row[7])],
                    "ood_score": float(row[8]),
                    "model_version": row[9],
                    "component_scores": json.loads(row[10]),
                    "intervention_tags": json.loads(row[11]),
                    "source_type": "model_prediction",
                    "evidence": [
                        {
                            "evidence_id": ev[0],
                            "channel": ev[1],
                            "statement": ev[2],
                            "score": float(ev[3]),
                            "source_id": ev[4],
                            "confidence": float(ev[5]),
                            "evidence_type": ev[6],
                        }
                        for ev in evidence_rows
                    ],
                }
            )
        hypothesis_rows = self.connection.execute(
            """
            SELECT hypothesis_id, round_id, statement, evidence_ids_json, status
            FROM hypotheses
            WHERE round_id < ?
            ORDER BY round_id DESC, hypothesis_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        prior_hypotheses = [
            {
                "hypothesis_id": row[0],
                "round_id": int(row[1]),
                "statement": row[2],
                "evidence_ids": json.loads(row[3]),
                "status": row[4],
            }
            for row in hypothesis_rows
        ]
        evidence_preview_rows = self.connection.execute(
            """
            SELECT evidence_id, variant_id, channel, statement, score, source_id,
                   confidence, evidence_type
            FROM evidence
            WHERE round_id = ?
            ORDER BY confidence * ABS(score) DESC, evidence_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        top_knowledge_evidence = [
            {
                "evidence_id": row[0],
                "variant_id": row[1],
                "channel": row[2],
                "statement": row[3],
                "score": float(row[4]),
                "source_id": row[5],
                "confidence": float(row[6]),
                "evidence_type": row[7],
                "source_type": "computed_evidence",
            }
            for row in evidence_preview_rows
        ]
        return {
            "visible_observation_count": observation_count,
            "visible_global_mean_fitness": global_mean,
            "beneficial_site_residues": beneficial_sites,
            "top_visible_observations": top_observed,
            "top_knowledge_evidence": top_knowledge_evidence,
            "current_candidate_predictions": candidates,
            "prior_hypotheses": prior_hypotheses,
        }

    def explain_variant(self, variant_id: str, *, round_id: int) -> dict[str, Any]:
        variant = self.connection.execute(
            """
            SELECT variant_id, code, sequence, mutation_notation, mutation_count, split_role
            FROM variants WHERE variant_id = ?
            """,
            (variant_id,),
        ).fetchone()
        if variant is None:
            return {"variant_id": variant_id, "found": False}
        observations = self.connection.execute(
            """
            SELECT observation_id, fitness, round_id, source
            FROM observations WHERE variant_id = ? AND round_id < ?
            ORDER BY round_id, observation_id
            """,
            (variant_id, round_id),
        ).fetchall()
        predictions = self.connection.execute(
            """
            SELECT prediction_id, round_id, fitness_mean, fitness_std, interval_low,
                   interval_high, ood_score, model_version, intervention_tags_json
            FROM predictions WHERE variant_id = ? AND round_id <= ?
            ORDER BY round_id DESC, prediction_id
            """,
            (variant_id, round_id),
        ).fetchall()
        evidence = self.connection.execute(
            """
            SELECT evidence_id, round_id, channel, statement, score, source_id,
                   confidence, evidence_type
            FROM evidence WHERE variant_id = ? AND round_id <= ?
            ORDER BY round_id DESC, confidence * ABS(score) DESC, evidence_id
            """,
            (variant_id, round_id),
        ).fetchall()
        return {
            "variant_id": variant[0],
            "found": True,
            "variant": variant[1],
            "sequence": variant[2],
            "mutation_notation": variant[3],
            "mutation_count": int(variant[4]),
            "split_role": variant[5],
            "visible_observations": [
                {
                    "observation_id": row[0],
                    "measured_fitness": float(row[1]),
                    "round_revealed": int(row[2]),
                    "source": row[3],
                }
                for row in observations
            ],
            "predictions": [
                {
                    "prediction_id": row[0],
                    "round_id": int(row[1]),
                    "fitness_mean": float(row[2]),
                    "fitness_std": float(row[3]),
                    "interval_90": [float(row[4]), float(row[5])],
                    "ood_score": float(row[6]),
                    "model_version": row[7],
                    "intervention_tags": json.loads(row[8]),
                }
                for row in predictions
            ],
            "evidence": [
                {
                    "evidence_id": row[0],
                    "round_id": int(row[1]),
                    "channel": row[2],
                    "statement": row[3],
                    "score": float(row[4]),
                    "source_id": row[5],
                    "confidence": float(row[6]),
                    "evidence_type": row[7],
                }
                for row in evidence
            ],
        }

    def record_agent_query(
        self,
        operation: str,
        *,
        round_id: int,
        parameters: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        canonical = json.dumps(
            {
                "operation": operation,
                "round_id": round_id,
                "parameters": parameters,
                "result": result,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        query_id = f"kgq:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO agent_queries
                    (query_id, operation, round_id, parameters_json, result_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    operation,
                    round_id,
                    json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                ),
            )
        return query_id

    def export_agent_queries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT query_id, operation, round_id, parameters_json, result_json
            FROM agent_queries ORDER BY round_id, query_id
            """
        ).fetchall()
        return [
            {
                "query_id": row[0],
                "operation": row[1],
                "round_id": int(row[2]),
                "parameters": json.loads(row[3]),
                "result": json.loads(row[4]),
            }
            for row in rows
        ]

    def export_edges(self) -> list[dict[str, object]]:
        observation_rows = self.connection.execute(
            """
            SELECT o.observation_id, o.variant_id, o.assay_id, o.fitness, o.round_id,
                   m.position, m.wt_aa, m.mutant_aa
            FROM observations o LEFT JOIN mutations m ON o.variant_id = m.variant_id
            ORDER BY o.round_id, o.variant_id, m.position
            """
        ).fetchall()
        edges: list[dict[str, object]] = [
            {
                "entity_id": row[0],
                "variant_id": row[1],
                "assay_id": row[2],
                "fitness": row[3],
                "round_id": row[4],
                "position": row[5],
                "wt_aa": row[6],
                "mutant_aa": row[7],
                "predicate": "OBSERVED_IN_CONTEXT",
            }
            for row in observation_rows
        ]
        prediction_rows = self.connection.execute(
            """
            SELECT prediction_id, variant_id, round_id, fitness_mean, fitness_std,
                   ood_score, model_version, intervention_tags_json
            FROM predictions ORDER BY round_id, variant_id
            """
        ).fetchall()
        edges.extend(
            {
                "entity_id": row[0],
                "variant_id": row[1],
                "round_id": row[2],
                "fitness_mean": row[3],
                "fitness_std": row[4],
                "ood_score": row[5],
                "model_version": row[6],
                "intervention_tags": json.loads(row[7]),
                "predicate": "PREDICTED_AS",
            }
            for row in prediction_rows
        )
        evidence_rows = self.connection.execute(
            """
            SELECT evidence_id, variant_id, round_id, channel, score, source_id,
                   confidence, evidence_type
            FROM evidence ORDER BY round_id, variant_id, channel
            """
        ).fetchall()
        edges.extend(
            {
                "entity_id": row[0],
                "variant_id": row[1],
                "round_id": row[2],
                "channel": row[3],
                "score": row[4],
                "source_id": row[5],
                "confidence": row[6],
                "evidence_type": row[7],
                "predicate": "SUPPORTED_BY_EVIDENCE",
            }
            for row in evidence_rows
        )
        return edges

    def close(self) -> None:
        self.connection.close()
