from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    HypothesisAssessment,
    HypothesisReflection,
    Prediction,
    ValidationRecord,
    Variant,
)


class ObservationKnowledgeGraph:
    """Small SQLite property graph with observations as first-class entities.

    Measured fitness, model predictions, computed evidence, and LLM hypotheses are stored as
    different entity types. Measured fitness is always tied to a complete variant, assay, reveal
    round, and source; predictions and evidence retain their model/source provenance.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        assay_id: str,
        recency_decay: float,
        wild_type_code: str,
        mutable_positions: Sequence[int],
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.assay_id = assay_id
        self.recency_decay = recency_decay
        self.wild_type_code = str(wild_type_code)
        self.mutable_positions = tuple(int(item) for item in mutable_positions)
        if len(self.wild_type_code) != len(self.mutable_positions):
            raise ValueError("wild_type_code and mutable_positions must have equal length")
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
                raw_features_json TEXT NOT NULL DEFAULT '{}',
                quality_status TEXT NOT NULL DEFAULT 'ok',
                applicability TEXT NOT NULL DEFAULT 'unknown',
                uncertainty REAL,
                calibrated_score REAL,
                calibrated INTEGER NOT NULL DEFAULT 0,
                contributes_to_selection INTEGER NOT NULL DEFAULT 1,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                provenance_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id TEXT PRIMARY KEY,
                round_id INTEGER NOT NULL,
                statement TEXT NOT NULL,
                evidence_ids_json TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS hypothesis_assessments (
                assessment_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                falsification_spec_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                status TEXT NOT NULL,
                criterion_results_json TEXT NOT NULL,
                observation_ids_json TEXT NOT NULL,
                decisive_criterion_ids_json TEXT NOT NULL,
                unresolved_criterion_ids_json TEXT NOT NULL,
                evaluator_version TEXT NOT NULL,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id)
            );
            CREATE TABLE IF NOT EXISTS hypothesis_reflections (
                reflection_id TEXT PRIMARY KEY,
                hypothesis_id TEXT NOT NULL,
                assessment_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                assessment_status TEXT NOT NULL,
                summary TEXT NOT NULL,
                retained_claims_json TEXT NOT NULL,
                invalidated_assumptions_json TEXT NOT NULL,
                unresolved_questions_json TEXT NOT NULL,
                recommended_actions_json TEXT NOT NULL,
                supporting_observation_ids_json TEXT NOT NULL,
                supporting_evidence_ids_json TEXT NOT NULL,
                provider TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                advisory_only INTEGER NOT NULL,
                selection_eligible INTEGER NOT NULL,
                dimension_assessments_json TEXT NOT NULL,
                dimension_group_advice_json TEXT NOT NULL,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(hypothesis_id),
                FOREIGN KEY (assessment_id) REFERENCES hypothesis_assessments(assessment_id)
            );
            CREATE TABLE IF NOT EXISTS agent_queries (
                query_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                parameters_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS validation_records (
                record_id TEXT PRIMARY KEY,
                variant_id TEXT NOT NULL,
                round_id INTEGER NOT NULL,
                validation_type TEXT NOT NULL CHECK(validation_type IN ('wet', 'dry')),
                mutation_notation TEXT NOT NULL,
                value REAL NOT NULL,
                uncertainty REAL NOT NULL,
                source_id TEXT NOT NULL,
                model_version TEXT,
                base_weight REAL NOT NULL,
                reliability REAL NOT NULL,
                agent_reason TEXT NOT NULL,
                hypothesis_id TEXT,
                evidence_ids_json TEXT NOT NULL,
                reflection_id TEXT,
                reflection_verdict TEXT,
                reflection_summary TEXT NOT NULL,
                reflection_quality_status TEXT,
                reflection_advisory_only INTEGER NOT NULL DEFAULT 1,
                assessment_id TEXT,
                FOREIGN KEY (variant_id) REFERENCES variants(variant_id)
            );
            CREATE INDEX IF NOT EXISTS idx_observations_round
                ON observations(round_id, variant_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_round
                ON predictions(round_id, fitness_mean DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_round
                ON evidence(round_id, variant_id, channel);
            CREATE INDEX IF NOT EXISTS idx_validation_round
                ON validation_records(round_id, validation_type, variant_id);
            """
        )
        self._migrate_legacy_variants_table()
        self._migrate_legacy_evidence_table()
        self._migrate_legacy_validation_table()
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

    def _migrate_legacy_evidence_table(self) -> None:
        existing = {
            str(row[1]) for row in self.connection.execute("PRAGMA table_info(evidence)").fetchall()
        }
        additions = {
            "raw_features_json": "TEXT NOT NULL DEFAULT '{}'",
            "quality_status": "TEXT NOT NULL DEFAULT 'ok'",
            "applicability": "TEXT NOT NULL DEFAULT 'unknown'",
            "uncertainty": "REAL",
            "calibrated_score": "REAL",
            "calibrated": "INTEGER NOT NULL DEFAULT 0",
            "contributes_to_selection": "INTEGER NOT NULL DEFAULT 1",
            "warnings_json": "TEXT NOT NULL DEFAULT '[]'",
            "provenance_json": "TEXT NOT NULL DEFAULT '{}'",
        }
        for column, definition in additions.items():
            if column not in existing:
                self.connection.execute(f"ALTER TABLE evidence ADD COLUMN {column} {definition}")

    def _migrate_legacy_validation_table(self) -> None:
        existing = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(validation_records)"
            ).fetchall()
        }
        additions = {
            "reflection_id": "TEXT",
            "reflection_verdict": "TEXT",
            "reflection_summary": "TEXT NOT NULL DEFAULT ''",
            "reflection_quality_status": "TEXT",
            "reflection_advisory_only": "INTEGER NOT NULL DEFAULT 1",
            "assessment_id": "TEXT",
        }
        for column, definition in additions.items():
            if column not in existing:
                self.connection.execute(
                    f"ALTER TABLE validation_records ADD COLUMN {column} {definition}"
                )

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
                    zip(self.wild_type_code, variant.variant, strict=True)
                ):
                    if wt == mutant:
                        continue
                    self.connection.execute(
                        """
                        INSERT OR REPLACE INTO mutations
                            (variant_id, position, wt_aa, mutant_aa)
                        VALUES (?, ?, ?, ?)
                        """,
                        (variant.variant_id, self.mutable_positions[index], wt, mutant),
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
                         confidence, round_id, evidence_type, raw_features_json,
                         quality_status, applicability, uncertainty, calibrated_score,
                         calibrated, contributes_to_selection, warnings_json, provenance_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                        json.dumps(item.raw_features, sort_keys=True),
                        item.quality_status,
                        item.applicability,
                        item.uncertainty,
                        item.calibrated_score,
                        int(item.calibrated),
                        int(item.contributes_to_selection),
                        json.dumps(item.warnings),
                        json.dumps(item.provenance, sort_keys=True),
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

    def add_hypothesis_assessment(self, assessment: HypothesisAssessment) -> None:
        criterion_results = [
            {
                "criterion_id": item.criterion_id,
                "signal": item.signal.value,
                "metric_value": item.metric_value,
                "comparator_value": item.comparator_value,
                "effect_size": item.effect_size,
                "observation_ids": list(item.observation_ids),
                "qc_status": item.qc_status,
                "detector_name": item.detector_name,
                "detector_version": item.detector_version,
                "reason_code": item.reason_code,
            }
            for item in assessment.criterion_results
        ]
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO hypothesis_assessments
                    (assessment_id, hypothesis_id, falsification_spec_id, round_id, status,
                     criterion_results_json, observation_ids_json,
                     decisive_criterion_ids_json, unresolved_criterion_ids_json,
                     evaluator_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.assessment_id,
                    assessment.hypothesis_id,
                    assessment.falsification_spec_id,
                    assessment.round_id,
                    assessment.status.value,
                    json.dumps(criterion_results),
                    json.dumps(assessment.observation_ids),
                    json.dumps(assessment.decisive_criterion_ids),
                    json.dumps(assessment.unresolved_criterion_ids),
                    assessment.evaluator_version,
                ),
            )
            self.connection.execute(
                "UPDATE hypotheses SET status = ? WHERE hypothesis_id = ?",
                (assessment.status.value, assessment.hypothesis_id),
            )

    def add_hypothesis_reflection(self, reflection: HypothesisReflection) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO hypothesis_reflections
                    (reflection_id, hypothesis_id, assessment_id, round_id,
                     assessment_status, summary, retained_claims_json,
                     invalidated_assumptions_json, unresolved_questions_json,
                     recommended_actions_json, supporting_observation_ids_json,
                     supporting_evidence_ids_json, provider, quality_status,
                     advisory_only, selection_eligible, dimension_assessments_json,
                     dimension_group_advice_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reflection.reflection_id,
                    reflection.hypothesis_id,
                    reflection.assessment_id,
                    reflection.round_id,
                    reflection.assessment_status,
                    reflection.summary,
                    json.dumps(reflection.retained_claims),
                    json.dumps(reflection.invalidated_assumptions),
                    json.dumps(reflection.unresolved_questions),
                    json.dumps(reflection.recommended_actions),
                    json.dumps(reflection.supporting_observation_ids),
                    json.dumps(reflection.supporting_evidence_ids),
                    reflection.provider,
                    reflection.quality_status,
                    int(reflection.advisory_only),
                    int(reflection.selection_eligible),
                    json.dumps(reflection.dimension_assessments),
                    json.dumps(reflection.dimension_group_advice),
                ),
            )

    def add_validation_records(self, records: Sequence[ValidationRecord]) -> None:
        """Append versioned wet/dry validation records without replacing earlier rounds."""

        known = {
            str(row[0])
            for row in self.connection.execute("SELECT variant_id FROM variants").fetchall()
        }
        missing = {item.variant_id for item in records}.difference(known)
        if missing:
            raise ValueError(f"Validation records reference missing variants: {sorted(missing)}")
        with self.connection:
            for item in records:
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO validation_records
                        (record_id, variant_id, round_id, validation_type,
                         mutation_notation, value, uncertainty, source_id, model_version,
                         base_weight, reliability, agent_reason, hypothesis_id,
                         evidence_ids_json, reflection_id, reflection_verdict,
                         reflection_summary, reflection_quality_status,
                         reflection_advisory_only, assessment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.record_id,
                        item.variant_id,
                        item.round_id,
                        item.validation_type,
                        item.mutation_notation,
                        item.value,
                        item.uncertainty,
                        item.source_id,
                        item.model_version,
                        item.base_weight,
                        item.reliability,
                        item.agent_reason,
                        item.hypothesis_id,
                        json.dumps(item.evidence_ids),
                        item.reflection_id,
                        item.reflection_verdict,
                        item.reflection_summary,
                        item.reflection_quality_status,
                        int(item.reflection_advisory_only),
                        item.assessment_id,
                    ),
                )

    def dry_model_reliability(self, model_version: str, *, floor: float = 0.05) -> float:
        rows = self.connection.execute(
            """
            SELECT dry.value, wet.value
            FROM validation_records dry
            JOIN validation_records wet
              ON dry.variant_id = wet.variant_id AND dry.round_id = wet.round_id
            WHERE dry.validation_type = 'dry' AND wet.validation_type = 'wet'
              AND dry.model_version = ?
            ORDER BY dry.round_id, dry.variant_id
            """,
            (model_version,),
        ).fetchall()
        if len(rows) < 2:
            return 0.5
        dry = [float(row[0]) for row in rows]
        wet = [float(row[1]) for row in rows]
        rmse = float((sum((left - right) ** 2 for left, right in zip(dry, wet)) / len(rows)) ** 0.5)
        scale = max(
            (sum((value - sum(wet) / len(wet)) ** 2 for value in wet) / len(wet)) ** 0.5,
            1e-6,
        )
        return max(floor, min(1.0, 1.0 / (1.0 + rmse / scale)))

    def validation_prior_context(self, *, round_id: int, limit: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT record_id, variant_id, round_id, validation_type, mutation_notation,
                   value, uncertainty, source_id, model_version, base_weight,
                   reliability, agent_reason, hypothesis_id, evidence_ids_json,
                   reflection_id, reflection_verdict, reflection_summary,
                   reflection_quality_status, reflection_advisory_only, assessment_id
            FROM validation_records
            WHERE round_id < ?
            ORDER BY round_id DESC, validation_type, variant_id, record_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        output = []
        for row in rows:
            age = max(0, round_id - int(row[2]) - 1)
            effective_weight = float(row[9]) * float(row[10]) * self.recency_decay**age
            output.append(
                {
                    "record_id": row[0],
                    "variant_id": row[1],
                    "source_round": int(row[2]),
                    "validation_type": row[3],
                    "mutation_notation": row[4],
                    "validation_value": float(row[5]),
                    "uncertainty": float(row[6]),
                    "source_id": row[7],
                    "model_version": row[8],
                    "base_weight": float(row[9]),
                    "reliability": float(row[10]),
                    "effective_weight": effective_weight,
                    "recency_age": age,
                    "agent_reason": row[11],
                    "hypothesis_id": row[12],
                    "evidence_ids": json.loads(row[13]),
                    "reflection_id": row[14],
                    "reflection_verdict": row[15],
                    "reflection_summary": row[16],
                    "reflection_quality_status": row[17],
                    "reflection_advisory_only": bool(row[18]),
                    "assessment_id": row[19],
                    "source_type": (
                        "measurement" if row[3] == "wet" else "model_prediction"
                    ),
                }
            )
        return output

    def validation_prior_statistics(
        self, *, round_id: int
    ) -> dict[tuple[int, str], tuple[float, float, int, int]]:
        rows = self.connection.execute(
            """
            SELECT v.round_id, v.validation_type, v.value, v.base_weight,
                   v.reliability, m.position, m.mutant_aa
            FROM validation_records v JOIN mutations m ON v.variant_id = m.variant_id
            WHERE v.round_id < ?
            ORDER BY v.round_id, v.record_id, m.position
            """,
            (round_id,),
        ).fetchall()
        buckets: dict[tuple[int, str], list[tuple[float, float, str]]] = {}
        for source_round, validation_type, value, base_weight, reliability, position, residue in rows:
            age = max(0, round_id - int(source_round) - 1)
            weight = float(base_weight) * float(reliability) * self.recency_decay**age
            buckets.setdefault((int(position), str(residue)), []).append(
                (float(value), weight, str(validation_type))
            )
        output: dict[tuple[int, str], tuple[float, float, int, int]] = {}
        for key, entries in buckets.items():
            total_weight = sum(weight for _, weight, _ in entries)
            if total_weight <= 0:
                continue
            output[key] = (
                sum(value * weight for value, weight, _ in entries) / total_weight,
                total_weight,
                sum(kind == "wet" for _, _, kind in entries),
                sum(kind == "dry" for _, _, kind in entries),
            )
        return output

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
                SELECT evidence_id, channel, statement, score, source_id, confidence,
                       evidence_type, raw_features_json, quality_status, applicability,
                       uncertainty, calibrated_score, calibrated,
                       contributes_to_selection, warnings_json, provenance_json
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
                            "raw_features": json.loads(ev[7]),
                            "quality_status": ev[8],
                            "applicability": ev[9],
                            "uncertainty": ev[10],
                            "calibrated_score": ev[11],
                            "calibrated": bool(ev[12]),
                            "contributes_to_selection": bool(ev[13]),
                            "warnings": json.loads(ev[14]),
                            "provenance": json.loads(ev[15]),
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
        memory_rows = self.connection.execute(
            """
            SELECT h.hypothesis_id, h.round_id, h.statement, h.status,
                   a.assessment_id, a.status, a.criterion_results_json,
                   a.decisive_criterion_ids_json, a.unresolved_criterion_ids_json,
                   r.reflection_id, r.summary, r.retained_claims_json,
                   r.invalidated_assumptions_json, r.unresolved_questions_json,
                   r.recommended_actions_json, r.supporting_observation_ids_json,
                   r.supporting_evidence_ids_json, r.quality_status,
                   r.advisory_only, r.selection_eligible
            FROM hypotheses h
            JOIN hypothesis_assessments a ON a.hypothesis_id = h.hypothesis_id
            LEFT JOIN hypothesis_reflections r ON r.assessment_id = a.assessment_id
            WHERE h.round_id < ?
            ORDER BY a.round_id DESC, a.assessment_id, r.reflection_id
            LIMIT ?
            """,
            (round_id, limit),
        ).fetchall()
        prior_hypothesis_memory = [
            {
                "hypothesis_id": row[0],
                "source_round": int(row[1]),
                "hypothesis": row[2],
                "hypothesis_status": row[3],
                "assessment_id": row[4],
                "assessment_status": row[5],
                "criterion_results": json.loads(row[6]),
                "decisive_criteria": json.loads(row[7]),
                "unresolved_criteria": json.loads(row[8]),
                "reflection_id": row[9],
                "reflection": row[10] or "",
                "retained_claims": json.loads(row[11]) if row[11] else [],
                "invalidated_assumptions": json.loads(row[12]) if row[12] else [],
                "unresolved_questions": json.loads(row[13]) if row[13] else [],
                "recommended_actions": json.loads(row[14]) if row[14] else [],
                "observation_ids": json.loads(row[15]) if row[15] else [],
                "evidence_ids": json.loads(row[16]) if row[16] else [],
                "quality_status": row[17],
                "advisory_only": bool(row[18]) if row[18] is not None else True,
                "selection_eligible": bool(row[19]) if row[19] is not None else False,
            }
            for row in memory_rows
        ]
        evidence_preview_rows = self.connection.execute(
            """
            SELECT evidence_id, variant_id, channel, statement, score, source_id,
                   confidence, evidence_type, raw_features_json, quality_status,
                   applicability, uncertainty, calibrated_score, calibrated,
                   contributes_to_selection, warnings_json, provenance_json
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
                "raw_features": json.loads(row[8]),
                "quality_status": row[9],
                "applicability": row[10],
                "uncertainty": row[11],
                "calibrated_score": row[12],
                "calibrated": bool(row[13]),
                "contributes_to_selection": bool(row[14]),
                "warnings": json.loads(row[15]),
                "provenance": json.loads(row[16]),
                "source_type": "computed_evidence",
            }
            for row in evidence_preview_rows
        ]
        validation_prior = self.validation_prior_context(round_id=round_id, limit=limit)
        return {
            "visible_observation_count": observation_count,
            "visible_global_mean_fitness": global_mean,
            "beneficial_site_residues": beneficial_sites,
            "top_visible_observations": top_observed,
            "top_knowledge_evidence": top_knowledge_evidence,
            "current_candidate_predictions": candidates,
            "prior_hypotheses": prior_hypotheses,
            "prior_hypothesis_memory": prior_hypothesis_memory,
            "validation_prior": validation_prior,
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
                   confidence, evidence_type, raw_features_json, quality_status,
                   applicability, uncertainty, calibrated_score, calibrated,
                   contributes_to_selection, warnings_json, provenance_json
            FROM evidence WHERE variant_id = ? AND round_id <= ?
            ORDER BY round_id DESC, confidence * ABS(score) DESC, evidence_id
            """,
            (variant_id, round_id),
        ).fetchall()
        validation = self.connection.execute(
            """
            SELECT record_id, round_id, validation_type, value, uncertainty, source_id,
                   model_version, base_weight, reliability, agent_reason,
                   hypothesis_id, evidence_ids_json, reflection_id,
                   reflection_verdict, reflection_summary,
                   reflection_quality_status, reflection_advisory_only, assessment_id
            FROM validation_records WHERE variant_id = ? AND round_id < ?
            ORDER BY round_id DESC, validation_type, record_id
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
                    "raw_features": json.loads(row[8]),
                    "quality_status": row[9],
                    "applicability": row[10],
                    "uncertainty": row[11],
                    "calibrated_score": row[12],
                    "calibrated": bool(row[13]),
                    "contributes_to_selection": bool(row[14]),
                    "warnings": json.loads(row[15]),
                    "provenance": json.loads(row[16]),
                }
                for row in evidence
            ],
            "validation_history": [
                {
                    "record_id": row[0],
                    "round_id": int(row[1]),
                    "validation_type": row[2],
                    "value": float(row[3]),
                    "uncertainty": float(row[4]),
                    "source_id": row[5],
                    "model_version": row[6],
                    "base_weight": float(row[7]),
                    "reliability": float(row[8]),
                    "agent_reason": row[9],
                    "hypothesis_id": row[10],
                    "evidence_ids": json.loads(row[11]),
                    "reflection_id": row[12],
                    "reflection_verdict": row[13],
                    "reflection_summary": row[14],
                    "reflection_quality_status": row[15],
                    "reflection_advisory_only": bool(row[16]),
                    "assessment_id": row[17],
                }
                for row in validation
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
        parameters_json = json.dumps(parameters, ensure_ascii=False, sort_keys=True)
        result_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
        existing = self.connection.execute(
            """
            SELECT query_id FROM agent_queries
            WHERE operation = ? AND round_id = ? AND parameters_json = ? AND result_json = ?
            LIMIT 1
            """,
            (operation, round_id, parameters_json, result_json),
        ).fetchone()
        if existing is not None:
            return str(existing[0])
        ordinal = int(
            self.connection.execute("SELECT COUNT(*) FROM agent_queries").fetchone()[0]
        ) + 1
        query_id = f"KGQ{ordinal:04d}"
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
                    parameters_json,
                    result_json,
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
        validation_rows = self.connection.execute(
            """
            SELECT record_id, variant_id, round_id, validation_type, value,
                   uncertainty, source_id, model_version, base_weight, reliability,
                   hypothesis_id, evidence_ids_json, reflection_id,
                   reflection_verdict, reflection_summary,
                   reflection_quality_status, reflection_advisory_only, assessment_id
            FROM validation_records ORDER BY round_id, validation_type, variant_id
            """
        ).fetchall()
        edges.extend(
            {
                "entity_id": row[0],
                "variant_id": row[1],
                "round_id": row[2],
                "validation_type": row[3],
                "value": row[4],
                "uncertainty": row[5],
                "source_id": row[6],
                "model_version": row[7],
                "base_weight": row[8],
                "reliability": row[9],
                "hypothesis_id": row[10],
                "evidence_ids": json.loads(row[11]),
                "reflection_id": row[12],
                "reflection_verdict": row[13],
                "reflection_summary": row[14],
                "reflection_quality_status": row[15],
                "reflection_advisory_only": bool(row[16]),
                "assessment_id": row[17],
                "predicate": "VALIDATED_BY_WET" if row[3] == "wet" else "VALIDATED_BY_DRY",
            }
            for row in validation_rows
        )
        return edges

    def close(self) -> None:
        self.connection.close()
