from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from fitness_agents.contracts.schemas import Evidence, Variant

from .contracts import RetrievalResult

SELECTION_SCHEMA_VERSION = "local-rag-selection-calibration:v1"


class CandidateEvidenceProjector:
    """Project retrieved claims into calibrated, candidate-specific evidence.

    Retrieval relevance is not a fitness effect. This class is therefore the only
    supported bridge from local RAG context to selection evidence, and it requires
    a versioned calibration file with ``status: validated``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        payload = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise TypeError("Local RAG selection calibration must be a mapping")
        if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
            raise ValueError("Unsupported local RAG selection calibration schema")
        if payload.get("status") != "validated":
            raise ValueError(
                "Local RAG selection calibration must have status=validated"
            )
        calibration_id = str(payload.get("calibration_id", "")).strip()
        if not calibration_id:
            raise ValueError("Local RAG selection calibration requires calibration_id")
        validation = payload.get("validation")
        if not isinstance(validation, dict):
            raise TypeError("Validated selection calibration requires validation metadata")
        for key in ("protocol_id", "dataset_manifest_hash", "metrics"):
            if validation.get(key) in (None, "", {}):
                raise ValueError(
                    f"Validated selection calibration is missing validation.{key}"
                )
        raw_rules = payload.get("rules", [])
        if not isinstance(raw_rules, list) or not raw_rules:
            raise ValueError("Local RAG selection calibration requires at least one rule")
        self.calibration_id = calibration_id
        self.version = str(payload.get("version", "")).strip()
        self.validation = {str(key): value for key, value in validation.items()}
        self.rules = tuple(self._validate_rule(item) for item in raw_rules)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @staticmethod
    def _validate_rule(raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise TypeError("Selection calibration rules must be mappings")
        rule = {str(key): value for key, value in raw.items()}
        for key in ("rule_id", "claim_id", "feature", "operator", "value"):
            if key not in rule or rule[key] in (None, ""):
                raise ValueError(f"Selection calibration rule is missing {key}")
        if rule["feature"] != "mutation_count":
            raise ValueError("Only mutation_count selection calibration is supported")
        if rule["operator"] not in {
            "equal",
            "less_than_or_equal",
            "greater_than_or_equal",
        }:
            raise ValueError("Selection calibration rule operator is unsupported")
        score = float(rule.get("score", 0.0))
        confidence = float(rule.get("confidence", 0.0))
        if not -1.0 <= score <= 1.0:
            raise ValueError("Selection calibration score must be in [-1, 1]")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Selection calibration confidence must be in [0, 1]")
        rule["value"] = int(rule["value"])
        rule["score"] = score
        rule["confidence"] = confidence
        return rule

    @staticmethod
    def _matches(variant: Variant, rule: dict[str, Any]) -> bool:
        actual = variant.mutation_count
        expected = int(rule["value"])
        operator = str(rule["operator"])
        if operator == "equal":
            return actual == expected
        if operator == "less_than_or_equal":
            return actual <= expected
        return actual >= expected

    def project(
        self,
        result: RetrievalResult,
        variants: Sequence[Variant],
    ) -> tuple[Evidence, ...]:
        retrieved_claims = {
            claim.claim_id: claim
            for claim in result.claims
            if claim.selection_eligible
            and claim.citation_support
            and all(
                item.get("verified_against_source") is True
                for item in claim.citation_support
            )
        }
        output: list[Evidence] = []
        for rule in self.rules:
            claim_id = str(rule["claim_id"])
            claim = retrieved_claims.get(claim_id)
            if claim is None:
                continue
            for variant in variants:
                if not self._matches(variant, rule):
                    continue
                evidence_digest = hashlib.sha256(
                    (
                        f"{result.query_id}|{claim_id}|{rule['rule_id']}|"
                        f"{variant.variant_id}"
                    ).encode()
                ).hexdigest()[:16]
                evidence_id = (
                    f"E{result.round_id}:local-projection:{evidence_digest}"
                )
                score = float(rule["score"])
                output.append(
                    Evidence(
                        evidence_id=evidence_id,
                        variant_id=variant.variant_id,
                        channel="local_rag_projection",
                        statement=claim.statement,
                        score=score,
                        source_id=f"calibration:{self.calibration_id}",
                        confidence=float(rule["confidence"]),
                        round_id=result.round_id,
                        evidence_type="calibrated_candidate_projection",
                        raw_features={
                            "mutation_count": variant.mutation_count,
                            "rule_id": rule["rule_id"],
                            "rule_operator": rule["operator"],
                            "rule_value": rule["value"],
                            "claim_scientific_confidence": claim.confidence,
                        },
                        quality_status="ok",
                        applicability="candidate_specific_calibrated_projection",
                        calibrated_score=score,
                        calibrated=True,
                        contributes_to_selection=True,
                        warnings=(),
                        provenance={
                            "calibration_id": self.calibration_id,
                            "calibration_version": self.version,
                            "calibration_sha256": self.sha256,
                            "calibration_path": str(self.path),
                            "retrieval_query_id": result.query_id,
                            "index_manifest_hash": result.index_manifest_hash,
                            "claim_id": claim_id,
                            "validation": self.validation,
                        },
                        claim_id=claim_id,
                        polarity=("support" if score >= 0 else "contradict"),
                        source_group="local_rag_calibration",
                        valid_from_round=result.round_id,
                    )
                )
        return tuple(output)
