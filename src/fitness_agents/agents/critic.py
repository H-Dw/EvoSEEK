from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any

from fitness_agents.contracts.schemas import (
    BatchRisk,
    CandidateIssue,
    ConflictReport,
    CritiqueDecision,
    DraftBatch,
    Evidence,
    EvidenceConflict,
    FalsificationReadiness,
    IssueScope,
    IssueSeverity,
    Prediction,
    RequiredChange,
    RequiredChangeAction,
    ReviewVerdict,
    UnsupportedClaim,
    Variant,
)
from fitness_agents.validation.batch import CritiqueDecisionValidator


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


CRITIQUE_DECISION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "decision_id", "draft_batch_id", "round_id", "review_attempt", "verdict",
        "falsification_readiness", "candidate_issues", "batch_level_risks",
        "evidence_conflicts", "unsupported_claims", "required_changes",
        "cited_evidence_ids", "confidence", "summary",
    ],
    "properties": {
        "decision_id": {"type": "string"},
        "draft_batch_id": {"type": "string"},
        "round_id": {"type": "integer"},
        "review_attempt": {"type": "integer"},
        "verdict": {"type": "string", "enum": [item.value for item in ReviewVerdict]},
        "falsification_readiness": {
            "type": "string", "enum": [item.value for item in FalsificationReadiness]
        },
        "candidate_issues": {"type": "array", "items": {"$ref": "#/$defs/candidate_issue"}},
        "batch_level_risks": {"type": "array", "items": {"$ref": "#/$defs/batch_risk"}},
        "evidence_conflicts": {"type": "array", "items": {"$ref": "#/$defs/evidence_conflict"}},
        "unsupported_claims": {"type": "array", "items": {"$ref": "#/$defs/unsupported_claim"}},
        "required_changes": {"type": "array", "items": {"$ref": "#/$defs/required_change"}},
        "cited_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
    "$defs": {
        "candidate_issue": {
            "type": "object", "additionalProperties": False,
            "required": ["issue_id", "candidate_id", "scope", "severity", "code", "claim", "evidence_ids", "conflict_ids", "suggested_action"],
            "properties": {
                "issue_id": {"type": "string"}, "candidate_id": {"type": "string"},
                "scope": {"type": "string", "enum": [item.value for item in IssueScope]},
                "severity": {"type": "string", "enum": [item.value for item in IssueSeverity]},
                "code": {"type": "string"}, "claim": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "conflict_ids": {"type": "array", "items": {"type": "string"}},
                "suggested_action": {"type": ["string", "null"], "enum": [item.value for item in RequiredChangeAction] + [None]},
            },
        },
        "batch_risk": {
            "type": "object", "additionalProperties": False,
            "required": ["risk_id", "code", "severity", "statement", "candidate_ids", "evidence_ids"],
            "properties": {
                "risk_id": {"type": "string"}, "code": {"type": "string"},
                "severity": {"type": "string", "enum": [item.value for item in IssueSeverity]},
                "statement": {"type": "string"},
                "candidate_ids": {"type": "array", "items": {"type": "string"}},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        "evidence_conflict": {
            "type": "object", "additionalProperties": False,
            "required": ["conflict_id", "topic", "supporting_ids", "opposing_ids", "source_independence", "unresolved_reason", "impact"],
            "properties": {
                "conflict_id": {"type": "string"}, "topic": {"type": "string"},
                "supporting_ids": {"type": "array", "items": {"type": "string"}},
                "opposing_ids": {"type": "array", "items": {"type": "string"}},
                "source_independence": {"type": "string"},
                "unresolved_reason": {"type": "string"}, "impact": {"type": "string"},
            },
        },
        "unsupported_claim": {
            "type": "object", "additionalProperties": False,
            "required": ["claim_id", "claim", "reason", "missing_evidence_type", "required_action"],
            "properties": {
                "claim_id": {"type": "string"}, "claim": {"type": "string"},
                "reason": {"type": "string"}, "missing_evidence_type": {"type": "string"},
                "required_action": {"type": "string", "enum": [item.value for item in RequiredChangeAction]},
            },
        },
        "required_change": {
            "type": "object", "additionalProperties": False,
            "required": ["action", "target_ids", "parameters", "rationale", "evidence_ids", "priority"],
            "properties": {
                "action": {"type": "string", "enum": [item.value for item in RequiredChangeAction]},
                "target_ids": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
                "rationale": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer", "minimum": 0},
            },
        },
    },
}


def load_critic_profile(profile: str) -> str:
    root = Path(__file__).with_name("critic_profiles") / profile
    skill = root / "SKILL.md"
    if not skill.is_file():
        raise FileNotFoundError(f"Unknown critic profile {profile!r}")
    return skill.read_text(encoding="utf-8")


class RuleBasedCriticClient:
    """Deterministic critic used offline and as a fail-closed remote fallback."""

    provider_name = "rule"

    def review(self, *, context: dict[str, Any], output_schema: dict[str, Any]) -> CritiqueDecision:
        draft: DraftBatch = context["draft"]
        report: ConflictReport = context["conflict_report"]
        evidence: Mapping[str, Sequence[Evidence]] = context["evidence"]
        candidate_issues = tuple(
            CandidateIssue(
                issue_id=f"issue:{item.conflict_id}",
                candidate_id=item.candidate_ids[0],
                scope=item.scope,
                severity=item.severity,
                code=item.code,
                claim=item.message,
                evidence_ids=item.evidence_ids,
                conflict_ids=(item.conflict_id,),
                suggested_action=(
                    RequiredChangeAction.EXCLUDE_CANDIDATE if item.candidate_ids else RequiredChangeAction.ABORT_ROUND
                ),
            )
            for item in report.hard_conflicts
            if item.candidate_ids
        )
        risks = tuple(
            BatchRisk(
                risk_id=f"risk:{item.conflict_id}",
                code=item.code,
                severity=item.severity,
                statement=item.message,
                candidate_ids=item.candidate_ids,
                evidence_ids=item.evidence_ids,
            )
            for item in report.conflicts
            if not item.hard
        )
        evidence_conflicts: list[EvidenceConflict] = []
        for item in report.conflicts:
            if item.code != "EVIDENCE_POLARITY_CONFLICT" or not item.candidate_ids:
                continue
            bundle = evidence.get(item.candidate_ids[0], ())
            evidence_conflicts.append(
                EvidenceConflict(
                    conflict_id=item.conflict_id,
                    topic=f"Evidence polarity for {item.candidate_ids[0]}",
                    supporting_ids=tuple(entry.evidence_id for entry in bundle if entry.score > 0),
                    opposing_ids=tuple(entry.evidence_id for entry in bundle if entry.score < 0),
                    source_independence="not_established",
                    unresolved_reason="Deterministic evidence channels disagree.",
                    impact="Treat the candidate as uncertain; do not present one-sided support.",
                )
            )
        if report.hard_conflicts:
            verdict = ReviewVerdict.REJECT
            readiness = FalsificationReadiness.UNTESTABLE
            changes = (
                RequiredChange(
                    action=RequiredChangeAction.ABORT_ROUND,
                    target_ids=(),
                    parameters={},
                    rationale="Unresolved deterministic hard conflicts block submission.",
                ),
            )
            confidence = 1.0
            summary = "Rejected because deterministic hard conflicts remain unresolved."
        elif draft.hypothesis_ids and draft.falsification_spec is None:
            verdict = ReviewVerdict.REVISE
            readiness = FalsificationReadiness.NEEDS_REVISION
            changes = (
                RequiredChange(
                    action=RequiredChangeAction.MAKE_FALSIFICATION_EXECUTABLE,
                    target_ids=(),
                    parameters={},
                    rationale="The hypothesis lacks a frozen executable falsification specification.",
                ),
            )
            confidence = 0.99
            summary = "Revision is required to preregister an executable falsification test."
        else:
            verdict = ReviewVerdict.APPROVE
            readiness = FalsificationReadiness.READY
            changes = ()
            confidence = 0.95
            summary = "Approved after hard validation and structured scientific-risk review."
        cited = tuple(
            dict.fromkeys(
                evidence_id
                for candidate_id in draft.candidate_ids
                for evidence_id in (entry.evidence_id for entry in evidence.get(candidate_id, ()))
            )
        )
        return CritiqueDecision(
            decision_id=f"critique:{uuid.uuid4().hex}",
            draft_batch_id=draft.draft_batch_id,
            round_id=draft.round_id,
            review_attempt=draft.review_attempt,
            verdict=verdict,
            falsification_readiness=readiness,
            candidate_issues=candidate_issues,
            batch_level_risks=risks,
            evidence_conflicts=tuple(evidence_conflicts),
            unsupported_claims=(),
            required_changes=changes,
            cited_evidence_ids=cited,
            confidence=confidence,
            summary=summary,
        )


class OpenAICriticClient:
    provider_name = "openai_critic"

    def __init__(self, *, model: str | None, profile: str, temperature: float = 0.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install requirements/llm.txt to use a remote Critic") from error
        api_key = os.environ.get("FITNESS_AGENTS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set FITNESS_AGENTS_LLM_API_KEY or OPENAI_API_KEY")
        self.model = model or os.environ.get("FITNESS_AGENTS_CRITIC_MODEL", "gpt-5-mini")
        self.temperature = temperature
        self.profile = load_critic_profile(profile)
        self.client = OpenAI(api_key=api_key, base_url=os.environ.get("OPENAI_BASE_URL"))

    def review(self, *, context: dict[str, Any], output_schema: dict[str, Any]) -> CritiqueDecision:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.profile},
                {"role": "user", "content": json.dumps(_jsonable(context), ensure_ascii=False)},
            ],
            text={
                "format": {
                    "type": "json_schema", "name": "critique_decision",
                    "strict": True, "schema": output_schema,
                }
            },
        )
        return _decision_from_payload(json.loads(response.output_text))


def _decision_from_payload(payload: dict[str, Any]) -> CritiqueDecision:
    return CritiqueDecision(
        decision_id=str(payload["decision_id"]),
        draft_batch_id=str(payload["draft_batch_id"]),
        round_id=int(payload["round_id"]),
        review_attempt=int(payload["review_attempt"]),
        verdict=ReviewVerdict(payload["verdict"]),
        falsification_readiness=FalsificationReadiness(payload["falsification_readiness"]),
        candidate_issues=tuple(
            CandidateIssue(
                issue_id=item["issue_id"], candidate_id=item["candidate_id"],
                scope=IssueScope(item["scope"]), severity=IssueSeverity(item["severity"]),
                code=item["code"], claim=item["claim"], evidence_ids=tuple(item["evidence_ids"]),
                conflict_ids=tuple(item["conflict_ids"]),
                suggested_action=(RequiredChangeAction(item["suggested_action"]) if item["suggested_action"] else None),
            )
            for item in payload["candidate_issues"]
        ),
        batch_level_risks=tuple(
            BatchRisk(
                risk_id=item["risk_id"], code=item["code"], severity=IssueSeverity(item["severity"]),
                statement=item["statement"], candidate_ids=tuple(item["candidate_ids"]),
                evidence_ids=tuple(item["evidence_ids"]),
            )
            for item in payload["batch_level_risks"]
        ),
        evidence_conflicts=tuple(EvidenceConflict(**{**item, "supporting_ids": tuple(item["supporting_ids"]), "opposing_ids": tuple(item["opposing_ids"])}) for item in payload["evidence_conflicts"]),
        unsupported_claims=tuple(UnsupportedClaim(**{**item, "required_action": RequiredChangeAction(item["required_action"])}) for item in payload["unsupported_claims"]),
        required_changes=tuple(
            RequiredChange(
                action=RequiredChangeAction(item["action"]), target_ids=tuple(item["target_ids"]),
                parameters=dict(item["parameters"]), rationale=item["rationale"],
                evidence_ids=tuple(item["evidence_ids"]), priority=int(item["priority"]),
            )
            for item in payload["required_changes"]
        ),
        cited_evidence_ids=tuple(payload["cited_evidence_ids"]),
        confidence=float(payload["confidence"]), summary=str(payload["summary"]),
    )


class CriticAgent:
    """Independent reviewer with frozen inputs and no experiment-backend capability."""

    def __init__(self, client: Any, *, max_retries: int = 2, fallback: Any | None = None) -> None:
        self.client = client
        self.max_retries = max_retries
        self.fallback = fallback
        self.validator = CritiqueDecisionValidator()

    def review(
        self,
        *,
        draft: DraftBatch,
        variants: Mapping[str, Variant],
        predictions: Mapping[str, Prediction],
        evidence: Mapping[str, Sequence[Evidence]],
        conflict_report: ConflictReport,
    ) -> CritiqueDecision:
        context = {
            "draft": draft,
            "variants": {item: variants[item] for item in draft.candidate_ids},
            "predictions": {item: predictions[item] for item in draft.candidate_ids},
            "evidence": {item: tuple(evidence.get(item, ())) for item in draft.candidate_ids},
            "conflict_report": conflict_report,
        }
        visible_ids = {
            entry.evidence_id
            for candidate_id in draft.candidate_ids
            for entry in evidence.get(candidate_id, ())
        }
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                decision = self.client.review(context=context, output_schema=CRITIQUE_DECISION_SCHEMA)
                self.validator.validate(
                    decision, draft=draft, report=conflict_report,
                    visible_evidence_ids=visible_ids,
                )
                return decision
            except Exception as error:  # noqa: BLE001 - remote/schema failures must fail closed
                last_error = error
        if self.fallback is not None:
            decision = self.fallback.review(context=context, output_schema=CRITIQUE_DECISION_SCHEMA)
            self.validator.validate(
                decision, draft=draft, report=conflict_report,
                visible_evidence_ids=visible_ids,
            )
            return decision
        raise RuntimeError("Critic failed without a configured safe fallback") from last_error
