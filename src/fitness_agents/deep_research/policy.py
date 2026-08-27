from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import Field

from .attestation import (
    AttestationError,
    TrustedHMACKeyring,
    sign_canonical_payload,
    verify_canonical_payload,
)
from .contracts import (
    PolicyReceipt,
    ScopeAssertion,
    StrictModel,
    SubjectRole,
    SubjectScope,
)

POLICY_VERSION = "external-evidence-scope:v1"


class SourceAccessPermit(StrictModel):
    artifact_id: str
    operation: Literal["metadata_use", "full_text_fetch", "parse", "release"]
    policy_hash: str
    assertion_hash: str
    resource_locator_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    expires_at: datetime
    nonce: str
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScopeDecision(StrictModel):
    decision: Literal["allowed", "denied", "quarantined"]
    matched_categories: tuple[str, ...] = ()
    reason: str
    policy_version: str = POLICY_VERSION
    policy_hash: str

class ExternalEvidenceScopePolicy:
    """Fail-closed subject-scope policy for external evidence.

    Verified structured assertions are authoritative. Metadata markers can deny or
    quarantine discovery results before any full-text fetch, but can never promote an
    unknown result to allowed.
    """

    _RESTRICTED_PATTERNS = (
        re.compile(r"\b(?:virus|viral|virion|bacteriophage|phage)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:capsid|viral\s+(?:polymerase|protease|glycoprotein|protein))\b",
            re.IGNORECASE,
        ),
    )
    _NEGATIVE_QUERY_TERMS = (
        "virus",
        "viral",
        "virion",
        "bacteriophage",
        "phage",
        "capsid",
    )
    _RESTRICTED_ROLES = frozenset(
        {
            SubjectRole.PRIMARY_SUBJECT,
            SubjectRole.EXPERIMENTAL_SYSTEM,
            SubjectRole.OPERATIONAL_METHOD,
        }
    )

    def __init__(
        self,
        *,
        default_unknown_action: Literal["quarantined", "denied"] = "quarantined",
        signing_key_id: str | None = None,
        signing_key: bytes | None = None,
    ):
        if (signing_key_id is None) != (signing_key is None):
            raise ValueError(
                "Policy receipt signing requires both signing_key_id and signing_key"
            )
        self.default_unknown_action = default_unknown_action
        self._signing_key_id = signing_key_id
        self._signing_key = signing_key
        self._permit_secret = secrets.token_bytes(32)
        payload = {
            "policy_version": POLICY_VERSION,
            "default_unknown_action": default_unknown_action,
            "excluded_scope": SubjectScope.VIRAL_PROTEIN.value,
            "restricted_roles": sorted(item.value for item in self._RESTRICTED_ROLES),
            "negative_query_terms": list(self._NEGATIVE_QUERY_TERMS),
            "trusted_assertion_issuer_kinds": [
                "authoritative_taxonomy",
                "curated_registry",
                "human_review",
            ],
            "scope_assertion_requires_explicit_excluded_subject_absence": True,
            "operation_and_resource_bound_permits": True,
        }
        self.policy_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @property
    def signing_key_id(self) -> str | None:
        return self._signing_key_id

    def issue_receipt(
        self,
        decision: ScopeDecision,
        *,
        stage: str,
        subject_id: str,
        subject_sha256: str,
        issued_at: datetime | None = None,
    ) -> PolicyReceipt:
        """Issue a domain-separated receipt; unsigned policy receipts are impossible."""

        if self._signing_key_id is None or self._signing_key is None:
            raise RuntimeError(
                "Policy receipt signing key is not configured; issuance fails closed"
            )
        if decision.policy_hash != self.policy_hash:
            raise ValueError("Cannot issue a receipt for a foreign policy decision")
        payload = {
            "schema_version": "external-policy-receipt:v2",
            "policy_version": decision.policy_version,
            "policy_hash": decision.policy_hash,
            "decision": decision.decision,
            "matched_categories": decision.matched_categories,
            "stage": stage,
            "subject_id": subject_id,
            "subject_sha256": subject_sha256,
            "issued_at": issued_at or datetime.now(timezone.utc),
            "issuer": "external_evidence_scope_policy",
        }
        attestation = sign_canonical_payload(
            payload,
            key_id=self._signing_key_id,
            key=self._signing_key,
            purpose="policy-receipt:v1",
        )
        return PolicyReceipt(**payload, attestation=attestation)

    def verify_receipt(self, receipt: PolicyReceipt) -> None:
        """Verify a receipt against this exact active policy and trust anchor."""

        if self._signing_key_id is None or self._signing_key is None:
            raise RuntimeError(
                "Policy receipt verification key is not configured; verification fails closed"
            )
        if (
            receipt.policy_hash != self.policy_hash
            or receipt.policy_version != POLICY_VERSION
        ):
            raise AttestationError("Policy receipt is bound to a different policy")
        payload = receipt.model_dump(mode="python", exclude={"attestation"})
        verify_canonical_payload(
            payload,
            receipt.attestation,
            trusted_keyring=TrustedHMACKeyring(
                {self._signing_key_id: self._signing_key}
            ),
            expected_purpose="policy-receipt:v1",
        )

    @property
    def negative_query_suffix(self) -> str:
        return " ".join(f"-{term}" for term in self._NEGATIVE_QUERY_TERMS)

    def inspect_query(self, query: str) -> ScopeDecision:
        inspected_query = re.sub(
            r"(?<!\w)-(?:virus|viral|virion|bacteriophage|phage|capsid)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        categories = self._marker_categories(inspected_query)
        if categories:
            return ScopeDecision(
                decision="denied",
                matched_categories=categories,
                reason="research_question_enters_excluded_subject_scope",
                policy_hash=self.policy_hash,
            )
        return ScopeDecision(
            decision="allowed",
            reason="query_within_nonviral_research_scope",
            policy_hash=self.policy_hash,
        )

    def decide_metadata(
        self,
        *,
        artifact_id: str,
        title: str,
        abstract: str | None,
        subjects: tuple[str, ...],
        assertion: ScopeAssertion | None,
        as_of: datetime | None = None,
    ) -> ScopeDecision:
        if assertion is not None:
            if assertion.artifact_id != artifact_id:
                return ScopeDecision(
                    decision="denied",
                    matched_categories=("scope_assertion_identity_mismatch",),
                    reason="scope assertion does not bind to this artifact",
                    policy_hash=self.policy_hash,
                )
            if assertion.assertion_status != "verified":
                return ScopeDecision(
                    decision=self.default_unknown_action,
                    matched_categories=("scope_assertion_not_verified",),
                    reason="only verified scope assertions can authorize source use",
                    policy_hash=self.policy_hash,
                )
            now = as_of or datetime.now(timezone.utc)
            if assertion.expires_at is not None and assertion.expires_at <= now:
                return ScopeDecision(
                    decision=self.default_unknown_action,
                    matched_categories=("scope_assertion_expired",),
                    reason="scope assertion has expired",
                    policy_hash=self.policy_hash,
                )
            restricted_role = bool(self._RESTRICTED_ROLES.intersection(assertion.roles))
            if assertion.subject_scope == SubjectScope.VIRAL_PROTEIN and restricted_role:
                return ScopeDecision(
                    decision="denied",
                    matched_categories=("verified_viral_protein_scope",),
                    reason="verified assertion identifies an excluded source scope",
                    policy_hash=self.policy_hash,
                )
            if assertion.subject_scope in {
                SubjectScope.MIXED,
                SubjectScope.UNKNOWN,
            }:
                return ScopeDecision(
                    decision=self.default_unknown_action,
                    matched_categories=("mixed_or_unknown_scope",),
                    reason="mixed or unknown source scope requires quarantine",
                    policy_hash=self.policy_hash,
                )
            if assertion.subject_scope in {
                SubjectScope.GENERIC_PROTEIN,
                SubjectScope.NONVIRAL_PROTEIN,
            }:
                return ScopeDecision(
                    decision="allowed",
                    reason="verified generic or nonviral protein scope",
                    policy_hash=self.policy_hash,
                )

        metadata = " ".join((title, abstract or "", *subjects))
        categories = self._marker_categories(metadata)
        if categories:
            return ScopeDecision(
                decision="denied",
                matched_categories=categories,
                reason="metadata indicates excluded source scope before full-text access",
                policy_hash=self.policy_hash,
            )
        return ScopeDecision(
            decision=self.default_unknown_action,
            matched_categories=("missing_verified_scope_assertion",),
            reason="unknown sources remain quarantined until independently scoped",
            policy_hash=self.policy_hash,
        )

    def issue_permit(
        self,
        *,
        artifact_id: str,
        operation: Literal["metadata_use", "full_text_fetch", "parse", "release"],
        assertion: ScopeAssertion,
        resource_locator: str,
        ttl_minutes: int = 30,
    ) -> SourceAccessPermit:
        if not resource_locator.strip():
            raise ValueError("External evidence permits require a resource locator")
        if not 1 <= ttl_minutes <= 120:
            raise ValueError("External evidence permit TTL must be between 1 and 120 minutes")
        decision = self.decide_metadata(
            artifact_id=artifact_id,
            title="",
            abstract=None,
            subjects=(),
            assertion=assertion,
        )
        if decision.decision != "allowed":
            raise PermissionError(f"External evidence policy denied permit: {decision.reason}")
        issued = datetime.now(timezone.utc)
        expires = issued + timedelta(minutes=ttl_minutes)
        if assertion.expires_at is not None:
            expires = min(expires, assertion.expires_at)
        locator_hash = hashlib.sha256(resource_locator.encode("utf-8")).hexdigest()
        nonce = secrets.token_hex(12)
        signature = self._permit_signature(
            artifact_id=artifact_id,
            operation=operation,
            assertion_hash=assertion.assertion_hash,
            resource_locator_sha256=locator_hash,
            issued_at=issued,
            expires_at=expires,
            nonce=nonce,
        )
        return SourceAccessPermit(
            artifact_id=artifact_id,
            operation=operation,
            policy_hash=self.policy_hash,
            assertion_hash=assertion.assertion_hash,
            resource_locator_sha256=locator_hash,
            issued_at=issued,
            expires_at=expires,
            nonce=nonce,
            signature=signature,
        )

    def verify_permit(
        self,
        permit: SourceAccessPermit,
        *,
        artifact_id: str,
        operation: str,
        assertion: ScopeAssertion,
        resource_locator: str,
    ) -> None:
        now = datetime.now(timezone.utc)
        locator_hash = hashlib.sha256(resource_locator.encode("utf-8")).hexdigest()
        expected_signature = self._permit_signature(
            artifact_id=permit.artifact_id,
            operation=permit.operation,
            assertion_hash=permit.assertion_hash,
            resource_locator_sha256=permit.resource_locator_sha256,
            issued_at=permit.issued_at,
            expires_at=permit.expires_at,
            nonce=permit.nonce,
        )
        current_decision = self.decide_metadata(
            artifact_id=artifact_id,
            title="",
            abstract=None,
            subjects=(),
            assertion=assertion,
        )
        if (
            permit.policy_hash != self.policy_hash
            or permit.artifact_id != artifact_id
            or permit.operation != operation
            or permit.assertion_hash != assertion.assertion_hash
            or permit.resource_locator_sha256 != locator_hash
            or permit.expires_at <= now
            or current_decision.decision != "allowed"
            or not hmac.compare_digest(permit.signature, expected_signature)
        ):
            raise PermissionError("External evidence access permit is invalid or expired")

    def _permit_signature(
        self,
        *,
        artifact_id: str,
        operation: str,
        assertion_hash: str,
        resource_locator_sha256: str,
        issued_at: datetime,
        expires_at: datetime,
        nonce: str,
    ) -> str:
        payload = (
            f"{artifact_id}|{operation}|{self.policy_hash}|{assertion_hash}|"
            f"{resource_locator_sha256}|{issued_at.isoformat()}|"
            f"{expires_at.isoformat()}|{nonce}"
        ).encode()
        return hmac.new(self._permit_secret, payload, hashlib.sha256).hexdigest()

    def _marker_categories(self, text: str) -> tuple[str, ...]:
        return tuple(
            f"excluded_scope_marker_{index}"
            for index, pattern in enumerate(self._RESTRICTED_PATTERNS, start=1)
            if pattern.search(text)
        )
