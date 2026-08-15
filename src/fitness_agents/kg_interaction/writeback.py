from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import ClassVar, Protocol

from .contracts import ChangeOperation, KGChangeProposal, KGUpdateResult


class ChangeProposalValidator(Protocol):
    def validate(self, proposal: KGChangeProposal) -> tuple[str, ...]: ...


class ChangeWriter(Protocol):
    def commit(self, proposal: KGChangeProposal) -> KGUpdateResult: ...


class TrustBoundaryValidator:
    """Enforce that agent writes remain claims, links, or hypothesis updates."""

    _evidence_required: ClassVar[frozenset[ChangeOperation]] = frozenset(
        {
            ChangeOperation.ADD_CURATED_CLAIM,
            ChangeOperation.CHANGE_HYPOTHESIS_STATUS,
            ChangeOperation.LINK_EVIDENCE,
        }
    )

    def validate(self, proposal: KGChangeProposal) -> tuple[str, ...]:
        errors: list[str] = []
        if not proposal.proposal_id.strip():
            errors.append("proposal_id is required")
        if not proposal.idempotency_key.strip():
            errors.append("idempotency_key is required")
        if not 0.0 <= proposal.confidence <= 1.0:
            errors.append("confidence must be in [0, 1]")
        if proposal.operation in self._evidence_required and not proposal.evidence_ids:
            errors.append(f"{proposal.operation.value} requires evidence_ids")
        if proposal.operation is ChangeOperation.CHANGE_HYPOTHESIS_STATUS:
            allowed = {
                "proposed",
                "testing",
                "supported",
                "contradicted",
                "inconclusive",
                "superseded",
            }
            if str(proposal.payload.get("status")) not in allowed:
                errors.append("invalid hypothesis status")
        return tuple(errors)


class ProposalGateway:
    """Validate first, then dry-run or transactionally delegate a KG change."""

    def __init__(
        self,
        writer: ChangeWriter,
        *,
        validators: Iterable[ChangeProposalValidator] = (),
        read_only: bool = True,
        allowed_operations: frozenset[ChangeOperation] | None = None,
    ) -> None:
        self.writer = writer
        self.validators = tuple(validators) or (TrustBoundaryValidator(),)
        self.read_only = read_only
        self.allowed_operations = allowed_operations

    def submit(self, proposal: KGChangeProposal) -> KGUpdateResult:
        errors: list[str] = []
        if (
            self.allowed_operations is not None
            and proposal.operation not in self.allowed_operations
        ):
            errors.append(f"operation {proposal.operation.value!r} is disabled")
        for validator in self.validators:
            errors.extend(validator.validate(proposal))
        if errors:
            return KGUpdateResult(
                proposal_id=proposal.proposal_id,
                status="rejected",
                errors=tuple(errors),
            )
        if self.read_only:
            return KGUpdateResult(
                proposal_id=proposal.proposal_id,
                status="dry_run",
            )
        return self.writer.commit(proposal)


class InMemoryChangeWriter:
    """Test/demonstration sink with idempotency semantics."""

    def __init__(self) -> None:
        self.proposals: dict[str, KGChangeProposal] = {}

    def commit(self, proposal: KGChangeProposal) -> KGUpdateResult:
        existing = self.proposals.get(proposal.idempotency_key)
        if existing is not None:
            return KGUpdateResult(
                proposal_id=proposal.proposal_id,
                status="duplicate",
                transaction_id=self._transaction_id(proposal.idempotency_key),
            )
        self.proposals[proposal.idempotency_key] = proposal
        return KGUpdateResult(
            proposal_id=proposal.proposal_id,
            status="committed",
            transaction_id=self._transaction_id(proposal.idempotency_key),
        )

    @staticmethod
    def _transaction_id(value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
        return f"kgtx:{digest}"
