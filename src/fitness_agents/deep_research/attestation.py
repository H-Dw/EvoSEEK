"""Offline, explicitly keyed attestations for canonical Deep Research payloads.

This module deliberately has no key generation, environment-variable lookup, file
loading, or default trust anchor.  Signing callers must provide one key explicitly;
verifying callers must provide a non-empty trusted keyring explicitly.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .canonical import canonical_json, content_sha256

ATTESTATION_SCHEMA_VERSION = "canonical-hmac-attestation:v1"
HMAC_ALGORITHM = "hmac-sha256"
MINIMUM_KEY_BYTES = 32

# Key rotation is explicit: a key identifier without a ``:vN`` suffix is invalid.
_VERSIONED_IDENTIFIER = re.compile(
    r"^[a-z][a-z0-9._-]{0,62}:v[1-9][0-9]{0,8}$"
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class AttestationError(ValueError):
    """Base class for attestation configuration and verification failures."""


class AttestationConfigurationError(AttestationError):
    """The caller did not provide a safe, explicit signing/trust configuration."""


class AttestationVerificationError(AttestationError):
    """An attestation is malformed, untrusted, or does not match its payload."""


class UnknownKeyIdError(AttestationVerificationError):
    """The attestation names a key that is absent from the trusted keyring."""


class HMACAttestation(BaseModel):
    """Portable signature envelope for one canonical payload digest.

    ``purpose`` is mandatory domain separation.  Callers should use a versioned
    value specific to the protocol, for example ``release-approval:v1``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["canonical-hmac-attestation:v1"] = (
        ATTESTATION_SCHEMA_VERSION
    )
    algorithm: Literal["hmac-sha256"] = HMAC_ALGORITHM
    key_id: str = Field(
        min_length=4,
        max_length=80,
        pattern=_VERSIONED_IDENTIFIER.pattern,
    )
    purpose: str = Field(
        min_length=4,
        max_length=80,
        pattern=_VERSIONED_IDENTIFIER.pattern,
    )
    payload_sha256: str = Field(pattern=_SHA256_HEX.pattern)
    signature: str = Field(pattern=_SHA256_HEX.pattern)


class TrustedHMACKeyring:
    """Immutable, caller-provisioned HMAC trust anchors keyed by versioned ID."""

    __slots__ = ("_keys",)

    def __init__(self, keys: Mapping[str, bytes]) -> None:
        if not isinstance(keys, Mapping) or not keys:
            raise AttestationConfigurationError(
                "A non-empty trusted HMAC keyring must be supplied explicitly"
            )
        copied: dict[str, bytes] = {}
        for key_id, key in keys.items():
            _require_versioned_identifier(key_id, label="key_id")
            copied[key_id] = _require_key_bytes(key, key_id=key_id)
        self._keys = MappingProxyType(copied)

    @property
    def key_ids(self) -> tuple[str, ...]:
        """Return trusted identifiers without exposing key material."""

        return tuple(sorted(self._keys))

    def _resolve(self, key_id: str) -> bytes:
        """Resolve one exact key ID; never fall back to another key or version."""

        key = self._keys.get(key_id)
        if key is None:
            raise UnknownKeyIdError(f"Untrusted HMAC key_id: {key_id!r}")
        return key

    def __repr__(self) -> str:
        return f"TrustedHMACKeyring(key_ids={self.key_ids!r})"


def sign_canonical_payload(
    payload: Any,
    *,
    key_id: str,
    key: bytes,
    purpose: str,
) -> HMACAttestation:
    """Sign a canonical payload with an explicitly supplied HMAC-SHA256 key."""

    _require_versioned_identifier(key_id, label="key_id")
    _require_versioned_identifier(purpose, label="purpose")
    key_bytes = _require_key_bytes(key, key_id=key_id)
    payload_sha256 = _canonical_payload_sha256(payload)
    signature = hmac.new(
        key_bytes,
        _signing_message(
            key_id=key_id,
            purpose=purpose,
            payload_sha256=payload_sha256,
        ),
        hashlib.sha256,
    ).hexdigest()
    return HMACAttestation(
        key_id=key_id,
        purpose=purpose,
        payload_sha256=payload_sha256,
        signature=signature,
    )


def verify_canonical_payload(
    payload: Any,
    attestation: HMACAttestation | Mapping[str, Any],
    *,
    trusted_keyring: TrustedHMACKeyring | Mapping[str, bytes],
    expected_purpose: str,
) -> HMACAttestation:
    """Verify an attestation or raise; this API never returns a permissive boolean.

    The verifier requires both an exact expected purpose and an explicit trust
    keyring.  Unknown key IDs, unsupported versions, malformed signatures, payload
    changes, and empty keyrings all fail closed.
    """

    _require_versioned_identifier(expected_purpose, label="expected_purpose")
    envelope = _parse_attestation(attestation)
    if envelope.purpose != expected_purpose:
        raise AttestationVerificationError(
            "Attestation purpose does not match the expected protocol purpose"
        )
    keyring = (
        trusted_keyring
        if isinstance(trusted_keyring, TrustedHMACKeyring)
        else TrustedHMACKeyring(trusted_keyring)
    )
    key = keyring._resolve(envelope.key_id)
    actual_payload_sha256 = _canonical_payload_sha256(payload)
    if not hmac.compare_digest(envelope.payload_sha256, actual_payload_sha256):
        raise AttestationVerificationError(
            "Attestation payload digest does not match the canonical payload"
        )
    expected_signature = hmac.new(
        key,
        _signing_message(
            key_id=envelope.key_id,
            purpose=envelope.purpose,
            payload_sha256=envelope.payload_sha256,
        ),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(envelope.signature, expected_signature):
        raise AttestationVerificationError("Attestation signature is invalid")
    return envelope


def _parse_attestation(
    value: HMACAttestation | Mapping[str, Any],
) -> HMACAttestation:
    if isinstance(value, HMACAttestation):
        return value
    if not isinstance(value, Mapping):
        raise AttestationVerificationError("Attestation must be a mapping or model")
    required_fields = {
        "schema_version",
        "algorithm",
        "key_id",
        "purpose",
        "payload_sha256",
        "signature",
    }
    if set(value) != required_fields:
        raise AttestationVerificationError(
            "Attestation envelope must contain the complete versioned schema"
        )
    try:
        return HMACAttestation.model_validate(dict(value))
    except ValidationError as error:
        raise AttestationVerificationError(
            "Attestation envelope is malformed or uses an unsupported version"
        ) from error


def _canonical_payload_sha256(payload: Any) -> str:
    try:
        encoded = canonical_json(payload)
        # The shared serializer permits Python's NaN/Infinity spellings by default;
        # reject them here because they are not canonical JSON across runtimes.
        json.loads(
            encoded,
            parse_constant=lambda value: _reject_nonfinite_number(value),
        )
        digest = content_sha256(payload)
    except (TypeError, ValueError, OverflowError) as error:
        raise AttestationConfigurationError(
            "Payload cannot be represented as finite canonical JSON"
        ) from error
    if not _SHA256_HEX.fullmatch(digest):
        raise AttestationConfigurationError("Canonical payload digest is invalid")
    if not hmac.compare_digest(
        digest,
        hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    ):
        raise AttestationConfigurationError(
            "Payload changed while it was being canonicalized"
        )
    return digest


def _reject_nonfinite_number(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is not permitted: {value}")


def _signing_message(*, key_id: str, purpose: str, payload_sha256: str) -> bytes:
    signed_core = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "algorithm": HMAC_ALGORITHM,
        "key_id": key_id,
        "purpose": purpose,
        "payload_sha256": payload_sha256,
    }
    return canonical_json(signed_core).encode("utf-8")


def _require_versioned_identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not _VERSIONED_IDENTIFIER.fullmatch(value):
        raise AttestationConfigurationError(
            f"{label} must be a lowercase versioned identifier ending in ':vN'"
        )
    return value


def _require_key_bytes(key: bytes, *, key_id: str) -> bytes:
    if type(key) is not bytes:
        raise AttestationConfigurationError(
            f"HMAC key {key_id!r} must be supplied as bytes"
        )
    if len(key) < MINIMUM_KEY_BYTES:
        raise AttestationConfigurationError(
            f"HMAC key {key_id!r} must contain at least {MINIMUM_KEY_BYTES} bytes"
        )
    return key


__all__ = [
    "ATTESTATION_SCHEMA_VERSION",
    "HMAC_ALGORITHM",
    "MINIMUM_KEY_BYTES",
    "AttestationConfigurationError",
    "AttestationError",
    "AttestationVerificationError",
    "HMACAttestation",
    "TrustedHMACKeyring",
    "UnknownKeyIdError",
    "sign_canonical_payload",
    "verify_canonical_payload",
]
