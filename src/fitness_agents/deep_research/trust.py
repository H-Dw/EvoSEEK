"""Explicit environment-backed trust configuration for Deep Research workflows."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass

from .policy import ExternalEvidenceScopePolicy

POLICY_KEY_ID_ENV = "FITNESS_DEEP_RESEARCH_POLICY_KEY_ID"
POLICY_KEY_HEX_ENV = "FITNESS_DEEP_RESEARCH_POLICY_KEY_HEX"
REVIEW_KEYRING_ENV = "FITNESS_DEEP_RESEARCH_REVIEW_KEYRING_JSON"
RELEASE_KEYRING_ENV = "FITNESS_DEEP_RESEARCH_RELEASE_KEYRING_JSON"


SignerKeyring = dict[str, tuple[str, bytes]]


@dataclass(frozen=True)
class ValidationTrust:
    active_policy: ExternalEvidenceScopePolicy
    reviewer_keys: Mapping[str, tuple[str, bytes]]
    release_approval_keys: Mapping[str, tuple[str, bytes]]


def decode_key_hex(value: str, *, label: str) -> bytes:
    try:
        key = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{label} must be hexadecimal key material") from error
    if len(key) < 32:
        raise ValueError(f"{label} must contain at least 32 bytes")
    return key


def load_active_policy_from_environment(
    *,
    signing_required: bool,
) -> ExternalEvidenceScopePolicy:
    key_id = os.environ.get(POLICY_KEY_ID_ENV)
    key_hex = os.environ.get(POLICY_KEY_HEX_ENV)
    if key_id is None and key_hex is None and not signing_required:
        return ExternalEvidenceScopePolicy()
    if not key_id or not key_hex:
        raise ValueError(
            f"Set both {POLICY_KEY_ID_ENV} and {POLICY_KEY_HEX_ENV}"
        )
    return ExternalEvidenceScopePolicy(
        signing_key_id=key_id,
        signing_key=decode_key_hex(key_hex, label=POLICY_KEY_HEX_ENV),
    )


def load_signer_keyring_from_environment(
    environment_name: str,
    *,
    required: bool = False,
) -> SignerKeyring:
    raw = os.environ.get(environment_name)
    if not raw:
        if required:
            raise ValueError(f"Set non-empty {environment_name}")
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{environment_name} must be valid JSON") from error
    if not isinstance(payload, dict) or (required and not payload):
        raise ValueError(f"{environment_name} must be a non-empty JSON object")
    output: SignerKeyring = {}
    for reviewer_id, record in payload.items():
        if (
            not isinstance(reviewer_id, str)
            or not reviewer_id
            or not isinstance(record, dict)
        ):
            raise ValueError(f"{environment_name} contains an invalid signer record")
        if set(record) != {"key_id", "key_hex"}:
            raise ValueError(
                f"{environment_name} signer records must contain exactly "
                "key_id and key_hex"
            )
        key_id = record["key_id"]
        key_hex = record["key_hex"]
        if not isinstance(key_id, str) or not isinstance(key_hex, str):
            raise TypeError(
                f"{environment_name} signer records require key_id and key_hex"
            )
        output[reviewer_id] = (
            key_id,
            decode_key_hex(key_hex, label=f"{environment_name}:{reviewer_id}"),
        )
    return output


def load_validation_trust_from_environment() -> ValidationTrust:
    return ValidationTrust(
        active_policy=load_active_policy_from_environment(signing_required=True),
        reviewer_keys=load_signer_keyring_from_environment(
            REVIEW_KEYRING_ENV,
            required=True,
        ),
        release_approval_keys=load_signer_keyring_from_environment(
            RELEASE_KEYRING_ENV,
            required=True,
        ),
    )
