from __future__ import annotations

import hashlib
import hmac


def effective_salt(public_salt: str | None, seed: int) -> bytes:
    value = public_salt if public_salt is not None else f"fitness-agents-seed:{seed}"
    return value.encode("utf-8")


def stable_digest(salt: bytes, *parts: object) -> str:
    message = "|".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(salt, message, hashlib.sha256).hexdigest()


def salt_commitment(salt: bytes) -> str:
    return hashlib.sha256(b"fitness-agents-split-salt|" + salt).hexdigest()


def stable_order(values: list[str], salt: bytes, *namespace: object) -> list[str]:
    return sorted(values, key=lambda value: (stable_digest(salt, *namespace, value), value))

