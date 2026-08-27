from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel


def _normalized(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalized(value.model_dump(mode="json", exclude_none=False))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        return {str(key): _normalized(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalized(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _normalized(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(text: str) -> str:
    normalized = unicodedata.normalize("NFC", " ".join(text.split()))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join(unicodedata.normalize("NFC", part) for part in parts).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}:{digest}"
