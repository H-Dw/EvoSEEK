"""Versioned local role profiles; profiles constrain cognition, never domain authority."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RoleProfile:
    role: str
    name: str
    instructions: str
    metadata: dict[str, object]
    sha256: str


def load_role_profile(role: str, name: str) -> RoleProfile:
    root = Path(__file__).with_name("profiles") / role / name
    skill_path = root / "SKILL.md"
    metadata_path = root / "profile.yaml"
    if not skill_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Unknown {role} profile {name!r}")
    instructions = skill_path.read_text(encoding="utf-8")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if metadata.get("role") != role or metadata.get("name") != name:
        raise ValueError(f"Profile metadata does not match {role}/{name}")
    digest = hashlib.sha256(
        (instructions + "\n" + metadata_path.read_text(encoding="utf-8")).encode()
    ).hexdigest()
    return RoleProfile(role, name, instructions, metadata, digest)
