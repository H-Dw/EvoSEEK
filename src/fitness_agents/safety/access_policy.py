from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

POLICY_VERSION = "workspace-access-policy:v1"
POLICY_FILES = (
    Path("AvoidRead.txt"),
    Path(".cursor/rules/avoid-biosafety-content.mdc"),
)
_RULE_DENY_LINE = re.compile(r"^\s*-\s+`([^`\r\n]+)`\s*$", re.MULTILINE)


def _normalize_relative_path(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKC", value.strip().strip("`\"'")
    ).replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        not normalized
        or candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or any(
            ":" in part or part.endswith((".", " "))
            for part in candidate.parts
        )
    ):
        raise ValueError(f"Access-policy paths must be safe project-relative paths: {value!r}")
    return candidate.as_posix()


def _path_identity(value: str) -> str:
    normalized = _normalize_relative_path(value)
    return normalized.casefold() if os.name == "nt" else normalized


def _paths_from_avoid_read(text: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for line in text.splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        paths.add(_normalize_relative_path(value))
    return tuple(sorted(paths))


def _paths_from_rule(text: str) -> tuple[str, ...]:
    paths: set[str] = set()
    for match in _RULE_DENY_LINE.finditer(text):
        value = match.group(1).strip()
        if "/" not in value and "\\" not in value:
            continue
        try:
            paths.add(_normalize_relative_path(value))
        except ValueError:
            continue
    return tuple(sorted(paths))


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    project_relative_path: str | None
    reason: str
    policy_hash: str


@dataclass(frozen=True)
class WorkspaceAccessPolicy:
    workspace_root: Path
    denied_relative_paths: tuple[str, ...]
    policy_sources: tuple[str, ...]
    policy_hash: str
    version: str = POLICY_VERSION

    @classmethod
    def load(cls, workspace_root: str | Path) -> WorkspaceAccessPolicy:
        root = Path(workspace_root).absolute()
        denied: set[str] = set()
        sources: list[str] = []
        for relative in POLICY_FILES:
            path = root / relative
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            parsed = (
                _paths_from_avoid_read(text)
                if relative.name == "AvoidRead.txt"
                else _paths_from_rule(text)
            )
            if not parsed:
                raise ValueError(f"Access-policy file contains no valid denied path: {path}")
            denied.update(parsed)
            sources.append(relative.as_posix())
        payload = {
            "version": POLICY_VERSION,
            "denied_relative_paths": sorted(denied),
            "policy_sources": sorted(sources),
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(
            workspace_root=root,
            denied_relative_paths=tuple(sorted(denied)),
            policy_sources=tuple(sorted(sources)),
            policy_hash=fingerprint,
        )

    @classmethod
    def empty(cls, root: str | Path) -> WorkspaceAccessPolicy:
        workspace_root = Path(root).absolute()
        payload = {
            "version": POLICY_VERSION,
            "denied_relative_paths": [],
            "policy_sources": [],
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return cls(workspace_root, (), (), fingerprint)

    def _project_relative(self, path: str | Path) -> str | None:
        absolute = Path(os.path.abspath(os.fspath(path)))
        try:
            relative = absolute.relative_to(self.workspace_root)
        except ValueError:
            return None
        return PurePosixPath(*relative.parts).as_posix()

    def decide(self, path: str | Path) -> AccessDecision:
        relative = self._project_relative(path)
        if relative is None:
            return AccessDecision(True, None, "outside_policy_workspace", self.policy_hash)
        if relative in {"", "."}:
            return AccessDecision(True, ".", "workspace_root_allowed", self.policy_hash)
        normalized = _normalize_relative_path(relative)
        denied_identities = {
            _path_identity(value) for value in self.denied_relative_paths
        }
        if _path_identity(normalized) in denied_identities:
            return AccessDecision(False, normalized, "explicitly_denied_path", self.policy_hash)
        return AccessDecision(True, normalized, "allowed", self.policy_hash)

    def require_allowed(self, path: str | Path) -> None:
        decision = self.decide(path)
        if not decision.allowed:
            raise PermissionError(
                f"Workspace access policy denied project path {decision.project_relative_path!r}"
            )

    def public_manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policy_hash": self.policy_hash,
            "policy_sources": list(self.policy_sources),
            "denied_path_count": len(self.denied_relative_paths),
        }


def discover_workspace_access_policy(start: str | Path) -> WorkspaceAccessPolicy:
    absolute = Path(start).absolute()
    # Do not probe ``start`` with is_dir()/stat before policy resolution.  A denied
    # file path may be supplied here, so walk the lexical ancestor chain and inspect
    # only the well-known policy-file locations.
    for candidate in (absolute, *absolute.parents):
        if any((candidate / relative).is_file() for relative in POLICY_FILES):
            return WorkspaceAccessPolicy.load(candidate)
    return WorkspaceAccessPolicy.empty(absolute)


def merge_denied_paths(
    policy: WorkspaceAccessPolicy,
    extra_paths: Iterable[str],
) -> WorkspaceAccessPolicy:
    denied = set(policy.denied_relative_paths)
    denied.update(_normalize_relative_path(value) for value in extra_paths)
    payload = {
        "version": POLICY_VERSION,
        "denied_relative_paths": sorted(denied),
        "policy_sources": sorted(policy.policy_sources),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return WorkspaceAccessPolicy(
        workspace_root=policy.workspace_root,
        denied_relative_paths=tuple(sorted(denied)),
        policy_sources=policy.policy_sources,
        policy_hash=fingerprint,
    )
