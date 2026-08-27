from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from fitness_agents.config import LocalKnowledgeRootConfig
from fitness_agents.safety import discover_workspace_access_policy

from .runtime_manifest import load_runtime_file_manifest

DOI_PATTERN = re.compile(r"^doi:10\.\d{4,9}/\S+$", re.IGNORECASE)


def _active_external_policy_hash() -> str:
    from fitness_agents.deep_research.policy import ExternalEvidenceScopePolicy

    return ExternalEvidenceScopePolicy().policy_hash


def _is_reparse_path(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


class PublicationCatalog:
    """Normalized publication records referenced by atomic local-knowledge claims."""

    def __init__(self, publications: dict[str, dict[str, Any]]) -> None:
        self.publications = publications

    @classmethod
    def from_roots(cls, roots: tuple[LocalKnowledgeRootConfig, ...]) -> PublicationCatalog:
        publications: dict[str, dict[str, Any]] = {}
        for root in roots:
            configured_root = root.path.absolute()
            access_policy = discover_workspace_access_policy(configured_root)
            root_decision = access_policy.decide(configured_root)
            if root.access_policy_mode == "required" and (
                not access_policy.policy_sources
                or root_decision.project_relative_path is None
            ):
                raise PermissionError(
                    "Local knowledge root is not bound to a workspace access policy"
                )
            access_policy.require_allowed(configured_root)
            if _is_reparse_path(configured_root):
                raise ValueError(
                    "Local knowledge roots must not be symlinks or junctions"
                )
            root_path = configured_root.resolve()
            runtime_manifest = load_runtime_file_manifest(
                root_path,
                access_policy=access_policy,
                expected_external_policy_hash=_active_external_policy_hash(),
            )
            if runtime_manifest is None:
                if root.runtime_manifest_mode == "required":
                    raise FileNotFoundError(
                        "Local knowledge root requires runtime-files.json"
                    )
                path = root_path / "catalog" / "publications.yaml"
                access_policy.require_allowed(path)
                if _is_reparse_path(path):
                    raise ValueError(
                        "Publication catalog must not be a symlink or junction"
                    )
                resolved = path.resolve()
                if not resolved.is_relative_to(root_path):
                    raise ValueError("Publication catalog escapes its configured root")
                access_policy.require_allowed(resolved)
                if not resolved.is_file():
                    continue
                raw = resolved.read_bytes()
            else:
                catalog_entries = runtime_manifest.entries_of_type(
                    "publication_catalog"
                )
                if len(catalog_entries) != 1:
                    raise ValueError(
                        "Runtime manifest must contain exactly one publication catalog"
                    )
                entry = catalog_entries[0]
                path = root_path / entry.relative_path
                access_policy.require_allowed(path)
                if _is_reparse_path(path):
                    raise ValueError(
                        "Publication catalog must not be a symlink or junction"
                    )
                resolved = path.resolve()
                access_policy.require_allowed(resolved)
                if not resolved.is_relative_to(root_path) or not resolved.is_file():
                    raise ValueError(
                        "Runtime-manifest publication catalog is missing or escapes its root"
                    )
                raw = resolved.read_bytes()
                if len(raw) != entry.bytes or hashlib.sha256(raw).hexdigest() != entry.sha256:
                    raise ValueError(
                        "Runtime-manifest publication catalog integrity mismatch"
                    )
            payload = yaml.safe_load(raw.decode("utf-8-sig")) or {}
            if payload.get("schema_version") != "scientific-publications:v1":
                raise ValueError(f"Unsupported publication catalog schema: {path}")
            if runtime_manifest is not None and (
                payload.get("generated_from") != runtime_manifest.source_release_id
            ):
                raise ValueError(
                    "Publication catalog source release does not match runtime manifest"
                )
            entries = payload.get("publications", [])
            if not isinstance(entries, list):
                raise TypeError(f"Publication catalog entries must be a list: {path}")
            for raw in entries:
                if not isinstance(raw, dict):
                    raise TypeError(f"Publication catalog entry must be a mapping: {path}")
                record = {str(key): value for key, value in raw.items()}
                publication_id = str(record.get("publication_id", "")).strip().casefold()
                if not DOI_PATTERN.fullmatch(publication_id):
                    raise ValueError(
                        f"Publication ID must be a normalized doi: identifier: {publication_id!r}"
                    )
                for key in ("title", "authors", "year", "venue", "doi", "url"):
                    if key not in record or record[key] in (None, "", []):
                        raise ValueError(f"Publication {publication_id} is missing {key}")
                doi = str(record["doi"]).strip().casefold()
                if publication_id != f"doi:{doi}":
                    raise ValueError(f"Publication ID/DOI mismatch for {publication_id}")
                prior = publications.get(publication_id)
                if prior is not None and prior != record:
                    raise ValueError(f"Conflicting publication definitions for {publication_id}")
                if runtime_manifest is not None:
                    verification = record.get("verification")
                    if not isinstance(verification, dict):
                        raise TypeError(
                            f"Publication {publication_id} lacks verification metadata"
                        )
                    if (
                        verification.get("metadata_verified") is not True
                        or verification.get("full_text_verified") is not True
                        or verification.get("source_release_id")
                        != runtime_manifest.source_release_id
                    ):
                        raise ValueError(
                            f"Publication {publication_id} is not release-verified"
                        )
                publications[publication_id] = record
        return cls(publications)

    def get(self, publication_id: str) -> dict[str, Any] | None:
        return self.publications.get(str(publication_id).strip().casefold())

    def require(self, publication_id: str) -> dict[str, Any]:
        normalized = str(publication_id).strip().casefold()
        record = self.get(normalized)
        if record is None:
            raise KeyError(f"Atomic claim references unknown publication {normalized!r}")
        return record
