from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from fitness_agents.safety import WorkspaceAccessPolicy

RUNTIME_MANIFEST_SCHEMA = "local-rag-runtime-files:v1"
NATIVE_RUNTIME_MANIFEST_SCHEMA = "local-rag-runtime-files:v2"


def _is_reparse_path(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


@dataclass(frozen=True)
class RuntimeFileEntry:
    relative_path: str
    record_type: str
    record_id: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class RuntimeFileManifest:
    root: Path
    source_release_id: str
    external_policy_hash: str
    workspace_access_policy_hash: str
    manifest_sha256: str
    entries: tuple[RuntimeFileEntry, ...]
    release_record_hashes: tuple[tuple[str, str], ...]

    def entries_of_type(self, record_type: str) -> tuple[RuntimeFileEntry, ...]:
        return tuple(item for item in self.entries if item.record_type == record_type)

    def release_record_hash(self, record_id: str) -> str | None:
        return dict(self.release_record_hashes).get(record_id)


def _safe_relative_path(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\\", "/")
    candidate = PurePosixPath(normalized)
    if (
        candidate.is_absolute()
        or not candidate.parts
        or ".." in candidate.parts
        or any(
            ":" in part or part.endswith((".", " "))
            for part in candidate.parts
        )
    ):
        raise ValueError(f"Unsafe runtime-manifest path: {value!r}")
    return candidate.as_posix()


def _canonical_manifest_hash(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_runtime_file_manifest(
    root: str | Path,
    *,
    access_policy: WorkspaceAccessPolicy,
    expected_external_policy_hash: str | None = None,
    active_policy: Any | None = None,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]] | None = None,
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]] | None = None,
) -> RuntimeFileManifest | None:
    """Load an exact-file manifest and its complete authorized evidence product.

    ``None`` means the root is a legacy corpus.  When a manifest exists, malformed
    content fails closed; callers must not fall back to recursive discovery. The
    signed Bundle v2 and every deterministic compatibility-projection byte are
    revalidated before the manifest is returned.
    """

    root_path = Path(root).absolute()
    manifest_path = root_path / "runtime-files.json"
    access_policy.require_allowed(manifest_path)
    if _is_reparse_path(manifest_path):
        raise ValueError("runtime-files.json must not be a symlink or junction")
    if not manifest_path.is_file():
        return None
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime-files.json must contain a mapping")
    manifest_schema = payload.get("schema_version")
    if manifest_schema not in {
        RUNTIME_MANIFEST_SCHEMA,
        NATIVE_RUNTIME_MANIFEST_SCHEMA,
    }:
        raise ValueError("Unsupported runtime-files.json schema")
    if payload.get("workspace_access_policy_hash") != access_policy.policy_hash:
        raise ValueError("Runtime manifest workspace access-policy hash mismatch")
    if (
        expected_external_policy_hash is not None
        and payload.get("policy_hash") != expected_external_policy_hash
    ):
        raise ValueError("Runtime manifest external evidence-policy hash mismatch")
    actual_manifest_hash = _canonical_manifest_hash(payload)
    if payload.get("manifest_sha256") != actual_manifest_hash:
        raise ValueError("Runtime manifest hash mismatch")
    raw_entries = payload.get("files")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Runtime manifest requires a non-empty files list")

    entries: list[RuntimeFileEntry] = []
    seen: set[str] = set()
    seen_records: set[tuple[str, str]] = set()
    allowed_record_types = {
        "atomic_claim",
        "logic_unit",
        "knowledge_decision_card",
        "publication_catalog",
        "evidence_release",
    }
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, dict):
            raise TypeError(f"Runtime manifest file[{index}] must be a mapping")
        relative = _safe_relative_path(str(raw.get("relative_path", "")))
        identity = relative.casefold() if os.name == "nt" else relative
        if identity in seen:
            raise ValueError(f"Duplicate runtime-manifest path: {relative}")
        seen.add(identity)
        size = raw.get("bytes")
        if type(size) is not int or size < 0:
            raise TypeError(f"Runtime manifest bytes must be a non-negative integer: {relative}")
        sha256 = str(raw.get("sha256", ""))
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            raise ValueError(f"Runtime manifest has an invalid sha256: {relative}")
        record_type = str(raw.get("record_type", "")).strip()
        record_id = str(raw.get("record_id", "")).strip()
        if not record_type or not record_id:
            raise ValueError(f"Runtime manifest entry lacks record identity: {relative}")
        if record_type not in allowed_record_types:
            raise ValueError(
                f"Runtime manifest contains unsupported record type: {record_type}"
            )
        record_identity = (record_type, record_id)
        if record_identity in seen_records:
            raise ValueError(
                f"Duplicate runtime-manifest record identity: {record_type}/{record_id}"
            )
        seen_records.add(record_identity)
        entries.append(
            RuntimeFileEntry(
                relative_path=relative,
                record_type=record_type,
                record_id=record_id,
                sha256=sha256,
                bytes=size,
            )
        )
    evidence_release_entries = [
        item for item in entries if item.record_type == "evidence_release"
    ]
    if len(evidence_release_entries) != 1:
        raise ValueError(
            "Runtime manifest must contain exactly one evidence release attestation"
        )
    release_entry = evidence_release_entries[0]
    release_path = root_path / release_entry.relative_path
    access_policy.require_allowed(release_path)
    if _is_reparse_path(release_path):
        raise ValueError(
            "Evidence release attestation must not be a symlink or junction"
        )
    resolved_release_path = release_path.resolve()
    if (
        not resolved_release_path.is_relative_to(root_path)
        or not resolved_release_path.is_file()
    ):
        raise ValueError("Evidence release attestation is missing or escapes its root")
    release_raw = resolved_release_path.read_bytes()
    if (
        len(release_raw) != release_entry.bytes
        or hashlib.sha256(release_raw).hexdigest() != release_entry.sha256
    ):
        raise ValueError("Evidence release attestation integrity mismatch")
    evidence_payload = json.loads(release_raw.decode("utf-8"))
    if not isinstance(evidence_payload, dict):
        raise TypeError("Evidence release attestation must contain a mapping")
    if evidence_payload.get("schema_version") != "scientific-evidence-product:v2":
        raise ValueError(
            "evidence-release.json must contain the complete signed EvidenceProduct v2"
        )
    if active_policy is None and trusted_reviewer_keys is None and (
        trusted_release_approval_keys is None
    ):
        from fitness_agents.deep_research.trust import (
            load_validation_trust_from_environment,
        )

        trust = load_validation_trust_from_environment()
        active_policy = trust.active_policy
        trusted_reviewer_keys = trust.reviewer_keys
        trusted_release_approval_keys = trust.release_approval_keys
    elif (
        active_policy is None
        or trusted_reviewer_keys is None
        or trusted_release_approval_keys is None
    ):
        raise ValueError(
            "Runtime evidence validation requires policy, reviewer, and release trust together"
        )
    from fitness_agents.deep_research.contracts import EvidenceProductBundle
    from fitness_agents.deep_research.pipeline import validate_evidence_product

    evidence_bundle = EvidenceProductBundle.model_validate_json(release_raw)
    validation = validate_evidence_product(
        evidence_bundle,
        active_policy=active_policy,
        trusted_reviewer_keys=trusted_reviewer_keys,
        trusted_release_approval_keys=trusted_release_approval_keys,
    )
    if not validation.release_ready:
        issue_codes = ", ".join(item.code for item in validation.issues[:8])
        raise ValueError(
            "Evidence release failed cryptographic/scientific validation: "
            + (issue_codes or "release_not_ready")
        )
    release_manifest = evidence_bundle.release_manifest
    if release_manifest is None:
        raise ValueError("Evidence product lacks a ReleaseManifest")
    release_payload = release_manifest.model_dump(mode="json")
    if release_payload.get("status") != "released":
        raise ValueError("Runtime indexing requires a released evidence attestation")
    if release_payload.get("release_id") != str(payload.get("source_release_id", "")):
        raise ValueError("Evidence release ID does not match runtime manifest")
    if release_entry.record_id != release_payload.get("release_id"):
        raise ValueError("Evidence release file identity does not match its manifest entry")
    if release_payload.get("policy_hash") != str(payload.get("policy_hash", "")):
        raise ValueError("Evidence release external policy hash mismatch")
    if release_payload.get("denied_path_operations") != 0:
        raise ValueError("Evidence release reports a denied-path operation")
    approval_ids = release_payload.get("release_approval_ids")
    if not isinstance(approval_ids, list) or not approval_ids:
        raise ValueError("Evidence release lacks approval receipts")
    raw_records = release_payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Evidence release lacks content-addressed records")
    record_hashes: dict[str, str] = {}
    record_types: dict[str, str] = {}
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise TypeError(f"Evidence release record[{index}] must be a mapping")
        record_id = str(raw_record.get("record_id", ""))
        content_hash = str(raw_record.get("content_sha256", ""))
        if not record_id or record_id in record_hashes:
            raise ValueError(f"Missing or duplicate evidence-release record ID: {record_id!r}")
        if len(content_hash) != 64 or any(
            char not in "0123456789abcdef" for char in content_hash
        ):
            raise ValueError(f"Invalid evidence-release record hash: {record_id}")
        record_hashes[record_id] = content_hash
        record_types[record_id] = str(raw_record.get("record_type", ""))
    for runtime_entry in entries:
        if runtime_entry.record_type not in {
            "atomic_claim",
            "logic_unit",
            "knowledge_decision_card",
        }:
            continue
        if record_types.get(runtime_entry.record_id) != runtime_entry.record_type:
            raise ValueError(
                "Runtime knowledge record lacks a matching canonical release record: "
                f"{runtime_entry.record_id}"
            )

    from fitness_agents.deep_research.export import (
        expected_legacy_projection_files,
        expected_native_projection_files,
    )

    expected_files = (
        expected_native_projection_files(evidence_bundle)
        if manifest_schema == NATIVE_RUNTIME_MANIFEST_SCHEMA
        else expected_legacy_projection_files(evidence_bundle)
    )
    entries_by_path = {item.relative_path: item for item in entries}
    if set(entries_by_path) != set(expected_files):
        missing = sorted(set(expected_files).difference(entries_by_path))
        unexpected = sorted(set(entries_by_path).difference(expected_files))
        raise ValueError(
            "Runtime projection file set differs from the signed evidence product: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, (record_type, record_id, expected_raw) in expected_files.items():
        entry = entries_by_path[relative]
        if entry.record_type != record_type or entry.record_id != record_id:
            raise ValueError(
                f"Runtime projection identity mismatch: {relative}"
            )
        target = root_path / PurePosixPath(relative)
        access_policy.require_allowed(target)
        if _is_reparse_path(target):
            raise ValueError(
                f"Runtime projection must not be a symlink or junction: {relative}"
            )
        resolved = target.resolve()
        if not resolved.is_relative_to(root_path) or not resolved.is_file():
            raise ValueError(
                f"Runtime projection is missing or escapes its root: {relative}"
            )
        actual_raw = resolved.read_bytes()
        if actual_raw != expected_raw:
            raise ValueError(
                "Runtime projection bytes differ from the deterministic projection "
                f"of the signed evidence product: {relative}"
            )
        if (
            len(actual_raw) != entry.bytes
            or hashlib.sha256(actual_raw).hexdigest() != entry.sha256
        ):
            raise ValueError(
                f"Runtime projection integrity mismatch: {relative}"
            )

    return RuntimeFileManifest(
        root=root_path,
        source_release_id=str(payload.get("source_release_id", "")),
        external_policy_hash=str(payload.get("policy_hash", "")),
        workspace_access_policy_hash=access_policy.policy_hash,
        manifest_sha256=actual_manifest_hash,
        entries=tuple(entries),
        release_record_hashes=tuple(sorted(record_hashes.items())),
    )
