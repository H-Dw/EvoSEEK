from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from fitness_agents.local_knowledge.runtime_manifest import (
    RuntimeFileManifest,
    load_runtime_file_manifest,
)
from fitness_agents.safety import discover_workspace_access_policy

from .policy import ExternalEvidenceScopePolicy

CLAIM_SCHEMA = "scientific-atomic-claim:v1"
PUBLICATION_SCHEMA = "scientific-publications:v1"
RUNTIME_MANIFEST_SCHEMA = "local-rag-runtime-files:v1"
VALIDATOR_VERSION = "legacy-runtime-bundle-validator:v3"
DOI_ID = re.compile(r"^doi:10\.\d{4,9}/\S+$", re.IGNORECASE)
CJK = re.compile(r"[\u3400-\u9fff]")
KNOWLEDGE_TYPE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
INSTRUCTION_MARKERS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "execute the shell",
    "run this command",
)


def _is_reparse_path(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("missing YAML front matter")
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise ValueError("missing closing front-matter delimiter")
    payload = yaml.safe_load("".join(lines[1:closing])) or {}
    if not isinstance(payload, dict):
        raise TypeError("front matter must be a mapping")
    return {str(key): value for key, value in payload.items()}, "".join(
        lines[closing + 1 :]
    ).strip()


def _safe_relative_path(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or not path.parts
        or ".." in path.parts
        or any(":" in part or part.endswith((".", " ")) for part in path.parts)
    ):
        raise ValueError(f"unsafe runtime-manifest path: {value!r}")
    return path.as_posix()


def load_manifest_files(
    root: Path,
    errors: list[str],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, dict[str, Any]]]:
    access_policy = discover_workspace_access_policy(root)
    manifest_path = root / "runtime-files.json"
    access_policy.require_allowed(manifest_path)
    if _is_reparse_path(manifest_path):
        errors.append("runtime-files.json must not be a symlink or junction")
        return {}, {}, {}
    if not manifest_path.is_file():
        errors.append(
            "missing runtime-files.json; recursive legacy bundle discovery is disabled"
        )
        return {}, {}, {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid runtime-files.json: {error}")
        return {}, {}, {}
    if not isinstance(payload, dict) or payload.get("schema_version") != RUNTIME_MANIFEST_SCHEMA:
        errors.append("unsupported runtime-files.json schema")
        return {}, {}, {}
    unsigned = dict(payload)
    expected_manifest_hash = str(unsigned.pop("manifest_sha256", ""))
    actual_manifest_hash = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if expected_manifest_hash != actual_manifest_hash:
        errors.append("runtime-files.json manifest hash mismatch")
        return payload, {}, {}
    if str(payload.get("workspace_access_policy_hash", "")) != access_policy.policy_hash:
        errors.append("workspace access-policy hash does not match runtime manifest")
        return payload, {}, {}
    raw_entries = payload.get("files")
    if not isinstance(raw_entries, list) or not raw_entries:
        errors.append("runtime manifest requires a non-empty files list")
        return payload, {}, {}

    raw_by_path: dict[str, bytes] = {}
    entries_by_path: dict[str, dict[str, Any]] = {}
    path_identities: set[str] = set()
    root_absolute = root.absolute()
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            errors.append(f"runtime manifest file[{index}] must be a mapping")
            continue
        try:
            relative = _safe_relative_path(str(raw_entry.get("relative_path", "")))
        except ValueError as error:
            errors.append(str(error))
            continue
        path_identity = relative.casefold() if os.name == "nt" else relative
        if path_identity in path_identities:
            errors.append(f"duplicate runtime-manifest path: {relative}")
            continue
        path_identities.add(path_identity)
        target = root / PurePosixPath(relative)
        decision = access_policy.decide(target)
        if not decision.allowed:
            errors.append(f"runtime-manifest path denied before stat: {relative}")
            continue
        if _is_reparse_path(target):
            errors.append(
                f"runtime-manifest path must not be a symlink or junction: {relative}"
            )
            continue
        resolved = target.resolve()
        if not resolved.is_relative_to(root_absolute):
            errors.append(f"runtime-manifest path escapes bundle root: {relative}")
            continue
        if not resolved.is_file():
            errors.append(f"runtime-manifest file is missing: {relative}")
            continue
        raw = resolved.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(raw_entry.get("sha256", "")):
            errors.append(f"runtime-manifest sha256 mismatch: {relative}")
            continue
        expected_bytes = raw_entry.get("bytes")
        if type(expected_bytes) is not int or expected_bytes < 0:
            errors.append(f"runtime-manifest byte count is invalid: {relative}")
            continue
        if len(raw) != expected_bytes:
            errors.append(f"runtime-manifest byte count mismatch: {relative}")
            continue
        raw_by_path[relative] = raw
        entries_by_path[relative] = {
            str(key): value for key, value in raw_entry.items()
        }
    return payload, raw_by_path, entries_by_path


def load_publications(
    raw: bytes | None,
    label: str,
    errors: list[str],
) -> dict[str, dict[str, Any]]:
    if raw is None:
        errors.append(f"missing publication catalog in runtime manifest: {label}")
        return {}
    payload = yaml.safe_load(raw.decode("utf-8-sig")) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != PUBLICATION_SCHEMA:
        errors.append(f"unsupported publication catalog schema: {label}")
        return {}
    publications: dict[str, dict[str, Any]] = {}
    for index, raw_publication in enumerate(payload.get("publications", [])):
        if not isinstance(raw_publication, dict):
            errors.append(f"publication[{index}] must be a mapping")
            continue
        publication_id = str(
            raw_publication.get("publication_id", "")
        ).strip().casefold()
        if not DOI_ID.fullmatch(publication_id):
            errors.append(f"invalid publication_id: {publication_id!r}")
            continue
        required = ("title", "authors", "year", "venue", "doi", "url", "verification")
        missing = [
            key for key in required if raw_publication.get(key) in (None, "", [])
        ]
        if missing:
            errors.append(f"publication {publication_id} missing: {', '.join(missing)}")
        doi = str(raw_publication.get("doi", "")).strip().casefold()
        if publication_id != f"doi:{doi}":
            errors.append(f"publication ID/DOI mismatch: {publication_id}")
        if publication_id in publications:
            errors.append(f"duplicate publication_id: {publication_id}")
        publications[publication_id] = dict(raw_publication)
    return publications


def model_token_counter(model_path: Path | None) -> tuple[Any, int | None]:
    if model_path is None:
        return None, None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Install sentence-transformers for model-token validation") from error
    model = SentenceTransformer(str(model_path.resolve()), local_files_only=True)
    tokenizer = model.tokenizer

    def count(text: str) -> int:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return len(encoded["input_ids"])

    return count, int(model.max_seq_length)


def validate_legacy_runtime_bundle(
    root: str | Path,
    *,
    embedding_model: Path | None = None,
    active_policy: ExternalEvidenceScopePolicy | None = None,
    trusted_reviewer_keys: Mapping[str, tuple[str, bytes]] | None = None,
    trusted_release_approval_keys: Mapping[str, tuple[str, bytes]] | None = None,
) -> dict[str, Any]:
    bundle_root = Path(root).absolute()
    errors: list[str] = []
    warnings: list[str] = []
    access_policy = discover_workspace_access_policy(bundle_root)
    attestation: RuntimeFileManifest | None = None
    try:
        attestation = load_runtime_file_manifest(
            bundle_root,
            access_policy=access_policy,
            expected_external_policy_hash=(
                active_policy.policy_hash
                if active_policy is not None
                else ExternalEvidenceScopePolicy().policy_hash
            ),
            active_policy=active_policy,
            trusted_reviewer_keys=trusted_reviewer_keys,
            trusted_release_approval_keys=trusted_release_approval_keys,
        )
    except (FileNotFoundError, PermissionError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"runtime release attestation failed: {error}")
    manifest, raw_files, entries = load_manifest_files(bundle_root, errors)
    if attestation is None:
        errors.append("runtime release attestation is missing")
    catalog_paths = [
        path
        for path, entry in entries.items()
        if entry.get("record_type") == "publication_catalog"
    ]
    publications = (
        load_publications(raw_files.get(catalog_paths[0]), catalog_paths[0], errors)
        if len(catalog_paths) == 1
        else {}
    )
    if len(catalog_paths) != 1:
        errors.append("runtime manifest must contain exactly one publication catalog")
    for publication_id, publication in publications.items():
        verification = publication.get("verification")
        if not isinstance(verification, dict):
            errors.append(f"publication {publication_id} lacks verification metadata")
            continue
        if (
            verification.get("metadata_verified") is not True
            or verification.get("full_text_verified") is not True
            or verification.get("source_release_id")
            != manifest.get("source_release_id")
        ):
            errors.append(f"publication {publication_id} is not release-verified")
    token_counter, model_limit = model_token_counter(embedding_model)
    claim_ids: set[str] = set()
    support_ids: set[str] = set()
    knowledge_types: Counter[str] = Counter()
    token_counts: list[int] = []
    verified_support = 0
    unverified_support = 0
    claim_paths = sorted(
        path
        for path, entry in entries.items()
        if entry.get("record_type") == "atomic_claim"
    )
    if not claim_paths:
        errors.append("runtime manifest contains no atomic claim files")

    for relative in claim_paths:
        raw = raw_files.get(relative)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8-sig")
            metadata, body = split_front_matter(text)
        except (UnicodeError, TypeError, ValueError, yaml.YAMLError) as error:
            errors.append(f"{relative}: {error}")
            continue
        required = (
            "schema_version",
            "record_type",
            "claim_id",
            "title",
            "language",
            "knowledge_type",
            "statement",
            "subject",
            "predicate",
            "object",
            "polarity",
            "claim_kind",
            "confidence",
            "applicability",
            "citation_support",
            "selection_eligible",
            "source_release_id",
            "source_record_hash",
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            errors.append(f"{relative}: missing fields: {', '.join(missing)}")
            continue
        if metadata["schema_version"] != CLAIM_SCHEMA or metadata["record_type"] != "atomic_claim":
            errors.append(f"{relative}: unsupported claim schema or record type")
        if metadata["language"] != "en" or CJK.search(text):
            errors.append(f"{relative}: claim must be English-only")
        knowledge_type = str(metadata["knowledge_type"])
        if not KNOWLEDGE_TYPE.fullmatch(knowledge_type):
            errors.append(f"{relative}: invalid knowledge_type {knowledge_type!r}")
        knowledge_types[knowledge_type] += 1
        claim_id = str(metadata["claim_id"])
        if claim_id in claim_ids:
            errors.append(f"{relative}: duplicate claim_id {claim_id}")
        claim_ids.add(claim_id)
        if attestation is not None:
            if metadata.get("source_release_id") != attestation.source_release_id:
                errors.append(f"{relative}: source_release_id mismatch")
            if metadata.get("source_record_hash") != attestation.release_record_hash(
                claim_id
            ):
                errors.append(f"{relative}: source_record_hash mismatch")
        statement = " ".join(str(metadata["statement"]).split())
        if " ".join(body.split()) != statement:
            errors.append(f"{relative}: body must equal the atomic statement")
        markers = [item for item in INSTRUCTION_MARKERS if item in body.casefold()]
        if markers:
            errors.append(f"{relative}: instruction-like content: {markers}")
        try:
            confidence = float(metadata["confidence"])
        except (TypeError, ValueError):
            errors.append(f"{relative}: confidence must be numeric")
        else:
            if not 0.0 <= confidence <= 1.0:
                errors.append(f"{relative}: confidence must be in [0, 1]")
        if type(metadata["selection_eligible"]) is not bool:
            errors.append(f"{relative}: selection_eligible must be a strict boolean")
        elif metadata["selection_eligible"]:
            errors.append(
                f"{relative}: v1 runtime claims cannot carry selection permission"
            )
        supports = metadata["citation_support"]
        if not isinstance(supports, list) or not supports:
            errors.append(f"{relative}: citation_support must be a non-empty list")
            continue
        for support in supports:
            if not isinstance(support, dict):
                errors.append(f"{relative}: citation support must be a mapping")
                continue
            support_id = str(support.get("support_id", ""))
            if not support_id or support_id in support_ids:
                errors.append(
                    f"{relative}: missing or duplicate support_id {support_id!r}"
                )
            support_ids.add(support_id)
            publication_id = str(support.get("publication_id", "")).casefold()
            if publication_id not in publications:
                errors.append(f"{relative}: unknown publication {publication_id!r}")
            if not support.get("locator") or not support.get("support_type"):
                errors.append(
                    f"{relative}: citation support requires locator and support_type"
                )
            verified = support.get("verified_against_source", False)
            if type(verified) is not bool:
                errors.append(
                    f"{relative}: verified_against_source must be a strict boolean"
                )
            elif verified:
                verified_support += 1
            else:
                unverified_support += 1
                errors.append(
                    f"{relative}: released citation support must be source-verified"
                )
        if token_counter is not None and model_limit is not None:
            count = int(token_counter(body))
            token_counts.append(count)
            if count > model_limit:
                errors.append(
                    f"{relative}: {count} model tokens exceed model limit {model_limit}"
                )

    if unverified_support:
        warnings.append(f"{unverified_support} citation supports are not full-source verified")
    return {
        "validator_version": VALIDATOR_VERSION,
        "bundle_root": str(bundle_root),
        "runtime_manifest_schema": manifest.get("schema_version"),
        "source_release_id": manifest.get("source_release_id"),
        "external_scope_policy_hash": manifest.get("policy_hash"),
        "workspace_access_policy_hash": manifest.get("workspace_access_policy_hash"),
        "valid": not errors,
        "claim_count": len(claim_paths),
        "publication_count": len(publications),
        "citation_support_count": len(support_ids),
        "verified_citation_support_count": verified_support,
        "unverified_citation_support_count": unverified_support,
        "knowledge_types": dict(sorted(knowledge_types.items())),
        "model_token_check": {
            "performed": token_counter is not None,
            "model_path": str(embedding_model.resolve()) if embedding_model else None,
            "model_max_tokens": model_limit,
            "maximum_claim_tokens": max(token_counts, default=None),
        },
        "denied_path_read_count": 0,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a manifest-driven legacy local-RAG export"
    )
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    args = parser.parse_args()
    report = validate_legacy_runtime_bundle(
        args.bundle_root,
        embedding_model=args.embedding_model,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
