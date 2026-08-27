from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from fitness_agents.config import LocalKnowledgeRootConfig
from fitness_agents.safety import discover_workspace_access_policy

from .contracts import ParsedDocument
from .runtime_manifest import load_runtime_file_manifest

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst"})
STRUCTURED_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".csv"})
DOCLING_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".xml"})
DEFAULT_KNOWLEDGE_TYPE = "unclassified"
KNOWLEDGE_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True)
class DiscoveredLocalFile:
    path: Path
    root_id: str
    relative_path: str
    expected_sha256: str | None = None
    expected_bytes: int | None = None
    runtime_manifest_sha256: str | None = None
    expected_record_id: str | None = None
    expected_record_type: str | None = None
    expected_record_content_sha256: str | None = None


def _markdown_front_matter(text: str) -> tuple[str, dict[str, object]]:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return text, {}
    closing = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )
    if closing is None:
        raise ValueError("Markdown front matter is missing its closing delimiter")
    payload = yaml.safe_load("".join(lines[1:closing])) or {}
    if not isinstance(payload, dict):
        raise TypeError("Markdown front matter must be a mapping")
    return "".join(lines[closing + 1 :]).lstrip(), {
        str(key): value for key, value in payload.items()
    }


def _knowledge_metadata(front_matter: dict[str, object]) -> tuple[str, dict[str, object]]:
    knowledge_type = str(
        front_matter.get("knowledge_type", DEFAULT_KNOWLEDGE_TYPE)
    ).strip()
    if not KNOWLEDGE_TYPE_PATTERN.fullmatch(knowledge_type):
        raise ValueError(
            "knowledge_type must be a lowercase snake_case identifier with 2-64 characters"
        )
    metadata: dict[str, object] = {
        "knowledge_type": knowledge_type,
        "front_matter": front_matter,
    }
    for key in (
        "schema_version",
        "record_type",
        "record_id",
        "claim_id",
        "logic_unit_id",
        "decision_card_id",
        "retrieval_text",
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
        "permission",
        "scientific_quality",
        "task_applicability",
        "boundary_conditions",
        "counterclaims",
        "abstain_if",
        "record_payload",
        "question_leaf_id",
        "decision_slot",
        "task_route",
        "feature_channel",
        "required_input",
        "expected_direction",
        "stage",
        "evidence_role",
        "source_release_id",
        "source_record_hash",
        "language",
        "version",
        "evidence_level",
        "rule_scope",
        "topics",
        "citation_keys",
        "applies_to",
        "excludes",
    ):
        if key in front_matter:
            metadata[key] = front_matter[key]
    if metadata.get("record_type") == "atomic_claim":
        required = ("schema_version", "claim_id", "statement", "subject", "predicate", "object")
        missing = [key for key in required if not str(metadata.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Atomic claim front matter is missing: {', '.join(missing)}")
        polarity = str(metadata.get("polarity", "support"))
        if polarity not in {"support", "contradict", "neutral", "unknown"}:
            raise ValueError("Atomic claim polarity is invalid")
        citation_support = metadata.get("citation_support", [])
        if not isinstance(citation_support, list):
            raise TypeError("Atomic claim citation_support must be a list")
    elif metadata.get("record_type") in {"logic_unit", "knowledge_decision_card"}:
        required = ("schema_version", "record_id", "retrieval_text", "permission")
        missing = [key for key in required if not str(metadata.get(key, "")).strip()]
        if missing:
            raise ValueError(
                f"Native knowledge record front matter is missing: {', '.join(missing)}"
            )
    return knowledge_type, metadata


def _matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        fnmatch(normalized, pattern)
        or (pattern.startswith("**/") and fnmatch(normalized, pattern[3:]))
        for pattern in patterns
    )


def _active_external_policy_hash() -> str:
    from fitness_agents.deep_research.policy import ExternalEvidenceScopePolicy

    return ExternalEvidenceScopePolicy().policy_hash


def _is_reparse_path(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(is_junction and is_junction())


def discover_local_files(
    roots: tuple[LocalKnowledgeRootConfig, ...],
    *,
    follow_symlinks: bool,
    policy_events: list[dict[str, str]] | None = None,
) -> tuple[DiscoveredLocalFile, ...]:
    discovered: dict[str, DiscoveredLocalFile] = {}
    resolved_root_ids: set[str] = set()
    for root_config in roots:
        configured_root = root_config.path.absolute()
        access_policy = discover_workspace_access_policy(configured_root)
        root_policy_decision = access_policy.decide(configured_root)
        if root_config.access_policy_mode == "required" and (
            not access_policy.policy_sources
            or root_policy_decision.project_relative_path is None
        ):
            raise PermissionError(
                "Local knowledge root is not bound to a workspace access policy"
            )
        access_policy.require_allowed(configured_root)
        if _is_reparse_path(configured_root):
            raise ValueError(
                "Local knowledge roots must not be symlinks or junctions"
            )
        root = configured_root.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Local knowledge root does not exist: {root}")
        root_id = root_config.root_id or re.sub(
            r"[^A-Za-z0-9_-]+", "-", root.name
        ).strip("-").upper()
        root_id = root_id or "ROOT"
        if root_id in resolved_root_ids:
            raise ValueError(
                f"Local knowledge roots resolve to duplicate root_id {root_id!r}; "
                "configure explicit unique root_id values"
            )
        resolved_root_ids.add(root_id)
        runtime_manifest = load_runtime_file_manifest(
            root,
            access_policy=access_policy,
            expected_external_policy_hash=_active_external_policy_hash(),
        )
        if (
            runtime_manifest is None
            and root_config.runtime_manifest_mode == "required"
        ):
            raise FileNotFoundError(
                "Local knowledge root requires runtime-files.json; recursive discovery is disabled"
            )
        if runtime_manifest is not None:
            for entry in runtime_manifest.entries:
                path = root / entry.relative_path
                access_decision = access_policy.decide(path)
                if not access_decision.allowed:
                    if policy_events is not None:
                        policy_events.append(
                            {
                                "event": "manifest_path_denied_before_stat",
                                "root_id": root_id,
                                "relative_path": entry.relative_path,
                                "policy_hash": access_decision.policy_hash,
                            }
                        )
                    raise PermissionError(
                        "Runtime manifest lists a workspace-policy-denied path"
                    )
                if entry.record_type not in {
                    "atomic_claim",
                    "logic_unit",
                    "knowledge_decision_card",
                }:
                    continue
                relative = entry.relative_path
                if not _matches(relative, root_config.include):
                    continue
                if _matches(relative, root_config.exclude):
                    continue
                if _is_reparse_path(path):
                    raise ValueError(
                        "Runtime-manifest files must not be symlinks or junctions"
                    )
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(
                        f"Runtime-manifest file escapes configured root: {relative}"
                    )
                access_policy.require_allowed(resolved)
                if not resolved.is_file():
                    raise FileNotFoundError(
                        f"Runtime-manifest file is missing: {relative}"
                    )
                discovered[str(resolved)] = DiscoveredLocalFile(
                    path=resolved,
                    root_id=root_id,
                    relative_path=relative,
                    expected_sha256=entry.sha256,
                    expected_bytes=entry.bytes,
                    runtime_manifest_sha256=runtime_manifest.manifest_sha256,
                    expected_record_id=entry.record_id,
                    expected_record_type=entry.record_type,
                    expected_record_content_sha256=(
                        runtime_manifest.release_record_hash(entry.record_id)
                    ),
                )
            continue
        for directory, directory_names, file_names in os.walk(
            root,
            topdown=True,
            followlinks=follow_symlinks,
        ):
            directory_path = Path(directory)
            directory_names[:] = sorted(
                name
                for name in directory_names
                if not _matches(
                    (directory_path / name).relative_to(root).as_posix() + "/",
                    root_config.exclude,
                )
            )
            for name in sorted(file_names):
                path = directory_path / name
                relative = path.relative_to(root).as_posix()
                if not _matches(relative, root_config.include):
                    continue
                if _matches(relative, root_config.exclude):
                    continue
                access_decision = access_policy.decide(path)
                if not access_decision.allowed:
                    if policy_events is not None:
                        policy_events.append(
                            {
                                "event": "path_denied_before_stat",
                                "root_id": root_id,
                                "relative_path": relative,
                                "policy_hash": access_decision.policy_hash,
                            }
                        )
                    continue
                if path.is_symlink() and not follow_symlinks:
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Local knowledge file escapes configured root: {path}")
                access_policy.require_allowed(resolved)
                if not resolved.is_file():
                    continue
                discovered[str(resolved)] = DiscoveredLocalFile(
                    path=resolved,
                    root_id=root_id,
                    relative_path=relative,
                )
    return tuple(discovered[key] for key in sorted(discovered))


def _stable_document_id(
    file_hash: str,
    path: Path,
    *,
    root_id: str | None = None,
    relative_path: str | None = None,
    claim_id: str | None = None,
) -> str:
    del file_hash  # content integrity remains a separate field, not an identifier contract
    root_label = root_id or re.sub(
        r"[^A-Za-z0-9_-]+", "-", path.resolve().parent.name
    ).strip("-").upper() or "ROOT"
    identity = (
        f"claim/{claim_id}"
        if claim_id
        else (relative_path or path.name).replace("\\", "/")
    )
    encoded = re.sub(r"[^A-Za-z0-9._/-]+", "-", identity).strip("-/") or "ITEM"
    return f"localdoc:{root_label}:{encoded}"


class AutoLocalParser:
    name = "auto_local_parser:v1"

    def __init__(self, *, rich_document_backend: str | None = None) -> None:
        self.rich_document_backend = rich_document_backend

    def supports(self, path: Path) -> bool:
        suffix = path.suffix.casefold()
        return suffix in TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS | DOCLING_EXTENSIONS

    def parse(
        self,
        path: Path | DiscoveredLocalFile,
    ) -> ParsedDocument:
        discovered = path if isinstance(path, DiscoveredLocalFile) else None
        if discovered is None:
            raise TypeError(
                "AutoLocalParser requires a policy-screened DiscoveredLocalFile"
            )
        path = discovered.path if discovered is not None else path
        access_policy = discover_workspace_access_policy(path.parent)
        access_policy.require_allowed(path)
        if _is_reparse_path(path):
            raise ValueError(
                "Policy-screened local knowledge files must not be symlinks or junctions"
            )
        suffix = path.suffix.casefold()
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        if (
            discovered is not None
            and discovered.expected_bytes is not None
            and len(raw) != discovered.expected_bytes
        ):
            raise ValueError(
                f"Runtime-manifest byte count mismatch: {discovered.relative_path}"
            )
        if (
            discovered is not None
            and discovered.expected_sha256 is not None
            and file_hash != discovered.expected_sha256
        ):
            raise ValueError(
                f"Runtime-manifest sha256 mismatch: {discovered.relative_path}"
            )
        front_matter: dict[str, object] = {}
        if suffix in TEXT_EXTENSIONS:
            text = raw.decode("utf-8-sig")
            if suffix in {".md", ".markdown"}:
                text, front_matter = _markdown_front_matter(text)
        elif suffix == ".json":
            text = json.dumps(json.loads(raw.decode("utf-8-sig")), ensure_ascii=False, indent=2)
        elif suffix in {".yaml", ".yml"}:
            payload = yaml.safe_load(raw.decode("utf-8-sig"))
            text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)
        elif suffix == ".csv":
            reader = csv.DictReader(StringIO(raw.decode("utf-8-sig")))
            text = "\n".join(
                "; ".join(f"{key}: {value}" for key, value in row.items()) for row in reader
            )
        elif suffix in DOCLING_EXTENSIONS:
            text = self._parse_with_docling(path)
        else:
            raise ValueError(f"Unsupported local knowledge file type: {path.suffix}")
        mime_type = mimetypes.guess_type(path.name)[0] or "text/plain"
        knowledge_type, knowledge_metadata = _knowledge_metadata(front_matter)
        if discovered.expected_record_id is not None:
            actual_record_id = str(
                knowledge_metadata.get("record_id")
                or knowledge_metadata.get("claim_id")
                or ""
            )
            if actual_record_id != discovered.expected_record_id:
                raise ValueError(
                    "Runtime-manifest record ID does not match knowledge-record front matter"
                )
            if (
                discovered.expected_record_type is not None
                and knowledge_metadata.get("record_type")
                != discovered.expected_record_type
            ):
                raise ValueError(
                    "Runtime-manifest record type does not match knowledge-record front matter"
                )
            if (
                str(knowledge_metadata.get("source_record_hash", ""))
                != discovered.expected_record_content_sha256
            ):
                raise ValueError(
                    "Knowledge-record hash does not match evidence release"
                )
        configured_title = front_matter.get("title")
        title = str(configured_title).strip() if configured_title else path.stem
        record_id = (
            str(
                knowledge_metadata.get("record_id")
                or knowledge_metadata.get("claim_id")
                or ""
            ).strip()
            if knowledge_metadata.get("record_type")
            in {"atomic_claim", "logic_unit", "knowledge_decision_card"}
            else None
        )
        return ParsedDocument(
            document_id=_stable_document_id(
                file_hash,
                path,
                root_id=discovered.root_id if discovered is not None else None,
                relative_path=(
                    discovered.relative_path if discovered is not None else None
                ),
                claim_id=record_id or None,
            ),
            path=path.resolve(),
            file_hash=file_hash,
            mime_type=mime_type,
            title=title,
            text=text.replace("\x00", "").strip(),
            knowledge_type=knowledge_type,
            metadata={
                "parser": self.name,
                "suffix": suffix,
                "root_id": discovered.root_id if discovered is not None else None,
                "relative_path": (
                    discovered.relative_path if discovered is not None else path.name
                ),
                "runtime_manifest_sha256": (
                    discovered.runtime_manifest_sha256
                    if discovered is not None
                    else None
                ),
                **knowledge_metadata,
            },
        )

    def _parse_with_docling(self, path: Path) -> str:
        if self.rich_document_backend != "docling":
            raise RuntimeError(
                f"Rich document {path.name!r} requires rich_document_backend=docling"
            )
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as error:
            raise RuntimeError(
                "Docling is required for rich local documents; install the rag-docs extra"
            ) from error
        with patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "DOCLING_OFFLINE": "1",
            },
            clear=False,
        ):
            result = DocumentConverter().convert(path)
        return str(result.document.export_to_markdown())
