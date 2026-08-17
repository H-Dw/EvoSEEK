from __future__ import annotations

import csv
import hashlib
import json
import mimetypes
import os
import re
from collections.abc import Iterable
from fnmatch import fnmatch
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import yaml

from fitness_agents.config import LocalKnowledgeRootConfig

from .contracts import ParsedDocument

TEXT_EXTENSIONS = frozenset({".txt", ".md", ".markdown", ".rst"})
STRUCTURED_EXTENSIONS = frozenset({".json", ".yaml", ".yml", ".csv"})
DOCLING_EXTENSIONS = frozenset({".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".xml"})
DEFAULT_KNOWLEDGE_TYPE = "unclassified"
KNOWLEDGE_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


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
    return knowledge_type, metadata


def _matches(path: str, patterns: Iterable[str]) -> bool:
    normalized = path.replace("\\", "/")
    return any(
        fnmatch(normalized, pattern)
        or (pattern.startswith("**/") and fnmatch(normalized, pattern[3:]))
        for pattern in patterns
    )


def discover_local_files(
    roots: tuple[LocalKnowledgeRootConfig, ...],
    *,
    follow_symlinks: bool,
) -> tuple[Path, ...]:
    discovered: dict[str, Path] = {}
    for root_config in roots:
        root = root_config.path.resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Local knowledge root does not exist: {root}")
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.is_symlink() and not follow_symlinks:
                continue
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                raise ValueError(f"Local knowledge file escapes configured root: {path}")
            relative = path.relative_to(root).as_posix()
            if not _matches(relative, root_config.include):
                continue
            if _matches(relative, root_config.exclude):
                continue
            discovered[str(resolved)] = resolved
    return tuple(discovered[key] for key in sorted(discovered))


def _stable_document_id(file_hash: str, path: Path) -> str:
    payload = f"{path.resolve()}|{file_hash}"
    return f"localdoc:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


class AutoLocalParser:
    name = "auto_local_parser:v1"

    def __init__(self, *, rich_document_backend: str | None = None) -> None:
        self.rich_document_backend = rich_document_backend

    def supports(self, path: Path) -> bool:
        suffix = path.suffix.casefold()
        return suffix in TEXT_EXTENSIONS | STRUCTURED_EXTENSIONS | DOCLING_EXTENSIONS

    def parse(self, path: Path) -> ParsedDocument:
        suffix = path.suffix.casefold()
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
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
        configured_title = front_matter.get("title")
        title = str(configured_title).strip() if configured_title else path.stem
        return ParsedDocument(
            document_id=_stable_document_id(file_hash, path),
            path=path.resolve(),
            file_hash=file_hash,
            mime_type=mime_type,
            title=title,
            text=text.replace("\x00", "").strip(),
            knowledge_type=knowledge_type,
            metadata={
                "parser": self.name,
                "suffix": suffix,
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
