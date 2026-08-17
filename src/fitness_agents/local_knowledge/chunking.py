from __future__ import annotations

import hashlib
import re

from .contracts import DocumentChunk, ParsedDocument

CHUNKER_VERSION = "section-char-v1"


def approximate_token_count(text: str) -> int:
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    return max(1, ascii_words + cjk_chars)


def _section_path(text: str, offset: int, title: str) -> tuple[str, ...]:
    headings: dict[int, str] = {}
    for line in text[:offset].splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if not match:
            continue
        level = len(match.group(1))
        headings[level] = match.group(2)
        for deeper in tuple(key for key in headings if key > level):
            headings.pop(deeper, None)
    ordered = tuple(headings[key] for key in sorted(headings))
    return ordered or (title,)


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    source_group: str,
) -> tuple[DocumentChunk, ...]:
    text = document.text.strip()
    if not text:
        return ()
    max_chars = max(256, chunk_tokens * 4)
    overlap_chars = chunk_overlap * 4
    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + max_chars)
        end = target_end
        if target_end < len(text):
            candidates = [
                text.rfind("\n\n", start + max_chars // 2, target_end),
                text.rfind("\n", start + max_chars // 2, target_end),
                text.rfind(". ", start + max_chars // 2, target_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] in {"\n\n", ". "} else 1)
        chunk_text = text[start:end].strip()
        if chunk_text:
            payload = (
                f"{document.file_hash}|{start}|{end}|{CHUNKER_VERSION}|"
                f"{hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()}"
            )
            chunk_id = f"chunk:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    text=chunk_text,
                    section_path=_section_path(text, start, document.title),
                    start_offset=start,
                    end_offset=end,
                    token_count=approximate_token_count(chunk_text),
                    source_group=source_group,
                    artifact_uri=str(document.path),
                    file_hash=document.file_hash,
                    knowledge_type=document.knowledge_type,
                    metadata={
                        **document.metadata,
                        "chunker_version": CHUNKER_VERSION,
                    },
                )
            )
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return tuple(chunks)
