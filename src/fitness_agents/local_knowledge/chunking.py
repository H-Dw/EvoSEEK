from __future__ import annotations

import re
from collections.abc import Callable

from .contracts import DocumentChunk, ParsedDocument

CHUNKER_VERSION = "markdown-token-atomic-v2"
TokenCounter = Callable[[str], int]


def approximate_token_count(text: str) -> int:
    ascii_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    cjk_chars = len(re.findall(r"[\u3400-\u9fff]", text))
    punctuation = len(re.findall(r"[^\w\s]", text, flags=re.UNICODE))
    return max(1, ascii_words + cjk_chars + punctuation // 4)


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


def _largest_fitting_end(text: str, start: int, counter: TokenCounter, limit: int) -> int:
    low = start + 1
    high = len(text)
    best = low
    while low <= high:
        middle = (low + high) // 2
        if counter(text[start:middle]) <= limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < len(text):
        floor = start + max(1, (best - start) // 2)
        boundaries = (
            text.rfind("\n\n", floor, best),
            text.rfind("\n", floor, best),
            text.rfind(". ", floor, best),
            text.rfind("; ", floor, best),
            text.rfind(" ", floor, best),
        )
        boundary = max(boundaries)
        if boundary > start:
            width = 2 if text[boundary : boundary + 2] in {"\n\n", ". ", "; "} else 1
            return boundary + width
    return best


def _overlap_start(
    text: str,
    *,
    start: int,
    end: int,
    counter: TokenCounter,
    overlap_tokens: int,
) -> int:
    if overlap_tokens <= 0:
        return end
    low = start
    high = end
    best = end
    while low <= high:
        middle = (low + high) // 2
        if counter(text[middle:end]) <= overlap_tokens:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    boundary = text.find(" ", best, end)
    return boundary + 1 if boundary >= 0 else best


def _chunk_id(document: ParsedDocument, start: int, end: int, text: str) -> str:
    del text
    document_label = re.sub(
        r"[^A-Za-z0-9]+", "-", document.document_id.rsplit(":", 1)[-1]
    ).strip("-")[:18]
    return f"chunk:CHK-{document_label}-{start:06X}-{end - start:05X}"


def chunk_document(
    document: ParsedDocument,
    *,
    chunk_tokens: int,
    chunk_overlap: int,
    source_group: str,
    token_counter: TokenCounter | None = None,
    max_input_tokens: int | None = None,
) -> tuple[DocumentChunk, ...]:
    text = document.text.strip()
    if not text:
        return ()
    counter = token_counter or approximate_token_count
    token_limit = min(chunk_tokens, max_input_tokens or chunk_tokens)
    record_type = str(document.metadata.get("record_type", "document"))
    if record_type == "atomic_claim":
        token_count = counter(text)
        if token_count > token_limit:
            claim_id = document.metadata.get("claim_id", document.document_id)
            raise ValueError(
                f"Atomic claim {claim_id!r} uses {token_count} tokens, "
                f"exceeding the model-safe limit {token_limit}"
            )
        return (
            DocumentChunk(
                chunk_id=_chunk_id(document, 0, len(text), text),
                document_id=document.document_id,
                text=text,
                section_path=(document.title,),
                start_offset=0,
                end_offset=len(text),
                token_count=token_count,
                source_group=source_group,
                artifact_uri=str(document.path),
                file_hash=document.file_hash,
                knowledge_type=document.knowledge_type,
                metadata={**document.metadata, "chunker_version": CHUNKER_VERSION},
            ),
        )

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(text):
        end = _largest_fitting_end(text, start, counter, token_limit)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=_chunk_id(document, start, end, chunk_text),
                    document_id=document.document_id,
                    text=chunk_text,
                    section_path=_section_path(text, start, document.title),
                    start_offset=start,
                    end_offset=end,
                    token_count=counter(chunk_text),
                    source_group=source_group,
                    artifact_uri=str(document.path),
                    file_hash=document.file_hash,
                    knowledge_type=document.knowledge_type,
                    metadata={**document.metadata, "chunker_version": CHUNKER_VERSION},
                )
            )
        if end >= len(text):
            break
        next_start = _overlap_start(
            text,
            start=start,
            end=end,
            counter=counter,
            overlap_tokens=chunk_overlap,
        )
        start = max(start + 1, next_start)
    return tuple(chunks)
