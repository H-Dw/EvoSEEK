from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

CLAIM_SCHEMA = "scientific-atomic-claim:v1"
PUBLICATION_SCHEMA = "scientific-publications:v1"
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


def load_publications(path: Path, errors: list[str]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        errors.append(f"missing publication catalog: {path}")
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict) or payload.get("schema_version") != PUBLICATION_SCHEMA:
        errors.append(f"unsupported publication catalog schema: {path}")
        return {}
    publications: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(payload.get("publications", [])):
        if not isinstance(raw, dict):
            errors.append(f"publication[{index}] must be a mapping")
            continue
        publication_id = str(raw.get("publication_id", "")).strip().casefold()
        if not DOI_ID.fullmatch(publication_id):
            errors.append(f"invalid publication_id: {publication_id!r}")
            continue
        required = ("title", "authors", "year", "venue", "doi", "url", "verification")
        missing = [key for key in required if raw.get(key) in (None, "", [])]
        if missing:
            errors.append(f"publication {publication_id} missing: {', '.join(missing)}")
        doi = str(raw.get("doi", "")).strip().casefold()
        if publication_id != f"doi:{doi}":
            errors.append(f"publication ID/DOI mismatch: {publication_id}")
        if publication_id in publications:
            errors.append(f"duplicate publication_id: {publication_id}")
        publications[publication_id] = dict(raw)
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate an English atomic knowledge bundle")
    parser.add_argument("bundle_root", type=Path)
    parser.add_argument("--embedding-model", type=Path)
    args = parser.parse_args()

    root = args.bundle_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    publications = load_publications(root / "catalog" / "publications.yaml", errors)
    token_counter, model_limit = model_token_counter(args.embedding_model)
    claim_ids: set[str] = set()
    support_ids: set[str] = set()
    knowledge_types: Counter[str] = Counter()
    token_counts: list[int] = []
    verified_support = 0
    unverified_support = 0
    files = tuple(sorted((root / "claims").glob("**/*.md")))
    if not files:
        errors.append(f"no claim files found below {root / 'claims'}")

    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            metadata, body = split_front_matter(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, TypeError, ValueError, yaml.YAMLError) as error:
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
        )
        missing = [key for key in required if key not in metadata]
        if missing:
            errors.append(f"{relative}: missing fields: {', '.join(missing)}")
            continue
        if metadata["schema_version"] != CLAIM_SCHEMA or metadata["record_type"] != "atomic_claim":
            errors.append(f"{relative}: unsupported claim schema or record type")
        if metadata["language"] != "en" or CJK.search(path.read_text(encoding="utf-8")):
            errors.append(f"{relative}: claim must be English-only")
        knowledge_type = str(metadata["knowledge_type"])
        if not KNOWLEDGE_TYPE.fullmatch(knowledge_type):
            errors.append(f"{relative}: invalid knowledge_type {knowledge_type!r}")
        knowledge_types[knowledge_type] += 1
        claim_id = str(metadata["claim_id"])
        if claim_id in claim_ids:
            errors.append(f"{relative}: duplicate claim_id {claim_id}")
        claim_ids.add(claim_id)
        statement = " ".join(str(metadata["statement"]).split())
        if " ".join(body.split()) != statement:
            errors.append(f"{relative}: body must equal the atomic statement")
        lowered = body.casefold()
        markers = [item for item in INSTRUCTION_MARKERS if item in lowered]
        if markers:
            errors.append(f"{relative}: instruction-like content: {markers}")
        try:
            confidence = float(metadata["confidence"])
        except (TypeError, ValueError):
            errors.append(f"{relative}: confidence must be numeric")
        else:
            if not 0.0 <= confidence <= 1.0:
                errors.append(f"{relative}: confidence must be in [0, 1]")
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
                errors.append(f"{relative}: missing or duplicate support_id {support_id!r}")
            support_ids.add(support_id)
            publication_id = str(support.get("publication_id", "")).casefold()
            if publication_id not in publications:
                errors.append(f"{relative}: unknown publication {publication_id!r}")
            if not support.get("locator") or not support.get("support_type"):
                errors.append(f"{relative}: citation support requires locator and support_type")
            if bool(support.get("verified_against_source", False)):
                verified_support += 1
            else:
                unverified_support += 1
        if token_counter is not None and model_limit is not None:
            count = int(token_counter(body))
            token_counts.append(count)
            if count > model_limit:
                errors.append(
                    f"{relative}: {count} model tokens exceed model limit {model_limit}"
                )

    if unverified_support:
        warnings.append(f"{unverified_support} citation supports are not full-source verified")
    report = {
        "bundle_root": str(root),
        "valid": not errors,
        "claim_count": len(files),
        "publication_count": len(publications),
        "citation_support_count": len(support_ids),
        "verified_citation_support_count": verified_support,
        "unverified_citation_support_count": unverified_support,
        "knowledge_types": dict(sorted(knowledge_types.items())),
        "model_token_check": {
            "performed": token_counter is not None,
            "model_path": str(args.embedding_model.resolve()) if args.embedding_model else None,
            "model_max_tokens": model_limit,
            "maximum_claim_tokens": max(token_counts, default=None),
        },
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
