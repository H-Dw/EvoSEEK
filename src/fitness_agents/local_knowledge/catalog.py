from __future__ import annotations

import re
from typing import Any

import yaml

from fitness_agents.config import LocalKnowledgeRootConfig

DOI_PATTERN = re.compile(r"^doi:10\.\d{4,9}/\S+$", re.IGNORECASE)


class PublicationCatalog:
    """Normalized publication records referenced by atomic local-knowledge claims."""

    def __init__(self, publications: dict[str, dict[str, Any]]) -> None:
        self.publications = publications

    @classmethod
    def from_roots(cls, roots: tuple[LocalKnowledgeRootConfig, ...]) -> PublicationCatalog:
        publications: dict[str, dict[str, Any]] = {}
        for root in roots:
            path = root.path / "catalog" / "publications.yaml"
            if not path.is_file():
                continue
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if payload.get("schema_version") != "scientific-publications:v1":
                raise ValueError(f"Unsupported publication catalog schema: {path}")
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
