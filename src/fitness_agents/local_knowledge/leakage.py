from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fitness_agents.config import LeakageGuardConfig, TaskConfig

POLICY_VERSION = "target-leakage-guard:v1"


def normalize_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^\w\u3400-\u9fff]+", " ", normalized).strip()


@dataclass(frozen=True)
class LeakageDecision:
    allowed: bool
    sanitized_query: str
    generalized: bool
    matched_categories: tuple[str, ...]
    policy_version: str = POLICY_VERSION

    def public_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "generalized": self.generalized,
            "matched_categories": self.matched_categories,
            "policy_version": self.policy_version,
        }


class TargetLeakageGuard:
    def __init__(
        self,
        config: LeakageGuardConfig,
        *,
        protein_name: str | None,
        protein_id: str,
        aliases: tuple[str, ...],
        accessions: tuple[str, ...],
        reference_sequence: str | None,
    ) -> None:
        self.config = config
        task_aliases = tuple(aliases) if config.derive_protected_terms_from_task else ()
        task_accessions = tuple(accessions) if config.derive_protected_terms_from_task else ()
        explicit_aliases = tuple(config.protected_aliases) + task_aliases
        explicit_accessions = tuple(config.protected_accessions) + task_accessions
        derived_name = protein_name if config.derive_protected_terms_from_task else None
        derived_id = protein_id if config.derive_protected_terms_from_task else ""
        if (
            config.enabled
            and config.strict_aliases_required
            and not derived_name
            and not explicit_aliases
            and not explicit_accessions
        ):
            raise ValueError(
                "Strict local-knowledge leakage protection requires protein_name, aliases, "
                "or accessions in the task/config"
            )
        terms = [derived_id, derived_name or "", *explicit_aliases, *explicit_accessions]
        self.protected_terms = tuple(
            sorted({item for value in terms if (item := normalize_identity(str(value)))})
        )
        sequence = re.sub(
            r"[^A-Za-z]",
            "",
            (reference_sequence or "")
            if config.derive_protected_terms_from_task
            else "",
        ).upper()
        fragments: set[str] = set()
        minimum = config.minimum_sequence_fragment_length
        if sequence:
            fragments.add(sequence)
            if len(sequence) >= minimum:
                for index in range(0, len(sequence) - minimum + 1, max(1, minimum // 2)):
                    fragments.add(sequence[index : index + minimum])
        self.sequence_fragments = tuple(sorted(fragments, key=lambda item: (-len(item), item)))
        payload = json.dumps(
            [
                {
                    "enabled": config.enabled,
                    "derive_from_task": config.derive_protected_terms_from_task,
                    "quarantine_target_documents": config.quarantine_target_documents,
                    "block_target_entities": config.block_target_entities,
                    "minimum_sequence_fragment_length": (
                        config.minimum_sequence_fragment_length
                    ),
                },
                self.protected_terms,
                self.sequence_fragments,
            ],
            separators=(",", ":"),
        )
        self.protected_terms_hash = hashlib.sha256(payload.encode()).hexdigest()

    @classmethod
    def from_task(
        cls, config: LeakageGuardConfig, task: TaskConfig, *, reference_sequence: str | None
    ) -> TargetLeakageGuard:
        return cls(
            config,
            protein_name=task.protein_name,
            protein_id=task.protein_id,
            aliases=task.protein_aliases,
            accessions=task.protein_accessions,
            reference_sequence=reference_sequence,
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def matches(self, *, text: str, path: str | Path | None = None) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        haystack = normalize_identity(" ".join(item for item in (str(path or ""), text) if item))
        categories: set[str] = set()
        for term in self.protected_terms:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", haystack):
                categories.add("protected_term")
        sequence_text = re.sub(r"[^A-Za-z]", "", text).upper()
        if any(fragment in sequence_text for fragment in self.sequence_fragments):
            categories.add("protected_sequence")
        return tuple(sorted(categories))

    def sanitize_query(self, query: str, *, generic_terms: tuple[str, ...]) -> LeakageDecision:
        if not self.enabled:
            return LeakageDecision(True, query.strip(), False, ())
        matched = self.matches(text=query)
        if not matched:
            return LeakageDecision(True, query.strip(), False, ())
        safe_terms = tuple(
            term for term in generic_terms if not self.matches(text=term) and normalize_identity(term)
        )
        if not safe_terms:
            return LeakageDecision(False, "", True, matched)
        sanitized = "general protein property knowledge: " + "; ".join(safe_terms)
        if self.matches(text=sanitized):
            return LeakageDecision(False, "", True, matched)
        return LeakageDecision(True, sanitized, True, matched)

    def validate_result(self, *, text: str, path: str | Path) -> tuple[str, ...]:
        return self.matches(text=text, path=path)
