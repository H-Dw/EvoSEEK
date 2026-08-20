"""Fail-closed context partitioning for channel-isolated child Scientists."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import asdict, replace
from typing import Any

from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelEvidenceInput,
    ChannelName,
    DescriptorObservationFact,
    MainSynthesisEvidenceCard,
)
from fitness_agents.contracts.mutation_evidence import mutation_evidence_prompt_payload
from fitness_agents.contracts.schemas import Evidence
from fitness_agents.kg_interaction.contracts import EvidencePack, InteractionResult

from .llm import _compact_prompt_pack, _prompt_row_identities

FEATURE_CHANNELS: tuple[ChannelName, ...] = ("physchem", "conservation", "structure")
FEATURE_OPERATOR_CHANNEL: dict[str, ChannelName] = {
    "query_physchem_delta": "physchem",
    "query_evolutionary_profile": "conservation",
    "query_structure_environment": "structure",
}
FEATURE_OPERATORS = frozenset({*FEATURE_OPERATOR_CHANNEL, "query_feature_bundle"})
_MUTATION_TOKEN_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _descriptor_observation_facts(
    *,
    sample_id: str,
    residues_by_position: dict[str, str],
    evidence_payloads: list[dict[str, Any]],
    mutable_positions: tuple[int, ...],
    wild_type_sites: str,
) -> tuple[DescriptorObservationFact, ...]:
    wild_type_by_position = {
        position: wild_type_sites[index]
        for index, position in enumerate(mutable_positions)
    }
    facts: dict[str, DescriptorObservationFact] = {}
    for payload in evidence_payloads:
        evidence_id = str(payload.get("evidence_id") or "")
        features = dict(payload.get("features") or {})
        if not evidence_id or features.get("kind") != "physchem":
            continue
        for raw_site in features.get("sites", ()):
            site = dict(raw_site)
            try:
                position = int(site.get("position"))
            except (TypeError, ValueError):
                continue
            from_residue = wild_type_by_position.get(position)
            to_residue = residues_by_position.get(str(position))
            mutation = str(site.get("mutation") or "")
            match = _MUTATION_TOKEN_RE.fullmatch(mutation)
            if match is not None and int(match.group(2)) == position:
                from_residue = match.group(1)
                to_residue = match.group(3)
            if from_residue is None or to_residue is None:
                continue
            for raw_delta in site.get("deltas", ()):
                delta = dict(raw_delta)
                descriptor = str(delta.get("name") or "")
                try:
                    value = float(delta.get("value"))
                except (TypeError, ValueError):
                    continue
                if not descriptor:
                    continue
                digest = hashlib.sha256(
                    (
                        f"{evidence_id}|{sample_id}|{position}|{from_residue}|"
                        f"{to_residue}|{descriptor}"
                    ).encode()
                ).hexdigest()[:24]
                fact = DescriptorObservationFact(
                    fact_id=f"fact:descriptor:{digest}",
                    evidence_id=evidence_id,
                    sample_id=sample_id,
                    position=position,
                    from_residue=from_residue,
                    to_residue=to_residue,
                    descriptor=descriptor,
                    delta=value,
                )
                facts[fact.fact_id] = fact
    return tuple(facts[key] for key in sorted(facts))


def _dedupe_dicts(items: tuple[dict[str, Any], ...], *, keys: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for item in items:
        identity = next((str(item[key]) for key in keys if item.get(key)), canonical_sha256(item))
        if identity in seen:
            continue
        seen.add(identity)
        output.append(item)
    return tuple(output)


def _channel_items(
    items: tuple[dict[str, Any], ...],
    *,
    channel: ChannelName,
    allow_untagged: bool,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        item
        for item in items
        if item.get("channel") == channel or (allow_untagged and item.get("channel") is None)
    )


class KGContextPartitioner:
    """Project raw KG output into three non-overlapping role-visible contexts."""

    def split_packs(
        self, interaction: InteractionResult | None
    ) -> tuple[dict[ChannelName, tuple[EvidencePack, ...]], InteractionResult | None]:
        channel_packs: dict[ChannelName, list[EvidencePack]] = {
            channel: [] for channel in FEATURE_CHANNELS
        }
        if interaction is None:
            return {key: () for key in FEATURE_CHANNELS}, None
        base_packs: list[EvidencePack] = []
        for pack in interaction.packs:
            if pack.operator not in FEATURE_OPERATORS:
                base_packs.append(pack)
                continue
            targets: tuple[ChannelName, ...]
            if pack.operator in FEATURE_OPERATOR_CHANNEL:
                targets = (FEATURE_OPERATOR_CHANNEL[pack.operator],)
            else:
                targets = FEATURE_CHANNELS
            for channel in targets:
                allow_untagged = pack.operator in FEATURE_OPERATOR_CHANNEL
                evidence = _dedupe_dicts(
                    _channel_items(pack.evidence, channel=channel, allow_untagged=allow_untagged),
                    keys=("evidence_id", "claim_id"),
                )
                facts = _channel_items(
                    pack.facts, channel=channel, allow_untagged=allow_untagged
                )
                predictions = _channel_items(
                    pack.predictions, channel=channel, allow_untagged=allow_untagged
                )
                counterevidence = _channel_items(
                    pack.counterevidence, channel=channel, allow_untagged=allow_untagged
                )
                signals = _channel_items(
                    pack.directional_signals,
                    channel=channel,
                    allow_untagged=allow_untagged,
                )
                visible_ids = {str(item.get("evidence_id")) for item in evidence}
                provenance = tuple(
                    item
                    for item in pack.provenance
                    if not item.get("evidence_id")
                    or str(item.get("evidence_id")) in visible_ids
                )
                if not any((evidence, facts, predictions, counterevidence, signals)):
                    continue
                channel_packs[channel].append(
                    replace(
                        pack,
                        query_id=f"{pack.query_id}:{channel}",
                        facts=facts,
                        predictions=predictions,
                        evidence=evidence,
                        counterevidence=counterevidence,
                        directional_signals=signals,
                        provenance=provenance,
                        metadata={
                            key: value
                            for key, value in pack.metadata.items()
                            if key not in {"channels", "channel_payloads"}
                        }
                        | {"projected_channel": channel},
                    )
                )
        deduplicated: dict[ChannelName, tuple[EvidencePack, ...]] = {}
        for channel, packs in channel_packs.items():
            seen_evidence: set[str] = set()
            seen_facts: set[str] = set()
            projected: list[EvidencePack] = []
            for pack in packs:
                evidence = tuple(
                    item
                    for item in pack.evidence
                    if not item.get("evidence_id")
                    or str(item["evidence_id"]) not in seen_evidence
                )
                seen_evidence.update(
                    str(item["evidence_id"])
                    for item in evidence
                    if item.get("evidence_id")
                )
                facts = tuple(
                    item
                    for item in pack.facts
                    if canonical_sha256(item) not in seen_facts
                )
                seen_facts.update(canonical_sha256(item) for item in facts)
                projected.append(replace(pack, evidence=evidence, facts=facts))
            deduplicated[channel] = tuple(projected)
        base = replace(interaction, packs=tuple(base_packs))
        return deduplicated, base

    def split_evidence(
        self, evidence: tuple[Evidence, ...] | list[Evidence]
    ) -> tuple[dict[ChannelName, tuple[Evidence, ...]], tuple[Evidence, ...]]:
        seen: set[str] = set()
        channel_items: dict[ChannelName, list[Evidence]] = {
            channel: [] for channel in FEATURE_CHANNELS
        }
        base: list[Evidence] = []
        for item in evidence:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            if item.channel in channel_items:
                channel_items[item.channel].append(item)  # type: ignore[index]
            else:
                base.append(item)
        return {key: tuple(value) for key, value in channel_items.items()}, tuple(base)

    def child_context(
        self,
        *,
        base_context: ScientistContextInput | dict[str, Any],
        channel: ChannelName,
        evidence: tuple[Evidence, ...],
        packs: tuple[EvidencePack, ...],
        retry_control: dict[str, Any] | None = None,
    ) -> ChannelEvidenceInput:
        context = ScientistContextInput.model_validate(base_context)
        seen_identities: set[tuple[str, str]] = set()
        evidence_payloads = []
        for item in evidence:
            payload = mutation_evidence_prompt_payload(item)
            identities = _prompt_row_identities(payload)
            if any(identity in seen_identities for identity in identities):
                continue
            evidence_payloads.append(payload)
            seen_identities.update(identities)
        pack_payloads = []
        for pack in packs:
            pack_payloads.append(
                _compact_prompt_pack(asdict(pack), seen_identities=seen_identities)
            )
        evidence_by_variant: dict[str, list[dict[str, Any]]] = {}
        for payload in evidence_payloads:
            evidence_by_variant.setdefault(str(payload.get("variant_id") or ""), []).append(
                payload
            )
        sample_cards = []
        for raw_observation in context.visible_observations:
            observation = dict(raw_observation)
            variant_id = str(observation.get("variant_id") or "")
            matching = evidence_by_variant.get(variant_id, [])
            sample_id = str(observation.get("sample_id") or variant_id)
            residues_by_position = {
                str(key): str(value)
                for key, value in dict(
                    observation.get("residues_by_position") or {}
                ).items()
            }
            sequence_sha256 = str(observation.get("sequence_sha256") or "")
            if len(sequence_sha256) != 64:
                sequence_sha256 = hashlib.sha256(
                    str(observation.get("variant") or variant_id).encode()
                ).hexdigest()
            sample_cards.append(
                {
                    "sample_id": sample_id,
                    "variant_id": variant_id,
                    "mutation_notation": str(
                        observation.get("mutation_notation") or "WT"
                    ),
                    "sequence_sha256": sequence_sha256,
                    "residues_by_position": residues_by_position,
                    "evidence_ids": tuple(
                        sorted(
                            str(item["evidence_id"])
                            for item in matching
                            if item.get("evidence_id")
                        )
                    ),
                    "feature_values": {
                        str(item["evidence_id"]): dict(item.get("features") or {})
                        for item in matching
                        if item.get("evidence_id")
                    },
                    "descriptor_facts": _descriptor_observation_facts(
                        sample_id=sample_id,
                        residues_by_position=residues_by_position,
                        evidence_payloads=matching,
                        mutable_positions=context.mutable_positions,
                        wild_type_sites=context.wild_type_sites,
                    ),
                }
            )
        return ChannelEvidenceInput(
            run_id=context.run_id,
            round_id=context.round_id,
            channel=channel,
            task=context.task,
            mutable_positions=context.mutable_positions,
            wild_type_sites=context.wild_type_sites,
            visible_observations=tuple(sample_cards),
            evidence=tuple(evidence_payloads),
            kg_packs=tuple(pack_payloads),
            retry_control=retry_control,
        )


def main_context_payload(
    approved: tuple[Any, ...], conflicts: tuple[Any, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved_payload = [approved_analysis_payload(item) for item in approved]
    conflict_payload = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in conflicts
    ]
    return approved_payload, conflict_payload


def approved_analysis_payload(item: Any) -> dict[str, Any]:
    """Project an approved branch to the only fields needed for main-role fusion."""

    analysis = getattr(item, "hypothesis", None)
    review = getattr(item, "review", None)
    if analysis is None or review is None:
        raise TypeError("approved analysis projection requires hypothesis and review")
    contribution_modes = []
    if analysis.candidate_hypotheses:
        contribution_modes.append("support")
    if analysis.counterevidence or any(
        finding.kind == "LIMITATION" for finding in analysis.findings
    ):
        contribution_modes.append("constraint_counterevidence")
    if not analysis.candidate_hypotheses or any(
        finding.kind in {"OBSERVATION", "INTERPRETATION"}
        for finding in analysis.findings
    ):
        contribution_modes.append("analysis_only")
    return {
        "channel": str(item.channel),
        "contribution_modes": list(dict.fromkeys(contribution_modes)),
        "analysis": analysis.model_dump(mode="json"),
        "semantic_review": {
            "verdict": str(review.verdict),
            "summary": str(review.summary),
            "cited_evidence_ids": list(review.cited_evidence_ids),
        },
    }


def _raw_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def main_synthesis_evidence_cards(
    *,
    evidence: Iterable[Evidence],
    interaction: InteractionResult | None,
    approved: tuple[Any, ...],
) -> tuple[MainSynthesisEvidenceCard, ...]:
    """Build one atomic, typed evidence universe for main synthesis review."""

    candidate_ids = {
        evidence_id
        for item in approved
        for candidate in item.hypothesis.candidate_hypotheses
        for evidence_id in candidate.evidence_ids
    }
    limitation_ids = {
        evidence_id
        for item in approved
        for finding in item.hypothesis.findings
        if finding.kind == "LIMITATION"
        for evidence_id in finding.evidence_ids
    }
    raw_items = [_raw_mapping(item) for item in evidence]
    if interaction is not None:
        raw_items.extend(
            dict(item)
            for pack in interaction.packs
            for item in pack.evidence
            if isinstance(item, dict)
        )
    by_id: dict[str, MainSynthesisEvidenceCard] = {}
    for raw in raw_items:
        evidence_id = str(raw.get("evidence_id") or "")
        if not evidence_id or evidence_id in by_id:
            continue
        polarity = str(raw.get("polarity") or "neutral").casefold()
        if polarity not in {"support", "contradict", "neutral", "unknown"}:
            polarity = "unknown"
        if evidence_id in limitation_ids or polarity == "contradict":
            contribution = "constraint_counterevidence"
        elif evidence_id in candidate_ids or polarity == "support":
            contribution = "support"
        else:
            contribution = "analysis_only"
        confidence = raw.get("confidence", 0.0)
        try:
            confidence = min(1.0, max(0.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0
        raw_span = raw.get("artifact_span") or raw.get("source_span")
        source_span = None
        if isinstance(raw_span, (list, tuple)) and len(raw_span) == 2:
            source_span = (int(raw_span[0]), int(raw_span[1]))
        warnings = raw.get("warnings") or ()
        by_id[evidence_id] = MainSynthesisEvidenceCard(
            evidence_id=evidence_id,
            atomic_statement=str(
                raw.get("statement") or raw.get("claim") or "Evidence statement unavailable."
            )[:600],
            channel=str(raw.get("channel") or "unknown")[:120],
            contribution=contribution,
            polarity=polarity,
            applicability=str(raw.get("applicability") or "unknown")[:240],
            confidence=confidence,
            quality_status=str(raw.get("quality_status") or "unknown")[:120],
            warnings=tuple(str(item)[:240] for item in warnings)[:8],
            source_uri=(
                str(raw.get("artifact_uri") or raw.get("source_uri"))[:1200]
                if raw.get("artifact_uri") or raw.get("source_uri")
                else None
            ),
            source_span=source_span,
        )
    return tuple(by_id[key] for key in sorted(by_id))


def select_main_review_evidence_cards(
    hypothesis: Any,
    cards: tuple[MainSynthesisEvidenceCard, ...],
    *,
    limit: int = 12,
) -> tuple[MainSynthesisEvidenceCard, ...]:
    """Keep cited evidence first, then the most relevant visible counterevidence."""

    by_id = {item.evidence_id: item for item in cards}
    cited = tuple(dict.fromkeys(getattr(hypothesis, "evidence_ids", ()) or ()))
    selected = [by_id[item] for item in cited if item in by_id][:limit]
    selected_ids = {item.evidence_id for item in selected}
    counterevidence = sorted(
        (
            item
            for item in cards
            if item.evidence_id not in selected_ids
            and item.contribution == "constraint_counterevidence"
        ),
        key=lambda item: (item.confidence, item.evidence_id),
        reverse=True,
    )
    selected.extend(counterevidence[: max(0, limit - len(selected))])
    return tuple(selected)
