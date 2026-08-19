"""Fail-closed context partitioning for channel-isolated child Scientists."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any

from fitness_agents.contracts.agent_io import ScientistContextInput
from fitness_agents.contracts.hypothesis_pipeline import ChannelEvidenceInput, ChannelName
from fitness_agents.contracts.schemas import Evidence
from fitness_agents.kg_interaction.contracts import EvidencePack, InteractionResult

from .llm import _compact_prompt_evidence

FEATURE_CHANNELS: tuple[ChannelName, ...] = ("physchem", "conservation", "structure")
FEATURE_OPERATOR_CHANNEL: dict[str, ChannelName] = {
    "query_physchem_delta": "physchem",
    "query_evolutionary_profile": "conservation",
    "query_structure_environment": "structure",
}
FEATURE_OPERATORS = frozenset({*FEATURE_OPERATOR_CHANNEL, "query_feature_bundle"})


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


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
        pack_payloads = []
        pack_evidence_ids: set[str] = set()
        for pack in packs:
            payload = asdict(pack)
            payload["evidence"] = [
                _compact_prompt_evidence(item) for item in pack.evidence
            ]
            pack_evidence_ids.update(
                str(item["evidence_id"])
                for item in payload["evidence"]
                if item.get("evidence_id")
            )
            pack_payloads.append(payload)
        return ChannelEvidenceInput(
            run_id=context.run_id,
            round_id=context.round_id,
            channel=channel,
            task=context.task,
            mutable_positions=context.mutable_positions,
            wild_type_sites=context.wild_type_sites,
            visible_observations=tuple(context.visible_observations),
            evidence=tuple(
                _compact_prompt_evidence(item)
                for item in evidence
                if item.evidence_id not in pack_evidence_ids
            ),
            kg_packs=tuple(pack_payloads),
            retry_control=retry_control,
        )


def main_context_payload(
    approved: tuple[Any, ...], conflicts: tuple[Any, ...]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    approved_payload = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in approved
    ]
    conflict_payload = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in conflicts
    ]
    return approved_payload, conflict_payload
