"""Bounded, typed evidence projection exposed to mutation-review LLM roles.

The canonical :class:`Evidence` objects remain unchanged in SQLite and run
artifacts.  This module only defines the smaller role-visible representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import Evidence


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NamedNumericFeature(_FrozenModel):
    name: str = Field(min_length=1, max_length=120)
    value: float


class EvidenceSourceCard(_FrozenModel):
    source_id: str = Field(min_length=1, max_length=320)
    claim_id: str | None = Field(default=None, max_length=320)
    publication_id: str | None = Field(default=None, max_length=320)
    doi: str | None = Field(default=None, max_length=320)
    artifact_uri: str | None = Field(default=None, max_length=1200)
    artifact_span: tuple[int, int] | None = None
    section_path: tuple[str, ...] = Field(default=(), max_length=12)
    source_group: str = Field(default="unknown", max_length=160)


class PhyschemSiteCard(_FrozenModel):
    position: int
    mutation: str | None = Field(default=None, max_length=32)
    deltas: tuple[NamedNumericFeature, ...] = Field(default=(), max_length=8)


class PhyschemEvidenceFeatures(_FrozenModel):
    kind: Literal["physchem"] = "physchem"
    sites: tuple[PhyschemSiteCard, ...] = Field(default=(), max_length=16)
    mean_normalized_absolute_delta: float | None = None
    special_flags: tuple[str, ...] = Field(default=(), max_length=16)
    global_sequence_deltas: tuple[NamedNumericFeature, ...] = Field(
        default=(), max_length=12
    )


class ConservationSiteCard(_FrozenModel):
    position: int
    mutation: str | None = Field(default=None, max_length=32)
    coverage: float | None = None
    gap_fraction: float | None = None
    effective_count: float | None = None
    mutant_frequency: float | None = None
    log_odds_vs_wild_type: float | None = None
    site_quality: str | None = Field(default=None, max_length=80)


class ConservationEvidenceFeatures(_FrozenModel):
    kind: Literal["conservation"] = "conservation"
    sites: tuple[ConservationSiteCard, ...] = Field(default=(), max_length=16)
    independent_log_odds: float | None = None
    independent_log_odds_sum: float | None = None
    independent_mean_log_odds_per_mutation: float | None = None
    pairwise_enabled: bool | None = None
    pairwise_eligible: bool | None = None
    pairwise_score: float | None = None


class StructureSiteCard(_FrozenModel):
    position: int
    mutation: str | None = Field(default=None, max_length=32)
    status: str | None = Field(default=None, max_length=80)
    relative_sasa: float | None = None
    contact_count: int | None = None
    secondary_structure: str | None = Field(default=None, max_length=80)
    missing_backbone_atoms: tuple[str, ...] = Field(default=(), max_length=8)
    missing_coordinates: bool = False
    mutant_side_chain_not_modelled: bool | None = None


class StructureEvidenceFeatures(_FrozenModel):
    kind: Literal["structure"] = "structure"
    sites: tuple[StructureSiteCard, ...] = Field(default=(), max_length=16)
    static_context_flag_count: int | None = None


class KGEvidenceFeatures(_FrozenModel):
    kind: Literal["kg"] = "kg"
    raw_association_score: float | None = None
    support: int | None = None
    global_visible_mean: float | None = None


class RAGEvidenceFeatures(_FrozenModel):
    kind: Literal["rag"] = "rag"
    knowledge_type: str | None = Field(default=None, max_length=120)
    retrieval_scores: tuple[NamedNumericFeature, ...] = Field(default=(), max_length=12)


class GenericEvidenceFeatures(_FrozenModel):
    kind: Literal["generic"] = "generic"


MutationEvidenceFeatures = Annotated[
    PhyschemEvidenceFeatures
    | ConservationEvidenceFeatures
    | StructureEvidenceFeatures
    | KGEvidenceFeatures
    | RAGEvidenceFeatures
    | GenericEvidenceFeatures,
    Field(discriminator="kind"),
]


class MutationEvidenceCard(_FrozenModel):
    """One mutation-relevant evidence claim with bounded channel-specific details."""

    schema_version: Literal["mutation_evidence_card.v1"] = "mutation_evidence_card.v1"
    evidence_id: str = Field(min_length=1, max_length=320)
    variant_id: str = Field(min_length=1, max_length=320)
    channel: str = Field(min_length=1, max_length=120)
    evidence_type: str = Field(min_length=1, max_length=120)
    statement: str = Field(min_length=1, max_length=1200)
    score: float
    confidence: float
    quality_status: str = Field(max_length=120)
    applicability: str | None = Field(default=None, max_length=240)
    contributes_to_selection: bool | None = None
    polarity: str | None = Field(default=None, max_length=40)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)
    source: EvidenceSourceCard
    features: MutationEvidenceFeatures


class ConservationBatchMetadata(_FrozenModel):
    sequence_count: int | None = None
    neff: float | None = None
    neff_per_length: float | None = None
    pairwise_enabled: bool | None = None
    pairwise_eligible: bool | None = None
    pairwise_score_method: str | None = Field(default=None, max_length=120)


class ChannelEvidenceSharedMetadata(_FrozenModel):
    channel: str = Field(min_length=1, max_length=120)
    warnings: tuple[str, ...] = Field(default=(), max_length=16)
    source_id: str | None = Field(default=None, max_length=320)


class MutationEvidenceBatchMetadata(_FrozenModel):
    """Candidate-invariant feature metadata hoisted out of repeated evidence cards."""

    schema_version: Literal["mutation_evidence_batch_metadata.v1"] = (
        "mutation_evidence_batch_metadata.v1"
    )
    assay_pH_values: tuple[float, ...] = Field(default=(), max_length=8)
    conservation_profiles: tuple[ConservationBatchMetadata, ...] = Field(
        default=(), max_length=8
    )
    structure_resource_ids: tuple[str, ...] = Field(default=(), max_length=8)
    channel_shared: tuple[ChannelEvidenceSharedMetadata, ...] = Field(
        default=(), max_length=16
    )


def _raw(value: Evidence | Mapping[str, Any]) -> dict[str, Any]:
    return dict(value.__dict__) if isinstance(value, Evidence) else dict(value)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _position(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strings(value: Any, *, limit: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set)):
        return ()
    return tuple(str(item)[:240] for item in value if str(item))[:limit]


def _numeric_features(
    value: Any,
    *,
    allowed_names: frozenset[str] | None = None,
    allowed_markers: tuple[str, ...] = (),
    limit: int = 12,
) -> tuple[NamedNumericFeature, ...]:
    output: list[NamedNumericFeature] = []
    for name, raw_value in _mapping(value).items():
        key = str(name)
        normalized = key.casefold()
        if allowed_names is not None and key not in allowed_names:
            continue
        if allowed_markers and not any(marker in normalized for marker in allowed_markers):
            continue
        number = _optional_float(raw_value)
        if number is None:
            continue
        output.append(NamedNumericFeature(name=key, value=number))
        if len(output) >= limit:
            break
    return tuple(output)


def _source_card(raw: dict[str, Any]) -> EvidenceSourceCard:
    provenance = _mapping(raw.get("provenance"))
    artifact_span = raw.get("artifact_span") or provenance.get("artifact_span")
    normalized_span = None
    if isinstance(artifact_span, (list, tuple)) and len(artifact_span) == 2:
        start = _optional_int(artifact_span[0])
        end = _optional_int(artifact_span[1])
        if start is not None and end is not None and 0 <= start <= end:
            normalized_span = (start, end)
    section_path = provenance.get("section_path")
    return EvidenceSourceCard(
        source_id=str(raw.get("source_id") or "unknown")[:320],
        claim_id=(str(raw["claim_id"])[:320] if raw.get("claim_id") else None),
        publication_id=(
            str(provenance["publication_id"])[:320]
            if provenance.get("publication_id")
            else None
        ),
        doi=str(provenance["doi"])[:320] if provenance.get("doi") else None,
        artifact_uri=(
            str(raw.get("artifact_uri") or provenance.get("artifact_uri"))[:1200]
            if raw.get("artifact_uri") or provenance.get("artifact_uri")
            else None
        ),
        artifact_span=normalized_span,
        section_path=_strings(section_path, limit=12),
        source_group=str(raw.get("source_group") or "unknown")[:160],
    )


def _physchem_features(raw_features: dict[str, Any]) -> PhyschemEvidenceFeatures:
    sites: list[PhyschemSiteCard] = []
    for raw_position, raw_site in _mapping(raw_features.get("sites")).items():
        position = _position(raw_position)
        if position is None:
            continue
        site = _mapping(raw_site)
        sites.append(
            PhyschemSiteCard(
                position=position,
                mutation=str(site["mutation"])[:32] if site.get("mutation") else None,
                deltas=_numeric_features(
                    site.get("deltas"),
                    allowed_markers=("mass", "hydropathy", "charge", "volume"),
                    limit=8,
                ),
            )
        )
    return PhyschemEvidenceFeatures(
        sites=tuple(sites[:16]),
        mean_normalized_absolute_delta=_optional_float(
            raw_features.get("mean_normalized_absolute_delta")
        ),
        special_flags=_strings(raw_features.get("special_flags"), limit=16),
        global_sequence_deltas=_numeric_features(
            raw_features.get("global_sequence_deltas"), limit=12
        ),
    )


def _conservation_features(
    raw_features: dict[str, Any],
) -> ConservationEvidenceFeatures:
    sites: list[ConservationSiteCard] = []
    for raw_position, raw_site in _mapping(raw_features.get("sites")).items():
        position = _position(raw_position)
        if position is None:
            continue
        site = _mapping(raw_site)
        sites.append(
            ConservationSiteCard(
                position=position,
                mutation=str(site["mutation"])[:32] if site.get("mutation") else None,
                coverage=_optional_float(site.get("coverage")),
                gap_fraction=_optional_float(site.get("gap_fraction")),
                effective_count=_optional_float(site.get("effective_count")),
                mutant_frequency=_optional_float(site.get("mutant_frequency")),
                log_odds_vs_wild_type=_optional_float(site.get("log_odds_vs_wild_type")),
                site_quality=(
                    str(site["site_quality"])[:80] if site.get("site_quality") else None
                ),
            )
        )
    pairwise_score = raw_features.get("pairwise_residual_log_odds")
    if pairwise_score is None:
        pairwise_score = raw_features.get("pairwise_frequency_log_odds")
    return ConservationEvidenceFeatures(
        sites=tuple(sites[:16]),
        independent_log_odds=_optional_float(raw_features.get("independent_log_odds")),
        independent_log_odds_sum=_optional_float(
            raw_features.get("independent_log_odds_sum")
        ),
        independent_mean_log_odds_per_mutation=_optional_float(
            raw_features.get("independent_mean_log_odds_per_mutation")
        ),
        pairwise_enabled=_optional_bool(raw_features.get("pairwise_enabled")),
        pairwise_eligible=_optional_bool(raw_features.get("pairwise_eligible")),
        pairwise_score=_optional_float(pairwise_score),
    )


def _structure_features(raw_features: dict[str, Any]) -> StructureEvidenceFeatures:
    sites: list[StructureSiteCard] = []
    for raw_position, raw_site in _mapping(raw_features.get("sites")).items():
        position = _position(raw_position)
        if position is None:
            continue
        site = _mapping(raw_site)
        missing_atoms = _strings(site.get("missing_backbone_atoms"), limit=8)
        status = str(site["status"])[:80] if site.get("status") else None
        sites.append(
            StructureSiteCard(
                position=position,
                mutation=str(site["mutation"])[:32] if site.get("mutation") else None,
                status=status,
                relative_sasa=_optional_float(site.get("relative_sasa")),
                contact_count=_optional_int(site.get("contact_count")),
                secondary_structure=(
                    str(site["secondary_structure"])[:80]
                    if site.get("secondary_structure")
                    else None
                ),
                missing_backbone_atoms=missing_atoms,
                missing_coordinates=bool(missing_atoms or (status and status != "ok")),
                mutant_side_chain_not_modelled=_optional_bool(
                    site.get("mutant_side_chain_not_modelled")
                ),
            )
        )
    return StructureEvidenceFeatures(
        sites=tuple(sites[:16]),
        static_context_flag_count=_optional_int(
            raw_features.get("static_context_flag_count")
        ),
    )


def _feature_projection(raw: dict[str, Any]) -> MutationEvidenceFeatures:
    channel = str(raw.get("channel") or "unknown").casefold()
    evidence_type = str(raw.get("evidence_type") or "")
    raw_features = _mapping(raw.get("raw_features"))
    if channel == "physchem":
        return _physchem_features(raw_features)
    if channel == "conservation":
        return _conservation_features(raw_features)
    if channel == "structure":
        return _structure_features(raw_features)
    if channel == "kg":
        return KGEvidenceFeatures(
            raw_association_score=_optional_float(
                raw_features.get("raw_association_score")
            ),
            support=_optional_int(raw_features.get("support")),
            global_visible_mean=_optional_float(raw_features.get("global_visible_mean")),
        )
    if channel == "local_rag" or evidence_type == "retrieved_document":
        return RAGEvidenceFeatures(
            knowledge_type=(
                str(raw_features["knowledge_type"])[:120]
                if raw_features.get("knowledge_type")
                else None
            ),
            retrieval_scores=_numeric_features(
                raw_features.get("retrieval_scores"), limit=12
            ),
        )
    return GenericEvidenceFeatures()


def mutation_evidence_card(
    evidence: Evidence | Mapping[str, Any],
) -> MutationEvidenceCard:
    """Project canonical evidence without mutating or replacing its persisted payload."""

    raw = _raw(evidence)
    channel = str(raw.get("channel") or "unknown")[:120]
    evidence_type = str(raw.get("evidence_type") or "computed")[:120]
    rag_visible = channel.casefold() == "local_rag" or evidence_type == "retrieved_document"
    polarity = str(raw.get("polarity") or "neutral")[:40]
    return MutationEvidenceCard(
        evidence_id=str(raw.get("evidence_id") or "unknown")[:320],
        variant_id=str(raw.get("variant_id") or "unknown")[:320],
        channel=channel,
        evidence_type=evidence_type,
        statement=str(raw.get("statement") or "No evidence statement supplied.")[:1200],
        score=_optional_float(raw.get("score")) or 0.0,
        confidence=_optional_float(raw.get("confidence")) or 0.0,
        quality_status=str(raw.get("quality_status") or "unknown")[:120],
        applicability=(
            str(raw.get("applicability") or "unknown")[:240] if rag_visible else None
        ),
        contributes_to_selection=(
            True if bool(raw.get("contributes_to_selection", False)) else None
        ),
        polarity=polarity if polarity != "neutral" else None,
        warnings=_strings(raw.get("warnings"), limit=16),
        source=_source_card(raw),
        features=_feature_projection(raw),
    )


def mutation_evidence_prompt_payload(
    evidence: Evidence | Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize a compact card while retaining its discriminated-union tag."""

    card = mutation_evidence_card(evidence)
    payload = card.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    payload.setdefault("features", {})["kind"] = card.features.kind
    return payload


def mutation_evidence_batch_metadata(
    evidence: Iterable[Evidence | Mapping[str, Any]],
) -> MutationEvidenceBatchMetadata:
    assay_ph: set[float] = set()
    conservation: set[tuple[Any, ...]] = set()
    structure_resources: set[str] = set()
    channel_rows: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        raw = _raw(item)
        features = _mapping(raw.get("raw_features"))
        channel = str(raw.get("channel") or "").casefold()
        channel_rows.setdefault(channel or "unknown", []).append(raw)
        if channel == "physchem":
            value = _optional_float(features.get("assay_pH"))
            if value is not None:
                assay_ph.add(value)
        elif channel == "conservation":
            conservation.add(
                (
                    _optional_int(features.get("sequence_count")),
                    _optional_float(features.get("neff")),
                    _optional_float(features.get("neff_per_length")),
                    _optional_bool(features.get("pairwise_enabled")),
                    _optional_bool(features.get("pairwise_eligible")),
                    (
                        str(features["pairwise_score_method"])[:120]
                        if features.get("pairwise_score_method")
                        else None
                    ),
                )
            )
        elif channel == "structure" and features.get("resource_id"):
            structure_resources.add(str(features["resource_id"])[:320])
    channel_shared: list[ChannelEvidenceSharedMetadata] = []
    for channel, rows in sorted(channel_rows.items()):
        warning_sets = [set(_strings(row.get("warnings"), limit=16)) for row in rows]
        common_warnings = (
            set.intersection(*warning_sets) if warning_sets else set()
        )
        source_ids = {str(row.get("source_id")) for row in rows if row.get("source_id")}
        channel_shared.append(
            ChannelEvidenceSharedMetadata(
                channel=channel,
                warnings=tuple(sorted(common_warnings)),
                source_id=next(iter(source_ids)) if len(source_ids) == 1 else None,
            )
        )
    return MutationEvidenceBatchMetadata(
        assay_pH_values=tuple(sorted(assay_ph)),
        conservation_profiles=tuple(
            ConservationBatchMetadata(
                sequence_count=item[0],
                neff=item[1],
                neff_per_length=item[2],
                pairwise_enabled=item[3],
                pairwise_eligible=item[4],
                pairwise_score_method=item[5],
            )
            for item in sorted(conservation, key=lambda value: repr(value))
        ),
        structure_resource_ids=tuple(sorted(structure_resources)),
        channel_shared=tuple(channel_shared),
    )


__all__ = [
    "MutationEvidenceBatchMetadata",
    "MutationEvidenceCard",
    "mutation_evidence_batch_metadata",
    "mutation_evidence_card",
    "mutation_evidence_prompt_payload",
]
