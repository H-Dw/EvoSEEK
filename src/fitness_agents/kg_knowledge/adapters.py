from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import combinations
from typing import Protocol

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Hypothesis,
    HypothesisAssessment,
    HypothesisReflection,
    Prediction,
    ValidationRecord,
    Variant,
)
from fitness_agents.local_knowledge.catalog import PublicationCatalog
from fitness_agents.local_knowledge.contracts import RetrievalResult
from fitness_agents.local_knowledge.leakage import TargetLeakageGuard
from fitness_agents.mutation.notation import InvalidMutationNotation, parse_mutation_notation

from .schema import (
    BuildContext,
    EntityRecord,
    KnowledgeBatch,
    KnowledgeLayer,
    Modality,
    RelationRecord,
    stable_record_id,
)

CANONICAL_RESIDUES = tuple("ACDEFGHIKLMNPQRSTVWY")


class KnowledgeAdapter(Protocol):
    name: str

    def extract(self, context: BuildContext) -> KnowledgeBatch: ...


class CallableKnowledgeAdapter:
    def __init__(self, name: str, function: Callable[[BuildContext], KnowledgeBatch]) -> None:
        self.name = name
        self.function = function

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        batch = self.function(context)
        if batch.adapter_name != self.name:
            raise ValueError(f"Adapter {self.name!r} returned batch for {batch.adapter_name!r}")
        return batch


class StaticKnowledgeAdapter:
    """Small adapter useful for tests and externally prepared graph fragments."""

    def __init__(
        self,
        name: str,
        *,
        entities: Iterable[EntityRecord] = (),
        relations: Iterable[RelationRecord] = (),
    ) -> None:
        self.name = name
        self.entities = tuple(entities)
        self.relations = tuple(relations)

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        del context
        return KnowledgeBatch(self.name, self.entities, self.relations)


def _relation(
    adapter: str,
    subject_id: str,
    predicate: str,
    object_id: str,
    layer: KnowledgeLayer,
    *,
    modality: Modality,
    source_id: str,
    source_group: str,
    context_id: str | None = None,
    valid_from_round: int | None = None,
    properties: dict[str, object] | None = None,
    evidence_ids: tuple[str, ...] = (),
) -> RelationRecord:
    return RelationRecord(
        relation_id=stable_record_id(
            "rel", adapter, subject_id, predicate, object_id, context_id, valid_from_round
        ),
        subject_id=subject_id,
        predicate=predicate,
        object_id=object_id,
        layer=layer,
        modalities=frozenset({modality}),
        properties=properties or {},
        source_ids=(source_id,),
        evidence_ids=evidence_ids,
        source_group=source_group,
        context_id=context_id,
        valid_from_round=valid_from_round,
    )


def _observation_record_id(context: BuildContext, observation: FitnessObservation) -> str:
    return stable_record_id(
        "observation",
        context.run_id,
        observation.variant_id,
        observation.round_revealed,
        observation.source,
    )


def _mutation_edits(variant: Variant):
    try:
        return parse_mutation_notation(variant.mutation_notation)
    except InvalidMutationNotation:
        return ()


def _mutation_signature(variant: Variant) -> frozenset[tuple[int, str, str]]:
    return frozenset(edit.identity for edit in _mutation_edits(variant))


def _feature_semantic_records(
    *,
    context: BuildContext,
    item: Evidence,
    variant: Variant,
) -> tuple[tuple[EntityRecord, ...], tuple[RelationRecord, ...]]:
    """Materialize typed feature semantics without treating descriptors as fitness labels."""

    if item.quality_status == "unavailable" or item.variant_id.startswith("context:"):
        return (), ()
    raw = item.raw_features if isinstance(item.raw_features, dict) else {}
    raw_sites = raw.get("sites", {})
    if not isinstance(raw_sites, dict):
        return (), ()
    edits = {edit.position: edit for edit in _mutation_edits(variant)}
    entities: dict[str, EntityRecord] = {}
    relations: list[RelationRecord] = []
    source_group = f"feature:{item.channel}"
    evidence_entity_id = f"evidence:{item.evidence_id}"
    provenance_hash = str(item.provenance.get("resource_sha256", item.source_id))

    def relation(
        subject_id: str,
        predicate: str,
        object_id: str,
        layer: KnowledgeLayer,
        modality: Modality,
        *,
        context_id: str | None = None,
        valid_from_round: int | None = None,
        evidence_ids: tuple[str, ...] = (),
    ) -> RelationRecord:
        return _relation(
            "inference_records",
            subject_id,
            predicate,
            object_id,
            layer,
            modality=modality,
            source_id=item.source_id,
            source_group=source_group,
            context_id=context_id,
            valid_from_round=valid_from_round,
            evidence_ids=evidence_ids or (item.evidence_id,),
        )

    for raw_position, raw_site in raw_sites.items():
        try:
            position = int(raw_position)
        except (TypeError, ValueError):
            continue
        edit = edits.get(position)
        if edit is None or not isinstance(raw_site, dict):
            continue
        mutation_id = f"mutation:{context.protein_id}:{edit.wt}{position}{edit.mutant}"
        position_id = f"residue:{context.protein_id}:{position}"

        if item.channel == "physchem":
            property_accessions = raw.get("property_accessions", {})
            property_accessions = (
                property_accessions if isinstance(property_accessions, dict) else {}
            )
            for residue, values_key in (
                (edit.wt, "wild_type_values"),
                (edit.mutant, "mutant_values"),
            ):
                residue_id = f"residue-type:{residue}"
                entities[residue_id] = EntityRecord(
                    residue_id,
                    "ResidueType",
                    KnowledgeLayer.SEQUENCE,
                    frozenset({Modality.SEQUENCE}),
                    {"one_letter_code": residue},
                    ("IUPAC:amino-acid-code",),
                    "canonical_amino_acid",
                )
                values = raw_site.get(values_key, {})
                if not isinstance(values, dict):
                    continue
                for property_name, value in sorted(values.items()):
                    descriptor_id = stable_record_id(
                        "physchem-property",
                        provenance_hash,
                        property_name,
                        residue,
                        value,
                    )
                    entities[descriptor_id] = EntityRecord(
                        descriptor_id,
                        "PhyschemPropertyValue",
                        KnowledgeLayer.SEQUENCE,
                        frozenset({Modality.TABULAR}),
                        {
                            "property": property_name,
                            "value": value,
                            "residue": residue,
                            "accession": property_accessions.get(property_name),
                        },
                        (item.source_id,),
                        source_group,
                        item.confidence,
                    )
                    relations.append(
                        relation(
                            residue_id,
                            "HAS_DESCRIPTOR",
                            descriptor_id,
                            KnowledgeLayer.SEQUENCE,
                            Modality.TABULAR,
                        )
                    )
            substitution_id = stable_record_id(
                "substitution-descriptor",
                provenance_hash,
                mutation_id,
            )
            entities[substitution_id] = EntityRecord(
                substitution_id,
                "SubstitutionDescriptor",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE, Modality.TABULAR}),
                {
                    "mutation": edit.hgvs_short,
                    "deltas": raw_site.get("deltas", {}),
                    "property_accessions": property_accessions,
                    "descriptor_only_not_fitness": True,
                },
                (item.source_id,),
                source_group,
                item.confidence,
            )
            relations.extend(
                (
                    relation(
                        mutation_id,
                        "HAS_PHYSCHEM_DELTA",
                        substitution_id,
                        KnowledgeLayer.SEQUENCE,
                        Modality.TABULAR,
                    ),
                    relation(
                        substitution_id,
                        "DERIVED_FROM",
                        evidence_entity_id,
                        KnowledgeLayer.PROVENANCE,
                        Modality.TABULAR,
                    ),
                )
            )

        elif item.channel == "conservation":
            profile_id = stable_record_id(
                "evolution-profile",
                provenance_hash,
                position,
                edit.mutant,
            )
            entities[profile_id] = EntityRecord(
                profile_id,
                "EvolutionProfile",
                KnowledgeLayer.EVOLUTIONARY,
                frozenset({Modality.MSA, Modality.TABULAR}),
                {
                    **raw_site,
                    "mutation": edit.hgvs_short,
                    "sequence_count": raw.get("sequence_count"),
                    "neff": raw.get("neff"),
                    "neff_per_length": raw.get("neff_per_length"),
                    "pseudocount_mode": raw.get("pseudocount_mode"),
                    "pseudocount_value": raw.get("pseudocount_value"),
                    "pairwise_enabled": raw.get("pairwise_enabled"),
                    "pairwise_eligible": raw.get("pairwise_eligible"),
                    "pairwise_score_method": raw.get("pairwise_score_method"),
                    "estimated_parameters": raw.get("estimated_parameters", []),
                    "evolutionary_prior_not_fitness": True,
                },
                (item.source_id,),
                source_group,
                item.confidence,
            )
            relations.extend(
                (
                    relation(
                        mutation_id,
                        "HAS_EVOLUTIONARY_CONTEXT",
                        profile_id,
                        KnowledgeLayer.EVOLUTIONARY,
                        Modality.MSA,
                    ),
                    relation(
                        position_id,
                        "HAS_EVOLUTION_PROFILE",
                        profile_id,
                        KnowledgeLayer.EVOLUTIONARY,
                        Modality.MSA,
                    ),
                    relation(
                        profile_id,
                        "DERIVED_FROM",
                        evidence_entity_id,
                        KnowledgeLayer.PROVENANCE,
                        Modality.MSA,
                    ),
                )
            )

        elif item.channel == "structure" and raw_site.get("status") == "ok":
            environment_properties = {
                key: value
                for key, value in raw_site.items()
                if key
                not in {
                    "mutation",
                    "mutant_side_chain_not_modelled",
                    "closest_contacts",
                    "interface_contacts",
                }
            }
            environment_id = stable_record_id(
                "residue-environment",
                provenance_hash,
                raw.get("resource_id"),
                position,
                environment_properties,
            )
            entities[environment_id] = EntityRecord(
                environment_id,
                "ResidueEnvironment",
                KnowledgeLayer.STRUCTURE,
                frozenset({Modality.STRUCTURE_3D, Modality.TABULAR}),
                {
                    **environment_properties,
                    "protein_position": position,
                    "resource_id": raw.get("resource_id"),
                    "static_environment_not_mutant_model": True,
                },
                (item.source_id,),
                source_group,
                item.confidence,
            )
            relations.extend(
                (
                    relation(
                        mutation_id,
                        "OCCURS_IN_ENVIRONMENT",
                        environment_id,
                        KnowledgeLayer.STRUCTURE,
                        Modality.STRUCTURE_3D,
                    ),
                    relation(
                        position_id,
                        "MAPPED_TO_STRUCTURE",
                        environment_id,
                        KnowledgeLayer.STRUCTURE,
                        Modality.STRUCTURE_3D,
                    ),
                    relation(
                        environment_id,
                        "DERIVED_FROM",
                        evidence_entity_id,
                        KnowledgeLayer.PROVENANCE,
                        Modality.STRUCTURE_3D,
                    ),
                )
            )
    return tuple(entities.values()), tuple(relations)


class SiteFeatureKnowledgeAdapter:
    """Ingest replacement-level conservation/structure/physchem tables once per campaign."""

    name = "site_features"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        tables = context.resources.get("site_feature_tables") or {}
        if not isinstance(tables, dict) or not tables:
            return KnowledgeBatch(self.name)

        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        protein_id = f"protein:{context.protein_id}"
        entities[protein_id] = EntityRecord(
            protein_id,
            "Protein",
            KnowledgeLayer.IDENTITY,
            frozenset({Modality.SEQUENCE}),
            {"accession": context.protein_id},
            (f"run:{context.run_id}",),
            "campaign",
        )

        def ensure_position(position: int, wild_type: str) -> str:
            position_id = f"residue:{context.protein_id}:{position}"
            entities[position_id] = EntityRecord(
                position_id,
                "ResiduePosition",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE}),
                {"position": position, "reference_residue": wild_type},
                (f"run:{context.run_id}",),
                "campaign",
            )
            return position_id

        def ensure_mutation(position: int, wild_type: str, mutant: str) -> str:
            mutation_id = f"mutation:{context.protein_id}:{wild_type}{position}{mutant}"
            position_id = ensure_position(position, wild_type)
            entities[mutation_id] = EntityRecord(
                mutation_id,
                "Mutation",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE}),
                {
                    "reference": wild_type,
                    "position": position,
                    "alternate": mutant,
                },
                (f"run:{context.run_id}",),
                "campaign",
            )
            relations.append(
                _relation(
                    self.name,
                    mutation_id,
                    "AT_POSITION",
                    position_id,
                    KnowledgeLayer.SEQUENCE,
                    modality=Modality.SEQUENCE,
                    source_id=f"run:{context.run_id}",
                    source_group="campaign",
                )
            )
            return mutation_id

        def ensure_residue_type(residue: str) -> str:
            residue_id = f"residue-type:{residue}"
            entities[residue_id] = EntityRecord(
                residue_id,
                "ResidueType",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE}),
                {"one_letter_code": residue},
                ("IUPAC:amino-acid-code",),
                "canonical_amino_acid",
            )
            return residue_id

        physchem = tables.get("physchem") if isinstance(tables.get("physchem"), dict) else {}
        if physchem:
            source_id = str(physchem.get("source_id") or "physchem")
            provenance_hash = str(physchem.get("resource_sha256") or source_id)
            source_group = "feature:physchem"
            property_accessions = physchem.get("property_accessions", {})
            property_accessions = (
                property_accessions if isinstance(property_accessions, dict) else {}
            )
            positions = physchem.get("positions", {})
            if isinstance(positions, dict):
                for raw_position, payload in positions.items():
                    if not isinstance(payload, dict):
                        continue
                    try:
                        position = int(raw_position)
                    except (TypeError, ValueError):
                        continue
                    wild_type = str(payload.get("wild_type") or "")
                    if not wild_type:
                        continue
                    ensure_position(position, wild_type)
                    substitutions = payload.get("substitutions", {})
                    if not isinstance(substitutions, dict):
                        continue
                    for mutant, site in substitutions.items():
                        if not isinstance(site, dict) or str(mutant) == wild_type:
                            continue
                        mutant_code = str(mutant)
                        mutation_id = ensure_mutation(position, wild_type, mutant_code)
                        for residue, values_key in (
                            (wild_type, "wild_type_values"),
                            (mutant_code, "mutant_values"),
                        ):
                            residue_id = ensure_residue_type(residue)
                            values = site.get(values_key, {})
                            if not isinstance(values, dict):
                                continue
                            for property_name, value in sorted(values.items()):
                                descriptor_id = stable_record_id(
                                    "physchem-property",
                                    provenance_hash,
                                    property_name,
                                    residue,
                                    value,
                                )
                                entities[descriptor_id] = EntityRecord(
                                    descriptor_id,
                                    "PhyschemPropertyValue",
                                    KnowledgeLayer.SEQUENCE,
                                    frozenset({Modality.TABULAR}),
                                    {
                                        "property": property_name,
                                        "value": value,
                                        "residue": residue,
                                        "accession": property_accessions.get(property_name),
                                    },
                                    (source_id,),
                                    source_group,
                                )
                                relations.append(
                                    _relation(
                                        self.name,
                                        residue_id,
                                        "HAS_DESCRIPTOR",
                                        descriptor_id,
                                        KnowledgeLayer.SEQUENCE,
                                        modality=Modality.TABULAR,
                                        source_id=source_id,
                                        source_group=source_group,
                                    )
                                )
                        substitution_id = stable_record_id(
                            "substitution-descriptor",
                            provenance_hash,
                            mutation_id,
                        )
                        entities[substitution_id] = EntityRecord(
                            substitution_id,
                            "SubstitutionDescriptor",
                            KnowledgeLayer.SEQUENCE,
                            frozenset({Modality.SEQUENCE, Modality.TABULAR}),
                            {
                                "mutation": site.get(
                                    "mutation", f"{wild_type}{position}{mutant_code}"
                                ),
                                "deltas": site.get("deltas", {}),
                                "property_accessions": property_accessions,
                                "descriptor_only_not_fitness": True,
                            },
                            (source_id,),
                            source_group,
                        )
                        relations.append(
                            _relation(
                                self.name,
                                mutation_id,
                                "HAS_PHYSCHEM_DELTA",
                                substitution_id,
                                KnowledgeLayer.SEQUENCE,
                                modality=Modality.TABULAR,
                                source_id=source_id,
                                source_group=source_group,
                            )
                        )

        conservation = (
            tables.get("conservation") if isinstance(tables.get("conservation"), dict) else {}
        )
        if conservation:
            provenance_hash = str(conservation.get("resource_sha256") or "conservation")
            source_id = f"msa_profile:{provenance_hash[:16]}"
            source_group = "feature:conservation"
            positions = conservation.get("positions", {})
            shared = {
                "sequence_count": conservation.get("sequence_count"),
                "neff": conservation.get("neff"),
                "neff_per_length": conservation.get("neff_per_length"),
                "pseudocount_mode": conservation.get("pseudocount_mode"),
                "pseudocount_value": conservation.get("pseudocount_value"),
                "pairwise_enabled": conservation.get("pairwise_enabled"),
                "pairwise_score_method": conservation.get("pairwise_mode"),
                "estimated_parameters": conservation.get("estimated_parameters", []),
                "evolutionary_prior_not_fitness": True,
            }
            if isinstance(positions, dict):
                for raw_position, payload in positions.items():
                    if not isinstance(payload, dict):
                        continue
                    try:
                        position = int(raw_position)
                    except (TypeError, ValueError):
                        continue
                    wild_type = str(payload.get("wild_type") or "")
                    if not wild_type:
                        continue
                    position_id = ensure_position(position, wild_type)
                    residues = payload.get("residues", {})
                    if not isinstance(residues, dict):
                        continue
                    for residue in CANONICAL_RESIDUES:
                        site = residues.get(residue)
                        if not isinstance(site, dict):
                            continue
                        profile_id = stable_record_id(
                            "evolution-profile",
                            provenance_hash,
                            position,
                            residue,
                        )
                        entities[profile_id] = EntityRecord(
                            profile_id,
                            "EvolutionProfile",
                            KnowledgeLayer.EVOLUTIONARY,
                            frozenset({Modality.MSA, Modality.TABULAR}),
                            {
                                **site,
                                "mutation": f"{wild_type}{position}{residue}",
                                **shared,
                            },
                            (source_id,),
                            source_group,
                        )
                        relations.append(
                            _relation(
                                self.name,
                                position_id,
                                "HAS_EVOLUTION_PROFILE",
                                profile_id,
                                KnowledgeLayer.EVOLUTIONARY,
                                modality=Modality.MSA,
                                source_id=source_id,
                                source_group=source_group,
                            )
                        )
                        if residue == wild_type:
                            continue
                        mutation_id = ensure_mutation(position, wild_type, residue)
                        relations.append(
                            _relation(
                                self.name,
                                mutation_id,
                                "HAS_EVOLUTIONARY_CONTEXT",
                                profile_id,
                                KnowledgeLayer.EVOLUTIONARY,
                                modality=Modality.MSA,
                                source_id=source_id,
                                source_group=source_group,
                            )
                        )

        structure = tables.get("structure") if isinstance(tables.get("structure"), dict) else {}
        if structure:
            provenance_hash = str(structure.get("resource_sha256") or "structure")
            resource_id = structure.get("resource_id")
            source_id = f"structure:{resource_id}"
            source_group = "feature:structure"
            positions = structure.get("positions", {})
            if isinstance(positions, dict):
                for raw_position, site in positions.items():
                    if not isinstance(site, dict) or site.get("status") != "ok":
                        continue
                    try:
                        position = int(raw_position)
                    except (TypeError, ValueError):
                        continue
                    wild_type = str(site.get("wild_type") or "")
                    if not wild_type:
                        continue
                    position_id = ensure_position(position, wild_type)
                    environment_properties = {
                        key: value
                        for key, value in site.items()
                        if key
                        not in {
                            "wild_type",
                            "mutation",
                            "mutant_side_chain_not_modelled",
                            "closest_contacts",
                            "interface_contacts",
                        }
                    }
                    environment_id = stable_record_id(
                        "residue-environment",
                        provenance_hash,
                        resource_id,
                        position,
                        environment_properties,
                    )
                    entities[environment_id] = EntityRecord(
                        environment_id,
                        "ResidueEnvironment",
                        KnowledgeLayer.STRUCTURE,
                        frozenset({Modality.STRUCTURE_3D, Modality.TABULAR}),
                        {
                            **environment_properties,
                            "protein_position": position,
                            "resource_id": resource_id,
                            "static_environment_not_mutant_model": True,
                        },
                        (source_id,),
                        source_group,
                    )
                    relations.append(
                        _relation(
                            self.name,
                            position_id,
                            "MAPPED_TO_STRUCTURE",
                            environment_id,
                            KnowledgeLayer.STRUCTURE,
                            modality=Modality.STRUCTURE_3D,
                            source_id=source_id,
                            source_group=source_group,
                        )
                    )
                    for mutant in CANONICAL_RESIDUES:
                        if mutant == wild_type:
                            continue
                        mutation_id = ensure_mutation(position, wild_type, mutant)
                        relations.append(
                            _relation(
                                self.name,
                                mutation_id,
                                "OCCURS_IN_ENVIRONMENT",
                                environment_id,
                                KnowledgeLayer.STRUCTURE,
                                modality=Modality.STRUCTURE_3D,
                                source_id=source_id,
                                source_group=source_group,
                            )
                        )

        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))


class CampaignObservationAdapter:
    """Convert current project variants and measurements into the common KG schema."""

    name = "campaign_observations"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        variants = tuple(context.resources.get("variants", ()))
        observations = tuple(context.resources.get("observations", ()))
        if not all(isinstance(item, Variant) for item in variants):
            raise TypeError("resources['variants'] must contain Variant records")
        if not all(isinstance(item, FitnessObservation) for item in observations):
            raise TypeError("resources['observations'] must contain FitnessObservation records")

        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        protein_id = f"protein:{context.protein_id}"
        run_source = f"run:{context.run_id}"
        entities[protein_id] = EntityRecord(
            protein_id,
            "Protein",
            KnowledgeLayer.IDENTITY,
            frozenset({Modality.SEQUENCE}),
            {"accession": context.protein_id},
            (run_source,),
            "campaign",
        )

        round_ids = {item.round_revealed for item in observations}
        for round_id in round_ids:
            entity_id = f"round:{context.run_id}:{round_id}"
            entities[entity_id] = EntityRecord(
                entity_id,
                "CampaignRound",
                KnowledgeLayer.PROVENANCE,
                frozenset({Modality.TIME_SERIES}),
                {"run_id": context.run_id, "round_id": round_id},
                (run_source,),
                "campaign",
                valid_from_round=round_id,
            )

        if context.assay_id:
            assay_id = f"assay:{context.assay_id}"
            entities[assay_id] = EntityRecord(
                assay_id,
                "Assay",
                KnowledgeLayer.EXPERIMENTAL,
                frozenset({Modality.TABULAR}),
                {"assay_id": context.assay_id},
                (run_source,),
                "campaign",
            )
        if context.condition_id:
            condition_id = f"condition:{context.condition_id}"
            entities[condition_id] = EntityRecord(
                condition_id,
                "Condition",
                KnowledgeLayer.EXPERIMENTAL,
                frozenset({Modality.TABULAR}),
                {"condition_id": context.condition_id},
                (run_source,),
                "campaign",
            )

        variant_lookup: dict[str, Variant] = {}
        for variant in variants:
            variant_lookup[variant.variant_id] = variant
            variant_id = f"variant:{variant.variant_id}"
            sequence_id = stable_record_id("sequence", variant.sequence)
            entities[variant_id] = EntityRecord(
                variant_id,
                "Variant",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE}),
                {
                    "label": variant.variant,
                    "mutation_notation": variant.mutation_notation,
                    "mutation_count": variant.mutation_count,
                    "split_role": variant.split_role,
                },
                (run_source,),
                "campaign",
            )
            entities[sequence_id] = EntityRecord(
                sequence_id,
                "Sequence",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.SEQUENCE}),
                {"sequence": variant.sequence, "length": len(variant.sequence)},
                (run_source,),
                "campaign",
            )
            relations.extend(
                (
                    _relation(
                        self.name,
                        variant_id,
                        "VARIANT_OF",
                        protein_id,
                        KnowledgeLayer.IDENTITY,
                        modality=Modality.SEQUENCE,
                        source_id=run_source,
                        source_group="campaign",
                    ),
                    _relation(
                        self.name,
                        variant_id,
                        "HAS_SEQUENCE",
                        sequence_id,
                        KnowledgeLayer.SEQUENCE,
                        modality=Modality.SEQUENCE,
                        source_id=run_source,
                        source_group="campaign",
                    ),
                )
            )
            mutation_edits = _mutation_edits(variant)
            for edit in mutation_edits:
                reference = edit.wt
                position = edit.position
                alternate = edit.mutant
                position_id = f"residue:{context.protein_id}:{position}"
                mutation_id = f"mutation:{context.protein_id}:{reference}{position}{alternate}"
                entities[position_id] = EntityRecord(
                    position_id,
                    "ResiduePosition",
                    KnowledgeLayer.SEQUENCE,
                    frozenset({Modality.SEQUENCE}),
                    {"position": position, "reference_residue": reference},
                    (run_source,),
                    "campaign",
                )
                entities[mutation_id] = EntityRecord(
                    mutation_id,
                    "Mutation",
                    KnowledgeLayer.SEQUENCE,
                    frozenset({Modality.SEQUENCE}),
                    {"reference": reference, "position": position, "alternate": alternate},
                    (run_source,),
                    "campaign",
                )
                relations.extend(
                    (
                        _relation(
                            self.name,
                            variant_id,
                            "HAS_MUTATION",
                            mutation_id,
                            KnowledgeLayer.SEQUENCE,
                            modality=Modality.SEQUENCE,
                            source_id=run_source,
                            source_group="campaign",
                        ),
                        _relation(
                            self.name,
                            mutation_id,
                            "AT_POSITION",
                            position_id,
                            KnowledgeLayer.SEQUENCE,
                            modality=Modality.SEQUENCE,
                            source_id=run_source,
                            source_group="campaign",
                        ),
                    )
                )

        for observation in observations:
            if observation.variant_id not in variant_lookup:
                continue
            source_id = f"observation-source:{observation.source}"
            observation_id = _observation_record_id(context, observation)
            entities[observation_id] = EntityRecord(
                observation_id,
                "Observation",
                KnowledgeLayer.EXPERIMENTAL,
                frozenset({Modality.TABULAR, Modality.TIME_SERIES}),
                {
                    "fitness": observation.fitness,
                    "split_role": observation.split_role,
                    "round_revealed": observation.round_revealed,
                    "source": observation.source,
                },
                (source_id,),
                observation.source,
                valid_from_round=observation.round_revealed,
            )
            context_id = observation_id
            relations.append(
                _relation(
                    self.name,
                    observation_id,
                    "OBSERVES_VARIANT",
                    f"variant:{observation.variant_id}",
                    KnowledgeLayer.EXPERIMENTAL,
                    modality=Modality.TABULAR,
                    source_id=source_id,
                    source_group=observation.source,
                    context_id=context_id,
                    valid_from_round=observation.round_revealed,
                )
            )
            relations.append(
                _relation(
                    self.name,
                    observation_id,
                    "REVEALED_IN",
                    f"round:{context.run_id}:{observation.round_revealed}",
                    KnowledgeLayer.PROVENANCE,
                    modality=Modality.TIME_SERIES,
                    source_id=source_id,
                    source_group=observation.source,
                    context_id=context_id,
                    valid_from_round=observation.round_revealed,
                )
            )
            if context.assay_id:
                relations.append(
                    _relation(
                        self.name,
                        observation_id,
                        "MEASURED_IN",
                        f"assay:{context.assay_id}",
                        KnowledgeLayer.EXPERIMENTAL,
                        modality=Modality.TABULAR,
                        source_id=source_id,
                        source_group=observation.source,
                        context_id=context_id,
                        valid_from_round=observation.round_revealed,
                    )
                )
            if context.condition_id:
                relations.append(
                    _relation(
                        self.name,
                        observation_id,
                        "UNDER_CONDITION",
                        f"condition:{context.condition_id}",
                        KnowledgeLayer.EXPERIMENTAL,
                        modality=Modality.TABULAR,
                        source_id=source_id,
                        source_group=observation.source,
                        context_id=context_id,
                        valid_from_round=observation.round_revealed,
                    )
                )
        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))


class InferenceKnowledgeAdapter:
    """Convert derived effects, model outputs, and agent evidence into versioned KG records."""

    name = "inference_records"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        variants = tuple(context.resources.get("variants", ()))
        observations = tuple(context.resources.get("observations", ()))
        predictions = tuple(context.resources.get("predictions", ()))
        evidence_items = tuple(context.resources.get("evidence", ()))
        hypotheses = tuple(context.resources.get("hypotheses", ()))
        if not all(isinstance(item, Variant) for item in variants):
            raise TypeError("resources['variants'] must contain Variant records")
        if not all(isinstance(item, FitnessObservation) for item in observations):
            raise TypeError("resources['observations'] must contain FitnessObservation records")
        if not all(isinstance(item, Prediction) for item in predictions):
            raise TypeError("resources['predictions'] must contain Prediction records")
        if not all(isinstance(item, Evidence) for item in evidence_items):
            raise TypeError("resources['evidence'] must contain Evidence records")
        if not all(isinstance(item, Hypothesis) for item in hypotheses):
            raise TypeError("resources['hypotheses'] must contain Hypothesis records")

        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        for prediction in predictions:
            model_id = f"model-run:{prediction.model_version}:{context.round_id}"
            prediction_id = stable_record_id(
                "prediction",
                context.run_id,
                context.round_id,
                prediction.variant_id,
                prediction.model_version,
            )
            entities[model_id] = EntityRecord(
                model_id,
                "ModelRun",
                KnowledgeLayer.MODEL,
                frozenset({Modality.TABULAR}),
                {"model_version": prediction.model_version, "round_id": context.round_id},
                (f"model:{prediction.model_version}",),
                prediction.model_version,
                valid_from_round=context.round_id,
            )
            entities[prediction_id] = EntityRecord(
                prediction_id,
                "Prediction",
                KnowledgeLayer.MODEL,
                frozenset({Modality.TABULAR}),
                {
                    "fitness_mean": prediction.fitness_mean,
                    "fitness_std": prediction.fitness_std,
                    "interval_90": prediction.interval_90,
                    "ood_score": prediction.ood_score,
                    "component_scores": prediction.component_scores,
                    "is_measured": prediction.is_measured,
                },
                (f"model:{prediction.model_version}",),
                prediction.model_version,
                valid_from_round=context.round_id,
            )
            relations.extend(
                (
                    _relation(
                        self.name,
                        prediction_id,
                        "PREDICTS",
                        f"variant:{prediction.variant_id}",
                        KnowledgeLayer.MODEL,
                        modality=Modality.TABULAR,
                        source_id=f"model:{prediction.model_version}",
                        source_group=prediction.model_version,
                        valid_from_round=context.round_id,
                    ),
                    _relation(
                        self.name,
                        prediction_id,
                        "GENERATED_BY",
                        model_id,
                        KnowledgeLayer.MODEL,
                        modality=Modality.TABULAR,
                        source_id=f"model:{prediction.model_version}",
                        source_group=prediction.model_version,
                        valid_from_round=context.round_id,
                    ),
                )
            )

        variant_lookup = {item.variant_id: item for item in variants}
        evidence_lookup: dict[str, Evidence] = {}
        for item in evidence_items:
            evidence_lookup[item.evidence_id] = item
            entity_id = f"evidence:{item.evidence_id}"
            entities[entity_id] = EntityRecord(
                entity_id,
                "Evidence",
                KnowledgeLayer.AGENT,
                frozenset({Modality.TEXT, Modality.TABULAR}),
                {
                    "channel": item.channel,
                    "statement": item.statement,
                    "score": item.score,
                    "calibrated_score": item.calibrated_score,
                    "calibrated": item.calibrated,
                    "contributes_to_selection": item.contributes_to_selection,
                    "evidence_type": item.evidence_type,
                    "quality_status": item.quality_status,
                    "applicability": item.applicability,
                    "uncertainty": item.uncertainty,
                    "raw_features": item.raw_features,
                    "warnings": item.warnings,
                    "provenance": item.provenance,
                    "round_id": item.round_id,
                    "claim_id": item.claim_id,
                    "polarity": item.polarity,
                    "source_group": item.source_group,
                    "artifact_uri": item.artifact_uri,
                    "artifact_span": item.artifact_span,
                },
                (item.source_id,),
                item.source_group if item.source_group != "unknown" else item.channel,
                item.confidence,
                valid_from_round=(
                    item.valid_from_round
                    if item.valid_from_round is not None
                    else item.round_id
                ),
                valid_to_round=item.valid_to_round,
            )
            if not item.variant_id.startswith("context:"):
                relations.append(
                    _relation(
                        self.name,
                        entity_id,
                        "ABOUT",
                        f"variant:{item.variant_id}",
                        KnowledgeLayer.AGENT,
                        modality=Modality.TEXT,
                        source_id=item.source_id,
                        source_group=(
                            item.source_group
                            if item.source_group != "unknown"
                            else item.channel
                        ),
                        valid_from_round=item.round_id,
                    )
                )
                variant = variant_lookup.get(item.variant_id)
                if variant is not None:
                    feature_entities, feature_relations = _feature_semantic_records(
                        context=context,
                        item=item,
                        variant=variant,
                    )
                    entities.update(
                        (feature_entity.entity_id, feature_entity)
                        for feature_entity in feature_entities
                    )
                    relations.extend(feature_relations)

        for hypothesis in hypotheses:
            entity_id = f"hypothesis:{hypothesis.hypothesis_id}"
            entities[entity_id] = EntityRecord(
                entity_id,
                "Hypothesis",
                KnowledgeLayer.AGENT,
                frozenset({Modality.TEXT}),
                {
                    "statement": hypothesis.statement,
                    "preferred_residues": hypothesis.preferred_residues,
                    "expected_outcome": hypothesis.expected_outcome,
                    "falsification_criterion": hypothesis.falsification_criterion,
                    "parent_hypothesis_id": hypothesis.parent_hypothesis_id,
                },
                (f"agent:{context.run_id}",),
                "agent",
                valid_from_round=context.round_id,
            )
            for evidence_id in hypothesis.evidence_ids:
                if evidence_id not in evidence_lookup:
                    continue
                evidence = evidence_lookup[evidence_id]
                predicate = (
                    "SUPPORTED_BY"
                    if evidence.contributes_to_selection and evidence.score >= 0
                    else (
                        "CONTRADICTED_BY"
                        if evidence.contributes_to_selection and evidence.score < 0
                        else "CITES_EVIDENCE"
                    )
                )
                relations.append(
                    RelationRecord(
                        relation_id=stable_record_id(
                            "rel", self.name, entity_id, predicate, evidence_id, context.round_id
                        ),
                        subject_id=entity_id,
                        predicate=predicate,
                        object_id=f"evidence:{evidence_id}",
                        layer=KnowledgeLayer.AGENT,
                        modalities=frozenset({Modality.TEXT}),
                        source_ids=(evidence.source_id,),
                        evidence_ids=(evidence_id,),
                        source_group="agent",
                        confidence=evidence.confidence,
                        valid_from_round=context.round_id,
                    )
                )
        latest_observation: dict[str, FitnessObservation] = {}
        for observation in observations:
            previous = latest_observation.get(observation.variant_id)
            if previous is None or (
                observation.round_revealed,
                observation.source,
            ) > (previous.round_revealed, previous.source):
                latest_observation[observation.variant_id] = observation

        signature_lookup: dict[frozenset[tuple[int, str, str]], Variant] = {}
        for variant in sorted(variants, key=lambda item: item.variant_id):
            signature_lookup.setdefault(_mutation_signature(variant), variant)

        def observed_variant(
            signature: frozenset[tuple[int, str, str]],
        ) -> tuple[Variant, FitnessObservation] | None:
            variant = signature_lookup.get(signature)
            if variant is None:
                return None
            observation = latest_observation.get(variant.variant_id)
            return (variant, observation) if observation is not None else None

        for child in sorted(variants, key=lambda item: item.variant_id):
            child_observation = latest_observation.get(child.variant_id)
            child_signature = _mutation_signature(child)
            if child_observation is None or not child_signature:
                continue
            for position, reference, alternate in sorted(child_signature):
                edit_identity = (position, reference, alternate)
                background_signature = frozenset(child_signature.difference({edit_identity}))
                background_record = observed_variant(background_signature)
                if background_record is None:
                    continue
                background, background_observation = background_record
                visible_round = max(
                    child_observation.round_revealed,
                    background_observation.round_revealed,
                )
                delta = float(child_observation.fitness - background_observation.fitness)
                mutation_id = f"mutation:{context.protein_id}:{reference}{position}{alternate}"
                estimate_id = stable_record_id(
                    "mutation-effect",
                    context.run_id,
                    context.assay_id,
                    child.variant_id,
                    background.variant_id,
                    mutation_id,
                    _observation_record_id(context, child_observation),
                    _observation_record_id(context, background_observation),
                )
                source_id = (
                    "derived:matched-background:"
                    f"{child_observation.source}:{background_observation.source}"
                )
                entities[estimate_id] = EntityRecord(
                    estimate_id,
                    "MutationEffectEstimate",
                    KnowledgeLayer.EXPERIMENTAL,
                    frozenset({Modality.TABULAR, Modality.TIME_SERIES}),
                    {
                        "delta_fitness": delta,
                        "direction": "improves" if delta > 0 else "worsens" if delta < 0 else "neutral",
                        "method": "matched_background_difference",
                        "child_variant_id": child.variant_id,
                        "background_variant_id": background.variant_id,
                        "requires_visible_pair": True,
                    },
                    (source_id,),
                    "derived_effect",
                    valid_from_round=visible_round,
                )
                context_id = f"variant:{background.variant_id}"
                for predicate, object_id, layer, modality in (
                    ("ABOUT_MUTATION", mutation_id, KnowledgeLayer.EXPERIMENTAL, Modality.TABULAR),
                    (
                        "IN_BACKGROUND",
                        f"variant:{background.variant_id}",
                        KnowledgeLayer.EXPERIMENTAL,
                        Modality.SEQUENCE,
                    ),
                ):
                    relations.append(
                        _relation(
                            self.name,
                            estimate_id,
                            predicate,
                            object_id,
                            layer,
                            modality=modality,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )
                if context.assay_id:
                    relations.append(
                        _relation(
                            self.name,
                            estimate_id,
                            "MEASURED_IN",
                            f"assay:{context.assay_id}",
                            KnowledgeLayer.EXPERIMENTAL,
                            modality=Modality.TABULAR,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )
                for source_observation in (child_observation, background_observation):
                    relations.append(
                        _relation(
                            self.name,
                            estimate_id,
                            "DERIVED_FROM",
                            _observation_record_id(context, source_observation),
                            KnowledgeLayer.PROVENANCE,
                            modality=Modality.TABULAR,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )

            if len(child_signature) < 2:
                continue
            for first, second in combinations(sorted(child_signature), 2):
                background_signature = frozenset(child_signature.difference({first, second}))
                required = (
                    observed_variant(background_signature),
                    observed_variant(frozenset((*background_signature, first))),
                    observed_variant(frozenset((*background_signature, second))),
                    observed_variant(frozenset((*background_signature, first, second))),
                )
                if any(item is None for item in required):
                    continue
                base_record, first_record, second_record, double_record = required
                assert base_record and first_record and second_record and double_record
                base_variant, base_observation = base_record
                _, first_observation = first_record
                _, second_observation = second_record
                double_variant, double_observation = double_record
                source_observations = (
                    base_observation,
                    first_observation,
                    second_observation,
                    double_observation,
                )
                visible_round = max(item.round_revealed for item in source_observations)
                epistasis = float(
                    double_observation.fitness
                    - first_observation.fitness
                    - second_observation.fitness
                    + base_observation.fitness
                )
                mutation_ids = tuple(
                    f"mutation:{context.protein_id}:{reference}{position}{alternate}"
                    for position, reference, alternate in (first, second)
                )
                interaction_id = stable_record_id(
                    "mutation-interaction",
                    context.protein_id,
                    *mutation_ids,
                    base_variant.variant_id,
                )
                estimate_id = stable_record_id(
                    "epistasis-effect",
                    context.run_id,
                    context.assay_id,
                    interaction_id,
                    *(_observation_record_id(context, item) for item in source_observations),
                )
                source_id = "derived:complete-four-state-epistasis"
                entities[interaction_id] = EntityRecord(
                    interaction_id,
                    "MutationInteraction",
                    KnowledgeLayer.EXPERIMENTAL,
                    frozenset({Modality.SEQUENCE, Modality.TABULAR}),
                    {
                        "background_variant_id": base_variant.variant_id,
                        "double_variant_id": double_variant.variant_id,
                        "mutation_ids": mutation_ids,
                    },
                    (source_id,),
                    "derived_effect",
                    valid_from_round=visible_round,
                )
                entities[estimate_id] = EntityRecord(
                    estimate_id,
                    "EffectEstimate",
                    KnowledgeLayer.EXPERIMENTAL,
                    frozenset({Modality.TABULAR, Modality.TIME_SERIES}),
                    {
                        "effect_type": "pairwise_epistasis",
                        "epistasis": epistasis,
                        "direction": "positive" if epistasis > 0 else "negative" if epistasis < 0 else "additive",
                        "method": "complete_four_state_difference",
                        "requires_complete_observation_square": True,
                    },
                    (source_id,),
                    "derived_effect",
                    valid_from_round=visible_round,
                )
                context_id = f"variant:{base_variant.variant_id}"
                for mutation_id in mutation_ids:
                    relations.append(
                        _relation(
                            self.name,
                            interaction_id,
                            "INCLUDES_MUTATION",
                            mutation_id,
                            KnowledgeLayer.EXPERIMENTAL,
                            modality=Modality.SEQUENCE,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )
                relations.extend(
                    (
                        _relation(
                            self.name,
                            interaction_id,
                            "HAS_EPISTASIS_ESTIMATE",
                            estimate_id,
                            KnowledgeLayer.EXPERIMENTAL,
                            modality=Modality.TABULAR,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        ),
                        _relation(
                            self.name,
                            interaction_id,
                            "IN_BACKGROUND",
                            f"variant:{base_variant.variant_id}",
                            KnowledgeLayer.EXPERIMENTAL,
                            modality=Modality.SEQUENCE,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        ),
                    )
                )
                if context.assay_id:
                    relations.append(
                        _relation(
                            self.name,
                            estimate_id,
                            "MEASURED_IN",
                            f"assay:{context.assay_id}",
                            KnowledgeLayer.EXPERIMENTAL,
                            modality=Modality.TABULAR,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )
                for source_observation in source_observations:
                    relations.append(
                        _relation(
                            self.name,
                            estimate_id,
                            "DERIVED_FROM",
                            _observation_record_id(context, source_observation),
                            KnowledgeLayer.PROVENANCE,
                            modality=Modality.TABULAR,
                            source_id=source_id,
                            source_group="derived_effect",
                            context_id=context_id,
                            valid_from_round=visible_round,
                        )
                    )

        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))


class LocalRAGKnowledgeAdapter:
    """Materialize only policy-approved, current-round local retrieval results."""

    name = "local_rag"

    def __init__(
        self,
        guard: TargetLeakageGuard | None = None,
        *,
        publication_catalog: PublicationCatalog | None = None,
    ) -> None:
        self.guard = guard
        self.publication_catalog = publication_catalog or PublicationCatalog({})

    @staticmethod
    def _chunk_source_id(document_id: str, chunk_id: str) -> str:
        del document_id
        return f"source:local_rag:{chunk_id}"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        results = tuple(context.resources.get("local_retrieval_results", ()))
        if not all(isinstance(item, RetrievalResult) for item in results):
            raise TypeError(
                "resources['local_retrieval_results'] must contain RetrievalResult records"
            )
        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        for result in results:
            if result.round_id != context.round_id:
                raise ValueError(
                    f"Local retrieval {result.query_id} belongs to round {result.round_id}, "
                    f"not {context.round_id}"
                )
            if not bool(result.policy_decision.get("allowed", False)):
                continue
            chunks = {item.chunk_id: item for item in result.chunks}
            for chunk in result.chunks:
                if (
                    self.guard is not None
                    and self.guard.enabled
                    and self.guard.config.block_target_entities
                ):
                    matched = self.guard.validate_result(
                        text=chunk.text, path=chunk.artifact_uri
                    )
                    if matched:
                        raise ValueError(
                            f"Target leakage guard rejected KG chunk {chunk.chunk_id}: {matched}"
                        )
                source_id = self._chunk_source_id(chunk.document_id, chunk.chunk_id)
                document_metadata = chunk.provenance.get("metadata", {})
                if not isinstance(document_metadata, dict):
                    document_metadata = {}
                entities[chunk.document_id] = EntityRecord(
                    entity_id=chunk.document_id,
                    entity_type="Document",
                    layer=KnowledgeLayer.LITERATURE,
                    modalities=frozenset({Modality.TEXT}),
                    properties={
                        "artifact_uri": chunk.artifact_uri,
                        "file_hash": chunk.provenance.get("file_hash"),
                        "knowledge_type": chunk.knowledge_type,
                        "metadata": document_metadata,
                        "index_manifest_hash": result.index_manifest_hash,
                    },
                    source_ids=(source_id,),
                    source_group=chunk.source_group,
                    confidence=1.0,
                    valid_from_round=result.round_id,
                )
                entities[chunk.chunk_id] = EntityRecord(
                    entity_id=chunk.chunk_id,
                    entity_type="DocumentChunk",
                    layer=KnowledgeLayer.LITERATURE,
                    modalities=frozenset(
                        {Modality.TEXT, Modality.EMBEDDING}
                        if "dense" in chunk.scores
                        else {Modality.TEXT}
                    ),
                    properties={
                        "text": chunk.text,
                        "artifact_uri": chunk.artifact_uri,
                        "section_path": chunk.section_path,
                        "start_offset": chunk.start_offset,
                        "end_offset": chunk.end_offset,
                        "knowledge_type": chunk.knowledge_type,
                        "document_metadata": document_metadata,
                        "scores": chunk.scores,
                        "query_id": result.query_id,
                        "policy_decision": result.policy_decision,
                    },
                    source_ids=(source_id,),
                    source_group=chunk.source_group,
                    confidence=min(
                        1.0,
                        max(
                            0.0,
                            float(chunk.scores.get("retrieval_confidence", 0.0)),
                        ),
                    ),
                    valid_from_round=result.round_id,
                )
                relations.append(
                    _relation(
                        self.name,
                        chunk.document_id,
                        "HAS_CHUNK",
                        chunk.chunk_id,
                        KnowledgeLayer.LITERATURE,
                        modality=Modality.TEXT,
                        source_id=source_id,
                        source_group=chunk.source_group,
                        context_id=result.query_id,
                        valid_from_round=result.round_id,
                    )
                )
            for claim in result.claims:
                supporting_chunks = [
                    chunks[item] for item in claim.evidence_chunk_ids if item in chunks
                ]
                if not supporting_chunks:
                    continue
                if (
                    self.guard is not None
                    and self.guard.enabled
                    and self.guard.config.block_target_entities
                ):
                    matched = self.guard.matches(text=claim.statement)
                    if matched:
                        raise ValueError(
                            f"Target leakage guard rejected KG claim {claim.claim_id}: {matched}"
                        )
                source_ids = tuple(
                    self._chunk_source_id(item.document_id, item.chunk_id)
                    for item in supporting_chunks
                )
                source_group = supporting_chunks[0].source_group
                knowledge_types = sorted(
                    {item.knowledge_type for item in supporting_chunks}
                )
                entities[claim.claim_id] = EntityRecord(
                    entity_id=claim.claim_id,
                    entity_type="Claim",
                    layer=KnowledgeLayer.LITERATURE,
                    modalities=frozenset({Modality.TEXT}),
                    properties={
                        "statement": claim.statement,
                        "subject": claim.subject,
                        "predicate": claim.predicate,
                        "object": claim.object,
                        "polarity": claim.polarity,
                        "applicability": claim.applicability,
                        "knowledge_types": knowledge_types,
                        "claim_kind": claim.claim_kind,
                        "selection_eligible": claim.selection_eligible,
                        "extraction_version": claim.extraction_version,
                    },
                    source_ids=source_ids,
                    source_group=source_group,
                    confidence=claim.confidence,
                    valid_from_round=result.round_id,
                )
                evidence_id = stable_record_id(
                    "local-rag-evidence",
                    claim.claim_id,
                    supporting_chunks[0].chunk_id,
                )
                evidence_entity_id = f"evidence:{evidence_id}"
                entities[evidence_entity_id] = EntityRecord(
                    entity_id=evidence_entity_id,
                    entity_type="Evidence",
                    layer=KnowledgeLayer.AGENT,
                    modalities=frozenset({Modality.TEXT}),
                    properties={
                        "channel": "local_rag",
                        "statement": claim.statement,
                        "claim_id": claim.claim_id,
                        "knowledge_types": knowledge_types,
                        "contributes_to_selection": False,
                        "selection_projection_required": claim.selection_eligible,
                        "round_id": result.round_id,
                    },
                    source_ids=source_ids,
                    source_group=source_group,
                    confidence=claim.confidence,
                    valid_from_round=result.round_id,
                )
                relations.append(
                    RelationRecord(
                        relation_id=stable_record_id(
                            "rel", self.name, claim.claim_id, evidence_entity_id, result.query_id
                        ),
                        subject_id=claim.claim_id,
                        predicate="SUPPORTED_BY_SOURCE",
                        object_id=evidence_entity_id,
                        layer=KnowledgeLayer.LITERATURE,
                        modalities=frozenset({Modality.TEXT}),
                        source_ids=source_ids,
                        evidence_ids=(evidence_id,),
                        source_group=source_group,
                        confidence=claim.confidence,
                        context_id=result.query_id,
                        valid_from_round=result.round_id,
                    )
                )
                for support in claim.citation_support:
                    publication_id = str(support.get("publication_id", "")).casefold()
                    publication = self.publication_catalog.require(publication_id)
                    support_id = str(
                        support.get("support_id")
                        or stable_record_id(
                            "citation-support",
                            claim.claim_id,
                            publication_id,
                            support.get("support_type"),
                            support.get("locator"),
                        )
                    )
                    entities[publication_id] = EntityRecord(
                        entity_id=publication_id,
                        entity_type="Publication",
                        layer=KnowledgeLayer.LITERATURE,
                        modalities=frozenset({Modality.TEXT}),
                        properties={
                            "title": publication["title"],
                            "authors": publication["authors"],
                            "year": publication["year"],
                            "venue": publication["venue"],
                            "doi": publication["doi"],
                            "url": publication["url"],
                            "publication_type": publication.get("publication_type"),
                            "verification": publication.get("verification", {}),
                        },
                        source_ids=(publication_id,),
                        source_group="publication_catalog",
                        confidence=1.0,
                    )
                    entities[support_id] = EntityRecord(
                        entity_id=support_id,
                        entity_type="CitationSupport",
                        layer=KnowledgeLayer.PROVENANCE,
                        modalities=frozenset({Modality.TEXT}),
                        properties={
                            "support_type": support.get("support_type"),
                            "locator": support.get("locator"),
                            "verified_against_source": bool(
                                support.get("verified_against_source", False)
                            ),
                            "claim_id": claim.claim_id,
                            "publication_id": publication_id,
                        },
                        source_ids=(publication_id, *source_ids),
                        source_group="citation_support",
                        confidence=claim.confidence,
                        valid_from_round=result.round_id,
                    )
                    relations.extend(
                        (
                            _relation(
                                self.name,
                                claim.claim_id,
                                "SUPPORTED_BY_CITATION",
                                support_id,
                                KnowledgeLayer.LITERATURE,
                                modality=Modality.TEXT,
                                source_id=publication_id,
                                source_group="citation_support",
                                context_id=result.query_id,
                                valid_from_round=result.round_id,
                            ),
                            _relation(
                                self.name,
                                support_id,
                                "CITES_PUBLICATION",
                                publication_id,
                                KnowledgeLayer.PROVENANCE,
                                modality=Modality.TEXT,
                                source_id=publication_id,
                                source_group="publication_catalog",
                                context_id=result.query_id,
                                valid_from_round=result.round_id,
                            ),
                            _relation(
                                self.name,
                                support_id,
                                "DERIVED_FROM",
                                supporting_chunks[0].chunk_id,
                                KnowledgeLayer.PROVENANCE,
                                modality=Modality.TEXT,
                                source_id=source_ids[0],
                                source_group=source_group,
                                context_id=result.query_id,
                                valid_from_round=result.round_id,
                            ),
                        )
                    )
                for chunk in supporting_chunks:
                    source_id = self._chunk_source_id(chunk.document_id, chunk.chunk_id)
                    relations.append(
                        _relation(
                            self.name,
                            chunk.chunk_id,
                            "ASSERTS",
                            claim.claim_id,
                            KnowledgeLayer.LITERATURE,
                            modality=Modality.TEXT,
                            source_id=source_id,
                            source_group=chunk.source_group,
                            context_id=result.query_id,
                            valid_from_round=result.round_id,
                        )
                    )
                    relations.append(
                        _relation(
                            self.name,
                            evidence_entity_id,
                            "DERIVED_FROM",
                            chunk.chunk_id,
                            KnowledgeLayer.PROVENANCE,
                            modality=Modality.TEXT,
                            source_id=source_id,
                            source_group=chunk.source_group,
                            context_id=result.query_id,
                            valid_from_round=result.round_id,
                        )
                    )
        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))


class ValidationKnowledgeAdapter:
    """Expose sample facts and hypothesis-level learning as distinct KG records."""

    name = "validation_records"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        records = tuple(context.resources.get("validation_records", ()))
        assessments = tuple(context.resources.get("hypothesis_assessments", ()))
        reflections = tuple(context.resources.get("hypothesis_reflections", ()))
        observations = tuple(context.resources.get("observations", ()))
        if not all(isinstance(item, ValidationRecord) for item in records):
            raise TypeError("resources['validation_records'] must contain ValidationRecord records")
        if not all(isinstance(item, HypothesisAssessment) for item in assessments):
            raise TypeError(
                "resources['hypothesis_assessments'] must contain HypothesisAssessment records"
            )
        if not all(isinstance(item, HypothesisReflection) for item in reflections):
            raise TypeError(
                "resources['hypothesis_reflections'] must contain HypothesisReflection records"
            )
        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        observation_lookup = {
            item.variant_id: _observation_record_id(context, item)
            for item in observations
            if isinstance(item, FitnessObservation)
        }
        for assessment in assessments:
            source_id = f"assessment:{assessment.evaluator_version}"
            entities[assessment.assessment_id] = EntityRecord(
                assessment.assessment_id,
                "HypothesisAssessment",
                KnowledgeLayer.AGENT,
                frozenset({Modality.TABULAR, Modality.TIME_SERIES}),
                {
                    "hypothesis_id": assessment.hypothesis_id,
                    "falsification_spec_id": assessment.falsification_spec_id,
                    "round_id": assessment.round_id,
                    "status": assessment.status.value,
                    "criterion_results": tuple(
                        {
                            "criterion_id": item.criterion_id,
                            "signal": item.signal.value,
                            "metric_value": item.metric_value,
                            "comparator_value": item.comparator_value,
                            "effect_size": item.effect_size,
                            "observation_ids": item.observation_ids,
                            "qc_status": item.qc_status,
                            "detector_name": item.detector_name,
                            "detector_version": item.detector_version,
                            "reason_code": item.reason_code,
                        }
                        for item in assessment.criterion_results
                    ),
                    "observation_ids": assessment.observation_ids,
                    "decisive_criterion_ids": assessment.decisive_criterion_ids,
                    "unresolved_criterion_ids": assessment.unresolved_criterion_ids,
                    "evaluator_version": assessment.evaluator_version,
                },
                (source_id,),
                "hypothesis_assessment",
                valid_from_round=assessment.round_id,
            )
            relations.append(
                _relation(
                    self.name,
                    assessment.assessment_id,
                    "ASSESSES",
                    assessment.hypothesis_id,
                    KnowledgeLayer.AGENT,
                    modality=Modality.TABULAR,
                    source_id=source_id,
                    source_group="hypothesis_assessment",
                    context_id=assessment.assessment_id,
                    valid_from_round=assessment.round_id,
                )
            )
            for observation_id in assessment.observation_ids:
                structured_id = observation_lookup.get(observation_id)
                if structured_id is None:
                    continue
                relations.append(
                    _relation(
                        self.name,
                        structured_id,
                        "CONTRIBUTES_TO_ASSESSMENT",
                        assessment.assessment_id,
                        KnowledgeLayer.PROVENANCE,
                        modality=Modality.TABULAR,
                        source_id=source_id,
                        source_group="hypothesis_assessment",
                        context_id=assessment.assessment_id,
                        valid_from_round=assessment.round_id,
                    )
                )
        for reflection in reflections:
            source_id = f"agent:{reflection.provider}"
            entities[reflection.reflection_id] = EntityRecord(
                reflection.reflection_id,
                "HypothesisReflection",
                KnowledgeLayer.AGENT,
                frozenset({Modality.TEXT, Modality.TIME_SERIES}),
                {
                    "hypothesis_id": reflection.hypothesis_id,
                    "round_id": reflection.round_id,
                    "summary": reflection.summary,
                    "retained_claims": reflection.retained_claims,
                    "invalidated_assumptions": reflection.invalidated_assumptions,
                    "unresolved_questions": reflection.unresolved_questions,
                    "recommended_actions": reflection.recommended_actions,
                    "supporting_observation_ids": reflection.supporting_observation_ids,
                    "supporting_evidence_ids": reflection.supporting_evidence_ids,
                    "provider": reflection.provider,
                    "assessment_id": reflection.assessment_id,
                    "assessment_status": reflection.assessment_status,
                    "quality_status": reflection.quality_status,
                    "advisory_only": reflection.advisory_only,
                    "selection_eligible": reflection.selection_eligible,
                    "dimension_assessments": reflection.dimension_assessments,
                    "dimension_group_advice": reflection.dimension_group_advice,
                },
                (source_id,),
                "rethink_agent",
                valid_from_round=reflection.round_id,
            )
            relations.extend(
                (
                    _relation(
                        self.name,
                        reflection.reflection_id,
                        "REFLECTS_ON",
                        reflection.hypothesis_id,
                        KnowledgeLayer.AGENT,
                        modality=Modality.TEXT,
                        source_id=source_id,
                        source_group="rethink_agent",
                        context_id=reflection.reflection_id,
                        valid_from_round=reflection.round_id,
                    ),
                    _relation(
                        self.name,
                        reflection.reflection_id,
                        "EXPLAINS_ASSESSMENT",
                        reflection.assessment_id,
                        KnowledgeLayer.AGENT,
                        modality=Modality.TEXT,
                        source_id=source_id,
                        source_group="rethink_agent",
                        context_id=reflection.reflection_id,
                        valid_from_round=reflection.round_id,
                    ),
                )
            )
            for evidence_id in reflection.supporting_evidence_ids:
                relations.append(
                    _relation(
                        self.name,
                        reflection.reflection_id,
                        "GROUNDED_IN",
                        evidence_id,
                        KnowledgeLayer.PROVENANCE,
                        modality=Modality.TEXT,
                        source_id=source_id,
                        source_group="rethink_agent",
                        context_id=reflection.reflection_id,
                        valid_from_round=reflection.round_id,
                    )
                )
        for record in records:
            layer = (
                KnowledgeLayer.EXPERIMENTAL
                if record.validation_type == "wet"
                else KnowledgeLayer.MODEL
            )
            entities[record.record_id] = EntityRecord(
                record.record_id,
                "WetValidation" if record.validation_type == "wet" else "DryValidation",
                layer,
                frozenset({Modality.TABULAR, Modality.TIME_SERIES}),
                {
                    "validation_type": record.validation_type,
                    "mutation_notation": record.mutation_notation,
                    "value": record.value,
                    "uncertainty": record.uncertainty,
                    "model_version": record.model_version,
                    "base_weight": record.base_weight,
                    "reliability": record.reliability,
                    "agent_reason": record.agent_reason,
                    "hypothesis_id": record.hypothesis_id,
                    "assessment_id": record.assessment_id,
                },
                (record.source_id,),
                "wet_validation" if record.validation_type == "wet" else "dry_validation",
                record.reliability,
                valid_from_round=record.round_id,
            )
            relations.append(
                _relation(
                    self.name,
                    record.record_id,
                    "VALIDATES",
                    f"variant:{record.variant_id}",
                    layer,
                    modality=Modality.TABULAR,
                    source_id=record.source_id,
                    source_group=(
                        "wet_validation" if record.validation_type == "wet" else "dry_validation"
                    ),
                    context_id=record.record_id,
                    valid_from_round=record.round_id,
                )
            )
        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))
