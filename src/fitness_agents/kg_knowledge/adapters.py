from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    ReThinkReflection,
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
        source_ids=(source_id,),
        source_group=source_group,
        context_id=context_id,
        valid_from_round=valid_from_round,
    )


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
            try:
                mutation_edits = parse_mutation_notation(variant.mutation_notation)
            except InvalidMutationNotation:
                mutation_edits = ()
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
            observation_id = stable_record_id(
                "observation",
                context.run_id,
                observation.variant_id,
                observation.round_revealed,
                observation.source,
            )
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
    """Convert model outputs and agent evidence/hypotheses into versioned KG records."""

    name = "inference_records"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        predictions = tuple(context.resources.get("predictions", ()))
        evidence_items = tuple(context.resources.get("evidence", ()))
        hypotheses = tuple(context.resources.get("hypotheses", ()))
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
        normalized_document = (
            document_id if document_id.startswith("localdoc:") else f"localdoc:{document_id}"
        )
        return f"{normalized_document}:{chunk_id}"

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
                evidence_id = f"ev:local_rag:{supporting_chunks[0].chunk_id.split(':', 1)[-1]}"
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
    """Expose append-only wet/dry validation and ReThink records in the external KG."""

    name = "validation_records"

    def extract(self, context: BuildContext) -> KnowledgeBatch:
        records = tuple(context.resources.get("validation_records", ()))
        reflections = tuple(context.resources.get("reflections", ()))
        if not all(isinstance(item, ValidationRecord) for item in records):
            raise TypeError("resources['validation_records'] must contain ValidationRecord records")
        if not all(isinstance(item, ReThinkReflection) for item in reflections):
            raise TypeError("resources['reflections'] must contain ReThinkReflection records")
        entities: dict[str, EntityRecord] = {}
        relations: list[RelationRecord] = []
        reflection_lookup = {item.reflection_id: item for item in reflections}
        for reflection in reflections:
            entities[reflection.reflection_id] = EntityRecord(
                reflection.reflection_id,
                "ReThinkReflection",
                KnowledgeLayer.AGENT,
                frozenset({Modality.TEXT, Modality.TIME_SERIES}),
                {
                    "variant_id": reflection.variant_id,
                    "round_id": reflection.round_id,
                    "verdict": reflection.verdict,
                    "summary": reflection.summary,
                    "positive_findings": reflection.positive_findings,
                    "negative_findings": reflection.negative_findings,
                    "revised_reason": reflection.revised_reason,
                    "next_round_advice": reflection.next_round_advice,
                    "provider": reflection.provider,
                },
                (f"agent:{reflection.provider}",),
                "rethink_agent",
                valid_from_round=reflection.round_id,
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
                    "reflection_verdict": record.reflection_verdict,
                    "reflection_summary": record.reflection_summary,
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
            if record.reflection_id and record.reflection_id in reflection_lookup:
                relations.append(
                    _relation(
                        self.name,
                        record.record_id,
                        "REFLECTED_BY",
                        record.reflection_id,
                        KnowledgeLayer.AGENT,
                        modality=Modality.TEXT,
                        source_id=f"agent:{reflection_lookup[record.reflection_id].provider}",
                        source_group="rethink_agent",
                        context_id=record.record_id,
                        valid_from_round=record.round_id,
                    )
                )
        return KnowledgeBatch(self.name, tuple(entities.values()), tuple(relations))
