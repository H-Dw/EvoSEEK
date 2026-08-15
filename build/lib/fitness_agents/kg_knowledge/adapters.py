from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Protocol

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    Variant,
)

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
    _mutation_pattern = re.compile(r"([A-Z])([0-9]+)([A-Z*])")

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
            for match in self._mutation_pattern.finditer(variant.mutation_notation):
                reference, position_text, alternate = match.groups()
                position = int(position_text)
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
                    "evidence_type": item.evidence_type,
                    "round_id": item.round_id,
                },
                (item.source_id,),
                item.channel,
                item.confidence,
                valid_from_round=item.round_id,
            )
            relations.append(
                _relation(
                    self.name,
                    entity_id,
                    "ABOUT",
                    f"variant:{item.variant_id}",
                    KnowledgeLayer.AGENT,
                    modality=Modality.TEXT,
                    source_id=item.source_id,
                    source_group=item.channel,
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
                predicate = "SUPPORTED_BY" if evidence.score >= 0 else "CONTRADICTED_BY"
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
