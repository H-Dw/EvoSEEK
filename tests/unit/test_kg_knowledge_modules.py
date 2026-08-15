import pytest

from fitness_agents.contracts.schemas import (
    Evidence,
    FitnessObservation,
    Hypothesis,
    Prediction,
    Variant,
)
from fitness_agents.kg_knowledge import (
    DEFAULT_ENTITY_SPECS,
    BuildContext,
    CampaignObservationAdapter,
    EntityRecord,
    InferenceKnowledgeAdapter,
    KnowledgeAblationConfig,
    KnowledgeBatch,
    KnowledgeGraphBuilder,
    KnowledgeLayer,
    Modality,
    ProvenanceAwareFusion,
    RelationRecord,
    StaticKnowledgeAdapter,
)
from fitness_agents.plugin_registry import PluginRegistry


def _context():
    variant = Variant("v1", "A3W", "AAWAA", "A3W", 1, "oracle_pool")
    observation = FitnessObservation("v1", 0.9, "oracle_pool", 1)
    prediction = Prediction("v1", 0.8, 0.1, (0.6, 1.0), 0.2, {"sequence": 0.7}, "m1")
    evidence = Evidence("e1", "v1", "structure", "packing improves", 0.6, "af:test", 0.8, 2)
    hypothesis = Hypothesis(
        "h1", "A3W improves packing", {3: ("W",)}, ("e1",), "+fitness", "fitness <= 0"
    )
    return BuildContext(
        "run1",
        2,
        "PTEST",
        "binding",
        "ph7",
        {
            "variants": [variant],
            "observations": [observation],
            "predictions": [prediction],
            "evidence": [evidence],
            "hypotheses": [hypothesis],
        },
    )


def test_builder_converts_current_records_and_keeps_all_edges_grounded():
    registry = PluginRegistry("knowledge_adapter")
    registry.register("campaign_observations", CampaignObservationAdapter())
    registry.register("inference_records", InferenceKnowledgeAdapter())
    result = KnowledgeGraphBuilder(registry).build(_context())
    entity_types = {item.entity_type for item in result.snapshot.entities}
    assert {
        "Protein",
        "Sequence",
        "ResiduePosition",
        "Variant",
        "Mutation",
        "Assay",
        "Condition",
        "Observation",
        "CampaignRound",
        "Prediction",
        "ModelRun",
        "Evidence",
        "Hypothesis",
    } <= entity_types
    entity_ids = {item.entity_id for item in result.snapshot.entities}
    assert all(
        edge.subject_id in entity_ids and edge.object_id in entity_ids
        for edge in result.snapshot.relations
    )
    assert not [item for item in result.report.validation_issues if item.severity == "error"]


def test_layer_ablation_drops_entities_and_newly_dangling_edges():
    protein = EntityRecord("protein:p", "Protein", KnowledgeLayer.IDENTITY)
    structure = EntityRecord(
        "structure:s", "Structure", KnowledgeLayer.STRUCTURE, frozenset({Modality.STRUCTURE_3D})
    )
    edge = RelationRecord(
        "rel:1",
        "protein:p",
        "HAS_STRUCTURE",
        "structure:s",
        KnowledgeLayer.STRUCTURE,
        frozenset({Modality.STRUCTURE_3D}),
        source_ids=("pdb:1",),
    )
    registry = PluginRegistry("knowledge_adapter")
    registry.register(
        "structure",
        StaticKnowledgeAdapter("structure", entities=[protein, structure], relations=[edge]),
    )
    config = KnowledgeAblationConfig(enabled_layers=frozenset({KnowledgeLayer.IDENTITY}))
    result = KnowledgeGraphBuilder(registry, config=config).build(BuildContext("run", 0, "p"))
    assert [item.entity_id for item in result.snapshot.entities] == ["protein:p"]
    assert result.snapshot.relations == ()
    assert result.report.filtered_entities == 1
    assert result.report.filtered_relations == 1


def test_fusion_counts_independent_sources_but_not_duplicate_source_family():
    records = (
        EntityRecord(
            "protein:p", "Protein", KnowledgeLayer.IDENTITY, confidence=0.6, source_group="db1"
        ),
        EntityRecord(
            "protein:p", "Protein", KnowledgeLayer.IDENTITY, confidence=0.8, source_group="db1"
        ),
        EntityRecord(
            "protein:p", "Protein", KnowledgeLayer.IDENTITY, confidence=0.5, source_group="db2"
        ),
    )
    batches = tuple(KnowledgeBatch(f"a{index}", (record,)) for index, record in enumerate(records))
    snapshot = ProvenanceAwareFusion().fuse(batches)
    assert snapshot.entities[0].confidence == pytest.approx(0.9)


def test_catalog_places_experimental_identity_and_sequence_anchors_first():
    p0_types = {item.entity_type for item in DEFAULT_ENTITY_SPECS if item.priority == "P0"}
    assert {"Protein", "Variant", "Mutation", "Assay", "Observation", "CampaignRound"} <= p0_types
