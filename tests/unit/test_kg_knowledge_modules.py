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
    SiteFeatureKnowledgeAdapter,
    StaticKnowledgeAdapter,
    stable_record_id,
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


def test_effect_estimates_and_feature_channels_materialize_typed_mutation_knowledge():
    variants = [
        Variant("wt", "AA", "AAAA", "WT", 0, "observed"),
        Variant("a", "WA", "WAAA", "A3W", 1, "observed"),
        Variant("b", "AF", "AAAF", "A4F", 1, "observed"),
        Variant("ab", "WF", "WAAF", "A3W;A4F", 2, "observed"),
    ]
    observations = [
        FitnessObservation("wt", 0.0, "observed", 0, "wet"),
        FitnessObservation("a", 1.0, "observed", 0, "wet"),
        FitnessObservation("b", 0.5, "observed", 0, "wet"),
        FitnessObservation("ab", 2.0, "observed", 1, "wet"),
    ]
    evidence = [
        Evidence(
            "e-physchem",
            "ab",
            "physchem",
            "descriptor",
            0.1,
            "aaindex:test",
            0.0,
            2,
            raw_features={
                "sites": {
                    "3": {
                        "mutation": "A3W",
                        "deltas": {"hydropathy": 1.2},
                        "wild_type_values": {"hydropathy": 0.2},
                        "mutant_values": {"hydropathy": 1.4},
                    }
                },
                "property_accessions": {"hydropathy": "TEST0001"},
            },
            contributes_to_selection=False,
            provenance={"resource_sha256": "physchem-hash"},
        ),
        Evidence(
            "e-conservation",
            "ab",
            "conservation",
            "profile",
            -0.2,
            "msa:test",
            0.0,
            2,
            raw_features={
                "sites": {
                    "3": {
                        "wild_type_frequency": 0.7,
                        "mutant_frequency": 0.1,
                        "log_odds_vs_wild_type": -1.9,
                        "entropy": 0.4,
                        "coverage": 1.0,
                        "gap_fraction": 0.0,
                    }
                },
                "sequence_count": 24,
                "neff": 12.0,
            },
            contributes_to_selection=False,
            provenance={"resource_sha256": "msa-hash"},
        ),
        Evidence(
            "e-structure",
            "ab",
            "structure",
            "environment",
            -1.0,
            "structure:1PGB",
            0.0,
            2,
            raw_features={
                "sites": {
                    "3": {
                        "status": "ok",
                        "mutation": "A3W",
                        "structure_chain": "A",
                        "structure_residue": 3,
                        "contact_count": 6,
                        "sasa_angstrom2": 12.0,
                        "mutant_side_chain_not_modelled": True,
                    }
                },
                "resource_id": "rcsb:1PGB",
            },
            contributes_to_selection=False,
            provenance={"resource_sha256": "structure-hash"},
        ),
    ]
    context = BuildContext(
        "run-effects",
        2,
        "PTEST",
        "binding",
        resources={
            "variants": variants,
            "observations": observations,
            "evidence": evidence,
        },
    )
    registry = PluginRegistry("knowledge_adapter")
    registry.register("campaign_observations", CampaignObservationAdapter())
    registry.register("inference_records", InferenceKnowledgeAdapter())

    snapshot = KnowledgeGraphBuilder(registry).build(context).snapshot
    entity_types = {item.entity_type for item in snapshot.entities}
    assert {
        "MutationEffectEstimate",
        "MutationInteraction",
        "EffectEstimate",
        "ResidueType",
        "PhyschemPropertyValue",
        "SubstitutionDescriptor",
        "EvolutionProfile",
        "ResidueEnvironment",
    } <= entity_types
    predicates = {item.predicate for item in snapshot.relations}
    assert {
        "ABOUT_MUTATION",
        "IN_BACKGROUND",
        "DERIVED_FROM",
        "HAS_DESCRIPTOR",
        "HAS_PHYSCHEM_DELTA",
        "HAS_EVOLUTIONARY_CONTEXT",
        "OCCURS_IN_ENVIRONMENT",
        "HAS_EPISTASIS_ESTIMATE",
    } <= predicates
    epistasis = next(
        item.properties["epistasis"]
        for item in snapshot.entities
        if item.entity_type == "EffectEstimate"
    )
    assert epistasis == pytest.approx(0.5)


def test_site_feature_adapter_writes_position_tables_without_combinatorial_variants():
    context = BuildContext(
        "run-sites",
        1,
        "PTEST",
        resources={
            "site_feature_tables": {
                "physchem": {
                    "source_id": "aaindex:test",
                    "resource_sha256": "physchem-hash",
                    "property_accessions": {"hydropathy": "HOPT810101"},
                    "positions": {
                        "3": {
                            "wild_type": "A",
                            "substitutions": {
                                "W": {
                                    "mutation": "A3W",
                                    "deltas": {"hydropathy": 1.2},
                                    "wild_type_values": {"hydropathy": 0.2},
                                    "mutant_values": {"hydropathy": 1.4},
                                }
                            },
                        }
                    },
                },
                "conservation": {
                    "resource_sha256": "msa-hash",
                    "sequence_count": 10,
                    "neff": 5.0,
                    "positions": {
                        "3": {
                            "wild_type": "A",
                            "residues": {
                                "A": {"log_odds_vs_wild_type": 0.0, "entropy": 0.4},
                                "W": {"log_odds_vs_wild_type": -1.2, "entropy": 0.4},
                            },
                        }
                    },
                },
                "structure": {
                    "resource_id": "rcsb:1PGB",
                    "resource_sha256": "structure-hash",
                    "positions": {
                        "3": {
                            "wild_type": "A",
                            "status": "ok",
                            "sasa_angstrom2": 12.0,
                            "contact_count": 6,
                        }
                    },
                },
            }
        },
    )
    batch = SiteFeatureKnowledgeAdapter().extract(context)
    entity_types = {item.entity_type for item in batch.entities}
    predicates = {item.predicate for item in batch.relations}
    assert {
        "ResiduePosition",
        "Mutation",
        "SubstitutionDescriptor",
        "EvolutionProfile",
        "ResidueEnvironment",
    } <= entity_types
    assert "Variant" not in entity_types
    assert {
        "HAS_PHYSCHEM_DELTA",
        "HAS_EVOLUTIONARY_CONTEXT",
        "HAS_EVOLUTION_PROFILE",
        "OCCURS_IN_ENVIRONMENT",
    } <= predicates


def test_feature_semantics_dedupe_substitution_entities_across_combo_variants():
    variants = [
        Variant("a", "WF", "WAAF", "A3W;A4F", 2, "observed"),
        Variant("b", "WA", "WAAA", "A3W", 1, "observed"),
    ]
    evidence = [
        Evidence(
            "e-a",
            "a",
            "physchem",
            "descriptor",
            0.1,
            "aaindex:test",
            0.0,
            1,
            raw_features={
                "sites": {
                    "3": {
                        "mutation": "A3W",
                        "deltas": {"hydropathy": 1.2},
                        "wild_type_values": {"hydropathy": 0.2},
                        "mutant_values": {"hydropathy": 1.4},
                    }
                },
                "property_accessions": {"hydropathy": "HOPT810101"},
            },
            provenance={"resource_sha256": "physchem-hash"},
        ),
        Evidence(
            "e-b",
            "b",
            "physchem",
            "descriptor",
            0.2,
            "aaindex:test",
            0.0,
            1,
            raw_features={
                "sites": {
                    "3": {
                        "mutation": "A3W",
                        "deltas": {"hydropathy": 1.2},
                        "wild_type_values": {"hydropathy": 0.2},
                        "mutant_values": {"hydropathy": 1.4},
                    }
                },
                "property_accessions": {"hydropathy": "HOPT810101"},
            },
            provenance={"resource_sha256": "physchem-hash"},
        ),
    ]
    context = BuildContext(
        "run-dedupe",
        1,
        "PTEST",
        resources={"variants": variants, "evidence": evidence},
    )
    batch = InferenceKnowledgeAdapter().extract(context)
    descriptors = [
        item for item in batch.entities if item.entity_type == "SubstitutionDescriptor"
    ]
    assert len(descriptors) == 1
    assert "evidence_id" not in descriptors[0].properties


def test_effect_estimates_require_visible_matched_backgrounds():
    context = BuildContext(
        "run-incomplete",
        1,
        "PTEST",
        "binding",
        resources={
            "variants": [Variant("a", "WA", "WAAA", "A3W", 1, "observed")],
            "observations": [FitnessObservation("a", 1.0, "observed", 0, "wet")],
        },
    )
    batch = InferenceKnowledgeAdapter().extract(context)
    assert not {
        "MutationEffectEstimate",
        "MutationInteraction",
        "EffectEstimate",
    }.intersection(item.entity_type for item in batch.entities)


def test_semantic_record_ids_are_order_independent_and_collision_free():
    first = stable_record_id("relation", "ABCDEFGH-one-tail", {"b", "a"})
    second = stable_record_id("relation", "ABCDEFGH-one-tail", {"a", "b"})
    colliding_old_slug = stable_record_id("relation", "ABCDEFGH-two-tail", {"a", "b"})

    assert first == second
    assert first != colliding_old_slug
    assert "N02" not in first
