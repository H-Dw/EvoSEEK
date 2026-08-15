from __future__ import annotations

from common import (
    ensure,
    load_config,
    make_evidence,
    make_observations,
    make_predictions,
    parse_args,
    resolve_output,
    variant_grid,
    write_result,
)

from fitness_agents.contracts.schemas import Hypothesis
from fitness_agents.kg_knowledge import (
    AliasNormalizer,
    BuildContext,
    CampaignObservationAdapter,
    EntityRecord,
    InferenceKnowledgeAdapter,
    InMemoryGraphSink,
    KnowledgeAblationConfig,
    KnowledgeGraphBuilder,
    KnowledgeLayer,
    Modality,
    StaticKnowledgeAdapter,
)
from fitness_agents.plugin_registry import PluginRegistry


def main() -> None:
    args = parse_args("configs/module_tests/kg_construction.yaml")
    config = load_config(args.config)
    output = resolve_output(config, args.output_dir)
    variants = variant_grid()[:8]
    observations = make_observations(variants[:4], round_revealed=0)
    predictions = make_predictions(variants[4:], model_version="kg-construction:v1")
    evidence_map = make_evidence(variants[4:])
    evidence = tuple(item for bundle in evidence_map.values() for item in bundle)
    hypothesis = Hypothesis(
        "hyp:kg-build",
        "External and campaign evidence jointly support a bounded test.",
        {39: ("A",), 40: ("W",)},
        tuple(item.evidence_id for item in evidence[:3]),
        "The selected batch should improve median fitness.",
        "Contradict if the selected median does not improve.",
    )
    context = BuildContext(
        run_id=str(config["run_id"]),
        round_id=int(config["round_id"]),
        protein_id=str(config["protein_id"]),
        assay_id=str(config["assay_id"]),
        condition_id=str(config["condition_id"]),
        resources={
            "variants": variants,
            "observations": observations,
            "predictions": predictions,
            "evidence": evidence,
            "hypotheses": (hypothesis,),
        },
    )

    alias_id = "protein:GB1_ALIAS"
    registry = PluginRegistry("knowledge_adapter")
    registry.register("campaign_observations", CampaignObservationAdapter())
    registry.register("inference_records", InferenceKnowledgeAdapter())
    registry.register(
        "external_alias",
        StaticKnowledgeAdapter(
            "external_alias",
            entities=(
                EntityRecord(
                    alias_id,
                    "Protein",
                    KnowledgeLayer.IDENTITY,
                    frozenset({Modality.SEQUENCE}),
                    {"external_accession": "GB1_ALIAS"},
                    ("external-db:record-1",),
                    "external-db",
                    0.7,
                ),
            ),
        ),
    )
    sink = InMemoryGraphSink()
    full_builder = KnowledgeGraphBuilder(
        registry,
        config=KnowledgeAblationConfig.from_mapping(config["full_profile"]),
        normalizers=(AliasNormalizer({alias_id: "protein:GB1"}),),
        sinks=(sink,),
        strict=True,
    )
    full = full_builder.build(context)
    ensure(sink.snapshot == full.snapshot, "Graph sink did not receive the built snapshot")
    entity_ids = [item.entity_id for item in full.snapshot.entities]
    ensure(len(entity_ids) == len(set(entity_ids)), "Fusion left duplicate entity IDs")
    protein = next(item for item in full.snapshot.entities if item.entity_id == "protein:GB1")
    ensure(
        protein.properties.get("external_accession") == "GB1_ALIAS",
        "Alias normalization/fusion lost the external record",
    )
    ensure(
        not [item for item in full.report.validation_issues if item.severity == "error"],
        "Core KG validation reported errors",
    )
    ensure(
        {"Observation", "Prediction", "Evidence", "Hypothesis"}.issubset(
            {item.entity_type for item in full.snapshot.entities}
        ),
        "Full KG profile missed a core entity family",
    )

    observations_only = KnowledgeGraphBuilder(
        registry,
        config=KnowledgeAblationConfig.from_mapping(config["observations_only_profile"]),
        strict=True,
    ).build(context)
    ensure(
        {"inference_records", "external_alias"}.issubset(observations_only.report.skipped_adapters),
        "Adapter ablation did not skip inference/external adapters",
    )
    ensure(
        "Prediction" not in {item.entity_type for item in observations_only.snapshot.entities},
        "Observations-only profile retained prediction entities",
    )

    write_result(
        output,
        "kg_construction",
        {
            "config": config["_config_path"],
            "full_snapshot": {
                "entities": len(full.snapshot.entities),
                "relations": len(full.snapshot.relations),
                "entity_types": sorted({item.entity_type for item in full.snapshot.entities}),
                "adapter_counts": full.report.adapter_counts,
                "dropped_dangling_relations": full.report.dropped_dangling_relations,
            },
            "observations_only_snapshot": {
                "entities": len(observations_only.snapshot.entities),
                "relations": len(observations_only.snapshot.relations),
                "skipped_adapters": observations_only.report.skipped_adapters,
            },
            "alias_fused": True,
        },
    )


if __name__ == "__main__":
    main()

