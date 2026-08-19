from __future__ import annotations

import json

from fitness_agents.contracts.mutation_evidence import (
    mutation_evidence_batch_metadata,
    mutation_evidence_card,
)
from fitness_agents.contracts.schemas import Evidence


def _evidence(channel: str, raw_features: dict) -> Evidence:
    return Evidence(
        evidence_id=f"ev:{channel}:1",
        variant_id="sha256:v1",
        channel=channel,
        statement=f"bounded {channel} claim",
        score=0.25,
        source_id=f"{channel}:source",
        confidence=0.7,
        round_id=1,
        evidence_type=("retrieved_document" if channel == "local_rag" else "computed"),
        raw_features=raw_features,
        quality_status="ok",
        applicability="task_context",
        contributes_to_selection=channel != "local_rag",
        warnings=("bounded_warning",),
        provenance={
            "artifact_uri": "file:///evidence.json",
            "artifact_span": [10, 40],
            "provider": "private-provider-detail",
            "embedding_fingerprint": "x" * 4000,
        },
        claim_id="claim:1",
    )


def test_mutation_evidence_cards_keep_decision_fields_and_remove_backend_bulk() -> None:
    cases = {
        "physchem": {
            "sites": {
                "39": {
                    "mutation": "V39W",
                    "deltas": {
                        "residue_mass": 87.1,
                        "hydropathy": -1.2,
                        "nominal_charge": 0.0,
                        "side_chain_volume": 45.0,
                        "unused_descriptor": 99.0,
                    },
                    "wild_type_values": {"huge": "x" * 3000},
                    "mutant_values": {"huge": "x" * 3000},
                }
            },
            "mean_normalized_absolute_delta": 0.4,
            "special_flags": ["aromatic_gain"],
            "global_sequence_deltas": {"molecular_weight_delta_da": 87.1},
            "property_accessions": ["x" * 3000],
            "assay_pH": 7.4,
        },
        "conservation": {
            "sites": {
                "39": {
                    "mutation": "V39W",
                    "coverage": 0.95,
                    "gap_fraction": 0.05,
                    "effective_count": 19.2,
                    "wild_type_frequency": 0.3,
                    "mutant_frequency": 0.1,
                    "log_odds_vs_wild_type": -1.1,
                    "site_quality": "ok",
                }
            },
            "independent_log_odds": -1.1,
            "independent_log_odds_sum": -1.1,
            "independent_mean_log_odds_per_mutation": -1.1,
            "pairwise_enabled": True,
            "pairwise_eligible": True,
            "pairwise_residual_log_odds": 0.2,
            "pairwise_score_method": "marginal_corrected_log_odds",
            "sequence_count": 100,
            "neff": 25.0,
            "neff_per_length": 0.45,
            "cache_status": "hit",
            "estimated_parameters": ["x" * 3000],
        },
        "structure": {
            "sites": {
                "39": {
                    "mutation": "V39W",
                    "status": "ok",
                    "relative_sasa": 0.22,
                    "contact_count": 8,
                    "interface_contact_count": 2,
                    "secondary_structure": "loop",
                    "missing_backbone_atoms": [],
                    "mutant_side_chain_not_modelled": True,
                    "closest_contacts": [{"payload": "x" * 3000}],
                }
            },
            "static_context_flag_count": 1,
            "resource_id": "rcsb:1PGB",
        },
        "kg": {
            "raw_association_score": 0.4,
            "support": 7,
            "global_visible_mean": 0.1,
        },
        "local_rag": {
            "knowledge_type": "directed_evolution_guidance",
            "retrieval_scores": {
                "retrieval_confidence": 0.8,
                "rerank_score": 0.7,
            },
            "full_document": "x" * 6000,
        },
    }

    cards = {
        channel: mutation_evidence_card(_evidence(channel, raw)).model_dump(
            mode="json", exclude_none=True
        )
        for channel, raw in cases.items()
    }

    assert cards["physchem"]["features"]["kind"] == "physchem"
    assert {item["name"] for item in cards["physchem"]["features"]["sites"][0]["deltas"]} == {
        "residue_mass",
        "hydropathy",
        "nominal_charge",
        "side_chain_volume",
    }
    assert cards["conservation"]["features"]["pairwise_score"] == 0.2
    assert cards["structure"]["features"]["sites"][0]["contact_count"] == 8
    assert cards["kg"]["features"]["support"] == 7
    assert cards["local_rag"]["features"]["retrieval_scores"]
    encoded = json.dumps(cards, ensure_ascii=False)
    for forbidden in (
        "raw_features",
        "property_accessions",
        "wild_type_values",
        "mutant_values",
        "cache_status",
        "estimated_parameters",
        "closest_contacts",
        "full_document",
        "embedding_fingerprint",
        "private-provider-detail",
    ):
        assert forbidden not in encoded


def test_candidate_invariant_metadata_is_hoisted_and_deduplicated() -> None:
    evidence = [
        _evidence(
            "physchem",
            {"assay_pH": 7.4, "sites": {}, "property_accessions": ["unused"]},
        ),
        _evidence(
            "physchem",
            {"assay_pH": 7.4, "sites": {}, "property_accessions": ["unused"]},
        ),
        _evidence(
            "conservation",
            {
                "sequence_count": 100,
                "neff": 25.0,
                "neff_per_length": 0.45,
                "pairwise_enabled": True,
                "pairwise_eligible": True,
                "pairwise_score_method": "marginal_corrected_log_odds",
            },
        ),
        _evidence("structure", {"resource_id": "rcsb:1PGB", "sites": {}}),
    ]

    metadata = mutation_evidence_batch_metadata(evidence).model_dump(mode="json")

    assert metadata["assay_pH_values"] == [7.4]
    assert metadata["structure_resource_ids"] == ["rcsb:1PGB"]
    assert metadata["conservation_profiles"] == [
        {
            "sequence_count": 100,
            "neff": 25.0,
            "neff_per_length": 0.45,
            "pairwise_enabled": True,
            "pairwise_eligible": True,
            "pairwise_score_method": "marginal_corrected_log_odds",
        }
    ]
