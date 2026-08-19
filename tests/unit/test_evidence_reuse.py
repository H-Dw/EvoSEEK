from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.knowledge import KnowledgeEngine
from fitness_agents.loop import CampaignRunner
from fitness_agents.loop.orchestrator import _decision_ingest_variants, _flatten_evidence


def test_decision_ingest_variants_stay_near_observed_and_selected_batch():
    observed = [
        Variant("obs-a", "VDGV", "VDGV", "WT", 0, "initial_observed"),
        Variant("obs-b", "ADGV", "ADGV", "V39A", 1, "initial_observed"),
    ]
    selected = [Variant("sel", "WDGV", "WDGV", "V39W", 1, "oracle_pool")]
    remaining = [
        Variant(f"rest-{index}", "FDGV", "FDGV", "V39F", 1, "oracle_pool")
        for index in range(20)
    ]
    observations = [
        FitnessObservation("obs-a", 0.1, "initial_observed", 0),
        FitnessObservation("obs-b", 0.9, "initial_observed", 0),
    ]
    ingested = _decision_ingest_variants(
        observed,
        selected_variants=selected,
        observations=observations,
    )
    ingest_ids = {item.variant_id for item in ingested}
    assert ingest_ids == {"obs-a", "obs-b", "sel"}
    assert not {item.variant_id for item in remaining} & ingest_ids


def test_flatten_evidence_uses_fixed_limit_not_pool_times_channels():
    class _Item:
        def __init__(self, evidence_id: str, channel: str) -> None:
            self.evidence_id = evidence_id
            self.channel = channel
            self.quality_status = "ok"
            self.contributes_to_selection = False
            self.confidence = 0.5
            self.score = 0.2

    evidence = {
        f"v{index}": [_Item(f"e{index}-{channel}", channel) for channel in ("physchem", "kg")]
        for index in range(200)
    }
    flattened = _flatten_evidence(evidence, limit=120)
    assert len(flattened) == 120


def test_orchestrator_scores_static_channels_only_on_observed_and_selected(
    config_factory, monkeypatch
):
    calls: list[dict[str, object]] = []
    sync_sizes: list[int] = []
    original_evidence_for = KnowledgeEngine.evidence_for
    original_sync = KnowledgeEngine.sync_structured_kg

    def wrapped_evidence_for(
        self,
        variants,
        *,
        round_id,
        delete_evidence=False,
        channels=None,
    ):
        calls.append(
            {
                "count": len(tuple(variants)),
                "channels": None if channels is None else tuple(channels),
                "round_id": round_id,
            }
        )
        return original_evidence_for(
            self,
            variants,
            round_id=round_id,
            delete_evidence=delete_evidence,
            channels=channels,
        )

    def wrapped_sync(self, *, run_id, round_id, variants, **kwargs):
        sync_sizes.append(len(tuple(variants)))
        return original_sync(
            self, run_id=run_id, round_id=round_id, variants=variants, **kwargs
        )

    monkeypatch.setattr(KnowledgeEngine, "evidence_for", wrapped_evidence_for)
    monkeypatch.setattr(KnowledgeEngine, "sync_structured_kg", wrapped_sync)
    config = config_factory(
        rounds=1,
        budget_per_round=4,
        candidate_limit=16,
        run_label="evidence-split",
    )
    CampaignRunner(config).run()

    remaining_kg_calls = [
        item for item in calls if item["channels"] == ("kg",)
    ]
    full_channel_calls = [item for item in calls if item["channels"] is None]
    assert remaining_kg_calls
    assert remaining_kg_calls[0]["count"] == 88
    assert full_channel_calls
    assert full_channel_calls[0]["count"] == 24
    assert any(item["count"] == 4 for item in full_channel_calls)
    assert not any(item["count"] >= 80 and item["channels"] is None for item in calls)
    assert sync_sizes
    assert max(sync_sizes) <= 28
