from fitness_agents.kg_interaction import (
    EvidencePack,
    InteractionResult,
    KGKeywordTruncationAuditor,
    KGQueryContext,
    KGQueryStep,
    KGTruncationAuditOperator,
    QueryIntent,
    runtime_truncation_audit_payload,
)
from fitness_agents.kg_knowledge import (
    EntityRecord,
    KnowledgeGraphSnapshot,
    KnowledgeLayer,
    Modality,
    RelationRecord,
    SQLiteGraphSink,
)


def _sink_with_physchem_rows(tmp_path):
    sink = SQLiteGraphSink(tmp_path / "structured.sqlite")
    entities = tuple(
        EntityRecord(
            f"descriptor:{index}",
            "SubstitutionDescriptor",
            KnowledgeLayer.SEQUENCE,
            frozenset({Modality.TABULAR}),
            {"channel": "physchem", "delta": index},
            ("aaindex:test",),
            "feature:physchem",
            valid_from_round=1,
        )
        for index in range(4)
    )
    relations = tuple(
        RelationRecord(
            f"relation:{index}",
            f"mutation:{index}",
            "HAS_PHYSCHEM_DELTA",
            f"descriptor:{index}",
            KnowledgeLayer.SEQUENCE,
            frozenset({Modality.TABULAR}),
            source_ids=("aaindex:test",),
            source_group="feature:physchem",
            valid_from_round=1,
        )
        for index in range(4)
    )
    sink.write(KnowledgeGraphSnapshot(entities, relations))
    return sink


def test_keyword_auditor_distinguishes_truncation_from_missing_kg_item(tmp_path):
    sink = _sink_with_physchem_rows(tmp_path)
    report = KGKeywordTruncationAuditor(sink).audit(
        ("HAS_PHYSCHEM_DELTA", "HAS_EPISTASIS_ESTIMATE"),
        round_id=1,
        max_rows=3,
        sample_rows=2,
    )

    physchem, epistasis = report.entries
    assert physchem.relation_match_count == 4
    assert physchem.returned_match_count == 3
    assert physchem.truncated
    assert len(physchem.sample_matches) == 2
    assert epistasis.status == "not_found"
    assert not epistasis.truncated
    assert report.any_truncated
    assert report.missing_items == ("HAS_EPISTASIS_ESTIMATE",)
    sink.close()


def test_truncation_audit_operator_is_llm_visible_and_runtime_report_checks_normal_packs(
    tmp_path,
):
    sink = _sink_with_physchem_rows(tmp_path)
    operator = KGTruncationAuditOperator(sink)
    pack = operator.execute(
        KGQueryStep(
            "audit",
            "query_kg_truncation_audit",
            QueryIntent.UNCERTAINTY,
            {"items": ["physchem", "HAS_PHYSCHEM_DELTA"], "sample_rows": 1},
        ),
        KGQueryContext("run", 1, max_rows=3),
    )
    interaction = InteractionResult(
        "plan",
        (
            EvidencePack(
                "feature-query",
                "query_physchem_delta",
                1,
                evidence=({"channel": "physchem", "raw_features": {"deltas": {}}},),
            ),
            pack,
        ),
        ("feature", "audit"),
        (),
        "evidence_sufficient",
    )

    payload = runtime_truncation_audit_payload(
        interaction,
        ("physchem", "HAS_PHYSCHEM_DELTA"),
    )

    assert pack.facts[0]["fact_type"] == "kg_truncation_audit"
    assert payload is not None
    assert payload["any_truncated"]
    assert payload["interaction_presence"][0] == {
        "item": "physchem",
        "present_in_non_audit_packs": True,
        "matching_operators": ("query_physchem_delta",),
    }
    assert not payload["interaction_presence"][1]["present_in_non_audit_packs"]
    sink.close()


def test_keyword_audit_matches_predicate_equality_not_property_substrings(tmp_path):
    sink = _sink_with_physchem_rows(tmp_path)
    physchem = sink.query_keyword(item="physchem", round_id=1, limit=12)
    predicate = sink.query_keyword(item="HAS_PHYSCHEM_DELTA", round_id=1, limit=12)

    assert physchem["status"] == "not_found"
    assert physchem["total_match_count"] == 0
    assert predicate["relation_match_count"] == 4
    assert predicate["status"] == "complete"
    sink.close()


def test_live_only_snapshot_does_not_copy_record_json(tmp_path):
    sink = SQLiteGraphSink(tmp_path / "live-only.sqlite", snapshot_mode="live_only")
    snapshot = KnowledgeGraphSnapshot(
        (
            EntityRecord(
                "descriptor:1",
                "SubstitutionDescriptor",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.TABULAR}),
                {"delta": 1},
                ("aaindex:test",),
                "feature:physchem",
            ),
        ),
        (
            RelationRecord(
                "relation:1",
                "mutation:1",
                "HAS_PHYSCHEM_DELTA",
                "descriptor:1",
                KnowledgeLayer.SEQUENCE,
                frozenset({Modality.TABULAR}),
                source_ids=("aaindex:test",),
                source_group="feature:physchem",
            ),
        ),
    )
    sink.write(snapshot)
    sink.write(snapshot)
    entity_copies = sink.connection.execute(
        "SELECT COUNT(*) FROM snapshot_entity_versions"
    ).fetchone()[0]
    relation_copies = sink.connection.execute(
        "SELECT COUNT(*) FROM snapshot_relation_versions"
    ).fetchone()[0]
    snapshots = sink.connection.execute(
        "SELECT COUNT(*) FROM graph_snapshots"
    ).fetchone()[0]
    assert entity_copies == 0
    assert relation_copies == 0
    assert snapshots == 1
    sink.close()
