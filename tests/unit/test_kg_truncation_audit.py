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
