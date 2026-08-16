from __future__ import annotations

import json

import pandas as pd
import pytest

from fitness_agents.contracts.schemas import Variant
from fitness_agents.data.loader import variants_from_fold_frame
from fitness_agents.mutation.conflicts import ResidueConflictDetector
from fitness_agents.mutation.notation import (
    InvalidMutationNotation,
    MutationEdit,
    edits_from_site_code,
    edits_from_tokens,
    format_canonical,
    parse_mutation_notation,
)


def _identities(edits):
    return {item.identity for item in edits}


def test_flip_site_code_and_canonical_hgvs_round_trip():
    edits = edits_from_site_code("ADGA", wild_type="VDGV", positions=(39, 40, 41, 54))
    assert format_canonical(edits) == "V39A;V54A"
    assert format_canonical(()) == "WT"
    parsed = parse_mutation_notation("V39A;V54A")
    assert _identities(parsed) == _identities(edits)


def test_proteingym_colon_joined_parses_as_hgvs_edits():
    parsed = parse_mutation_notation("A23C:D45E")
    assert [item.hgvs_short for item in parsed] == ["A23C", "D45E"]
    assert all(item.component is None for item in parsed)


def test_component_prefix_is_not_treated_as_proteingym():
    parsed = parse_mutation_notation("GB1:V39C")
    assert len(parsed) == 1
    assert parsed[0] == MutationEdit(wt="V", position=39, mutant="C", component="GB1")
    with pytest.raises(InvalidMutationNotation):
        parse_mutation_notation("not-a-mutation")


def test_prefixed_and_canonical_and_tokens_share_identities():
    tokens = [
        json.dumps(["GB1_WT", "GB1", 39, "V", "C"], separators=(",", ":")),
        json.dumps(["GB1_WT", "GB1", 40, "D", "Y"], separators=(",", ":")),
    ]
    from_tokens = edits_from_tokens(tokens)
    from_prefix = parse_mutation_notation("GB1:V39C;GB1:D40Y")
    from_canonical = parse_mutation_notation("V39C;D40Y")
    assert _identities(from_tokens) == _identities(from_prefix) == _identities(from_canonical)
    assert format_canonical(from_tokens) == "V39C;D40Y"


def test_fold_tokens_normalize_and_pass_residue_detector():
    tokens = json.dumps(
        [
            json.dumps(["GB1_WT", "GB1", 39, "V", "A"], separators=(",", ":")),
            json.dumps(["GB1_WT", "GB1", 54, "V", "A"], separators=(",", ":")),
        ],
        separators=(",", ":"),
    )
    frame = pd.DataFrame(
        [
            {
                "variant_id": "v-adga",
                "variant": "ADGA",
                "sequence": "ADGA",
                "mutation_count": 2,
                "mutation_tokens": tokens,
            }
        ]
    )
    loaded = variants_from_fold_frame(frame, "oracle_pool")[0]
    assert loaded.mutation_notation == "V39A;V54A"
    conflicts = ResidueConflictDetector().detect(
        [loaded],
        wild_type_sites="VDGV",
        mutable_positions=(39, 40, 41, 54),
    )
    assert not any(
        item.code in {"INVALID_MUTATION_NOTATION", "MUTATION_NOTATION_MISMATCH"}
        for item in conflicts
    )


def test_prefixed_display_string_matches_sequence_for_critic(experiment_config):
    variant = Variant(
        variant_id="prefixed",
        variant="ADGA",
        sequence="ADGA",
        mutation_notation="GB1:V39A;GB1:V54A",
        mutation_count=2,
        split_role="oracle_pool",
    )
    conflicts = ResidueConflictDetector().detect(
        [variant],
        wild_type_sites=experiment_config.task.wild_type_sites,
        mutable_positions=experiment_config.task.mutable_positions,
    )
    assert {item.code for item in conflicts}.isdisjoint(
        {"INVALID_MUTATION_NOTATION", "MUTATION_NOTATION_MISMATCH"}
    )
