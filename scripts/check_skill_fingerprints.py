"""Print generated schema and normalized Skill fingerprints for parity updates."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from fitness_agents.agents.critic import CritiqueDecisionBodyOutput
from fitness_agents.agents.output_contracts import MainSynthesisOutput
from fitness_agents.contracts.hypothesis_pipeline import (
    ChannelHypothesisOutput,
    ConservationReviewBody,
    MainReviewBody,
    PhyschemReviewBody,
    StructureReviewBody,
)

ROOT = Path(__file__).parents[1]
CASES = (
    ("profiles/subscientist/physchem_v1/SKILL.md", ChannelHypothesisOutput),
    ("profiles/subscientist/conservation_v1/SKILL.md", ChannelHypothesisOutput),
    ("profiles/subscientist/structure_v1/SKILL.md", ChannelHypothesisOutput),
    ("profiles/subcritic/physchem_v1/SKILL.md", PhyschemReviewBody),
    ("profiles/subcritic/conservation_v1/SKILL.md", ConservationReviewBody),
    ("profiles/subcritic/structure_v1/SKILL.md", StructureReviewBody),
    ("profiles/critic/hypothesis_v1/SKILL.md", MainReviewBody),
    ("critic_profiles/scientific_v1/SKILL.md", CritiqueDecisionBodyOutput),
    ("profiles/scientist/synthesis_v1/SKILL.md", MainSynthesisOutput),
)


def _schema_sha256(model: type) -> str:
    payload = json.dumps(
        model.model_json_schema(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _skill_sha256(text: str) -> str:
    normalized = re.sub(
        r"(?m)^- skill_sha256: (?:PENDING|[0-9a-f]{64})$",
        "- skill_sha256: <normalized>",
        text,
    )
    return hashlib.sha256(normalized.encode()).hexdigest()


def main() -> None:
    agent_root = ROOT / "src/fitness_agents/agents"
    for relative, model in CASES:
        path = agent_root / relative
        text = path.read_text(encoding="utf-8")
        print(f"{relative}\tschema={_schema_sha256(model)}\tskill={_skill_sha256(text)}")


if __name__ == "__main__":
    main()
