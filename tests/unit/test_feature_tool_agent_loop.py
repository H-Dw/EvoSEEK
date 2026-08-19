from types import SimpleNamespace

from fitness_agents.config import KGInteractionRuntimeConfig
from fitness_agents.contracts.schemas import FitnessObservation, Variant
from fitness_agents.loop.orchestrator import CampaignRunner


class _CapturingController:
    def __init__(self) -> None:
        self.plan = None
        self.context = None

    def execute(self, plan, context):
        self.plan = plan
        self.context = context
        return plan


def test_agent_loop_plans_independent_and_joint_feature_tools_before_explanation():
    runtime = KGInteractionRuntimeConfig(
        feature_tool_strategy="independent_and_joint",
        feature_channels=("physchem", "conservation", "structure"),
        feature_variant_limit=1,
        truncation_audit_enabled=True,
        truncation_audit_items=("physchem", "HAS_PHYSCHEM_DELTA"),
        max_tool_calls=10,
        stop_when_sufficient=False,
    )
    controller = _CapturingController()
    runner = object.__new__(CampaignRunner)
    runner.config = SimpleNamespace(kg_interaction=runtime)
    runner.kg_interaction = controller
    runner.run_id = "run-feature-loop"
    runner.knowledge = SimpleNamespace(local_knowledge=None)
    runner._scientist_local_context_allowed = False
    runner.state = SimpleNamespace(
        observed=[
            FitnessObservation("v-best", 1.0, "observed", 0),
            FitnessObservation("v-other", 0.5, "observed", 0),
        ]
    )
    variants = [
        Variant("v-best", "WA", "WA", "A1W", 1, "observed"),
        Variant("v-other", "AA", "AA", "WT", 0, "observed"),
    ]

    result = runner._run_kg_interaction(round_id=1, observed_variants=variants)

    assert result is controller.plan
    assert [step.operator for step in controller.plan.steps] == [
        "hypothesis_context",
        "query_assay_association",
        "query_physchem_delta",
        "query_evolutionary_profile",
        "query_structure_environment",
        "query_feature_bundle",
        "query_kg_truncation_audit",
        "explain_variant",
        "compare_variants",
    ]
    bundle = controller.plan.steps[5]
    assert bundle.arguments["channels"] == ["physchem", "conservation", "structure"]
    assert bundle.arguments["variant_id"] == "v-best"
    assert controller.plan.steps[6].arguments["items"] == [
        "physchem",
        "HAS_PHYSCHEM_DELTA",
    ]
    assert controller.context.allowed_variant_ids == frozenset({"v-best", "v-other"})
