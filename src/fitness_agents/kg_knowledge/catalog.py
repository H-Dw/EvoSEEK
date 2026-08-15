from __future__ import annotations

from dataclasses import dataclass

from .schema import KnowledgeLayer


@dataclass(frozen=True)
class EntitySpec:
    entity_type: str
    layer: KnowledgeLayer
    priority: str
    contribution: int
    implementation_difficulty: int
    ablation_group: str
    purpose: str


@dataclass(frozen=True)
class RelationSpec:
    predicate: str
    subject_type: str
    object_type: str
    layer: KnowledgeLayer
    priority: str


DEFAULT_ENTITY_SPECS = (
    EntitySpec("Protein", KnowledgeLayer.IDENTITY, "P0", 5, 1, "identity", "实验靶蛋白"),
    EntitySpec("Sequence", KnowledgeLayer.SEQUENCE, "P0", 5, 1, "sequence", "版本化氨基酸序列"),
    EntitySpec(
        "ResiduePosition", KnowledgeLayer.SEQUENCE, "P0", 5, 2, "sequence", "统一残基坐标锚点"
    ),
    EntitySpec("Variant", KnowledgeLayer.SEQUENCE, "P0", 5, 1, "sequence", "待评估突变体"),
    EntitySpec("Mutation", KnowledgeLayer.SEQUENCE, "P0", 5, 1, "sequence", "原子化替换事件"),
    EntitySpec("Assay", KnowledgeLayer.EXPERIMENTAL, "P0", 5, 1, "experiment", "测量定义与标度"),
    EntitySpec(
        "Condition", KnowledgeLayer.EXPERIMENTAL, "P0", 5, 2, "experiment", "温度、pH等上下文"
    ),
    EntitySpec(
        "Observation", KnowledgeLayer.EXPERIMENTAL, "P0", 5, 1, "experiment", "真实测量及不确定性"
    ),
    EntitySpec(
        "CampaignRound", KnowledgeLayer.PROVENANCE, "P0", 5, 1, "history", "轮次与可见性边界"
    ),
    EntitySpec("Prediction", KnowledgeLayer.MODEL, "P1", 5, 1, "model", "模型输出与校准不确定性"),
    EntitySpec("ModelRun", KnowledgeLayer.MODEL, "P1", 5, 2, "model", "模型和特征版本"),
    EntitySpec("Evidence", KnowledgeLayer.AGENT, "P1", 5, 1, "reasoning", "可追溯证据单元"),
    EntitySpec("Hypothesis", KnowledgeLayer.AGENT, "P1", 5, 2, "reasoning", "可证伪设计假设"),
    EntitySpec("Decision", KnowledgeLayer.AGENT, "P1", 5, 2, "reasoning", "候选选择与理由"),
    EntitySpec(
        "Structure", KnowledgeLayer.STRUCTURE, "P1", 5, 2, "structure", "实验或预测结构版本"
    ),
    EntitySpec(
        "ResidueEnvironment", KnowledgeLayer.STRUCTURE, "P1", 5, 3, "structure", "局部几何和暴露度"
    ),
    EntitySpec(
        "AtomicInteraction",
        KnowledgeLayer.ATOM_CHEMISTRY,
        "P1",
        4,
        3,
        "atom_chemistry",
        "氢键、盐桥、接触等",
    ),
    EntitySpec(
        "EvolutionProfile", KnowledgeLayer.EVOLUTIONARY, "P1", 5, 2, "evolution", "保守性与共进化"
    ),
    EntitySpec("Domain", KnowledgeLayer.FUNCTIONAL, "P2", 4, 2, "function", "功能结构域"),
    EntitySpec("OntologyTerm", KnowledgeLayer.FUNCTIONAL, "P2", 4, 2, "function", "GO等标准概念"),
    EntitySpec("Homolog", KnowledgeLayer.EVOLUTIONARY, "P2", 3, 3, "evolution", "同源蛋白证据"),
    EntitySpec(
        "BindingPartner", KnowledgeLayer.FUNCTIONAL, "P2", 3, 3, "function", "互作或配体对象"
    ),
    EntitySpec("Publication", KnowledgeLayer.LITERATURE, "P2", 3, 2, "literature", "来源文献"),
    EntitySpec(
        "Claim", KnowledgeLayer.LITERATURE, "P2", 4, 3, "literature", "带限定条件的文献主张"
    ),
    EntitySpec(
        "Artifact", KnowledgeLayer.PROVENANCE, "P2", 4, 2, "artifacts", "大对象、坐标或向量引用"
    ),
)


DEFAULT_RELATION_SPECS = (
    RelationSpec("VARIANT_OF", "Variant", "Protein", KnowledgeLayer.IDENTITY, "P0"),
    RelationSpec("HAS_SEQUENCE", "Variant", "Sequence", KnowledgeLayer.SEQUENCE, "P0"),
    RelationSpec("HAS_MUTATION", "Variant", "Mutation", KnowledgeLayer.SEQUENCE, "P0"),
    RelationSpec("AT_POSITION", "Mutation", "ResiduePosition", KnowledgeLayer.SEQUENCE, "P0"),
    RelationSpec("OBSERVES_VARIANT", "Observation", "Variant", KnowledgeLayer.EXPERIMENTAL, "P0"),
    RelationSpec("MEASURED_IN", "Observation", "Assay", KnowledgeLayer.EXPERIMENTAL, "P0"),
    RelationSpec("UNDER_CONDITION", "Observation", "Condition", KnowledgeLayer.EXPERIMENTAL, "P0"),
    RelationSpec("REVEALED_IN", "Observation", "CampaignRound", KnowledgeLayer.PROVENANCE, "P0"),
    RelationSpec("PREDICTS", "Prediction", "Variant", KnowledgeLayer.MODEL, "P1"),
    RelationSpec("GENERATED_BY", "Prediction", "ModelRun", KnowledgeLayer.MODEL, "P1"),
    RelationSpec("ABOUT", "Evidence", "Variant", KnowledgeLayer.AGENT, "P1"),
    RelationSpec("SUPPORTED_BY", "Hypothesis", "Evidence", KnowledgeLayer.AGENT, "P1"),
    RelationSpec("CONTRADICTED_BY", "Hypothesis", "Evidence", KnowledgeLayer.AGENT, "P1"),
    RelationSpec(
        "MAPPED_TO_STRUCTURE",
        "ResiduePosition",
        "ResidueEnvironment",
        KnowledgeLayer.STRUCTURE,
        "P1",
    ),
    RelationSpec(
        "HAS_EVOLUTION_PROFILE",
        "ResiduePosition",
        "EvolutionProfile",
        KnowledgeLayer.EVOLUTIONARY,
        "P1",
    ),
    RelationSpec("DERIVED_FROM", "Evidence", "Artifact", KnowledgeLayer.PROVENANCE, "P2"),
    RelationSpec("ANNOTATED_WITH", "Protein", "OntologyTerm", KnowledgeLayer.FUNCTIONAL, "P2"),
)


def entity_specs_by_priority(priority: str) -> tuple[EntitySpec, ...]:
    return tuple(item for item in DEFAULT_ENTITY_SPECS if item.priority == priority)
