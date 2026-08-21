"""Central configuration for the GB1 AL96 report analysis."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]

BASELINE_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "random-fitness-direct-s42-al96-collected-20260820T102640Z"
)
HIERARCHICAL_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "hierarchical-scientist-kg_base_kg_base_rag_kg_base_al"
)
THREE_FEATURE_ARTIFACT_ROOT = (
    REPO_ROOT / "artifacts" / "hierarchical-scientist-kg_3features_rag"
)
ADDITIONAL_ABLATION_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "hierarchical-scientist-kg_3features_base_agent_only"
)
ARTIFACT_ROOTS = (
    BASELINE_ARTIFACT_ROOT,
    HIERARCHICAL_ARTIFACT_ROOT,
    THREE_FEATURE_ARTIFACT_ROOT,
    ADDITIONAL_ABLATION_ARTIFACT_ROOT,
)

OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
SOURCE_DATA_DIR = OUTPUT_ROOT / "source_data"
CASE_STUDY_DIR = OUTPUT_ROOT / "case_studies"
FIGURE_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"
ANOMALY_DIR = OUTPUT_ROOT / "anomaly_diagnostics"
ANOMALY_SOURCE_DATA_DIR = ANOMALY_DIR / "source_data"
ANOMALY_FIGURE_DIR = ANOMALY_DIR / "figures"
RAG_DIAGNOSTIC_DIR = OUTPUT_ROOT / "rag_effect_diagnostics"
RAG_DIAGNOSTIC_SOURCE_DIR = RAG_DIAGNOSTIC_DIR / "source_data"
RAG_DIAGNOSTIC_CASE_DIR = RAG_DIAGNOSTIC_DIR / "evidence_cases"
MUTATION_BEHAVIOR_DIR = OUTPUT_ROOT / "mutation_behavior_diagnostics"
MUTATION_BEHAVIOR_SOURCE_DIR = MUTATION_BEHAVIOR_DIR / "source_data"

CONDITION_ORDER = (
    "random",
    "fitness_direct",
    "agent_only",
    "kg_base",
    "kg_3features_base",
    "kg_base_rag",
    "kg_3features_rag",
    "kg_base_al",
)
SUPERSEDED_FAILED_CONDITION = "kg_3features_rag"
CONDITION_LABELS = {
    "random": "random",
    "fitness_direct": "fitness_direct",
    "agent_only": "agent_only",
    "kg_base": "kg_base",
    "kg_3features_base": "kg_3features_base",
    "kg_base_rag": "kg_base_rag",
    "kg_base_al": "kg_base_al",
    "kg_3features_rag": "kg_3features_rag",
}
PLOT_LABELS = {
    "random": "Random",
    "fitness_direct": "Kermut direct",
    "agent_only": "Agent only",
    "kg_base": "KG base",
    "kg_3features_base": "KG + 3 channels",
    "kg_base_rag": "KG + RAG",
    "kg_base_al": "KG + active learning",
    "kg_3features_rag": "KG + 3 channels + RAG",
}
PLOT_COLORS = {
    "random": "#8F8F8F",
    "fitness_direct": "#484878",
    "agent_only": "#C47B36",
    "kg_base": "#42949E",
    "kg_3features_base": "#68A77A",
    "kg_base_rag": "#9A4D8E",
    "kg_base_al": "#B64342",
    "kg_3features_rag": "#287C48",
}
PLOT_MARKERS = {
    "random": "o",
    "fitness_direct": "s",
    "agent_only": "v",
    "kg_base": "^",
    "kg_3features_base": "X",
    "kg_base_rag": "D",
    "kg_base_al": "P",
    "kg_3features_rag": "h",
}

PLOT_LINESTYLES = {
    "random": "--",
    "fitness_direct": "--",
    "agent_only": ":",
    "kg_base": "-",
    "kg_3features_base": "-",
    "kg_base_rag": "-",
    "kg_3features_rag": "-",
    "kg_base_al": "-",
}

KG_CONDITIONS = (
    "kg_base",
    "kg_3features_base",
    "kg_base_rag",
    "kg_base_al",
    "kg_3features_rag",
)
AGENT_CONDITIONS = ("agent_only", *KG_CONDITIONS)
THREE_FEATURE_CONDITIONS = ("kg_3features_base", "kg_3features_rag")

EXPECTED_FOLDS = (0, 1, 2)
EXPECTED_ROUNDS = (1, 2, 3)
EXPECTED_BATCH_SIZE = 16
EXPECTED_BATCH_SIZES = (16, 16, 16)
INITIAL_OBSERVATIONS = 96
TOTAL_QUERY_BUDGET = 48
FINAL_VISIBLE_OBSERVATIONS = 144
EVALUATION_TOP_K = 10

PRIMARY_METRICS = {
    "final_best_seen": "higher",
    "delta_best_seen": "higher",
    "best_seen_aulc": "higher",
    "r3_batch_best": "higher",
    "r3_batch_mean": "higher",
    "r3_batch_median": "higher",
}
PREDICTION_METRICS = {
    "spearman": "higher",
    "pearson": "higher",
    "mse": "lower",
    "rmse": "lower",
    "ndcg": "higher",
    "top_k_hit": "higher",
    "top_k_recall": "higher",
    "regret_at_k": "lower",
    "interval_90_coverage_error": "lower",
    "gaussian_nll": "lower",
}
METRIC_DIRECTIONS = {**PRIMARY_METRICS, **PREDICTION_METRICS}

DELTA_COMPARISONS = (
    ("kg_memory", "kg_base", "agent_only"),
    ("three_channels_without_rag", "kg_3features_base", "kg_base"),
    ("rag_without_three_channels", "kg_base_rag", "kg_base"),
    ("three_channels_with_rag", "kg_3features_rag", "kg_base_rag"),
    ("rag_with_three_channels", "kg_3features_rag", "kg_3features_base"),
    ("active_learning", "kg_base_al", "kg_base"),
)
DELTA_METRICS = (
    "final_best_seen",
    "r3_batch_mean",
    "r3_batch_median",
    "best_seen_aulc",
)
