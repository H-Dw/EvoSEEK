"""Nature-style Python figures generated only from exported source-data CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import (
    CONDITION_ORDER,
    FIGURE_DIR,
    PLOT_COLORS,
    PLOT_LABELS,
    PLOT_LINESTYLES,
    PLOT_MARKERS,
)


def _set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.7)
    axis.grid(axis="y", color="#D7D7D7", linewidth=0.45, alpha=0.7)
    axis.set_axisbelow(True)


def _save_all(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        path = stem.with_suffix(suffix)
        if suffix == ".tiff":
            rendered_path = path.with_name(f".{path.stem}.rendering.tiff")
            rgb_path = path.with_name(f".{path.stem}.rgb-writing.tiff")
            fig.savefig(
                rendered_path,
                bbox_inches="tight",
                facecolor="white",
                **kwargs,
            )
            with Image.open(rendered_path) as image:
                rgb_image = image.convert("RGB").copy()
            rgb_image.save(
                rgb_path,
                dpi=(600, 600),
                compression="tiff_lzw",
            )
            rgb_path.replace(path)
            rendered_path.unlink(missing_ok=True)
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    return paths


def _plot_trajectory_panel(
    axis: plt.Axes,
    fold_data: pd.DataFrame,
    summary_data: pd.DataFrame,
    metric: str,
    ylabel: str,
    include_round_zero: bool,
) -> None:
    folds = fold_data.copy()
    if not include_round_zero:
        folds = folds[folds["round_id"] > 0]
    for condition in CONDITION_ORDER:
        raw = folds[folds["condition"] == condition]
        for _, group in raw.groupby("fold"):
            axis.plot(
                group["query_budget"],
                group[metric],
                color=PLOT_COLORS[condition],
                alpha=0.18,
                linewidth=0.7,
                linestyle=PLOT_LINESTYLES[condition],
            )
        summary = summary_data[
            (summary_data["condition"] == condition)
            & (summary_data["metric"] == metric)
        ].sort_values("round_id")
        if not include_round_zero:
            summary = summary[summary["round_id"] > 0]
        x = summary["query_budget"].to_numpy(float)
        mean = summary["mean"].to_numpy(float)
        sd = summary["sd"].to_numpy(float)
        axis.fill_between(
            x,
            mean - sd,
            mean + sd,
            color=PLOT_COLORS[condition],
            alpha=0.10,
            linewidth=0,
        )
        axis.plot(
            x,
            mean,
            marker=PLOT_MARKERS[condition],
            color=PLOT_COLORS[condition],
            linestyle=PLOT_LINESTYLES[condition],
            label=PLOT_LABELS[condition],
        )
    axis.set_xlabel("Cumulative wet-lab queries")
    axis.set_ylabel(ylabel)
    axis.set_xticks([0, 16, 32, 48] if include_round_zero else [16, 32, 48])
    _clean_axis(axis)


def plot_fitness_trajectories(
    round_csv: Path, round_summary_csv: Path, output_dir: Path = FIGURE_DIR
) -> list[Path]:
    _set_style()
    fold_data = pd.read_csv(round_csv)
    summary = pd.read_csv(round_summary_csv)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.90))
    specs = (
        ("best_seen_fitness", "Best-seen fitness", True),
        ("batch_mean_fitness", "Selected-batch mean fitness", False),
        ("batch_median_fitness", "Selected-batch median fitness", False),
    )
    for label, axis, (metric, ylabel, include_zero) in zip("abc", axes, specs):
        _plot_trajectory_panel(
            axis, fold_data, summary, metric, ylabel, include_zero
        )
        axis.text(
            -0.18,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.52, 1.02),
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.18, top=0.72, wspace=0.34)
    paths = _save_all(fig, output_dir / "figure2_fitness_trajectories")
    plt.close(fig)
    return paths


def plot_module_deltas(delta_csv: Path, output_dir: Path = FIGURE_DIR) -> list[Path]:
    _set_style()
    data = pd.read_csv(delta_csv)
    metric_specs = (
        ("final_best_seen", "Final best-seen Δ"),
        ("best_seen_aulc", "Best-seen AULC Δ"),
        ("r3_batch_mean", "Round-3 batch mean Δ"),
        ("r3_batch_median", "Round-3 batch median Δ"),
    )
    comparisons = (
        "kg_memory",
        "three_channels_without_rag",
        "rag_without_three_channels",
        "three_channels_with_rag",
        "rag_with_three_channels",
        "active_learning",
    )
    comparison_labels = {
        "kg_memory": "KG base − Agent only",
        "three_channels_without_rag": "3 channels − KG base",
        "rag_without_three_channels": "KG + RAG − KG base",
        "three_channels_with_rag": "3 channels + RAG − KG + RAG",
        "rag_with_three_channels": "3 channels + RAG − 3 channels",
        "active_learning": "KG + active learning − KG base",
    }
    comparison_colors = {
        "kg_memory": PLOT_COLORS["kg_base"],
        "three_channels_without_rag": PLOT_COLORS["kg_3features_base"],
        "rag_without_three_channels": PLOT_COLORS["kg_base_rag"],
        "three_channels_with_rag": PLOT_COLORS["kg_3features_rag"],
        "rag_with_three_channels": "#76508A",
        "active_learning": PLOT_COLORS["kg_base_al"],
    }
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 3.05))
    for label, axis, (metric, xlabel) in zip("abcd", axes, metric_specs):
        subset = data[data["metric"] == metric]
        for y, comparison_id in enumerate(comparisons):
            values = subset[subset["comparison_id"] == comparison_id][
                "delta"
            ].to_numpy(float)
            jitter = np.linspace(-0.08, 0.08, len(values))
            axis.scatter(
                values,
                y + jitter,
                s=18,
                facecolor="white",
                edgecolor=comparison_colors[comparison_id],
                linewidth=0.9,
                zorder=3,
            )
            axis.scatter(
                [float(values.mean())],
                [y],
                marker="D",
                s=28,
                color=comparison_colors[comparison_id],
                zorder=4,
            )
            axis.plot(
                [float(values.min()), float(values.max())],
                [y, y],
                color=comparison_colors[comparison_id],
                linewidth=1.0,
                alpha=0.75,
            )
        axis.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
        axis.set_yticks(range(len(comparisons)))
        axis.set_yticklabels(
            [comparison_labels[c] for c in comparisons]
            if axis is axes[0]
            else [""] * len(comparisons)
        )
        axis.set_xlabel(xlabel)
        axis.text(
            -0.18,
            1.04,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )
        _clean_axis(axis)
        axis.grid(axis="x", color="#D7D7D7", linewidth=0.45, alpha=0.7)
        axis.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.20, top=0.91, wspace=0.46)
    paths = _save_all(fig, output_dir / "figure3_module_deltas")
    plt.close(fig)
    return paths


def generate_all_figures(source_data_dir: Path, output_dir: Path = FIGURE_DIR) -> list[Path]:
    paths = []
    paths.extend(
        plot_fitness_trajectories(
            source_data_dir / "round_metrics_by_fold.csv",
            source_data_dir / "round_metrics_mean_sd.csv",
            output_dir,
        )
    )
    paths.extend(
        plot_module_deltas(
            source_data_dir / "kg_module_fold_deltas.csv", output_dir
        )
    )
    return paths
