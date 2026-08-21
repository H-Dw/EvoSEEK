"""Publication-ready diagnostic figure for the best-vs-batch anomaly."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import ANOMALY_FIGURE_DIR, PLOT_COLORS, PLOT_LABELS, PLOT_MARKERS


KG_CONDITIONS = ("kg_base", "kg_base_rag", "kg_base_al")


def _set_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.6,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.35,
            "lines.markersize": 4.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.transparent": False,
        }
    )


def _clean_axis(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=2.5, width=0.7)
    axis.grid(axis=grid_axis, color="#D7D7D7", linewidth=0.45, alpha=0.7)
    axis.set_axisbelow(True)


def _save_all(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, kwargs in (
        (".svg", {}),
        (".pdf", {}),
        (".png", {"dpi": 300}),
    ):
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        paths.append(path)
    tiff_path = stem.with_name(f"{stem.name}_600dpi").with_suffix(".tiff")
    tiff_buffer = BytesIO()
    fig.savefig(
        tiff_buffer,
        format="tiff",
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )
    tiff_buffer.seek(0)
    with Image.open(tiff_buffer) as image:
        rgb_image = image.convert("RGB").copy()
    rgb_image.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
    paths.append(tiff_path)
    return paths


def plot_anomaly_diagnostics(source_dir: Path, output_dir: Path = ANOMALY_FIGURE_DIR) -> list[Path]:
    _set_style()
    pooled = pd.read_csv(source_dir / "batch_distribution_pooled.csv")
    candidates = pd.read_csv(source_dir / "selected_candidate_diagnostics.csv")
    score = pd.read_csv(source_dir / "score_alignment_mean_sd.csv")

    fig = plt.figure(figsize=(7.2, 5.1))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], hspace=0.42, wspace=0.34)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    # a: sparse record events, the direct reason cumulative best can rise.
    axis = axes[0]
    x = np.arange(1, 4)
    width = 0.22
    for index, condition in enumerate(KG_CONDITIONS):
        subset = pooled[pooled["condition"] == condition].set_index("round_id")
        values = [subset.loc[round_id, "new_record_folds"] for round_id in x]
        axis.bar(
            x + (index - 1) * width,
            values,
            width=width,
            color=PLOT_COLORS[condition],
            label=PLOT_LABELS[condition],
        )
    axis.set_xticks(x)
    axis.set_xticklabels(["Round 1", "Round 2", "Round 3"])
    axis.set_ylabel("Folds setting a new record (n=3)")
    axis.set_ylim(0, 3.35)
    axis.set_yticks([0, 1, 2, 3])
    axis.set_title("New best-seen events are sparse after round 1", loc="left")
    _clean_axis(axis)

    # b: full wet-fitness distributions reveal a falling body and rare upper tail.
    axis = axes[1]
    positions = []
    data = []
    colors = []
    labels = []
    base_positions = {"kg_base": 1, "kg_base_rag": 5, "kg_base_al": 9}
    for condition in KG_CONDITIONS:
        for round_id in (1, 2, 3):
            positions.append(base_positions[condition] + round_id - 1)
            values = candidates[
                (candidates["condition"] == condition)
                & (candidates["round_id"] == round_id)
            ]["wet_fitness"].to_numpy(float)
            data.append(values)
            colors.append(PLOT_COLORS[condition])
            labels.append(f"R{round_id}")
    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 0.9},
        whiskerprops={"linewidth": 0.7},
        capprops={"linewidth": 0.7},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
    for position, values, color in zip(positions, data, colors):
        jitter = np.linspace(-0.22, 0.22, len(values))
        axis.scatter(
            position + jitter,
            values,
            s=6,
            color=color,
            alpha=0.38,
            linewidths=0,
            rasterized=True,
        )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Selected wet fitness")
    axis.set_title("Batch bodies shift downward despite rare high outliers", loc="left")
    group_centers = [2, 6, 10]
    for center, condition in zip(group_centers, KG_CONDITIONS):
        axis.text(
            center,
            -0.20,
            PLOT_LABELS[condition],
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color=PLOT_COLORS[condition],
        )
    _clean_axis(axis)

    # c: mass transfers from high-fitness to near-zero candidates.
    axis = axes[2]
    for condition in KG_CONDITIONS:
        subset = pooled[pooled["condition"] == condition].sort_values("round_id")
        axis.plot(
            subset["round_id"],
            100 * subset["fraction_le_0_05"],
            color=PLOT_COLORS[condition],
            marker=PLOT_MARKERS[condition],
            linestyle="-",
            label=f"{PLOT_LABELS[condition]}: ≤0.05",
        )
        axis.plot(
            subset["round_id"],
            100 * subset["fraction_ge_2"],
            color=PLOT_COLORS[condition],
            marker=PLOT_MARKERS[condition],
            linestyle="--",
            alpha=0.72,
            label=f"{PLOT_LABELS[condition]}: ≥2",
        )
    axis.set_xticks([1, 2, 3])
    axis.set_xlabel("Round")
    axis.set_ylabel("Candidates in threshold class (%)")
    axis.set_title("Low-fitness mass rises as high-fitness mass falls", loc="left")
    _clean_axis(axis)

    # d: the selected set is predicted to be weaker in later rounds too.
    axis = axes[3]
    for condition in KG_CONDITIONS:
        subset = score[score["condition"] == condition].sort_values("round_id")
        x_values = subset["selected_predicted_mean_mean"].to_numpy(float)
        y_values = subset["wet_mean_mean"].to_numpy(float)
        axis.plot(
            x_values,
            y_values,
            color=PLOT_COLORS[condition],
            marker=PLOT_MARKERS[condition],
            label=PLOT_LABELS[condition],
        )
        for round_id, x_value, y_value in zip(
            subset["round_id"], x_values, y_values
        ):
            if int(round_id) in {1, 3}:
                axis.text(
                    x_value + 0.025,
                    y_value + 0.025,
                    f"R{int(round_id)}",
                    fontsize=6,
                )
    limits = [0.9, 3.55]
    axis.plot(limits, limits, color="#777777", linestyle=":", linewidth=0.8)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Mean dry-predicted fitness of selected batch")
    axis.set_ylabel("Mean revealed wet fitness")
    axis.set_title("Later selected batches are also predicted to be weaker", loc="left")
    _clean_axis(axis)

    for label, axis in zip("abcd", axes):
        axis.text(
            -0.16,
            1.05,
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
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.51, 1.005),
    )
    handles_c, labels_c = axes[2].get_legend_handles_labels()
    axes[2].legend(
        handles_c,
        labels_c,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.90)
    paths = _save_all(fig, output_dir / "figure4_best_vs_batch_anomaly")
    plt.close(fig)
    return paths
