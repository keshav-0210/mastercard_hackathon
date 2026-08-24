from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "adaptive" / "adaptive_qwen_ctgan_results_20260824T105805Z.json"
OUTPUT = ROOT / "adaptive" / "adaptive_qwen_ctgan_graphs_v2_20260824.png"


def plot_mean_band(axis, data: pd.DataFrame, columns: list[str], title: str, ylabel: str) -> None:
    grouped = data.groupby("round")[columns].agg(["mean", "std"])
    colors = ["#1677a8", "#d6673d", "#2f8f5b"]
    for column, color in zip(columns, colors):
        mean = grouped[(column, "mean")]
        deviation = grouped[(column, "std")].fillna(0)
        axis.plot(mean.index, mean, marker="o", linewidth=2, label=column, color=color)
        axis.fill_between(mean.index, mean - deviation, mean + deviation, color=color, alpha=0.12)
    axis.set_title(title)
    axis.set_xlabel("Round")
    axis.set_ylabel(ylabel)
    axis.set_ylim(0, 1.05)
    axis.set_xticks(sorted(data["round"].unique()))
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=8)


def main() -> None:
    artifact = json.loads(RESULTS.read_text(encoding="utf-8"))
    data = pd.DataFrame(artifact["round_metrics"])
    data["decision_label"] = np.where(data["family_decision"].map(lambda item: item.get("source") == "seeded_plan"), "Seeded start", "Adaptive recommendation")

    figure, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    figure.patch.set_facecolor("#f7f5ef")
    for axis in axes.flat:
        axis.set_facecolor("#fffdf8")

    plot_mean_band(axes[0, 0], data, ["behavioural_plausibility", "novelty_score", "unique_row_ratio"], "Red-team quality across seeds", "Score")
    plot_mean_band(axes[0, 1], data, ["recall", "f1", "precision"], "Blue-team response across seeds", "Score")

    family_table = data.pivot_table(index="attack_family", columns="round", values="recall", aggfunc="mean")
    image = axes[1, 0].imshow(family_table, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    axes[1, 0].set_title("Family-level recall heatmap")
    axes[1, 0].set_xlabel("Round")
    axes[1, 0].set_ylabel("Attack family")
    axes[1, 0].set_xticks(range(len(family_table.columns)), family_table.columns)
    axes[1, 0].set_yticks(range(len(family_table.index)), family_table.index)
    axes[1, 0].tick_params(axis="y", labelsize=8)
    for row_index in range(len(family_table.index)):
        for column_index in range(len(family_table.columns)):
            value = family_table.iloc[row_index, column_index]
            if not np.isnan(value):
                axes[1, 0].text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black" if value > 0.45 else "white")
    figure.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Mean recall")

    families = sorted(data["attack_family"].unique())
    palette = plt.get_cmap("tab10")
    for family_index, family in enumerate(families):
        subset = data[data["attack_family"] == family]
        axes[1, 1].scatter(subset["behavioural_plausibility"], 1 - subset["recall"], s=70, color=palette(family_index % 10), alpha=0.9, label=family)
        for _, row in subset.iterrows():
            axes[1, 1].annotate(f"r{int(row['round'])}", (row["behavioural_plausibility"], 1 - row["recall"]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    axes[1, 1].set_title("Challenge frontier: plausible and difficult")
    axes[1, 1].set_xlabel("Behavioural plausibility")
    axes[1, 1].set_ylabel("Detector difficulty (1 - recall)")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.22)
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")

    figure.suptitle("AI Defence Lab | Adaptive Qwen + CTGAN analysis", fontsize=16, color="#123c3b", fontweight="bold")
    figure.savefig(OUTPUT, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"Created: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    main()
