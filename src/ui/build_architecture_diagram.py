r"""Generates artifacts/architecture_diagram.png from the current codebase's closed-loop design.

Re-run this script any time the pipeline architecture changes:
    .\.venv\Scripts\python.exe src\ui\build_architecture_diagram.py

The submission dashboard references this PNG by relative filename, so simply replacing the file
(keeping the same name) and refreshing the browser shows the revised diagram with no rebuild step.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts" / "architecture_diagram.png"

BLUE = "#1B3A6B"
RED = "#EB001B"
ORANGE = "#F79E1B"
GREY = "#4A4A4A"


def box(ax, xy, width, height, text, color, text_color="white", fontsize=9.3):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.3, edgecolor=color, facecolor=color, alpha=0.94,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color, weight="bold")
    return {
        "top": (x + width / 2, y + height),
        "bottom": (x + width / 2, y),
        "left": (x, y + height / 2),
        "right": (x + width, y + height / 2),
    }


def arrow(ax, start, end, color=GREY, connection="arc3,rad=0.0"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=1.6, connectionstyle=connection))


def build() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 8.2), dpi=200)
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 8.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(6.7, 7.85, "Adaptive Red-Team / Blue-Team Fraud Defence Architecture", ha="center", fontsize=15.5, weight="bold", color=BLUE)
    ax.text(6.7, 7.48, "Closed synthetic attack-generation loop with continual detector hardening", ha="center", fontsize=10, color=GREY)

    memory = box(ax, (0.4, 5.7), 2.7, 0.95, "Attack Memory\n(SQLite, append-only)\n+ Reviewed Public RAG", BLUE)
    a1 = box(ax, (3.7, 5.7), 2.7, 0.95, "Agent 1\nResearch + Family\nRecommendation (Qwen2.5-7B)", RED)
    ctrl = box(ax, (7.0, 5.7), 2.7, 0.95, "Controller\n7 Approved Families Only\nWeakness-Weighted Sampling", GREY)
    a2 = box(ax, (10.3, 5.7), 2.7, 0.95, "Agent 2\nAttack Specification\nStrategist (Qwen2.5-7B)", RED)

    gen = box(ax, (10.3, 4.05), 2.7, 0.95, "Synthetic Generator\nProcedural / CTGAN", ORANGE, text_color="black")
    split = box(ax, (7.0, 4.05), 2.7, 0.95, "Train Attacks | Unseen Attacks\n(disjoint synthetic rows)", ORANGE, text_color="black")
    det = box(ax, (3.7, 4.05), 2.7, 0.95, "Fraud Detector\nHistGradientBoosting\nContinual + Replay + Versioning", BLUE, fontsize=8.6)
    ev = box(ax, (0.4, 4.05), 2.7, 0.95, "Evaluation\nPrecision/Recall/F1/ROC-AUC/FPR\n+ Historical Robustness Pool", BLUE, fontsize=8.3)

    a3 = box(ax, (0.4, 2.4), 2.7, 0.95, "Agent 3\nSecurity Analyst\nWeakness Report (Qwen2.5-7B)", RED)
    metrics = box(ax, (3.7, 2.4), 2.7, 0.95, "Metrics Dump\nDiversity, Fidelity, Novelty,\nCoverage, Blue-Team Scores", GREY)
    dash = box(ax, (7.0, 2.4), 2.7, 0.95, "Local JSON Artifacts\n(no pipeline rerun needed)", GREY)
    ui = box(ax, (10.3, 2.4), 2.7, 0.95, "Submission Dashboard\n(this one-page UI)", BLUE)

    arrow(ax, memory["right"], a1["left"])
    arrow(ax, a1["right"], ctrl["left"])
    arrow(ax, ctrl["right"], a2["left"])
    arrow(ax, a2["bottom"], gen["top"])
    arrow(ax, gen["left"], split["right"])
    arrow(ax, split["left"], det["right"])
    arrow(ax, det["left"], ev["right"])
    arrow(ax, ev["bottom"], a3["top"])
    arrow(ax, a3["right"], metrics["left"])
    arrow(ax, metrics["right"], dash["left"])
    arrow(ax, dash["right"], ui["left"])

    arrow(ax, a3["bottom"], (1.75, 1.15))
    arrow(ax, (1.75, 1.15), memory["bottom"], color=RED, connection="arc3,rad=-0.35")
    ax.text(1.75, 0.85, "Feedback loop: weakness -> Attack Memory -> next round's Agent 1", ha="center", fontsize=8.6, color=RED, style="italic")

    legend_y = 1.55
    legend_items = [
        ("Agentic reasoning (Qwen2.5-7B)", RED),
        ("Detector / defence state", BLUE),
        ("Synthetic data generation", ORANGE),
        ("Control / reporting", GREY),
    ]
    for index, (label, color) in enumerate(legend_items):
        ax.add_patch(Rectangle((0.4 + index * 3.25, legend_y), 0.28, 0.28, color=color))
        ax.text(0.78 + index * 3.25, legend_y + 0.14, label, va="center", fontsize=8.4, color=GREY)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, facecolor="white")
    plt.close(fig)
    print(f"Architecture diagram saved to {OUTPUT}")


if __name__ == "__main__":
    build()
