from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = (
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "false_positive_rate",
    "behavioural_plausibility",
    "novelty_score",
    "channel_entropy",
    "unique_row_ratio",
)

FAMILY_DIRECTORIES = (
    "account_takeover",
    "trusted_device",
    "beneficiary_manipulation",
    "low_and_slow",
    "social_engineering",
    "merchant_abuse",
    "cross_channel_anomaly",
    "trusted_device_normal_velocity",
    "low_and_slow_common_channel",
    "beneficiary_manipulation_moderate_amount",
    "trusted_device_low_and_slow",
    "social_engineering_beneficiary_manipulation",
)


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("round_metrics"):
        return data["round_metrics"]
    rows = []
    for run in data.get("by_seed", []):
        for result in run.get("results", []):
            rows.append({
                "seed": run.get("seed"),
                "round": result.get("round"),
                "attack_family": result.get("specification", {}).get("attack_family"),
                "family_decision": result.get("family_decision", {}),
                "all_family_metrics": result.get("detection", {}).get("all_family_metrics", {}),
                **{metric: result.get("detection", {}).get(metric, result.get("fidelity", {}).get(metric, result.get("novelty", {}).get(metric))) for metric in METRICS},
            })
    return rows


def build_charts(results_path: Path, output_dir: Path) -> None:
    rows = load_rows(results_path)
    family_values: dict[str, list[dict]] = {}
    for row in rows:
        all_metrics = row.get("all_family_metrics") or row.get("by_attack_family", {})
        if all_metrics:
            for family, values in all_metrics.items():
                family_values.setdefault(family, []).append({
                    "round": row["round"],
                    "seed": row.get("seed"),
                    **values,
                    "chosen": family == row.get("attack_family"),
                })
        elif row.get("attack_family"):
            family_values.setdefault(row["attack_family"], []).append({
                "round": row["round"],
                "seed": row.get("seed"),
                "chosen": True,
                **row,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    for family in FAMILY_DIRECTORIES:
        (output_dir / family).mkdir(parents=True, exist_ok=True)
    for family, family_rows in sorted(family_values.items()):
        family_dir = output_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        for metric in METRICS:
            points = [row for row in family_rows if row.get(metric) is not None]
            if not points:
                continue
            figure, axis = plt.subplots(figsize=(8, 4.5), dpi=160)
            for chosen, color, label in ((True, "#c0392b", "chosen by Agent 1"), (False, "#d6a21e", "not chosen")):
                selected = [row for row in points if row["chosen"] == chosen]
                if selected:
                    axis.scatter(
                        [row["round"] for row in selected],
                        [float(row[metric]) for row in selected],
                        color=color,
                        edgecolors="white",
                        linewidths=0.7,
                        s=48,
                        label=label,
                        zorder=3,
                    )
            axis.plot([row["round"] for row in sorted(points, key=lambda item: item["round"])], [float(row[metric]) for row in sorted(points, key=lambda item: item["round"])], color="#7f8c8d", alpha=0.35, linewidth=1)
            axis.set_title(f"{family}: {metric.replace('_', ' ').title()}")
            axis.set_xlabel("Round")
            axis.set_ylabel(metric.replace("_", " ").title())
            axis.grid(alpha=0.2)
            axis.legend(loc="best")
            figure.tight_layout()
            figure.savefig(family_dir / f"{metric}.png")
            plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create one metric chart per adaptive fraud family.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("charts"))
    arguments = parser.parse_args()
    build_charts(arguments.results, arguments.output)
    print(f"Charts written to {arguments.output}")
