from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .agents import ATTACK_FAMILIES
from .synthetic import build_metrics_dump


BLUE_TEAM_DIRECTIONS = {
    "blue_team_benchmark_f1": 1,
    "blue_team_benchmark_recall": 1,
    "blue_team_benchmark_precision": 1,
    "blue_team_benchmark_roc_auc": 1,
    "blue_team_benchmark_false_positive_rate": -1,
}


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _values(rows: list[dict], metric: str) -> np.ndarray:
    missing_rounds = [row.get("round") for row in rows if row.get(metric) is None]
    if missing_rounds:
        raise ValueError(f"Metric {metric!r} is missing for rounds: {missing_rounds}")
    return np.asarray([float(row[metric]) for row in rows], dtype=float)


def _directional_evidence(
    rows: list[dict],
    metric: str,
    direction: int,
    window_size: int,
) -> dict:
    values = _values(rows, metric)
    rounds = np.asarray([float(row["round"]) for row in rows], dtype=float)
    first_mean = float(values[:window_size].mean())
    last_mean = float(values[-window_size:].mean())
    slope = float(np.polyfit(rounds, values, 1)[0])
    expected_change = direction * (last_mean - first_mean)
    return {
        "expectation": "increase" if direction > 0 else "decrease",
        "first_window_mean": round(first_mean, 6),
        "last_window_mean": round(last_mean, 6),
        "absolute_change": round(last_mean - first_mean, 6),
        "linear_slope": round(slope, 6),
        "passed": bool(expected_change > 0 and direction * slope > 0),
    }


def assess_submission_trends(
    rows: list[dict],
    *,
    window_size: int,
    include_red_team: bool,
    novelty_floor: float = 0.7,
) -> dict:
    if window_size < 1 or len(rows) < window_size * 2:
        raise ValueError("Trend assessment needs at least two complete comparison windows")

    evidence = {
        metric: _directional_evidence(rows, metric, direction, window_size)
        for metric, direction in BLUE_TEAM_DIRECTIONS.items()
    }

    if include_red_team:
        evidence["attack_fidelity_behavioural_plausibility"] = _directional_evidence(
            rows,
            "attack_fidelity_behavioural_plausibility",
            1,
            window_size,
        )

        coverage = _values(rows, "family_coverage_diversity_ratio")
        evidence["family_coverage_diversity_ratio"] = {
            "expectation": "non-decreasing and complete",
            "first_value": round(float(coverage[0]), 6),
            "last_value": round(float(coverage[-1]), 6),
            "passed": bool(np.all(np.diff(coverage) >= 0) and np.isclose(coverage[-1], 1.0)),
        }

        novelty = _values(rows, "attack_novelty_score")
        final_novelty_mean = float(novelty[-window_size:].mean())
        evidence["attack_novelty_score"] = {
            "expectation": f"final-window mean at least {novelty_floor}",
            "first_window_mean": round(float(novelty[:window_size].mean()), 6),
            "last_window_mean": round(final_novelty_mean, 6),
            "linear_slope": round(
                float(np.polyfit(np.arange(len(novelty), dtype=float), novelty, 1)[0]),
                6,
            ),
            "passed": bool(final_novelty_mean >= novelty_floor),
        }

    return {
        "passed": all(item["passed"] for item in evidence.values()),
        "rounds": len(rows),
        "window_size": window_size,
        "metrics": evidence,
    }


def write_submission_artifacts(
    suite: dict,
    config: dict,
    artifacts_dir: Path,
    run_stamp: str,
    gate_assessment: dict,
    final_assessment: dict,
) -> dict[str, Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    flattened_results = [
        result
        for run in suite["by_seed"]
        for result in run["results"]
    ]
    metrics = build_metrics_dump(flattened_results)
    explored = sorted({row["attack_family"] for row in metrics if row.get("attack_family")})
    missing = sorted(set(ATTACK_FAMILIES) - set(explored))

    result_artifact = {
        "run_timestamp_utc": run_stamp,
        "experiment": "adaptive_qwen_ctgan_continual_fixed_benchmark",
        "agent_backend": "QwenAgents",
        "generator_backend": "conditional_ctgan",
        "detector_mode": config["detector_mode"],
        "fraud_rate_target": config["pipeline"]["fraud_rate"],
        "seed_count": suite["seed_count"],
        "rounds": suite["rounds"],
        "summary": suite["summary"],
        "family_analysis": suite.get("family_analysis", []),
        "detector_version_analysis": suite.get("detector_version_analysis", []),
        "by_seed": suite["by_seed"],
    }
    summary = {
        "run_timestamp_utc": run_stamp,
        "experiment": result_artifact["experiment"],
        "agent_backend": result_artifact["agent_backend"],
        "generator_backend": result_artifact["generator_backend"],
        "detector_mode": result_artifact["detector_mode"],
        "rounds": suite["rounds"],
        "seed_count": suite["seed_count"],
        "families_explored": explored,
        "families_never_reached": missing,
        "five_round_gate": gate_assessment,
        "final_trend_assessment": final_assessment,
    }

    paths = {
        "results": artifacts_dir / f"adaptive_v2_results_{run_stamp}.json",
        "metrics": artifacts_dir / f"adaptive_v2_metrics_dump_{run_stamp}.json",
        "summary": artifacts_dir / f"adaptive_v2_summary_{run_stamp}.json",
    }
    payloads = {
        "results": result_artifact,
        "metrics": metrics,
        "summary": summary,
    }
    for name, path in paths.items():
        path.write_text(json.dumps(to_jsonable(payloads[name]), indent=2), encoding="utf-8")
    return paths