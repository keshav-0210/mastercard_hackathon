import numpy as np
import pytest

from mastercard_defence.submission import assess_submission_trends, to_jsonable


def _improving_rows(rounds: int = 10) -> list[dict]:
    rows = []
    for round_id in range(1, rounds + 1):
        progress = round_id / rounds
        rows.append(
            {
                "round": round_id,
                "blue_team_benchmark_f1": 0.3 + 0.3 * progress,
                "blue_team_benchmark_recall": 0.35 + 0.25 * progress,
                "blue_team_benchmark_precision": 0.4 + 0.2 * progress,
                "blue_team_benchmark_roc_auc": 0.7 + 0.2 * progress,
                "blue_team_benchmark_false_positive_rate": 0.03 - 0.015 * progress,
                "attack_fidelity_behavioural_plausibility": 0.5 + 0.2 * progress,
                "family_coverage_diversity_ratio": min(1.0, round_id / 7),
                "attack_novelty_score": 0.82,
            }
        )
    return rows


def test_assess_submission_trends_accepts_expected_directions() -> None:
    assessment = assess_submission_trends(
        _improving_rows(),
        window_size=3,
        include_red_team=True,
    )

    assert assessment["passed"] is True
    assert assessment["metrics"]["blue_team_benchmark_roc_auc"]["passed"] is True
    assert assessment["metrics"]["family_coverage_diversity_ratio"]["last_value"] == 1.0


def test_assess_submission_trends_rejects_declining_auc() -> None:
    rows = _improving_rows()
    for row in rows:
        row["blue_team_benchmark_roc_auc"] = 0.95 - row["round"] * 0.01

    assessment = assess_submission_trends(
        rows,
        window_size=3,
        include_red_team=False,
    )

    assert assessment["passed"] is False
    assert assessment["metrics"]["blue_team_benchmark_roc_auc"]["passed"] is False


def test_assess_submission_trends_rejects_missing_values() -> None:
    rows = _improving_rows()
    rows[4]["blue_team_benchmark_f1"] = None

    with pytest.raises(ValueError, match="blue_team_benchmark_f1"):
        assess_submission_trends(rows, window_size=3, include_red_team=False)


def test_to_jsonable_converts_numpy_scalars() -> None:
    value = to_jsonable({"float": np.float64(0.75), "integer": np.int64(7)})

    assert value == {"float": 0.75, "integer": 7}