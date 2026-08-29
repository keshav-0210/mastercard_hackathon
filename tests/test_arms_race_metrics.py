import tempfile

from mastercard_defence.loop import ClosedLoop, load_config
from mastercard_defence.synthetic import build_metrics_dump, summarize_detector_version_performance


def test_cumulative_coverage_redundancy_and_historical_robustness_track_across_rounds() -> None:
    config = load_config("config/default.yaml")
    config["paths"]["memory_db"] = tempfile.mktemp(suffix="_arms_race_memory.sqlite")
    config["detector_mode"] = "continual"
    config["pipeline"]["synthetic_transactions"] = 200
    config["pipeline"]["max_generated_attacks"] = 20
    loop = ClosedLoop(config)
    try:
        results = loop.run(rounds=8, seed=555001)
    finally:
        loop.close()

    assert len(results) == 8

    coverage = [result["diversity"]["family_coverage_diversity_ratio"] for result in results]
    assert coverage == sorted(coverage)
    assert coverage[-1] > coverage[0]
    assert all(0.0 <= value <= 1.0 for value in coverage)

    assert all("cross_round_redundancy_ratio" in result["diversity"] for result in results)
    assert all("cross_round_unique_ratio" in result["diversity"] for result in results)
    assert results[0]["diversity"]["cross_round_redundancy_ratio"] == 0.0

    assert all("detector_version" in result["detection"] for result in results)
    assert all("attack_difficulty_score" in result["detection"] for result in results)
    assert all("historical_robustness" in result["detection"] for result in results)
    assert all("blue_team_benchmark" in result["detection"] for result in results)
    assert all(result["detection"]["blue_team_benchmark"]["evaluation_protocol"] == "fixed_unseen_seven_family_benchmark" for result in results)
    assert results[0]["detection"]["historical_robustness"]["insufficient_history"] is True
    assert results[-1]["detection"]["historical_robustness"]["insufficient_history"] is False

    dump = build_metrics_dump(results)
    assert len(dump) == 8
    required_keys = {
        "round", "attack_family", "detector_version",
        "attack_diversity_unique_row_ratio",
        "attack_fidelity_behavioural_plausibility", "attack_novelty_score", "attack_difficulty_score",
        "family_coverage_diversity_ratio", "family_coverage_diversity_count",
        "variant_redundancy_ratio", "variant_unique_ratio",
        "precision", "recall", "f1", "roc_auc", "false_positive_rate",
        "decision_threshold", "blue_team_benchmark_precision", "blue_team_benchmark_recall",
        "blue_team_benchmark_f1", "blue_team_benchmark_roc_auc",
        "blue_team_benchmark_false_positive_rate", "blue_team_benchmark_protocol",
        "unseen_attack_evaluation_protocol",
        "historical_robustness_insufficient", "historical_robustness_precision",
        "historical_robustness_recall", "historical_robustness_f1", "historical_robustness_roc_auc",
    }
    assert required_keys.issubset(dump[0].keys())
    assert "attack_diversity_channel_entropy" not in dump[0]
    assert "family_coverage_cumulative_ratio" not in dump[0]
    assert dump[0]["attack_family"] is not None
    assert dump[0]["blue_team_benchmark_protocol"] == "fixed_unseen_seven_family_benchmark"


def test_summarize_detector_version_performance_groups_by_version() -> None:
    results = [
        {"detection": {"detector_version": 1, "precision": 0.7, "recall": 0.5, "f1": 0.58, "roc_auc": 0.8}},
        {"detection": {"detector_version": 1, "precision": 0.75, "recall": 0.55, "f1": 0.63, "roc_auc": 0.82}},
        {"detection": {"detector_version": 2, "precision": 0.85, "recall": 0.65, "f1": 0.74, "roc_auc": 0.9}},
    ]

    summary = summarize_detector_version_performance(results)

    assert len(summary) == 2
    assert summary[0]["detector_version"] == 1
    assert summary[0]["round_count"] == 2
    assert summary[1]["detector_version"] == 2
    assert summary[1]["f1"] > summary[0]["f1"]
