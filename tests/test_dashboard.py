from ui.generate_dashboard import CHART_SPECS, normalize_metrics


def test_dashboard_uses_final_metric_selection() -> None:
    chart_keys = {spec["key"] for spec in CHART_SPECS}

    assert "blue_team_benchmark_roc_auc" in chart_keys
    assert "family_coverage_diversity_ratio" in chart_keys
    assert "attack_diversity_channel_entropy" not in chart_keys
    assert "family_coverage_cumulative_ratio" not in chart_keys


def test_normalize_metrics_migrates_legacy_coverage_without_changing_value() -> None:
    metrics = normalize_metrics(
        [
            {
                "round": 1,
                "family_coverage_cumulative_ratio": 0.4286,
                "attack_diversity_channel_entropy": 0.9,
            }
        ]
    )

    assert metrics[0]["family_coverage_diversity_ratio"] == 0.4286
    assert "family_coverage_cumulative_ratio" not in metrics[0]
    assert "attack_diversity_channel_entropy" not in metrics[0]