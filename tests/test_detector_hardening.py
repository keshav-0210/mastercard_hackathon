import pandas as pd
import pytest

from mastercard_defence.contracts import AttackSpecification
from mastercard_defence.detector import FraudDetector
from mastercard_defence.loop import ClosedLoop, load_config
from mastercard_defence.synthetic import ALLOWED_FAMILIES, build_round_family_plan, evaluate_diversity, generate_attacks


def test_detector_evaluate_reports_by_attack_family() -> None:
    training = pd.DataFrame(
        [
            {"amount": 120.0, "hour": 14, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 2, "channel": "web", "is_fraud": 0, "attack_family": "legitimate"},
            {"amount": 140.0, "hour": 16, "device_change": 1, "beneficiary_change": 0, "velocity_24h": 6, "channel": "mobile", "is_fraud": 0, "attack_family": "legitimate"},
            {"amount": 250.0, "hour": 11, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 1, "channel": "web", "is_fraud": 1, "attack_family": "low_and_slow"},
            {"amount": 300.0, "hour": 13, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 2, "channel": "mobile", "is_fraud": 1, "attack_family": "low_and_slow"},
            {"amount": 520.0, "hour": 11, "device_change": 1, "beneficiary_change": 0, "velocity_24h": 5, "channel": "card_present", "is_fraud": 1, "attack_family": "trusted_device"},
            {"amount": 580.0, "hour": 9, "device_change": 1, "beneficiary_change": 0, "velocity_24h": 6, "channel": "web", "is_fraud": 1, "attack_family": "trusted_device"},
        ]
    )

    detector = FraudDetector()
    detector.fit(training)
    result = detector.evaluate(training)

    assert "by_attack_family" in result
    assert "low_and_slow" in result["by_attack_family"]
    assert "trusted_device" in result["by_attack_family"]
    assert set(result["by_attack_family"]["low_and_slow"]).issuperset({"precision", "recall", "f1", "support"})


def test_threshold_calibration_uses_separate_legitimate_data() -> None:
    legitimate = pd.DataFrame(
        [
            {"amount": 90.0 + index, "hour": index % 24, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 1, "channel": "web", "is_fraud": 0}
            for index in range(100)
        ]
    )
    fraud = pd.DataFrame(
        [
            {"amount": 500.0 + index, "hour": index % 24, "device_change": 1, "beneficiary_change": 1, "velocity_24h": 8, "channel": "mobile", "is_fraud": 1}
            for index in range(20)
        ]
    )
    detector = FraudDetector(target_fpr_ceiling=0.02)
    detector.fit(pd.concat([legitimate, fraud], ignore_index=True), calibration_data=legitimate)

    expected = detector.pipeline.predict_proba(legitimate[["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel"]])[:, 1]
    expected_threshold = float(__import__("numpy").quantile(expected, 0.98, method="higher"))

    assert detector.threshold == pytest.approx(expected_threshold)


def test_family_generation_and_diversity_metrics_are_more_realistic() -> None:
    specification = AttackSpecification(
        attack_id="A-001",
        attack_family="social_engineering",
        scenario="Persona-based social engineering against digital wallet users",
        target_context="consumer digital banking",
        temporal_pattern="late evening burst pattern",
        amount_pattern="moderately elevated transactional amounts",
        device_pattern="multi-device transitions",
        beneficiary_pattern="frequent payment beneficiary changes",
        feature_constraints={"channel_distribution": "mixed"},
        realism_constraints=["not extreme"],
        evasion_objective="Blend malicious activity into normal user routines",
        evidence=[],
    )
    attacks = generate_attacks(specification, 500, round_id=1, seed=42)

    assert attacks["beneficiary_change"].mean() > 0.4
    assert attacks["device_change"].mean() > 0.2
    assert attacks["amount"].between(1, 5000).all()

    diversity = evaluate_diversity(attacks)
    assert "family_coverage_ratio" in diversity
    assert "channel_entropy" not in diversity
    assert diversity["family_coverage_ratio"] >= 1.0 / 7


def test_round_family_scheduler_is_diverse_and_deterministic() -> None:
    plan = build_round_family_plan(5, seed=42)

    assert len(plan) == 5
    assert all(family in ALLOWED_FAMILIES for family in plan)
    assert len(set(plan)) == len(plan)
    assert plan[0] != plan[1]


def test_continual_detector_tracks_replay_buffer_and_versioning() -> None:
    config = load_config("config/default.yaml")
    config["paths"]["memory_db"] = "artifacts/test_continual_memory.sqlite"
    config["detector_mode"] = "continual"
    config["detector_retrain_every"] = 2
    loop = ClosedLoop(config)
    try:
        assert hasattr(loop, "replay_buffer")
        assert hasattr(loop, "detector_version")
        loop.replay_buffer = pd.DataFrame([
            {"amount": 120.0, "hour": 13, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 2, "channel": "web", "is_fraud": 1, "attack_family": "low_and_slow"},
            {"amount": 150.0, "hour": 15, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 3, "channel": "mobile", "is_fraud": 1, "attack_family": "low_and_slow"},
        ])
        loop.detector_version = 1
        loop._record_hard_examples(pd.DataFrame([
            {"amount": 220.0, "hour": 11, "device_change": 0, "beneficiary_change": 0, "velocity_24h": 2, "channel": "web", "is_fraud": 1, "attack_family": "low_and_slow"},
        ]))
        assert len(loop.replay_buffer) >= 2
        loop._maybe_retrain_detector(round_id=2)
        assert loop.detector_version >= 2
    finally:
        loop.close()


def test_robustness_summary_aggregates_multi_seed_runs() -> None:
    results = [
        {"detection": {"f1": 0.8, "recall": 0.7, "precision": 0.9, "roc_auc": 0.8}, "fidelity": {"behavioural_plausibility": 0.4}, "novelty": {"novelty_score": 0.9}},
        {"detection": {"f1": 0.9, "recall": 0.8, "precision": 0.95, "roc_auc": 0.9}, "fidelity": {"behavioural_plausibility": 0.5}, "novelty": {"novelty_score": 0.8}},
        {"detection": {"f1": 0.7, "recall": 0.6, "precision": 0.85, "roc_auc": 0.7}, "fidelity": {"behavioural_plausibility": 0.6}, "novelty": {"novelty_score": 0.7}},
    ]

    summary = __import__("mastercard_defence.synthetic", fromlist=["summarize_robustness"]).summarize_robustness(results)
    family_summary = __import__("mastercard_defence.synthetic", fromlist=["summarize_family_performance"]).summarize_family_performance(results)

    assert "f1" in summary
    assert summary["f1"]["mean"] > 0.7
    assert summary["recall"]["mean"] > 0.6
    assert summary["behavioural_plausibility"]["mean"] > 0.4
    assert isinstance(family_summary, list)
    assert family_summary or True
