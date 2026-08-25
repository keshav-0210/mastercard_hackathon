import json

from mastercard_defence.agents import GENERATABLE_FAMILIES, HeuristicAgents
from mastercard_defence.contracts import WeaknessReport

DETECTION = {
    "precision": 0.6,
    "recall": 0.5,
    "f1": 0.55,
    "roc_auc": 0.7,
    "false_positive_rate": 0.1,
    "confusion_matrix": [[10, 1], [2, 3]],
    "by_attack_family": {
        "account_takeover": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 5},
        "low_and_slow": {"precision": 0.2, "recall": 0.1, "f1": 0.13, "support": 5},
    },
}
FIDELITY = {"behavioural_plausibility": 0.4, "passed": True}


def test_heuristic_agent3_analyze_produces_full_schema() -> None:
    report = HeuristicAgents().analyze(round_id=3, detection=DETECTION, fidelity=FIDELITY)

    assert isinstance(report, WeaknessReport)
    assert report.round_id == 3
    assert report.detector_version is None
    assert isinstance(report.analysis_summary, str) and report.analysis_summary
    assert isinstance(report.observed_weaknesses, list) and report.observed_weaknesses
    assert isinstance(report.hard_sample_patterns, list)
    assert isinstance(report.affected_attack_families, list)
    assert set(report.affected_attack_families).issubset(set(GENERATABLE_FAMILIES))
    assert isinstance(report.supporting_evidence, list) and report.supporting_evidence
    assert report.priority in {"low", "medium", "high"}
    assert isinstance(report.recommended_next_attack_directions, list) and report.recommended_next_attack_directions
    assert 0.0 <= report.confidence <= 1.0


def test_analyze_passes_through_detector_version_and_hard_sample_patterns() -> None:
    report = HeuristicAgents().analyze(
        round_id=1,
        detection=DETECTION,
        fidelity=FIDELITY,
        detector_version="detector_v3",
        hard_sample_patterns=["low-confidence borderline low_and_slow rows"],
    )

    assert report.detector_version == "detector_v3"
    assert report.hard_sample_patterns == ["low-confidence borderline low_and_slow rows"]


def test_weakness_report_schema_round_trips_through_json() -> None:
    report = WeaknessReport(
        round_id=2,
        detector_version="detector_v2",
        analysis_summary="Detector underperforms on low_and_slow.",
        observed_weaknesses=["Recall is low for slow-drip fraud."],
        hard_sample_patterns=["small amounts spread over many hours"],
        affected_attack_families=["low_and_slow"],
        supporting_evidence=["recall=0.10"],
        priority="high",
        recommended_next_attack_directions=["Explore lower-velocity variants."],
        confidence=0.8,
    )

    restored = WeaknessReport.model_validate(json.loads(report.model_dump_json()))

    assert restored == report
