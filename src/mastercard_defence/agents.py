from __future__ import annotations

from .contracts import AttackHypothesis, AttackSpecification, EvidenceReference, WeaknessReport


class HeuristicAgents:
    """Deterministic fallback used for local smoke tests before model setup."""

    def research(self, round_id: int, evidence: list[EvidenceReference], memory: list[str]) -> AttackHypothesis:
        direction = "trusted-device account takeover" if any("device" in item.lower() for item in memory) else "low-and-slow beneficiary manipulation"
        family = "trusted_device" if "trusted" in direction else "low_and_slow"
        return AttackHypothesis(
            attack_id=f"round-{round_id}-{family}", attack_family=family,
            scenario=f"Synthetic {direction} payment pattern",
            target_context="Offline synthetic payment security stress test",
            behavioural_mechanism="Fraudulent activity is shaped to reduce reliance on one obvious signal.",
            novelty_rationale="Selects a direction not used in the prior memory context.",
            research_direction=direction, evidence=evidence, memory_context=memory[-4:],
        )

    def specify(self, hypothesis: AttackHypothesis) -> AttackSpecification:
        return AttackSpecification(
            attack_id=hypothesis.attack_id, attack_family=hypothesis.attack_family,
            scenario=hypothesis.scenario, target_context=hypothesis.target_context,
            temporal_pattern="mixed hours with a short burst for validation",
            amount_pattern="moderate amounts with controlled variance",
            device_pattern="trusted device" if hypothesis.attack_family == "trusted_device" else "new device mix",
            beneficiary_pattern="new beneficiary mix", feature_constraints={"synthetic_only": True},
            realism_constraints=["stay within synthetic schema", "retain attack labels and round metadata"],
            evasion_objective="Expose detector reliance on a single behavioural feature.", evidence=hypothesis.evidence,
        )

    def analyze(self, round_id: int, detection: dict, fidelity: dict) -> WeaknessReport:
        weakness = "Detector should be tested against attacks with low velocity." if detection.get("recall", 0) > 0.7 else "Detector recall is weak on the current synthetic attack family."
        return WeaknessReport(
            round_id=round_id, observed_weaknesses=[weakness],
            supporting_evidence=[f"F1={detection.get('f1', 0):.3f}", f"fidelity={fidelity.get('behavioural_plausibility', 0):.3f}"],
            priority="high" if detection.get("recall", 0) < 0.7 else "medium",
            recommended_next_attack_direction="Test trusted-device and low-and-slow variants with reduced velocity signals.",
            confidence=0.65,
        )
