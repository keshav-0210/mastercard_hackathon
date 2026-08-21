from __future__ import annotations

import json

from .contracts import AttackHypothesis, AttackSpecification, EvidenceReference, WeaknessReport
from .llm import SharedLocalLLM


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


class QwenAgents:
    """Three logical roles backed by one shared local Qwen model."""

    def __init__(self, config: dict, llm: SharedLocalLLM | None = None) -> None:
        self.llm = llm or SharedLocalLLM(config)

    def research(self, round_id: int, evidence: list[EvidenceReference], memory: list[str]) -> AttackHypothesis:
        payload = self.llm.complete_json(
            """You are Agent 1, the Attack Researcher. Stay within an offline synthetic payment-security stress test. Return only JSON with keys: attack_id, attack_family, scenario, target_context, behavioural_mechanism, novelty_rationale, research_direction, evidence, memory_context. Do not create raw transactions.""",
            json.dumps({"round_id": round_id, "public_evidence": [{"source_id": item.source_id, "excerpt": item.excerpt[:300]} for item in evidence[:3]], "attack_memory": [item[:500] for item in memory[-4:]]}, ensure_ascii=True),
        )
        payload["evidence"] = [item.model_dump() for item in evidence]
        payload["memory_context"] = memory[-4:]
        return AttackHypothesis.model_validate(payload)

    def specify(self, hypothesis: AttackHypothesis) -> AttackSpecification:
        payload = self.llm.complete_json(
            """You are Agent 2, the Attack Specification Strategist. Convert only the supplied Agent 1 hypothesis into a structured synthetic simulation recipe. Do not use detector feedback. Return only JSON with keys: attack_id, attack_family, scenario, target_context, temporal_pattern, amount_pattern, device_pattern, beneficiary_pattern, feature_constraints, realism_constraints, evasion_objective, evidence.""",
            hypothesis.model_dump_json(),
        )
        return AttackSpecification.model_validate(payload)

    def analyze(self, round_id: int, detection: dict, fidelity: dict) -> WeaknessReport:
        payload = self.llm.complete_json(
            """You are Agent 3, the Security Analyst. Analyze detector and fidelity evidence from an offline synthetic experiment. Return only JSON with keys: round_id, observed_weaknesses, supporting_evidence, priority, recommended_next_attack_direction, confidence. Your report will be stored in Attack Memory and consumed by Agent 1 in the next round.""",
            json.dumps({"round_id": round_id, "detection": detection, "fidelity": fidelity}, ensure_ascii=True),
        )
        payload["round_id"] = round_id
        weaknesses = payload.get("observed_weaknesses", [])
        payload["observed_weaknesses"] = [weaknesses] if isinstance(weaknesses, str) else weaknesses
        payload["supporting_evidence"] = [
            f"Detection metrics: {json.dumps(detection, ensure_ascii=True)}",
            f"Fidelity metrics: {json.dumps(fidelity, ensure_ascii=True)}",
        ]
        payload["priority"] = str(payload.get("priority", "medium")).lower()
        if payload["priority"] not in {"low", "medium", "high"}:
            payload["priority"] = "medium"
        payload["confidence"] = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
        return WeaknessReport.model_validate(payload)
