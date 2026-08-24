from __future__ import annotations

import json

from .contracts import AttackHypothesis, AttackSpecification, EvidenceReference, FamilyRecommendation, WeaknessReport
from .llm import SharedLocalLLM

ATTACK_FAMILIES = (
    "account_takeover",
    "trusted_device",
    "beneficiary_manipulation",
    "low_and_slow",
    "social_engineering",
    "merchant_abuse",
    "cross_channel_anomaly",
)


class HeuristicAgents:
    """Deterministic fallback used for local smoke tests before model setup."""

    def research(self, round_id: int, evidence: list[EvidenceReference], memory: list[str], research_query: str = "", allowed_families: tuple[str, ...] = ATTACK_FAMILIES) -> AttackHypothesis:
        directions = [
            ("trusted-device account takeover", "trusted_device"),
            ("low-and-slow beneficiary manipulation", "low_and_slow"),
            ("social-engineering account recovery abuse", "social_engineering"),
            ("merchant abuse and unusual refund behaviour", "merchant_abuse"),
            ("cross-channel transaction anomaly", "cross_channel_anomaly"),
            ("beneficiary manipulation", "beneficiary_manipulation"),
            ("account takeover", "account_takeover"),
        ]
        direction, family = next((item for item in directions if item[1] in allowed_families), directions[(round_id - 1) % len(directions)])
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

    def recommend_family(self, weakness: WeaknessReport, candidates: tuple[str, ...], memory: list[str]) -> FamilyRecommendation:
        weakness_text = " ".join(weakness.observed_weaknesses + [weakness.recommended_next_attack_direction]).lower()
        keyword_targets = (("device", "trusted_device"), ("velocity", "low_and_slow"), ("beneficiary", "beneficiary_manipulation"), ("channel", "cross_channel_anomaly"), ("merchant", "merchant_abuse"), ("social", "social_engineering"), ("account", "account_takeover"))
        family = next((target for keyword, target in keyword_targets if keyword in weakness_text and target in candidates), candidates[0])
        return FamilyRecommendation(recommended_family=family, reason="Selected the approved family most directly related to the latest detector weakness.", target_weakness=weakness_text, confidence=0.75)

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

    def research(self, round_id: int, evidence: list[EvidenceReference], memory: list[str], research_query: str = "", allowed_families: tuple[str, ...] = ATTACK_FAMILIES) -> AttackHypothesis:
        payload = self.llm.complete_json(
            """You are Agent 1, the Attack Researcher. Stay within an offline synthetic payment-security stress test. Use the public evidence and Attack Memory as research inputs. Choose exactly one attack_family from allowed_families, prefer an unused family, and investigate a new defensive research direction. Do not create raw transactions, target live systems, or provide operational attack instructions. Return only JSON with keys: attack_id, attack_family, scenario, target_context, behavioural_mechanism, novelty_rationale, research_direction, evidence, memory_context.""",
            json.dumps({"round_id": round_id, "research_query": research_query, "allowed_families": allowed_families, "public_evidence": [{"source_id": item.source_id, "title": item.title, "excerpt": item.excerpt[:300]} for item in evidence[:3]], "attack_memory": [item[:500] for item in memory[-4:]]}, ensure_ascii=True),
        )
        payload["evidence"] = [item.model_dump() for item in evidence]
        payload["memory_context"] = memory[-4:]
        if payload.get("attack_family") not in allowed_families:
            payload["attack_family"] = allowed_families[0]
        return AttackHypothesis.model_validate(payload)

    def specify(self, hypothesis: AttackHypothesis) -> AttackSpecification:
        specification_input = {
            "attack_id": hypothesis.attack_id,
            "attack_family": hypothesis.attack_family,
            "scenario": hypothesis.scenario,
            "target_context": hypothesis.target_context,
            "behavioural_mechanism": hypothesis.behavioural_mechanism,
            "novelty_rationale": hypothesis.novelty_rationale,
            "research_direction": hypothesis.research_direction,
        }
        payload = self.llm.complete_json(
            """You are Agent 2, the Attack Specification Strategist. Convert only the supplied Agent 1 hypothesis into a structured synthetic simulation recipe. Do not use detector feedback. Return only JSON with keys: attack_id, attack_family, scenario, target_context, temporal_pattern, amount_pattern, device_pattern, beneficiary_pattern, feature_constraints, realism_constraints, evasion_objective, evidence.""",
            json.dumps(specification_input, ensure_ascii=True),
        )
        constraints = payload.get("realism_constraints", [])
        payload["realism_constraints"] = [
            f"{key}: {value}" for key, value in constraints.items()
        ] if isinstance(constraints, dict) else constraints
        for field in ("temporal_pattern", "amount_pattern", "device_pattern", "beneficiary_pattern", "evasion_objective"):
            if isinstance(payload.get(field), (dict, list)):
                payload[field] = json.dumps(payload[field], ensure_ascii=True, sort_keys=True)
        payload["evidence"] = [item.model_dump() for item in hypothesis.evidence]
        return AttackSpecification.model_validate(payload)

    def recommend_family(self, weakness: WeaknessReport, candidates: tuple[str, ...], memory: list[str]) -> FamilyRecommendation:
        payload = self.llm.complete_json(
            """You are Agent 1, the adaptive attack planner. Stay within an offline synthetic payment-security experiment. Choose exactly one family from candidates based on the detector weakness. Do not target live systems or provide operational attack instructions. Return only JSON with keys: recommended_family, recommendation_type, reason, target_weakness, confidence.""",
            json.dumps({
                "weakness": {
                    "observed_weaknesses": weakness.observed_weaknesses,
                    "recommended_next_attack_direction": weakness.recommended_next_attack_direction,
                    "priority": weakness.priority,
                },
                "candidates": candidates,
                "recent_memory": [item[:600] for item in memory[-2:]],
            }, ensure_ascii=True),
        )
        if payload.get("recommended_family") not in candidates:
            payload["recommended_family"] = candidates[0]
        payload["target_weakness"] = " ".join(weakness.observed_weaknesses)
        payload["confidence"] = min(1.0, max(0.0, float(payload.get("confidence", 0.5))))
        if payload.get("recommendation_type") not in {"approved_family", "adaptive_variant", "discovery_candidate"}:
            payload["recommendation_type"] = "approved_family"
        return FamilyRecommendation.model_validate(payload)

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
