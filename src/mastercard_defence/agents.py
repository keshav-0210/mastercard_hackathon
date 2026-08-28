from __future__ import annotations

import json

from .contracts import APPROVED_FAMILIES, AttackHypothesis, AttackSpecification, EvidenceReference, FamilyRecommendation, WeaknessReport
from .llm import SharedLocalLLM

ATTACK_FAMILIES = APPROVED_FAMILIES

GENERATABLE_FAMILIES = ATTACK_FAMILIES


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
        weakness_text = " ".join(weakness.observed_weaknesses + weakness.recommended_next_attack_directions).lower()
        keyword_targets = (("device", "trusted_device"), ("velocity", "low_and_slow"), ("beneficiary", "beneficiary_manipulation"), ("channel", "cross_channel_anomaly"), ("merchant", "merchant_abuse"), ("social", "social_engineering"), ("account", "account_takeover"))
        family = next((target for keyword, target in keyword_targets if keyword in weakness_text and target in candidates), candidates[0])
        return FamilyRecommendation(recommended_family=family, reason="Selected the approved family most directly related to the latest detector weakness.", target_weakness=weakness_text, confidence=0.75)

    def analyze(self, round_id: int, detection: dict, fidelity: dict, detector_version: str | None = None, hard_sample_patterns: list[str] | None = None) -> WeaknessReport:
        weak_recall = detection.get("recall", 0) < 0.7
        weakness = "Detector should be tested against attacks with low velocity." if not weak_recall else "Detector recall is weak on the current synthetic attack family."
        family_metrics = detection.get("by_attack_family", {})
        affected_attack_families = [family for family, metrics in sorted(family_metrics.items(), key=lambda item: item[1].get("recall", 0.0))[:2]]
        patterns = hard_sample_patterns if hard_sample_patterns is not None else ([f"Low recall concentrated in {', '.join(affected_attack_families)}"] if affected_attack_families else [])
        return WeaknessReport(
            round_id=round_id, detector_version=detector_version,
            analysis_summary=f"Detector f1={detection.get('f1', 0):.3f}, recall={detection.get('recall', 0):.3f} on this round's evaluation set.",
            observed_weaknesses=[weakness],
            hard_sample_patterns=patterns,
            affected_attack_families=affected_attack_families,
            supporting_evidence=[f"F1={detection.get('f1', 0):.3f}", f"fidelity={fidelity.get('behavioural_plausibility', 0):.3f}"],
            priority="high" if weak_recall else "medium",
            recommended_next_attack_directions=["Test trusted-device and low-and-slow variants with reduced velocity signals."],
            confidence=0.65,
        )


class QwenAgents:
    """Three logical roles backed by one shared local Qwen model."""

    def __init__(self, config: dict, llm: SharedLocalLLM | None = None) -> None:
        self.llm = llm or SharedLocalLLM(config)

    def research(self, round_id: int, evidence: list[EvidenceReference], memory: list[str], research_query: str = "", allowed_families: tuple[str, ...] = ATTACK_FAMILIES) -> AttackHypothesis:
        payload = self.llm.complete_json(
            """You are Agent 1, the Attack Researcher. Stay within an offline synthetic payment-security stress test. Use the evidence and memory as inputs. Choose exactly one attack_family from allowed_families. Return one compact JSON object only with short one-sentence string values for attack_id, attack_family, scenario, target_context, behavioural_mechanism, novelty_rationale, and research_direction. Do not include evidence or memory_context; the controller supplies those fields. Do not create raw transactions, target live systems, or provide operational attack instructions.""",
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
        if isinstance(constraints, dict):
            constraints = [f"{key}: {value}" for key, value in constraints.items()]
        elif isinstance(constraints, str):
            constraints = [constraints]
        elif not isinstance(constraints, list):
            constraints = [str(constraints)]
        payload["realism_constraints"] = constraints
        feature_constraints = payload.get("feature_constraints", {})
        if isinstance(feature_constraints, str):
            feature_constraints = {"note": feature_constraints}
        elif isinstance(feature_constraints, list):
            feature_constraints = {"items": feature_constraints}
        elif not isinstance(feature_constraints, dict):
            feature_constraints = {"note": str(feature_constraints)}
        payload["feature_constraints"] = feature_constraints
        for field in ("temporal_pattern", "amount_pattern", "device_pattern", "beneficiary_pattern", "evasion_objective"):
            if isinstance(payload.get(field), (dict, list)):
                payload[field] = json.dumps(payload[field], ensure_ascii=True, sort_keys=True)
        # Qwen occasionally omits a required string field from its JSON output; fall back to the
        # hypothesis's own text instead of letting the whole round fail on a validation error.
        required_string_fields = {
            "scenario": hypothesis.scenario,
            "target_context": hypothesis.target_context,
            "temporal_pattern": hypothesis.behavioural_mechanism,
            "amount_pattern": hypothesis.behavioural_mechanism,
            "device_pattern": hypothesis.behavioural_mechanism,
            "beneficiary_pattern": hypothesis.behavioural_mechanism,
            "evasion_objective": hypothesis.novelty_rationale,
        }
        for field, fallback in required_string_fields.items():
            value = payload.get(field)
            if not isinstance(value, str) or not value.strip():
                payload[field] = fallback
        # attack_id/attack_family are authoritative from the hypothesis (the caller overrides
        # attack_family right after this call anyway); never let Qwen's echo of these break validation.
        payload["attack_id"] = hypothesis.attack_id
        payload["attack_family"] = hypothesis.attack_family
        payload["evidence"] = [item.model_dump() for item in hypothesis.evidence]
        return AttackSpecification.model_validate(payload)

    def recommend_family(self, weakness: WeaknessReport, candidates: tuple[str, ...], memory: list[str]) -> FamilyRecommendation:
        payload = self.llm.complete_json(
            """You are Agent 1, the adaptive attack planner. Stay within an offline synthetic payment-security experiment. Choose exactly one family from candidates based on the detector weakness. Do not target live systems or provide operational attack instructions. Return only JSON with keys: recommended_family, recommendation_type, reason, target_weakness, confidence.""",
            json.dumps({
                "weakness": {
                    "observed_weaknesses": weakness.observed_weaknesses,
                    "affected_attack_families": weakness.affected_attack_families,
                    "recommended_next_attack_directions": weakness.recommended_next_attack_directions,
                    "priority": weakness.priority,
                },
                "candidates": candidates,
                "recent_memory": [item[:600] for item in memory[-2:]],
            }, ensure_ascii=True),
        )
        if payload.get("recommended_family") not in candidates:
            payload["recommended_family"] = candidates[0]
        payload["target_weakness"] = " ".join(weakness.observed_weaknesses)
        confidence = payload.get("confidence", 0.5)
        if isinstance(confidence, str):
            confidence = {"low": 0.35, "medium": 0.6, "high": 0.85}.get(confidence.lower(), 0.5)
        payload["confidence"] = min(1.0, max(0.0, float(confidence)))
        payload["recommendation_type"] = "approved_family"
        return FamilyRecommendation.model_validate(payload)

    def analyze(self, round_id: int, detection: dict, fidelity: dict, detector_version: str | None = None, hard_sample_patterns: list[str] | None = None) -> WeaknessReport:
        payload = self.llm.complete_json(
            """You are Agent 3, the Security Analyst. Analyze detector and fidelity evidence from an offline synthetic experiment. Return only JSON with keys: analysis_summary, observed_weaknesses, hard_sample_patterns, affected_attack_families, supporting_evidence, priority, recommended_next_attack_directions, confidence. affected_attack_families must list the attack family labels whose metrics show the weakness. hard_sample_patterns must list short descriptions of recurring hard-to-classify patterns; do not hard-code a single fixed weakness category. recommended_next_attack_directions must be a list of distinct next directions. Your report will be stored in Attack Memory and consumed by Agent 1 in the next round, never directly by Agent 2.""",
            json.dumps({"round_id": round_id, "detection": detection, "fidelity": fidelity}, ensure_ascii=True),
        )
        payload["round_id"] = round_id
        payload["detector_version"] = detector_version
        payload["analysis_summary"] = str(payload.get("analysis_summary", ""))
        weaknesses = payload.get("observed_weaknesses", [])
        payload["observed_weaknesses"] = [weaknesses] if isinstance(weaknesses, str) else weaknesses
        patterns = hard_sample_patterns if hard_sample_patterns is not None else (payload.get("hard_sample_patterns", []) or [])
        payload["hard_sample_patterns"] = [patterns] if isinstance(patterns, str) else patterns
        payload["affected_attack_families"] = payload.get("affected_attack_families", []) or []
        if isinstance(payload["affected_attack_families"], str):
            payload["affected_attack_families"] = [payload["affected_attack_families"]]
        payload["affected_attack_families"] = [
            family for family in payload["affected_attack_families"]
            if family in GENERATABLE_FAMILIES
        ]
        directions = payload.get("recommended_next_attack_directions", [])
        payload["recommended_next_attack_directions"] = [directions] if isinstance(directions, str) else directions
        payload["supporting_evidence"] = [
            f"Detection metrics: {json.dumps(detection, ensure_ascii=True)}",
            f"Fidelity metrics: {json.dumps(fidelity, ensure_ascii=True)}",
        ]
        payload["priority"] = str(payload.get("priority", "medium")).lower()
        if payload["priority"] not in {"low", "medium", "high"}:
            payload["priority"] = "medium"
        confidence = payload.get("confidence", 0.5)
        if isinstance(confidence, str):
            confidence = {"low": 0.35, "medium": 0.6, "high": 0.85}.get(confidence.lower(), 0.5)
        payload["confidence"] = min(1.0, max(0.0, float(confidence)))
        return WeaknessReport.model_validate(payload)
