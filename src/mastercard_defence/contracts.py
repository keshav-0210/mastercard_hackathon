from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EvidenceReference(BaseModel):
    source_id: str
    title: str
    excerpt: str


class AttackHypothesis(BaseModel):
    attack_id: str
    attack_family: str
    scenario: str
    target_context: str
    behavioural_mechanism: str
    novelty_rationale: str
    research_direction: str
    evidence: list[EvidenceReference] = Field(default_factory=list)
    memory_context: list[str] = Field(default_factory=list)


class AttackSpecification(BaseModel):
    attack_id: str
    attack_family: str
    scenario: str
    target_context: str
    temporal_pattern: str
    amount_pattern: str
    device_pattern: str
    beneficiary_pattern: str
    feature_constraints: dict[str, Any] = Field(default_factory=dict)
    realism_constraints: list[str] = Field(default_factory=list)
    evasion_objective: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


class GeneratedTransaction(BaseModel):
    transaction_id: str
    attack_id: str
    attack_family: str
    is_fraud: Literal[0, 1]
    amount: float
    hour: int
    device_change: int
    beneficiary_change: int
    velocity_24h: int
    channel: str
    generation_round: int
    generation_method: str


class DetectionResult(BaseModel):
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_positive_rate: float
    confusion_matrix: list[list[int]]
    by_attack_family: dict[str, dict[str, float]] = Field(default_factory=dict)


class FidelityReport(BaseModel):
    amount_mean_delta: float
    amount_std_delta: float
    fraud_rate: float
    behavioural_plausibility: float
    passed: bool
    notes: list[str] = Field(default_factory=list)


class WeaknessReport(BaseModel):
    round_id: int
    observed_weaknesses: list[str]
    weakness_families: list[str] = Field(default_factory=list)
    supporting_evidence: list[str]
    priority: Literal["low", "medium", "high"]
    recommended_next_attack_direction: str
    confidence: float = Field(ge=0.0, le=1.0)


class FamilyRecommendation(BaseModel):
    recommended_family: str
    recommendation_type: Literal["approved_family", "adaptive_variant", "discovery_candidate"] = "approved_family"
    reason: str
    target_weakness: str
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryRecord(BaseModel):
    round_id: int
    record_type: Literal["hypothesis", "specification", "evaluation", "weakness"]
    content: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)
