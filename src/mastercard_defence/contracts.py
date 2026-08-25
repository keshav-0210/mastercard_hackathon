from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


APPROVED_FAMILIES = (
    "account_takeover",
    "trusted_device",
    "beneficiary_manipulation",
    "low_and_slow",
    "social_engineering",
    "merchant_abuse",
    "cross_channel_anomaly",
)


def validate_family(value: str) -> str:
    if value not in APPROVED_FAMILIES:
        raise ValueError(f"attack_family must be one of {APPROVED_FAMILIES}")
    return value


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

    _validate_family = field_validator("attack_family")(validate_family)


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

    _validate_family = field_validator("attack_family")(validate_family)


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

    _validate_family = field_validator("attack_family")(validate_family)


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
    detector_version: str | None = None
    analysis_summary: str = ""
    observed_weaknesses: list[str]
    hard_sample_patterns: list[str] = Field(default_factory=list)
    affected_attack_families: list[str] = Field(default_factory=list)
    supporting_evidence: list[str]
    priority: Literal["low", "medium", "high"]
    recommended_next_attack_directions: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("affected_attack_families")
    @classmethod
    def validate_affected_families(cls, values: list[str]) -> list[str]:
        invalid = set(values) - set(APPROVED_FAMILIES)
        if invalid:
            raise ValueError(f"affected_attack_families contains unsupported labels: {sorted(invalid)}")
        return values


class FamilyRecommendation(BaseModel):
    recommended_family: str
    recommendation_type: Literal["approved_family"] = "approved_family"
    reason: str
    target_weakness: str
    confidence: float = Field(ge=0.0, le=1.0)

    _validate_family = field_validator("recommended_family")(validate_family)


class RoundRecord(BaseModel):
    """One immutable, append-only Attack Memory entry per (seed, round_id); never overwritten."""

    seed: int
    round_id: int
    attack_id: str
    fraud_family: str
    status: Literal["explored"] = "explored"
    detector_version: str | None = None
    attack_hypothesis: dict[str, Any] = Field(default_factory=dict)
    attack_specification: dict[str, Any] = Field(default_factory=dict)
    generator_metadata: dict[str, Any] = Field(default_factory=dict)
    generation_stats: dict[str, Any] = Field(default_factory=dict)
    fidelity_evaluation: dict[str, Any] = Field(default_factory=dict)
    detector_evaluation: dict[str, Any] = Field(default_factory=dict)
    hard_sample_summary: dict[str, Any] = Field(default_factory=dict)
    agent3_analysis: dict[str, Any] = Field(default_factory=dict)
    identified_weaknesses: list[str] = Field(default_factory=list)
    recommended_next_attack_directions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    _validate_family = field_validator("fraud_family")(validate_family)
