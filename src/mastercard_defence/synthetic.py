from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AttackHypothesis, AttackSpecification, GeneratedTransaction

CHANNELS = ("web", "mobile", "card_present")


def make_reference_transactions(size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": np.round(rng.lognormal(3.3, 1.0, size), 2),
            "hour": rng.integers(0, 24, size),
            "device_change": rng.binomial(1, 0.08, size),
            "beneficiary_change": rng.binomial(1, 0.05, size),
            "velocity_24h": rng.poisson(2.0, size),
            "channel": rng.choice(CHANNELS, size, p=[0.35, 0.5, 0.15]),
            "is_fraud": 0,
        }
    )


def generate_attacks(specification: AttackSpecification, size: int, round_id: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    family = specification.attack_family
    data = pd.DataFrame(
        {
            "amount": np.round(rng.lognormal(4.1, 0.65, size), 2),
            "hour": rng.integers(0, 24, size),
            "device_change": rng.binomial(1, 0.42, size),
            "beneficiary_change": rng.binomial(1, 0.38, size),
            "velocity_24h": rng.poisson(5.0, size),
            "channel": rng.choice(CHANNELS, size),
            "is_fraud": 1,
        }
    )
    if "low_and_slow" in family:
        data["amount"] = np.round(rng.lognormal(2.7, 0.35, size), 2)
        data["velocity_24h"] = rng.poisson(2.0, size)
    if "trusted_device" in family:
        data["device_change"] = rng.binomial(1, 0.04, size)
    if "merchant_abuse" in family:
        data["beneficiary_change"] = rng.binomial(1, 0.02, size)
        data["velocity_24h"] = rng.poisson(7.0, size)
    if "cross_channel" in family:
        data["channel"] = rng.choice(CHANNELS, size)
        data["device_change"] = rng.binomial(1, 0.55, size)
    if "beneficiary_manipulation" in family:
        data["beneficiary_change"] = rng.binomial(1, 0.75, size)
    if "account_takeover" in family:
        data["device_change"] = rng.binomial(1, 0.65, size)
        data["velocity_24h"] = rng.poisson(6.0, size)
    data["attack_id"] = specification.attack_id
    data["attack_family"] = family
    data["generation_round"] = round_id
    data["generation_method"] = "deterministic_conditional_baseline"
    return data


def to_contracts(data: pd.DataFrame) -> list[GeneratedTransaction]:
    return [GeneratedTransaction(**row) for row in data.to_dict(orient="records")]


def evaluate_fidelity(reference: pd.DataFrame, attacks: pd.DataFrame) -> dict:
    mean_delta = abs(float(reference["amount"].mean()) - float(attacks["amount"].mean()))
    std_delta = abs(float(reference["amount"].std()) - float(attacks["amount"].std()))
    reference_rates = reference[["device_change", "beneficiary_change", "velocity_24h"]].mean()
    attack_rates = attacks[["device_change", "beneficiary_change", "velocity_24h"]].mean()
    behaviour_delta = float((reference_rates - attack_rates).abs().mean())
    channel_coverage = float(attacks["channel"].nunique() / len(CHANNELS))
    plausibility = max(0.0, 1.0 - min(1.0, (mean_delta / max(reference["amount"].mean(), 1.0)) * 0.25 + (std_delta / max(reference["amount"].std(), 1.0)) * 0.25 + behaviour_delta * 0.25 + (1.0 - channel_coverage) * 0.25))
    return {
        "amount_mean_delta": round(mean_delta, 4),
        "amount_std_delta": round(std_delta, 4),
        "fraud_rate": round(float(attacks["is_fraud"].mean()), 4),
        "behavioural_signal_delta": round(behaviour_delta, 4),
        "channel_coverage": round(channel_coverage, 4),
        "behavioural_plausibility": round(plausibility, 4),
        "passed": bool(plausibility >= 0.35),
        "notes": ["Synthetic-only reference and attack data; compare with permitted reference data before making realism claims."],
    }


def evaluate_diversity(attacks: pd.DataFrame) -> dict:
    numeric = attacks[["amount", "hour", "device_change", "beneficiary_change", "velocity_24h"]]
    return {
        "attack_family_count": int(attacks["attack_family"].nunique()),
        "channel_count": int(attacks["channel"].nunique()),
        "unique_row_ratio": round(float(attacks.drop_duplicates().shape[0] / max(len(attacks), 1)), 4),
        "numeric_feature_mean_count": int(numeric.nunique().mean()),
    }


def evaluate_novelty(hypothesis: AttackHypothesis, prior_memory: list[str]) -> dict:
    current_terms = set((hypothesis.attack_family + " " + hypothesis.behavioural_mechanism + " " + hypothesis.research_direction).lower().split())
    prior_terms = [set(item.lower().split()) for item in prior_memory if item.strip()]
    similarities = [len(current_terms & terms) / max(len(current_terms | terms), 1) for terms in prior_terms]
    max_similarity = max(similarities, default=0.0)
    return {
        "novelty_score": round(1.0 - max_similarity, 4),
        "max_prior_similarity": round(max_similarity, 4),
        "comparison_count": len(prior_terms),
        "novelty_basis": "token-level structured hypothesis distance; internal heuristic",
    }
