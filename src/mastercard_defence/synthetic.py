from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AttackSpecification, GeneratedTransaction

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
    plausibility = max(0.0, 1.0 - min(1.0, (mean_delta / max(reference["amount"].mean(), 1.0)) * 0.5 + std_delta / max(reference["amount"].std(), 1.0) * 0.5))
    return {
        "amount_mean_delta": round(mean_delta, 4),
        "amount_std_delta": round(std_delta, 4),
        "fraud_rate": round(float(attacks["is_fraud"].mean()), 4),
        "behavioural_plausibility": round(plausibility, 4),
        "passed": bool(plausibility >= 0.35),
        "notes": ["Synthetic-only reference and attack data; no production data used."],
    }
