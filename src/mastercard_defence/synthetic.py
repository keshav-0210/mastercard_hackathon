from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import AttackHypothesis, AttackSpecification, GeneratedTransaction

CHANNELS = ("web", "mobile", "card_present")
ALLOWED_FAMILIES = (
    "account_takeover",
    "trusted_device",
    "beneficiary_manipulation",
    "low_and_slow",
    "social_engineering",
    "merchant_abuse",
    "cross_channel_anomaly",
)


def build_round_family_plan(rounds: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    families = list(ALLOWED_FAMILIES)
    rng.shuffle(families)
    plan: list[str] = []
    while len(plan) < rounds:
        for family in families:
            if len(plan) >= rounds:
                break
            if family not in plan:
                plan.append(family)
    return plan


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


def _family_profile(family: str) -> dict[str, float | tuple[float, ...]]:
    family = family.lower()
    defaults = {
        "amount_mu": 3.9,
        "amount_sigma": 0.5,
        "device_prob": 0.32,
        "beneficiary_prob": 0.35,
        "velocity_mean": 4.5,
        "channel_p": (0.35, 0.45, 0.20),
    }
    if "low_and_slow" in family:
        return {
            "amount_mu": 3.0,
            "amount_sigma": 0.3,
            "device_prob": 0.12,
            "beneficiary_prob": 0.18,
            "velocity_mean": 1.8,
            "channel_p": (0.4, 0.4, 0.2),
        }
    if "trusted_device" in family:
        return {
            "amount_mu": 3.7,
            "amount_sigma": 0.45,
            "device_prob": 0.08,
            "beneficiary_prob": 0.20,
            "velocity_mean": 4.0,
            "channel_p": (0.25, 0.55, 0.20),
        }
    if "merchant_abuse" in family:
        return {
            "amount_mu": 4.2,
            "amount_sigma": 0.55,
            "device_prob": 0.22,
            "beneficiary_prob": 0.14,
            "velocity_mean": 6.5,
            "channel_p": (0.15, 0.35, 0.50),
        }
    if "cross_channel" in family:
        return {
            "amount_mu": 4.1,
            "amount_sigma": 0.6,
            "device_prob": 0.46,
            "beneficiary_prob": 0.43,
            "velocity_mean": 5.5,
            "channel_p": (0.34, 0.33, 0.33),
        }
    if "beneficiary_manipulation" in family:
        return {
            "amount_mu": 4.0,
            "amount_sigma": 0.5,
            "device_prob": 0.24,
            "beneficiary_prob": 0.72,
            "velocity_mean": 4.8,
            "channel_p": (0.28, 0.52, 0.20),
        }
    if "account_takeover" in family:
        return {
            "amount_mu": 4.3,
            "amount_sigma": 0.6,
            "device_prob": 0.62,
            "beneficiary_prob": 0.45,
            "velocity_mean": 5.8,
            "channel_p": (0.20, 0.60, 0.20),
        }
    if "social_engineering" in family:
        return {
            "amount_mu": 4.0,
            "amount_sigma": 0.45,
            "device_prob": 0.32,
            "beneficiary_prob": 0.58,
            "velocity_mean": 4.6,
            "channel_p": (0.30, 0.50, 0.20),
        }
    return defaults


def generate_attacks(specification: AttackSpecification, size: int, round_id: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    family = specification.attack_family.lower()
    profile = _family_profile(family)
    channel_probs = profile["channel_p"]
    data = pd.DataFrame(
        {
            "amount": np.clip(np.round(rng.lognormal(float(profile["amount_mu"]), float(profile["amount_sigma"]), size), 2), 20.0, 4000.0),
            "hour": rng.integers(0, 24, size),
            "device_change": rng.binomial(1, float(profile["device_prob"]), size),
            "beneficiary_change": rng.binomial(1, float(profile["beneficiary_prob"]), size),
            "velocity_24h": rng.poisson(float(profile["velocity_mean"]), size),
            "channel": rng.choice(CHANNELS, size, p=channel_probs),
            "is_fraud": 1,
        }
    )
    if "social_engineering" in family:
        data["hour"] = rng.choice(np.array([18, 19, 20, 21, 22, 23]), size=size, replace=True)
        data["amount"] = np.clip(np.round(rng.lognormal(3.9, 0.42, size), 2), 25.0, 2500.0)
    if "low_and_slow" in family:
        data["amount"] = np.clip(np.round(rng.lognormal(2.8, 0.35, size), 2), 15.0, 1500.0)
        data["velocity_24h"] = rng.poisson(1.8, size)
    if "trusted_device" in family:
        data["device_change"] = rng.binomial(1, 0.08, size)
    if "merchant_abuse" in family:
        data["beneficiary_change"] = rng.binomial(1, 0.12, size)
        data["velocity_24h"] = rng.poisson(6.6, size)
    if "cross_channel" in family:
        data["channel"] = rng.choice(CHANNELS, size, p=[0.34, 0.33, 0.33])
        data["device_change"] = rng.binomial(1, 0.42, size)
    if "beneficiary_manipulation" in family:
        data["beneficiary_change"] = rng.binomial(1, 0.68, size)
    if "account_takeover" in family:
        data["device_change"] = rng.binomial(1, 0.58, size)
        data["velocity_24h"] = rng.poisson(5.8, size)
    data["attack_id"] = specification.attack_id
    data["attack_family"] = specification.attack_family
    data["generation_round"] = round_id
    data["generation_method"] = "realistic_family_conditional_generator"
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
    channel_counts = attacks["channel"].value_counts(normalize=True)
    entropy = 0.0
    if not channel_counts.empty:
        entropy = -sum((p * np.log2(p)) for p in channel_counts if p > 0)
    family_count = max(attacks["attack_family"].nunique(), 1)
    return {
        "attack_family_count": int(attacks["attack_family"].nunique()),
        "channel_count": int(attacks["channel"].nunique()),
        "unique_row_ratio": round(float(attacks.drop_duplicates().shape[0] / max(len(attacks), 1)), 4),
        "numeric_feature_mean_count": int(numeric.nunique().mean()),
        "family_coverage_ratio": round(float(family_count / len(ALLOWED_FAMILIES)), 4),
        "channel_entropy": round(float(entropy / max(np.log2(len(CHANNELS)), 1e-9)), 4),
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


def summarize_robustness(results: list[dict]) -> dict:
    metrics = {
        "f1": [],
        "recall": [],
        "precision": [],
        "roc_auc": [],
        "novelty_score": [],
        "behavioural_plausibility": [],
    }
    for result in results:
        detection = result.get("detection", {})
        fidelity = result.get("fidelity", {})
        novelty = result.get("novelty", {})
        metrics["f1"].append(float(detection.get("f1", 0.0)))
        metrics["recall"].append(float(detection.get("recall", 0.0)))
        metrics["precision"].append(float(detection.get("precision", 0.0)))
        metrics["roc_auc"].append(float(detection.get("roc_auc", 0.0)))
        metrics["novelty_score"].append(float(novelty.get("novelty_score", 0.0)))
        metrics["behavioural_plausibility"].append(float(fidelity.get("behavioural_plausibility", 0.0)))

    summary: dict[str, dict[str, float]] = {}
    for name, values in metrics.items():
        if not values:
            summary[name] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            continue
        arr = np.asarray(values, dtype=float)
        summary[name] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=0)), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
        }
    return summary
