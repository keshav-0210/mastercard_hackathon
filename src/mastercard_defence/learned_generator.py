from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

from .contracts import AttackSpecification
from .synthetic import ALLOWED_FAMILIES, CHANNELS, generate_attacks

MODEL_COLUMNS = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "attack_family"]
DISCRETE_COLUMNS = ["hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "attack_family"]


class ConditionalCTGANGenerator:
    """CTGAN wrapper that samples rows conditioned on the requested attack family."""

    def __init__(self, seed: int = 0, epochs: int = 20) -> None:
        self.seed = seed
        self.epochs = epochs
        self.model: CTGAN | None = None

    def fit(self, training_data: pd.DataFrame) -> None:
        missing = set(MODEL_COLUMNS) - set(training_data.columns)
        if missing:
            raise ValueError(f"CTGAN training data is missing columns: {sorted(missing)}")
        self.model = CTGAN(
            batch_size=max(10, min(100, len(training_data))),
            epochs=self.epochs,
            pac=10,
            verbose=False,
            cuda=torch.cuda.is_available(),
        )
        self.model.set_random_state(self.seed)
        self.model.fit(training_data[MODEL_COLUMNS], discrete_columns=DISCRETE_COLUMNS)

    def generate(self, specification: AttackSpecification, size: int, round_id: int, seed: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("ConditionalCTGANGenerator must be fitted before generation")
        self.model.set_random_state(seed)
        conditioned_batches = []
        remaining = size
        for attempt in range(8):
            batch_size = max(remaining * 3, 30)
            candidate = self.model.sample(batch_size, condition_column="attack_family", condition_value=specification.attack_family)
            matched = candidate[candidate["attack_family"] == specification.attack_family]
            if not matched.empty:
                conditioned_batches.append(matched)
                remaining -= len(matched)
            if remaining <= 0:
                break
            self.model.set_random_state(seed + attempt + 1)
        if remaining > 0:
            matched_count = sum(len(batch) for batch in conditioned_batches)
            raise RuntimeError(
                f"CTGAN produced only {matched_count} rows for attack family "
                f"{specification.attack_family!r}; refusing to relabel mixed-family samples."
            )
        data = pd.concat(conditioned_batches, ignore_index=True).iloc[:size].copy()
        data["is_fraud"] = 1
        data["amount"] = np.clip(pd.to_numeric(data["amount"], errors="coerce").fillna(100.0), 20.0, 4000.0).round(2)
        for column in ("hour", "device_change", "beneficiary_change", "velocity_24h"):
            data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).round().astype(int)
        data["hour"] = data["hour"].clip(0, 23)
        data["device_change"] = data["device_change"].clip(0, 1)
        data["beneficiary_change"] = data["beneficiary_change"].clip(0, 1)
        data["velocity_24h"] = data["velocity_24h"].clip(0, 50)
        data["channel"] = data["channel"].where(data["channel"].isin(CHANNELS), CHANNELS[0])
        data["attack_id"] = specification.attack_id
        data["attack_family"] = specification.attack_family
        data["generation_round"] = round_id
        data["generation_method"] = "conditional_ctgan"
        return data


def build_training_corpus(seed: int, attack_size: int = 100, reference_size: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame({
        "amount": np.round(rng.lognormal(3.3, 1.0, reference_size), 2),
        "hour": rng.integers(0, 24, reference_size),
        "device_change": rng.binomial(1, 0.08, reference_size),
        "beneficiary_change": rng.binomial(1, 0.05, reference_size),
        "velocity_24h": rng.poisson(2.0, reference_size),
        "channel": rng.choice(CHANNELS, reference_size, p=[0.35, 0.5, 0.15]),
        "attack_family": "legitimate",
    })
    attack_rows = []
    for index, family in enumerate(ALLOWED_FAMILIES):
        specification = AttackSpecification(
            attack_id=f"training-{family}", attack_family=family,
            scenario="synthetic training scenario", target_context="synthetic payment security",
            temporal_pattern="mixed", amount_pattern="family conditional", device_pattern="family conditional",
            beneficiary_pattern="family conditional", evasion_objective="synthetic evaluation",
        )
        rows = generate_attacks(specification, attack_size, 0, seed + index + 1)
        attack_rows.append(rows[MODEL_COLUMNS])
    attacks = pd.concat(attack_rows, ignore_index=True)
    return pd.concat([reference, attacks], ignore_index=True)[MODEL_COLUMNS]
