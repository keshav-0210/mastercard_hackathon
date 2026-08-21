from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .agents import HeuristicAgents, QwenAgents
from .contracts import MemoryRecord
from .detector import FraudDetector
from .memory import AttackMemory
from .rag import LocalKnowledgeBase
from .synthetic import evaluate_diversity, evaluate_fidelity, generate_attacks, make_reference_transactions


class ClosedLoop:
    def __init__(self, config: dict, agents=None) -> None:
        self.config = config
        if agents is not None:
            self.agents = agents
        elif os.getenv("RUN_MODE", "LOCAL").upper() == "KAGGLE_GPU":
            self.agents = QwenAgents(config)
        else:
            self.agents = HeuristicAgents()
        self.knowledge = LocalKnowledgeBase(config["paths"]["knowledge_base"])
        self.memory = AttackMemory(config["paths"]["memory_db"])

    def run(self, rounds: int | None = None) -> list[dict]:
        rounds = rounds or self.config["pipeline"]["rounds"]
        seed = self.config["seed"]
        reference = make_reference_transactions(self.config["pipeline"]["synthetic_transactions"], seed)
        holdout = make_reference_transactions(max(80, self.config["pipeline"]["synthetic_transactions"] // 4), seed + 1)
        train_reference = reference.sample(frac=0.8, random_state=seed)
        fidelity_reference = reference.drop(train_reference.index).reset_index(drop=True)
        train_reference = train_reference.reset_index(drop=True)
        results = []
        for round_id in range(1, rounds + 1):
            memory_context = self.memory.recent_context()
            prior_direction = " ".join(memory_context[-4:])
            query = "payment fraud attack detector weakness new direction public evidence"
            if prior_direction:
                query += " " + prior_direction[:500]
            evidence = self.knowledge.retrieve(query, self.config["pipeline"]["rag_top_k"])
            hypothesis = self.agents.research(round_id, evidence, memory_context, query)
            specification = self.agents.specify(hypothesis)
            attack_count = self.config["pipeline"]["max_generated_attacks"]
            train_attacks = generate_attacks(specification, attack_count, round_id, seed + round_id)
            unseen_attacks = generate_attacks(specification, attack_count, round_id, seed + 1000 + round_id)
            train_size = int(attack_count * self.config["pipeline"].get("detector_train_fraction", 0.6))
            detector_attacks = train_attacks.iloc[:train_size].copy()
            fidelity = evaluate_fidelity(fidelity_reference, unseen_attacks)
            diversity = evaluate_diversity(unseen_attacks)
            training = pd.concat([train_reference, detector_attacks], ignore_index=True)
            detector = FraudDetector()
            detector.fit(training)
            evaluation = detector.evaluate(pd.concat([holdout.assign(is_fraud=0), unseen_attacks], ignore_index=True))
            evaluation["evaluation_protocol"] = "unseen_attack_rows_and_legitimate_holdout"
            evaluation["train_attack_rows"] = len(detector_attacks)
            evaluation["unseen_attack_rows"] = len(unseen_attacks)
            weakness = self.agents.analyze(round_id, evaluation, fidelity)
            self.memory.add(MemoryRecord(round_id=round_id, record_type="hypothesis", content=hypothesis.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="specification", content=specification.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="evaluation", content={"detection": evaluation, "fidelity": fidelity, "diversity": diversity}))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="weakness", content=weakness.model_dump()))
            results.append({"round": round_id, "research_query": query, "hypothesis": hypothesis, "specification": specification, "fidelity": fidelity, "diversity": diversity, "detection": evaluation, "weakness": weakness})
        return results

    def close(self) -> None:
        self.memory.close()


def load_config(path: str = "config/default.yaml") -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configuration.") from exc
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
