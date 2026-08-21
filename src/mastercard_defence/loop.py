from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .agents import HeuristicAgents, QwenAgents
from .contracts import MemoryRecord
from .detector import FraudDetector
from .memory import AttackMemory
from .rag import LocalKnowledgeBase
from .synthetic import evaluate_fidelity, generate_attacks, make_reference_transactions


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
        results = []
        for round_id in range(1, rounds + 1):
            query = "payment fraud attack detector weakness device beneficiary velocity"
            evidence = self.knowledge.retrieve(query, self.config["pipeline"]["rag_top_k"])
            hypothesis = self.agents.research(round_id, evidence, self.memory.recent_context())
            specification = self.agents.specify(hypothesis)
            attacks = generate_attacks(specification, self.config["pipeline"]["max_generated_attacks"], round_id, seed + round_id)
            fidelity = evaluate_fidelity(reference, attacks)
            training = pd.concat([reference, attacks], ignore_index=True)
            detector = FraudDetector()
            detector.fit(training)
            evaluation = detector.evaluate(pd.concat([holdout.assign(is_fraud=0), attacks], ignore_index=True))
            weakness = self.agents.analyze(round_id, evaluation, fidelity)
            self.memory.add(MemoryRecord(round_id=round_id, record_type="hypothesis", content=hypothesis.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="specification", content=specification.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="evaluation", content={"detection": evaluation, "fidelity": fidelity}))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="weakness", content=weakness.model_dump()))
            results.append({"round": round_id, "hypothesis": hypothesis, "specification": specification, "fidelity": fidelity, "detection": evaluation, "weakness": weakness})
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
