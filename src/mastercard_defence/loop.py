from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .agents import ATTACK_FAMILIES, HeuristicAgents, QwenAgents
from .contracts import MemoryRecord
from .detector import FraudDetector
from .memory import AttackMemory
from .rag import LocalKnowledgeBase
from .synthetic import build_round_family_plan, evaluate_diversity, evaluate_fidelity, evaluate_novelty, generate_attacks, make_reference_transactions, summarize_robustness


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

    def run(self, rounds: int | None = None, family_plan: list[str] | None = None, seed: int | None = None, attack_generator=None, detector_mode: str = "static", hard_examples: pd.DataFrame | None = None) -> list[dict]:
        rounds = rounds or self.config["pipeline"]["rounds"]
        seed = self.config["seed"] if seed is None else seed
        family_plan = family_plan or build_round_family_plan(rounds, seed)
        reference = make_reference_transactions(self.config["pipeline"]["synthetic_transactions"], seed)
        holdout = make_reference_transactions(max(80, self.config["pipeline"]["synthetic_transactions"] // 4), seed + 1)
        train_reference = reference.sample(frac=0.8, random_state=seed)
        fidelity_reference = reference.drop(train_reference.index).reset_index(drop=True)
        train_reference = train_reference.reset_index(drop=True)
        results = []
        print(f"[seed={seed}] starting {rounds}-round run with family plan: {family_plan}")
        previous_weakness = None
        decisions = []
        for round_id in range(1, rounds + 1):
            print(f"[seed={seed}] round {round_id}/{rounds} starting")
            memory_context = self.memory.recent_context()
            prior_direction = " ".join(memory_context[-4:])
            query = "payment fraud attack detector weakness new direction public evidence"
            if prior_direction:
                query += " " + prior_direction[:500]
            evidence = self.knowledge.retrieve(query, self.config["pipeline"]["rag_top_k"])
            prior_families = tuple(family for family in ATTACK_FAMILIES if family in " ".join(memory_context).lower())
            fallback_family = family_plan[min(round_id - 1, len(family_plan) - 1)]
            if previous_weakness is None:
                chosen_family = fallback_family
                recommendation = {"source": "seeded_plan", "family": chosen_family, "reason": "Initial family selected from the reproducible seeded plan."}
            else:
                tested_families = {item["family"] for item in decisions}
                candidates = tuple(family for family in ATTACK_FAMILIES if family not in tested_families) or ATTACK_FAMILIES
                family_recommendation = self.agents.recommend_family(previous_weakness, candidates, memory_context)
                chosen_family = family_recommendation.recommended_family if family_recommendation.recommended_family in candidates else fallback_family
                recommendation = {"source": "agent_1_adaptive_recommendation", **family_recommendation.model_dump()}
            decisions.append({"round": round_id, "family": chosen_family, "recommendation": recommendation})
            allowed_families = (chosen_family,) if chosen_family not in prior_families else tuple(family for family in ATTACK_FAMILIES if family not in prior_families) or ATTACK_FAMILIES
            hypothesis = self.agents.research(round_id, evidence, memory_context, query, allowed_families)
            hypothesis.attack_family = chosen_family
            novelty = evaluate_novelty(hypothesis, memory_context)
            specification = self.agents.specify(hypothesis)
            specification.attack_family = chosen_family
            attack_count = self.config["pipeline"]["max_generated_attacks"]
            if attack_generator is None:
                train_attacks = generate_attacks(specification, attack_count, round_id, seed + round_id)
                unseen_attacks = generate_attacks(specification, attack_count, round_id, seed + 1000 + round_id)
            else:
                train_attacks = attack_generator.generate(specification, attack_count, round_id, seed + round_id)
                unseen_attacks = attack_generator.generate(specification, attack_count, round_id, seed + 1000 + round_id)
            train_size = int(attack_count * self.config["pipeline"].get("detector_train_fraction", 0.6))
            detector_attacks = train_attacks.iloc[:train_size].copy()
            fidelity = evaluate_fidelity(fidelity_reference, unseen_attacks)
            diversity = evaluate_diversity(unseen_attacks)
            training_parts = [train_reference, detector_attacks]
            if detector_mode == "continual" and hard_examples is not None and not hard_examples.empty:
                training_parts.append(hard_examples)
            training = pd.concat(training_parts, ignore_index=True)
            detector = FraudDetector()
            detector.fit(training)
            validation_data = pd.concat([holdout.assign(is_fraud=0), unseen_attacks], ignore_index=True)
            validation_data["attack_family"] = validation_data.get("attack_family", specification.attack_family)
            evaluation = detector.evaluate(validation_data)
            evaluation["evaluation_protocol"] = "unseen_attack_rows_and_legitimate_holdout"
            evaluation["train_attack_rows"] = len(detector_attacks)
            evaluation["unseen_attack_rows"] = len(unseen_attacks)
            evaluation["train_reference_rows"] = len(train_reference)
            evaluation["validation_legitimate_rows"] = len(holdout)
            if detector_mode == "continual" and hard_examples is not None:
                predictions = detector.predict(unseen_attacks)
                missed = unseen_attacks.loc[predictions == 0].copy()
                hard_examples = missed if hard_examples.empty else pd.concat([hard_examples, missed], ignore_index=True).drop_duplicates()
            evaluation["hard_examples_replayed"] = int(len(hard_examples)) if hard_examples is not None else 0
            weakness = self.agents.analyze(round_id, evaluation, fidelity)
            previous_weakness = weakness
            self.memory.add(MemoryRecord(round_id=round_id, record_type="hypothesis", content=hypothesis.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="specification", content=specification.model_dump()))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="evaluation", content={"detection": evaluation, "fidelity": fidelity, "diversity": diversity, "novelty": novelty}))
            self.memory.add(MemoryRecord(round_id=round_id, record_type="weakness", content=weakness.model_dump()))
            results.append({"round": round_id, "research_query": query, "hypothesis": hypothesis, "specification": specification, "fidelity": fidelity, "diversity": diversity, "novelty": novelty, "detection": evaluation, "weakness": weakness, "family_decision": recommendation, "detector_mode": detector_mode})
            print(f"[seed={seed}] round {round_id}/{rounds} complete | family={chosen_family} | f1={evaluation.get('f1', 0.0):.4f} | novelty={novelty.get('novelty_score', 0.0):.4f}")
        print(f"[seed={seed}] run complete. Total rounds: {len(results)}")
        return results

    def run_robustness_suite(self, seeds: int = 3, rounds: int | None = None) -> dict:
        rounds = rounds or self.config["pipeline"]["rounds"]
        all_runs: list[list[dict]] = []
        attack_generator = None
        if self.config.get("generator_backend", "procedural").lower() == "ctgan":
            from .learned_generator import ConditionalCTGANGenerator, build_training_corpus

            attack_generator = ConditionalCTGANGenerator(seed=self.config["seed"], epochs=self.config.get("generator_epochs", 20))
            attack_generator.fit(build_training_corpus(
                self.config["seed"],
                attack_size=self.config.get("generator_training_attack_size", 100),
                reference_size=self.config.get("generator_training_reference_size", 400),
            ))
        print(f"[robustness] starting suite: seeds={seeds}, rounds={rounds}")
        for seed_index in range(seeds):
            run_seed = self.config["seed"] + seed_index
            family_plan = build_round_family_plan(rounds, run_seed)
            print(f"[robustness] starting seed {run_seed} with family plan: {family_plan}")
            if attack_generator is not None:
                attack_generator.model.set_random_state(run_seed)
            run_results = self.run(rounds=rounds, family_plan=family_plan, seed=run_seed, attack_generator=attack_generator, detector_mode=self.config.get("detector_mode", "static"), hard_examples=pd.DataFrame())
            all_runs.append(run_results)
            print(f"[robustness] completed seed {run_seed} with {len(run_results)} rounds")

        flattened = [item for group in all_runs for item in group]
        summary = summarize_robustness(flattened)
        print(f"[robustness] suite complete. Aggregated summary: {summary}")
        return {
            "seed_count": seeds,
            "rounds": rounds,
            "families_per_run": list(build_round_family_plan(rounds, self.config["seed"])),
            "by_seed": [{"seed": self.config["seed"] + i, "results": run_results} for i, run_results in enumerate(all_runs)],
            "summary": summary,
        }

    def close(self) -> None:
        self.memory.close()


def load_config(path: str = "config/default.yaml") -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load configuration.") from exc
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)
