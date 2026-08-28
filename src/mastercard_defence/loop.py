from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import ATTACK_FAMILIES, HeuristicAgents, QwenAgents
from .contracts import RoundRecord
from .detector import FraudDetector
from .memory import AttackMemory
from .rag import LocalKnowledgeBase
from .synthetic import build_round_family_plan, calibrate_attacks_toward_reference, evaluate_diversity, evaluate_fidelity, evaluate_novelty, fraud_rate_for, generate_attacks, legitimate_rows_for_fraud_rate, make_reference_transactions, summarize_detector_version_performance, summarize_family_performance, summarize_robustness


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
        self.replay_buffer = pd.DataFrame()
        self.detector_version = 1
        self.detector_retrain_every = int(self.config.get("detector_retrain_every", 3))
        self.max_replay_examples = int(self.config.get("max_replay_examples", 200))
        # Cross-round state: lets us measure cumulative family coverage, redundancy against
        # prior rounds' attacks, and whether the detector still catches old attack patterns.
        self.historical_unseen_pool = pd.DataFrame()
        self.historical_signatures: set[tuple] = set()
        self.explored_families: set[str] = set()
        self.max_historical_pool = int(self.config.get("max_historical_pool", 300))
        # How many times each family has already been generated this run; drives both the
        # fidelity-calibration blend and the detector cold-start training ramp.
        self.family_exposure_count: dict[str, int] = {}

    @staticmethod
    def _stratified_cap(combined: pd.DataFrame, cap: int) -> pd.DataFrame:
        """Trims a pool to `cap` rows by sampling evenly across families instead of dropping the
        oldest rows first, so earlier rounds' patterns are not forgotten just because they age out."""
        if len(combined) <= cap:
            return combined
        families = combined["attack_family"].fillna("unknown")
        distinct = max(families.nunique(), 1)
        per_family_cap = max(1, cap // distinct)
        rng = np.random.default_rng(len(combined))
        parts = []
        for _, group in combined.groupby(families):
            if len(group) > per_family_cap:
                parts.append(group.sample(n=per_family_cap, random_state=int(rng.integers(0, 1_000_000))))
            else:
                parts.append(group)
        trimmed = pd.concat(parts, ignore_index=True)
        if len(trimmed) > cap:
            trimmed = trimmed.sample(n=cap, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        return trimmed.reset_index(drop=True)

    def _record_hard_examples(self, new_examples: pd.DataFrame | None) -> None:
        if new_examples is None or new_examples.empty:
            return
        columns = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "is_fraud", "attack_family"]
        for column in columns:
            if column not in new_examples.columns:
                new_examples = new_examples.copy()
                new_examples[column] = 0 if column != "channel" else "web"
                if column == "is_fraud":
                    new_examples[column] = 1
        combined = pd.concat([self.replay_buffer, new_examples], ignore_index=True)
        combined = combined[columns].drop_duplicates().reset_index(drop=True)
        self.replay_buffer = self._stratified_cap(combined, self.max_replay_examples)

    def _maybe_retrain_detector(self, round_id: int) -> bool:
        if self.replay_buffer.empty:
            return False
        if round_id % self.detector_retrain_every == 0:
            self.detector_version += 1
            return True
        return False

    def _cross_round_redundancy(self, attacks: pd.DataFrame) -> tuple[dict, set]:
        """Compares this round's generated rows against every prior round's rows to catch
        near-duplicate (redundant) attack variants rather than only within-round uniqueness."""
        signature_columns = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel"]
        signatures = {
            (round(float(row["amount"])), int(row["hour"]), int(row["device_change"]), int(row["beneficiary_change"]), int(row["velocity_24h"]), str(row["channel"]))
            for row in attacks[signature_columns].to_dict(orient="records")
        }
        overlap = len(signatures & self.historical_signatures)
        total = max(len(signatures), 1)
        redundancy_ratio = round(overlap / total, 4)
        return {
            "cross_round_redundancy_ratio": redundancy_ratio,
            "cross_round_unique_ratio": round(1.0 - redundancy_ratio, 4),
        }, signatures

    def _update_historical_pool(self, new_fraud_rows: pd.DataFrame | None) -> None:
        if new_fraud_rows is None or new_fraud_rows.empty:
            return
        columns = ["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "is_fraud", "attack_family"]
        available = [column for column in columns if column in new_fraud_rows.columns]
        combined = pd.concat([self.historical_unseen_pool, new_fraud_rows[available]], ignore_index=True)
        combined = combined.drop_duplicates().reset_index(drop=True)
        self.historical_unseen_pool = self._stratified_cap(combined, self.max_historical_pool)

    def _evaluate_historical_robustness(self, detector: FraudDetector, seed: int, round_id: int) -> dict:
        """Checks whether the just-fitted detector still catches attacks from earlier rounds,
        so detector-version improvement and non-forgetting can be measured, not assumed."""
        if self.historical_unseen_pool.empty:
            return {"insufficient_history": True, "historical_fraud_rows": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.5}
        historical_legit = make_reference_transactions(len(self.historical_unseen_pool), seed + 9000 + round_id)
        historical_validation = pd.concat([historical_legit.assign(is_fraud=0), self.historical_unseen_pool], ignore_index=True)
        historical_validation["attack_family"] = historical_validation["attack_family"].fillna("legitimate")
        result = detector.evaluate(historical_validation)
        return {
            "insufficient_history": False,
            "historical_fraud_rows": int(len(self.historical_unseen_pool)),
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "roc_auc": result["roc_auc"],
        }

    def run(self, rounds: int | None = None, family_plan: list[str] | None = None, seed: int | None = None, attack_generator=None, detector_mode: str = "static", hard_examples: pd.DataFrame | None = None) -> list[dict]:
        rounds = rounds or self.config["pipeline"]["rounds"]
        seed = self.config["seed"] if seed is None else seed
        family_plan = family_plan or build_round_family_plan(rounds, seed)
        target_fraud_rate = float(self.config["pipeline"].get("fraud_rate", 0.02))
        if not 0.01 <= target_fraud_rate <= 0.03:
            raise ValueError("pipeline.fraud_rate must be between 0.01 and 0.03")
        attack_count = self.config["pipeline"]["max_generated_attacks"]
        train_attack_count = int(attack_count * self.config["pipeline"].get("detector_train_fraction", 0.6))
        reference = make_reference_transactions(self.config["pipeline"]["synthetic_transactions"], seed)
        train_reference = make_reference_transactions(legitimate_rows_for_fraud_rate(train_attack_count, target_fraud_rate), seed + 2)
        holdout = make_reference_transactions(legitimate_rows_for_fraud_rate(attack_count, target_fraud_rate), seed + 1)
        fidelity_reference = reference.reset_index(drop=True)
        results = []
        print(f"[seed={seed}] starting {rounds}-round run with family plan: {family_plan}")
        previous_weakness = None
        weakness_history = []
        decisions = []
        sampler = np.random.default_rng(seed)
        for round_id in range(1, rounds + 1):
            print(f"[seed={seed}] round {round_id}/{rounds} starting")
            memory_context = self.memory.recent_context(seed=seed)
            prior_direction = " ".join(memory_context[-4:])
            prior_families = tuple(family for family in ATTACK_FAMILIES if family in " ".join(memory_context).lower())
            fallback_family = family_plan[min(round_id - 1, len(family_plan) - 1)]
            if previous_weakness is None:
                chosen_family = fallback_family
                recommendation = {"source": "seeded_plan", "family": chosen_family, "reason": "Initial family selected from the reproducible seeded plan."}
            else:
                candidates = self._adaptive_candidates(previous_weakness, weakness_history)
                agent_start = time.perf_counter()
                family_recommendation = self.agents.recommend_family(previous_weakness, candidates, memory_context)
                print(f"[seed={seed}] round {round_id} Agent1.recommend_family complete in {time.perf_counter() - agent_start:.2f}s")
                weights = self._family_sampling_weights(candidates, previous_weakness, family_recommendation.recommended_family)
                chosen_family = str(sampler.choice(candidates, p=np.asarray(weights) / sum(weights)))
                recommendation = {
                    "source": "agent_1_adaptive_recommendation",
                    **family_recommendation.model_dump(),
                    "candidate_families": list(candidates),
                    "sampling_weights": dict(zip(candidates, weights)),
                    "sampled_family": chosen_family,
                }
            # Ground retrieval in the chosen family so Agent 1 receives family-specific evidence, not generic background.
            family_terms = chosen_family.replace("_", " ")
            query = f"{family_terms} {chosen_family} payment fraud attack detector weakness new direction public evidence"
            if prior_direction:
                query += " " + prior_direction[:500]
            evidence = self.knowledge.retrieve_for_family(chosen_family, query, self.config["pipeline"]["rag_top_k"])
            decision_memory = self.memory.relevant_context(seed=seed, fraud_family=chosen_family, limit=12)
            memory_context = decision_memory or memory_context
            decisions.append({"round": round_id, "family": chosen_family, "recommendation": recommendation})
            allowed_families = (chosen_family,) if chosen_family not in prior_families else tuple(family for family in ATTACK_FAMILIES if family not in prior_families) or ATTACK_FAMILIES
            agent_start = time.perf_counter()
            hypothesis = self.agents.research(round_id, evidence, memory_context, query, allowed_families)
            print(f"[seed={seed}] round {round_id} Agent1.research complete in {time.perf_counter() - agent_start:.2f}s")
            # allowed_families may be broader than chosen_family, so the agent's own attack_id can drift; keep it consistent.
            hypothesis.attack_family = chosen_family
            hypothesis.attack_id = f"round-{round_id}-{chosen_family}"
            novelty = evaluate_novelty(hypothesis, memory_context, round_id)
            agent_start = time.perf_counter()
            specification = self.agents.specify(hypothesis)
            print(f"[seed={seed}] round {round_id} Agent2.specify complete in {time.perf_counter() - agent_start:.2f}s")
            specification.attack_family = chosen_family
            generation_start = time.perf_counter()
            if attack_generator is None:
                train_attacks = generate_attacks(specification, attack_count, round_id, seed + round_id)
                unseen_attacks = generate_attacks(specification, attack_count, round_id, seed + 1000 + round_id)
            else:
                train_attacks = attack_generator.generate(specification, attack_count, round_id, seed + round_id)
                unseen_attacks = attack_generator.generate(specification, attack_count, round_id, seed + 1000 + round_id)
            print(f"[seed={seed}] round {round_id} train/unseen generation complete in {time.perf_counter() - generation_start:.2f}s")
            exposure_count = self.family_exposure_count.get(chosen_family, 0)
            train_attacks = calibrate_attacks_toward_reference(train_attacks, fidelity_reference, exposure_count)
            unseen_attacks = calibrate_attacks_toward_reference(unseen_attacks, fidelity_reference, exposure_count)
            self.family_exposure_count[chosen_family] = exposure_count + 1
            # Note: the current round's own attack batch is always used at full size so that a
            # family's first-ever exposure is never starved of its own training signal. The
            # genuine cold-start effect comes from the replay/historical buffers starting empty
            # and growing round over round (see continual training below), not from shrinking
            # this round's fresh sample.
            base_train_size = int(attack_count * self.config["pipeline"].get("detector_train_fraction", 0.6))
            detector_attacks = train_attacks.iloc[:base_train_size].copy()
            fidelity = evaluate_fidelity(fidelity_reference, unseen_attacks)
            diversity = evaluate_diversity(unseen_attacks)
            redundancy_metrics, new_signatures = self._cross_round_redundancy(unseen_attacks)
            diversity.update(redundancy_metrics)
            self.explored_families.add(chosen_family)
            diversity["cumulative_families_explored"] = len(self.explored_families)
            diversity["cumulative_family_coverage_ratio"] = round(len(self.explored_families) / len(ATTACK_FAMILIES), 4)
            training_parts = [train_reference, detector_attacks]
            if detector_mode == "continual":
                replay_examples = self.replay_buffer.copy()
                if hard_examples is not None and not hard_examples.empty:
                    replay_examples = pd.concat([replay_examples, hard_examples], ignore_index=True)
                if not replay_examples.empty:
                    training_parts.append(replay_examples)
            training = pd.concat(training_parts, ignore_index=True)
            fraud_rows = int(training["is_fraud"].sum())
            legitimate_rows = len(training) - fraud_rows
            required_legitimate_rows = legitimate_rows_for_fraud_rate(fraud_rows, target_fraud_rate)
            if legitimate_rows < required_legitimate_rows:
                additional_legitimate = make_reference_transactions(
                    required_legitimate_rows - legitimate_rows,
                    seed + 3000 + round_id,
                )
                training = pd.concat([training, additional_legitimate], ignore_index=True)
            training_fraud_rate = fraud_rate_for(int(training["is_fraud"].sum()), len(training) - int(training["is_fraud"].sum()))
            detector = FraudDetector()
            detector.fit(training)
            validation_data = pd.concat([holdout.assign(is_fraud=0), unseen_attacks], ignore_index=True)
            validation_data["attack_family"] = validation_data.get("attack_family", specification.attack_family)
            validation_fraud_rate = fraud_rate_for(int(validation_data["is_fraud"].sum()), len(validation_data) - int(validation_data["is_fraud"].sum()))
            evaluation = detector.evaluate(validation_data)
            evaluation["evaluation_protocol"] = "unseen_attack_rows_and_legitimate_holdout"
            evaluation["train_attack_rows"] = len(detector_attacks)
            evaluation["unseen_attack_rows"] = len(unseen_attacks)
            evaluation["train_reference_rows"] = len(train_reference)
            evaluation["validation_legitimate_rows"] = len(holdout)
            evaluation["training_fraud_rate"] = round(training_fraud_rate, 4)
            evaluation["validation_fraud_rate"] = round(validation_fraud_rate, 4)
            evaluation["detector_version"] = self.detector_version
            evaluation["historical_robustness"] = self._evaluate_historical_robustness(detector, seed, round_id)
            evaluation["attack_difficulty_score"] = round(1.0 - float(evaluation.get("recall", 0.0)), 4)
            if detector_mode == "continual":
                predictions = detector.predict(unseen_attacks)
                missed = unseen_attacks.loc[predictions == 0].copy()
                if hard_examples is None:
                    hard_examples = pd.DataFrame(columns=["amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "is_fraud", "attack_family"])
                if not missed.empty:
                    hard_examples = missed if hard_examples.empty else pd.concat([hard_examples, missed], ignore_index=True).drop_duplicates()
                self._record_hard_examples(hard_examples)
                self._maybe_retrain_detector(round_id)
            evaluation["hard_examples_replayed"] = int(len(self.replay_buffer)) if detector_mode == "continual" else int(len(hard_examples)) if hard_examples is not None else 0
            agent_start = time.perf_counter()
            weakness = self.agents.analyze(round_id, evaluation, fidelity)
            print(f"[seed={seed}] round {round_id} Agent3.analyze complete in {time.perf_counter() - agent_start:.2f}s")
            previous_weakness = weakness
            weakness_history.append(weakness)
            self._update_historical_pool(unseen_attacks[unseen_attacks["is_fraud"] == 1])
            self.historical_signatures.update(new_signatures)
            generator_metadata = {
                "backend": "ctgan" if attack_generator is not None else "procedural",
                "epochs": self.config.get("generator_epochs") if attack_generator is not None else None,
                "seed": seed + round_id,
            }
            generation_stats = {
                "train_attack_rows": len(detector_attacks),
                "unseen_attack_rows": len(unseen_attacks),
                "train_reference_rows": len(train_reference),
                "validation_legitimate_rows": len(holdout),
                "training_fraud_rate": evaluation["training_fraud_rate"],
                "validation_fraud_rate": evaluation["validation_fraud_rate"],
                "diversity": diversity,
                "novelty": novelty,
            }
            self.memory.add_round(
                RoundRecord(
                    seed=seed,
                    round_id=round_id,
                    attack_id=hypothesis.attack_id,
                    fraud_family=chosen_family,
                    detector_version=f"v{self.detector_version}",
                    attack_hypothesis=hypothesis.model_dump(),
                    attack_specification=specification.model_dump(),
                    generator_metadata=generator_metadata,
                    generation_stats=generation_stats,
                    fidelity_evaluation=fidelity,
                    detector_evaluation=evaluation,
                    hard_sample_summary={"patterns": weakness.hard_sample_patterns},
                    agent3_analysis=weakness.model_dump(),
                    identified_weaknesses=weakness.observed_weaknesses,
                    recommended_next_attack_directions=weakness.recommended_next_attack_directions,
                )
            )
            results.append({"round": round_id, "research_query": query, "hypothesis": hypothesis, "specification": specification, "fidelity": fidelity, "diversity": diversity, "novelty": novelty, "detection": evaluation, "weakness": weakness, "family_decision": recommendation, "detector_mode": detector_mode})
            print(f"[seed={seed}] round {round_id}/{rounds} complete | family={chosen_family} | f1={evaluation.get('f1', 0.0):.4f} | novelty={novelty.get('novelty_score', 0.0):.4f}")
        print(f"[seed={seed}] run complete. Total rounds: {len(results)}")
        return results

    @staticmethod
    def _adaptive_candidates(weakness, history) -> tuple[str, ...]:
        return ATTACK_FAMILIES

    def _family_sampling_weights(self, candidates, weakness, agent_family: str) -> list[float]:
        confidence = float(weakness.confidence)
        weakness_multiplier = float(self.config.get("weakness_weight_multiplier", 8.0))
        affected_attack_families = set(weakness.affected_attack_families)
        weights = []
        for family in candidates:
            parent_match = family in affected_attack_families or any(parent in affected_attack_families for parent in ATTACK_FAMILIES if parent in family)
            weight = 1.0 + (weakness_multiplier * confidence if parent_match else 0.0)
            if family == agent_family:
                weight *= 1.25
            weights.append(round(weight, 4))
        return weights

    def run_robustness_suite(self, seeds: int = 3, rounds: int | None = None) -> dict:
        suite_start = time.perf_counter()
        rounds = rounds or self.config["pipeline"]["rounds"]
        all_runs: list[list[dict]] = []
        attack_generator = None
        if self.config.get("generator_backend", "procedural").lower() == "ctgan":
            from .learned_generator import ConditionalCTGANGenerator, build_training_corpus

            fit_start = time.perf_counter()
            print("[robustness] preparing CTGAN training corpus")
            attack_generator = ConditionalCTGANGenerator(seed=self.config["seed"], epochs=self.config.get("generator_epochs", 20))
            training_corpus = build_training_corpus(
                self.config["seed"],
                attack_size=self.config.get("generator_training_attack_size", 100),
                reference_size=self.config.get("generator_training_reference_size", 400),
            )
            print(f"[robustness] fitting CTGAN: rows={len(training_corpus)} epochs={attack_generator.epochs}")
            attack_generator.fit(training_corpus)
            print(f"[robustness] CTGAN ready in {time.perf_counter() - fit_start:.2f}s; entering round loop")
        print(f"[robustness] starting suite: seeds={seeds}, rounds={rounds}")
        for seed_index in range(seeds):
            seed_start = time.perf_counter()
            run_seed = self.config["seed"] + seed_index
            family_plan = build_round_family_plan(rounds, run_seed)
            print(f"[robustness] starting seed {run_seed} with family plan: {family_plan}")
            if attack_generator is not None:
                attack_generator.model.set_random_state(run_seed)
            run_results = self.run(rounds=rounds, family_plan=family_plan, seed=run_seed, attack_generator=attack_generator, detector_mode=self.config.get("detector_mode", "static"), hard_examples=pd.DataFrame())
            all_runs.append(run_results)
            print(f"[robustness] completed seed {run_seed} with {len(run_results)} rounds in {time.perf_counter() - seed_start:.2f}s")

        flattened = [item for group in all_runs for item in group]
        summary = summarize_robustness(flattened)
        family_analysis = summarize_family_performance(flattened)
        detector_version_analysis = summarize_detector_version_performance(flattened)
        print(f"[robustness] suite complete in {time.perf_counter() - suite_start:.2f}s. Aggregated summary: {summary}")
        return {
            "seed_count": seeds,
            "rounds": rounds,
            "families_per_run": list(build_round_family_plan(rounds, self.config["seed"])),
            "by_seed": [{"seed": self.config["seed"] + i, "results": run_results} for i, run_results in enumerate(all_runs)],
            "summary": summary,
            "family_analysis": family_analysis,
            "detector_version_analysis": detector_version_analysis,
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
