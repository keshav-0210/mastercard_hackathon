from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .contracts import RoundRecord

_JSON_COLUMNS = (
    "attack_hypothesis",
    "attack_specification",
    "generator_metadata",
    "generation_stats",
    "fidelity_evaluation",
    "detector_evaluation",
    "hard_sample_summary",
    "agent3_analysis",
    "identified_weaknesses",
    "recommended_next_attack_directions",
)


class AttackMemory:
    """Structured, append-only round history: the only channel from Agent 3 back to Agent 1 (never to Agent 2)."""

    def __init__(self, database_path: str) -> None:
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rounds (
                seed INTEGER NOT NULL,
                round_id INTEGER NOT NULL,
                attack_id TEXT,
                fraud_family TEXT,
                status TEXT,
                detector_version TEXT,
                attack_hypothesis TEXT,
                attack_specification TEXT,
                generator_metadata TEXT,
                generation_stats TEXT,
                fidelity_evaluation TEXT,
                detector_evaluation TEXT,
                hard_sample_summary TEXT,
                agent3_analysis TEXT,
                identified_weaknesses TEXT,
                recommended_next_attack_directions TEXT,
                created_at TEXT,
                PRIMARY KEY (seed, round_id)
            )
            """
        )
        self.connection.commit()

    def add_round(self, record: RoundRecord) -> None:
        """Insert one round; raises if (seed, round_id) already exists so history is never overwritten."""
        existing = self.connection.execute(
            "SELECT 1 FROM rounds WHERE seed = ? AND round_id = ?", (record.seed, record.round_id)
        ).fetchone()
        if existing is not None:
            raise ValueError(f"round (seed={record.seed}, round_id={record.round_id}) already exists in Attack Memory")
        values = record.model_dump()
        self.connection.execute(
            """
            INSERT INTO rounds (
                seed, round_id, attack_id, fraud_family, status, detector_version,
                attack_hypothesis, attack_specification, generator_metadata, generation_stats,
                fidelity_evaluation, detector_evaluation, hard_sample_summary, agent3_analysis,
                identified_weaknesses, recommended_next_attack_directions, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                values["seed"],
                values["round_id"],
                values["attack_id"],
                values["fraud_family"],
                values["status"],
                values["detector_version"],
                json.dumps(values["attack_hypothesis"]),
                json.dumps(values["attack_specification"]),
                json.dumps(values["generator_metadata"]),
                json.dumps(values["generation_stats"]),
                json.dumps(values["fidelity_evaluation"]),
                json.dumps(values["detector_evaluation"]),
                json.dumps(values["hard_sample_summary"]),
                json.dumps(values["agent3_analysis"]),
                json.dumps(values["identified_weaknesses"]),
                json.dumps(values["recommended_next_attack_directions"]),
                values["created_at"].isoformat(),
            ),
        )
        self.connection.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        for column in _JSON_COLUMNS:
            record[column] = json.loads(record[column])
        return record

    def get_round(self, seed: int, round_id: int) -> dict[str, Any] | None:
        self.connection.row_factory = sqlite3.Row
        row = self.connection.execute(
            "SELECT * FROM rounds WHERE seed = ? AND round_id = ?", (seed, round_id)
        ).fetchone()
        self.connection.row_factory = None
        return self._row_to_dict(row) if row is not None else None

    def get_recent_rounds(self, limit: int = 12, fraud_family: str | None = None, seed: int | None = None) -> list[dict[str, Any]]:
        """Structured retrieval for Agent 1: most-recent-first, optionally scoped to one fraud family and/or seed."""
        clauses = []
        params: list[Any] = []
        if seed is not None:
            clauses.append("seed = ?")
            params.append(seed)
        if fraud_family:
            clauses.append("fraud_family = ?")
            params.append(fraud_family)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        self.connection.row_factory = sqlite3.Row
        rows = self.connection.execute(f"SELECT * FROM rounds {where} ORDER BY round_id DESC LIMIT ?", params).fetchall()
        self.connection.row_factory = None
        return [self._row_to_dict(row) for row in rows]

    def get_weakness_history(self, limit: int = 12, seed: int | None = None) -> list[dict[str, Any]]:
        """Only Agent 3's findings + recommended directions, most recent first (feeds Agent 1, never Agent 2)."""
        records = self.get_recent_rounds(limit=limit, seed=seed)
        return [
            {
                "round_id": record["round_id"],
                "fraud_family": record["fraud_family"],
                "detector_version": record["detector_version"],
                "agent3_analysis": record["agent3_analysis"],
                "identified_weaknesses": record["identified_weaknesses"],
                "recommended_next_attack_directions": record["recommended_next_attack_directions"],
            }
            for record in records
        ]

    def get_explored_families(self, seed: int | None = None) -> dict[str, int]:
        """Round counts per fraud family, used to detect which directions are already explored."""
        if seed is not None:
            rows = self.connection.execute(
                "SELECT fraud_family, COUNT(*) FROM rounds WHERE seed = ? GROUP BY fraud_family", (seed,)
            ).fetchall()
        else:
            rows = self.connection.execute("SELECT fraud_family, COUNT(*) FROM rounds GROUP BY fraud_family").fetchall()
        return {family: count for family, count in rows}

    def relevant_context(self, limit: int = 12, seed: int | None = None, fraud_family: str | None = None) -> list[str]:
        """Deterministic, relevance-scoped memory for Agent 1: prefer same-family rounds, then recent weakness records from the same seed."""
        same_family = self.get_recent_rounds(limit=max(8, limit), fraud_family=fraud_family, seed=seed) if fraud_family else []
        recent_records = self.get_recent_rounds(limit=max(8, limit), seed=seed)
        seen_ids: set[str] = set()
        selected: list[dict[str, Any]] = []

        for record in same_family + recent_records:
            record_key = f"{record['seed']}:{record['round_id']}"
            if record_key in seen_ids:
                continue
            seen_ids.add(record_key)
            selected.append(record)
            if len(selected) >= max(1, limit):
                break

        context: list[str] = []
        for record in selected:
            context.append(f"round={record['round_id']} family={record['fraud_family']} attack_id={record['attack_id']}")
            context.append(f"hypothesis: {json.dumps(record['attack_hypothesis'])}")
            context.append(f"specification: {json.dumps(record['attack_specification'])}")
            context.append(
                "weakness: "
                + json.dumps(
                    {
                        "identified_weaknesses": record["identified_weaknesses"],
                        "recommended_next_attack_directions": record["recommended_next_attack_directions"],
                        "agent3_analysis": record["agent3_analysis"],
                    }
                )
            )
        return context[-limit:]

    def recent_context(self, limit: int = 12, seed: int | None = None) -> list[str]:
        """Backward-compatible flattened text view of recent rounds, rendered from the structured table."""
        rounds_needed = max(1, -(-limit // 4))
        records = self.get_recent_rounds(limit=rounds_needed, seed=seed)
        context = []
        for record in reversed(records):
            context.append(f"hypothesis: {json.dumps(record['attack_hypothesis'])}")
            context.append(f"specification: {json.dumps(record['attack_specification'])}")
            context.append(
                "evaluation: " + json.dumps({"detection": record["detector_evaluation"], "fidelity": record["fidelity_evaluation"]})
            )
            context.append(f"weakness: {json.dumps(record['agent3_analysis'])}")
        return context[-limit:]

    def close(self) -> None:
        self.connection.close()
