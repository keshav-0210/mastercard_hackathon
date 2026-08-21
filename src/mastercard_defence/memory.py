from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .contracts import MemoryRecord


class AttackMemory:
    def __init__(self, database_path: str) -> None:
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(database_path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, round_id INTEGER, record_type TEXT, content TEXT, created_at TEXT)")
        self.connection.commit()

    def add(self, record: MemoryRecord) -> None:
        self.connection.execute("INSERT INTO records(round_id, record_type, content, created_at) VALUES (?, ?, ?, ?)", (record.round_id, record.record_type, json.dumps(record.content), record.created_at.isoformat()))
        self.connection.commit()

    def recent_context(self, limit: int = 12) -> list[str]:
        rows = self.connection.execute("SELECT record_type, content FROM records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [f"{record_type}: {content}" for record_type, content in reversed(rows)]

    def close(self) -> None:
        self.connection.close()
