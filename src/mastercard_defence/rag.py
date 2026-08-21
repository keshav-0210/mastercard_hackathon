from __future__ import annotations

import json
from pathlib import Path

from .contracts import EvidenceReference


class LocalKnowledgeBase:
    def __init__(self, directory: str) -> None:
        root = Path(directory)
        manifest_path = root / "manifest.json"
        metadata = {item["source_id"]: item for item in json.loads(manifest_path.read_text(encoding="utf-8"))} if manifest_path.exists() else {}
        self.documents = []
        for path in sorted(root.glob("*.txt")) + sorted((root / "sources").glob("*.txt")):
            source_id = path.stem
            item = metadata.get(source_id, {})
            self.documents.append((item.get("source_id", source_id), item.get("title", path.name), path.read_text(encoding="utf-8")))

    def retrieve(self, query: str, top_k: int = 4) -> list[EvidenceReference]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for source_id, title, text in self.documents:
            score = sum(text.lower().count(term) for term in terms)
            scored.append((score, source_id, title, text))
        scored.sort(reverse=True)
        return [EvidenceReference(source_id=source_id, title=title, excerpt=text[:600]) for score, source_id, title, text in scored[:top_k] if score > 0]
