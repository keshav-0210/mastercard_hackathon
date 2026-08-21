from __future__ import annotations

from pathlib import Path

from .contracts import EvidenceReference


class LocalKnowledgeBase:
    def __init__(self, directory: str) -> None:
        self.documents = []
        for path in sorted(Path(directory).glob("*.txt")):
            self.documents.append((path.stem, path.name, path.read_text(encoding="utf-8")))

    def retrieve(self, query: str, top_k: int = 4) -> list[EvidenceReference]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for source_id, title, text in self.documents:
            score = sum(text.lower().count(term) for term in terms)
            scored.append((score, source_id, title, text))
        scored.sort(reverse=True)
        return [EvidenceReference(source_id=source_id, title=title, excerpt=text[:600]) for score, source_id, title, text in scored[:top_k] if score > 0]
