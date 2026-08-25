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
        self._by_source_id = {source_id: (title, text) for source_id, title, text in self.documents}

    def retrieve(self, query: str, top_k: int = 4) -> list[EvidenceReference]:
        terms = {term.lower() for term in query.split() if len(term) > 2}
        scored = []
        for source_id, title, text in self.documents:
            score = sum(text.lower().count(term) for term in terms)
            scored.append((score, source_id, title, text))
        scored.sort(reverse=True)
        return [EvidenceReference(source_id=source_id, title=title, excerpt=text[:600]) for score, source_id, title, text in scored[:top_k] if score > 0]

    def retrieve_for_family(self, family: str, query: str, top_k: int = 4) -> list[EvidenceReference]:
        """Guarantee the family's own grounding document is always returned first,
        so retrieval quality cannot be diluted by memory-context term overlap."""
        results: list[EvidenceReference] = []
        seen_ids: set[str] = set()
        family_source_id = f"family_{family}"
        if family_source_id in self._by_source_id:
            title, text = self._by_source_id[family_source_id]
            results.append(EvidenceReference(source_id=family_source_id, title=title, excerpt=text[:600]))
            seen_ids.add(family_source_id)
        for item in self.retrieve(query, top_k):
            if item.source_id in seen_ids:
                continue
            results.append(item)
            seen_ids.add(item.source_id)
            if len(results) >= top_k:
                break
        return results[:top_k]
