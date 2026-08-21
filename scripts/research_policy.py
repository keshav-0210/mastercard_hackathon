from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ApprovedResearchSource:
    source_id: str
    url: str
    allowed_use: str


APPROVED_SOURCES = (
    ApprovedResearchSource(
        "nist_ai_rmf_2023",
        "https://www.nist.gov/itl/ai-risk-management-framework",
        "Defensive AI risk, governance, evaluation, and auditability summaries.",
    ),
    ApprovedResearchSource(
        "nist_genai_profile_2024",
        "https://doi.org/10.6028/NIST.AI.600-1",
        "Defensive generative-AI risk and misuse-control summaries.",
    ),
    ApprovedResearchSource(
        "enisa_threat_landscape_2024",
        "https://www.enisa.europa.eu/publications/enisa-threat-landscape-2024",
        "High-level public threat taxonomy and mitigation summaries.",
    ),
)


def assert_safe_source(source_id: str) -> ApprovedResearchSource:
    for source in APPROVED_SOURCES:
        if source.source_id == source_id:
            return source
    raise ValueError(f"Source is not on the approved allowlist: {source_id}")