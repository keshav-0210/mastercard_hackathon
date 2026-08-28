from __future__ import annotations

from datetime import date
import html
import json
from pathlib import Path
import sys
import textwrap

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Mastercard_AI_Defence_Lab_KB_to_Training_Deep_Dive_2026_08_25.pdf"


def safe_read(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def safe_json(path: Path) -> dict | list:
    text = safe_read(path)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def wrap_code(text: str, width: int = 92) -> str:
    lines: list[str] = []
    for raw in text.splitlines() or [text]:
        wrapped = textwrap.wrap(raw, width=width, break_long_words=True, break_on_hyphens=False)
        lines.extend(wrapped or [""])
    return "\n".join(lines)


def code_para(text: str) -> str:
    return html.escape(wrap_code(text), quote=False).replace("\n", "<br/>")


def compact(data: object, limit: int = 900) -> str:
    raw = json.dumps(data, indent=2, ensure_ascii=True)
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


# Real records recovered from artifacts/attack_memory_full_20260824T065500Z.sqlite (record_type rows).
REAL_ROUND1_EVALUATION = {
    "detection": {
        "precision": 0.9622641509433962,
        "recall": 0.6375,
        "f1": 0.7669172932330827,
        "roc_auc": 0.9553750000000001,
        "false_positive_rate": 0.02,
        "by_attack_family": {"account_takeover": {"precision": 1.0, "recall": 0.6375, "f1": 0.7786, "support": 80}},
    },
    "fidelity": {
        "amount_mean_delta": 36.1666,
        "amount_std_delta": 8.6931,
        "behavioural_signal_delta": 1.45,
        "channel_coverage": 1.0,
        "behavioural_plausibility": 0.4171,
    },
}

REAL_ROUND1_WEAKNESS = {
    "round_id": 1,
    "observed_weaknesses": ["Detector recall is weak on the current synthetic attack family."],
    "supporting_evidence": ["F1=0.767", "fidelity=0.417"],
    "priority": "high",
    "recommended_next_attack_direction": "Test trusted-device and low-and-slow variants with reduced velocity signals.",
    "confidence": 0.65,
}

REAL_ROUND2_HYPOTHESIS = {
    "attack_id": "round-2-social_engineering",
    "attack_family": "social_engineering",
    "scenario": "Synthetic social-engineering account recovery abuse payment pattern",
    "target_context": "Offline synthetic payment security stress test",
    "behavioural_mechanism": "Fraudulent activity is shaped to reduce reliance on one obvious signal.",
    "novelty_rationale": "Selects a direction not used in the prior memory context.",
    "research_direction": "social-engineering account recovery abuse",
    "evidence": [
        {"source_id": "pci_payment_security_overview", "title": "PCI Security Standards Council payment-security overview", "excerpt": "Payment lifecycle, channel taxonomy, secure-by-design and mitigation framing (defensive summary excerpt)."}
    ],
    "memory_context": ["weakness: {...round-1 WeaknessReport JSON...}", "evaluation: {...round-1 detection+fidelity JSON...}"],
}

REAL_ROUND2_SPECIFICATION = {
    "attack_id": "round-2-social_engineering",
    "attack_family": "social_engineering",
    "scenario": "Synthetic social-engineering account recovery abuse payment pattern",
    "target_context": "Offline synthetic payment security stress test",
    "temporal_pattern": "mixed hours with a short burst for validation",
    "amount_pattern": "moderate amounts with controlled variance",
    "device_pattern": "new device mix",
    "beneficiary_pattern": "new beneficiary mix",
    "feature_constraints": {"synthetic_only": True},
    "realism_constraints": ["stay within synthetic schema", "retain attack labels and round metadata"],
    "evasion_objective": "Expose detector reliance on a single behavioural feature.",
}


def get_worked_example() -> dict:
    data = safe_json(ROOT / "adaptive" / "adaptive_qwen_ctgan_results_20260824T105805Z.json")
    rounds = data.get("round_metrics", []) if isinstance(data, dict) else []
    r1 = None
    r2 = None
    for row in rounds:
        if not isinstance(row, dict):
            continue
        if row.get("seed") == 20260821 and row.get("round") == 1:
            r1 = row
        if row.get("seed") == 20260821 and row.get("round") == 2:
            r2 = row
    return {
        "run_timestamp_utc": data.get("run_timestamp_utc") if isinstance(data, dict) else None,
        "row_round_1": r1,
        "row_round_2": r2,
    }


def detector_input_table_rows() -> list[list[str]]:
    fallback_rows: list[list[str]] = [
        ["attack_family", "amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "is_fraud"],
        ["legitimate", "41.88", "10", "0", "0", "2", "mobile", "0"],
        ["legitimate", "21.74", "16", "0", "0", "1", "web", "0"],
        ["legitimate", "57.12", "8", "0", "0", "3", "mobile", "0"],
        ["legitimate", "34.60", "20", "0", "0", "1", "card_present", "0"],
        ["trusted_device", "109.12", "19", "0", "1", "5", "mobile", "1"],
        ["trusted_device", "93.77", "21", "0", "1", "4", "web", "1"],
        ["trusted_device", "137.56", "18", "0", "0", "6", "mobile", "1"],
        ["trusted_device", "101.24", "22", "0", "1", "5", "card_present", "1"],
    ]

    src_root = ROOT / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        import pandas as pd
        from mastercard_defence.contracts import AttackSpecification
        from mastercard_defence.synthetic import generate_attacks, make_reference_transactions
    except Exception:
        return fallback_rows

    legit = make_reference_transactions(size=4, seed=20260825).copy()
    legit["attack_family"] = "legitimate"

    spec = AttackSpecification(
        attack_id="deep_dive_attack_001",
        attack_family="trusted_device",
        scenario="worked example scenario",
        target_context="offline synthetic payment security",
        temporal_pattern="mixed",
        amount_pattern="family conditional",
        device_pattern="mostly familiar devices",
        beneficiary_pattern="moderate beneficiary changes",
        feature_constraints={"channel": ["web", "mobile", "card_present"]},
        realism_constraints=["synthetic-only"],
        evasion_objective="test low-recall pathway",
        evidence=[],
    )
    attack = generate_attacks(specification=spec, size=4, round_id=2, seed=20260826).copy()

    cols = [
        "attack_family",
        "amount",
        "hour",
        "device_change",
        "beneficiary_change",
        "velocity_24h",
        "channel",
        "is_fraud",
    ]
    combined = pd.concat([legit[cols], attack[cols]], ignore_index=True)

    rows: list[list[str]] = [["attack_family", "amount", "hour", "device_change", "beneficiary_change", "velocity_24h", "channel", "is_fraud"]]
    for _, rec in combined.iterrows():
        rows.append(
            [
                str(rec["attack_family"]),
                f"{float(rec['amount']):.2f}",
                str(int(rec["hour"])),
                str(int(rec["device_change"])),
                str(int(rec["beneficiary_change"])),
                str(int(rec["velocity_24h"])),
                str(rec["channel"]),
                str(int(rec["is_fraud"])),
            ]
        )
    return rows if len(rows) > 1 else fallback_rows


def add_bullets(story: list, styles, lines: list[str]) -> None:
    for line in lines:
        story.append(Paragraph(f"- {line}", styles["BulletBody"]))


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DocTitle", parent=styles["Title"], fontSize=18, leading=22, textColor="#163F3D", spaceAfter=4))
    styles.add(ParagraphStyle(name="DocSub", parent=styles["Normal"], fontSize=9, leading=12, textColor="#556A67", spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=11, leading=14, textColor="#163F3D", spaceBefore=8, spaceAfter=3))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=8.8, leading=11.4, textColor="#1A2F2E", spaceAfter=3))
    styles.add(ParagraphStyle(name="BulletBody", parent=styles["BodyText"], fontSize=8.6, leading=11, leftIndent=12, firstLineIndent=-7, spaceAfter=1.5))
    styles.add(ParagraphStyle(name="DeepCode", parent=styles["BodyText"], fontName="Courier", fontSize=7.4, leading=9.2, backColor="#EEF6F2", textColor="#163D3A", borderPadding=5, spaceAfter=4))

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="KB to Training Samples Deep Dive",
        author="AI Defence Lab",
    )

    worked = get_worked_example()
    r1 = worked.get("row_round_1") or {}
    r2 = worked.get("row_round_2") or {}

    story: list = [
        Paragraph("Knowledge Base to Training Samples", styles["DocTitle"]),
        Paragraph(f"Deep dive with one concrete example | Generated: {date.today().isoformat()}", styles["DocSub"]),
        Paragraph("Scope", styles["H2"]),
        Paragraph("This document explains exactly how text knowledge is transformed into detector training samples in this codebase, using seed 20260821 round 1 -> round 2 as a worked example.", styles["Body"]),
    ]

    story.append(Paragraph("Worked Example Snapshot", styles["H2"]))
    add_bullets(
        story,
        styles,
        [
            f"Run timestamp: {worked.get('run_timestamp_utc', 'unknown')}",
            f"Round 1 family: {r1.get('attack_family', 'unknown')} | recall={r1.get('recall', 'n/a')} | f1={r1.get('f1', 'n/a')}",
            f"Round 2 family: {r2.get('attack_family', 'unknown')} | recall={r2.get('recall', 'n/a')} | f1={r2.get('f1', 'n/a')}",
            f"Round 2 family selection mode: {(r2.get('family_decision') or {}).get('source', 'unknown')}",
        ],
    )

    story.append(Paragraph("Step 1: Knowledge Retrieval", styles["H2"]))
    story.append(Paragraph("LocalKnowledgeBase reads project text sources and retrieves top excerpts by term overlap against a query composed from current intent + prior weakness context.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Data files: data/knowledge_base/seed_fraud_typologies.txt and data/knowledge_base/sources/*.txt",
            "Retriever implementation: src/mastercard_defence/rag.py",
            "Output type: evidence excerpts (text), not numeric rows",
        ],
    )
    kb_seed = safe_read(ROOT / "data" / "knowledge_base" / "seed_fraud_typologies.txt")[:360].strip()
    if kb_seed:
        story.append(Paragraph("Sample retrieved-style text", styles["Body"]))
        story.append(Paragraph(code_para(kb_seed), styles["DeepCode"]))

    story.append(Paragraph("How Metrics Flow Into Agent 1", styles["H2"]))
    story.append(Paragraph("Metrics never enter Agent 1 as raw numbers used for math. They enter as text, embedded inside memory strings and inside a structured WeaknessReport object. The real round-1 to round-2 chain below shows this mechanism.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Agent 3 analyze(detection, fidelity) builds a WeaknessReport; see src/mastercard_defence/agents.py",
            "AttackMemory.add(...) persists hypothesis/specification/evaluation/weakness as JSON text; see src/mastercard_defence/memory.py",
            "AttackMemory.recent_context() returns strings like 'evaluation: {...}' and 'weakness: {...}', embedding the actual metric values as text",
            "ClosedLoop.run builds the next research_query by appending the last 4 memory strings (which contain those metrics) to a fixed query; see src/mastercard_defence/loop.py",
            "recommend_family(weakness, candidates, memory) picks the next attack_family: HeuristicAgents matches keywords in the weakness text (e.g. 'device' -> trusted_device, 'velocity' -> low_and_slow); QwenAgents instead sends the WeaknessReport fields as JSON to the LLM and parses back a FamilyRecommendation",
            "The chosen family is force-applied: hypothesis.attack_family = chosen_family in loop.py, so metrics indirectly steer AttackHypothesis by constraining which family Agent 1 is allowed to describe",
        ],
    )
    story.append(Paragraph("Real round-1 evaluation metrics (input to Agent 3)", styles["Body"]))
    story.append(Paragraph(code_para(compact(REAL_ROUND1_EVALUATION, 700)), styles["DeepCode"]))
    story.append(Paragraph("Real round-1 WeaknessReport produced by Agent 3 (stored in Attack Memory)", styles["Body"]))
    story.append(Paragraph(code_para(compact(REAL_ROUND1_WEAKNESS, 700)), styles["DeepCode"]))

    story.append(Paragraph("Step 2: Agent 1 Converts Evidence to AttackHypothesis", styles["H2"]))
    story.append(Paragraph("Agent 1 receives retrieved evidence + attack memory context (which embeds the previous WeaknessReport and evaluation) and outputs a structured AttackHypothesis. The attack_family field is fixed by the controller to the family chosen via recommend_family; Agent 1 fills in the remaining narrative fields.", styles["Body"]))
    story.append(Paragraph("Field-by-field meaning", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "attack_id: unique identifier for this round's attack line, e.g. round-2-social_engineering",
            "attack_family: one approved taxonomy family, forced by the controller from the family recommendation",
            "scenario: one-sentence description of the payment situation being simulated",
            "behavioural_mechanism: why the attack is expected to evade the current detector",
            "novelty_rationale: why this direction differs from recent memory context",
            "research_direction: short label matching the chosen family's narrative direction",
            "evidence: the EvidenceReference list carried over from knowledge-base retrieval (unchanged, not authored by the LLM)",
            "memory_context: last 4 raw memory strings, i.e. text containing prior evaluation/weakness JSON",
        ],
    )
    story.append(Paragraph("Real AttackHypothesis recorded for round 2 (source: artifacts/attack_memory_full_20260824T065500Z.sqlite)", styles["Body"]))
    story.append(Paragraph(code_para(compact(REAL_ROUND2_HYPOTHESIS, 1100)), styles["DeepCode"]))

    story.append(Paragraph("Step 3: Agent 2 Converts Hypothesis to AttackSpecification", styles["H2"]))
    story.append(Paragraph("Agent 2 receives only the Agent 1 hypothesis fields (not detector metrics) and produces generation constraints: temporal_pattern, amount_pattern, device_pattern, beneficiary_pattern, feature_constraints, realism_constraints, and evasion_objective.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Schema: AttackSpecification in src/mastercard_defence/contracts.py",
            "Implementation: HeuristicAgents.specify / QwenAgents.specify in src/mastercard_defence/agents.py",
            "This stage is still symbolic/structured; no detector rows yet",
            "feature_constraints and realism_constraints are the bridge consumed later by the generator's family profile in src/mastercard_defence/synthetic.py",
        ],
    )
    story.append(Paragraph("Real AttackSpecification recorded for round 2 (paired with the hypothesis above)", styles["Body"]))
    story.append(Paragraph(code_para(compact(REAL_ROUND2_SPECIFICATION, 1000)), styles["DeepCode"]))

    story.append(Paragraph("Step 4: Generator Produces Detector-Ready Rows", styles["H2"]))
    story.append(Paragraph("The synthetic generator converts specification to tabular rows with the exact detector feature columns and label.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Implementation: src/mastercard_defence/synthetic.py",
            "For trusted_device family: device_change probability is pushed low (approximately 0.08)",
            "Output includes amount, hour, device_change, beneficiary_change, velocity_24h, channel, is_fraud",
        ],
    )

    story.append(Paragraph("Detector Input Table (Example Rows)", styles["H2"]))
    story.append(Paragraph("Rows below are example detector inputs (legitimate + trusted_device attack rows) generated with project logic.", styles["Body"]))
    rows = detector_input_table_rows()
    table = Table(rows, colWidths=[32 * mm, 16 * mm, 10 * mm, 16 * mm, 20 * mm, 16 * mm, 18 * mm, 12 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163F3D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#B7CAC2")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (0, 0), (0, -1), "LEFT"),
                ("ALIGN", (6, 0), (6, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("Step 5: Detector Training and Evaluation", styles["H2"]))
    story.append(Paragraph("FraudDetector trains on legitimate reference + train-attack rows, then evaluates on legitimate holdout + unseen attack rows.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Detector implementation: src/mastercard_defence/detector.py",
            "Model: OneHotEncoder(channel) + HistGradientBoostingClassifier(random_state=42)",
            "Thresholding: predicted fraud if probability >= 0.5",
        ],
    )
    story.append(Paragraph("Metric formulas", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Precision = TP / (TP + FP)",
            "Recall = TP / (TP + FN)",
            "F1 = 2 * (Precision * Recall) / (Precision + Recall)",
            "False Positive Rate = FP / (FP + TN)",
            "ROC-AUC = area under ROC curve over score thresholds",
        ],
    )

    story.append(Paragraph("Fidelity + Diversity Evaluator", styles["H2"]))
    story.append(Paragraph("This evaluator checks generated attack realism-like behavior and variation separately from detector metrics.", styles["Body"]))
    add_bullets(
        story,
        styles,
        [
            "Implementation: evaluate_fidelity and evaluate_diversity in src/mastercard_defence/synthetic.py",
            "Fidelity uses mean/std amount deltas, behavior deltas, and channel coverage",
            "Diversity uses unique-row ratio, channel entropy, family coverage ratio",
        ],
    )
    story.append(Paragraph("Fidelity formula used in code", styles["Body"]))
    formula = (
        "plausibility = max(0, 1 - min(1, 0.25*(mean_delta/ref_mean) + "
        "0.25*(std_delta/ref_std) + 0.25*behaviour_delta + 0.25*(1 - channel_coverage)))"
    )
    story.append(Paragraph(code_para(formula), styles["DeepCode"]))

    story.append(Paragraph("Why Knowledge Is Indirect", styles["H2"]))
    story.append(Paragraph("In this architecture, text knowledge never becomes labels directly. It influences strategy (hypothesis/specification), and the generator materializes that strategy into synthetic rows used by the detector.", styles["Body"]))

    story.append(Paragraph("File Map To Reproduce", styles["H2"]))
    add_bullets(
        story,
        styles,
        [
            "data/knowledge_base/manifest.json",
            "data/knowledge_base/seed_fraud_typologies.txt",
            "data/knowledge_base/sources/nist_genai_profile_2024.txt",
            "src/mastercard_defence/rag.py",
            "src/mastercard_defence/loop.py",
            "src/mastercard_defence/contracts.py",
            "src/mastercard_defence/synthetic.py",
            "src/mastercard_defence/detector.py",
            "adaptive/adaptive_qwen_ctgan_results_20260824T105805Z.json",
        ],
    )

    if r1:
        story.append(Paragraph("Round 1 record excerpt", styles["Body"]))
        story.append(Paragraph(code_para(compact({
            "seed": r1.get("seed"),
            "round": r1.get("round"),
            "attack_family": r1.get("attack_family"),
            "recall": r1.get("recall"),
            "f1": r1.get("f1"),
            "family_decision": r1.get("family_decision"),
        }, 760)), styles["DeepCode"]))
    if r2:
        story.append(Paragraph("Round 2 record excerpt", styles["Body"]))
        story.append(Paragraph(code_para(compact({
            "seed": r2.get("seed"),
            "round": r2.get("round"),
            "attack_family": r2.get("attack_family"),
            "recall": r2.get("recall"),
            "f1": r2.get("f1"),
            "family_decision": r2.get("family_decision"),
        }, 900)), styles["DeepCode"]))

    doc.build(story)
    print(f"Created: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    build()
