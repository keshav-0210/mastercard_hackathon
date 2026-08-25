from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Mastercard_AI_Defence_Lab_Adaptive_Design_v1_20260824.pdf"


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="AdaptiveTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=21, textColor="#123c3b", spaceAfter=4))
    styles.add(ParagraphStyle(name="AdaptiveSubtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, leading=11, textColor="#5d6a65", spaceAfter=8))
    styles.add(ParagraphStyle(name="AdaptiveH2", parent=styles["Heading2"], fontSize=11, leading=13, textColor="#123c3b", spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="AdaptiveBody", parent=styles["BodyText"], fontSize=8.2, leading=10.2, spaceAfter=3))
    styles.add(ParagraphStyle(name="AdaptiveBullet", parent=styles["BodyText"], fontSize=8.1, leading=10, leftIndent=10, firstLineIndent=-6, spaceAfter=1.5))
    styles.add(ParagraphStyle(name="AdaptiveCode", parent=styles["BodyText"], fontName="Courier", fontSize=7.2, leading=8.5, textColor="#123c3b", backColor="#eef2eb", borderPadding=5, spaceAfter=5))

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=11 * mm, bottomMargin=11 * mm, title="Adaptive Attack Family Design", author="AI Defence Lab")
    story = [
        Paragraph("AI Defence Lab", styles["AdaptiveTitle"]),
        Paragraph("Adaptive Attack Family Design v1 · 24 August 2026", styles["AdaptiveSubtitle"]),
        Paragraph("Purpose", styles["AdaptiveH2"]),
        Paragraph("The final loop should be a stable benchmark with evidence-driven adaptation. The approved taxonomy keeps experiments comparable; detector weaknesses guide the next family or variant; new patterns become official only after validation.", styles["AdaptiveBody"]),
        Paragraph("The family levels", styles["AdaptiveH2"]),
        Paragraph("<b>Approved benchmark families:</b> account_takeover, trusted_device, beneficiary_manipulation, low_and_slow, social_engineering, merchant_abuse, cross_channel_anomaly.", styles["AdaptiveBody"]),
        Paragraph("<b>Family policy:</b> Agent 1 may change feature-level constraints, but every round must use exactly one of the seven approved families. Variants and composite families are not created.", styles["AdaptiveBody"]),
        Paragraph("How one round works", styles["AdaptiveH2"]),
        Paragraph("Base documents provide general payment-security knowledge. Attack Memory provides previous detector weaknesses. Agent 1 uses both to recommend a direction. The controller validates it, Agent 2 writes constraints, the generator creates rows, and the detector evaluates unseen rows.", styles["AdaptiveBody"]),
        Paragraph("Knowledge + memory → Agent 1 recommendation → controller validation → Agent 2 specification → generator → detector → Agent 3 weakness → memory", styles["AdaptiveCode"]),
        Paragraph("Simple example", styles["AdaptiveH2"]),
        Paragraph("<b>Round 1:</b> The seeded schedule starts with account_takeover. The detector misses attacks that use normal devices.", styles["AdaptiveBody"]),
        Paragraph("<b>Round 2:</b> Agent 1 recommends trusted_device with normal device behavior, moderate beneficiary changes, ordinary velocity, and controlled amounts.", styles["AdaptiveBody"]),
        Paragraph("<b>Round 3:</b> If the detector still misses attacks spread over time, Agent 1 recommends low_and_slow.", styles["AdaptiveBody"]),
        Paragraph("<b>Round 4:</b> If both weaknesses combine, Agent 1 continues with an approved family and changes its structured constraints. No composite family is created.", styles["AdaptiveBody"]),
        PageBreak(),
        Paragraph("How the recommendation is controlled", styles["AdaptiveH2"]),
    ]
    checks = [
        [Paragraph("Agent 1 proposes", styles["AdaptiveBody"]), Paragraph("family, variant or candidate; reason; target weakness; confidence", styles["AdaptiveBody"])],
        [Paragraph("Controller checks", styles["AdaptiveBody"]), Paragraph("approved taxonomy, relevance to weakness, recent repetition, generator support", styles["AdaptiveBody"])],
        [Paragraph("Agent 2 writes", styles["AdaptiveBody"]), Paragraph("time, amount, device, beneficiary, channel and evasion constraints", styles["AdaptiveBody"])],
        [Paragraph("Generator validates", styles["AdaptiveBody"]), Paragraph("schema, ranges, labels, family purity, diversity and reference similarity", styles["AdaptiveBody"])],
        [Paragraph("Promotion gate", styles["AdaptiveBody"]), Paragraph("novelty, realism, generator support and repeated multi-seed evidence", styles["AdaptiveBody"])],
    ]
    story.append(Table(checks, colWidths=[40 * mm, 136 * mm], style=TableStyle([("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8eee8")), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5d0c5")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)])))
    story.extend([
        Paragraph("How realistic quality is protected", styles["AdaptiveH2"]),
        Paragraph("The LLM does not generate raw transaction rows. It proposes intent and constraints. Family profiles and the learned generator create the data, while validators keep it within the synthetic schema.", styles["AdaptiveBody"]),
    ])
    for item in [
        "Family profiles map each direction to feature relationships.",
        "Agent 2 states what should remain normal and what should change.",
        "Amounts, hours, binary fields, velocity, channels and labels are validated.",
        "Generated attacks are compared with legitimate synthetic reference data.",
        "Mixed-family rows are rejected rather than silently relabelled.",
        "Unseen rows are kept separate from generator and detector training data.",
    ]:
        story.append(Paragraph(f"- {item}", styles["AdaptiveBullet"]))
    story.extend([
        Paragraph("Final policy", styles["AdaptiveH2"]),
        Paragraph("1. Start with a seeded approved family. 2. Use the latest weakness to guide Agent 1. 3. Validate the recommendation. 4. Let Agent 2 create structured constraints. 5. Generate and evaluate. 6. Store the weakness. 7. Promote candidates only after repeated evidence.", styles["AdaptiveBody"]),
        Paragraph("Stable benchmark + adaptive variants + controlled discovery", styles["AdaptiveCode"]),
        Paragraph("The taxonomy is versioned: taxonomy_v1, taxonomy_v2, and so on. This preserves old reports while allowing the method to improve. The result is neither a completely fixed schedule nor uncontrolled automatic discovery.", styles["AdaptiveBody"]),
        Paragraph("This design note describes the intended method. Metrics from the existing baseline and generator reports remain separate experiment evidence and are not official Mastercard scores.", styles["AdaptiveBody"]),
    ])
    document.build(story)
    print(f"Created: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    build()
