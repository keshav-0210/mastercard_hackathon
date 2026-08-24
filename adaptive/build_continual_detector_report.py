from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "adaptive" / "adaptive_qwen_ctgan_continual_results_20260824T123152Z.json"
GRAPH = ROOT / "adaptive" / "adaptive_qwen_ctgan_continual_graphs_20260824T123152Z.png"
OUTPUT = ROOT / "adaptive" / "Mastercard_AI_Defence_Lab_Continual_Detector_v1_20260824.pdf"

REFERENCE_3 = {
    "F1": 0.8010,
    "Recall": 0.6842,
    "Precision": 0.9685,
    "ROC-AUC": 0.9248,
    "Novelty": 0.9569,
    "Plausibility": 0.5126,
}


def build() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    summary = data["summary"]
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle2", parent=styles["Title"], fontSize=17, leading=20, textColor="#123c3b", spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportH2", parent=styles["Heading2"], fontSize=11, leading=13, textColor="#123c3b", spaceBefore=6, spaceAfter=3))
    styles.add(ParagraphStyle(name="ReportBody2", parent=styles["BodyText"], fontSize=8.3, leading=10.5, spaceAfter=3))
    styles.add(ParagraphStyle(name="ReportSmall", parent=styles["BodyText"], fontSize=7.4, leading=9, textColor="#4f5b66", spaceAfter=2))

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=14 * mm, leftMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm, title="Continual Detector Improvement", author="AI Defence Lab")
    story = [
        Paragraph("AI Defence Lab: Continual Detector Improvement", styles["ReportTitle2"]),
        Paragraph("Reference 3 comparison · Adaptive QwenAgents + conditional CTGAN · 3 seeds x 5 rounds", styles["ReportBody2"]),
        Paragraph("Objective", styles["ReportH2"]),
        Paragraph("This experiment keeps the adaptive red team fixed and tests whether a continual blue team improves by replaying validated missed synthetic attacks. Reference 3 is the sole comparison: adaptive QwenAgents + CTGAN with the static detector.", styles["ReportBody2"]),
        Paragraph("Continual detector policy", styles["ReportH2"]),
        Paragraph("After each unseen evaluation, fraud rows missed by the detector are retained as hard examples. The next round trains with the normal training data plus the accumulated replay buffer. Round 1 starts with no replay; later rounds use the stored hard examples.", styles["ReportBody2"]),
        Paragraph("Reference 3 versus continual detector", styles["ReportH2"]),
    ]
    comparison_rows = [["Metric", "Reference 3", "Continual", "Change"]]
    for label, key in (("F1", "f1"), ("Recall", "recall"), ("Precision", "precision"), ("ROC-AUC", "roc_auc"), ("Novelty", "novelty_score"), ("Plausibility", "behavioural_plausibility")):
        current = summary[key]["mean"]
        reference = REFERENCE_3[label]
        comparison_rows.append([label, f"{reference:.4f}", f"{current:.4f}", f"{current - reference:+.4f}"])
    story.append(Table(comparison_rows, colWidths=[48 * mm, 40 * mm, 40 * mm, 40 * mm], style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#123c3b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c5d0c5")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)])))
    story.extend([
        Paragraph("Observed replay and response", styles["ReportH2"]),
        Paragraph("Replay grew across rounds, showing that the continual detector was actually carrying forward missed examples. Mean recall increased from 0.6842 in Reference 3 to 0.7600, while mean F1 increased from 0.8010 to 0.8115. Precision decreased from 0.9685 to 0.8801, so the improvement is a recall-focused trade-off rather than a universal gain.", styles["ReportBody2"]),
        Paragraph("Red-team quality remained measurable: novelty averaged 0.9709 and behavioural plausibility averaged 0.5914. These are internal synthetic indicators, not official Mastercard scores or live-payment performance claims.", styles["ReportBody2"]),
        PageBreak(),
        Paragraph("Convergence evidence", styles["ReportH2"]),
        Image(str(GRAPH), width=180 * mm, height=120 * mm),
        Spacer(1, 3),
        Paragraph("How to read the graph", styles["ReportH2"]),
        Paragraph("The upper-left panel shows recall, F1, and precision for the continual blue team across rounds. The upper-right panel shows hard-example replay growth. The lower-left panel shows family-level recall, and the lower-right panel shows the plausibility versus detector-difficulty frontier.", styles["ReportBody2"]),
        Paragraph("Interpretation", styles["ReportH2"]),
        Paragraph("The continual approach demonstrates a useful convergence mechanism: detector misses are converted into training evidence for later rounds. The result is higher recall and slightly higher F1 than Reference 3, with controlled ROC-AUC change and a measurable precision cost. The next detector refinement should tune the decision threshold or calibrate probabilities to recover precision while retaining replay-driven recall gains.", styles["ReportBody2"]),
        Paragraph("Scope and limitations", styles["ReportH2"]),
        Paragraph("All transactions are synthetic. The experiment does not use real cardholder or production payment data, and the realism score measures similarity to a controlled synthetic legitimate reference. The comparison is a repeated seeded experiment using the same architecture and protocol, not an identical-row paired test because the generated batches are regenerated.", styles["ReportSmall"]),
    ])
    document.build(story)
    print(f"Created: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    build()
