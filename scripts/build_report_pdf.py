from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "competition_architecture_report.md"
OUTPUT = Path(os.getenv("REPORT_PDF_OUTPUT", ROOT / "Mastercard_AI_Defence_Lab_Architecture_Report.pdf"))


def inline_markup(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("`", "")


def build() -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, textColor="#17365D", spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSubtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, leading=15, textColor="#4F5B66", spaceAfter=18))
    styles.add(ParagraphStyle(name="H2Report", parent=styles["Heading2"], fontSize=14, leading=18, textColor="#17365D", spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="H3Report", parent=styles["Heading3"], fontSize=11, leading=14, textColor="#008ABC", spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyReport", parent=styles["BodyText"], fontSize=9.2, leading=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="BulletReport", parent=styles["BodyText"], fontSize=9.2, leading=13, leftIndent=14, firstLineIndent=-8, bulletIndent=4, spaceAfter=3))

    document = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm, topMargin=16 * mm, bottomMargin=16 * mm, title="Mastercard AI Defence Lab Architecture Report", author="AI Defence Lab")
    story = []
    in_code = False
    code_lines: list[str] = []

    for raw_line in SOURCE.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                story.append(Preformatted("\n".join(code_lines), styles["Code"]))
                story.append(Spacer(1, 5))
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not line:
            story.append(Spacer(1, 2))
        elif line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["ReportTitle"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), styles["H2Report"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["H3Report"]))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:]), styles["BulletReport"], bulletText="-"))
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(inline_markup(line[2:-2]), styles["ReportSubtitle"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["BodyReport"]))

    document.build(story)
    print(f"Created: {OUTPUT}")
    print(f"Bytes: {OUTPUT.stat().st_size}")


if __name__ == "__main__":
    build()
