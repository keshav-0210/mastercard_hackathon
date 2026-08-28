from __future__ import annotations

import html
import os
import re
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.getenv("PIPELINE_GUIDE_SOURCE", ROOT / "PIPELINE_AND_METRICS_GUIDE.md"))
OUTPUT = Path(os.getenv("PIPELINE_GUIDE_PDF", ROOT / "PIPELINE_AND_METRICS_GUIDE.pdf"))
ASSET_DIR = ROOT / ".pdf_assets"

NAVY = colors.HexColor("#123C3B")
TEAL = colors.HexColor("#007F7B")
GOLD = colors.HexColor("#D69E2E")
INK = colors.HexColor("#172B2A")
MUTED = colors.HexColor("#536361")
PALE = colors.HexColor("#EEF5F2")
LINE = colors.HexColor("#C8D8D2")


def inline_markup(text: str) -> str:
    escaped = html.escape(text, quote=False)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font name='Courier' color='#007F7B'>\1</font>", escaped)
    escaped = escaped.replace("  ", "<br/>")
    return escaped


def normalize_mathtext(equation: str) -> str:
    equation = equation.replace(r"\ge ", r"\geq ").replace(r"\le ", r"\leq ")
    equation = re.sub(r"\\text\{([^{}]*)\}", r"\\mathrm{\1}", equation)
    return equation


def render_equation(equation: str, index: int) -> Path:
    ASSET_DIR.mkdir(exist_ok=True)
    path = ASSET_DIR / f"equation_{index}.png"
    figure = plt.figure(figsize=(10, 0.65), dpi=180)
    figure.patch.set_alpha(0)
    axis = figure.add_axes([0, 0, 1, 1])
    axis.axis("off")
    axis.text(0.5, 0.5, f"${normalize_mathtext(equation.strip())}$", ha="center", va="center", fontsize=13, color="#172B2A")
    figure.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)
    return path


def render_pipeline_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6), dpi=180)
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.axis("off")

    box_specs = {
        "knowledge": (0.7, 5.8, 1.8, 0.8, "Knowledge"),
        "rag": (2.8, 5.8, 1.8, 0.8, "RAG"),
        "taxonomy": (4.9, 5.8, 2.0, 0.8, "Taxonomy"),
        "plan": (7.1, 5.8, 2.0, 0.8, "Family\nplan"),
        "agent1": (9.5, 5.8, 1.9, 0.8, "Agent 1"),
        "controller": (11.9, 5.8, 2.1, 0.8, "Controller"),
        "agent2": (14.4, 5.8, 1.9, 0.8, "Agent 2"),
        "generator": (7.8, 3.6, 2.2, 0.8, "Generator"),
        "train": (10.5, 3.6, 2.2, 0.8, "Training\nbatch"),
        "unseen": (13.2, 3.6, 2.1, 0.8, "Unseen\nbatch"),
        "detector": (9.1, 1.7, 2.1, 0.8, "Detector"),
        "evaluation": (12.0, 1.7, 2.6, 0.8, "Evaluation"),
        "agent3": (15.1, 1.7, 1.8, 0.8, "Agent 3"),
        "memory": (15.1, 3.6, 2.0, 0.8, "Attack\nmemory"),
        "replay": (7.8, 0.3, 2.0, 0.8, "Replay"),
        "recommend": (10.8, 0.3, 2.7, 0.8, "Recommendation"),
    }

    colors_by_box = {
        "knowledge": "#0d5f63",
        "rag": "#0d5f63",
        "taxonomy": "#0d5f63",
        "plan": "#eaf2ef",
        "agent1": "#eaf2ef",
        "controller": "#eaf2ef",
        "agent2": "#eaf2ef",
        "generator": "#eaf2ef",
        "train": "#eaf2ef",
        "unseen": "#eaf2ef",
        "detector": "#eaf2ef",
        "evaluation": "#eaf2ef",
        "agent3": "#eaf2ef",
        "memory": "#d9b45b",
        "replay": "#d9b45b",
        "recommend": "#eaf2ef",
    }

    for key, (x, y, w, h, label) in box_specs.items():
        rect = plt.Rectangle((x, y), w, h, facecolor=colors_by_box[key], edgecolor="#1b4a4a", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.8, color="#0e2c2d")

    arrow_specs = [
        ("knowledge", "rag"),
        ("rag", "agent1"),
        ("taxonomy", "plan"),
        ("plan", "agent1"),
        ("agent1", "controller"),
        ("controller", "agent2"),
        ("agent2", "generator"),
        ("generator", "train"),
        ("generator", "unseen"),
        ("train", "detector"),
        ("unseen", "evaluation"),
        ("detector", "evaluation"),
        ("evaluation", "agent3"),
        ("agent3", "memory"),
        ("memory", "recommend"),
        ("recommend", "controller"),
        ("memory", "replay"),
        ("replay", "detector"),
    ]

    for src, dst in arrow_specs:
        x1, y1, w1, h1, _ = box_specs[src]
        x2, y2, w2, h2, _ = box_specs[dst]
        start = (x1 + w1, y1 + h1 / 2)
        end = (x2, y2 + h2 / 2)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle='-|>',
                lw=1.5,
                color="#0d5f63",
                shrinkA=2,
                shrinkB=2,
                connectionstyle="arc3,rad=0.0",
            ),
        )

    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor(), transparent=False)
    plt.close(fig)


def table_from_lines(lines: list[str], styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[Paragraph]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= set("-:") for cell in cells):
            continue
        rows.append([Paragraph(inline_markup(cell), styles["TableCell"]) for cell in cells])
    widths = [42 * mm, 34 * mm, 34 * mm, 34 * mm] if len(rows[0]) == 4 else None
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build() -> None:
    output_path = OUTPUT
    if OUTPUT.exists():
        try:
            OUTPUT.unlink()
        except PermissionError:
            output_path = OUTPUT.with_name(f"{OUTPUT.stem}_{uuid.uuid4().hex}{OUTPUT.suffix}")
            print(f"Output locked: {OUTPUT}. Falling back to {output_path.name}")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="GuideTitle", parent=styles["Title"], alignment=TA_CENTER, fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=NAVY, spaceAfter=6))
    styles.add(ParagraphStyle(name="GuideSubtitle", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10, leading=14, textColor=MUTED, spaceAfter=16))
    styles.add(ParagraphStyle(name="GuideH2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle(name="GuideH3", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=10.5, leading=13, textColor=TEAL, spaceBefore=8, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle(name="GuideBody", parent=styles["BodyText"], fontSize=8.8, leading=12.2, textColor=INK, spaceAfter=5))
    styles.add(ParagraphStyle(name="GuideBullet", parent=styles["BodyText"], fontSize=8.8, leading=12, leftIndent=13, firstLineIndent=-8, bulletIndent=3, textColor=INK, spaceAfter=2))
    styles.add(ParagraphStyle(name="GuideCode", parent=styles["Code"], fontName="Courier", fontSize=7.4, leading=9.4, textColor=INK, backColor=PALE, borderColor=LINE, borderWidth=0.4, borderPadding=6, spaceBefore=3, spaceAfter=6))
    styles.add(ParagraphStyle(name="GuideQuote", parent=styles["GuideBody"], borderColor=GOLD, borderWidth=0.5, borderPadding=5, backColor=colors.HexColor("#FFF8E7")))
    styles.add(ParagraphStyle(name="TableCell", parent=styles["BodyText"], fontSize=7.6, leading=9.3, textColor=INK))

    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm, title="AI Defence Lab: Pipeline, Metrics, and Iteration Guide", author="AI Defence Lab")
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story: list[Flowable] = []
    in_code = False
    code_lines: list[str] = []
    code_language = ""
    in_equation = False
    equation_lines: list[str] = []
    equation_index = 0
    table_lines: list[str] = []
    saw_title = False

    def flush_table() -> None:
        nonlocal table_lines
        if table_lines:
            story.append(table_from_lines(table_lines, styles))
            story.append(Spacer(1, 5))
            table_lines = []

    for raw_line in lines:
        line = raw_line.rstrip()
        if in_equation:
            if line.strip() == "$$":
                equation_index += 1
                equation_path = render_equation(" ".join(equation_lines), equation_index)
                story.append(Image(str(equation_path), width=165 * mm, height=11 * mm))
                story.append(Spacer(1, 2))
                equation_lines = []
                in_equation = False
            else:
                equation_lines.append(line)
            continue
        if line.strip() == "$$":
            flush_table()
            in_equation = True
            continue
        if line.startswith("```"):
            flush_table()
            if in_code:
                if code_language == "mermaid":
                    ASSET_DIR.mkdir(exist_ok=True)
                    diagram_path = ASSET_DIR / "pipeline_diagram.png"
                    render_pipeline_diagram(diagram_path)
                    story.append(Image(str(diagram_path), width=175 * mm, height=64 * mm))
                else:
                    story.append(Preformatted("\n".join(code_lines), styles["GuideCode"]))
                story.append(Spacer(1, 4))
                code_lines = []
                code_language = ""
                in_code = False
            else:
                in_code = True
                code_language = line[3:].strip().lower()
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if not line:
            story.append(Spacer(1, 2))
        elif line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["GuideTitle"]))
            saw_title = True
        elif line.startswith("## "):
            if saw_title:
                story.append(Spacer(1, 3))
            story.append(Paragraph(inline_markup(line[3:]), styles["GuideH2"]))
        elif line.startswith("### "):
            story.append(Paragraph(inline_markup(line[4:]), styles["GuideH3"]))
        elif line.startswith("- "):
            story.append(Paragraph(inline_markup(line[2:]), styles["GuideBullet"], bulletText="-"))
        elif line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["GuideQuote"]))
        elif line.startswith("**") and line.endswith("**"):
            story.append(Paragraph(inline_markup(line[2:-2]), styles["GuideSubtitle"]))
        else:
            story.append(Paragraph(inline_markup(line), styles["GuideBody"]))

    flush_table()
    document.build(story)
    for asset in ASSET_DIR.glob("equation_*.png"):
        asset.unlink()
    try:
        ASSET_DIR.rmdir()
    except OSError:
        pass
    print(f"Created: {output_path}")
    print(f"Bytes: {output_path.stat().st_size}")
    print(f"Equations rendered: {equation_index}")


if __name__ == "__main__":
    build()
