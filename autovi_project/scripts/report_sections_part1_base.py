"""Shared styling and helper primitives for AutoVI IEEE report generator."""

from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image

OUTPUT_PDF = Path("outputs/AutoVI_IEEE_Report_25Pages.pdf")
FIG_DIR = Path("outputs/figures")
RENDER_DIR = Path("outputs/edge_windows_test_renders")
EXP_DIR = Path("autovi_project/outputs/experiments/pipe_clip")

def get_styles():
    styles = {}
    styles["Title"] = ParagraphStyle("IEEETitle", fontName="Times-Bold", fontSize=15, leading=18, alignment=1, spaceAfter=5, textColor=colors.HexColor("#0D1B2A"))
    styles["Authors"] = ParagraphStyle("IEEEAuthors", fontName="Times-Roman", fontSize=8.5, leading=11, alignment=1, spaceAfter=3, textColor=colors.HexColor("#1A202C"))
    styles["Affiliations"] = ParagraphStyle("IEEEAffiliations", fontName="Times-Italic", fontSize=7, leading=9, alignment=1, spaceAfter=7, textColor=colors.HexColor("#4A5568"))
    styles["Abstract"] = ParagraphStyle("IEEEAbstract", fontName="Times-Roman", fontSize=7.4, leading=9.4, alignment=4, spaceAfter=4, textColor=colors.HexColor("#1A202C"))
    styles["IndexTerms"] = ParagraphStyle("IEEEIndexTerms", fontName="Times-Roman", fontSize=7.4, leading=9.4, alignment=4, spaceAfter=6, textColor=colors.HexColor("#1A202C"))
    styles["Heading1"] = ParagraphStyle("IEEEHeading1", fontName="Times-Bold", fontSize=8.8, leading=11, alignment=1, spaceBefore=6, spaceAfter=2.5, keepWithNext=True, textColor=colors.HexColor("#0D1B2A"))
    styles["Heading2"] = ParagraphStyle("IEEEHeading2", fontName="Times-BoldItalic", fontSize=8, leading=10, alignment=0, spaceBefore=4.5, spaceAfter=2, keepWithNext=True, textColor=colors.HexColor("#1B263B"))
    styles["Heading3"] = ParagraphStyle("IEEEHeading3", fontName="Times-Italic", fontSize=7.5, leading=9.5, alignment=0, spaceBefore=3.5, spaceAfter=1.5, keepWithNext=True, textColor=colors.HexColor("#2C3E50"))
    styles["Body"] = ParagraphStyle("IEEEBody", fontName="Times-Roman", fontSize=7.6, leading=9.4, alignment=4, spaceAfter=3, firstLineIndent=8, textColor=colors.HexColor("#111111"))
    styles["BodyNoIndent"] = ParagraphStyle("IEEEBodyNoIndent", fontName="Times-Roman", fontSize=7.6, leading=9.4, alignment=4, spaceAfter=3, firstLineIndent=0, textColor=colors.HexColor("#111111"))
    styles["Equation"] = ParagraphStyle("IEEEEquation", fontName="Times-Italic", fontSize=7.5, leading=10, alignment=1, spaceBefore=2, spaceAfter=2, textColor=colors.HexColor("#0B3954"))
    styles["Caption"] = ParagraphStyle("IEEECaption", fontName="Times-Roman", fontSize=6.8, leading=8.5, alignment=1, spaceBefore=2, spaceAfter=3, textColor=colors.HexColor("#2B2D42"))
    styles["TableText"] = ParagraphStyle("IEEETableText", fontName="Times-Roman", fontSize=6.2, leading=7.5, alignment=0, textColor=colors.HexColor("#111111"))
    styles["TableTextBold"] = ParagraphStyle("IEEETableTextBold", fontName="Times-Bold", fontSize=6.2, leading=7.5, alignment=0, textColor=colors.HexColor("#0D1B2A"))
    styles["Reference"] = ParagraphStyle("IEEEReference", fontName="Times-Roman", fontSize=6.5, leading=8.2, alignment=4, spaceAfter=1.5, firstLineIndent=-8, leftIndent=8, textColor=colors.HexColor("#222222"))
    return styles

STYLES = get_styles()

def p(text, style_name="Body"):
    return Paragraph(text, STYLES[style_name])

def eq(text):
    return Paragraph(f"<i>{text}</i>", STYLES["Equation"])

def h1(text):
    return Paragraph(f"<b>{text.upper()}</b>", STYLES["Heading1"])

def h2(text):
    return Paragraph(f"<i><b>{text}</b></i>", STYLES["Heading2"])

def h3(text):
    return Paragraph(f"<i>{text}</i>", STYLES["Heading3"])

def make_table(header, rows, widths, caption=None):
    elems = []
    if caption:
        elems.append(Paragraph(f"<b>TABLE {caption}</b>", STYLES["Caption"]))
    table_data = [[Paragraph(f"<b>{c}</b>", STYLES["TableTextBold"]) for c in header]]
    for r in rows:
        table_data.append([Paragraph(str(c), STYLES["TableText"]) for c in r])
    t = Table(table_data, colWidths=widths)
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.HexColor("#111111")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#111111")),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#111111")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 2))
    return elems

def make_fig(path, w, h, num, text):
    elems = []
    p_path = Path(path)
    if p_path.exists():
        elems.append(Image(str(p_path), width=w, height=h))
        elems.append(Paragraph(f"<b>Fig. {num}.</b> {text}", STYLES["Caption"]))
    else:
        elems.append(Paragraph(f"[Figure {num}: {path}]", STYLES["Caption"]))
    return elems
