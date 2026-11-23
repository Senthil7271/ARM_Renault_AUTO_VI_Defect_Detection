"""Assembles the full 25-page IEEE report and builds outputs/AutoVI_IEEE_Report_25Pages.pdf."""

import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    FrameBreak,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
)

from report_sections_part1_base import OUTPUT_PDF
from report_sections_part1 import (
    get_page_1, get_page_2, get_page_3, get_page_4,
    get_page_5, get_page_6, get_page_7, get_page_8
)
from report_sections_part2 import (
    get_page_9, get_page_10, get_page_11, get_page_12,
    get_page_13, get_page_14, get_page_15, get_page_16
)
from report_sections_part3 import (
    get_page_17, get_page_18, get_page_19, get_page_20,
    get_page_21, get_page_22, get_page_23, get_page_24, get_page_25
)


class IEEENumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages):
        self.saveState()
        self.setFont("Times-Roman", 7.5)
        self.setFillColor(colors.HexColor("#333333"))
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)

        if self._pageNumber > 1:
            if self._pageNumber % 2 == 0:
                self.drawString(54, 750, "IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS, VOL. 22, NO. 4, APRIL 2026")
                self.drawRightString(558, 750, "LINGA ET AL.: AUTOVI END-TO-END INDUSTRIAL ANOMALY DETECTION")
            else:
                self.drawString(54, 750, "AUTOVI: END-TO-END CLOUD-TO-EDGE INDUSTRIAL INSPECTION")
                self.drawRightString(558, 750, "IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS")
            self.line(54, 744, 558, 744)

        self.line(54, 45, 558, 45)
        self.drawString(54, 34, "CONFIDENTIAL & PROPRIETARY — AUTOVI CONSORTIUM (RENAULT GROUP / ARM CHALLENGE)")
        self.drawRightString(558, 34, f"Page {self._pageNumber} of {total_pages}")
        self.restoreState()


def build_pdf():
    print(f"[PDF] Initializing ReportLab document: {OUTPUT_PDF}...")
    doc = BaseDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    page_w, page_h = letter  # 612 x 792
    margin_x = 54
    margin_y = 54
    usable_w = page_w - 2 * margin_x  # 504
    col_gap = 18
    col_w = (usable_w - col_gap) / 2  # 243

    # Page 1 frames: Top banner (height=320 pt, y=420) + 2 half columns below (height=356 pt, y=54)
    f_top = Frame(margin_x, 420, usable_w, 320, id="p1_top", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_p1_c1 = Frame(margin_x, margin_y, col_w, 356, id="p1_c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_p1_c2 = Frame(margin_x + col_w + col_gap, margin_y, col_w, 356, id="p1_c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    tmpl_first = PageTemplate(id="FirstPage", frames=[f_top, f_p1_c1, f_p1_c2])

    # Standard TwoCol frames: Full height (686 pt)
    f_std_c1 = Frame(margin_x, margin_y, col_w, 686, id="std_c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_std_c2 = Frame(margin_x + col_w + col_gap, margin_y, col_w, 686, id="std_c2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    tmpl_std = PageTemplate(id="TwoCol", frames=[f_std_c1, f_std_c2])

    doc.addPageTemplates([tmpl_first, tmpl_std])

    story = []

    # Page 1
    top1, c1_1, c2_1 = get_page_1()
    story.extend(top1)
    story.append(FrameBreak())
    story.extend(c1_1)
    story.append(FrameBreak())
    story.extend(c2_1)
    story.append(NextPageTemplate("TwoCol"))
    story.append(PageBreak())

    # Pages 2 to 25
    page_funcs = [
        get_page_2, get_page_3, get_page_4, get_page_5,
        get_page_6, get_page_7, get_page_8, get_page_9,
        get_page_10, get_page_11, get_page_12, get_page_13,
        get_page_14, get_page_15, get_page_16, get_page_17,
        get_page_18, get_page_19, get_page_20, get_page_21,
        get_page_22, get_page_23, get_page_24, get_page_25,
    ]

    for idx, fn in enumerate(page_funcs, start=2):
        _, c1, c2 = fn()
        story.extend(c1)
        story.append(FrameBreak())
        story.extend(c2)
        if idx < 25:
            story.append(PageBreak())

    print(f"[PDF] Assembling {len(story)} flowable elements across 25 pages...")
    doc.build(story, canvasmaker=IEEENumberedCanvas)
    print(f"[PDF] Successfully built: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
