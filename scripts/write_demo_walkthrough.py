"""Write docs/demo_walkthrough_script.pdf — 3–5 min Loom script."""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.paths import DOCS_DIR  # noqa: E402

NAVY = HexColor("#0B2A4A")
TEAL = HexColor("#1B6B8A")
GOLD = HexColor("#C49A3C")
INK = HexColor("#1C2430")
MUTED = HexColor("#5A6570")
PALE = HexColor("#F3F6F9")
PALE_TEAL = HexColor("#E7F2F6")
PALE_GOLD = HexColor("#F8F1DE")


def _styles():
    ss = getSampleStyleSheet()
    ss.add(
        ParagraphStyle(
            "Kicker",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=TEAL,
            spaceAfter=2 * mm,
            tracking=0.6,
        )
    )
    ss.add(
        ParagraphStyle(
            "H1",
            parent=ss["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            textColor=NAVY,
            spaceAfter=3 * mm,
            leading=22,
        )
    )
    ss.add(
        ParagraphStyle(
            "H2",
            parent=ss["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            textColor=NAVY,
            spaceBefore=5 * mm,
            spaceAfter=2.2 * mm,
            leading=15,
        )
    )
    ss.add(
        ParagraphStyle(
            "Body",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.6,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=2.2 * mm,
        )
    )
    ss.add(
        ParagraphStyle(
            "Small",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=8.4,
            leading=11.4,
            textColor=INK,
            spaceAfter=1.4 * mm,
        )
    )
    ss.add(
        ParagraphStyle(
            "Say",
            parent=ss["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9.1,
            leading=12.4,
            textColor=INK,
            leftIndent=3 * mm,
            spaceAfter=2.4 * mm,
            borderPadding=2 * mm,
        )
    )
    ss.add(
        ParagraphStyle(
            "Click",
            parent=ss["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12.2,
            textColor=TEAL,
            spaceBefore=1 * mm,
            spaceAfter=1.2 * mm,
        )
    )
    ss.add(
        ParagraphStyle(
            "Cell",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=INK,
        )
    )
    ss.add(
        ParagraphStyle(
            "CellB",
            parent=ss["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=11,
            textColor=NAVY,
        )
    )
    ss.add(
        ParagraphStyle(
            "HeadWhite",
            parent=ss["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.2,
            leading=11,
            textColor=white,
        )
    )
    ss.add(
        ParagraphStyle(
            "Foot",
            parent=ss["Normal"],
            fontName="Helvetica",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_LEFT,
        )
    )
    return ss


def _header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, A4[1] - 16 * mm, A4[0], 16 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, A4[1] - 16.7 * mm, A4[0], 0.8 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(16 * mm, A4[1] - 10 * mm, "CREDIT RISK PORTFOLIO MONITOR  ·  DEMO VIDEO SCRIPT")
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 16 * mm, A4[1] - 10 * mm, "3–5 minutes  ·  Loom / YouTube unlisted")
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, A4[0], 12 * mm, fill=1, stroke=0)
    canvas.setFillColor(GOLD)
    canvas.rect(0, 12 * mm, A4[0], 0.7 * mm, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(16 * mm, 5 * mm, "Record the hosted Streamlit app  ·  not localhost  ·  1080p")
    canvas.drawRightString(A4[0] - 16 * mm, 5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _card(ss, title, body_paras):
    data = [[Paragraph(f"<b>{title}</b>", ss["CellB"])]]
    for p in body_paras:
        data.append([Paragraph(p, ss["Cell"])])
    t = Table(data, colWidths=[178 * mm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_TEAL),
                ("BACKGROUND", (0, 1), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.4, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build():
    ss = _styles()
    story = []
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Demo walkthrough script", ss["H1"]))
    story.append(
        Paragraph(
            "Use this while recording. Target <b>3.5–4.5 minutes</b> (assignment: 3–5). "
            "Open the <b>live Streamlit URL</b> (your <i>*.streamlit.app</i> link), full screen, 1080p. "
            "Hide bookmarks. Do not record localhost.",
            ss["Body"],
        )
    )

    story.append(Paragraph("0. Before you hit record", ss["H2"]))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Reboot the Streamlit Cloud app and hard-refresh the browser.", ss["Small"])),
                ListItem(Paragraph("Confirm Ask now caption shows <b>groq:openai/gpt-oss-20b</b> (secrets saved). If it says local_retriever, still demo it and say numbers come from facts.json.", ss["Small"])),
                ListItem(Paragraph("GitHub: https://github.com/howsthejoshi/pdf-artifact-creator", ss["Small"])),
                ListItem(Paragraph("Paste box ready for three questions (below).", ss["Small"])),
            ],
            bulletType="bullet",
            leftIndent=8 * mm,
            bulletFontName="Helvetica",
            bulletFontSize=8,
        )
    )
    story.append(Spacer(1, 2 * mm))
    paste = Table(
        [
            [Paragraph("Keep these on a second screen / notes app", ss["CellB"])],
            [Paragraph("<b>Ask 1</b> — Which loan type has the highest default rate?", ss["Cell"])],
            [Paragraph("<b>Ask 2</b> — What is the portfolio default rate?", ss["Cell"])],
            [Paragraph("<b>Ask 3</b> — What will GDP be next year?", ss["Cell"])],
            [Paragraph("Account tab uses the prefilled high-PD account_id — do not hunt IDs in Ask.", ss["Cell"])],
        ],
        colWidths=[178 * mm],
    )
    paste.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PALE_GOLD),
                ("BACKGROUND", (0, 1), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.4, GOLD),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    story.append(paste)

    story.append(Paragraph("Run of show", ss["H2"]))
    timing = [
        ["Clock", "Tab", "What you show"],
        ["0:00–0:20", "Portfolio", "Set the scene — KPIs only"],
        ["0:20–1:15", "Portfolio", "Charts + one filter (uncheck Mortgage, restore)"],
        ["1:15–2:05", "Account", "PD, band, bureau / repay / collections drivers"],
        ["2:05–3:05", "Anomalies", "Trigger / FP / FN + Dismiss (or Confirm)"],
        ["3:05–4:20", "Ask the Portfolio", "Two grounded answers + one decline"],
        ["4:20–4:45", "Portfolio", "Close on the book view"],
    ]
    trows = []
    for i, row in enumerate(timing):
        sty = ss["HeadWhite"] if i == 0 else ss["Cell"]
        trows.append([Paragraph(c, sty) for c in row])
    tbl = Table(trows, colWidths=[32 * mm, 42 * mm, 104 * mm])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), white),
                ("BACKGROUND", (0, 1), (-1, -1), PALE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PALE, white]),
                ("GRID", (0, 0), (-1, -1), 0.3, HexColor("#D5DDE6")),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    # header cells need white bold - Paragraph in header used CellB navy text. Fix header row via extra style
    story.append(tbl)
    story.append(
        Paragraph(
            "If you run long, drop Ask 2 — keep GDP decline. If Groq is slow, wait; do not double-click Ask now.",
            ss["Foot"],
        )
    )

    # Beat 1
    story.append(Paragraph("1.  0:00–0:20  ·  Open and set the scene", ss["H2"]))
    story.append(Paragraph("CLICK — Nothing. Camera on the Portfolio tab. Five KPI tiles visible.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “This is a credit-risk monitor on 10,000 customers and 14,685 loans. Observed default is 16.1%, exposure about 601 million dollars. I’ll show the portfolio view, one account, the anomaly queue, and Ask the Portfolio — both AI features are live on this hosted app.”",
            ss["Say"],
        )
    )

    story.append(Paragraph("2.  0:20–1:15  ·  Portfolio + one filter", ss["H2"]))
    story.append(Paragraph("CLICK — Hover Accounts, Customers, Exposure, Observed default, Mean predicted PD.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Morning view: book size, default rate, and average predicted PD. Default by loan type — Mortgage is the weakest at 24.6% versus Auto around 14%. Exposure by region. The heatmap is employment times product. Risk bands are the model scores.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Sidebar <b>Loan type</b> → uncheck <b>Mortgage</b>. Pause until the account count drops.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Filters apply to every tab. Taking Mortgage out cuts accounts and exposure immediately.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Check <b>Mortgage</b> again so the full book is back.", ss["Click"]))

    story.append(Paragraph("3.  1:15–2:05  ·  Account view", ss["H2"]))
    story.append(Paragraph("CLICK — Tab <b>Account</b>. Leave the prefilled high-PD account_id.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Account lookup. This ID is a high predicted PD. You see PD, risk band, whether it already defaulted, and loan amount.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Scroll to <b>Key drivers</b>, then the three boxes (bureau / repay / collections).", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Drivers are versus the book median — bureau DPD and utilisation, missed EMIs, collections. An analyst would open this before calling the customer. I won’t chase a single ID in Ask; that’s this tab on purpose.”",
            ss["Say"],
        )
    )

    story.append(PageBreak())
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("4.  2:05–3:05  ·  Anomalies (AI feature 2)", ss["H2"]))
    story.append(Paragraph("CLICK — Tab <b>Anomalies</b>. Open the expander “What triggers a flag…” if it is closed.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Second AI feature: anomaly detection. Trigger is IsolationForest plus peer z-scores on bureau DPD, write-offs, missed EMIs. False positive: a large mortgage with a one-off utilisation spike still paying on time. False negative: a two-month-old loan with almost no history. Human step is this queue.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Glance the table. <b>Record a decision</b> → first account. Read the reason line (short).", ss["Click"]))
    story.append(Paragraph("CLICK — <b>Dismiss — false positive</b> (or Confirm). Wait for the success toast.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “I dismiss this as a false positive. The decision is stored; open items stay on the list. That is the human-in-the-loop.”",
            ss["Say"],
        )
    )

    story.append(Paragraph("5.  3:05–4:20  ·  Ask the Portfolio (AI feature 1)", ss["H2"]))
    story.append(Paragraph("CLICK — Tab <b>Ask the Portfolio</b>. Point at the engine caption.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “First AI feature: LangGraph over dashboard facts. Groq only sees grouped rates, not raw customer rows. Generate, then a number check. If a figure isn’t in the tables, we decline or fall back.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Sample button <b>Which loan type has the highest default rate?</b> (or paste + <b>Ask now</b>). Wait.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Mortgage, about 24.6%, with the count and exposure. Source is by loan type — that matches the bar chart we saw.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Paste <b>What is the portfolio default rate?</b> → <b>Ask now</b>.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “16.1 percent, 2,364 defaults on 14,685 accounts. Same KPI as the header.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("CLICK — Paste <b>What will GDP be next year?</b> → <b>Ask now</b>.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “Declined. The graph will not invent a macro forecast. That’s the false-positive control: no number if it isn’t in facts.json. Human step: I still check the Portfolio tab before I would change policy.”",
            ss["Say"],
        )
    )

    story.append(Paragraph("6.  4:20–4:45  ·  Close", ss["H2"]))
    story.append(Paragraph("CLICK — Tab <b>Portfolio</b> so the last frame is the book.", ss["Click"]))
    story.append(
        Paragraph(
            "SAY — “That’s the working monitor: portfolio and account views, anomaly review with confirm or dismiss, and grounded Ask the Portfolio on the hosted link. Code is on GitHub; this Streamlit app stays live for the review window.”",
            ss["Say"],
        )
    )
    story.append(Paragraph("Stop recording. Export 1080p. Loom or YouTube <b>unlisted</b>.", ss["Body"]))

    story.append(Paragraph("Submission email (three links, no zip)", ss["H2"]))
    story.append(
        Paragraph(
            "1. Hosted dashboard — your <b>https://&lt;name&gt;.streamlit.app</b> URL (must stay live 14 days).<br/>"
            "2. GitHub — https://github.com/howsthejoshi/pdf-artifact-creator<br/>"
            "3. Demo video — Loom or YouTube unlisted URL from this recording.",
            ss["Body"],
        )
    )

    story.append(Paragraph("If something breaks while recording", ss["H2"]))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("<b>Ask now missing</b> — you are not on the fourth tab. Click Ask the Portfolio.", ss["Small"])),
                ListItem(Paragraph("<b>Oven / hang</b> — you are not recording a build. Use the already-running Cloud app.", ss["Small"])),
                ListItem(Paragraph("<b>Groq slow</b> — pause and wait; one click only.", ss["Small"])),
                ListItem(Paragraph("<b>Filter empty</b> — restore all loan types / regions in the sidebar.", ss["Small"])),
                ListItem(Paragraph("<b>Went over 5:00</b> — cut hover on KPIs and Ask 2; keep GDP decline.", ss["Small"])),
            ],
            bulletType="bullet",
            leftIndent=8 * mm,
        )
    )

    out = DOCS_DIR / "demo_walkthrough_script.pdf"
    doc = SimpleDocTemplate(
        str(out),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=22 * mm,
        bottomMargin=18 * mm,
        title="Demo walkthrough script — Credit Risk Portfolio Monitor",
        author="Credit Risk Portfolio Monitor",
    )
    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    print("wrote", out)


if __name__ == "__main__":
    build()
