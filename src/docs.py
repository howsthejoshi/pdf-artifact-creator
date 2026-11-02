"""One-page architecture and solution overview PDFs required by the brief."""

from __future__ import annotations

from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from src.paths import DOCS_DIR, PROCESSED_DIR, MODEL_DIR

NAVY = HexColor("#0B2A4A")
TEAL = HexColor("#1B6B8A")
GOLD = HexColor("#C49A3C")
INK = HexColor("#1C2430")
MUTED = HexColor("#5A6570")
RULE = HexColor("#D5DDE6")
PALE = HexColor("#F3F6F9")
PALE_TEAL = HexColor("#E7F2F6")
PALE_GOLD = HexColor("#F8F1DE")
BOX = HexColor("#FFFFFF")
RED = HexColor("#8B2E2E")


def _wrap(c: Canvas, text: str, font: str, size: float, max_w: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = w if not cur else f"{cur} {w}"
        if stringWidth(trial, font, size) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _header(c: Canvas, width: float, height: float, kicker: str, title: str, sub: str) -> float:
    c.setFillColor(NAVY)
    c.rect(0, height - 28 * mm, width, 28 * mm, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, height - 28.8 * mm, width, 1.1 * mm, fill=1, stroke=0)
    c.setFillColor(HexColor("#8FB8C9"))
    c.setFont("Helvetica", 7.2)
    c.drawString(14 * mm, height - 9 * mm, kicker.upper())
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(14 * mm, height - 16.5 * mm, title)
    c.setFont("Helvetica", 8)
    c.setFillColor(HexColor("#D5E4EC"))
    c.drawString(14 * mm, height - 22.5 * mm, sub)
    return height - 34 * mm


def _section(c: Canvas, x: float, y: float, title: str) -> float:
    c.setFillColor(TEAL)
    c.roundRect(x, y - 1.2 * mm, 2.2 * mm, 4.2 * mm, 0.4 * mm, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 4 * mm, y, title)
    return y - 5.2 * mm


def _para(c: Canvas, x: float, y: float, text: str, width: float, size: float = 7.4, leading: float = 9.4, font: str = "Helvetica") -> float:
    c.setFont(font, size)
    c.setFillColor(INK)
    for line in _wrap(c, text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def _box(c: Canvas, x: float, y: float, w: float, h: float, fill=BOX, stroke=RULE) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.4)
    c.roundRect(x, y, w, h, 2 * mm, fill=1, stroke=1)


def write_architecture(facts: dict, n_flagged: int) -> None:
    path = DOCS_DIR / "architecture.pdf"
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    y = _header(
        c,
        width,
        height,
        "Credit risk portfolio analytics  ·  take-home v2",
        "Architecture",
        "Raw tables → clean panel → model & AI features → Streamlit (Community Cloud). One page.",
    )

    # Flow boxes
    stages = [
        ("1. Raw pack", "5 CSVs + dictionary"),
        ("2. Ingest", "pandas / pyarrow"),
        ("3. Features", "bureau · repay · coll."),
        ("4. Score", "sklearn + LightGBM"),
        ("5. Serve", "Streamlit + Plotly"),
    ]
    box_w = 33.2 * mm
    gap = 4.6 * mm
    x0 = 14 * mm
    flow_y = y - 18 * mm
    for i, (t, s) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        _box(c, x, flow_y, box_w, 16 * mm, PALE_TEAL, TEAL)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(x + 2.2 * mm, flow_y + 9.6 * mm, t)
        c.setFont("Helvetica", 6.5)
        c.setFillColor(MUTED)
        c.drawString(x + 2.2 * mm, flow_y + 4.2 * mm, s)
        if i < len(stages) - 1:
            c.setStrokeColor(TEAL)
            c.setFillColor(TEAL)
            c.setLineWidth(1)
            ax = x + box_w + 0.6 * mm
            c.line(ax, flow_y + 8 * mm, ax + gap - 1.8 * mm, flow_y + 8 * mm)
            c.drawCentredString(ax + gap - 1.4 * mm, flow_y + 7.1 * mm, "›")
    y = flow_y - 6 * mm

    left_x = 14 * mm
    col_w = 88 * mm
    right_x = 110 * mm
    right_w = 86 * mm

    y_l = _section(c, left_x, y, "Pipeline and tools")
    y_l = _para(
        c,
        left_x,
        y_l,
        "Grain: one row per loan account. loan_accounts LEFT JOIN customers on customer_id; "
        "LEFT JOIN bureau features rolled to customer; repayment and collections rolled to account_id.",
        col_w,
    )
    y_l -= 1.2 * mm
    bullets = [
        "Ingest (pandas, pyarrow): map gender M/F→Male/Female; impute years_employed (by employment), dependents, residence years, interest rate (by product), bureau DPD missing→0. Drop only unusable target/amount. Deduplicate ids.",
        "Analytics: group-bys → data/processed/facts.json. This JSON is the only numeric payload the LLM may see.",
        "Model: sklearn OneHot+scaler+LogisticRegression (champion on test Gini) and LightGBM on bureau utilisation/DPD/write-offs, repayment DPD/missed EMI/paid ratio, collections counts. joblib in models/.",
        "Anomaly: IsolationForest on bureau/repay/collections mix plus peer z-scores (loan_type × region). Top 200 review queue.",
        "Dashboard: Streamlit + Plotly. Sidebar filters apply to every tab. Hosted on Streamlit Community Cloud (Python 3.11).",
    ]
    for b in bullets:
        c.setFillColor(TEAL)
        c.circle(left_x + 1.2 * mm, y_l + 1.4 * mm, 0.9 * mm, fill=1, stroke=0)
        y_l = _para(c, left_x + 4 * mm, y_l, b, col_w - 4 * mm, size=7.15, leading=9.1)
        y_l -= 1.6 * mm

    y_r = _section(c, right_x, y, "Ask the Portfolio — LangGraph")
    _box(c, right_x, y_r - 52 * mm, right_w, 51 * mm, PALE)
    y_r -= 5 * mm
    y_r = _para(
        c,
        right_x + 3 * mm,
        y_r,
        "Graph: load facts → classify → retrieve evidence → generate → validate numbers → retry (max 2) or local fallback.",
        right_w - 6 * mm,
        size=7.2,
        leading=9.2,
    )
    y_r -= 1.2 * mm
    y_r = _para(
        c,
        right_x + 3 * mm,
        y_r,
        "Where the LLM is called: generate node only, temperature 0, JSON response. Provider order: Groq openai/gpt-oss-20b (GROQ_API_KEY / Streamlit secrets) → Gemini → OpenAI. If no key: local retriever, still cited.",
        right_w - 6 * mm,
        size=7.2,
        leading=9.2,
    )
    y_r -= 1.2 * mm
    c.setFont("Helvetica-Bold", 7.1)
    c.setFillColor(NAVY)
    c.drawString(right_x + 3 * mm, y_r, "Sent to the LLM")
    y_r -= 3.6 * mm
    y_r = _para(
        c,
        right_x + 3 * mm,
        y_r,
        "Question + last few chat turns + slim facts.json (book totals and grouped default rates). Never account_id, customer_id, names, or raw rows.",
        right_w - 6 * mm,
        size=7.2,
        leading=9.2,
    )
    y_r -= 1.4 * mm
    c.setFont("Helvetica-Bold", 7.1)
    c.setFillColor(NAVY)
    c.drawString(right_x + 3 * mm, y_r, "Guardrails")
    y_r -= 3.6 * mm
    y_r = _para(
        c,
        right_x + 3 * mm,
        y_r,
        "Classify declines GDP/forecasts/PII/ACC ids. Validate rejects any % or $ not in evidence and falls back to local facts.",
        right_w - 6 * mm,
        size=7.2,
        leading=9.2,
    )

    y_r -= 6 * mm
    y_r = _section(c, right_x, y_r, f"Anomaly detection — {n_flagged} accounts")
    y_r = _para(
        c,
        right_x,
        y_r,
        "sklearn IsolationForest + peer z on max DPD, write-off rate, missed EMIs, collections intensity. Each flag has a reason string. Confirm / Dismiss is stored for QA (session/local JSON). Not sent to the LLM.",
        right_w,
        size=7.2,
        leading=9.2,
    )

    # Footer strip
    c.setFillColor(NAVY)
    c.rect(0, 0, width, 12 * mm, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, 12 * mm, width, 0.7 * mm, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica", 6.8)
    c.drawString(
        14 * mm,
        5 * mm,
        "Host  Streamlit Community Cloud  ·  secrets: GROQ_API_KEY  ·  artefacts: data/processed/*.parquet, models/*.joblib  ·  no secrets in git",
    )
    c.showPage()
    c.save()


def write_overview(facts: dict, metrics: dict, n_flagged: int) -> None:
    path = DOCS_DIR / "solution_overview.pdf"
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    y = _header(
        c,
        width,
        height,
        "Credit risk portfolio analytics  ·  take-home v2",
        "Solution overview",
        f"Book snapshot as-of {facts.get('as_of', '2026-10-31')}  ·  "
        f"{facts['n_customers']:,} customers  ·  {facts['n_accounts']:,} accounts  ·  "
        f"observed default {facts['portfolio_default_rate']*100:.1f}%  ·  "
        f"exposure ${facts['total_exposure']/1e6:.1f}m",
    )

    y = _section(c, 14 * mm, y, "How a risk analyst uses the dashboard")
    views = [
        ("Portfolio", "Morning book health: KPIs, default by product/region, employment × loan heatmap, predicted risk bands. Sidebar filters are the drill-down."),
        ("Account", "Paste account_id. PD, risk band, observed default, bureau / repayment / collections drivers. Use this before calling a customer."),
        ("Anomalies", f"Work the {n_flagged}-account queue. Read why it flagged. Confirm (collections) or dismiss (false positive). Open items remain listed."),
        ("Ask", "Natural-language Q&A. Example: “Which loan type has the highest default rate?” Answer cites the grouping. Check the Portfolio chart before changing policy."),
    ]
    vw = 43.5 * mm
    vh = 28 * mm
    for i, (t, body) in enumerate(views):
        x = 14 * mm + (i % 4) * (vw + 3.2 * mm)
        _box(c, x, y - vh, vw, vh, PALE if i % 2 == 0 else PALE_TEAL)
        c.setFillColor(TEAL)
        c.setFont("Helvetica-Bold", 7.6)
        c.drawString(x + 2.4 * mm, y - 5.5 * mm, t)
        _para(c, x + 2.4 * mm, y - 10 * mm, body, vw - 4.6 * mm, size=6.5, leading=8.2)
    y = y - vh - 6 * mm

    y = _section(c, 14 * mm, y, "Mandatory AI features")
    # two cards
    card_h = 38 * mm
    card_w = 90 * mm
    _box(c, 14 * mm, y - card_h, card_w, card_h, PALE_TEAL, TEAL)
    _box(c, 108 * mm, y - card_h, card_w, card_h, PALE_GOLD, GOLD)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(NAVY)
    c.drawString(17 * mm, y - 5.5 * mm, "1  Ask the Portfolio (LangGraph + Groq)")
    c.drawString(111 * mm, y - 5.5 * mm, "2  Anomaly detection")
    ask_txt = (
        "Trigger: Ask now on a natural-language question. "
        "False positive: a fluent rate for a cut that is not in facts.json — the validate node rejects it and we decline or fall back. "
        "False negative: an awkwardly phrased question the local retriever misses if Groq is down. "
        "Human step: read the cited grouping on the Portfolio tab before acting. No raw rows leave the app."
    )
    anom_txt = (
        "Trigger: IsolationForest tail and/or peer-extreme bureau DPD, write-off share, missed EMIs, collections intensity. "
        "False positive: a large mortgage with a one-off utilisation spike that is still paying on time. "
        "False negative: a two-month-old account with too little EMI history. "
        "Human step: confirm or dismiss on the review list; decision stored for QA."
    )
    _para(c, 17 * mm, y - 10.5 * mm, ask_txt, card_w - 6 * mm, size=6.7, leading=8.4)
    _para(c, 111 * mm, y - 10.5 * mm, anom_txt, card_w - 6 * mm, size=6.7, leading=8.4)
    y = y - card_h - 6 * mm

    y = _section(c, 14 * mm, y, "Model results (stratified 75/25 on target_default)")
    lo, lg = metrics["logistic"], metrics["lightgbm"]
    # metric pills
    pills = [
        ("Champion  logistic", f"AUC {lo['auc']:.3f}   KS {lo['ks']:.3f}   Gini {lo['gini']:.3f}"),
        ("Challenger  LightGBM", f"AUC {lg['auc']:.3f}   KS {lg['ks']:.3f}   Gini {lg['gini']:.3f}"),
        ("Sample", f"train {metrics['n_train']:,}   test {metrics['n_test']:,}   test DR {metrics['test_default_rate']*100:.1f}%"),
    ]
    pw = 60 * mm
    for i, (lab, val) in enumerate(pills):
        x = 14 * mm + i * (pw + 3 * mm)
        _box(c, x, y - 14 * mm, pw, 14 * mm, PALE)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 6.4)
        c.drawString(x + 2.5 * mm, y - 4.8 * mm, lab.upper())
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x + 2.5 * mm, y - 10.5 * mm, val)
    y = y - 16.5 * mm
    drivers = ", ".join(i["feature"] for i in metrics["feature_importance"][:7])
    y = _para(
        c,
        14 * mm,
        y,
        f"LightGBM gain leaders (for explanation, not selection): {drivers}. "
        "Serve logistic as the production ranker until LightGBM beats it on Gini for two consecutive months. "
        "Monitor weekly PSI on income, loan amount, bureau DPD, write-off rate; monthly observed default vs PD band.",
        182 * mm,
        size=7.15,
        leading=9.1,
    )
    y -= 2.4 * mm

    y = _section(c, 14 * mm, y, "Top three portfolio insights")
    insights = facts["insights"][:3]
    for i, item in enumerate(insights, 1):
        c.setFillColor(GOLD)
        c.roundRect(14 * mm, y - 1.1 * mm, 4.4 * mm, 4.4 * mm, 0.8 * mm, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 6.6)
        c.drawCentredString(16.2 * mm, y, str(i))
        c.setFont("Helvetica-Bold", 7.4)
        c.drawString(20.5 * mm, y, item["title"])
        y -= 3.8 * mm
        y = _para(c, 20.5 * mm, y, item["so_what"], 175 * mm, size=7.05, leading=8.9)
        y -= 1.8 * mm

    y = _section(c, 14 * mm, y, "Assumptions  ·  what we need from bank IT and data to go live")
    y = _para(
        c,
        14 * mm,
        y,
        "Assumptions: snapshot as-of 2026-10-31; USD as provided; missing bureau DPD treated as 0 (not reported / current); "
        "missing interest rates imputed by product median; target_default is observed default, not a through-the-cycle PD; "
        "logistic is champion because it outranks LightGBM on this tape (Gini 0.334 vs 0.317). "
        "IT: scheduled warehouse extract, model registry, SSO in front of Streamlit, secrets vault for Groq, "
        "immutable audit of anomaly confirm/dismiss. Data: official default definition, vintage/closed-account policy, "
        "region hierarchy, monthly repayment tape, and a locked recent-origination hold-out month.",
        182 * mm,
        size=7.1,
        leading=9.0,
    )

    c.setFillColor(NAVY)
    c.rect(0, 0, width, 11 * mm, fill=1, stroke=0)
    c.setFillColor(GOLD)
    c.rect(0, 11 * mm, width, 0.7 * mm, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica", 6.7)
    c.drawString(
        14 * mm,
        4.6 * mm,
        "One page  ·  no slide deck  ·  dashboard: app/app.py  ·  graph: src/ask_graph.py  ·  decisions: data/processed/data_quality_decisions.json",
    )
    c.showPage()
    c.save()


def write_pdfs(facts: dict, metrics: dict, n_flagged: int) -> None:
    write_architecture(facts, n_flagged)
    write_overview(facts, metrics, n_flagged)


if __name__ == "__main__":
    import json

    facts = json.loads((PROCESSED_DIR / "facts.json").read_text())
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    n_flagged = int(json.loads((PROCESSED_DIR / "anomaly_meta.json").read_text()).get("n_flagged", 200))
    write_pdfs(facts, metrics, n_flagged)
    print("wrote", DOCS_DIR / "architecture.pdf", DOCS_DIR / "solution_overview.pdf")
