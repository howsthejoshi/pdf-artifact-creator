"""One-page architecture and solution overview PDFs required by the brief."""

from __future__ import annotations

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas

from src.paths import DOCS_DIR


def _page(path, title: str, lines: list[str]) -> None:
    c = Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 16 * mm
    c.setFont("Times-Bold", 13)
    c.drawString(16 * mm, y, title)
    y -= 7 * mm
    c.setFont("Times-Roman", 8.4)
    for line in lines:
        if y < 14 * mm:
            break
        c.drawString(16 * mm, y, line[:118])
        y -= 4.1 * mm
    c.showPage()
    c.save()


def write_pdfs(facts: dict, metrics: dict, n_flagged: int) -> None:
    lg = metrics["lightgbm"]
    lo = metrics["logistic"]
    insights = facts["insights"]
    top_loan = max(facts["by_loan_type"], key=lambda r: r["default_rate"])
    _page(
        DOCS_DIR / "architecture.pdf",
        "Credit-risk monitor — architecture",
        [
            "Flow  raw CSVs  →  ingestion (pandas)  →  analysis panel (parquet)  →  analytics facts JSON",
            "                 →  sklearn LogisticRegression + LightGBM  →  scores parquet",
            "                 →  IsolationForest anomaly list  →  Streamlit dashboard (app/app.py).",
            "",
            "Tools by stage",
            "  Ingestion: pandas, pyarrow. Gender M/F mapped; missing employment tenure / dependents /",
            "  residence years / interest rate / bureau DPD imputed; rows kept unless target or amount unusable.",
            "  Join grain: one row per loan. loans ⋈ customers (customer_id) ⋈ bureau agg ⋈ repay agg ⋈ collections agg.",
            "  Analytics: pandas group-bys written to data/processed/facts.json (the only numbers the LLM may use).",
            "  Model: sklearn pipeline (OneHot + scaler + logistic, class_weight=balanced) and LightGBM on bureau,",
            "  repayment (max/mean DPD, missed EMIs, paid ratio) and collections counts. joblib in models/.",
            "  Anomaly: IsolationForest (contamination 3%) plus peer z-scores (loan_type × region).",
            "  Dashboard: Streamlit + Plotly. Filters apply to every view.",
            "",
            "Ask the Portfolio (LLM)",
            "  Called from the Ask tab when OPENAI_API_KEY is set. Payload: the question plus a slim facts.json",
            "  (portfolio totals and grouped rates only). No account-level rows, names, or IDs leave the app.",
            "  If the key is missing, a local retriever answers from the same facts and still cites the grouping.",
            "  Unknown questions are declined. Temperature 0.",
            "",
            "Anomaly detection",
            f"  {n_flagged} accounts flagged into a review queue. Analyst confirm/dismiss is stored locally.",
            "",
            "Host: Streamlit process; optional OpenAI Chat Completions (gpt-4o-mini by default).",
        ],
    )
    _page(
        DOCS_DIR / "solution_overview.pdf",
        "Credit-risk monitor — solution overview",
        [
            "How a risk analyst uses it",
            "  Portfolio: KPIs (accounts, exposure, default rate, predicted PD) then cuts by product, region,",
            "  employment, income. Filters in the sidebar are the drill-down. Use this for morning book health.",
            "  Account: paste an account_id. See PD, risk band, bureau/repay drivers, collections, score reasons.",
            "  Anomalies: work the queue, read the why-flagged line, confirm or dismiss. Open items stay listed.",
            "  Ask: type a question such as 'Which loan type has the highest default rate?' and get a cited figure.",
            "",
            "AI feature 1 — Ask the Portfolio",
            "  Trigger: analyst submits a question. Grounding: facts.json only. False positive: a well-formed",
            "  question about a cut we did not pre-aggregate (we decline rather than invent). False negative:",
            "  a phrasing we fail to parse locally when no API key is set. Human step: analyst checks the cited",
            "  grouping against the chart on the Portfolio tab before acting.",
            "AI feature 2 — Anomaly detection",
            "  Trigger: peer-extreme bureau DPD/utilisation/write-off, missed EMIs, or IsolationForest tail.",
            "  False positive: a large mortgage with a one-off utilisation spike that is still paying on time.",
            "  False negative: a two-month-old account that has not yet missed. Human step: confirm/dismiss.",
            "",
            f"Model  train {metrics['n_train']:,} / test {metrics['n_test']:,} accounts, stratified on target_default.",
            f"  Logistic  AUC {lo['auc']:.3f}  KS {lo['ks']:.3f}  Gini {lo['gini']:.3f}",
            f"  LightGBM  AUC {lg['auc']:.3f}  KS {lg['ks']:.3f}  Gini {lg['gini']:.3f}  ← production score",
            f"  Top drivers: {', '.join(i['feature'] for i in metrics['feature_importance'][:6])}.",
            "  Monitor weekly PSI on income, loan amount, bureau DPD, write-off rate; monthly default vs PD band.",
            "",
            f"Top insights  (1) {insights[0]['title']}. (2) {insights[1]['title']}. (3) {insights[2]['title']}.",
            f"  Book default rate {facts['portfolio_default_rate']*100:.1f}%; weakest product {top_loan['loan_type']}"
            f" at {top_loan['default_rate']*100:.1f}%.",
            "",
            "Assumptions  Snapshot as-of 2026-10-31; missing bureau DPD = 0; missing rates imputed by product.",
            "  target_default is the observed outcome, not a through-the-cycle PD. USD amounts as provided.",
            "To production  IT: scheduled CSV/warehouse extract, model registry, Streamlit/SSO, secrets vault,",
            "  audit log of anomaly decisions. Data: monthly repayment tape, collections codes, official default",
            "  definition, vintage and closed-account policy, region hierarchy, and a hold-out recent origination month.",
        ],
    )
