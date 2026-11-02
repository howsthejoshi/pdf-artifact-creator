# Credit Risk Portfolio Monitor

Working take-home (v2): ingest the five provided bank tables, profile the book, score default risk, and serve a Streamlit monitor with **Ask the Portfolio** (LangGraph + Groq) and **anomaly review**.

| | |
|---|---|
| **GitHub** | https://github.com/howsthejoshi/pdf-artifact-creator |
| **Hosted app** | Streamlit Community Cloud, deployed from that GitHub `main` branch, entry file `app/app.py` |
| **Docs** | [`docs/architecture.pdf`](docs/architecture.pdf) · [`docs/solution_overview.pdf`](docs/solution_overview.pdf) · [`docs/demo_walkthrough_script.pdf`](docs/demo_walkthrough_script.pdf) |

Open the **Cloud app URL** from https://share.streamlit.io (your app’s `*.streamlit.app` link). That is the submission hosted dashboard — not localhost.

Python **3.11** on Streamlit Cloud (required). Locally, 3.11 or 3.12.

## What the app does

- **Portfolio** — KPIs, default by product/region, employment × loan heatmap, predicted risk bands. Sidebar filters apply to every tab.
- **Account** — lookup by `account_id`: PD, risk band, bureau / repayment / collections drivers.
- **Anomalies** — 200-account review queue with reasons; confirm or dismiss.
- **Ask the Portfolio** — LangGraph over `facts.json`. Groq `openai/gpt-oss-20b` when `GROQ_API_KEY` is set; otherwise a local retriever. Off-topic questions are declined. No account rows are sent to an API.

Book snapshot (as-of 2026-10-31): 10,000 customers, 14,685 accounts, **16.1%** observed default, **$601.1m** original-balance exposure.

Champion model: **logistic** test AUC **0.667** / KS **0.247** / Gini **0.334**. LightGBM is the challenger (Gini 0.317).

## Streamlit Community Cloud (reviewers)

Already deployed from GitHub. If you need to recreate it:

1. https://share.streamlit.io → sign in with GitHub.
2. **Create app**
   - Repository: `howsthejoshi/pdf-artifact-creator`
   - Branch: `main`
   - Main file path: **`app/app.py`**
3. **Advanced settings → Python version → 3.11** (do not use 3.13; pandas will compile from source and hang on “in the oven”).
4. **Secrets** (never commit these):

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-20b"
```

5. **Deploy**, then **Reboot** after saving secrets.

If Ask now shows `local_retriever`, the secret is missing — add it and reboot. Do not put `.env` on GitHub.

## Run locally (optional)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
.venv/bin/python scripts/build.py
.venv/bin/streamlit run app/app.py --server.port 8512 --server.address 0.0.0.0
```

Then http://127.0.0.1:8512 for development only.

```bash
cp .env.example .env
# set GROQ_API_KEY
```

## Ask now (LangGraph)

`src/ask_graph.py`: **load facts → classify → retrieve → generate → validate numbers → retry or local fallback**.

Temperature 0. Payload is grouped rates from `data/processed/facts.json` plus the question. GDP / forecasts / PII / raw `ACC…` ids are declined.

Try: “Which loan type has the highest default rate?” (Mortgage ~24.6%) · “Which region has most salaried employees having a business loan?” (South, 207 customers / 214 accounts) · “Default rate by employment and loan type” · “What will GDP be next year?” (declined).

Cross-join questions are planned in `src/cross_query.py` against `analysis_panel.parquet` (filters × group-by, max 40 grouped rows). The LLM never sees account-level rows.

## Where the brief lives

| Brief item | Path |
|---|---|
| Raw tables + dictionary | `data/raw/` |
| Join + data-quality decisions | `src/ingestion.py`, `data/processed/data_quality_decisions.json` |
| Portfolio analytics / insights | `src/analytics.py` |
| Logistic + LightGBM, AUC/KS/Gini | `src/model.py`, `models/metrics.json` |
| Anomaly detection | `src/anomaly.py` |
| Ask the Portfolio | `src/ask_graph.py`, `src/ask_portfolio.py` |
| Dashboard | `app/app.py` |
| Architecture (1 page) | `docs/architecture.pdf` |
| Solution overview (1 page) | `docs/solution_overview.pdf` |
| Demo video script | `docs/demo_walkthrough_script.pdf` |
| Tests | `tests/` |

`repayment_history.csv` is the monthly tape (176,220 rows) in `data/raw/` with the other four CSVs. Sample grounded Q&A: `sample_output/ask_portfolio_example.json`.
