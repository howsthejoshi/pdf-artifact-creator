# Credit Risk Portfolio Monitor

Working take-home for the v2 brief: ingest five bank tables, profile the book, score default risk, and serve a Streamlit monitor with **Ask the Portfolio** and **anomaly review**.

Python **3.12.3**.

## Run (three commands)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/build.py
.venv/bin/streamlit run app/app.py --server.port 8512 --server.address 0.0.0.0
```

Open `http://127.0.0.1:8512`.

## Ask now (LangGraph)

`src/ask_graph.py` is the chatbot. Flow: **load facts → classify → retrieve evidence → LLM generate → number check → retry or local fallback**.

Free LLM: **Groq `llama-3.1-8b-instant`**. Create a key at https://console.groq.com (no card), then:

```bash
cp .env.example .env
# set GROQ_API_KEY=gsk_...
```

If no key is set, the generate node still answers from `data/processed/facts.json` and declines questions the book cannot support. Account-level rows are never sent to an API.

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
| Tests | `tests/` |

`repayment_history.csv` is the 7.6 MB monthly tape (176,220 rows), downloaded into `data/raw/` with the other four CSVs from the pack.

A grounded Q&A sample (for wiring an evaluator script) is in `sample_output/ask_portfolio_example.json`.
