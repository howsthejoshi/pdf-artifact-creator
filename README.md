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

## LLM key

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` if you want the hosted-model phrasing. The dashboard also reads Streamlit secrets of the same name.

If no key is set, Ask the Portfolio still answers from `data/processed/facts.json` (grouped rates and totals only) and **declines** questions those facts cannot support. It does not invent numbers. No account-level rows are sent to an LLM when a key is present.

## Where the brief lives

| Brief item | Path |
|---|---|
| Raw tables + dictionary | `data/raw/` |
| Join + data-quality decisions | `src/ingestion.py`, `data/processed/data_quality_decisions.json` |
| Portfolio analytics / insights | `src/analytics.py` |
| Logistic + LightGBM, AUC/KS/Gini | `src/model.py`, `models/metrics.json` |
| Anomaly detection | `src/anomaly.py` |
| Ask the Portfolio | `src/ask_portfolio.py` |
| Dashboard | `app/app.py` |
| Architecture (1 page) | `docs/architecture.pdf` |
| Solution overview (1 page) | `docs/solution_overview.pdf` |
| Tests | `tests/` |

`repayment_history.csv` is the 7.6 MB monthly tape (176,220 rows), downloaded into `data/raw/` with the other four CSVs from the pack.

A grounded Q&A sample (for wiring an evaluator script) is in `sample_output/ask_portfolio_example.json`.
