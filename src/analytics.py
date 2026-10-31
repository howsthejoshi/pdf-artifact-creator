"""Portfolio profiling and grounded facts for the dashboard and LLM."""

from __future__ import annotations

import json

import pandas as pd

from src.paths import PROCESSED_DIR


def _rate_table(df: pd.DataFrame, col: str) -> list[dict]:
    g = (
        df.groupby(col, dropna=False)
        .agg(
            accounts=("account_id", "count"),
            defaults=("target_default", "sum"),
            exposure=("exposure", "sum"),
            default_rate=("target_default", "mean"),
            avg_income=("annual_income", "mean"),
        )
        .reset_index()
        .sort_values("default_rate", ascending=False)
    )
    g["default_rate"] = g["default_rate"].round(4)
    g["exposure"] = g["exposure"].round(2)
    g["avg_income"] = g["avg_income"].round(2)
    g["defaults"] = g["defaults"].astype(int)
    return g.to_dict(orient="records")


def build_facts(panel: pd.DataFrame) -> dict:
    n = len(panel)
    facts = {
        "as_of": "2026-10-31",
        "source_tables": [
            "customers.csv",
            "loan_accounts.csv",
            "bureau_credit.csv",
            "collections_events.csv",
            "repayment_history.csv",
        ],
        "n_accounts": int(n),
        "n_customers": int(panel["customer_id"].nunique()),
        "n_defaults": int(panel["target_default"].sum()),
        "portfolio_default_rate": round(float(panel["target_default"].mean()), 4),
        "total_exposure": round(float(panel["exposure"].sum()), 2),
        "avg_loan_amount": round(float(panel["loan_amount"].mean()), 2),
        "avg_income": round(float(panel["annual_income"].mean()), 2),
        "by_loan_type": _rate_table(panel, "loan_type"),
        "by_region": _rate_table(panel, "region"),
        "by_employment_type": _rate_table(panel, "employment_type"),
        "by_income_band": _rate_table(panel, "income_band"),
        "by_age_band": _rate_table(panel, "age_band"),
        "by_city_tier": _rate_table(panel, "city_tier"),
        "by_housing_type": _rate_table(panel, "housing_type"),
        "by_disbursal_channel": _rate_table(panel, "disbursal_channel"),
        "by_education_level": _rate_table(panel, "education_level"),
    }
    cross = (
        panel.groupby(["employment_type", "loan_type"], dropna=False)
        .agg(
            accounts=("account_id", "count"),
            default_rate=("target_default", "mean"),
            exposure=("exposure", "sum"),
        )
        .reset_index()
    )
    cross = cross[cross["accounts"] >= 80].sort_values("default_rate", ascending=False)
    facts["cross_employment_loan_type"] = json.loads(
        cross.round(4).to_json(orient="records")
    )
    income_emp = (
        panel.groupby(["income_band", "employment_type"], dropna=False)
        .agg(accounts=("account_id", "count"), default_rate=("target_default", "mean"))
        .reset_index()
    )
    facts["cross_income_employment"] = json.loads(
        income_emp[income_emp["accounts"] >= 80].round(4).to_json(orient="records")
    )
    return facts


def insights(facts: dict) -> list[dict]:
    loan = sorted(facts["by_loan_type"], key=lambda r: -r["default_rate"])
    emp = sorted(facts["by_employment_type"], key=lambda r: -r["default_rate"])
    inc = sorted(facts["by_income_band"], key=lambda r: -r["default_rate"])
    cross = facts["cross_employment_loan_type"]
    region = sorted(facts["by_region"], key=lambda r: -r["default_rate"])
    top_cross = cross[0] if cross else {}
    return [
        {
            "title": f"{loan[0]['loan_type']} carries the highest default rate",
            "so_what": (
                f"{loan[0]['loan_type']} defaults at {loan[0]['default_rate']*100:.1f}% vs "
                f"{loan[-1]['loan_type']} at {loan[-1]['default_rate']*100:.1f}%. "
                "Price and limit unsecured products tighter; keep secured book as the ballast."
            ),
        },
        {
            "title": f"{emp[0]['employment_type']} customers are the weakest employment segment",
            "so_what": (
                f"Default rate {emp[0]['default_rate']*100:.1f}% versus "
                f"{emp[-1]['employment_type']} at {emp[-1]['default_rate']*100:.1f}%. "
                "Policy should require extra bureau quality or a lower LTI for this segment."
            ),
        },
        {
            "title": f"Income band {inc[0]['income_band']} has the highest default rate",
            "so_what": (
                f"{inc[0]['default_rate']*100:.1f}% default versus {inc[-1]['income_band']} at "
                f"{inc[-1]['default_rate']*100:.1f}%. Affordability cuts belong in origination, not only collections."
            ),
        },
        {
            "title": "Cross-cut: employment × loan type concentrates risk",
            "so_what": (
                f"{top_cross.get('employment_type')} {top_cross.get('loan_type')} prints "
                f"{float(top_cross.get('default_rate', 0))*100:.1f}% default on "
                f"{int(top_cross.get('accounts', 0))} accounts. That cell should be on the watchlist, not averaged away."
            ),
        },
        {
            "title": f"{region[0]['region']} is the weakest region",
            "so_what": (
                f"{region[0]['region']} defaults at {region[0]['default_rate']*100:.1f}% vs "
                f"{region[-1]['region']} at {region[-1]['default_rate']*100:.1f}%. "
                "Check local underwriting and collections coverage before raising regional limits."
            ),
        },
    ]


def run(panel: pd.DataFrame) -> dict:
    facts = build_facts(panel)
    facts["insights"] = insights(facts)
    (PROCESSED_DIR / "facts.json").write_text(json.dumps(facts, indent=2))
    return facts
