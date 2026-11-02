"""Grounded 'Ask the Portfolio' answers. Numbers come only from facts.json."""

from __future__ import annotations

import json
import re
from typing import Any

from dotenv import load_dotenv

from src.paths import PROCESSED_DIR, ROOT

load_dotenv(ROOT / ".env")


def load_facts() -> dict[str, Any]:
    return json.loads((PROCESSED_DIR / "facts.json").read_text())


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _best(rows: list[dict], key: str = "default_rate") -> dict:
    return max(rows, key=lambda r: r[key])


EMPLOYMENT_ALIASES = {
    "salaried": "Salaried",
    "unemployed": "Unemployed",
    "self-employed": "Self-Employed",
    "self employed": "Self-Employed",
    "contract": "Contract",
    "business owner": "Business Owner",
}
LOAN_ALIASES = {
    "business loan": "Business Loan",
    "personal loan": "Personal Loan",
    "credit card": "Credit Card",
    "auto loan": "Auto Loan",
    "mortgage": "Mortgage",
    "auto": "Auto Loan",
    "business": "Business Loan",
}


def _match_alias(q: str, aliases: dict[str, str]) -> str | None:
    for needle, label in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        if needle in q:
            return label
    return None


def answer_triple_count(question: str, facts: dict[str, Any]) -> dict[str, Any] | None:
    q = question.lower()
    emp = _match_alias(q, EMPLOYMENT_ALIASES)
    loan = _match_alias(q, LOAN_ALIASES)
    rows = facts.get("cross_region_employment_loan_type") or []
    if not rows or "region" not in q or emp is None or loan is None:
        return None
    cells = [r for r in rows if r.get("employment_type") == emp and r.get("loan_type") == loan]
    if not cells:
        return {
            "declined": False,
            "answer": f"No accounts in the book for {emp} with a {loan}.",
            "source": "region × employment_type × loan_type",
        }
    metric = "customers" if any(w in q for w in ("customer", "employee", "people", "borrower")) else "accounts"
    if metric not in cells[0]:
        metric = "accounts"
    top = max(cells, key=lambda r: int(r.get(metric) or 0))
    n_acc = int(top.get("accounts") or 0)
    n_cust = int(top.get("customers") or 0)
    return {
        "declined": False,
        "answer": (
            f"{top['region']} has the most {emp.lower()} {loan.lower()}s: "
            f"{n_cust:,} customers and {n_acc:,} accounts "
            f"(default rate {_fmt_pct(float(top.get('default_rate') or 0))}, "
            f"exposure {_fmt_money(float(top.get('exposure') or 0))})."
        ),
        "source": "analysis panel grouped by region × employment_type × loan_type",
    }


def answer_local(question: str, facts: dict[str, Any]) -> dict[str, Any]:
    q = question.lower().strip()
    declined = {
        "declined": True,
        "answer": "I cannot answer that from the dashboard tables. I only know the joined customers, loans, bureau and collections facts.",
        "source": None,
    }
    if not q:
        return declined

    from src.cross_query import format_cross, run_cross

    try:
        cross = run_cross(question)
    except FileNotFoundError:
        cross = None
    if cross:
        return format_cross(cross)

    triple = answer_triple_count(q, facts)
    if triple:
        return triple

    if any(w in q for w in ["how many customer", "number of customer", "n_customer"]):
        return {
            "declined": False,
            "answer": f"There are {facts['n_customers']:,} customers and {facts['n_accounts']:,} loan accounts.",
            "source": "customers.csv ⋈ loan_accounts.csv",
        }
    if "exposure" in q or "book size" in q:
        return {
            "declined": False,
            "answer": f"Total original-balance exposure is {_fmt_money(facts['total_exposure'])} across {facts['n_accounts']:,} accounts.",
            "source": "loan_accounts.loan_amount",
        }
    if "portfolio default" in q or "overall default" in q or "default rate of the portfolio" in q:
        return {
            "declined": False,
            "answer": (
                f"Portfolio default rate is {_fmt_pct(facts['portfolio_default_rate'])} "
                f"({facts['n_defaults']:,} defaults / {facts['n_accounts']:,} accounts)."
            ),
            "source": "loan_accounts.target_default",
        }

    dim_map = {
        "loan type": "by_loan_type",
        "loan_type": "by_loan_type",
        "product": "by_loan_type",
        "region": "by_region",
        "employment": "by_employment_type",
        "income": "by_income_band",
        "age": "by_age_band",
        "city": "by_city_tier",
        "housing": "by_housing_type",
        "channel": "by_disbursal_channel",
        "education": "by_education_level",
    }
    asks_rate = any(w in q for w in ("default", "rate", "riskiest", "risk"))
    wants_highest = any(w in q for w in ["highest", "worst", "riskiest", "most default"])
    wants_lowest = any(w in q for w in ["lowest", "best", "safest"])
    if asks_rate:
        for word, key in dim_map.items():
            if word in q:
                rows = facts[key]
                row = _best(rows) if not wants_lowest else min(rows, key=lambda r: r["default_rate"])
                label = list(row.values())[0]
                return {
                    "declined": False,
                    "answer": (
                        f"{label} has the {'highest' if not wants_lowest else 'lowest'} default rate at "
                        f"{_fmt_pct(row['default_rate'])} on {row['accounts']:,} accounts "
                        f"({row['defaults']} defaults), exposure {_fmt_money(row['exposure'])}."
                    ),
                    "source": f"analysis panel grouped by {key.replace('by_', '')}",
                }

    if "unemployed" in q:
        row = next(r for r in facts["by_employment_type"] if r["employment_type"] == "Unemployed")
        return {
            "declined": False,
            "answer": (
                f"Unemployed accounts default at {_fmt_pct(row['default_rate'])} "
                f"({row['defaults']} of {row['accounts']})."
            ),
            "source": "customers.employment_type × loan_accounts.target_default",
        }

    if "cross" in q or "employment" in q and "loan" in q:
        top = facts["cross_employment_loan_type"][0]
        return {
            "declined": False,
            "answer": (
                f"Highest employment × product cell with ≥80 accounts: {top['employment_type']} / {top['loan_type']} "
                f"at {_fmt_pct(top['default_rate'])} ({top['accounts']} accounts)."
            ),
            "source": "cross tab employment_type × loan_type",
        }

    if re.search(r"\b(hello|hi)\b", q):
        return {
            "declined": False,
            "answer": "Ask a portfolio question, for example: which loan type has the highest default rate?",
            "source": None,
        }

    # Unknown dimensions / macro / PII
    if any(w in q for w in ["why did", "forecast", "next year", "gdp", "ceo", "ssn", "password"]):
        return declined

    return declined


def ask(question: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    from src.ask_graph import run_ask

    return run_ask(question, history=history)
