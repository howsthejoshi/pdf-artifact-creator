"""Grounded cross-cuts from the analysis panel (no account rows sent to the LLM)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.paths import PROCESSED_DIR

EMPLOYMENT = {
    "salaried": "Salaried",
    "unemployed": "Unemployed",
    "self-employed": "Self-Employed",
    "self employed": "Self-Employed",
    "contract": "Contract",
    "business owner": "Business Owner",
}
LOAN = {
    "business loan": "Business Loan",
    "personal loan": "Personal Loan",
    "credit card": "Credit Card",
    "auto loan": "Auto Loan",
    "mortgage": "Mortgage",
    "auto": "Auto Loan",
}
REGION = {n.lower(): n for n in ("North", "South", "East", "West", "Central")}
INCOME = {
    "<25k": "<25k",
    "under 25": "<25k",
    "below 25": "<25k",
    "25-50": "25-50k",
    "50-80": "50-80k",
    "80-120": "80-120k",
    "120k": "120k+",
    "120k+": "120k+",
}
AGE = {
    "21-25": "<=25",
    "under 25": "<=25",
    "<=25": "<=25",
    "26-35": "26-35",
    "36-45": "36-45",
    "46-60": "46-60",
    "60+": "60+",
}
TIER = {"tier 1": "Tier 1", "tier 2": "Tier 2", "tier 3": "Tier 3"}
HOUSING = {
    "owned": "Owned",
    "mortgaged": "Mortgaged",
    "rented": "Rented",
    "living with family": "Living with Family",
}
CHANNEL = {
    "branch": "Branch",
    "online": "Online",
    "mobile": "Mobile App",
    "partner": "Partner/DSA",
    "dsa": "Partner/DSA",
}
GENDER = {"female": "Female", "male": "Male"}

DIMS: list[dict[str, Any]] = [
    {"col": "region", "which": ("which region", "by region", "across region", "per region"), "values": REGION},
    {"col": "employment_type", "which": ("which employment", "by employment", "per employment"), "values": EMPLOYMENT},
    {"col": "loan_type", "which": ("which loan type", "which product", "by loan type", "by product"), "values": LOAN},
    {"col": "income_band", "which": ("which income", "by income", "income band"), "values": INCOME},
    {"col": "age_band", "which": ("which age", "by age", "age band"), "values": AGE},
    {"col": "city_tier", "which": ("which city", "by city", "by tier", "city tier"), "values": TIER},
    {"col": "housing_type", "which": ("which housing", "by housing"), "values": HOUSING},
    {"col": "disbursal_channel", "which": ("which channel", "by channel"), "values": CHANNEL},
    {"col": "gender", "which": ("which gender", "by gender"), "values": GENDER},
]


def _match_value(q: str, values: dict[str, str]) -> str | None:
    for needle, label in sorted(values.items(), key=lambda kv: -len(kv[0])):
        if needle in q:
            return label
    return None


@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "analysis_panel.parquet")


def parse_spec(question: str) -> dict[str, Any] | None:
    q = question.lower().strip()
    if not q:
        return None
    filters: dict[str, str] = {}
    group_by: list[str] = []
    for dim in DIMS:
        mentioned_which = any(w in q for w in dim["which"])
        value = _match_value(q, dim["values"])
        if mentioned_which:
            group_by.append(dim["col"])
        elif value:
            filters[dim["col"]] = value
        elif dim["col"] == "employment_type" and "employee" in q and value is None:
            pass
    if "cross" in q or ("by " in q and " and " in q) or " vs " in q:
        for dim in DIMS:
            if dim["col"] not in filters and dim["col"] not in group_by:
                pretty = dim["col"].replace("_", " ")
                last_words = [w.split()[-1] for w in dim["which"] if w.split()[-1] not in {"type", "band"}]
                if pretty in q or any(tok in q for tok in last_words):
                    group_by.append(dim["col"])

    wants_customers = any(w in q for w in ("employee", "customer", "people", "borrower"))
    if any(w in q for w in ("default rate", "highest default", "riskiest", "default %")):
        metric = "default_rate"
    elif "exposure" in q or "book size" in q:
        metric = "exposure"
    elif wants_customers:
        metric = "customers"
    else:
        metric = "accounts"

    is_cross = bool(filters) and bool(group_by)
    is_multi = len(group_by) >= 2
    is_cell = bool(filters) and not group_by and any(w in q for w in ("how many", "number of", "count of", "share of"))
    if not (is_cross or is_multi or is_cell or "cross" in q):
        return None
    if is_cell:
        group_by = []
    # unique preserve order
    seen: set[str] = set()
    gb = []
    for c in group_by:
        if c not in seen:
            seen.add(c)
            gb.append(c)
    return {"filters": filters, "group_by": gb, "metric": metric, "question": q}


def run_cross(question: str) -> dict[str, Any] | None:
    spec = parse_spec(question)
    if spec is None:
        return None
    df = load_panel()
    work = df
    for col, val in spec["filters"].items():
        if col not in work.columns:
            return None
        work = work.loc[work[col] == val]
    if work.empty:
        return {
            "spec": spec,
            "table": [],
            "headline": "No accounts match that combination in the analysis panel.",
            "source": "analysis panel cross-cut",
        }
    metric = spec["metric"]
    gb = spec["group_by"]
    if not gb:
        row = {
            "accounts": int(len(work)),
            "customers": int(work["customer_id"].nunique()),
            "defaults": int(work["target_default"].sum()),
            "default_rate": round(float(work["target_default"].mean()), 4),
            "exposure": round(float(work["exposure"].sum()), 2),
            **spec["filters"],
        }
        table = [_jsonable(row)]
        headline = (
            f"{row['customers']:,} customers / {row['accounts']:,} accounts"
            f" match {', '.join(f'{k}={v}' for k, v in spec['filters'].items())}"
            f" (default {_pct(row['default_rate'])}, exposure {_money(row['exposure'])})."
        )
        return {"spec": spec, "table": table, "headline": headline, "source": "analysis panel filtered cell"}

    grouped = work.groupby(gb, dropna=False, as_index=False).agg(
        accounts=("account_id", "count"),
        customers=("customer_id", "nunique"),
        defaults=("target_default", "sum"),
        default_rate=("target_default", "mean"),
        exposure=("exposure", "sum"),
    )
    grouped["default_rate"] = grouped["default_rate"].round(4)
    grouped["exposure"] = grouped["exposure"].round(2)
    sort_col = metric if metric in grouped.columns else "accounts"
    grouped = grouped.sort_values(sort_col, ascending=False)
    table = [_jsonable(r) for r in grouped.head(40).to_dict(orient="records")]
    top = table[0]
    keys = " × ".join(str(top[c]) for c in gb)
    filt = ", ".join(f"{v}" for v in spec["filters"].values()) or "the book"
    if metric == "default_rate":
        lead = f"{keys} has the highest default rate among {filt}"
        stat = f"{_pct(top['default_rate'])} on {int(top['accounts']):,} accounts"
    elif metric == "exposure":
        lead = f"{keys} has the largest exposure among {filt}"
        stat = f"{_money(top['exposure'])} ({int(top['accounts']):,} accounts, default {_pct(top['default_rate'])})"
    elif metric == "customers":
        lead = f"{keys} has the most among {filt}"
        stat = f"{int(top['customers']):,} customers / {int(top['accounts']):,} accounts (default {_pct(top['default_rate'])})"
    else:
        lead = f"{keys} has the most among {filt}"
        stat = f"{int(top['accounts']):,} accounts / {int(top['customers']):,} customers (default {_pct(top['default_rate'])})"
    headline = f"{lead}: {stat}."
    return {
        "spec": spec,
        "table": table,
        "headline": headline,
        "source": f"analysis panel grouped by {' × '.join(gb)}"
        + (f" filtered to {filt}" if spec["filters"] else ""),
    }


def format_cross(result: dict[str, Any]) -> dict[str, Any]:
    answer = result["headline"]
    gb = (result.get("spec") or {}).get("group_by") or []
    table = result.get("table") or []
    if gb and len(table) > 1:
        bits = []
        for row in table[:5]:
            keys = " / ".join(str(row[c]) for c in gb)
            bits.append(
                f"{keys}: {int(row['customers']):,} customers, {int(row['accounts']):,} accounts, "
                f"default {_pct(float(row['default_rate']))}"
            )
        answer = f"{answer} Ranking — " + "; ".join(bits) + "."
    return {
        "declined": False,
        "answer": answer,
        "source": result["source"],
    }


def _jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, val in row.items():
        if hasattr(val, "item"):
            val = val.item()
        if isinstance(val, float):
            out[key] = float(val)
        elif isinstance(val, int) and not isinstance(val, bool):
            out[key] = int(val)
        else:
            out[key] = val
    return out


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _money(x: float) -> str:
    return f"${x:,.0f}"
