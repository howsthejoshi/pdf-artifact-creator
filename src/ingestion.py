"""Load, repair and join the provided bank tables into one analysis-ready dataset."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.paths import PROCESSED_DIR, RAW_DIR

GENDER_MAP = {"M": "Male", "F": "Female", "MALE": "Male", "FEMALE": "Female"}


def _log(decisions: list[dict[str, Any]], step: str, **kwargs: Any) -> None:
    decisions.append({"step": step, **kwargs})


def load_raw() -> dict[str, pd.DataFrame]:
    tables = {
        "customers": pd.read_csv(RAW_DIR / "customers.csv"),
        "loans": pd.read_csv(RAW_DIR / "loan_accounts.csv"),
        "bureau": pd.read_csv(RAW_DIR / "bureau_credit.csv"),
        "collections": pd.read_csv(RAW_DIR / "collections_events.csv"),
    }
    repay_path = RAW_DIR / "repayment_history.csv"
    if repay_path.exists():
        tables["repayment"] = pd.read_csv(repay_path)
    else:
        tables["repayment"] = pd.DataFrame()
    return tables


def clean_customers(df: pd.DataFrame, decisions: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    before = len(out)
    out["customer_id"] = out["customer_id"].astype(str).str.strip()
    out["gender"] = out["gender"].astype(str).str.strip()
    mapped = out["gender"].str.upper().map(GENDER_MAP)
    n_gender = int(mapped.notna().sum() & out["gender"].isin(["M", "F"]).sum()) if False else int(
        (out["gender"].isin(["M", "F"])).sum()
    )
    out["gender"] = mapped.fillna(out["gender"])
    _log(
        decisions,
        "standardise_gender",
        n=n_gender,
        rule="Map M→Male and F→Female; keep Male/Female. Do not drop rows.",
    )
    for col in ["employment_type", "region", "city_tier", "housing_type", "marital_status", "education_level"]:
        out[col] = out[col].astype(str).str.strip()

    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["annual_income"] = pd.to_numeric(out["annual_income"], errors="coerce")
    out["years_employed"] = pd.to_numeric(out["years_employed"], errors="coerce")
    out["num_dependents"] = pd.to_numeric(out["num_dependents"], errors="coerce")
    out["years_at_residence"] = pd.to_numeric(out["years_at_residence"], errors="coerce")
    out["customer_since"] = pd.to_datetime(out["customer_since"], errors="coerce")

    n_ye = int(out["years_employed"].isna().sum())
    out["years_employed"] = out["years_employed"].fillna(
        out.groupby("employment_type")["years_employed"].transform("median")
    )
    n_dep = int(out["num_dependents"].isna().sum())
    out["num_dependents"] = out["num_dependents"].fillna(out["num_dependents"].median()).clip(lower=0)
    n_res = int(out["years_at_residence"].isna().sum())
    out["years_at_residence"] = out["years_at_residence"].fillna(out["years_at_residence"].median())
    _log(decisions, "impute_years_employed", n=n_ye, rule="median within employment_type; rows retained")
    _log(decisions, "impute_num_dependents", n=n_dep, rule="global median; rows retained")
    _log(decisions, "impute_years_at_residence", n=n_res, rule="global median; rows retained")

    out = out.drop_duplicates(subset=["customer_id"], keep="first")
    _log(decisions, "dedupe_customers", dropped=before - len(out), keep="first")
    return out


def clean_loans(df: pd.DataFrame, decisions: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    before = len(out)
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["customer_id"] = out["customer_id"].astype(str).str.strip()
    out["loan_type"] = out["loan_type"].astype(str).str.strip()
    out["loan_purpose"] = out["loan_purpose"].astype(str).str.strip()
    out["disbursal_channel"] = out["disbursal_channel"].astype(str).str.strip()
    out["loan_amount"] = pd.to_numeric(out["loan_amount"], errors="coerce").abs()
    out["interest_rate_pct"] = pd.to_numeric(out["interest_rate_pct"], errors="coerce")
    n_rate = int(out["interest_rate_pct"].isna().sum())
    out["interest_rate_pct"] = out["interest_rate_pct"].fillna(
        out.groupby("loan_type")["interest_rate_pct"].transform("median")
    )
    _log(decisions, "impute_interest_rate", n=n_rate, rule="median within loan_type; rows retained")
    out["tenure_months"] = pd.to_numeric(out["tenure_months"], errors="coerce")
    out["origination_date"] = pd.to_datetime(out["origination_date"], errors="coerce")
    out["collateral_flag"] = pd.to_numeric(out["collateral_flag"], errors="coerce").fillna(0).astype(int)
    out["target_default"] = pd.to_numeric(out["target_default"], errors="coerce")
    bad = out["target_default"].isna() | out["account_id"].eq("") | out["loan_amount"].isna()
    _log(decisions, "drop_unusable_loans", n=int(bad.sum()), rule="only drop if target, id or amount unusable")
    out = out.loc[~bad].copy()
    out["target_default"] = out["target_default"].astype(int)
    out = out.drop_duplicates(subset=["account_id"], keep="first")
    _log(decisions, "dedupe_accounts", dropped=before - len(out), keep="first")
    return out


def clean_bureau(df: pd.DataFrame, decisions: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    out["customer_id"] = out["customer_id"].astype(str).str.strip()
    out["sanctioned_amount"] = pd.to_numeric(out["sanctioned_amount"], errors="coerce").clip(lower=0)
    out["outstanding_amount"] = pd.to_numeric(out["outstanding_amount"], errors="coerce").clip(lower=0)
    out["days_past_due"] = pd.to_numeric(out["days_past_due"], errors="coerce")
    n_dpd = int(out["days_past_due"].isna().sum())
    out["days_past_due"] = out["days_past_due"].fillna(0)
    _log(
        decisions,
        "impute_bureau_dpd",
        n=n_dpd,
        rule="missing days_past_due treated as 0 (not reported / current); rows retained",
    )
    out["date_opened"] = pd.to_datetime(out["date_opened"], errors="coerce")
    out["date_closed"] = pd.to_datetime(out["date_closed"], errors="coerce")
    out["utilisation"] = np.where(
        out["sanctioned_amount"] > 0, out["outstanding_amount"] / out["sanctioned_amount"], np.nan
    )
    out["utilisation"] = out["utilisation"].clip(0, 5)
    return out


def clean_collections(df: pd.DataFrame, decisions: list[dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    out["amount_recovered"] = pd.to_numeric(out["amount_recovered"], errors="coerce").clip(lower=0).fillna(0)
    _log(decisions, "collections_clean", n=len(out), rule="no row drops; recoveries clipped at 0")
    return out


def clean_repayment(df: pd.DataFrame, decisions: list[dict[str, Any]]) -> pd.DataFrame:
    if df.empty:
        _log(
            decisions,
            "repayment_history_missing",
            n=0,
            rule="File was not in the provided pack. Features that would come from monthly EMI/DPD are skipped; bureau DPD and collections are used instead.",
        )
        return df
    out = df.copy()
    out["account_id"] = out["account_id"].astype(str).str.strip()
    out["emi_due"] = pd.to_numeric(out.get("emi_due"), errors="coerce").abs()
    out["emi_paid"] = pd.to_numeric(out.get("emi_paid"), errors="coerce").clip(lower=0)
    out["days_past_due"] = pd.to_numeric(out.get("days_past_due"), errors="coerce").fillna(0).clip(lower=0)
    return out


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    tmp = bureau.copy()
    tmp["is_written_off"] = (tmp["account_status"] == "Written-Off").astype(int)
    tmp["is_active"] = (tmp["account_status"] == "Active").astype(int)
    g = tmp.groupby("customer_id", as_index=False).agg(
        bureau_tradelines=("bureau_id", "count"),
        bureau_limit=("sanctioned_amount", "sum"),
        bureau_balance=("outstanding_amount", "sum"),
        bureau_utilisation=("utilisation", "mean"),
        bureau_max_dpd=("days_past_due", "max"),
        bureau_mean_dpd=("days_past_due", "mean"),
        bureau_writeoffs=("is_written_off", "sum"),
        bureau_active=("is_active", "sum"),
    )
    g["bureau_writeoff_rate"] = np.where(
        g["bureau_tradelines"] > 0, g["bureau_writeoffs"] / g["bureau_tradelines"], 0
    )
    return g


def aggregate_repayment(repay: pd.DataFrame) -> pd.DataFrame:
    if repay.empty:
        return pd.DataFrame(
            columns=[
                "account_id",
                "repay_months",
                "repay_max_dpd",
                "repay_mean_dpd",
                "repay_times_30",
                "repay_paid_ratio",
            ]
        )
    tmp = repay.copy()
    tmp["paid_ratio"] = np.where(tmp["emi_due"] > 0, tmp["emi_paid"] / tmp["emi_due"], 1.0)
    tmp["is_missed"] = tmp["payment_status"].astype(str).str.strip().eq("Missed").astype(int)
    tmp["is_late"] = tmp["payment_status"].astype(str).str.strip().eq("Late").astype(int)
    last = tmp.sort_values(["account_id", "payment_month"]).groupby("account_id").tail(1)[
        ["account_id", "days_past_due", "payment_status", "outstanding_balance"]
    ].rename(
        columns={
            "days_past_due": "repay_last_dpd",
            "payment_status": "repay_last_status",
            "outstanding_balance": "repay_last_balance",
        }
    )
    tmp["is_30"] = (tmp["days_past_due"] >= 30).astype(int)
    tmp["is_90"] = (tmp["days_past_due"] >= 90).astype(int)
    g = tmp.groupby("account_id", as_index=False).agg(
        repay_months=("payment_month", "count"),
        repay_max_dpd=("days_past_due", "max"),
        repay_mean_dpd=("days_past_due", "mean"),
        repay_times_30=("is_30", "sum"),
        repay_times_90=("is_90", "sum"),
        repay_paid_ratio=("paid_ratio", "mean"),
        repay_missed=("is_missed", "sum"),
        repay_late=("is_late", "sum"),
    )
    return g.merge(last, on="account_id", how="left")


def aggregate_collections(coll: pd.DataFrame) -> pd.DataFrame:
    tmp = coll.copy()
    tmp["is_legal"] = tmp["action_type"].isin(["Legal Notice", "Written-Off Referral"]).astype(int)
    tmp["is_no_response"] = (tmp["outcome"] == "No Response").astype(int)
    tmp["is_paid"] = (tmp["outcome"] == "Payment Received").astype(int)
    return tmp.groupby("account_id", as_index=False).agg(
        collections_events=("collection_event_id", "count"),
        collections_recovered=("amount_recovered", "sum"),
        collections_legal=("is_legal", "sum"),
        collections_no_response=("is_no_response", "sum"),
        collections_paid=("is_paid", "sum"),
    )


def build_analysis_set() -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    raw = load_raw()
    customers = clean_customers(raw["customers"], decisions)
    loans = clean_loans(raw["loans"], decisions)
    bureau = clean_bureau(raw["bureau"], decisions)
    coll = clean_collections(raw["collections"], decisions)
    repay = clean_repayment(raw["repayment"], decisions)

    valid_customers = set(customers["customer_id"])
    orphan = ~loans["customer_id"].isin(valid_customers)
    _log(decisions, "orphan_loans", n=int(orphan.sum()), rule="left-join customers; unmatched loans kept with missing demo flags")
    valid_accounts = set(loans["account_id"])
    coll_orphan = ~coll["account_id"].isin(valid_accounts)
    _log(decisions, "orphan_collections", n=int(coll_orphan.sum()), rule="drop collections whose account_id is not in loan_accounts")
    coll = coll.loc[~coll_orphan].copy()
    bureau_ok = bureau["customer_id"].isin(valid_customers)
    _log(decisions, "bureau_unknown_customer", n=int((~bureau_ok).sum()), rule="drop bureau lines without a customer")
    bureau = bureau.loc[bureau_ok].copy()

    _log(
        decisions,
        "join_logic",
        grain="one row per loan account",
        joins=[
            "loan_accounts LEFT JOIN customers USING customer_id",
            "LEFT JOIN bureau features aggregated to customer_id (tradelines, utilisation, max DPD, write-offs)",
            "LEFT JOIN collections features aggregated to account_id",
            "LEFT JOIN repayment features aggregated to account_id IF repayment_history.csv is present",
        ],
    )

    panel = loans.merge(customers, on="customer_id", how="left", validate="m:1")
    panel = panel.merge(aggregate_bureau(bureau), on="customer_id", how="left")
    panel = panel.merge(aggregate_collections(coll), on="account_id", how="left")
    repay_agg = aggregate_repayment(repay)
    if not repay_agg.empty:
        panel = panel.merge(repay_agg, on="account_id", how="left")

    zero_fill = [
        "bureau_tradelines",
        "bureau_limit",
        "bureau_balance",
        "bureau_utilisation",
        "bureau_max_dpd",
        "bureau_mean_dpd",
        "bureau_writeoffs",
        "bureau_active",
        "bureau_writeoff_rate",
        "collections_events",
        "collections_recovered",
        "collections_legal",
        "collections_no_response",
        "collections_paid",
        "repay_months",
        "repay_max_dpd",
        "repay_mean_dpd",
        "repay_times_30",
        "repay_times_90",
        "repay_paid_ratio",
        "repay_missed",
        "repay_late",
        "repay_last_dpd",
        "repay_last_balance",
    ]
    for c in zero_fill:
        if c not in panel.columns:
            panel[c] = 0
        panel[c] = panel[c].fillna(0)
    if "repay_last_status" not in panel.columns:
        panel["repay_last_status"] = "Unknown"
    panel["repay_last_status"] = panel["repay_last_status"].fillna("Unknown")

    panel["income_band"] = pd.cut(
        panel["annual_income"],
        bins=[-1, 25_000, 50_000, 80_000, 120_000, 10_000_000],
        labels=["<25k", "25-50k", "50-80k", "80-120k", "120k+"],
    ).astype(str)
    panel["age_band"] = pd.cut(
        panel["age"], bins=[0, 25, 35, 45, 60, 120], labels=["<=25", "26-35", "36-45", "46-60", "60+"]
    ).astype(str)
    panel["exposure"] = panel["loan_amount"].fillna(0)
    as_of = pd.Timestamp("2026-10-31")
    panel["relationship_years"] = ((as_of - panel["customer_since"]).dt.days / 365.25).clip(lower=0)
    panel["loan_age_months"] = ((as_of - panel["origination_date"]).dt.days / 30.44).clip(lower=0)
    panel["has_bureau"] = (panel["bureau_tradelines"] > 0).astype(int)
    panel["has_collections"] = (panel["collections_events"] > 0).astype(int)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "analysis_panel.parquet", index=False)
    customers.to_parquet(PROCESSED_DIR / "customers_clean.parquet", index=False)
    (PROCESSED_DIR / "data_quality_decisions.json").write_text(json.dumps(decisions, indent=2, default=str))
    return panel


if __name__ == "__main__":
    df = build_analysis_set()
    print(df.shape, float(df["target_default"].mean()))
