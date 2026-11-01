from src.ask_graph import evidence_for, run_ask, validate_answer_numbers
from src.ask_portfolio import answer_local
from src.ingestion import clean_customers, clean_loans
from src.model import gini_from_auc, ks_statistic
import numpy as np
import pandas as pd


def test_gender_standardised_and_rows_kept():
    raw = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3"],
            "age": [30, 40, 50],
            "gender": ["M", "Female", "F"],
            "marital_status": ["Single", "Married", "Single"],
            "education_level": ["High School", "Bachelor's", "High School"],
            "employment_type": ["Salaried", "Salaried", "Unemployed"],
            "occupation_sector": ["Retail", "Retail", "Retail"],
            "years_employed": [2.0, np.nan, 0.0],
            "annual_income": [40000, 55000, 12000],
            "region": ["North", "South", "East"],
            "city_tier": ["Tier 1", "Tier 1", "Tier 2"],
            "num_dependents": [np.nan, 1, 0],
            "housing_type": ["Rented", "Owned", "Rented"],
            "years_at_residence": [1.0, np.nan, 2.0],
            "customer_since": ["2020-01-01", "2019-06-01", "2021-01-01"],
        }
    )
    decisions = []
    out = clean_customers(raw, decisions)
    assert set(out["gender"]) <= {"Male", "Female"}
    assert len(out) == 3
    assert out["years_employed"].isna().sum() == 0


def test_interest_rate_imputed_not_dropped():
    raw = pd.DataFrame(
        {
            "account_id": ["A1", "A2"],
            "customer_id": ["C1", "C1"],
            "loan_type": ["Personal Loan", "Personal Loan"],
            "loan_purpose": ["Other", "Other"],
            "loan_amount": [1000, 2000],
            "interest_rate_pct": [10.0, np.nan],
            "tenure_months": [12, 12],
            "origination_date": ["2022-01-01", "2022-02-01"],
            "disbursal_channel": ["Online", "Online"],
            "collateral_flag": [0, 0],
            "target_default": [0, 1],
        }
    )
    out = clean_loans(raw, [])
    assert len(out) == 2
    assert out["interest_rate_pct"].isna().sum() == 0


def test_ks_and_gini():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert ks_statistic(y, s) == 1.0
    assert abs(gini_from_auc(0.9) - 0.8) < 1e-9


def test_ask_uses_facts_and_declines_unknown():
    facts = {
        "n_customers": 10,
        "n_accounts": 12,
        "n_defaults": 2,
        "portfolio_default_rate": 0.1667,
        "total_exposure": 1000.0,
        "by_loan_type": [
            {
                "loan_type": "Personal Loan",
                "accounts": 8,
                "defaults": 2,
                "exposure": 800,
                "default_rate": 0.25,
                "avg_income": 1,
            },
            {
                "loan_type": "Mortgage",
                "accounts": 4,
                "defaults": 0,
                "exposure": 200,
                "default_rate": 0.0,
                "avg_income": 1,
            },
        ],
        "by_region": [],
        "by_employment_type": [],
        "by_income_band": [],
        "by_age_band": [],
        "by_city_tier": [],
        "by_housing_type": [],
        "by_disbursal_channel": [],
        "by_education_level": [],
        "cross_employment_loan_type": [],
    }
    hit = answer_local("Which loan type has the highest default rate?", facts)
    assert hit["declined"] is False
    assert "25.0%" in hit["answer"]
    assert "Personal Loan" in hit["answer"]
    miss = answer_local("What will GDP be next year?", facts)
    assert miss["declined"] is True


def test_graph_declines_gdp_and_answers_loan_type():
    hit = run_ask("Which loan type has the highest default rate?")
    assert hit["declined"] is False
    assert "Mortgage" in hit["answer"]
    assert any(tok in hit["answer"] for tok in ("24.6%", "24.65%", "24.7%"))
    miss = run_ask("What will GDP be next year?")
    assert miss["declined"] is True
    assert miss["engine"] == "rules"


def test_number_validator_rejects_hallucinated_rate():
    evidence = evidence_for("loan type", {"n_accounts": 10, "portfolio_default_rate": 0.161, "by_loan_type": [{"loan_type": "Mortgage", "default_rate": 0.2465, "accounts": 1432, "defaults": 353, "exposure": 100.0, "avg_income": 1}]})
    assert validate_answer_numbers("Mortgage defaults at 24.6% on 1,432 accounts.", evidence) is None
    assert validate_answer_numbers("Mortgage defaults at 91.0%.", evidence) is not None
