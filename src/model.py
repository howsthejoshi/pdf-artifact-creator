"""Logistic baseline + LightGBM default model, metrics, and monitoring notes."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.paths import MODEL_DIR, PROCESSED_DIR

CAT_FEATURES = [
    "loan_type",
    "employment_type",
    "region",
    "city_tier",
    "housing_type",
    "education_level",
    "disbursal_channel",
    "gender",
    "income_band",
    "age_band",
]
NUM_FEATURES = [
    "age",
    "annual_income",
    "years_employed",
    "num_dependents",
    "years_at_residence",
    "loan_amount",
    "interest_rate_pct",
    "tenure_months",
    "collateral_flag",
    "bureau_tradelines",
    "bureau_utilisation",
    "bureau_max_dpd",
    "bureau_mean_dpd",
    "bureau_writeoffs",
    "bureau_writeoff_rate",
    "bureau_active",
    "collections_events",
    "collections_legal",
    "collections_no_response",
    "relationship_years",
    "loan_age_months",
    "repay_max_dpd",
    "repay_mean_dpd",
    "repay_times_30",
    "repay_times_90",
    "repay_paid_ratio",
    "repay_missed",
    "repay_late",
    "repay_last_dpd",
]


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y = y_true[order]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.cumsum(y) / n_pos
    fpr = np.cumsum(1 - y) / n_neg
    return float(np.max(np.abs(tpr - fpr)))


def gini_from_auc(auc: float) -> float:
    return 2 * auc - 1


def _split(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        panel,
        test_size=0.25,
        random_state=42,
        stratify=panel["target_default"],
    )
    return train, test


def train(panel: pd.DataFrame) -> dict:
    df = panel.copy()
    features = CAT_FEATURES + NUM_FEATURES
    for c in NUM_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).fillna("Unknown").astype("category")

    train_df, test_df = _split(df)
    y_train = train_df["target_default"].to_numpy()
    y_test = test_df["target_default"].to_numpy()

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
            ("num", StandardScaler(), NUM_FEATURES),
        ]
    )
    logit = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
            ),
        ]
    )
    logit.fit(train_df[features], y_train)
    logit_scores = logit.predict_proba(test_df[features])[:, 1]
    logit_auc = float(roc_auc_score(y_test, logit_scores))

    gbm = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbosity=-1,
    )
    gbm.fit(train_df[features], y_train, categorical_feature=CAT_FEATURES)
    gbm_scores = gbm.predict_proba(test_df[features])[:, 1]
    gbm_auc = float(roc_auc_score(y_test, gbm_scores))
    gbm_ks = ks_statistic(y_test, gbm_scores)
    gbm_gini = gini_from_auc(gbm_auc)
    logit_ks = ks_statistic(y_test, logit_scores)

    importance = (
        pd.DataFrame({"feature": features, "importance": gbm.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )

    df["pd_lgbm"] = gbm.predict_proba(df[features])[:, 1]
    df["pd_logit"] = logit.predict_proba(df[features])[:, 1]
    df["risk_band"] = pd.cut(
        df["pd_lgbm"], bins=[-0.01, 0.10, 0.20, 0.35, 1.01], labels=["Low", "Medium", "High", "Severe"]
    ).astype(str)

    contrib = _local_drivers(df, gbm, features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": gbm, "features": features, "cat": CAT_FEATURES, "num": NUM_FEATURES}, MODEL_DIR / "lgbm.joblib")
    joblib.dump(logit, MODEL_DIR / "logit.joblib")
    scored = df[["account_id", "customer_id", "pd_lgbm", "pd_logit", "risk_band", "target_default"]].merge(
        contrib, on="account_id", how="left"
    )
    scored.to_parquet(PROCESSED_DIR / "scores.parquet", index=False)

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_default_rate": round(float(y_test.mean()), 4),
        "logistic": {
            "auc": round(logit_auc, 4),
            "ks": round(logit_ks, 4),
            "gini": round(gini_from_auc(logit_auc), 4),
        },
        "lightgbm": {
            "auc": round(gbm_auc, 4),
            "ks": round(gbm_ks, 4),
            "gini": round(gbm_gini, 4),
        },
        "feature_importance": importance.to_dict(orient="records"),
        "monitoring": {
            "psi_features": ["annual_income", "bureau_max_dpd", "loan_amount", "bureau_writeoff_rate"],
            "cadence": "weekly score distribution vs training; monthly default vs predicted PD by band",
            "alerts": "PSI > 0.2 on any listed feature, or observed default outside predicted band CI",
            "champion_challenger": "keep logistic as challenger; promote only if LightGBM Gini lift holds for 2 months",
        },
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    panel_out = panel.merge(
        df[["account_id", "pd_lgbm", "pd_logit", "risk_band"]], on="account_id", how="left"
    )
    panel_out = panel_out.merge(contrib, on="account_id", how="left")
    panel_out.to_parquet(PROCESSED_DIR / "analysis_panel.parquet", index=False)
    return metrics


def _local_drivers(df: pd.DataFrame, model: LGBMClassifier, features: list[str]) -> pd.DataFrame:
    """Cheap account-level drivers: deviation of top numeric features from population median, signed by model gain."""
    imp = pd.Series(model.feature_importances_, index=features)
    num = [f for f in features if f in NUM_FEATURES]
    top = list(imp[num].sort_values(ascending=False).head(6).index)
    med = df[top].median()
    rows = []
    for _, r in df.iterrows():
        parts = []
        for f in top:
            delta = float(r[f] - med[f])
            if abs(delta) < 1e-9:
                continue
            direction = "above" if delta > 0 else "below"
            parts.append(f"{f} {direction} book median ({r[f]:.2f} vs {med[f]:.2f})")
        rows.append({"account_id": r["account_id"], "score_drivers": "; ".join(parts[:4])})
    return pd.DataFrame(rows)
