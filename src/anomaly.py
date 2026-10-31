"""Peer-relative and IsolationForest anomaly flags for unusual bureau/collections patterns."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.paths import PROCESSED_DIR

ANOMALY_COLS = [
    "bureau_max_dpd",
    "bureau_utilisation",
    "bureau_writeoff_rate",
    "bureau_tradelines",
    "collections_events",
    "collections_legal",
    "collections_no_response",
    "interest_rate_pct",
    "loan_amount",
    "repay_max_dpd",
    "repay_missed",
    "repay_paid_ratio",
]


def run(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    for c in ANOMALY_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    peer_key = df["loan_type"].astype(str) + "|" + df["region"].astype(str)
    reasons = []
    for col in ["bureau_max_dpd", "bureau_utilisation", "bureau_writeoff_rate", "collections_events", "repay_max_dpd", "repay_missed"]:
        med = df.groupby(peer_key)[col].transform("median")
        iqr = df.groupby(peer_key)[col].transform(lambda s: s.quantile(0.75) - s.quantile(0.25)).replace(0, np.nan)
        z = (df[col] - med) / iqr.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0)
        df[f"z_{col}"] = z

    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    df["iforest_score"] = -iso.fit_predict(df[ANOMALY_COLS])  # 1 = inlier mapped? fit_predict -1 anomaly
    # Use decision_function: lower is more anomalous
    df["anomaly_score"] = -iso.decision_function(df[ANOMALY_COLS])

    for _, r in df.iterrows():
        bits = []
        if r["z_bureau_max_dpd"] >= 1.5:
            bits.append(f"bureau max DPD {r['bureau_max_dpd']:.0f} is high vs {r['loan_type']} in {r['region']}")
        if r["z_bureau_utilisation"] >= 1.5:
            bits.append(f"bureau utilisation {r['bureau_utilisation']:.2f} is high vs peers")
        if r["z_bureau_writeoff_rate"] >= 1.5:
            bits.append(f"write-off share {r['bureau_writeoff_rate']:.0%} vs peer book")
        if r["z_collections_events"] >= 1.5:
            bits.append(f"{int(r['collections_events'])} collections events vs peer median")
        if r.get("z_repay_max_dpd", 0) >= 1.5:
            bits.append(f"own-book max DPD {r['repay_max_dpd']:.0f} is high vs {r['loan_type']} in {r['region']}")
        if r.get("z_repay_missed", 0) >= 1.5:
            bits.append(f"{int(r['repay_missed'])} missed EMIs vs peer segment")
        if r["pd_lgbm"] >= 0.35 and r["has_collections"] == 0:
            bits.append("severe model PD with no collections file yet")
        reasons.append("; ".join(bits) if bits else "multivariate isolation-forest outlier vs peer feature mix")

    df["anomaly_reason"] = reasons
    strong = (
        (df["z_bureau_max_dpd"] >= 1.5)
        | (df["z_bureau_writeoff_rate"] >= 1.5)
        | (df["z_repay_max_dpd"] >= 1.5)
        | (df["z_repay_missed"] >= 1.5)
        | (df["z_collections_events"] >= 1.5)
    )
    df["is_anomaly"] = strong & (df["anomaly_score"] >= df["anomaly_score"].quantile(0.90))
    flagged = df.loc[df["is_anomaly"]].copy()
    flagged = flagged.sort_values("anomaly_score", ascending=False).head(200)
    keep = [
        "account_id",
        "customer_id",
        "loan_type",
        "region",
        "employment_type",
        "loan_amount",
        "target_default",
        "pd_lgbm",
        "risk_band",
        "anomaly_score",
        "anomaly_reason",
        "bureau_max_dpd",
        "bureau_utilisation",
        "bureau_writeoff_rate",
        "collections_events",
        "repay_max_dpd",
        "repay_missed",
    ]
    flagged = flagged[keep].sort_values("anomaly_score", ascending=False)
    flagged["review_status"] = "Open"
    flagged.to_parquet(PROCESSED_DIR / "anomalies.parquet", index=False)
    meta = {
        "n_flagged": int(len(flagged)),
        "trigger": "IsolationForest contamination 3% on bureau/collections/loan features, plus peer z>=2 on DPD or write-off rate",
        "false_positive": "A high-limit mortgage with temporarily high utilisation after a top-up, still paying on time.",
        "false_negative": "A recently originated account with only one or two EMI months, still looking current, that will miss from month three — the detector has too little history.",
        "human_step": "Analyst opens the account, confirms or dismisses, and that decision is stored for QA.",
    }
    (PROCESSED_DIR / "anomaly_meta.json").write_text(json.dumps(meta, indent=2))
    return flagged
