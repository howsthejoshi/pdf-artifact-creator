"""Streamlit credit-risk monitoring dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.ask_portfolio import ask  # noqa: E402
from src.paths import PROCESSED_DIR  # noqa: E402

st.set_page_config(page_title="Credit Risk Portfolio Monitor", layout="wide")

DECISIONS_PATH = PROCESSED_DIR / "anomaly_decisions.json"


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    path = PROCESSED_DIR / "analysis_panel.parquet"
    if not path.exists():
        st.error("Build artefacts missing. Run: python scripts/build.py")
        st.stop()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_anomalies() -> pd.DataFrame:
    path = PROCESSED_DIR / "anomalies.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((PROCESSED_DIR / name).read_text()) if (PROCESSED_DIR / name).exists() else {}


def load_decisions() -> dict:
    if DECISIONS_PATH.exists():
        return json.loads(DECISIONS_PATH.read_text())
    return {}


def save_decisions(d: dict) -> None:
    DECISIONS_PATH.write_text(json.dumps(d, indent=2))


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    loan_types = sorted(df["loan_type"].dropna().unique())
    regions = sorted(df["region"].dropna().unique())
    emps = sorted(df["employment_type"].dropna().unique())
    sel_lt = st.sidebar.multiselect("Loan type", loan_types, default=loan_types)
    sel_rg = st.sidebar.multiselect("Region", regions, default=regions)
    sel_em = st.sidebar.multiselect("Employment", emps, default=emps)
    out = df[df["loan_type"].isin(sel_lt) & df["region"].isin(sel_rg) & df["employment_type"].isin(sel_em)]
    st.sidebar.caption(f"{len(out):,} of {len(df):,} accounts")
    return out


def kpi_row(df: pd.DataFrame) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accounts", f"{len(df):,}")
    c2.metric("Customers", f"{df['customer_id'].nunique():,}")
    c3.metric("Exposure", f"${df['exposure'].sum()/1e6:.1f}m")
    c4.metric("Observed default", f"{df['target_default'].mean()*100:.1f}%")
    c5.metric("Mean predicted PD", f"{df['pd_lgbm'].mean()*100:.1f}%")


def portfolio_view(df: pd.DataFrame, facts: dict) -> None:
    st.subheader("Portfolio view")
    kpi_row(df)
    left, right = st.columns(2)
    by_lt = df.groupby("loan_type", as_index=False).agg(
        default_rate=("target_default", "mean"), exposure=("exposure", "sum"), accounts=("account_id", "count")
    )
    fig = px.bar(by_lt, x="loan_type", y="default_rate", hover_data=["accounts", "exposure"], title="Default rate by loan type")
    fig.update_yaxes(tickformat=".0%")
    left.plotly_chart(fig, use_container_width=True)
    by_rg = df.groupby("region", as_index=False).agg(default_rate=("target_default", "mean"), exposure=("exposure", "sum"))
    fig2 = px.bar(by_rg, x="region", y="exposure", color="default_rate", title="Exposure by region (colour = default rate)")
    right.plotly_chart(fig2, use_container_width=True)

    mid1, mid2 = st.columns(2)
    cross = (
        df.groupby(["employment_type", "loan_type"], as_index=False)
        .agg(default_rate=("target_default", "mean"), accounts=("account_id", "count"))
        .query("accounts >= 40")
    )
    fig3 = px.density_heatmap(
        cross,
        x="loan_type",
        y="employment_type",
        z="default_rate",
        title="Cross-cut: employment × loan type (default rate)",
        color_continuous_scale="Reds",
    )
    mid1.plotly_chart(fig3, use_container_width=True)
    bands = df.groupby("risk_band", as_index=False).agg(
        accounts=("account_id", "count"), exposure=("exposure", "sum"), default_rate=("target_default", "mean")
    )
    fig4 = px.bar(bands, x="risk_band", y="accounts", color="default_rate", title="Accounts by predicted risk band")
    mid2.plotly_chart(fig4, use_container_width=True)

    st.markdown("**Insights (so what)**")
    for item in facts.get("insights", []):
        st.markdown(f"- **{item['title']}** — {item['so_what']}")


def account_view(df: pd.DataFrame) -> None:
    st.subheader("Account view")
    default_id = str(df.sort_values("pd_lgbm", ascending=False)["account_id"].iloc[0])
    acc = st.text_input("Account ID", value=default_id)
    row = df.loc[df["account_id"] == acc]
    if row.empty:
        st.warning("No account in the current filter matches that ID.")
        return
    r = row.iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Predicted PD", f"{r['pd_lgbm']*100:.1f}%")
    k2.metric("Risk band", r["risk_band"])
    k3.metric("Observed default", "Yes" if int(r["target_default"]) == 1 else "No")
    k4.metric("Loan amount", f"${r['loan_amount']:,.0f}")
    st.write(
        f"{r['loan_type']} to {r['employment_type']} customer in {r['region']} "
        f"(income ${r['annual_income']:,.0f}, age {int(r['age'])})."
    )
    st.markdown("**Key drivers**")
    st.write(r.get("score_drivers") or "Median-like profile on the main numeric features.")
    c1, c2, c3 = st.columns(3)
    c1.write(
        {
            "bureau_max_dpd": r["bureau_max_dpd"],
            "bureau_utilisation": round(float(r["bureau_utilisation"]), 3),
            "bureau_writeoffs": r["bureau_writeoffs"],
            "bureau_tradelines": r["bureau_tradelines"],
        }
    )
    c2.write(
        {
            "repay_max_dpd": r["repay_max_dpd"],
            "repay_missed": r["repay_missed"],
            "repay_paid_ratio": round(float(r["repay_paid_ratio"]), 3),
            "repay_last_status": r.get("repay_last_status"),
        }
    )
    c3.write(
        {
            "collections_events": r["collections_events"],
            "collections_legal": r["collections_legal"],
            "logistic_pd": round(float(r["pd_logit"]), 3),
        }
    )


def anomaly_view(df: pd.DataFrame, flags: pd.DataFrame, meta: dict) -> None:
    st.subheader("Anomaly review list")
    with st.expander("What triggers a flag, errors, and the human step", expanded=True):
        st.markdown(
            f"""
- **Trigger:** {meta.get('trigger', '')}
- **False positive:** {meta.get('false_positive', '')}
- **False negative:** {meta.get('false_negative', '')}
- **Human-in-the-loop:** {meta.get('human_step', '')}
"""
        )
    if flags.empty:
        st.info("No anomalies in artefacts. Re-run the build.")
        return
    view = flags.merge(df[["account_id"]], on="account_id", how="inner")
    decisions = load_decisions()
    view["review_status"] = view["account_id"].map(lambda a: decisions.get(a, "Open"))
    show_open = st.checkbox("Open items only", value=True)
    if show_open:
        view = view[view["review_status"] == "Open"]
    st.caption(f"{len(view):,} accounts in this list")
    st.dataframe(
        view[
            [
                "account_id",
                "customer_id",
                "loan_type",
                "region",
                "pd_lgbm",
                "anomaly_score",
                "anomaly_reason",
                "review_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    if view.empty:
        return
    pick = st.selectbox("Record a decision", view["account_id"].tolist())
    reason = view.loc[view["account_id"] == pick, "anomaly_reason"].iloc[0]
    st.write(reason)
    b1, b2 = st.columns(2)
    if b1.button("Confirm — send to collections"):
        decisions[pick] = "Confirmed"
        save_decisions(decisions)
        st.success(f"{pick} confirmed")
        st.rerun()
    if b2.button("Dismiss — false positive"):
        decisions[pick] = "Dismissed"
        save_decisions(decisions)
        st.success(f"{pick} dismissed")
        st.rerun()


def ask_view() -> None:
    st.subheader("Ask the Portfolio")
    with st.expander("What triggers it, errors, and the human step", expanded=True):
        st.markdown(
            """
- **Trigger:** the analyst submits a natural-language question. The app answers from pre-computed
  portfolio facts only (grouped rates and totals). If `OPENAI_API_KEY` is set, an LLM rewrites the
  answer from that JSON; otherwise a local retriever is used. **No raw account rows are sent to the LLM.**
- **False positive:** a fluent answer to a question the facts cannot support — we decline instead of guessing.
- **False negative:** an awkwardly phrased question the local retriever misses when no API key is set.
- **Human-in-the-loop:** check the cited grouping against the Portfolio tab before changing policy.
"""
        )
    if "chat" not in st.session_state:
        st.session_state.chat = []
    for turn in st.session_state.chat:
        with st.chat_message(turn["role"]):
            st.write(turn["content"])
            if turn.get("source"):
                st.caption(turn["source"])
    q = st.chat_input("e.g. Which loan type has the highest default rate?")
    if q:
        result = ask(q)
        st.session_state.chat.append({"role": "user", "content": q})
        prefix = "" if not result["declined"] else "Declined. "
        st.session_state.chat.append(
            {"role": "assistant", "content": prefix + result["answer"], "source": result.get("source")}
        )
        st.rerun()


def main() -> None:
    st.title("Credit Risk Portfolio Monitor")
    st.caption("Working book from the five provided tables. Filters apply to every view.")
    panel = load_panel()
    filtered = apply_filters(panel)
    facts = load_json("facts.json")
    meta = json.loads((PROCESSED_DIR / "anomaly_meta.json").read_text()) if (PROCESSED_DIR / "anomaly_meta.json").exists() else {}
    flags = load_anomalies()
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Account", "Anomalies", "Ask the Portfolio"])
    with tab1:
        if filtered.empty:
            st.warning("No accounts match the current filters.")
        else:
            portfolio_view(filtered, facts)
    with tab2:
        if filtered.empty:
            st.warning("No accounts match the current filters.")
        else:
            account_view(filtered)
    with tab3:
        anomaly_view(filtered, flags, meta)
    with tab4:
        ask_view()


if __name__ == "__main__":
    main()
