"""Streamlit dashboard for the e-commerce order fulfillment analysis.

Run from the repo root:
    streamlit run dashboard/app.py

Views:
    1. Overview       - KPI row + platform/category mix
    2. Delivery Delay - delay rate by platform, category, weekday, hour
    3. Refund         - refund rate by rating, delay, category, value
    4. Model          - trained-model vs rule-based baseline, top-risk orders

Data / artifacts read:
    data/processed/orders_clean.csv
    data/processed/features.csv
    models/metrics.json
    models/refund_model.joblib
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"


@st.cache_data(show_spinner=False)
def load_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA / "orders_clean.csv", parse_dates=["Order Date & Time"])
    df["is_delayed"] = df["is_delayed"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    return pd.read_csv(DATA / "features.csv")


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((MODELS / "metrics.json").read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODELS / "refund_model.joblib")


def kpi(metric_delta: tuple[str, str]) -> None:
    label, value = metric_delta
    st.metric(label, value)


def main() -> None:
    st.set_page_config(page_title="Order Fulfillment Analytics", layout="wide")
    st.title("E-Commerce Order Fulfillment Analytics")
    st.caption("Delivery-delay and refund analysis + a trained refund-risk model.")

    df = load_clean()
    feat = load_features()
    metrics = load_metrics()

    # ---- Sidebar filters -----------------------------------------------------
    st.sidebar.header("Filters")
    platforms = st.sidebar.multiselect(
        "Platforms", sorted(df["Platform"].unique()), default=sorted(df["Platform"].unique())
    )
    categories = st.sidebar.multiselect(
        "Categories", sorted(df["Product Category"].unique()), default=sorted(df["Product Category"].unique())
    )
    date_range = st.sidebar.date_input(
        "Date range",
        value=(df["Order Date & Time"].min().date(), df["Order Date & Time"].max().date()),
        min_value=df["Order Date & Time"].min().date(),
        max_value=df["Order Date & Time"].max().date(),
    )

    mask = (
        df["Platform"].isin(platforms)
        & df["Product Category"].isin(categories)
        & (df["Order Date & Time"].dt.date >= date_range[0])
        & (df["Order Date & Time"].dt.date <= date_range[1])
    )
    view = df[mask]

    view_tab, delay_tab, refund_tab, model_tab = st.tabs(
        ["Overview", "Delivery Delay", "Refund Analysis", "Model"]
    )

    # =========================================================================
    # Overview
    # =========================================================================
    with view_tab:
        st.subheader("Key performance indicators")
        kpis = [
            ("Orders", f"{len(view):,}"),
            ("On-time rate", f"{100 * (1 - view['is_delayed'].mean()):.1f}%"),
            ("Refund rate", f"{100 * view['Refund Requested'].eq('Yes').mean():.1f}%"),
            ("Avg delivery (min)", f"{view['Actual Delivery (Minutes)'].mean():.1f}"),
        ]
        cols = st.columns(len(kpis))
        for col, kp in zip(cols, kpis):
            with col:
                kpi(kp)

        st.markdown("### Order mix")
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(view["Platform"].value_counts())
        with c2:
            st.bar_chart(view["Product Category"].value_counts())

    # =========================================================================
    # Delivery Delay
    # =========================================================================
    with delay_tab:
        st.subheader("Delivery delay rate")
        c1, c2 = st.columns(2)
        with c1:
            by_plat = 100 * view.groupby("Platform")["is_delayed"].mean().sort_values()
            st.bar_chart(by_plat, color="#C44E52")
            st.caption("Delay rate by platform (%)")
        with c2:
            by_cat = 100 * view.groupby("Product Category")["is_delayed"].mean().sort_values()
            st.bar_chart(by_cat, color="#C44E52")
            st.caption("Delay rate by category (%)")

        c3, c4 = st.columns(2)
        with c3:
            wd = view["Order Date & Time"].dt.day_name()
            order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            by_wd = 100 * view.groupby(wd)["is_delayed"].mean().reindex(order)
            st.bar_chart(by_wd, color="#55A868")
            st.caption("Delay rate by weekday (%)")
        with c4:
            by_hr = 100 * view.groupby(view["Order Date & Time"].dt.hour)["is_delayed"].mean()
            st.bar_chart(by_hr, color="#55A868")
            st.caption("Delay rate by hour of day (%)")

    # =========================================================================
    # Refund Analysis
    # =========================================================================
    with refund_tab:
        st.subheader("Refund behavior")
        c1, c2 = st.columns(2)
        with c1:
            by_rating = 100 * view.groupby("Service Rating")["Refund Requested"].apply(
                lambda s: s.eq("Yes").mean()
            )
            st.bar_chart(by_rating, color="#C44E52")
            st.caption("Refund rate by service rating (%)")
        with c2:
            by_status = (
                view.groupby("is_delayed")["Refund Requested"]
                .apply(lambda s: 100 * s.eq("Yes").mean())
                .rename(index={False: "On time", True: "Delayed"})
            )
            st.bar_chart(by_status, color="#4C72B0")
            st.caption("Refund rate by delivery status (%)")

        st.markdown("#### Refund rate by weekday x hour (heatmap)")
        piv = view.pivot_table(
            index=view["Order Date & Time"].dt.day_name(),
            columns=view["Order Date & Time"].dt.hour,
            values="Refund Requested",
            aggfunc=lambda s: 100 * s.eq("Yes").mean(),
        )
        st.dataframe(piv)

    # =========================================================================
    # Model
    # =========================================================================
    with model_tab:
        st.subheader("Refund-risk model vs rule-based baseline")
        summary = pd.DataFrame(
            {
                "Model": ["Logistic regression", "Random forest", "Rule-based baseline (SQL)"],
                "ROC-AUC": [
                    metrics["logistic_regression"]["roc_auc"],
                    metrics["random_forest"]["roc_auc"],
                    metrics["rule_based_baseline"]["roc_auc"],
                ],
                "F1": [
                    metrics["logistic_regression"]["f1"],
                    metrics["random_forest"]["f1"],
                    metrics["rule_based_baseline"]["f1"],
                ],
                "Precision": [
                    metrics["logistic_regression"]["precision"],
                    metrics["random_forest"]["precision"],
                    metrics["rule_based_baseline"]["precision"],
                ],
                "Recall": [
                    metrics["logistic_regression"]["recall"],
                    metrics["random_forest"]["recall"],
                    metrics["rule_based_baseline"]["recall"],
                ],
            }
        )
        st.dataframe(summary, hide_index=True)

        st.markdown("#### Feature importance (random forest)")
        imp = pd.Series(metrics["random_forest"]["feature_importance"]).sort_values()
        st.bar_chart(imp, color="#4C72B0")

        st.markdown("#### Top refund-risk orders")
        model_bundle = load_model()
        rf = model_bundle["model"]
        feature_cols = model_bundle["feature_cols"]

        # Score the (filtered) orders with the saved model.
        feat_masked = feat.merge(view[["Order ID"]], left_on="order_id", right_on="Order ID", how="inner")
        if feat_masked.empty:
            st.info("No orders match the current filters.")
        else:
            X = feat_masked[feature_cols]
            proba = rf.predict_proba(X)[:, 1]
            top = feat_masked.assign(risk_score=proba).nlargest(10, "risk_score")[
                ["order_id", "risk_score"]
            ]
            top = top.merge(view, left_on="order_id", right_on="Order ID", how="left")
            st.dataframe(
                top[
                    [
                        "order_id",
                        "Platform",
                        "Product Category",
                        "Service Rating",
                        "Order Value (INR)",
                        "Delivery Delay",
                        "Refund Requested",
                        "risk_score",
                    ]
                ],
                hide_index=True,
            )


if __name__ == "__main__":
    main()
