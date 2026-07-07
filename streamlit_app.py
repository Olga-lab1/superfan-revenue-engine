"""
Superfan Revenue Engine — interactive demo
==========================================
Two tabs: score a fan (with SHAP drivers + Expected Fan Value), and plan an
optimized tour under a budget (integer programming, live).

Run locally:
    streamlit run streamlit_app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features import CATEGORICAL_FEATURES, build_feature_frame  # noqa: E402
from src.tour import compute_city_economics, optimize_tour, naive_big_city_tour  # noqa: E402

st.set_page_config(page_title="Superfan Revenue Engine", page_icon="🎸", layout="wide")


@st.cache_resource
def load_all():
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(ROOT / "models" / "superfan_model.json")
    rev = xgb.XGBRegressor(enable_categorical=True)
    rev.load_model(ROOT / "models" / "revenue_model.json")
    explainer = shap.TreeExplainer(model)
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    econ = compute_city_economics(pd.read_csv(ROOT / "data" / "cities.csv"))
    return model, rev, explainer, meta, econ


model, rev_model, explainer, meta, econ = load_all()
levels = meta["categorical_levels"]

st.title("🎸 Superfan Revenue Engine")
st.caption(
    "RevOps for independent artists: streams pay ~€0.003 — superfans pay €30–300/yr. "
    "Score fans like leads, tour where you're underpriced. Synthetic data; source on GitHub."
)

tab_fan, tab_tour = st.tabs(["🧑‍🎤 Score a fan", "🗺️ Plan a tour"])

# ---------------- Fan scoring ----------------
with tab_fan:
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Fan profile")
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", levels["city"], index=levels["city"].index("Leipzig") if "Leipzig" in levels["city"] else 0)
            age = st.selectbox("Age band", levels["age_band"], index=2)
            channel = st.selectbox("Discovered via", levels["discovery_channel"], index=0)
            months = st.slider("Months since discovery", 0.5, 36.0, 6.0, 0.5)
        with c2:
            streams = st.slider("Monthly streams", 0, 200, 25)
            depth = st.slider("Catalog depth (%)", 0, 100, 35)
            skip = st.slider("Skip rate", 0.0, 1.0, 0.25, 0.01)
            playlist_adds = st.slider("Playlist adds", 0, 20, 2)

        st.markdown("**Beyond streaming**")
        c3, c4 = st.columns(2)
        with c3:
            saves = st.toggle("Saved to library", True)
            follows = st.toggle("Follows socials", True)
            mailing = st.toggle("On mailing list", False)
            open_rate = st.slider("Email open rate", 0.0, 1.0, 0.5, 0.05,
                                  disabled=not mailing)
        with c4:
            show = st.toggle("Attended a show", False)
            shared = st.toggle("Shared a track", False)
            engagements = st.slider("Social engagements (90d)", 0, 60, 5)
            dms = st.slider("Comments / DMs", 0, 20, 0)

    fan = pd.DataFrame([{
        "city": city, "age_band": age, "discovery_channel": channel,
        "months_since_discovery": months, "monthly_streams": streams,
        "catalog_depth_pct": float(depth), "skip_rate": skip,
        "playlist_adds": playlist_adds, "saves_library": int(saves),
        "follows_socials": int(follows),
        "social_engagements_90d": engagements if follows else 0,
        "on_mailing_list": int(mailing),
        "email_open_rate": open_rate if mailing else 0.0,
        "attended_show": int(show), "shared_track": int(shared),
        "comments_dms": dms,
    }])

    X = build_feature_frame(fan)
    for col in CATEGORICAL_FEATURES:
        X[col] = pd.Categorical(X[col], categories=levels[col])
    p = float(model.predict_proba(X)[:, 1][0])
    efv = round(p * max(float(np.expm1(rev_model.predict(X))[0]), 0.0), 2)

    TIERS = [
        (0.40, "Superfan-ready", "Personal invite: presale access, signed edition, direct thank-you.", "💜"),
        (0.20, "Warming", "Move to owned channels: mailing-list invite with an exclusive track.", "🔥"),
        (0.08, "Casual+", "Retarget with live session video; nudge toward a library save.", "🌱"),
        (0.00, "Passive", "Broad reach campaigns only; no direct spend.", "❄️"),
    ]
    tier, action, emoji = next((t, a, e) for th, t, a, e in TIERS if p >= th)

    with right:
        st.subheader("Score")
        m1, m2, m3 = st.columns(3)
        m1.metric("Superfan probability", f"{p:.0%}")
        m2.metric("Expected Fan Value", f"€{efv:.2f}")
        m3.metric("Tier", f"{emoji} {tier}")
        st.progress(min(p, 1.0))
        st.info(f"**Recommended action:** {action}")
        st.caption(f"Equivalent stream value: one fan at €{efv:.2f}/yr ≈ "
                   f"{int(efv / 0.003):,} streams.")

        st.subheader("Why (SHAP)")
        sv = explainer.shap_values(X)[0]
        contrib = pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False).head(8)
        chart_df = pd.DataFrame(
            {"impact": contrib.values[::-1]},
            index=[f"{f} = {X.iloc[0][f]}" for f in contrib.index][::-1],
        )
        st.bar_chart(chart_df, horizontal=True, color="#7c3aed")

# ---------------- Tour planner ----------------
with tab_tour:
    st.subheader("Budget-constrained tour optimization")
    c1, c2, c3 = st.columns(3)
    budget = c1.slider("Tour budget (EUR)", 3000, 30000, 12000, 500)
    min_c = c2.slider("Min cities", 2, 15, 6)
    max_c = c3.slider("Max cities", min_c, 20, 12)

    plan = optimize_tour(econ, budget, min_c, max_c)
    if plan["status"] != "Optimal":
        st.error("No feasible tour for these constraints — raise the budget or lower the minimum cities.")
    else:
        naive = naive_big_city_tour(econ, budget, len(plan["cities"]))
        uplift = plan["total_expected_profit_eur"] - naive["total_expected_profit_eur"]

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected profit", f"€{plan['total_expected_profit_eur']:,.0f}")
        m2.metric("Total cost", f"€{plan['total_cost_eur']:,.0f}")
        m3.metric("vs. naive big-city tour", f"+€{uplift:,.0f}",
                  help="Same budget spent on the biggest cities you can afford.")

        sel = pd.DataFrame(plan["cities"])
        st.bar_chart(sel.set_index("city")["expected_profit"], color="#7c3aed",
                     y_label="Expected profit (EUR)")

        st.dataframe(
            sel[["city", "country", "expected_attendance", "expected_revenue",
                 "cost", "expected_profit", "arbitrage_score"]],
            hide_index=True, width="stretch",
        )
        st.caption(
            "High arbitrage score = underpriced market: unusually strong listener density "
            "and supporter rate relative to venue cost. The optimizer keeps profitable big "
            "rooms but adds the cities big-city instinct skips."
        )

st.divider()
mets = meta["metrics"]
st.markdown(
    f"**Model card:** conversion ROC-AUC **{mets['roc_auc']}** "
    f"(logistic baseline {mets['baseline_logreg_roc_auc']} — reported, not hidden) · "
    f"top-decile lift **{mets['top_decile_lift']}×** · top 10% of fans by EFV hold "
    f"**{mets['revenue_capture_top_10pct']:.0%}** of supporter revenue."
)
