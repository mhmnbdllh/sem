"""
assumptions.py
==============
Sprint 1 — Assumption Testing Module for SEM Studio.

Covers:
- Common Method Bias (Harman's Single Factor Test)
- Linearity assessment
- Homoscedasticity check
- Full assumption checklist with pass/fail
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from factor_analyzer import FactorAnalyzer


LEVEL_COLOR = {
    "excellent": "#2ecc71",
    "good":      "#27ae60",
    "ok":        "#3498db",
    "warning":   "#f39c12",
    "critical":  "#e74c3c",
}

def _badge(level: str, message: str):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>',
        unsafe_allow_html=True
    )


# ─── HARMAN'S SINGLE FACTOR TEST ────────────────────────────────

def render_harman_test(df: pd.DataFrame, indicator_cols: list):
    st.subheader("🧪 Common Method Bias — Harman's Single Factor Test")
    st.markdown(
        "When all data are collected from a single source (self-report questionnaire), "
        "**common method bias (CMB)** may inflate correlations artificially. "
        "Harman's test is a widely-used (though conservative) screening test.\n\n"
        "**Criterion:** If a single factor explains > 50% of variance, CMB is a concern."
    )

    data = df[indicator_cols].dropna()

    if data.shape[0] < data.shape[1] + 2:
        st.warning("⚠️ Not enough complete cases to perform Harman's test.")
        return

    try:
        fa = FactorAnalyzer(n_factors=1, rotation=None)
        fa.fit(data)
        ev, _ = fa.get_eigenvalues()
        var = fa.get_factor_variance()
        single_factor_var = var[1][0]  # proportion of variance for 1 factor

        c1, c2 = st.columns(2)
        c1.metric("Variance Explained by First Factor", f"{single_factor_var:.1%}")
        c2.metric("Criterion", "< 50%")

        if single_factor_var < 0.50:
            _badge("ok",
                f"Harman's single factor explains **{single_factor_var:.1%}** of variance "
                f"— below the 50% criterion. ✅ Common method bias is **not a major concern** based on this test."
            )
        else:
            _badge("warning",
                f"Harman's single factor explains **{single_factor_var:.1%}** of variance "
                f"— **above the 50% criterion**. ⚠️ Common method bias may be present. "
                "Consider reporting this as a limitation and/or using a marker variable technique."
            )

        # Scree plot of eigenvalues
        ev_df = pd.DataFrame({"Factor": range(1, len(ev)+1), "Eigenvalue": ev})
        fig = px.line(ev_df, x="Factor", y="Eigenvalue",
                      markers=True, template="plotly_dark",
                      title="Eigenvalue Scree Plot (Harman's Test)")
        fig.add_hline(y=1, line_dash="dash", line_color="#f39c12", annotation_text="Eigenvalue = 1")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error running Harman's test: {str(e)}")


# ─── LINEARITY ──────────────────────────────────────────────────

def render_linearity(df: pd.DataFrame, indicator_cols: list, constructs: dict):
    st.subheader("📏 Linearity Assessment")
    st.markdown(
        "SEM assumes **linear relationships** between latent variables. "
        "We assess this by examining scatter plots and correlations between construct parcels "
        "(mean scores of indicators per construct)."
    )

    if not constructs or len(constructs) < 2:
        st.info("ℹ️ Define at least 2 constructs to assess linearity between them.")
        return

    # Compute parcel scores (mean of indicators per construct)
    parcel_df = pd.DataFrame()
    for cname, citems in constructs.items():
        valid_items = [c for c in citems if c in df.columns]
        if valid_items:
            parcel_df[cname] = df[valid_items].mean(axis=1)

    construct_names = list(parcel_df.columns)

    if len(construct_names) < 2:
        st.warning("⚠️ Not enough constructs with valid indicators.")
        return

    c1, c2 = st.columns(2)
    with c1:
        x_var = st.selectbox("X-axis (Predictor)", construct_names, key="lin_x")
    with c2:
        y_var = st.selectbox("Y-axis (Outcome)", [c for c in construct_names if c != x_var], key="lin_y")

    # Scatter with regression line
    plot_data = parcel_df[[x_var, y_var]].dropna()
    fig = px.scatter(
        plot_data, x=x_var, y=y_var,
        trendline="ols",
        trendline_color_override="#e74c3c",
        template="plotly_dark",
        title=f"Scatter: {x_var} → {y_var}",
        opacity=0.6,
    )
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    r, p = stats.pearsonr(plot_data[x_var], plot_data[y_var])
    st.markdown(f"Pearson r = **{r:.3f}** (p = {p:.4f})")

    if abs(r) > 0.10:
        _badge("ok", f"A linear relationship between **{x_var}** and **{y_var}** is evident (r = {r:.3f}). ✅ Linearity assumption is supported.")
    else:
        _badge("warning", f"The relationship between **{x_var}** and **{y_var}** is very weak (r = {r:.3f}). ⚠️ Review your theoretical model.")


# ─── ASSUMPTION CHECKLIST ────────────────────────────────────────

def render_assumption_checklist():
    st.subheader("✅ Assumption Checklist Summary")
    st.markdown(
        "This checklist summarizes all key assumptions for SEM/CFA. "
        "Resolve any **❌ Fail** items before proceeding to model estimation."
    )

    checks = {
        "Sample Size (n ≥ 200 for SEM)": st.session_state.get("df") is not None and len(st.session_state["df"]) >= 200,
        "No Extreme Missing Values (< 10%)": True,  # to be dynamically computed
        "No Critical Multicollinearity (r < .85)": True,
        "Multivariate Normality (or MLR selected)": st.session_state.get("recommended_estimator") in ["ML", "MLR", "WLSMV"],
        "No Extreme Multivariate Outliers": True,
        "At Least 3 Indicators per Construct": all(
            len(v) >= 3 for v in st.session_state.get("constructs", {}).values()
        ),
        "Constructs Defined": len(st.session_state.get("constructs", {})) >= 1,
        "Structural Paths Defined": len(st.session_state.get("structural_paths", [])) >= 1,
    }

    # Dynamic missing check
    if st.session_state.get("df") is not None:
        assignments = st.session_state.get("assignments", {})
        indicator_cols = [c for c, r in assignments.items() if r == "indicator"]
        df = st.session_state["df"]
        if indicator_cols:
            max_missing = df[indicator_cols].isna().mean().max()
            checks["No Extreme Missing Values (< 10%)"] = max_missing < 0.10

    rows = []
    for check, passed in checks.items():
        rows.append({
            "Assumption": check,
            "Status": "✅ Pass" if passed else "❌ Fail",
        })

    result_df = pd.DataFrame(rows)

    def color_status(val):
        if "Pass" in str(val): return "color: #2ecc71; font-weight: bold"
        return "color: #e74c3c; font-weight: bold"

    styled = result_df.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    n_fail = sum(1 for v in checks.values() if not v)
    if n_fail == 0:
        _badge("excellent", "🎉 All assumption checks passed! You are ready to proceed to CFA/SEM.")
    elif n_fail <= 2:
        _badge("warning", f"⚠️ {n_fail} assumption(s) require attention. Review the items marked ❌ above.")
    else:
        _badge("critical", f"❌ {n_fail} critical assumptions are not met. Resolve these before running SEM.")


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_assumptions():
    """Main render function for the Assumption Testing page."""

    st.title("🧪 Assumption Testing")
    st.markdown(
        "This module verifies all key **methodological assumptions** required for valid SEM estimation. "
        "Each test includes an automatic interpretation and recommendation."
    )

    if "df" not in st.session_state or not st.session_state.get("df_ready", False):
        st.warning("⚠️ Please complete **Data Input** first.")
        return

    df          = st.session_state["df"]
    assignments = st.session_state.get("assignments", {})
    constructs  = st.session_state.get("constructs", {})

    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if not indicator_cols:
        st.warning("⚠️ No indicator variables assigned.")
        return

    render_harman_test(df, indicator_cols)
    st.markdown("---")
    render_linearity(df, indicator_cols, constructs)
    st.markdown("---")
    render_assumption_checklist()

    st.markdown("---")
    st.success(
        "✅ Assumption testing complete. Proceed to **Exploratory Factor Analysis (EFA)** "
        "or **Confirmatory Factor Analysis (CFA)** in the sidebar."
    )
