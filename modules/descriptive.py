"""
descriptive.py
==============
Sprint 1 — Descriptive Statistics Module for SEM Studio.

Covers:
- Per-item descriptive statistics (mean, SD, skewness, kurtosis)
- Missing value analysis with recommendations
- Univariate outlier detection (z-scores)
- Multivariate outlier detection (Mahalanobis D²)
- Multivariate normality (Mardia's test)
- Correlation matrix with significance
- Estimator recommendation
- Auto-interpretation for every output
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.spatial.distance import mahalanobis

from utils.thresholds import DESCRIPTIVE, OUTLIER
from utils.interpretation import (
    interpret_sample_size, interpret_missing, interpret_skewness,
    interpret_kurtosis, interpret_mardia, interpret_outliers
)
from utils.apa_tables import descriptive_table, correlation_table, style_df


# ─── HELPER: LEVEL COLOR ────────────────────────────────────────

LEVEL_COLOR = {
    "excellent": "#2ecc71",
    "good":      "#27ae60",
    "ok":        "#3498db",
    "warning":   "#f39c12",
    "critical":  "#e74c3c",
    "info":      "#95a5a6",
}

def _badge(level: str, message: str):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>',
        unsafe_allow_html=True
    )


# ─── SECTION 1: OVERVIEW ────────────────────────────────────────

def render_overview(df: pd.DataFrame, validation: dict):
    st.subheader("📊 Dataset Overview")

    n_rows, n_cols = df.shape
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    missing_total  = df.isnull().sum().sum()
    missing_pct    = missing_total / (n_rows * n_cols)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Respondents (n)", f"{n_rows:,}")
    c2.metric("Total Variables", n_cols)
    c3.metric("Numeric Variables", len(numeric_cols))
    c4.metric("Total Missing", f"{missing_total:,}")
    c5.metric("Missing %", f"{missing_pct:.1%}")

    # Sample size interpretation
    result = interpret_sample_size(n_rows)
    _badge(result["level"], result["message"])


# ─── SECTION 2: DESCRIPTIVE STATS TABLE ─────────────────────────

def render_descriptive_table(df: pd.DataFrame, indicator_cols: list):
    st.subheader("📋 Descriptive Statistics")
    st.markdown(
        "The table below presents descriptive statistics for all indicator variables. "
        "**Skewness** and **kurtosis** are key for assessing normality assumptions required for ML estimation."
    )

    desc_df = descriptive_table(df[indicator_cols])

    # Color-code skewness and kurtosis
    def color_skew(val):
        try:
            v = float(val)
            if abs(v) <= 1.0: return "color: #2ecc71"
            elif abs(v) <= 2.0: return "color: #f39c12"
            else: return "color: #e74c3c"
        except: return ""

    def color_kurt(val):
        try:
            v = float(val)
            if abs(v) <= 3.0: return "color: #2ecc71"
            elif abs(v) <= 7.0: return "color: #f39c12"
            else: return "color: #e74c3c"
        except: return ""

    styled = (
        desc_df.style
        .applymap(color_skew, subset=["Skewness"])
        .applymap(color_kurt, subset=["Kurtosis"])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{
            "selector": "th",
            "props": [("background-color", "#1e2130"), ("color", "white"),
                      ("font-weight", "bold"), ("text-align", "center")]
        }])
    )
    st.dataframe(styled, use_container_width=True)

    # Interpretation for each item
    with st.expander("🔍 Item-by-Item Normality Interpretation"):
        for _, row in desc_df.iterrows():
            col_name = row["Variable"]
            sk = interpret_skewness(row["Skewness"], col_name)
            ku = interpret_kurtosis(row["Kurtosis"], col_name)
            if sk["level"] in ("warning", "critical") or ku["level"] in ("warning", "critical"):
                _badge(sk["level"], sk["message"])
                _badge(ku["level"], ku["message"])

        st.markdown("*Only items with notable skewness/kurtosis are shown.*")

    # Distribution plots
    with st.expander("📈 Distribution Plots"):
        cols_to_plot = indicator_cols[:12]  # limit for performance
        n_plot = len(cols_to_plot)
        cols_grid = st.columns(min(4, n_plot))

        for i, col in enumerate(cols_to_plot):
            with cols_grid[i % min(4, n_plot)]:
                fig = px.histogram(
                    df, x=col, nbins=10,
                    title=col,
                    color_discrete_sequence=["#2E86AB"],
                    template="plotly_dark",
                )
                fig.update_layout(
                    height=220, margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False,
                    title_font_size=12,
                )
                st.plotly_chart(fig, use_container_width=True)


# ─── SECTION 3: MISSING VALUE ANALYSIS ──────────────────────────

def render_missing_analysis(df: pd.DataFrame, indicator_cols: list):
    st.subheader("🔍 Missing Value Analysis")

    missing_summary = pd.DataFrame({
        "Variable":  indicator_cols,
        "Missing N": [df[c].isna().sum() for c in indicator_cols],
        "Missing %": [df[c].isna().mean() * 100 for c in indicator_cols],
    }).sort_values("Missing %", ascending=False)

    total_missing = missing_summary["Missing N"].sum()

    if total_missing == 0:
        st.success("✅ **No missing values detected.** Your dataset is complete.")
        return

    # Bar chart
    fig = px.bar(
        missing_summary[missing_summary["Missing N"] > 0],
        x="Variable", y="Missing %",
        color="Missing %",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 15],
        template="plotly_dark",
        title="Missing Values by Variable (%)",
    )
    fig.add_hline(y=5,  line_dash="dash", line_color="#f39c12", annotation_text="5% threshold")
    fig.add_hline(y=10, line_dash="dash", line_color="#e74c3c", annotation_text="10% critical")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Table with interpretations
    st.dataframe(
        missing_summary.style.background_gradient(subset=["Missing %"], cmap="RdYlGn_r"),
        use_container_width=True
    )

    # Per-variable interpretation
    with st.expander("🔍 Recommendations per Variable"):
        for _, row in missing_summary.iterrows():
            pct = row["Missing %"] / 100
            if pct > 0:
                result = interpret_missing(pct, row["Variable"])
                _badge(result["level"], result["message"])

    # Overall recommendation
    total_pct = df[indicator_cols].isna().mean().mean()
    if total_pct <= DESCRIPTIVE["missing_acceptable"]:
        _badge("ok", f"Overall missing rate: {total_pct:.1%}. **Listwise deletion** or **pairwise** is acceptable. FIML is also suitable.")
    else:
        _badge("warning", f"Overall missing rate: {total_pct:.1%}. **Full Information Maximum Likelihood (FIML)** or **multiple imputation** is strongly recommended.")


# ─── SECTION 4: OUTLIER DETECTION ───────────────────────────────

def render_outlier_detection(df: pd.DataFrame, indicator_cols: list):
    st.subheader("⚠️ Outlier Detection")

    tab1, tab2 = st.tabs(["Univariate Outliers (Z-score)", "Multivariate Outliers (Mahalanobis D²)"])

    # ── Univariate ──────────────────────────────────────────────
    with tab1:
        st.markdown(
            "Univariate outliers are cases with extreme scores on individual variables "
            f"(|z| > {OUTLIER['z_score_threshold']}, corresponding to p < .001)."
        )

        data = df[indicator_cols].dropna()
        z_scores = np.abs(stats.zscore(data))
        outlier_mask = (z_scores > OUTLIER["z_score_threshold"]).any(axis=1)
        n_uni_outliers = outlier_mask.sum()

        if n_uni_outliers == 0:
            st.success("✅ No univariate outliers detected.")
        else:
            st.warning(f"⚠️ {n_uni_outliers} respondent(s) have at least one extreme univariate score.")
            outlier_rows = data[outlier_mask].copy()
            outlier_rows["Outlier Variables"] = [
                ", ".join([indicator_cols[j] for j in range(len(indicator_cols)) if z_scores[i, j] > OUTLIER["z_score_threshold"]])
                for i in outlier_rows.index
            ]
            st.dataframe(outlier_rows[["Outlier Variables"]], use_container_width=True)

    # ── Multivariate ─────────────────────────────────────────────
    with tab2:
        st.markdown(
            "Multivariate outliers are detected using **Mahalanobis distance (D²)**, "
            f"which accounts for correlations among variables. Cases with p < {OUTLIER['mahalanobis_pvalue']} are flagged."
        )

        data = df[indicator_cols].dropna()

        if data.shape[0] < data.shape[1] + 2:
            st.warning("⚠️ Not enough complete cases to compute Mahalanobis distance.")
            return

        try:
            mean_vec = data.mean().values
            cov_mat  = np.cov(data.values.T)

            if np.linalg.matrix_rank(cov_mat) < cov_mat.shape[0]:
                st.warning("⚠️ Singular covariance matrix. Some items may be perfectly correlated. Skipping Mahalanobis test.")
                return

            inv_cov = np.linalg.inv(cov_mat)
            d2 = np.array([mahalanobis(row, mean_vec, inv_cov)**2 for row in data.values])
            p_vals = 1 - stats.chi2.cdf(d2, df=data.shape[1])

            outlier_flag = p_vals < OUTLIER["mahalanobis_pvalue"]
            n_mv_outliers = outlier_flag.sum()

            result = interpret_outliers(n_mv_outliers, len(data))
            _badge(result["level"], result["message"])

            # Plot
            plot_df = pd.DataFrame({"D²": d2, "p-value": p_vals, "Outlier": outlier_flag})
            plot_df["Respondent"] = data.index

            fig = px.scatter(
                plot_df, x="Respondent", y="D²",
                color="Outlier",
                color_discrete_map={True: "#e74c3c", False: "#2E86AB"},
                template="plotly_dark",
                title="Mahalanobis D² per Respondent",
            )
            chi2_crit = stats.chi2.ppf(1 - OUTLIER["mahalanobis_pvalue"], df=data.shape[1])
            fig.add_hline(y=chi2_crit, line_dash="dash", line_color="#e74c3c",
                         annotation_text=f"χ²({data.shape[1]}, p=.001) = {chi2_crit:.2f}")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            if n_mv_outliers > 0:
                st.dataframe(
                    plot_df[plot_df["Outlier"]][["Respondent", "D²", "p-value"]]
                    .sort_values("D²", ascending=False)
                    .round(3),
                    use_container_width=True
                )

            st.session_state["mv_outliers"] = plot_df[outlier_flag]["Respondent"].tolist()
            st.session_state["d2_values"]   = d2

        except Exception as e:
            st.error(f"❌ Error computing Mahalanobis distance: {str(e)}")


# ─── SECTION 5: MULTIVARIATE NORMALITY ──────────────────────────

def render_normality(df: pd.DataFrame, indicator_cols: list):
    st.subheader("📐 Multivariate Normality — Mardia's Test")
    st.markdown(
        "Mardia's test assesses **multivariate normality**, which is a key assumption for "
        "**Maximum Likelihood (ML)** estimation in SEM. "
        "Violation suggests using **Robust ML (MLR)** instead."
    )

    data = df[indicator_cols].dropna()

    if data.shape[0] < 50:
        st.warning("⚠️ Too few complete cases for reliable normality testing.")
        return

    try:
        # Mardia's multivariate skewness & kurtosis
        n, p = data.shape
        X = data.values - data.mean().values  # center
        S = np.cov(X.T)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            st.warning("⚠️ Singular covariance matrix. Cannot compute Mardia's test.")
            return

        # Mardia's skewness
        A = X @ S_inv @ X.T
        b1p = (A**3).sum() / (n**2)
        chi2_skew = n * b1p / 6
        df_skew   = p * (p+1) * (p+2) / 6
        p_skew    = 1 - stats.chi2.cdf(chi2_skew, df_skew)

        # Mardia's kurtosis
        b2p   = np.trace(A**2) / n
        e_b2p = p * (p+2)
        var_b2p = 8 * p * (p+2) / n
        z_kurt = (b2p - e_b2p) / np.sqrt(var_b2p)
        p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))

        # Display
        c1, c2 = st.columns(2)
        c1.metric("Mardia's Skewness (χ²)", f"{chi2_skew:.3f}", f"p = {p_skew:.4f}")
        c2.metric("Mardia's Kurtosis (z)",  f"{z_kurt:.3f}",   f"p = {p_kurt:.4f}")

        result = interpret_mardia(p_skew, p_kurt)
        _badge(result["level"], result["message"])

        # Save recommended estimator
        st.session_state["recommended_estimator"] = result.get("recommended_estimator", "ML")

        # Q-Q plots for visual check
        with st.expander("📈 Q-Q Plots (Univariate Normality Check per Item)"):
            cols_to_plot = indicator_cols[:8]
            grid_cols = st.columns(min(4, len(cols_to_plot)))

            for i, col in enumerate(cols_to_plot):
                with grid_cols[i % min(4, len(cols_to_plot))]:
                    s = data[col].dropna()
                    (osm, osr), (slope, intercept, _) = stats.probplot(s)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=osm, y=osr, mode="markers",
                        marker=dict(color="#2E86AB", size=4), name="Data"
                    ))
                    fig.add_trace(go.Scatter(
                        x=[min(osm), max(osm)],
                        y=[slope*min(osm)+intercept, slope*max(osm)+intercept],
                        mode="lines", line=dict(color="#e74c3c"), name="Normal"
                    ))
                    fig.update_layout(
                        title=col, height=220, template="plotly_dark",
                        margin=dict(l=5, r=5, t=30, b=5),
                        showlegend=False, title_font_size=11
                    )
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error in Mardia's test: {str(e)}")


# ─── SECTION 6: CORRELATION MATRIX ──────────────────────────────

def render_correlation_matrix(df: pd.DataFrame, indicator_cols: list):
    st.subheader("🔗 Correlation Matrix")
    st.markdown(
        "Pearson correlations among all indicator variables. "
        "Correlations > .85 may indicate **multicollinearity**, which can cause issues in SEM."
    )

    data = df[indicator_cols].dropna()
    corr_matrix = data.corr()

    # Heatmap
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="plotly_dark",
        title="Correlation Heatmap",
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(height=max(350, len(indicator_cols) * 30))
    st.plotly_chart(fig, use_container_width=True)

    # APA table
    with st.expander("📋 APA-Style Correlation Table (lower triangle)"):
        apa_corr = correlation_table(df, indicator_cols)
        st.dataframe(apa_corr, use_container_width=True)
        st.caption("Note. * p < .05. ** p < .01. *** p < .001.")

    # Multicollinearity warnings
    high_corr = []
    for i in range(len(indicator_cols)):
        for j in range(i+1, len(indicator_cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.85:
                high_corr.append((indicator_cols[i], indicator_cols[j], r))

    if high_corr:
        st.warning("⚠️ **High correlations detected** (r > .85) — potential multicollinearity:")
        for c1, c2, r in high_corr:
            st.markdown(f"  - **{c1}** ↔ **{c2}**: r = {r:.3f}")
        st.markdown(
            "Consider whether these items are measuring the same thing. "
            "If so, one may be redundant and could be dropped."
        )
    else:
        st.success("✅ No problematic multicollinearity detected (all r ≤ .85).")


# ─── SECTION 7: ESTIMATOR RECOMMENDATION ────────────────────────

def render_estimator_recommendation():
    st.subheader("⚙️ Recommended Estimator")

    estimator = st.session_state.get("recommended_estimator", None)

    estimator_info = {
        "ML": {
            "name": "Maximum Likelihood (ML)",
            "when": "Multivariate normality is satisfied",
            "pros": "Most efficient estimator; widely accepted; provides fit indices",
            "cons": "Sensitive to non-normality",
            "color": "#2ecc71",
        },
        "MLR": {
            "name": "Robust Maximum Likelihood (MLR)",
            "when": "Multivariate normality is violated; continuous data",
            "pros": "Provides Satorra-Bentler corrected χ²; robust SEs; handles non-normality",
            "cons": "Slightly less efficient than ML under normality",
            "color": "#f39c12",
        },
        "WLSMV": {
            "name": "Weighted Least Squares (WLSMV)",
            "when": "Ordinal / categorical data (e.g., 5-point Likert treated as ordinal)",
            "pros": "Designed for ordinal data; no normality assumption",
            "cons": "Requires larger samples; fewer fit indices available",
            "color": "#3498db",
        },
    }

    if estimator and estimator in estimator_info:
        info = estimator_info[estimator]
        st.markdown(
            f'<div style="background:{info["color"]}22;border:2px solid {info["color"]};'
            f'border-radius:8px;padding:16px;">'
            f'<h4 style="color:{info["color"]}">✅ Recommended: {info["name"]}</h4>'
            f'<b>When to use:</b> {info["when"]}<br>'
            f'<b>Advantages:</b> {info["pros"]}<br>'
            f'<b>Limitations:</b> {info["cons"]}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("⚙️ Run the **Multivariate Normality** test above to receive an automatic estimator recommendation.")

    st.markdown("---")
    st.markdown("**Override Estimator (optional):**")
    override = st.selectbox(
        "You can manually select an estimator if needed:",
        options=["Auto (recommended)", "ML", "MLR", "WLSMV"],
        index=0,
    )
    if override != "Auto (recommended)":
        st.session_state["recommended_estimator"] = override
        st.success(f"✅ Estimator manually set to **{override}**.")


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_descriptive():
    """Main render function for the Descriptive Statistics page."""

    st.title("📊 Descriptive Statistics & Assumption Testing")
    st.markdown(
        "This module provides a **complete pre-analysis examination** of your data. "
        "Every output includes automatic interpretation to guide your methodological decisions."
    )

    # Check prerequisites
    if "df" not in st.session_state or not st.session_state.get("df_ready", False):
        st.warning("⚠️ Please complete **Data Input & Model Setup** first.")
        return

    df          = st.session_state["df"]
    assignments = st.session_state.get("assignments", {})
    validation  = st.session_state.get("validation", {})
    constructs  = st.session_state.get("constructs", {})

    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if not indicator_cols:
        st.warning("⚠️ No indicator variables assigned. Please go back to Data Input.")
        return

    # Render sections
    render_overview(df, validation)
    st.markdown("---")
    render_descriptive_table(df, indicator_cols)
    st.markdown("---")
    render_missing_analysis(df, indicator_cols)
    st.markdown("---")
    render_outlier_detection(df, indicator_cols)
    st.markdown("---")
    render_normality(df, indicator_cols)
    st.markdown("---")
    render_correlation_matrix(df, indicator_cols)
    st.markdown("---")
    render_estimator_recommendation()

    # Mark complete
    st.session_state["descriptive_complete"] = True

    st.markdown("---")
    st.success(
        "✅ **Descriptive analysis complete.** "
        "Proceed to **Exploratory Factor Analysis (EFA)** or **CFA** in the sidebar."
    )
