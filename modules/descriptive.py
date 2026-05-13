"""
descriptive.py - Descriptive Statistics & Assumption Testing Module.
Uses R/psych via r_bridge for Mardia's test and Harman's test.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats

from utils.interpretation import (
    interpret_sample_size, interpret_missing,
    interpret_skewness, interpret_kurtosis,
    interpret_mardia, interpret_kmo, interpret_bartlett
)

COLORS = {
    "excellent": "#1a7a4a",
    "good":      "#2ecc71",
    "ok":        "#1a6fa8",
    "warning":   "#b7770d",
    "critical":  "#c0392b",
}

def badge(level, message):
    color = COLORS.get(level, "#555555")
    import re
    # Convert **text** to <b>text</b> for HTML rendering
    message = re.sub(r'[*][*](.+?)[*][*]', r'<b>\1</b>', str(message))
    st.markdown(
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;'
        f'color:#1a1a1a;font-size:0.92rem">{message}</div>',
        unsafe_allow_html=True,
    )


def render_overview(df, validation):
    st.subheader("Dataset Overview")
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

    result = interpret_sample_size(n_rows)
    badge(result["level"], result["message"])


def render_descriptive_table(df, indicator_cols):
    st.subheader("Descriptive Statistics")
    st.markdown(
        "Key statistics for all indicator variables. "
        "**Skewness** and **kurtosis** assess normality for ML estimation."
    )

    rows = []
    for col in indicator_cols:
        x = df[col].dropna()
        if len(x) < 3:
            continue
        rows.append({
            "Variable": col,
            "N":        int(x.count()),
            "Mean":     round(float(x.mean()), 3),
            "SD":       round(float(x.std()), 3),
            "Min":      round(float(x.min()), 3),
            "Max":      round(float(x.max()), 3),
            "Skewness": round(float(scipy_stats.skew(x)), 3),
            "Kurtosis": round(float(scipy_stats.kurtosis(x)), 3),
            "Missing":  int(df[col].isna().sum()),
        })

    if not rows:
        st.warning("No valid indicator data to display.")
        return

    desc_df = pd.DataFrame(rows)

    # Color code skewness and kurtosis
    def color_skew(val):
        try:
            v = float(val)
            if abs(v) <= 1.0:  return "color:#1a7a4a;font-weight:600"
            elif abs(v) <= 2.0: return "color:#b7770d"
            else:               return "color:#c0392b;font-weight:600"
        except: return ""

    def color_kurt(val):
        try:
            v = float(val)
            if abs(v) <= 3.0:  return "color:#1a7a4a;font-weight:600"
            elif abs(v) <= 7.0: return "color:#b7770d"
            else:               return "color:#c0392b;font-weight:600"
        except: return ""

    styled = (
        desc_df.style
        .map(color_skew, subset=["Skewness"])
        .map(color_kurt, subset=["Kurtosis"])
        .set_properties(**{"text-align": "center", "color": "#1a1a1a"})
        .set_table_styles([{
            "selector": "th",
            "props": [
                ("background-color", "#2E86AB"),
                ("color", "white"),
                ("font-weight", "bold"),
                ("text-align", "center"),
                ("padding", "8px"),
            ]
        }, {
            "selector": "tr:nth-child(even)",
            "props": [("background-color", "#f0f4f8")]
        }])
    )
    st.dataframe(styled, use_container_width=True)

    # Item-by-item normality interpretation
    with st.expander("Item-by-Item Normality Interpretation"):
        st.markdown("Color coding: **Green** = normal | **Orange** = mild | **Red** = problematic")
        for row in rows:
            sk = interpret_skewness(row["Skewness"], row["Variable"])
            ku = interpret_kurtosis(row["Kurtosis"], row["Variable"])
            badge(sk["level"], sk["message"])
            badge(ku["level"], ku["message"])

    # Distribution plots
    with st.expander("Distribution Plots"):
        cols_to_plot = indicator_cols[:12]
        n_plot = len(cols_to_plot)
        grid = st.columns(min(4, n_plot))
        for i, col in enumerate(cols_to_plot):
            with grid[i % min(4, n_plot)]:
                fig = px.histogram(
                    df, x=col, nbins=10,
                    title=col,
                    color_discrete_sequence=["#2E86AB"],
                    template="simple_white",
                )
                fig.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=30, b=10),
                    showlegend=False,
                    title_font_size=12,
                    font_color="#1a1a1a",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                st.plotly_chart(fig, use_container_width=True)


def render_missing_analysis(df, indicator_cols):
    st.subheader("Missing Value Analysis")

    missing_df = pd.DataFrame({
        "Variable":  indicator_cols,
        "Missing N": [int(df[c].isna().sum()) for c in indicator_cols],
        "Missing %": [round(df[c].isna().mean() * 100, 2) for c in indicator_cols],
    }).sort_values("Missing %", ascending=False)

    total_missing = missing_df["Missing N"].sum()

    if total_missing == 0:
        badge("excellent", "No missing values detected. Your dataset is complete.")
        return

    # Bar chart
    fig = px.bar(
        missing_df[missing_df["Missing N"] > 0],
        x="Variable", y="Missing %",
        color="Missing %",
        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
        range_color=[0, 15],
        template="simple_white",
        title="Missing Values by Variable (%)",
    )
    fig.add_hline(y=5,  line_dash="dash", line_color="#b7770d", annotation_text="5% threshold")
    fig.add_hline(y=10, line_dash="dash", line_color="#c0392b", annotation_text="10% critical")
    fig.update_layout(height=320, font_color="#1a1a1a", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(missing_df, use_container_width=True, hide_index=True)

    # Interpretations
    with st.expander("Recommendations per Variable"):
        for _, row in missing_df.iterrows():
            pct = row["Missing %"] / 100
            if pct > 0:
                result = interpret_missing(pct, row["Variable"])
                badge(result["level"], result["message"])

    # Overall recommendation
    total_pct = df[indicator_cols].isna().mean().mean()
    if total_pct <= 0.05:
        badge("ok", f"Overall missing rate: {total_pct:.1%}. Listwise deletion or FIML is acceptable.")
    else:
        badge("warning", f"Overall missing rate: {total_pct:.1%}. FIML or multiple imputation is recommended.")


def render_outlier_detection(df, indicator_cols):
    st.subheader("Outlier Detection")

    tab1, tab2 = st.tabs(["Univariate (Z-score)", "Multivariate (Mahalanobis D2)"])

    with tab1:
        st.markdown("Univariate outliers: cases with |z| > 3.29 (p < .001).")
        data = df[indicator_cols].dropna()
        if len(data) < 5:
            st.warning("Not enough complete cases.")
            return
        z_scores     = np.abs(scipy_stats.zscore(data))
        outlier_mask = (z_scores > 3.29).any(axis=1)
        n_uni        = int(outlier_mask.sum())
        if n_uni == 0:
            badge("excellent", "No univariate outliers detected.")
        else:
            badge("warning", f"{n_uni} respondent(s) have at least one extreme univariate score (|z| > 3.29).")

    with tab2:
        st.markdown("Multivariate outliers: Mahalanobis D2, p < .001.")
        data = df[indicator_cols].dropna()
        if data.shape[0] < data.shape[1] + 2:
            st.warning("Not enough complete cases to compute Mahalanobis distance.")
            return
        try:
            mean_vec = data.mean().values
            cov_mat  = np.cov(data.values.T)
            if np.linalg.matrix_rank(cov_mat) < cov_mat.shape[0]:
                st.warning("Singular covariance matrix. Some items may be perfectly correlated.")
                return
            inv_cov = np.linalg.inv(cov_mat)
            from scipy.spatial.distance import mahalanobis
            d2 = np.array([mahalanobis(row, mean_vec, inv_cov)**2 for row in data.values])
            p_vals = 1 - scipy_stats.chi2.cdf(d2, df=data.shape[1])
            outlier_flag = p_vals < 0.001
            n_mv = int(outlier_flag.sum())
            n_total = len(data)
            pct = n_mv / n_total

            if n_mv == 0:
                badge("excellent", "No multivariate outliers detected (Mahalanobis D2, p < .001).")
            elif pct <= 0.02:
                badge("ok", f"{n_mv} multivariate outlier(s) detected ({pct:.1%} of sample). Review individually.")
            else:
                badge("warning", f"{n_mv} multivariate outlier(s) detected ({pct:.1%} of sample). Robust estimation (MLR) is recommended.")

            plot_df = pd.DataFrame({"Respondent": data.index, "D2": d2, "Outlier": outlier_flag})
            chi2_crit = scipy_stats.chi2.ppf(0.999, df=data.shape[1])
            fig = px.scatter(
                plot_df, x="Respondent", y="D2",
                color="Outlier",
                color_discrete_map={True: "#c0392b", False: "#2E86AB"},
                template="simple_white",
                title="Mahalanobis D2 per Respondent",
            )
            fig.add_hline(y=chi2_crit, line_dash="dash", line_color="#c0392b",
                         annotation_text=f"Critical value (p=.001) = {chi2_crit:.2f}")
            fig.update_layout(height=320, font_color="#1a1a1a", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state["mv_outliers"] = list(plot_df[outlier_flag]["Respondent"])
        except Exception as e:
            st.error(f"Error computing Mahalanobis distance: {str(e)}")


def render_normality(df, indicator_cols):
    st.subheader("Multivariate Normality — Mardia's Test (via R/psych)")
    st.markdown(
        "Mardia's test assesses multivariate normality — a key assumption for ML estimation. "
        "Violation suggests using **MLR** (Robust ML) instead."
    )

    data = df[indicator_cols].dropna()
    if len(data) < 50:
        st.warning("Too few complete cases for reliable normality testing (minimum 50).")
        return

    try:
        from r_scripts.r_bridge import run_mardia, check_r_available
        r_check = check_r_available()
        if not r_check["available"]:
            badge("warning", f"R not available: {r_check['message']}. Using Python fallback.")
            _render_normality_python(data)
            return

        with st.spinner("Running Mardia's test via R/psych..."):
            result = run_mardia(df, indicator_cols)

        if "error" in result:
            badge("warning", f"R error: {result['error']}. Using Python fallback.")
            _render_normality_python(data)
            return

        sk_p = result.get("skewness_p")
        ku_p = result.get("kurtosis_p")

        c1, c2 = st.columns(2)
        c1.metric("Mardia's Skewness", f"{result.get('skewness', 'N/A'):.3f}" if result.get("skewness") else "N/A",
                  f"p = {sk_p:.4f}" if sk_p else "N/A")
        c2.metric("Mardia's Kurtosis (z)", f"{result.get('kurtosis', 'N/A'):.3f}" if result.get("kurtosis") else "N/A",
                  f"p = {ku_p:.4f}" if ku_p else "N/A")

        interp = interpret_mardia(sk_p, ku_p)
        badge(interp["level"], interp["message"])
        st.session_state["recommended_estimator"] = interp.get("estimator", "MLR")

    except Exception as e:
        badge("warning", f"Could not run Mardia's test: {str(e)}. Using Python fallback.")
        _render_normality_python(data)


def _render_normality_python(data):
    """Python fallback for normality check using univariate skewness/kurtosis."""
    st.markdown("*Univariate normality check (Python fallback):*")
    skews = [abs(float(scipy_stats.skew(data[c].dropna()))) for c in data.columns]
    kurts = [abs(float(scipy_stats.kurtosis(data[c].dropna()))) for c in data.columns]
    max_skew = max(skews) if skews else 0
    max_kurt = max(kurts) if kurts else 0
    if max_skew <= 2.0 and max_kurt <= 7.0:
        badge("ok", f"Max |skewness| = {max_skew:.3f}, max |kurtosis| = {max_kurt:.3f}. Data appears approximately normal. ML estimation may be appropriate.")
        st.session_state["recommended_estimator"] = "ML"
    else:
        badge("warning", f"Max |skewness| = {max_skew:.3f}, max |kurtosis| = {max_kurt:.3f}. Non-normality detected. MLR estimation is recommended.")
        st.session_state["recommended_estimator"] = "MLR"


def render_correlation_matrix(df, indicator_cols):
    st.subheader("Correlation Matrix")
    st.markdown("Pearson correlations among all indicator variables. Correlations > .85 may indicate multicollinearity.")

    data = df[indicator_cols].dropna()
    corr = data.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="simple_white",
        title="Correlation Heatmap",
        text_auto=".2f",
        aspect="auto",
    )
    fig.update_layout(
        height=max(350, len(indicator_cols) * 35),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Correlation Table (lower triangle)"):
        from utils.apa_tables import correlation_table
        apa_corr = correlation_table(df, indicator_cols)
        st.dataframe(apa_corr, use_container_width=True)
        st.caption("Note: * p < .05;  p < .01; * p < .001")

    # Multicollinearity check
    high = [(indicator_cols[i], indicator_cols[j], corr.iloc[i,j])
            for i in range(len(indicator_cols))
            for j in range(i+1, len(indicator_cols))
            if abs(corr.iloc[i,j]) > 0.85]
    if high:
        st.warning("High correlations detected (r > .85) — potential multicollinearity:")
        for c1, c2, r in high:
            st.markdown(f"  - **{c1}** and **{c2}**: r = {r:.3f}")
    else:
        badge("excellent", "No problematic multicollinearity detected (all r <= .85).")


def render_harman_test(df, indicator_cols):
    st.subheader("Common Method Bias — Harman's Single Factor Test")
    st.markdown(
        "When all data come from a single source (self-report), common method bias (CMB) "
        "may inflate correlations. **Criterion:** If a single factor explains > 50% of variance, "
        "CMB is a concern."
    )

    try:
        from r_scripts.r_bridge import run_harman, check_r_available
        r_check = check_r_available()

        if r_check["available"]:
            with st.spinner("Running Harman's test via R/psych..."):
                result = run_harman(df, indicator_cols)

            if "error" not in result:
                prop = result.get("single_factor_var", 0)
                if isinstance(prop, list): prop = prop[0]
                prop = float(prop)

                c1, c2 = st.columns(2)
                c1.metric("Variance by First Factor", f"{prop:.1%}")
                c2.metric("Criterion", "< 50%")

                if prop < 0.50:
                    badge("ok", f"Harman's single factor explains {prop:.1%} of variance — below the 50% criterion. Common method bias is not a major concern.")
                else:
                    badge("warning", f"Harman's single factor explains {prop:.1%} — above 50% criterion. CMB may be present. Report as a limitation.")

                evs = result.get("eigenvalues", [])
                if evs:
                    if isinstance(evs, (int, float)): evs = [evs]
                    ev_df = pd.DataFrame({"Factor": range(1, len(evs)+1), "Eigenvalue": [float(e) for e in evs]})
                    fig = px.line(ev_df, x="Factor", y="Eigenvalue", markers=True,
                                 template="simple_white", title="Eigenvalue Scree Plot (Harman's Test)")
                    fig.add_hline(y=1, line_dash="dash", line_color="#b7770d", annotation_text="Eigenvalue = 1")
                    fig.update_layout(height=300, font_color="#1a1a1a", plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
                    st.plotly_chart(fig, use_container_width=True)
                return

        # Python fallback
        data   = df[indicator_cols].dropna()
        corr   = data.corr().values
        ev     = np.linalg.eigvalsh(corr)[::-1]
        ev     = ev[ev > 0]
        prop   = float(ev[0] / ev.sum()) if ev.sum() > 0 else 0

        c1, c2 = st.columns(2)
        c1.metric("Variance by First Factor", f"{prop:.1%}")
        c2.metric("Criterion", "< 50%")

        if prop < 0.50:
            badge("ok", f"Harman's: {prop:.1%} variance by first factor — below 50% criterion. CMB is not a major concern.")
        else:
            badge("warning", f"Harman's: {prop:.1%} variance by first factor — above 50% criterion. CMB may be present.")

    except Exception as e:
        badge("warning", f"Could not run Harman's test: {str(e)}")


def render_estimator_recommendation():
    st.subheader("Recommended Estimator")
    estimator = st.session_state.get("recommended_estimator", None)

    info = {
        "ML": {
            "name":  "Maximum Likelihood (ML)",
            "when":  "Multivariate normality is satisfied",
            "pros":  "Most efficient; widely accepted; provides standard fit indices",
            "cons":  "Sensitive to non-normality",
            "color": "#1a7a4a",
        },
        "MLR": {
            "name":  "Robust Maximum Likelihood (MLR)",
            "when":  "Multivariate normality is violated",
            "pros":  "Satorra-Bentler corrected chi-square; robust standard errors",
            "cons":  "Slightly less efficient than ML under normality",
            "color": "#b7770d",
        },
    }

    if estimator and estimator in info:
        d = info[estimator]
        st.markdown(
            f'<div style="background:{d["color"]}15;border:2px solid {d["color"]};'
            f'border-radius:8px;padding:16px;color:#1a1a1a">'
            f'<h4 style="color:{d["color"]};margin:0 0 8px 0">Recommended: {d["name"]}</h4>'
            f'<b>When to use:</b> {d["when"]}<br>'
            f'<b>Advantages:</b> {d["pros"]}<br>'
            f'<b>Limitations:</b> {d["cons"]}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Run the normality test above to receive an automatic estimator recommendation.")

    st.markdown("**Override estimator (optional):**")
    override = st.selectbox(
        "Select estimator manually if needed:",
        options=["Auto (recommended)", "ML", "MLR"],
        key="estimator_override"
    )
    if override != "Auto (recommended)":
        st.session_state["recommended_estimator"] = override
        badge("ok", f"Estimator manually set to **{override}**.")


def render_descriptive():
    st.title("Descriptive Statistics and Assumption Testing")
    st.markdown(
        "Complete pre-analysis examination of your data. "
        "Every output includes automatic interpretation to guide methodological decisions."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df             = st.session_state["df"]
    assignments    = st.session_state.get("assignments", {})
    validation     = st.session_state.get("validation", {})
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if not indicator_cols:
        st.warning("No indicator variables assigned. Please go back to Data Input.")
        return

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
    render_harman_test(df, indicator_cols)
    st.markdown("---")
    render_estimator_recommendation()

    st.session_state["descriptive_complete"] = True
    st.markdown("---")
    badge("excellent", "Descriptive analysis complete. Proceed to EFA or CFA in the sidebar.")
