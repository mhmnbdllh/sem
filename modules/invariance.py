"""
invariance.py - Measurement Invariance Module.
Uses R/lavaan via r_bridge for methodologically correct invariance testing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.interpretation import interpret_invariance_level

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

def _safe_float(val, default=None):
    if val is None: return default
    try:
        if isinstance(val, list): val = val[0]
        f = float(val)
        return None if np.isnan(f) else f
    except: return default

def _extract_fit(fit_dict):
    if not isinstance(fit_dict, dict): return {}
    normalized = {}
    key_map = {
        "chisq": "chi2", "pvalue": "p", "dof": "df",
        "rmsea": "rmsea", "cfi": "cfi", "tli": "tli",
        "srmr": "srmr", "aic": "aic", "bic": "bic",
    }
    for k, v in fit_dict.items():
        nk = key_map.get(k.lower().replace(".", "_"), k.lower().replace(".", "_"))
        val = _safe_float(v)
        if val is not None:
            normalized[nk] = val
    return normalized


def render_group_setup(df, assignments):
    st.subheader("Step 1: Group Variable Setup")
    st.markdown(
        "Measurement invariance tests whether your measurement model "
        "**functions equivalently across groups** (e.g., gender, school, country). "
        "Select the grouping variable below."
    )

    demographic_cols = [c for c, r in assignments.items() if r == "demographic"]
    non_numeric      = df.select_dtypes(exclude=[np.number]).columns.tolist()
    grouping_options = list(set(demographic_cols + non_numeric))

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 10 and col not in grouping_options:
            grouping_options.append(col)

    if not grouping_options:
        st.warning(
            "No suitable grouping variables found. "
            "Ensure you have a demographic variable (e.g., gender, school) "
            "assigned in Data Input."
        )
        return None

    group_var = st.selectbox(
        "Select grouping variable",
        options=grouping_options,
        help="Variable that defines the groups (e.g., gender, school type)"
    )

    unique_vals = sorted(df[group_var].dropna().unique())
    st.markdown(f"Unique values in {group_var}: {list(unique_vals)}")

    if len(unique_vals) < 2:
        st.error("Grouping variable must have at least 2 unique values.")
        return None
    if len(unique_vals) > 10:
        st.warning("More than 10 unique values. Invariance testing works best with 2-4 groups.")

    c1, c2 = st.columns(2)
    with c1:
        group1 = st.selectbox("Group 1", options=unique_vals, key="inv_g1")
    with c2:
        group2_opts = [v for v in unique_vals if v != group1]
        group2 = st.selectbox("Group 2", options=group2_opts, key="inv_g2")

    n1 = int((df[group_var] == group1).sum())
    n2 = int((df[group_var] == group2).sum())

    col1, col2 = st.columns(2)
    col1.metric(f"n ({group1})", f"{n1:,}")
    col2.metric(f"n ({group2})", f"{n2:,}")

    if min(n1, n2) < 100:
        badge("warning",
            f"The smaller group has n = {min(n1,n2)}. "
            "Invariance testing requires n >= 100 per group for reliable results."
        )
    else:
        badge("ok", f"Both groups meet the minimum sample size (n >= 100). ✅")

    return group_var, group1, group2


def render_invariance_tests(df, constructs, group_var, group1, group2):
    st.subheader("Step 2: Invariance Model Tests")
    st.markdown(
        "Three sequential models are tested, each adding more constraints:\n\n"
        "1. Configural — same factor structure (least constrained)\n"
        "2. Metric — equal factor loadings across groups\n"
        "3. Scalar — equal item intercepts (most constrained)"
    )

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))
    cfa_syntax = st.session_state.get("cfa_syntax", "")

    if not cfa_syntax:
        st.error("CFA syntax not found. Please run CFA first.")
        return None

    estimator = st.session_state.get("recommended_estimator", "MLR")

    try:
        from r_scripts.r_bridge import run_invariance, check_r_available
        r_check = check_r_available()
        if not r_check["available"]:
            st.error(f"R is not available: {r_check['message']}")
            return None

        with st.spinner("Running configural, metric, and scalar invariance models via R/lavaan..."):
            result = run_invariance(
                df           = df,
                indicator_cols = all_items + [group_var],
                model_syntax = cfa_syntax,
                group_var    = group_var,
                estimator    = estimator
            )

        if "error" in result:
            st.error(f"Invariance testing failed: {result['error']}")
            return None

        st.session_state["invariance_results"] = result
        st.success("Invariance models estimated successfully.")
        return result

    except Exception as e:
        st.error(f"Invariance error: {str(e)}")
        return None


def render_fit_comparison(result, group1, group2):
    st.subheader("Step 3: Model Fit Comparison")

    model_names  = ["configural", "metric", "scalar"]
    labels       = ["Configural", "Metric", "Scalar"]

    # Build comparison table
    rows = []
    fits = {}
    for name, label in zip(model_names, labels):
        raw = result.get(name, {})
        fit = _extract_fit(raw) if isinstance(raw, dict) else {}
        fits[name] = fit

        chi2 = fit.get("chi2")
        df_  = fit.get("df")
        rows.append({
            "Model":  label,
            "chi2":   round(chi2, 3) if chi2 is not None else "—",
            "df":     int(df_) if df_ else "—",
            "RMSEA":  round(fit.get("rmsea", 0), 3) if fit.get("rmsea") else "—",
            "CFI":    round(fit.get("cfi",   0), 3) if fit.get("cfi")   else "—",
            "TLI":    round(fit.get("tli",   0), 3) if fit.get("tli")   else "—",
            "SRMR":   round(fit.get("srmr",  0), 3) if fit.get("srmr")  else "—",
        })

    fit_df = pd.DataFrame(rows)
    st.dataframe(
        fit_df.style
              .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
              .set_table_styles([{
                  "selector":"th",
                  "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
              }]),
        use_container_width=True, hide_index=True
    )

    # Difference tests
    st.markdown("**Model Comparison (Delta Tests):**")
    st.markdown(
        "Criterion: **ΔCFI >= -.010** supports invariance (Cheung & Rensvold, 2002). "
        "**ΔRMSEA <= .015** supports invariance (Chen, 2007)."
    )

    diff_rows   = []
    inv_results = {}

    comparisons = [
        ("configural", "metric",  "Metric vs Configural"),
        ("metric",     "scalar",  "Scalar vs Metric"),
    ]

    from scipy import stats as scipy_stats

    for base_name, constrained_name, label in comparisons:
        base_fit  = fits.get(base_name, {})
        cons_fit  = fits.get(constrained_name, {})

        chi2_base  = base_fit.get("chi2")
        df_base    = base_fit.get("df")
        chi2_cons  = cons_fit.get("chi2")
        df_cons    = cons_fit.get("df")
        cfi_base   = base_fit.get("cfi")
        cfi_cons   = cons_fit.get("cfi")
        rmsea_base = base_fit.get("rmsea")
        rmsea_cons = cons_fit.get("rmsea")

        delta_cfi   = None
        delta_rmsea = None
        delta_chi2  = None
        delta_df    = None
        p_chi2      = None

        if cfi_base and cfi_cons:
            delta_cfi = round(cfi_cons - cfi_base, 4)
        if rmsea_base and rmsea_cons:
            delta_rmsea = round(rmsea_cons - rmsea_base, 4)
        if chi2_base and chi2_cons and df_base and df_cons:
            delta_chi2 = abs(chi2_cons - chi2_base)
            delta_df   = abs(int(df_cons) - int(df_base))
            if delta_df > 0:
                p_chi2 = round(1 - scipy_stats.chi2.cdf(delta_chi2, delta_df), 4)

        # ΔCFI criterion (Cheung & Rensvold, 2002)
        cfi_ok   = delta_cfi is not None and delta_cfi >= -0.010
        rmsea_ok = delta_rmsea is not None and delta_rmsea <= 0.015
        supported = cfi_ok

        inv_results[constrained_name] = supported

        diff_rows.append({
            "Comparison":  label,
            "Delta chi2":  round(delta_chi2, 3) if delta_chi2 is not None else "—",
            "Delta df":    delta_df               if delta_df    else "—",
            "p(Delta chi2)":round(p_chi2, 4) if p_chi2 is not None else "—",
            "Delta CFI":   round(delta_cfi, 4) if delta_cfi is not None else "—",
            "Delta RMSEA": round(delta_rmsea, 4) if delta_rmsea is not None else "—",
            "Invariance":  "Supported" if supported else "Not Supported",
        })

    diff_df = pd.DataFrame(diff_rows)

    def color_inv(val):
        if val == "Supported":     return "color:#1a7a4a;font-weight:700"
        if val == "Not Supported": return "color:#c0392b;font-weight:700"
        return ""

    st.dataframe(
        diff_df.style.map(color_inv, subset=["Invariance"])
                     .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                     .set_table_styles([{
                         "selector":"th",
                         "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                     }]),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Note: ΔCFI >= -.010 supports invariance (Cheung & Rensvold, 2002). "
        "ΔRMSEA <= .015 supports invariance (Chen, 2007). "
        "ΔCFI is preferred over Δchi2 due to robustness to sample size."
    )

    # CFI comparison bar chart
    cfi_vals   = [fits[n].get("cfi") for n in model_names]
    rmsea_vals = [fits[n].get("rmsea") for n in model_names]

    if any(v is not None for v in cfi_vals):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="CFI", x=labels,
            y=[v if v else 0 for v in cfi_vals],
            marker_color="#1a6fa8", opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            name="RMSEA", x=labels,
            y=[v if v else 0 for v in rmsea_vals],
            marker_color="#b7770d", opacity=0.85,
        ))
        fig.add_hline(y=0.95, line_dash="dash", line_color="#1a7a4a",
                     annotation_text="CFI >= .95")
        fig.add_hline(y=0.06, line_dash="dot",  line_color="#c0392b",
                     annotation_text="RMSEA <= .06")
        fig.update_layout(
            barmode="group", template="simple_white", height=320,
            title="Fit Indices Across Invariance Models",
            yaxis=dict(range=[0, 1.05],
        margin=dict(t=60, b=40, l=40, r=120),
    ),
            legend=dict(orientation="h", y=-0.2, x=0),
            font_color="#1a1a1a",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

    return inv_results


def render_invariance_interpretation(inv_results, group1, group2):
    st.subheader("Step 4: Invariance Interpretation and Implications")

    configural_ok = True  # If we got here, configural worked
    metric_ok     = inv_results.get("metric",  False)
    scalar_ok     = inv_results.get("scalar",  False)

    interpretations = interpret_invariance_level(
        configural_ok, metric_ok, scalar_ok,
        str(group1), str(group2)
    )
    for r in interpretations:
        badge(r["level"], r["message"])

    # Practical implications table
    st.markdown("**What Can You Compare? (Practical Implications)**")
    impl_rows = [
        {
            "Invariance Level": "Configural",
            "What is Allowed":  "Factor structure is the same — basic comparability established",
            "Achieved":         "Yes",
        },
        {
            "Invariance Level": "Metric",
            "What is Allowed":  "Compare correlations, regression paths, and factor variances across groups",
            "Achieved":         "Yes" if metric_ok else "No",
        },
        {
            "Invariance Level": "Scalar",
            "What is Allowed":  "Compare latent means and all structural relationships across groups",
            "Achieved":         "Yes" if scalar_ok else "No",
        },
    ]
    impl_df = pd.DataFrame(impl_rows)

    def color_achieved(val):
        if val == "Yes": return "color:#1a7a4a;font-weight:700"
        return "color:#c0392b;font-weight:700"

    st.dataframe(
        impl_df.style.map(color_achieved, subset=["Achieved"])
                     .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                     .set_table_styles([{
                         "selector":"th",
                         "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                     }]),
        use_container_width=True, hide_index=True
    )

    # Reporting recommendation
    highest = (
        "scalar"     if scalar_ok  else
        "metric"     if metric_ok  else
        "configural"
    )
    badge("ok",
        f"**Reporting Recommendation:** Report that **{highest} invariance** was established "
        f"between {group1} and {group2}. "
        "Include the fit indices table and difference tests in your methods section."
    )

    st.session_state["invariance_level"] = highest


def render_invariance():
    st.title("Measurement Invariance")
    st.markdown(
        "Measurement invariance (MI) tests whether a measurement model "
        "**functions equivalently across different groups**. "
        "Without establishing MI, cross-group comparisons may be methodologically invalid.\n\n"
        "> MI is mandatory before comparing latent means or structural paths across groups "
        "(Vandenberg & Lance, 2000)."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    if not st.session_state.get("cfa_complete"):
        badge("warning", "Please complete CFA first before testing measurement invariance.")

    df          = st.session_state["df"]
    assignments = st.session_state.get("assignments", {})
    constructs  = st.session_state.get("constructs", {})

    if not constructs:
        st.error("No constructs defined. Complete Data Input first.")
        return

    st.markdown("---")
    setup = render_group_setup(df, assignments)
    if setup is None:
        return

    group_var, group1, group2 = setup
    st.markdown("---")

    if st.button("Run Invariance Tests via R/lavaan", type="primary",
                 key="run_inv_btn", use_container_width=True):
        result = render_invariance_tests(df, constructs, group_var, group1, group2)
        if result:
            st.markdown("---")
            inv_results = render_fit_comparison(result, group1, group2)
            st.markdown("---")
            render_invariance_interpretation(inv_results, group1, group2)

    elif st.session_state.get("invariance_results"):
        result = st.session_state["invariance_results"]
        st.info("Showing previously computed results. Click the button above to re-run.")
        st.markdown("---")
        inv_results = render_fit_comparison(result, group1, group2)
        st.markdown("---")
        render_invariance_interpretation(inv_results, group1, group2)

    st.markdown("---")
    badge("ok", "Measurement invariance complete. Proceed to Model Comparison or Export Report.")
