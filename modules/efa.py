"""
efa.py - Exploratory Factor Analysis Module.
Uses R/psych via r_bridge for methodologically correct EFA.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.interpretation import interpret_kmo, interpret_bartlett, interpret_loading

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


def render_factorability(result):
    st.subheader("Step 1: Factorability Tests")
    st.markdown(
        "**KMO** and **Bartlett's Test** confirm whether data are suitable for factor analysis."
    )

    kmo = result.get("kmo")
    if isinstance(kmo, list): kmo = kmo[0]
    kmo = float(kmo) if kmo is not None else None

    bart_chi2 = result.get("bartlett_chi2")
    bart_p    = result.get("bartlett_p")
    if isinstance(bart_chi2, list): bart_chi2 = bart_chi2[0]
    if isinstance(bart_p, list):    bart_p    = bart_p[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("KMO Overall",       f"{kmo:.3f}" if kmo else "N/A")
    c2.metric("Bartlett's chi2",   f"{float(bart_chi2):.3f}" if bart_chi2 else "N/A")
    c3.metric("Bartlett's p-value","< .0001" if bart_p and float(bart_p) < 0.0001 else f"{float(bart_p):.4f}" if bart_p else "N/A")

    if kmo:
        r = interpret_kmo(kmo)
        badge(r["level"], r["message"])
    if bart_p is not None:
        r = interpret_bartlett(float(bart_p))
        badge(r["level"], r["message"])

    # Per-item KMO
    kmo_per = result.get("kmo_per_item")
    if kmo_per and isinstance(kmo_per, dict):
        with st.expander("KMO Per Item (MSA — Measures of Sampling Adequacy)"):
            st.markdown("Items with MSA < .50 should be considered for removal.")
            msa_rows = []
            for item, val in kmo_per.items():
                v = float(val) if val is not None else 0
                msa_rows.append({
                    "Item":   item,
                    "MSA":    round(v, 3),
                    "Status": "Good" if v >= 0.70 else "Acceptable" if v >= 0.60 else "Warning" if v >= 0.50 else "Poor",
                })
            msa_df = pd.DataFrame(msa_rows).sort_values("MSA")

            def color_msa(val):
                try:
                    v = float(val)
                    if v >= 0.70: return "color:#1a7a4a;font-weight:600"
                    elif v >= 0.60: return "color:#1a6fa8"
                    elif v >= 0.50: return "color:#b7770d"
                    else: return "color:#c0392b;font-weight:600"
                except: return ""

            st.dataframe(
                msa_df.style.map(color_msa, subset=["MSA"]),
                use_container_width=True, hide_index=True
            )

    factorable = (kmo and kmo >= 0.60) and (bart_p and float(bart_p) < 0.05)
    return factorable


def render_factor_number(result):
    st.subheader("Step 2: Determining Number of Factors")
    st.markdown(
        "**Parallel Analysis** (gold standard via R/psych) + **Kaiser criterion** "
        "(eigenvalue > 1) + **Scree plot**."
    )

    suggested = result.get("suggested_factors")
    if isinstance(suggested, list): suggested = suggested[0]
    suggested = int(suggested) if suggested else 1

    # Scree plot using loadings SS
    var_explained = result.get("var_explained")
    if var_explained and isinstance(var_explained, dict):
        # Extract eigenvalues from SS loadings row
        ss_row = var_explained.get("SS loadings", var_explained.get("ss_loadings"))
        if ss_row:
            if isinstance(ss_row, dict):
                ev_vals = [float(v) for v in ss_row.values()]
            elif isinstance(ss_row, list):
                ev_vals = [float(v) for v in ss_row]
            else:
                ev_vals = []

            if ev_vals:
                ev_df = pd.DataFrame({
                    "Factor":     range(1, len(ev_vals)+1),
                    "SS Loading": ev_vals
                })
                fig = px.line(
                    ev_df, x="Factor", y="SS Loading",
                    markers=True, template="simple_white",
                    title="Scree Plot (SS Loadings from R/psych PAF)",
                )
                fig.add_hline(y=1, line_dash="dash", line_color="#b7770d",
                             annotation_text="Eigenvalue = 1 (Kaiser)")
                fig.update_layout(
                    height=350, font_color="#1a1a1a",
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff"
                )
                st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Parallel Analysis Suggests", f"{suggested} factor(s)")
    c2.metric("Recommendation", f"{suggested} factor(s)")

    badge("ok",
        f"Parallel analysis (R/psych, 100 iterations) suggests **{suggested}** factor(s). "
        "Always combine statistical criteria with theoretical justification."
    )

    n_factors = st.number_input(
        "Number of factors to extract:",
        min_value=1,
        max_value=20,
        value=suggested,
        step=1,
        key="efa_n_factors_widget"
    )
    return int(n_factors)


def render_loadings_table(result, item_names, n_factors):
    st.subheader("Step 4: Factor Loading Matrix")
    st.markdown(
        f"Factor loadings (PAF extraction via R/psych). "
        f"**Strong:** lambda >= .70 | **Acceptable:** lambda >= .50 | **Weak:** lambda < .50. "
        f"Cross-loadings > .32 are flagged."
    )

    loadings_raw = result.get("loadings")
    if loadings_raw is None:
        st.warning("No loadings available.")
        return None

    # Convert to DataFrame
    if isinstance(loadings_raw, pd.DataFrame):
        loadings_mat = loadings_raw
    elif isinstance(loadings_raw, dict):
        loadings_mat = pd.DataFrame(loadings_raw)
    elif isinstance(loadings_raw, list):
        if len(loadings_raw) > 0 and isinstance(loadings_raw[0], list):
            actual_cols = len(loadings_raw[0])
        else:
            actual_cols = n_factors
        loadings_mat = pd.DataFrame(
            np.array(loadings_raw),
            index=item_names[:len(loadings_raw)],
            columns=[f"F{i+1}" for i in range(actual_cols)]
        )
    else:
        st.warning("Could not parse loadings.")
        return None

    # Ensure index is item names
    if list(loadings_mat.index) != item_names[:len(loadings_mat)]:
        loadings_mat.index = item_names[:len(loadings_mat)]

    # Rename columns to F1, F2, ...
    factor_cols = [f"F{i+1}" for i in range(min(n_factors, loadings_mat.shape[1]))]
    loadings_mat.columns = factor_cols[:loadings_mat.shape[1]]

    # Communalities
    communalities_raw = result.get("communalities")
    if communalities_raw:
        if isinstance(communalities_raw, dict):
            comm_vals = [float(v) for v in communalities_raw.values()]
        elif isinstance(communalities_raw, list):
            comm_vals = [float(v) for v in communalities_raw]
        else:
            comm_vals = []
        if comm_vals:
            loadings_mat["h2"] = comm_vals[:len(loadings_mat)]

    # Color code
    def color_loading(val):
        try:
            v = float(val)
            if abs(v) >= 0.70: return "color:#1a7a4a;font-weight:700"
            elif abs(v) >= 0.50: return "color:#1a6fa8;font-weight:600"
            elif abs(v) >= 0.32: return "color:#b7770d"
            else: return "color:#aaaaaa"
        except: return ""

    display_cols = factor_cols[:loadings_mat.shape[1]]
    st.dataframe(
        loadings_mat.round(3).style.map(color_loading, subset=display_cols),
        use_container_width=True
    )

    # Variance explained
    var_explained = result.get("var_explained")
    if var_explained and isinstance(var_explained, dict):
        st.markdown("**Variance Explained:**")
        var_rows = []
        row_names = list(var_explained.keys())
        for rn in row_names:
            row_vals = var_explained[rn]
            if isinstance(row_vals, dict):
                vals = list(row_vals.values())
            elif isinstance(row_vals, list):
                vals = row_vals
            else:
                vals = [row_vals]
            var_rows.append({"Statistic": rn, **{f"F{i+1}": round(float(v), 3) for i, v in enumerate(vals)}})
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

        # Total variance
        cum_row = var_explained.get("Cumulative Var") or var_explained.get("cumulative_var")
        if cum_row:
            if isinstance(cum_row, dict): vals = list(cum_row.values())
            elif isinstance(cum_row, list): vals = cum_row
            else: vals = [cum_row]
            if vals:
                total_var = float(vals[-1]) if vals else 0
                if total_var >= 0.50:
                    badge("ok", f"Total variance explained: **{total_var:.1%}** (criterion: >= 50%). ✅")
                else:
                    badge("warning", f"Total variance explained: **{total_var:.1%}** (criterion: >= 50%). ⚠️ Consider adding factors.")

    # Cross-loading detection
    cross = []
    for item in loadings_mat.index:
        row = loadings_mat.loc[item, display_cols].abs()
        if (row >= 0.32).sum() > 1:
            cross.append(item)
    if cross:
        badge("warning", f"Cross-loadings detected: **{', '.join(cross)}**. These items load on multiple factors. Consider removing or rewriting.")
    else:
        badge("excellent", "No cross-loadings detected. Each item loads primarily on one factor. ✅")

    # Low communalities
    if "h2" in loadings_mat.columns:
        low_h2 = [str(idx) for idx in loadings_mat.index if loadings_mat.loc[idx, "h2"] < 0.40]
        if low_h2:
            badge("warning", f"Low communality (h2 < .40): **{', '.join(low_h2)}**. These items share little variance with their factors.")

    # Heatmap
    with st.expander("Factor Loading Heatmap"):
        fig = px.imshow(
            loadings_mat[display_cols].T,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template="simple_white",
            text_auto=".2f",
            title="Factor Loading Heatmap",
            aspect="auto",
        )
        fig.update_layout(
            height=max(200, n_factors * 70),
            font_color="#1a1a1a",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Item-by-item interpretation
    with st.expander("Item-by-Item Loading Interpretation"):
        for item in loadings_mat.index:
            row = loadings_mat.loc[item, display_cols].abs()
            best_f   = row.idxmax()
            best_val = loadings_mat.loc[item, best_f]
            r = interpret_loading(float(best_val), str(item))
            badge(r["level"], f"{r['message']} — Primary factor: **{best_f}**")

    return loadings_mat


def render_factor_naming(loadings_mat, n_factors):
    st.subheader("Step 5: Factor Naming")
    st.markdown(
        "Based on items loading on each factor, assign a **theoretical name**. "
        "This bridges EFA (data-driven) and CFA (theory-driven)."
    )

    factor_cols  = [f"F{i+1}" for i in range(n_factors) if f"F{i+1}" in loadings_mat.columns]
    factor_names = {}

    for f in factor_cols:
        top_items = loadings_mat[f].abs().nlargest(5).index.tolist()
        c1, c2 = st.columns([1, 2])
        with c1:
            name = st.text_input(
                f"Name for {f}",
                value=f"Factor{f[1:]}",
                key=f"efa_fname_{f}"
            )
            factor_names[f] = name
        with c2:
            st.markdown(f"**Top items loading on {f}:**")
            for item in top_items:
                val = loadings_mat.loc[item, f]
                st.markdown(f"  - {item}: lambda = `{val:.3f}`")

    st.session_state["efa_factor_names"] = factor_names
    return factor_names


def render_efa_summary(loadings_mat, n_factors, factor_names, item_names):
    st.subheader("Step 6: EFA Summary and CFA Preparation")
    st.markdown("Suggested construct structure for CFA based on EFA results.")

    factor_cols = [f"F{i+1}" for i in range(n_factors) if f"F{i+1}" in loadings_mat.columns]
    suggested   = {}

    for f in factor_cols:
        fname = factor_names.get(f, f)
        primary_items = [
            str(item) for item in loadings_mat.index
            if loadings_mat.loc[item, factor_cols].abs().idxmax() == f
            and abs(loadings_mat.loc[item, f]) >= 0.40
        ]
        suggested[fname] = primary_items
        n = len(primary_items)
        status = "OK" if n >= 3 else "Warning"
        st.markdown(f"**{fname}** ({n} items): {', '.join(primary_items)}")
        if n < 3:
            badge("warning", f"{fname} has only {n} item(s). CFA requires at least 3 indicators.")

    st.session_state["efa_suggested_constructs"] = suggested

    if st.button("Use EFA Results for CFA Model", type="primary", key="efa_to_cfa"):
        st.session_state["constructs"] = suggested
        # Rebuild CFA syntax
        cfa_lines = [f"{c} =~ {' + '.join(items)}" for c, items in suggested.items() if items]
        st.session_state["cfa_syntax"] = "\n".join(cfa_lines)
        # Rebuild SEM syntax
        sem_lines = cfa_lines.copy()
        for pred, out in st.session_state.get("structural_paths", []):
            sem_lines.append(f"{out} ~ {pred}")
        st.session_state["sem_syntax"] = "\n".join(sem_lines)
        badge("excellent", "EFA factor structure transferred to CFA. Navigate to CFA in the sidebar.")

    badge("ok",
        "Note: EFA is data-driven. Ensure the suggested factor structure is "
        "**theoretically justified** before proceeding to CFA. "
        "Do not proceed to CFA solely based on EFA statistics."
    )


def render_efa():
    st.title("Exploratory Factor Analysis (EFA)")
    st.markdown(
        "EFA is used when the factor structure is **unknown or unconfirmed**. "
        "It explores how items cluster empirically, providing the foundation for CFA.\n\n"
        "> If your instrument has been validated in prior research, you may skip EFA and go directly to CFA."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df             = st.session_state["df"]
    assignments    = st.session_state.get("assignments", {})
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if len(indicator_cols) < 3:
        st.error("At least 3 indicator variables are required for EFA.")
        return

    data = df[indicator_cols].dropna()
    st.info(f"Analyzing **{len(indicator_cols)} items** across **{len(data)} complete cases**.")
    st.markdown("---")

    # EFA settings
    st.subheader("Step 3: Extraction and Rotation Settings")
    col1, col2 = st.columns(2)
    with col1:
        rotation = st.selectbox(
            "Rotation Method",
            options=["oblimin", "varimax", "promax", "none"],
            format_func=lambda x: {
                "oblimin": "Oblimin (oblique — recommended for social science)",
                "varimax": "Varimax (orthogonal — assumes uncorrelated factors)",
                "promax":  "Promax (oblique — alternative)",
                "none":    "No rotation",
            }[x],
            key="efa_rotation_widget"
        )
    with col2:
        st.markdown("**Extraction Method:** Principal Axis Factoring (PAF)")
        st.markdown("*Standard for SEM/CFA preparation (Hair et al., 2019)*")

    badge("ok",
        "Recommendation: Use **PAF with oblimin rotation** for social science. "
        "Oblique rotation allows factors to correlate, which is realistic in behavioral research."
    )

    if st.button("Run EFA via R/psych", type="primary", key="run_efa_btn"):
        try:
            from r_scripts.r_bridge import run_efa, check_r_available
            r_check = check_r_available()

            if not r_check["available"]:
                st.error(f"R is not available: {r_check['message']}")
                return

            n_factors_preview = st.session_state.get("efa_n_factors_widget", 2)

            with st.spinner("Running EFA via R/psych... (PAF extraction, parallel analysis)"):
                result = run_efa(
                    df          = df,
                    indicator_cols = indicator_cols,
                    n_factors   = n_factors_preview,
                    rotation    = rotation
                )

            if "error" in result:
                st.error(f"EFA failed: {result['error']}")
                return

            st.session_state["efa_result"]     = result
            st.session_state["efa_rotation"]   = rotation
            st.session_state["efa_data_cols"]  = indicator_cols
            st.success("EFA completed successfully via R/psych.")

        except Exception as e:
            st.error(f"EFA error: {str(e)}")
            return

    result = st.session_state.get("efa_result")
    if result is None:
        st.info("Configure settings above and click **Run EFA** to begin.")
        return

    st.markdown("---")
    factorable = render_factorability(result)
    if not factorable:
        badge("critical", "Data may not be suitable for factor analysis. Review KMO and Bartlett results.")
    st.markdown("---")

    n_factors = render_factor_number(result)
    st.markdown("---")

    # Re-run if n_factors changed
    stored_result = st.session_state.get("efa_result", {})
    stored_nf     = stored_result.get("n_factors")
    if isinstance(stored_nf, list): stored_nf = stored_nf[0]
    if stored_nf and int(stored_nf) != n_factors:
        badge("warning", f"You changed the number of factors to {n_factors}. Click **Run EFA** again to update.")

    loadings_mat = render_loadings_table(result, indicator_cols, n_factors)
    if loadings_mat is None:
        return
    st.markdown("---")

    factor_names = render_factor_naming(loadings_mat, n_factors)
    st.markdown("---")

    render_efa_summary(loadings_mat, n_factors, factor_names, indicator_cols)

    st.session_state["efa_complete"] = True
    st.markdown("---")
    badge("excellent", "EFA complete. Proceed to Confirmatory Factor Analysis (CFA).")
