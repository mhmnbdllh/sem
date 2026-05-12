"""
sem_model.py - Structural Equation Model Module.
Uses R/lavaan via r_bridge for methodologically correct SEM.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.interpretation import (
    interpret_path, interpret_r2,
    interpret_fit_index, interpret_overall_fit
)
from utils.apa_tables import fit_indices_table, structural_paths_table

COLORS = {
    "excellent": "#1a7a4a",
    "good":      "#2ecc71",
    "ok":        "#1a6fa8",
    "warning":   "#b7770d",
    "critical":  "#c0392b",
}

def badge(level, message):
    color = COLORS.get(level, "#555555")
    st.markdown(
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;'
        f'color:#1a1a1a;font-size:0.92rem">{message}</div>',
        unsafe_allow_html=True,
    )


def build_sem_syntax(constructs, structural_paths):
    lines = []
    lines.append("# Measurement Model")
    for c, items in constructs.items():
        if items:
            lines.append(f"{c} =~ {' + '.join(items)}")
    lines.append("")
    lines.append("# Structural Model")
    for pred, out in structural_paths:
        lines.append(f"{out} ~ {pred}")
    return "\n".join(lines)


def render_sem_spec(constructs, structural_paths):
    st.subheader("Step 1: Full SEM Model Specification")
    st.markdown(
        "The syntax combines your **measurement model** (from CFA) "
        "and **structural paths** (your hypotheses). Edit directly if needed."
    )

    syntax = build_sem_syntax(constructs, structural_paths)
    current = st.session_state.get("sem_syntax", syntax)

    edited = st.text_area(
        "lavaan Full SEM Syntax",
        value=current,
        height=max(200, (len(constructs) + len(structural_paths) + 3) * 28),
        help="=~ defines measurement model; ~ defines structural paths"
    )
    st.session_state["sem_syntax"] = edited

    with st.expander("Hypothesis Summary"):
        if structural_paths:
            for i, (pred, out) in enumerate(structural_paths):
                st.markdown(f"**H{i+1}:** {pred} --> {out}")
        else:
            st.warning("No structural paths defined. Go back to Data Input.")

    all_outcomes   = set(out for _, out in structural_paths)
    all_predictors = set(pred for pred, _ in structural_paths)
    exogenous  = all_predictors - all_outcomes
    endogenous = all_outcomes

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Exogenous constructs** (predictors only):")
        for c in sorted(exogenous):
            st.markdown(f"  - {c}")
    with c2:
        st.markdown("**Endogenous constructs** (have incoming paths):")
        for c in sorted(endogenous):
            st.markdown(f"  - {c}")

    st.session_state["sem_endogenous"] = list(endogenous)
    st.session_state["sem_exogenous"]  = list(exogenous)
    return edited


def render_sem_estimation(syntax, df, constructs):
    st.subheader("Step 2: SEM Estimation")

    estimator = st.session_state.get("recommended_estimator", "MLR")
    st.info(f"**Estimator:** {estimator} — based on Mardia's normality test.")

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))

    if st.button("Run SEM via R/lavaan", type="primary", key="run_sem_btn", use_container_width=True):
        try:
            from r_scripts.r_bridge import run_sem, check_r_available
            r_check = check_r_available()
            if not r_check["available"]:
                st.error(f"R is not available: {r_check['message']}")
                return None

            with st.spinner("Estimating full SEM via R/lavaan... please wait."):
                result = run_sem(
                    df          = df,
                    all_cols    = all_items,
                    model_syntax= syntax,
                    estimator   = estimator
                )

            if "error" in result:
                st.error(f"SEM estimation failed: {result['error']}")
                st.markdown("**Troubleshooting:**")
                st.markdown("- Ensure all item names match your data columns exactly")
                st.markdown("- Verify the measurement model (CFA) was validated first")
                st.markdown("- Check for missing values in indicator columns")
                return None

            n = result.get("n")
            if isinstance(n, list): n = n[0]
            converged = result.get("converged")
            if isinstance(converged, list): converged = converged[0]

            st.session_state["sem_result"]    = result
            st.session_state["sem_all_items"] = all_items
            st.success(f"SEM estimated on n = {int(float(n)) if n else 'N/A'} complete cases. Converged: {converged}")

        except Exception as e:
            st.error(f"SEM error: {str(e)}")
            return None

    return st.session_state.get("sem_result")


def render_sem_fit(result):
    st.subheader("Step 3: Model Fit Assessment")
    st.markdown(
        "Fit indices for the **full SEM**. "
        "Compare with CFA fit — SEM fit should not be substantially worse."
    )

    fit_raw = result.get("fit_indices")
    if fit_raw is None:
        st.warning("No fit indices available.")
        return {}

    if isinstance(fit_raw, dict):
        fit = {k: (v[0] if isinstance(v, list) else v) for k, v in fit_raw.items()}
    else:
        fit = {}

    # Normalize
    key_map = {
        "chisq": "chi2", "pvalue": "p", "dof": "df",
        "rmsea_robust": "rmsea", "cfi_robust": "cfi", "tli_robust": "tli",
    }
    normalized = {}
    for k, v in fit.items():
        nk = key_map.get(k, k.lower().replace(".", "_"))
        if v is not None:
            try: normalized[nk] = float(v)
            except: pass

    for base in ["rmsea", "cfi", "tli", "chisq", "pvalue"]:
        for suffix in ["_scaled", "scaled"]:
            sk = f"{base}{suffix}"
            if sk in normalized and base not in normalized:
                normalized[base] = normalized[sk]

    # Metric cards
    c1, c2, c3, c4, c5 = st.columns(5)
    chi2 = normalized.get("chi2")
    df_  = normalized.get("df")
    for col, key in [(c1,"rmsea"),(c2,"cfi"),(c3,"tli"),(c4,"srmr"),(c5,"aic")]:
        val = normalized.get(key)
        col.metric(key.upper(), f"{val:.3f}" if val is not None else "—")

    if chi2 and df_ and float(df_) > 0:
        ratio = float(chi2) / float(df_)
        badge(
            "ok" if ratio <= 5 else "warning",
            f"chi2/df = {ratio:.3f} — {'Good' if ratio <= 2 else 'Acceptable' if ratio <= 5 else 'Poor'} "
            f"(criterion: <= 2.0 good; <= 5.0 acceptable)"
        )

    # Full table
    st.markdown("**Complete Fit Indices:**")
    fit_df = fit_indices_table(normalized)
    if not fit_df.empty:
        st.dataframe(
            fit_df.style.set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                        .set_table_styles([{
                            "selector":"th",
                            "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                        }]),
            use_container_width=True, hide_index=True
        )

    with st.expander("Fit Index Interpretations"):
        for idx in ["rmsea", "cfi", "tli", "srmr"]:
            val = normalized.get(idx)
            if val is not None:
                r = interpret_fit_index(idx, val)
                badge(r["level"], r["message"])

    overall = interpret_overall_fit(normalized)
    badge(
        "ok" if "acceptable" in overall.lower() or "good" in overall.lower() else
        "warning" if "marginal" in overall.lower() else "critical",
        overall
    )

    # Compare with CFA fit
    cfa_fit = st.session_state.get("cfa_fit", {})
    if cfa_fit:
        st.markdown("**CFA vs SEM Fit Comparison:**")
        comp_rows = []
        for idx in ["rmsea", "cfi", "tli", "srmr", "aic", "bic"]:
            cfa_val = cfa_fit.get(idx)
            sem_val = normalized.get(idx)
            if cfa_val is not None and sem_val is not None:
                change = sem_val - cfa_val
                worse = (
                    (idx in ["rmsea","srmr"] and change > 0.01) or
                    (idx in ["cfi","tli"] and change < -0.01)
                )
                comp_rows.append({
                    "Index":  idx.upper(),
                    "CFA":    round(cfa_val, 3),
                    "SEM":    round(sem_val, 3),
                    "Change": round(change, 3),
                    "Note":   "Warning" if worse else "OK",
                })
        if comp_rows:
            comp_df = pd.DataFrame(comp_rows)
            def color_note(val):
                if val == "OK": return "color:#1a7a4a;font-weight:600"
                return "color:#b7770d;font-weight:600"
            st.dataframe(
                comp_df.style.map(color_note, subset=["Note"])
                             .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                             .set_table_styles([{
                                 "selector":"th",
                                 "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                             }]),
                use_container_width=True, hide_index=True
            )

    st.session_state["sem_fit"] = normalized
    return normalized


def render_structural_paths(result, structural_paths):
    st.subheader("Step 4: Structural Path Coefficients")
    st.markdown(
        "Path coefficients (beta) represent **direct effects** between latent constructs. "
        "Standardized beta allows comparison across paths. "
        "Significance assessed at p < .05 (two-tailed)."
    )

    paths_raw = result.get("paths")
    if paths_raw is None:
        st.warning("No structural paths available.")
        return []

    if isinstance(paths_raw, pd.DataFrame):
        paths_df = paths_raw
    elif isinstance(paths_raw, dict):
        paths_df = pd.DataFrame(paths_raw)
    else:
        st.warning("Could not parse structural paths.")
        return []

    if paths_df.empty:
        st.warning("No structural paths found in results.")
        return []

    # Normalize column names
    col_map = {}
    for col in paths_df.columns:
        cl = col.lower().replace(".", "_")
        if cl in ["lhs","outcome"]:     col_map[col] = "outcome"
        elif cl in ["rhs","predictor"]: col_map[col] = "predictor"
        elif "std" in cl:               col_map[col] = "beta"
        elif "se" == cl:                col_map[col] = "se"
        elif "z" == cl:                 col_map[col] = "z"
        elif "pvalue" in cl or cl == "p": col_map[col] = "p"
        elif "est" == cl:               col_map[col] = "unstd"
    paths_df = paths_df.rename(columns=col_map)

    # Filter to hypothesized paths only
    hyp_pairs = [(str(pred), str(out)) for pred, out in structural_paths]
    path_results = []

    for _, row in paths_df.iterrows():
        pred = str(row.get("predictor", ""))
        out  = str(row.get("outcome", ""))
        if not hyp_pairs or (pred, out) in hyp_pairs:
            try:
                beta  = float(row.get("beta",  row.get("unstd", 0)))
                se    = float(row.get("se",    0))
                z     = float(row.get("z",     0))
                p_val = float(row.get("p",     1))
                ci_lo = beta - 1.96 * se
                ci_hi = beta + 1.96 * se
                path_results.append({
                    "predictor": pred, "outcome": out,
                    "beta": round(beta, 3), "se": round(se, 3),
                    "z": round(z, 3), "p": round(p_val, 4),
                    "ci_lower": round(ci_lo, 3),
                    "ci_upper": round(ci_hi, 3),
                })
            except (TypeError, ValueError):
                pass

    if not path_results:
        st.warning("Could not extract path coefficients. Check model syntax and results.")
        return []

    # APA table
    apa_df = structural_paths_table(path_results, hyp_pairs)
    if not apa_df.empty:
        def color_decision(val):
            if "Supported" in str(val) and "Not" not in str(val):
                return "color:#1a7a4a;font-weight:700"
            return "color:#c0392b;font-weight:700"

        st.dataframe(
            apa_df.style.map(color_decision, subset=["Decision"])
                        .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                        .set_table_styles([{
                            "selector":"th",
                            "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                        }]),
            use_container_width=True, hide_index=True
        )
        st.caption("Note: * p < .05; ** p < .01; *** p < .001; dag p < .10")

    # Hypothesis testing table
    st.markdown("**Hypothesis Testing Results:**")
    hyp_rows = []
    for i, p in enumerate(path_results):
        hyp_rows.append({
            "Hypothesis": f"H{i+1}",
            "Path":       f"{p['predictor']} --> {p['outcome']}",
            "beta":       f"{p['beta']:.3f}",
            "p":          f"{p['p']:.3f}",
            "Decision":   "Supported" if p["p"] < 0.05 else "Not Supported",
        })
    hyp_df = pd.DataFrame(hyp_rows)
    def color_hyp(val):
        if val == "Supported": return "color:#1a7a4a;font-weight:700"
        return "color:#c0392b;font-weight:700"
    st.dataframe(
        hyp_df.style.map(color_hyp, subset=["Decision"])
                    .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                    .set_table_styles([{
                        "selector":"th",
                        "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                    }]),
        use_container_width=True, hide_index=True
    )

    # Path interpretations
    with st.expander("Path-by-Path Interpretation"):
        for p in path_results:
            r = interpret_path(p["beta"], p["se"], p["p"], p["predictor"], p["outcome"])
            badge(r["level"], r["message"])

    # Effect sizes (Cohen's f2)
    st.markdown("**Effect Sizes (Cohen's f2):**")
    ef_rows = []
    for p in path_results:
        b = p["beta"]
        f2 = b**2 / (1 - b**2) if abs(b) < 1 else None
        size = (
            "Large"      if f2 and f2 >= 0.35 else
            "Medium"     if f2 and f2 >= 0.15 else
            "Small"      if f2 and f2 >= 0.02 else
            "Negligible"
        )
        ef_rows.append({
            "Path":   f"{p['predictor']} --> {p['outcome']}",
            "beta":   round(b, 3),
            "f2":     round(f2, 3) if f2 else "—",
            "Effect": size,
        })
    st.dataframe(pd.DataFrame(ef_rows).style.set_properties(**{"color":"#1a1a1a"}),
                use_container_width=True, hide_index=True)
    st.caption("Note: f2 >= .02 = small; >= .15 = medium; >= .35 = large (Cohen, 1988).")

    # Effect size bar chart
    betas  = [p["beta"] for p in path_results]
    labels = [f"{p['predictor']} to {p['outcome']}" for p in path_results]
    p_vals = [p["p"] for p in path_results]
    colors = [
        "#1a7a4a" if (pv < 0.05 and b > 0) else
        "#c0392b" if (pv < 0.05 and b < 0) else
        "#aaaaaa"
        for b, pv in zip(betas, p_vals)
    ]
    fig = go.Figure(go.Bar(
        x=betas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"beta={b:.3f}" for b in betas],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#555", line_width=1)
    fig.update_layout(
        template="simple_white", height=max(280, len(path_results)*55+100),
        title="Standardized Path Coefficients (beta)",
        xaxis_title="Standardized Coefficient (beta)",
        xaxis=dict(range=[-1, 1]),
        margin=dict(l=180, r=80, t=60, b=40),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green = significant positive; Red = significant negative; Gray = non-significant.")

    st.session_state["sem_paths"] = path_results
    return path_results


def render_r_squared(result, endogenous):
    st.subheader("Step 5: Explained Variance (R2)")
    st.markdown(
        "R2 indicates the proportion of variance in each **endogenous construct** "
        "explained by its predictors. "
        "Criterion: R2 >= .26 (substantial), >= .13 (moderate), >= .02 (weak) — Cohen (1988)."
    )

    r2_raw = result.get("r2")
    if r2_raw is None:
        st.warning("R2 values not available.")
        return

    if isinstance(r2_raw, dict):
        r2_dict = {}
        for k, v in r2_raw.items():
            try: r2_dict[k] = float(v[0] if isinstance(v, list) else v)
            except: pass
    else:
        st.warning("Could not parse R2 values.")
        return

    r2_rows = []
    for construct, r2 in r2_dict.items():
        if construct in endogenous or not endogenous:
            result_interp = interpret_r2(r2, construct)
            badge(result_interp["level"], result_interp["message"])
            r2_rows.append({
                "Construct": construct,
                "R2":        round(r2, 3),
                "R2 (%)":   f"{r2:.1%}",
                "Level":    result_interp["level"].capitalize(),
            })

    if r2_rows:
        r2_df = pd.DataFrame(r2_rows)
        plot_data = r2_df[r2_df["R2"].apply(lambda x: isinstance(x, float))].copy()
        if not plot_data.empty:
            fig = px.bar(
                plot_data, x="Construct", y="R2",
                color="R2",
                color_continuous_scale=["#c0392b","#f39c12","#1a7a4a"],
                range_color=[0, 0.5],
                template="simple_white",
                title="R2 by Endogenous Construct",
                text="R2 (%)",
            )
            fig.add_hline(y=0.26, line_dash="dash", line_color="#1a7a4a", annotation_text="Substantial (.26)")
            fig.add_hline(y=0.13, line_dash="dash", line_color="#b7770d", annotation_text="Moderate (.13)")
            fig.add_hline(y=0.02, line_dash="dot",  line_color="#c0392b", annotation_text="Weak (.02)")
            fig.update_layout(
                height=350, yaxis=dict(range=[0, 1]),
                font_color="#1a1a1a",
                plot_bgcolor="#ffffff",
                paper_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.session_state["sem_r2"] = r2_rows


def render_sem_checklist(fit, path_results):
    st.subheader("Step 6: SEM Methodological Checklist")

    from utils.thresholds import FIT
    cfa_complete = st.session_state.get("cfa_complete", False)
    n_sig = sum(1 for p in path_results if p.get("p", 1) < 0.05)
    n_total = len(path_results)
    df = st.session_state.get("df")

    checks = {
        "CFA measurement model validated first":
            cfa_complete,
        f"Model fit — RMSEA <= {FIT['rmsea_acceptable']}":
            (fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"],
        f"Model fit — CFI >= {FIT['cfi_acceptable']}":
            (fit.get("cfi") or 0) >= FIT["cfi_acceptable"],
        f"Model fit — SRMR <= {FIT['srmr_acceptable']}":
            (fit.get("srmr") or 999) <= FIT["srmr_acceptable"],
        "At least one significant structural path":
            n_sig > 0,
        "Structural paths theoretically justified":
            n_total > 0,
        "Sample size >= 200":
            len(df) >= 200 if df is not None else False,
    }

    rows = [{"Check": k, "Status": "Pass" if v else "Fail"} for k, v in checks.items()]
    cdf  = pd.DataFrame(rows)

    def color_status(val):
        if val == "Pass": return "color:#1a7a4a;font-weight:700"
        return "color:#c0392b;font-weight:700"

    st.dataframe(
        cdf.style.map(color_status, subset=["Status"])
                 .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                 .set_table_styles([{
                     "selector":"th",
                     "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                 }]),
        use_container_width=True, hide_index=True
    )

    n_fail = sum(1 for v in checks.values() if not v)
    if n_fail == 0:
        badge("excellent",
            f"All SEM criteria passed! {n_sig}/{n_total} hypotheses supported. "
            "Proceed to Mediation/Moderation or Export Report. ✅"
        )
        st.session_state["sem_complete"] = True
    elif n_fail <= 2:
        badge("warning", f"{n_fail} criterion/criteria require attention.")
        st.session_state["sem_complete"] = False
    else:
        badge("critical", f"{n_fail} criteria not met. Re-examine model before reporting.")
        st.session_state["sem_complete"] = False


def render_sem():
    st.title("Structural Equation Model (SEM)")
    st.markdown(
        "The **structural model** tests the hypothesized directional relationships "
        "between latent constructs. It combines the **measurement model** (CFA) "
        "with **structural paths** (your hypotheses).\n\n"
        "> The measurement model (CFA) must be validated before interpreting structural paths."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    if not st.session_state.get("cfa_complete"):
        badge("warning",
            "CFA has not been validated yet. "
            "Complete Confirmatory Factor Analysis before running SEM."
        )

    df               = st.session_state["df"]
    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])

    if not constructs:
        st.error("No constructs defined. Return to Data Input.")
        return
    if not structural_paths:
        st.error("No structural paths defined. Return to Data Input to define hypotheses.")
        return

    st.markdown("---")
    syntax = render_sem_spec(constructs, structural_paths)
    st.markdown("---")
    result = render_sem_estimation(syntax, df, constructs)

    if result is None:
        st.info("Run the SEM model above to see results.")
        return

    st.markdown("---")
    fit = render_sem_fit(result)
    st.markdown("---")
    path_results = render_structural_paths(result, structural_paths)
    st.markdown("---")
    endogenous = st.session_state.get("sem_endogenous", [])
    render_r_squared(result, endogenous)
    st.markdown("---")
    render_sem_checklist(fit, path_results)

    st.markdown("---")
    badge("ok",
        "SEM complete. Proceed to Mediation Analysis, Moderation Analysis, "
        "or Export Report in the sidebar."
    )
