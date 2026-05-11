"""
sem_model.py
============
Sprint 3 — Structural Equation Model (SEM) Module for SEM Studio.

Covers:
- Full SEM model specification (measurement + structural)
- Model estimation via semopy
- Standardized & unstandardized path coefficients
- Standard errors, z-values, p-values, confidence intervals
- R² for all endogenous variables
- Effect sizes (f²)
- Complete fit indices (RMSEA, CFI, TLI, SRMR, GFI, AIC, BIC, χ²)
- Hypothesis testing table
- Structural path interpretation (auto)
- Comparison: CFA fit vs SEM fit
- Full methodological checklist

References:
    - Kline (2016). Principles and Practice of SEM (4th ed.)
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
    - Cohen (1988). Statistical Power Analysis for the Behavioral Sciences.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from utils.thresholds import FIT, SEM as SEM_THRESH, CFA as CFA_THRESH, get_fit_label
from utils.interpretation import (
    interpret_fit_index, interpret_overall_fit,
    interpret_path, interpret_r2
)
from utils.apa_tables import fit_indices_table, structural_paths_table, style_df


# ─── HELPERS ────────────────────────────────────────────────────

LEVEL_COLOR = {
    "excellent": "#2ecc71", "good": "#27ae60",
    "ok": "#3498db", "warning": "#f39c12", "critical": "#e74c3c",
    "significant": "#2ecc71", "marginal": "#f39c12",
    "nonsignificant": "#e74c3c", "strong": "#2ecc71",
    "moderate": "#f39c12", "weak": "#e67e22", "negligible": "#e74c3c",
}

def _badge(level: str, message: str):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>', unsafe_allow_html=True
    )


# ─── MODEL SYNTAX BUILDER ───────────────────────────────────────

def build_sem_syntax(constructs: dict, structural_paths: list) -> str:
    """
    Build full SEM syntax: measurement model + structural paths.

    Parameters
    ----------
    constructs : dict
        {construct_name: [item1, item2, ...]}
    structural_paths : list
        [(predictor, outcome), ...]

    Returns
    -------
    str : semopy model syntax
    """
    lines = ["# Measurement Model"]
    for construct, items in constructs.items():
        if items:
            lines.append(f"{construct} =~ {' + '.join(items)}")

    lines.append("\n# Structural Model")
    for predictor, outcome in structural_paths:
        lines.append(f"{outcome} ~ {predictor}")

    return "\n".join(lines)


# ─── SECTION 1: MODEL SPECIFICATION ─────────────────────────────

def render_sem_spec(constructs: dict, structural_paths: list) -> str:
    st.subheader("📝 Step 1: Full SEM Model Specification")
    st.markdown(
        "The syntax below combines your **measurement model** (from CFA) "
        "and **structural paths** (your hypotheses). "
        "You can edit the syntax directly if needed."
    )

    syntax = build_sem_syntax(constructs, structural_paths)
    edited_syntax = st.text_area(
        "Full SEM Model Syntax",
        value=syntax,
        height=max(180, (len(constructs) + len(structural_paths) + 3) * 28),
        help="=~ defines measurement; ~ defines structural paths"
    )

    # Hypothesis summary
    with st.expander("📋 Structural Hypotheses Summary"):
        if structural_paths:
            for i, (pred, out) in enumerate(structural_paths):
                st.markdown(f"**H{i+1}:** {pred} → {out}")
        else:
            st.warning("⚠️ No structural paths defined. Go back to Data Input to define hypotheses.")

    # Endogenous vs exogenous
    all_outcomes   = set(out for _, out in structural_paths)
    all_predictors = set(pred for pred, _ in structural_paths)
    exogenous      = all_predictors - all_outcomes
    endogenous     = all_outcomes

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Exogenous constructs** (predictors only):")
        for c in exogenous:
            st.markdown(f"  - {c}")
    with c2:
        st.markdown("**Endogenous constructs** (have incoming paths):")
        for c in endogenous:
            st.markdown(f"  - {c}")

    st.session_state["sem_endogenous"] = list(endogenous)
    st.session_state["sem_exogenous"]  = list(exogenous)

    return edited_syntax


# ─── SECTION 2: MODEL ESTIMATION ────────────────────────────────

def render_sem_estimation(syntax: str, df: pd.DataFrame):
    st.subheader("⚙️ Step 2: SEM Estimation")

    estimator = st.session_state.get("recommended_estimator", "ML")
    st.info(f"**Estimator:** {estimator} — based on normality assessment.")

    if not SEMOPY_AVAILABLE:
        st.error("❌ semopy is not installed. Add `semopy>=2.3.5` to requirements.txt.")
        return None

    col1, col2 = st.columns([1, 3])
    with col1:
        run = st.button("▶️ Run SEM", type="primary", key="run_sem_btn", use_container_width=True)

    if run:
        with st.spinner("Estimating full SEM... please wait."):
            constructs = st.session_state.get("constructs", {})
            all_items  = [item for items in constructs.values() for item in items]
            data       = df[all_items].dropna()

            try:
                model = semopy.Model(syntax)
                model.fit(data)
                st.session_state["sem_model"]  = model
                st.session_state["sem_syntax"] = syntax
                st.session_state["sem_data"]   = data
                st.success(f"✅ SEM estimated on n = {len(data)} complete cases.")
            except Exception as e:
                st.error(f"❌ SEM estimation failed: {str(e)}")
                st.markdown("**Troubleshooting:**")
                st.markdown("- Ensure all item names match your data columns exactly")
                st.markdown("- Check for missing values")
                st.markdown("- Verify each construct has ≥ 3 indicators")
                return None

    return st.session_state.get("sem_model")


# ─── SECTION 3: FIT INDICES ─────────────────────────────────────

def render_sem_fit(model) -> dict:
    st.subheader("📐 Step 3: Model Fit Assessment")
    st.markdown(
        "Fit indices for the **full SEM** (measurement + structural model combined). "
        "Compare with CFA fit — SEM fit should not be substantially worse."
    )

    try:
        stats_result = semopy.calc_stats(model)
        fit = {}

        stat_map = {
            "chi2":  ["chi2", "Chi2", "chisq"],
            "df":    ["dof", "df", "DoF"],
            "p":     ["pvalue", "p", "chi2_p"],
            "rmsea": ["rmsea", "RMSEA"],
            "cfi":   ["cfi", "CFI"],
            "tli":   ["tli", "TLI", "nnfi"],
            "srmr":  ["srmr", "SRMR"],
            "gfi":   ["gfi", "GFI"],
            "aic":   ["aic", "AIC"],
            "bic":   ["bic", "BIC"],
            "nfi":   ["nfi", "NFI"],
        }

        if isinstance(stats_result, pd.DataFrame):
            stats_dict = stats_result.iloc[0].to_dict()
        elif hasattr(stats_result, 'to_dict'):
            stats_dict = stats_result.to_dict()
        else:
            stats_dict = {}

        for key, aliases in stat_map.items():
            for alias in aliases:
                if alias in stats_dict:
                    try:
                        fit[key] = float(stats_dict[alias])
                    except:
                        pass
                    break

        # Metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        for col, label, key in [
            (c1, "RMSEA", "rmsea"), (c2, "CFI", "cfi"),
            (c3, "TLI", "tli"),    (c4, "SRMR", "srmr"),
            (c5, "AIC", "aic"),
        ]:
            val = fit.get(key)
            col.metric(label, f"{val:.3f}" if val is not None else "—")

        # Full table
        st.markdown("**Complete Fit Indices:**")
        fit_table = fit_indices_table(fit)
        st.dataframe(fit_table, use_container_width=True, hide_index=True)

        # Individual interpretations
        with st.expander("🔍 Fit Index Interpretations"):
            for idx in ["rmsea", "cfi", "tli", "srmr"]:
                val = fit.get(idx)
                if val is not None:
                    result = interpret_fit_index(idx, val)
                    _badge(result["level"], result["message"])

        # Overall verdict
        overall = interpret_overall_fit(fit)
        st.info(overall)

        # χ²/df
        if fit.get("chi2") and fit.get("df") and fit["df"] > 0:
            ratio = fit["chi2"] / fit["df"]
            fit["chisq_df"] = ratio
            _badge("ok" if ratio <= 5 else "warning",
                f"χ²/df = {ratio:.3f} — {'✅ acceptable' if ratio <= 5 else '❌ poor'} "
                f"(criterion: ≤ 2.0 good; ≤ 5.0 acceptable)"
            )

        # Compare with CFA fit
        cfa_fit = st.session_state.get("cfa_fit", {})
        if cfa_fit:
            st.markdown("**📊 CFA vs SEM Fit Comparison:**")
            comp_data = []
            for idx in ["rmsea", "cfi", "tli", "srmr", "aic", "bic"]:
                cfa_val = cfa_fit.get(idx)
                sem_val = fit.get(idx)
                if cfa_val and sem_val:
                    comp_data.append({
                        "Index": idx.upper(),
                        "CFA":   round(cfa_val, 3),
                        "SEM":   round(sem_val, 3),
                        "Change": round(sem_val - cfa_val, 3),
                        "Note":  "⚠️ Worse" if (
                            (idx in ["rmsea","srmr"] and sem_val > cfa_val + 0.01) or
                            (idx in ["cfi","tli"]    and sem_val < cfa_val - 0.01)
                        ) else "✅ OK"
                    })
            if comp_data:
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        st.session_state["sem_fit"] = fit
        return fit

    except Exception as e:
        st.error(f"❌ Could not compute SEM fit indices: {str(e)}")
        return {}


# ─── SECTION 4: STRUCTURAL PATHS ────────────────────────────────

def render_structural_paths(model, structural_paths: list) -> list:
    st.subheader("➡️ Step 4: Structural Path Coefficients")
    st.markdown(
        "Path coefficients (β) represent the **direct effects** between latent constructs. "
        "Standardized β allows comparison across paths. "
        "Significance is assessed at p < .05 (two-tailed)."
    )

    try:
        params = model.inspect(std_est=True)

        # Extract structural paths
        path_results = []
        for pred, out in structural_paths:
            try:
                mask = (params["lval"] == out) & (params["rval"] == pred)
                if mask.any():
                    row = params[mask].iloc[0]
                    beta   = float(row.get("Estimate", np.nan))
                    se     = float(row.get("Std. Err", row.get("SE", np.nan)))
                    z_val  = float(row.get("z-value", row.get("z", beta/se if se > 0 else np.nan)))
                    p_val  = float(row.get("p-value", row.get("p", np.nan)))

                    # Confidence interval (Wald)
                    ci_lo = beta - 1.96 * se if not np.isnan(se) else np.nan
                    ci_hi = beta + 1.96 * se if not np.isnan(se) else np.nan

                    path_results.append({
                        "predictor": pred,
                        "outcome":   out,
                        "beta":      round(beta, 3),
                        "se":        round(se, 3) if not np.isnan(se) else None,
                        "z":         round(z_val, 3) if not np.isnan(z_val) else None,
                        "p":         round(p_val, 4) if not np.isnan(p_val) else None,
                        "ci_lower":  round(ci_lo, 3) if not np.isnan(ci_lo) else None,
                        "ci_upper":  round(ci_hi, 3) if not np.isnan(ci_hi) else None,
                    })
            except Exception:
                path_results.append({
                    "predictor": pred, "outcome": out,
                    "beta": None, "se": None, "z": None, "p": None,
                    "ci_lower": None, "ci_upper": None,
                })

        if path_results:
            # APA table
            apa_table = structural_paths_table(path_results)
            st.dataframe(apa_table, use_container_width=True, hide_index=True)
            st.caption(
                "Note. β = standardized path coefficient; SE = standard error; "
                "† p < .10. * p < .05. ** p < .01. *** p < .001."
            )

            # Hypothesis testing table
            st.markdown("**Hypothesis Testing Results:**")
            hyp_rows = []
            for i, p in enumerate(path_results):
                p_val    = p.get("p") or 1.0
                beta_val = p.get("beta") or 0
                supported = p_val < 0.05
                hyp_rows.append({
                    "Hypothesis": f"H{i+1}",
                    "Path":       f"{p['predictor']} → {p['outcome']}",
                    "β":          f"{beta_val:.3f}" if beta_val else "—",
                    "p":          f"{p_val:.3f}" if p_val else "—",
                    "Decision":   "✅ Supported" if supported else "❌ Not Supported",
                })

            hyp_df = pd.DataFrame(hyp_rows)

            def color_decision(val):
                if "Supported" in str(val) and "Not" not in str(val):
                    return "color:#2ecc71;font-weight:bold"
                return "color:#e74c3c;font-weight:bold"

            st.dataframe(
                hyp_df.style.map(color_decision, subset=["Decision"]),
                use_container_width=True, hide_index=True
            )

            # Individual path interpretations
            with st.expander("🔍 Path-by-Path Interpretation"):
                for p in path_results:
                    if p.get("beta") is not None and p.get("p") is not None:
                        result = interpret_path(
                            p["beta"], p.get("se", 0),
                            p["p"], p["predictor"], p["outcome"]
                        )
                        _badge(result["level"], result["message"])

            # Effect size (f²)
            st.markdown("**Effect Sizes (Cohen's f²):**")
            ef_rows = []
            for p in path_results:
                beta = p.get("beta")
                if beta is not None:
                    f2 = beta**2 / (1 - beta**2) if abs(beta) < 1 else np.nan
                    size = "Large" if f2 >= 0.35 else "Medium" if f2 >= 0.15 else "Small" if f2 >= 0.02 else "Negligible"
                    ef_rows.append({
                        "Path":   f"{p['predictor']} → {p['outcome']}",
                        "β":      round(beta, 3),
                        "f²":     round(f2, 3) if not np.isnan(f2) else "—",
                        "Effect": size,
                    })
            if ef_rows:
                st.dataframe(pd.DataFrame(ef_rows), use_container_width=True, hide_index=True)
                st.caption("Note. f² ≥ .02 = small; ≥ .15 = medium; ≥ .35 = large (Cohen, 1988).")

        st.session_state["sem_paths"] = path_results
        return path_results

    except Exception as e:
        st.error(f"❌ Could not extract structural paths: {str(e)}")
        return []


# ─── SECTION 5: R² ──────────────────────────────────────────────

def render_r_squared(model, endogenous: list):
    st.subheader("📊 Step 5: Explained Variance (R²)")
    st.markdown(
        "R² indicates the proportion of variance in each **endogenous construct** "
        "explained by its predictors. "
        "Criterion: R² ≥ .26 (substantial), ≥ .13 (moderate), ≥ .02 (weak) — Cohen (1988)."
    )

    try:
        params = model.inspect(std_est=True)

        r2_rows = []
        for construct in endogenous:
            # R² = 1 - error variance (for standardized solution)
            try:
                mask = (params["lval"] == construct) & (params["op"] == "~~") & (params["rval"] == construct)
                if mask.any():
                    error_var = float(params[mask]["Estimate"].values[0])
                    r2 = max(0, 1 - error_var)
                    result = interpret_r2(r2, construct)
                    r2_rows.append({
                        "Construct": construct,
                        "R²":        round(r2, 3),
                        "R² (%)":    f"{r2:.1%}",
                        "Level":     result["level"].capitalize(),
                    })
                    _badge(result["level"], result["message"])
            except Exception:
                r2_rows.append({
                    "Construct": construct,
                    "R²": "—", "R² (%)": "—", "Level": "—"
                })

        if r2_rows:
            r2_df = pd.DataFrame(r2_rows)

            # Bar chart
            plot_data = r2_df[r2_df["R²"] != "—"].copy()
            if len(plot_data) > 0:
                plot_data["R²"] = plot_data["R²"].astype(float)
                fig = px.bar(
                    plot_data, x="Construct", y="R²",
                    color="R²",
                    color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
                    range_color=[0, 0.5],
                    template="plotly_dark",
                    title="R² by Endogenous Construct",
                    text="R² (%)",
                )
                fig.add_hline(y=0.26, line_dash="dash", line_color="#2ecc71",
                             annotation_text="Substantial (.26)")
                fig.add_hline(y=0.13, line_dash="dash", line_color="#f39c12",
                             annotation_text="Moderate (.13)")
                fig.add_hline(y=0.02, line_dash="dot",  line_color="#e74c3c",
                             annotation_text="Weak (.02)")
                fig.update_layout(height=350, yaxis=dict(range=[0, 1]))
                st.plotly_chart(fig, use_container_width=True)

        st.session_state["sem_r2"] = r2_rows

    except Exception as e:
        st.error(f"❌ Could not compute R²: {str(e)}")


# ─── SECTION 6: SEM CHECKLIST ────────────────────────────────────

def render_sem_checklist(fit: dict, path_results: list):
    st.subheader("✅ Step 6: SEM Methodological Checklist")

    cfa_complete = st.session_state.get("cfa_complete", False)
    n_sig_paths  = sum(1 for p in path_results if (p.get("p") or 1) < 0.05)
    n_paths      = len(path_results)

    checks = {
        "CFA measurement model validated first":
            cfa_complete,
        f"Model fit — RMSEA ≤ {FIT['rmsea_acceptable']}":
            (fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"],
        f"Model fit — CFI ≥ {FIT['cfi_acceptable']}":
            (fit.get("cfi") or 0) >= FIT["cfi_acceptable"],
        f"Model fit — SRMR ≤ {FIT['srmr_acceptable']}":
            (fit.get("srmr") or 999) <= FIT["srmr_acceptable"],
        "At least one significant structural path":
            n_sig_paths > 0,
        "Structural paths theoretically justified":
            n_paths > 0,
        "Sample size ≥ 200":
            len(st.session_state.get("df", [])) >= 200,
    }

    rows = [{"Check": k, "Status": "✅ Pass" if v else "❌ Fail"} for k, v in checks.items()]

    def color_status(val):
        if "Pass" in str(val): return "color:#2ecc71;font-weight:bold"
        return "color:#e74c3c;font-weight:bold"

    st.dataframe(
        pd.DataFrame(rows).style.map(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True
    )

    n_fail = sum(1 for v in checks.values() if not v)
    if n_fail == 0:
        _badge("excellent",
            f"🎉 All SEM criteria passed! {n_sig_paths}/{n_paths} hypotheses supported. "
            "Proceed to **Mediation/Moderation** analysis if applicable, or **Export Report**."
        )
    elif n_fail <= 2:
        _badge("warning",
            f"⚠️ {n_fail} criterion/criteria require attention. "
            "Review ❌ items above before finalizing results."
        )
    else:
        _badge("critical",
            f"❌ {n_fail} criteria not met. Re-examine model specification and fit before reporting."
        )

    st.session_state["sem_complete"] = n_fail == 0


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_sem():
    st.title("🔗 Structural Equation Model (SEM)")
    st.markdown(
        "The **structural model** tests the hypothesized directional relationships "
        "between latent constructs. It combines the **measurement model** (CFA) "
        "with **structural paths** (hypotheses) into one unified model.\n\n"
        "> 📌 *The measurement model (CFA) must be validated before interpreting structural paths.*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    if not st.session_state.get("cfa_complete"):
        st.warning(
            "⚠️ CFA has not been validated yet. "
            "Please complete **Confirmatory Factor Analysis** before running SEM."
        )

    df               = st.session_state["df"]
    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])

    if not constructs:
        st.error("❌ No constructs defined. Return to Data Input.")
        return

    if not structural_paths:
        st.error("❌ No structural paths defined. Return to Data Input to define hypotheses.")
        return

    if not SEMOPY_AVAILABLE:
        st.error("❌ semopy not installed. Add to requirements.txt and restart.")
        return

    st.markdown("---")

    # Step 1
    syntax = render_sem_spec(constructs, structural_paths)
    st.markdown("---")

    # Step 2
    model = render_sem_estimation(syntax, df)
    if model is None:
        st.info("👆 Run the SEM model above to see results.")
        return

    st.markdown("---")

    # Step 3
    fit = render_sem_fit(model)
    st.markdown("---")

    # Step 4
    path_results = render_structural_paths(model, structural_paths)
    st.markdown("---")

    # Step 5
    endogenous = st.session_state.get("sem_endogenous", [])
    render_r_squared(model, endogenous)
    st.markdown("---")

    # Step 6
    render_sem_checklist(fit, path_results)

    st.markdown("---")
    st.success(
        "✅ SEM complete. Proceed to **Mediation Analysis**, **Moderation Analysis**, "
        "or **Export Report** in the sidebar."
    )
