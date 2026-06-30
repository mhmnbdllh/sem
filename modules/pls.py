"""
pls.py — PLS-SEM Analysis Module
=================================
Partial Least Squares Structural Equation Modeling via R (composite-based).

Methodological references:
  Hair et al. (2022) A Primer on Partial Least Squares SEM, 3rd ed.
  Hair et al. (2022) PLS-SEM methodology
  Henseler et al. (2015) HTMT criterion
  Dijkstra & Henseler (2015) rho_A reliability
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.apa_tables import _safe_float, _fmt, _stars

COLORS = {
    "excellent": "#1a7a4a",
    "good":      "#2ecc71",
    "ok":        "#1a6fa8",
    "warning":   "#b7770d",
    "critical":  "#c0392b",
}

def badge(level, message):
    color = COLORS.get(level, "#555555")
    import re as _re
    message = _re.sub(r'[*][*](.+?)[*][*]', r'<b>\\1</b>', str(message))
    st.markdown(
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;'
        f'color:#1a1a1a;font-size:0.92rem">{message}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: significance stars
# ─────────────────────────────────────────────────────────────────────────────
def _sig(p):
    if p is None: return "—"
    p = float(p)
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "†"
    return "ns"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Measurement type setup
# ─────────────────────────────────────────────────────────────────────────────
def render_measurement_type(constructs):
    st.subheader("Step 1: Measurement Model Type")
    st.markdown(
        "In PLS-SEM, each construct must be specified as **Reflective** or **Formative**. "
        "This is a critical methodological decision."
    )

    with st.expander("❓ Reflective vs Formative — Which one?", expanded=True):
        st.markdown("""
**Reflective (Mode A)** — items are caused by the construct:
- Items are interchangeable — removing one item shouldn't change the construct meaning
- Items should be highly correlated (high loadings)
- Example: satisfaction items all reflect underlying satisfaction level
- Assessed with: AVE, CR, Cronbach's alpha

**Formative (Mode B)** — items cause/form the construct:
- Items are NOT interchangeable — each captures a unique dimension
- Items may NOT be correlated
- Example: socioeconomic status = income + education + occupation
- Assessed with: VIF (no multicollinearity), indicator weights

> **Most survey research uses Reflective constructs.**
        """)

    construct_types = {}
    for cname in constructs.keys():
        saved = st.session_state.get(f"pls_type_{cname}", "Reflective")
        choice = st.selectbox(
            f"{cname}:",
            options=["Reflective", "Formative"],
            index=0 if saved == "Reflective" else 1,
            key=f"pls_type_{cname}",
            help="Reflective: items reflect construct (most common). "
                 "Formative: items form/cause construct."
        )
        construct_types[cname] = choice.lower()

    return construct_types


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Bootstrap settings
# ─────────────────────────────────────────────────────────────────────────────
def render_bootstrap_settings():
    st.subheader("Step 2: Bootstrap Settings")
    st.markdown(
        "PLS-SEM uses bootstrapping for significance testing. "
        "More resamples = more accurate p-values but longer computation time."
    )
    n_boot = st.select_slider(
        "Number of bootstrap resamples:",
        options=[500, 1000, 2000, 5000],
        value=st.session_state.get("pls_n_boot", 1000),
        key="pls_n_boot",
        help="Hair et al. (2022) recommend minimum 1000. 5000 for publication."
    )
    st.caption(
        f"Estimated time: {n_boot // 500} – {n_boot // 250} minutes on Streamlit Cloud."
    )
    return n_boot


# ─────────────────────────────────────────────────────────────────────────────
# Render outer loadings / weights
# ─────────────────────────────────────────────────────────────────────────────
def render_outer_model(result):
    st.subheader("Step 3: Outer Model (Measurement)")
    st.markdown(
        "**Outer loadings** (reflective) or **outer weights** (formative) "
        "indicate how strongly each item relates to its construct."
    )

    loadings = result.get("loadings", [])
    if not loadings:
        st.info("No loadings available.")
        return

    rows = []
    for l in loadings:
        cname  = l.get("construct", "")
        item   = l.get("item", "")
        lam    = _safe_float(l.get("loading"))
        ctype  = l.get("type", "reflective")
        status = l.get("status", "—")
        col_label = "Loading (λ)" if ctype == "reflective" else "Weight (w)"
        rows.append({
            "Construct": cname,
            "Item":      item,
            "Type":      ctype.capitalize(),
            col_label:   f"{lam:.3f}" if lam is not None else "—",
            "Status":    status,
        })

    if rows:
        df_rows = pd.DataFrame(rows)
        st.dataframe(df_rows, use_container_width=True, hide_index=True)
        st.caption(
            "Reflective loadings: ≥ .70 = Strong ✅; ≥ .50 = Acceptable ⚠️; < .50 = Weak ❌ "
            "(Hair et al., 2022). "
            "Formative weights: check VIF < 5.0 instead of loading threshold."
        )

        # Recommendations
        weak = [f"{r['Item']} (λ={r.get('Loading (λ)', r.get('Weight (w)','?'))})"
                for r in rows if r.get("Status") == "Weak"]
        if weak:
            badge("warning",
                f"Weak items detected: {', '.join(weak)}. "
                "Consider removing items with loading < .50 for reflective constructs."
            )
        else:
            badge("ok", "All outer loadings meet the acceptable threshold (≥ .50). ✅")


# ─────────────────────────────────────────────────────────────────────────────
# Render reliability & validity
# ─────────────────────────────────────────────────────────────────────────────
def render_reliability(result):
    st.subheader("Step 4: Reliability and Validity")

    rel = result.get("reliability", {})
    if not rel:
        st.info("No reliability data available.")
        return

    rows = []
    for cname, m in rel.items():
        alpha   = _safe_float(m.get("alpha"))
        cr      = _safe_float(m.get("cr"))
        rho_a   = _safe_float(m.get("rho_a"))
        ave     = _safe_float(m.get("ave"))
        rows.append({
            "Construct":       cname,
            "α (Cronbach)":    _fmt(alpha),
            "ρc (Comp. Rel)":  _fmt(cr),
            "ρA (D-H Rel)":    _fmt(rho_a),
            "AVE":             _fmt(ave),
            "α OK":            "✅" if m.get("alpha_ok") else "❌",
            "ρc OK":           "✅" if m.get("cr_ok")    else "❌",
            "AVE OK":          "✅" if m.get("ave_ok")   else "❌",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "Criteria: α ≥ .70 (Nunnally, 1978); ρc ≥ .70; ρA ≥ .70 "
            "(Dijkstra & Henseler, 2015); AVE ≥ .50 (Fornell & Larcker, 1981). "
            "Use ρc (rho_c) as preferred reliability measure in PLS-SEM."
        )

        all_ok = all(
            m.get("alpha_ok") and m.get("cr_ok") and m.get("ave_ok")
            for m in rel.values()
        )
        if all_ok:
            badge("excellent", "All reliability and validity criteria met. ✅")
        else:
            failed = [c for c, m in rel.items()
                      if not (m.get("alpha_ok") and m.get("cr_ok") and m.get("ave_ok"))]
            badge("warning",
                f"Criteria not met for: {', '.join(failed)}. "
                "Consider removing weak items from the outer model."
            )

    # HTMT
    htmt = result.get("htmt")
    if htmt is not None:
        st.markdown("**HTMT (Heterotrait-Monotrait Ratio) — Discriminant Validity:**")
        try:
            htmt_df = pd.DataFrame(htmt)
            htmt_df = htmt_df.round(3).fillna("—")
            st.dataframe(htmt_df, use_container_width=True)
            st.caption(
                "HTMT < .85 = discriminant validity supported (Henseler et al., 2015). "
                "Conservative threshold: HTMT < .90."
            )
            # Check HTMT
            htmt_arr = pd.DataFrame(htmt).values.astype(float)
            np.fill_diagonal(htmt_arr, 0)
            max_htmt = np.nanmax(htmt_arr)
            if max_htmt < 0.85:
                badge("ok", f"Discriminant validity supported (max HTMT = {max_htmt:.3f} < .85). ✅")
            elif max_htmt < 0.90:
                badge("warning",
                    f"Max HTMT = {max_htmt:.3f} — borderline. "
                    "Some constructs may overlap (threshold .85 not met, but < .90)."
                )
            else:
                badge("warning",
                    f"Max HTMT = {max_htmt:.3f} ≥ .90 — discriminant validity not supported. "
                    "Constructs may be too similar."
                )
        except Exception:
            st.dataframe(pd.DataFrame(htmt), use_container_width=True)

    # Fornell-Larcker
    fl = result.get("fl_criterion")
    if fl is not None:
        st.markdown("**Fornell-Larcker Criterion:**")
        try:
            fl_df = pd.DataFrame(fl).round(3)
            st.dataframe(fl_df, use_container_width=True)
            st.caption(
                "Diagonal = √AVE. Off-diagonal = inter-construct correlations. "
                "√AVE should exceed all correlations in the same row/column."
            )
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Render inner model (structural paths)
# ─────────────────────────────────────────────────────────────────────────────
def render_inner_model(result):
    st.subheader("Step 5: Inner Model (Structural Paths)")
    st.markdown(
        "Path coefficients (β) represent the strength and direction of relationships "
        "between constructs. Significance via bootstrapping."
    )

    paths_data = result.get("paths", [])
    if not paths_data:
        st.info("No path data available.")
        return

    rows = []
    for i, p in enumerate(paths_data, 1):
        beta  = _safe_float(p.get("beta"))
        t_val = _safe_float(p.get("t_stat"))
        p_val = _safe_float(p.get("p_value"))
        ci_lo = _safe_float(p.get("ci_lo"))
        ci_hi = _safe_float(p.get("ci_hi"))
        ci_str = f"[{ci_lo:.3f}, {ci_hi:.3f}]" if ci_lo is not None and ci_hi is not None else "—"
        rows.append({
            "H":          f"H{i}",
            "Path":       f"{p.get('predictor','?')} → {p.get('outcome','?')}",
            "β":          _fmt(beta),
            "t-stat":     _fmt(t_val, 3),
            "p-value":    _fmt(p_val, 4),
            "Sig.":       _sig(p_val),
            "95% BCa CI": ci_str,
            "Decision":   p.get("decision", "—"),
        })

    if rows:
        df_paths = pd.DataFrame(rows)
        st.dataframe(df_paths, use_container_width=True, hide_index=True)
        st.caption(
            "β = standardized path coefficient. t-stat via bootstrapping. "
            "Sig.: * p < .05; ** p < .01; *** p < .001; † p < .10 (marginal). "
            "95% BCa CI: if CI excludes 0, path is significant."
        )

        supported = sum(1 for p in paths_data if p.get("supported"))
        total     = len(paths_data)
        # Per-path narrative
        st.markdown("**Path Interpretation:**")
        for i, p in enumerate(paths_data, 1):
            beta  = _safe_float(p.get("beta"))
            p_val = _safe_float(p.get("p_value"))
            pred  = p.get("predictor","?")
            out   = p.get("outcome","?")
            supp  = p.get("supported", False)
            if beta is not None:
                direction = "positive" if beta > 0 else "negative"
                strength  = "strong" if abs(beta) >= 0.50 else "moderate" if abs(beta) >= 0.30 else "weak"
                sig_txt   = f"statistically significant (p = {p_val:.4f})" if supp else f"not statistically significant (p = {p_val:.4f})"
                st.markdown(
                    f"**H{i}: {pred} → {out}** — β = {beta:.3f}, {sig_txt}. "
                    f"The effect is {direction} and {strength}. "
                    f"{'✅ Hypothesis supported.' if supp else '❌ Hypothesis not supported.'}"
                )

        # Overall interpretation
        badge(
            "excellent" if supported == total else "ok" if supported > 0 else "warning",
            f"{supported} of {total} hypothesized path(s) are statistically significant (p < .05). "
            f"{'All hypotheses are supported.' if supported == total else f'{total-supported} hypothesis/hypotheses not supported — review theoretical justification.'}"
        )

    # VIF
    # NOTE: R structure is vif[outcome][predictor] = value
    # (R groups VIF by outcome construct, since multicollinearity is assessed
    #  among predictors of the SAME outcome)
    vif = result.get("vif")
    if vif:
        st.markdown("**VIF (Variance Inflation Factor) — Inner Model Collinearity:**")
        st.caption(
            "VIF measures multicollinearity among predictors of the same outcome construct."
        )
        vif_rows = []
        for outcome, vif_vals in vif.items():
            if isinstance(vif_vals, dict):
                for predictor, v in vif_vals.items():
                    vif_rows.append({"Predictor": predictor, "Outcome": outcome, "VIF": _fmt(v, 3)})
            elif isinstance(vif_vals, (int, float)):
                vif_rows.append({"Predictor": "?", "Outcome": outcome, "VIF": _fmt(vif_vals, 3)})
        if vif_rows:
            st.dataframe(pd.DataFrame(vif_rows), use_container_width=True, hide_index=True)
            st.caption("VIF < 5.0 = no collinearity concern (Hair et al., 2022). VIF < 3.3 = ideal.")


# ─────────────────────────────────────────────────────────────────────────────
# Render R² and Q²
# ─────────────────────────────────────────────────────────────────────────────
def render_r2_q2(result):
    st.subheader("Step 6: Explained Variance (R²) and Predictive Relevance (Q²)")

    # R²
    r2_data = result.get("r2", [])
    if r2_data:
        rows = []
        for r in r2_data:
            r2    = _safe_float(r.get("r2"))
            r2adj = _safe_float(r.get("r2_adj"))
            rows.append({
                "Construct":  r.get("construct", "?"),
                "R²":         _fmt(r2),
                "Adj. R²":    _fmt(r2adj),
                "Level":      r.get("level", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "R² benchmarks (Hair et al., 2022): ≥ .75 = Substantial; "
            "≥ .50 = Moderate; ≥ .25 = Weak."
        )

    # Q²
    q2_data = result.get("q2", {})
    if q2_data:
        st.markdown("**Predictive Relevance (Q² via PLSpredict):**")
        q2_rows = []
        for cname, q in q2_data.items():
            q2_val = _safe_float(q.get("q2") if isinstance(q, dict) else q)
            q2_rows.append({
                "Construct": cname,
                "Q²":        _fmt(q2_val),
                "Relevant":  "✅ Yes" if q2_val is not None and q2_val > 0 else "❌ No",
            })
        if q2_rows:
            st.dataframe(pd.DataFrame(q2_rows), use_container_width=True, hide_index=True)
            st.caption(
                "Q² > 0 = model has predictive relevance (Stone, 1974; Geisser, 1975). "
                "Q² > .25 = large; > .15 = medium; > .02 = small (Hair et al., 2022)."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Render model fit
# ─────────────────────────────────────────────────────────────────────────────
def render_fit(result):
    st.subheader("Step 7: Model Fit")
    st.markdown(
        "PLS-SEM uses **SRMR** as primary fit index. "
        "Traditional CB-SEM indices (CFI, RMSEA) are not applicable to PLS-SEM."
    )

    fit = result.get("fit", {})
    if not fit:
        st.info("Fit indices not available.")
        return

    srmr      = _safe_float(fit.get("srmr"))
    rms_theta = _safe_float(fit.get("rms_theta"))
    nfi       = _safe_float(fit.get("nfi"))

    rows = []
    if srmr is not None:
        srmr_ok = srmr < 0.08
        rows.append({
            "Index":       "SRMR",
            "Value":       f"{srmr:.3f}",
            "Threshold":   "< .080",
            "Status":      "✅ Good" if srmr_ok else "❌ Poor",
            "Interpretation": "Acceptable" if srmr_ok else "Model fit needs improvement",
        })
    if rms_theta is not None:
        rows.append({
            "Index":       "RMS_theta",
            "Value":       f"{rms_theta:.3f}",
            "Threshold":   "< .120",
            "Status":      "✅ Good" if rms_theta < 0.12 else "❌ Poor",
            "Interpretation": "Acceptable" if rms_theta < 0.12 else "Poor",
        })
    if nfi is not None:
        rows.append({
            "Index":       "NFI",
            "Value":       f"{nfi:.3f}",
            "Threshold":   "> .900",
            "Status":      "✅ Good" if nfi > 0.90 else "❌ Poor",
            "Interpretation": "Acceptable" if nfi > 0.90 else "Poor",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "SRMR < .080 = acceptable fit (Henseler et al., 2015). "
            "Note: PLS-SEM is primarily a predictive technique — "
            "model fit is less critical than in CB-SEM (Hair et al., 2022)."
        )

        if srmr is not None:
            badge(
                "ok" if srmr < 0.08 else "warning",
                f"SRMR = {srmr:.3f} — "
                f"{'Model fit is acceptable.' if srmr < 0.08 else 'Model fit is below threshold. Consider re-specifying the model.'}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main render function
# ─────────────────────────────────────────────────────────────────────────────
def render_pls():
    st.title("PLS-SEM Analysis")
    st.warning(
        "⚠️ **You are now in the PLS-SEM track.** "
        "This is a separate analytical method from CB-SEM (EFA/CFA/SEM with lavaan). "
        "You do **not** need to complete EFA, CFA, or SEM before running PLS-SEM. "
        "Only **Data Input** and **Descriptive Statistics** are required first."
    )
    st.markdown(
        "**Partial Least Squares Structural Equation Modeling** via R (composite-based). "
        "PLS-SEM is suitable for prediction-oriented research, non-normal data, "
        "and smaller sample sizes."
    )

    with st.expander("📖 CB-SEM vs PLS-SEM — When to use which?", expanded=False):
        st.markdown("""
| Criterion | CB-SEM (lavaan) | PLS-SEM (R) |
|---|---|---|
| Goal | Theory testing | Prediction |
| Sample size | Large (n ≥ 200) | Smaller (n ≥ 30) |
| Data normality | Required | Not required |
| Fit indices | CFI, RMSEA, SRMR | SRMR, Q² |
| Reliability | Cronbach's α, CR | ρc, ρA |
| Construct type | Reflective only | Reflective & Formative |
| Latent scores | True latent | Composite proxies |

> Use **CB-SEM** for confirmatory theory testing.
> Use **PLS-SEM** for exploratory/predictive research or non-normal data.
        """)

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input first.")
        return

    df         = st.session_state.get("df")
    constructs = st.session_state.get("constructs", {})
    paths_list = st.session_state.get("structural_paths", [])

    if not constructs:
        badge("warning", "No constructs defined. Please complete Data Input first.")
        return

    if not paths_list:
        badge("warning", "No structural paths defined. Please complete Data Input first.")
        return

    # Step 1: Measurement types
    construct_types = render_measurement_type(constructs)
    st.markdown("---")

    # Step 2: Bootstrap settings
    n_boot = render_bootstrap_settings()
    st.markdown("---")

    # Run button
    st.subheader("Run PLS-SEM")
    n_items = sum(len(v) for v in constructs.values())
    n_paths = len(paths_list)
    st.markdown(
        f"Model: **{len(constructs)} constructs**, "
        f"**{n_items} items**, "
        f"**{n_paths} structural paths**. "
        f"Bootstrap: **{n_boot} resamples**."
    )

    if st.button("▶ Run PLS-SEM via R", type="primary",
                 key="run_pls_btn", use_container_width=True):

        from r_scripts.r_bridge import run_plssem, check_r_available

        r_check = check_r_available()
        if not r_check.get("available"):
            badge("warning", "R is not available. Please check the server configuration.")
            return

        all_items = [item for items in constructs.values() for item in items]
        with st.spinner(f"Running PLS-SEM with {n_boot} bootstrap resamples... "
                        f"This may take {n_boot // 500}–{n_boot // 250} minutes."):
            result = run_plssem(
                df             = df,
                constructs     = constructs,
                paths          = paths_list,
                construct_types= construct_types,
                n_boot         = n_boot,
            )

        if result.get("error"):
            err = result["error"]
            if "sample size" in err.lower():
                st.error(f"PLS-SEM failed: {err}")
            elif "not found in dataset" in err.lower():
                st.error(f"Item mismatch: {err}")
            else:
                st.error(f"PLS-SEM estimation failed: {err}")
            return

        st.session_state["pls_result"] = result
        st.session_state["pls_complete"] = True
        badge("excellent", f"PLS-SEM estimated successfully on n = {result.get('n','?')} cases.")
        st.rerun()

    # Show results if available
    result = st.session_state.get("pls_result")
    if result and not result.get("error"):
        st.markdown("---")
        # Clearly label this as PLS-SEM output
        method = result.get("method", "Composite-based PLS")
        st.success(
            f"**PLS-SEM Results** — {method}. "
            f"n = {result.get('n','?')}, "
            f"Bootstrap = {result.get('n_boot',1000)} resamples. "
            "Note: These are PLS-SEM estimates, NOT CB-SEM (lavaan) estimates."
        )
        render_outer_model(result)
        st.markdown("---")
        render_reliability(result)
        st.markdown("---")
        render_inner_model(result)
        st.markdown("---")
        render_r2_q2(result)
        st.markdown("---")
        render_fit(result)
        st.markdown("---")

        # Navigation
        badge("ok",
            f"PLS-SEM complete. n = {result.get('n','?')}, "
            f"{n_boot} bootstrap resamples."
        )
        # Overall model narrative
        st.markdown("---")
        st.subheader("📋 Model Summary and Interpretation")

        paths_data = result.get("paths", [])
        rel        = result.get("reliability", {})
        fit        = result.get("fit", {})
        r2_data    = result.get("r2", [])
        n_obs      = result.get("n", "?")
        n_boot_res = result.get("n_boot", 1000)

        supported  = sum(1 for p in paths_data if p.get("supported"))
        total_p    = len(paths_data)
        srmr       = _safe_float(fit.get("srmr"))
        all_rel_ok = all(m.get("cr_ok") and m.get("ave_ok") for m in rel.values())

        # Narrative
        narrative_parts = [
            f"A PLS-SEM analysis was conducted using composite-based PLS (Hair et al., 2022) with n = {n_obs} complete cases "
            f"and {n_boot_res}-resample bootstrapping for significance testing.",
        ]

        if srmr is not None:
            fit_desc = "acceptable" if srmr < 0.08 else "below acceptable threshold"
            narrative_parts.append(
                f"Model fit was {fit_desc} (SRMR = {srmr:.3f}; threshold < .080; "
                "Henseler et al., 2015)."
            )

        if all_rel_ok:
            narrative_parts.append(
                "All constructs demonstrated adequate reliability (ρc ≥ .70) "
                "and convergent validity (AVE ≥ .50; Fornell & Larcker, 1981)."
            )
        else:
            failed_rel = [c for c, m in rel.items() if not (m.get("cr_ok") and m.get("ave_ok"))]
            narrative_parts.append(
                f"Reliability or validity criteria were not fully met for: "
                f"{', '.join(failed_rel)}. Interpret results with caution."
            )

        narrative_parts.append(
            f"Of {total_p} hypothesized structural paths, "
            f"{supported} were statistically significant (p < .05). "
        )

        for r2 in r2_data:
            r2_val = _safe_float(r2.get("r2"))
            if r2_val is not None:
                narrative_parts.append(
                    f"The model explains {r2_val:.1%} of variance in {r2.get('construct','?')} "
                    f"(R² = {r2_val:.3f})."
                )

        narrative_parts.append(
            "Note: PLS-SEM generates composite-based estimates rather than true latent variable scores. "
            "Results should be interpreted in the context of prediction-oriented research goals "
            "(Hair et al., 2022)."
        )

        for part in narrative_parts:
            st.markdown(f"- {part}")

        st.markdown("---")
        badge("ok",
            "PLS-SEM analysis complete. Use 'Export Report' to generate the full HTML report. "
            "For theory-testing research, consider also running CB-SEM (lavaan) for comparison."
        )

        if st.button("▶ Export Report →", type="primary",
                     key="pls_to_export_btn", use_container_width=True):
            st.session_state["current_page"] = "export"
            st.rerun()
