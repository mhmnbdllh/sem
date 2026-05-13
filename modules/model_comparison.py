"""
model_comparison.py - Model Comparison Module.
Uses R/lavaan via r_bridge for methodologically correct model comparison.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
        "srmr": "srmr", "aic": "aic", "bic": "bic", "nfi": "nfi",
    }
    for k, v in fit_dict.items():
        nk = key_map.get(k.lower().replace(".", "_"), k.lower().replace(".", "_"))
        val = _safe_float(v)
        if val is not None:
            normalized[nk] = val
    return normalized


def render_model_setup(constructs, structural_paths):
    st.subheader("Step 1: Define Models to Compare")
    st.markdown(
        "Compare your **hypothesized model** against one or more **rival (alternative) models**. "
        "Rival models should be theoretically meaningful — not arbitrary.\n\n"
        "Common alternatives: measurement-only model (CFA), nested sub-models, "
        "or models with reversed/additional paths."
    )

    # Build default hypothesized model syntax
    meas_lines   = [f"{c} =~ {' + '.join(items)}" for c, items in constructs.items() if items]
    struct_lines = [f"{out} ~ {pred}" for pred, out in structural_paths]
    hyp_syntax   = "\n".join(meas_lines + ([""] + struct_lines if struct_lines else []))

    models = {}

    # Model 1: Hypothesized
    with st.expander("Model 1: Hypothesized Model (your main model)", expanded=True):
        st.markdown("This is your primary theoretical model — the one you are arguing for.")
        m1_syntax = st.text_area(
            "Model 1 Syntax (lavaan)",
            value=st.session_state.get("sem_syntax", hyp_syntax),
            height=max(120, (len(constructs) + len(structural_paths)) * 28),
            key="comp_m1_syntax"
        )
        models["Hypothesized Model"] = m1_syntax

    # Model 2: Rival
    with st.expander("Model 2: Rival Model A (add alternative)"):
        st.markdown(
            "Specify an alternative model. Examples:\n"
            "- Remove one structural path\n"
            "- Measurement-only model (no structural paths)\n"
            "- Reverse the direction of a path\n"
            "- Add a path not in the hypothesized model"
        )
        m2_name   = st.text_input("Model 2 name", value="Measurement Model (CFA only)", key="comp_m2_name")
        m2_default = "\n".join(meas_lines)
        m2_syntax = st.text_area("Model 2 Syntax", value=m2_default, height=120, key="comp_m2_syntax")
        if m2_name and m2_syntax.strip():
            models[m2_name] = m2_syntax

    # Model 3: Optional rival
    with st.expander("Model 3: Rival Model B (optional)"):
        m3_name   = st.text_input("Model 3 name", value="", key="comp_m3_name")
        m3_syntax = st.text_area("Model 3 Syntax", value="", height=100, key="comp_m3_syntax")
        if m3_name and m3_syntax.strip():
            models[m3_name] = m3_syntax

    badge("ok",
        "A good rival model is one that a reasonable skeptic might propose as an alternative. "
        "Testing against it strengthens your argument for the hypothesized model."
    )

    return models


def render_fit_models(models, df, constructs):
    st.subheader("Step 2: Estimate All Models")

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))
    estimator = st.session_state.get("recommended_estimator", "MLR")

    if st.button("Estimate All Models via R/lavaan", type="primary",
                 key="run_comp_btn", use_container_width=True):
        try:
            from r_scripts.r_bridge import run_model_comparison, check_r_available
            r_check = check_r_available()
            if not r_check["available"]:
                st.error(f"R is not available: {r_check['message']}")
                return {}

            with st.spinner("Estimating all models via R/lavaan..."):
                result = run_model_comparison(
                    df        = df,
                    all_cols  = all_items,
                    models    = models,
                    estimator = estimator
                )

            if "error" in result:
                st.error(f"Model comparison failed: {result['error']}")
                return {}

            # Extract fit results
            fit_results = {}
            raw_fits = result.get("fit_results", {})
            if isinstance(raw_fits, dict):
                for name, fit_raw in raw_fits.items():
                    if isinstance(fit_raw, dict) and "error" not in fit_raw:
                        fit_results[name] = _extract_fit(fit_raw)
                    elif isinstance(fit_raw, dict) and "error" in fit_raw:
                        fit_results[name] = {"error": fit_raw["error"]}

            st.session_state["comparison_results"] = fit_results
            st.session_state["comparison_models"]  = models

            n_success = sum(1 for v in fit_results.values() if "error" not in v)
            st.success(f"{n_success}/{len(models)} model(s) estimated successfully.")

        except Exception as e:
            st.error(f"Model comparison error: {str(e)}")
            return {}

    return st.session_state.get("comparison_results", {})


def render_comparison_table(fit_results):
    st.subheader("Step 3: Model Fit Comparison")

    if not fit_results:
        st.info("Estimate models above to see comparison.")
        return pd.DataFrame()

    rows = []
    for name, fit in fit_results.items():
        if "error" in fit:
            rows.append({
                "Model": name, "Status": "Error",
                "chi2": "—", "df": "—", "RMSEA": "—",
                "CFI": "—", "TLI": "—", "SRMR": "—",
                "AIC": "—", "BIC": "—",
            })
        else:
            chi2 = fit.get("chi2")
            df_  = fit.get("df")
            rows.append({
                "Model": name,
                "Status": "OK",
                "chi2":  round(chi2, 3) if chi2 else "—",
                "df":    int(df_)       if df_  else "—",
                "RMSEA": round(fit.get("rmsea", 0), 3) if fit.get("rmsea") else "—",
                "CFI":   round(fit.get("cfi",   0), 3) if fit.get("cfi")   else "—",
                "TLI":   round(fit.get("tli",   0), 3) if fit.get("tli")   else "—",
                "SRMR":  round(fit.get("srmr",  0), 3) if fit.get("srmr")  else "—",
                "AIC":   round(fit.get("aic",   0), 1) if fit.get("aic")   else "—",
                "BIC":   round(fit.get("bic",   0), 1) if fit.get("bic")   else "—",
            })

    comp_df = pd.DataFrame(rows)

    def color_status(val):
        if val == "OK":    return "color:#1a7a4a;font-weight:700"
        if val == "Error": return "color:#c0392b;font-weight:700"
        return ""

    st.dataframe(
        comp_df.style.map(color_status, subset=["Status"])
                     .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                     .set_table_styles([{
                         "selector":"th",
                         "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                     }]),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Note: Lower AIC/BIC = better fit for non-nested models. "
        "For nested models, use the chi-square difference test below."
    )

    # Radar chart
    valid_models = {k: v for k, v in fit_results.items() if "error" not in v}
    if len(valid_models) >= 2:
        with st.expander("Fit Profile Radar Chart"):
            categories = ["CFI", "TLI", "1-RMSEA", "1-SRMR"]
            fig = go.Figure()
            plot_colors = ["#1a6fa8", "#c0392b", "#1a7a4a", "#b7770d"]
            for i, (name, fit) in enumerate(valid_models.items()):
                rmsea = fit.get("rmsea", 0) or 0
                srmr  = fit.get("srmr",  0) or 0
                values = [
                    fit.get("cfi",  0) or 0,
                    fit.get("tli",  0) or 0,
                    max(0, 1 - rmsea),
                    max(0, 1 - srmr),
                ]
                values = [min(1, max(0, v)) for v in values]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    name=name,
                    line_color=plot_colors[i % len(plot_colors)],
                    fill="toself", opacity=0.25,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="simple_white", height=420,
                title="Fit Profile Comparison",
                font_color="#1a1a1a",
                paper_bgcolor="#ffffff",
            )
            st.plotly_chart(fig, use_container_width=True)

    return comp_df


def render_nested_test(fit_results):
    st.subheader("Step 4: Chi-Square Difference Test (Nested Models)")
    st.markdown(
        "The **Δchi2 test** is used when models are **nested** "
        "(one is a restricted version of another). "
        "A significant Δchi2 (p < .05) means the constraints significantly worsen fit."
    )

    valid_models = [k for k, v in fit_results.items() if "error" not in v]
    if len(valid_models) < 2:
        st.info("At least 2 successfully estimated models are needed.")
        return

    c1, c2 = st.columns(2)
    with c1:
        m1_name = st.selectbox("More constrained model", valid_models, key="nested_m1")
    with c2:
        m2_opts = [m for m in valid_models if m != m1_name]
        m2_name = st.selectbox("Less constrained model", m2_opts, key="nested_m2")

    fit1 = fit_results[m1_name]
    fit2 = fit_results[m2_name]

    chi2_1 = fit1.get("chi2")
    chi2_2 = fit2.get("chi2")
    df1    = fit1.get("df")
    df2    = fit2.get("df")

    if not all([chi2_1, chi2_2, df1, df2]):
        st.warning("Chi-square or df values missing. Cannot perform difference test.")
        return

    delta_chi2 = abs(float(chi2_1) - float(chi2_2))
    delta_df   = abs(int(float(df1)) - int(float(df2)))

    if delta_df == 0:
        st.warning("Models have the same degrees of freedom — they may not be nested.")
        return

    from scipy import stats as scipy_stats
    p_value = 1 - scipy_stats.chi2.cdf(delta_chi2, df=delta_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Delta chi2",  round(delta_chi2, 3))
    c2.metric("Delta df",    delta_df)
    c3.metric("p(Delta chi2)", f"{p_value:.4f}")

    if p_value < 0.05:
        badge("warning",
            f"Delta chi2({delta_df}) = {delta_chi2:.3f}, p = {p_value:.4f} — **significant**. "
            f"**{m1_name}** fits significantly worse than **{m2_name}**. "
            f"The additional constraints are not supported by the data."
        )
    else:
        badge("ok",
            f"Delta chi2({delta_df}) = {delta_chi2:.3f}, p = {p_value:.4f} — **not significant**. "
            f"**{m1_name}** fits as well as **{m2_name}** despite being more constrained. "
            "The more parsimonious model is preferred."
        )


def render_information_criteria(fit_results):
    st.subheader("Step 5: Information Criteria (AIC and BIC)")
    st.markdown(
        "AIC and BIC are used for **non-nested model comparison**. "
        "Lower values = better fit. "
        "**ΔAIC > 10** provides strong evidence against the worse model "
        "(Burnham & Anderson, 2002)."
    )

    valid = {k: v for k, v in fit_results.items()
             if "error" not in v and v.get("aic") is not None}
    if len(valid) < 2:
        st.info("At least 2 models with AIC values needed.")
        return

    aic_vals = {k: float(v["aic"]) for k, v in valid.items()}
    bic_vals = {k: float(v["bic"]) for k, v in valid.items() if v.get("bic")}
    min_aic  = min(aic_vals.values())
    best_aic = min(aic_vals, key=aic_vals.get)

    # Akaike weights
    d_aics   = {k: v - min_aic for k, v in aic_vals.items()}
    weights_raw = {k: np.exp(-0.5 * d) for k, d in d_aics.items()}
    total_w  = sum(weights_raw.values())
    weights  = {k: v / total_w for k, v in weights_raw.items()}

    rows = []
    for name in valid:
        aic    = aic_vals.get(name)
        bic    = bic_vals.get(name)
        d_aic  = round(aic - min_aic, 2)
        weight = round(weights.get(name, 0), 4)
        evidence = (
            "Best"         if d_aic == 0   else
            "Substantial"  if d_aic <= 2   else
            "Weak"         if d_aic <= 7   else
            "No support"
        )
        rows.append({
            "Model":          name,
            "AIC":            round(aic, 2) if aic else "—",
            "BIC":            round(bic, 2) if bic else "—",
            "Delta AIC":      d_aic,
            "Akaike Weight":  weight,
            "Evidence":       evidence,
        })

    ic_df = pd.DataFrame(rows)

    def color_evidence(val):
        if val == "Best":        return "color:#1a7a4a;font-weight:700"
        if val == "Substantial": return "color:#1a6fa8"
        if val == "No support":  return "color:#c0392b"
        return "color:#b7770d"

    st.dataframe(
        ic_df.style.map(color_evidence, subset=["Evidence"])
                   .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                   .set_table_styles([{
                       "selector":"th",
                       "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                   }]),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Note: Delta AIC = AIC difference from best model. "
        "Delta AIC <= 2: substantial support; 4-7: weak support; > 10: essentially no support. "
        "Akaike weight = probability that model is best-fitting in the set."
    )

    # AIC/BIC bar chart
    names  = list(valid.keys())
    aics   = [aic_vals[n] for n in names]
    bics   = [bic_vals.get(n, 0) for n in names]
    min_val = min(aics + [b for b in bics if b > 0]) * 0.998

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="AIC", x=names, y=aics,
        marker_color="#1a6fa8", opacity=0.85,
    ))
    if any(b > 0 for b in bics):
        fig.add_trace(go.Bar(
            name="BIC", x=names, y=bics,
            marker_color="#b7770d", opacity=0.85,
        ))
    fig.update_layout(
        barmode="group", template="simple_white", height=320,
        title="AIC and BIC by Model (lower = better)",
        yaxis=dict(range=[min_val, max(aics) * 1.002]),
        legend=dict(orientation="h", y=1.1),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

    badge("ok",
        f"Best-fitting model by AIC: <b>{best_aic}</b> (AIC = {round(min_aic, 2)}). "
        "Report the full model comparison table in your methods section."
    )


def render_model_recommendation(fit_results):
    st.subheader("Step 6: Model Selection Recommendation")

    from utils.thresholds import FIT
    valid = {k: v for k, v in fit_results.items() if "error" not in v}
    if not valid:
        return

    # Score each model
    scores = {name: 0 for name in valid}
    for name, fit in valid.items():
        if (fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"]: scores[name] += 1
        if (fit.get("cfi")   or 0)   >= FIT["cfi_acceptable"]:   scores[name] += 1
        if (fit.get("tli")   or 0)   >= FIT["tli_acceptable"]:   scores[name] += 1
        if (fit.get("srmr")  or 999) <= FIT["srmr_acceptable"]:  scores[name] += 1

    aic_vals = {k: v.get("aic", np.inf) for k, v in valid.items()}
    best_aic = min(aic_vals, key=aic_vals.get)
    scores[best_aic] = scores.get(best_aic, 0) + 2

    best_model = max(scores, key=scores.get)
    max_score  = max(scores.values())

    score_rows = [
        {
            "Model":        name,
            "Criteria Met": score,
            "Selected":     "Yes" if name == best_model else "",
        }
        for name, score in scores.items()
    ]
    score_df = pd.DataFrame(score_rows)

    def color_selected(val):
        if val == "Yes": return "color:#1a7a4a;font-weight:700"
        return ""

    st.dataframe(
        score_df.style.map(color_selected, subset=["Selected"])
                      .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                      .set_table_styles([{
                          "selector":"th",
                          "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                      }]),
        use_container_width=True, hide_index=True
    )

    badge("excellent",
        f"Recommended model: <b>{best_model}</b> "
        f"({max_score}/6 criteria met, best or tied AIC). "
        "This recommendation is based on fit index adequacy and AIC. "
        "<b>Final model selection must always be guided by theoretical considerations</b>, not statistics alone."
    )

    st.session_state["best_model"] = best_model


def render_model_comparison():
    st.title("Model Comparison")
    st.markdown(
        "Model comparison evaluates your **hypothesized model** against alternative "
        "(**rival**) models. This strengthens your argument by demonstrating "
        "your model fits better than plausible alternatives.\n\n"
        "> Testing against rival models is increasingly required by top journals "
        "in social science (Kline, 2016)."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df               = st.session_state["df"]
    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])

    if not constructs:
        st.error("No constructs defined.")
        return

    st.markdown("---")
    models = render_model_setup(constructs, structural_paths)
    st.markdown("---")
    fit_results = render_fit_models(models, df, constructs)

    if not fit_results:
        return

    st.markdown("---")
    render_comparison_table(fit_results)
    st.markdown("---")
    render_nested_test(fit_results)
    st.markdown("---")
    render_information_criteria(fit_results)
    st.markdown("---")
    render_model_recommendation(fit_results)

    st.markdown("---")
    badge("ok", "Model comparison complete. Proceed to Export Report.")
