"""
model_comparison.py
===================
Sprint 4 — Model Comparison Module for SEM Studio.

Covers:
- Rival/alternative model specification
- Chi-square difference test (Δχ²) for nested models
- AIC & BIC comparison for non-nested models
- CFI, RMSEA, SRMR comparison across models
- Model selection guidance
- Parsimony indices (PNFI, PGFI)
- Evidence ratio (Burnham & Anderson)
- Auto-interpretation and model selection recommendation

References:
    - Kline (2016). Principles and Practice of SEM (4th ed.)
    - Burnham & Anderson (2002). Model selection and multimodel inference.
    - Akaike (1974). AIC.
    - Schwarz (1978). BIC.
    - Bentler & Mooijaart (1989). Choice of structural model via parsimony.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from utils.thresholds import FIT
from utils.apa_tables import fit_indices_table


# ─── HELPERS ────────────────────────────────────────────────────

LEVEL_COLOR = {
    "excellent": "#2ecc71", "good": "#27ae60",
    "ok": "#3498db", "warning": "#f39c12", "critical": "#e74c3c",
}

def _badge(level: str, message: str):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>', unsafe_allow_html=True
    )


def extract_fit(model) -> dict:
    try:
        stats_result = semopy.calc_stats(model)
        fit = {}
        stat_map = {
            "chi2":  ["chi2", "Chi2"],
            "df":    ["dof", "df"],
            "p":     ["pvalue", "p"],
            "rmsea": ["rmsea", "RMSEA"],
            "cfi":   ["cfi", "CFI"],
            "tli":   ["tli", "TLI"],
            "srmr":  ["srmr", "SRMR"],
            "aic":   ["aic", "AIC"],
            "bic":   ["bic", "BIC"],
            "nfi":   ["nfi", "NFI"],
        }
        if isinstance(stats_result, pd.DataFrame):
            stats_dict = stats_result.iloc[0].to_dict()
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
        return fit
    except:
        return {}


# ─── SECTION 1: MODEL SETUP ─────────────────────────────────────

def render_model_setup(constructs: dict, structural_paths: list) -> dict:
    st.subheader("📝 Step 1: Define Models to Compare")
    st.markdown(
        "Compare your **hypothesized model** against one or more **rival (alternative) models**. "
        "Rival models should be theoretically meaningful — not arbitrary. "
        "Common alternatives: null model, saturated model, nested sub-models, or competing theories."
    )

    # Build default hypothesized model syntax
    meas_lines = [f"{c} =~ {' + '.join(items)}" for c, items in constructs.items() if items]
    struct_lines = [f"{out} ~ {pred}" for pred, out in structural_paths]
    hyp_syntax = "\n".join(meas_lines + ([""] + struct_lines if struct_lines else []))

    models = {}

    # ── Hypothesized model ───────────────────────────────────────
    with st.expander("📌 Model 1: Hypothesized Model (your main model)", expanded=True):
        st.markdown("This is your primary theoretical model.")
        m1_syntax = st.text_area(
            "Model 1 Syntax", value=hyp_syntax,
            height=max(120, (len(constructs) + len(structural_paths)) * 28),
            key="comp_m1"
        )
        models["Hypothesized Model"] = m1_syntax

    # ── Rival model 2 ───────────────────────────────────────────
    with st.expander("🔄 Model 2: Rival Model (add alternative)"):
        st.markdown(
            "Specify an alternative model. Examples:\n"
            "- Remove one path from the hypothesized model\n"
            "- Add a path not in the hypothesized model\n"
            "- Reverse the direction of a path\n"
            "- Saturated model (all possible paths)"
        )
        m2_name   = st.text_input("Model 2 name", value="Rival Model A", key="comp_m2_name")
        m2_syntax = st.text_area(
            "Model 2 Syntax",
            value="\n".join(meas_lines),  # measurement only as default rival
            height=120, key="comp_m2"
        )
        if m2_name and m2_syntax:
            models[m2_name] = m2_syntax

    # ── Rival model 3 (optional) ─────────────────────────────────
    with st.expander("🔄 Model 3: Additional Rival Model (optional)"):
        m3_name   = st.text_input("Model 3 name", value="Rival Model B", key="comp_m3_name")
        m3_syntax = st.text_area(
            "Model 3 Syntax", value="", height=100, key="comp_m3"
        )
        if m3_name and m3_syntax.strip():
            models[m3_name] = m3_syntax

    _badge("ok",
        "💡 **Tip:** A good rival model is one that a reasonable skeptic might propose as an "
        "alternative explanation. Testing against it strengthens your argument for the hypothesized model."
    )

    return models


# ─── SECTION 2: FIT MODELS ──────────────────────────────────────

def render_fit_models(models: dict, df: pd.DataFrame, constructs: dict) -> dict:
    st.subheader("⚙️ Step 2: Estimate All Models")

    if not SEMOPY_AVAILABLE:
        st.error("❌ semopy not installed.")
        return {}

    all_items = [item for items in constructs.values() for item in items]
    data      = df[all_items].dropna()

    if st.button("▶️ Estimate All Models", type="primary", key="run_comp"):
        fit_results = {}

        progress = st.progress(0)
        for i, (name, syntax) in enumerate(models.items()):
            if not syntax.strip():
                continue
            with st.spinner(f"Estimating: {name}..."):
                try:
                    model = semopy.Model(syntax)
                    model.fit(data)
                    fit = extract_fit(model)
                    fit["n_params"] = len(model.inspect())
                    fit_results[name] = fit
                    st.success(f"✅ {name} estimated.")
                except Exception as e:
                    st.error(f"❌ {name} failed: {str(e)}")
                    fit_results[name] = {"error": str(e)}
            progress.progress((i + 1) / len(models))

        st.session_state["comparison_results"] = fit_results
        return fit_results

    return st.session_state.get("comparison_results", {})


# ─── SECTION 3: FIT COMPARISON TABLE ────────────────────────────

def render_comparison_table(fit_results: dict) -> pd.DataFrame:
    st.subheader("📊 Step 3: Model Fit Comparison")

    if not fit_results:
        st.info("ℹ️ Run model estimation above to see comparison.")
        return pd.DataFrame()

    rows = []
    for name, fit in fit_results.items():
        if "error" in fit:
            rows.append({"Model": name, "Status": "❌ Error", **{k: "—" for k in
                         ["χ²", "df", "RMSEA", "CFI", "TLI", "SRMR", "AIC", "BIC"]}})
        else:
            rows.append({
                "Model": name,
                "χ²":    round(fit.get("chi2", 0) or 0, 3),
                "df":    int(fit.get("df", 0) or 0),
                "p":     round(fit.get("p", 0) or 0, 4),
                "RMSEA": round(fit.get("rmsea", 0) or 0, 3),
                "CFI":   round(fit.get("cfi", 0) or 0, 3),
                "TLI":   round(fit.get("tli", 0) or 0, 3),
                "SRMR":  round(fit.get("srmr", 0) or 0, 3),
                "AIC":   round(fit.get("aic", 0) or 0, 1),
                "BIC":   round(fit.get("bic", 0) or 0, 1),
            })

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.caption(
        "Note. Lower AIC/BIC = better fit (non-nested models). "
        "For nested models, use Δχ² test below."
    )

    # ── Radar Chart ──────────────────────────────────────────────
    with st.expander("📊 Fit Profile Radar Chart"):
        valid_models = {k: v for k, v in fit_results.items() if "error" not in v}
        if len(valid_models) >= 2:
            categories = ["CFI", "TLI", "1-RMSEA", "1-SRMR"]
            fig = go.Figure()
            colors = ["#2E86AB", "#e74c3c", "#2ecc71", "#f39c12"]
            for i, (name, fit) in enumerate(valid_models.items()):
                values = [
                    fit.get("cfi", 0) or 0,
                    fit.get("tli", 0) or 0,
                    1 - (fit.get("rmsea", 1) or 1),
                    1 - (fit.get("srmr", 1) or 1),
                ]
                values = [max(0, min(1, v)) for v in values]
                fig.add_trace(go.Scatterpolar(
                    r=values + [values[0]],
                    theta=categories + [categories[0]],
                    name=name,
                    line_color=colors[i % len(colors)],
                    fill="toself", opacity=0.3,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                template="plotly_dark", height=400,
                title="Fit Profile Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

    return comp_df


# ─── SECTION 4: NESTED MODEL TEST ───────────────────────────────

def render_nested_test(fit_results: dict):
    st.subheader("🔬 Step 4: Chi-Square Difference Test (Nested Models)")
    st.markdown(
        "The **Δχ² test** is used when models are **nested** (one is a restricted version of another). "
        "A significant Δχ² (p < .05) means the added constraint significantly worsens fit."
    )

    valid_models = [k for k, v in fit_results.items() if "error" not in v]
    if len(valid_models) < 2:
        st.info("ℹ️ At least 2 successfully estimated models are needed.")
        return

    col1, col2 = st.columns(2)
    with col1:
        m1_name = st.selectbox("More constrained model", valid_models, key="nested_m1")
    with col2:
        m2_opts = [m for m in valid_models if m != m1_name]
        m2_name = st.selectbox("Less constrained model", m2_opts, key="nested_m2")

    fit1 = fit_results[m1_name]
    fit2 = fit_results[m2_name]

    delta_chi2 = abs((fit1.get("chi2", 0) or 0) - (fit2.get("chi2", 0) or 0))
    delta_df   = abs(int((fit1.get("df", 0) or 0) - (fit2.get("df", 0) or 0)))

    if delta_df == 0:
        st.warning("⚠️ Models have same degrees of freedom — they may not be nested.")
        return

    p_value = 1 - stats.chi2.cdf(delta_chi2, df=delta_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Δχ²", round(delta_chi2, 3))
    c2.metric("Δdf", delta_df)
    c3.metric("p(Δχ²)", f"{p_value:.4f}")

    if p_value < 0.05:
        _badge("warning",
            f"Δχ²({delta_df}) = {delta_chi2:.3f}, p = {p_value:.4f} — **significant**. "
            f"**{m1_name}** fits significantly worse than **{m2_name}**. "
            f"The additional constraints in {m1_name} are not supported by the data."
        )
    else:
        _badge("ok",
            f"Δχ²({delta_df}) = {delta_chi2:.3f}, p = {p_value:.4f} — **not significant**. "
            f"**{m1_name}** fits as well as **{m2_name}** despite being more constrained. "
            "The more parsimonious model is preferred."
        )


# ─── SECTION 5: AIC/BIC COMPARISON ─────────────────────────────

def render_information_criteria(fit_results: dict):
    st.subheader("📉 Step 5: Information Criteria (AIC & BIC)")
    st.markdown(
        "AIC and BIC are used for **non-nested model comparison**. "
        "Lower values indicate better fit. "
        "**ΔAIC > 10** provides strong evidence against the worse-fitting model (Burnham & Anderson, 2002)."
    )

    valid = {k: v for k, v in fit_results.items() if "error" not in v and v.get("aic")}
    if len(valid) < 2:
        st.info("ℹ️ At least 2 models with AIC values needed.")
        return

    aic_vals  = {k: v["aic"] for k, v in valid.items()}
    bic_vals  = {k: v["bic"] for k, v in valid.items() if v.get("bic")}
    min_aic   = min(aic_vals.values())
    best_aic  = [k for k, v in aic_vals.items() if v == min_aic][0]

    rows = []
    for name in valid:
        aic  = aic_vals.get(name, None)
        bic  = bic_vals.get(name, None)
        daic = round(aic - min_aic, 2) if aic else None
        # Akaike weight
        all_daic   = [v - min_aic for v in aic_vals.values()]
        weights    = np.exp(-0.5 * np.array(all_daic))
        weights    /= weights.sum()
        w_dict     = dict(zip(aic_vals.keys(), weights))

        rows.append({
            "Model":   name,
            "AIC":     round(aic, 2) if aic else "—",
            "BIC":     round(bic, 2) if bic else "—",
            "ΔAIC":    daic,
            "Akaike Weight": round(float(w_dict.get(name, 0)), 4),
            "Evidence": "✅ Best" if name == best_aic else
                        ("Substantial" if daic and daic <= 2 else
                         ("Weak" if daic and daic <= 7 else "❌ Poor")),
        })

    ic_df = pd.DataFrame(rows)

    def color_ev(val):
        if "Best" in str(val): return "color:#2ecc71;font-weight:bold"
        if "Poor" in str(val): return "color:#e74c3c"
        return ""

    st.dataframe(
        ic_df.style.applymap(color_ev, subset=["Evidence"]),
        use_container_width=True, hide_index=True
    )
    st.caption(
        "Note. ΔAIC = AIC difference from best model. "
        "ΔAIC ≤ 2: substantial support; 4–7: weak support; > 10: essentially no support. "
        "Akaike weight = probability that model is the best-fitting model in the set."
    )

    _badge("ok",
        f"**Best-fitting model by AIC: {best_aic}** "
        f"(AIC = {round(min_aic, 2)}). "
        "Report the full model comparison table in your methods section."
    )


# ─── SECTION 6: RECOMMENDATION ──────────────────────────────────

def render_model_recommendation(fit_results: dict):
    st.subheader("🏆 Step 6: Model Selection Recommendation")

    valid = {k: v for k, v in fit_results.items() if "error" not in v}
    if not valid:
        return

    # Score each model on multiple criteria
    scores = {name: 0 for name in valid}
    for name, fit in valid.items():
        if (fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"]: scores[name] += 1
        if (fit.get("cfi")   or 0)   >= FIT["cfi_acceptable"]:   scores[name] += 1
        if (fit.get("tli")   or 0)   >= FIT["tli_acceptable"]:   scores[name] += 1
        if (fit.get("srmr")  or 999) <= FIT["srmr_acceptable"]:  scores[name] += 1

    aic_vals = {k: v.get("aic", np.inf) for k, v in valid.items()}
    best_aic = min(aic_vals, key=aic_vals.get)
    scores[best_aic] += 2  # extra weight for best AIC

    best_model = max(scores, key=scores.get)

    _badge("excellent",
        f"🏆 **Recommended model: {best_model}** "
        f"(score = {scores[best_model]}/6 criteria met). "
        "This recommendation is based on fit index adequacy and AIC comparison. "
        "**Final model selection must be guided by theoretical considerations**, not statistics alone."
    )

    st.markdown("**Full Scoring:**")
    score_df = pd.DataFrame([
        {"Model": k, "Criteria Met": v, "Selected": "✅" if k == best_model else ""}
        for k, v in scores.items()
    ])
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    st.session_state["best_model"] = best_model


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_model_comparison():
    st.title("📑 Model Comparison")
    st.markdown(
        "Model comparison evaluates your **hypothesized model** against alternative "
        "(**rival**) models. This strengthens your argument for the chosen model by "
        "demonstrating it fits better than plausible alternatives.\n\n"
        "> 📌 *Testing against rival models is increasingly required by top journals "
        "in social science (Kline, 2016).*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    if not SEMOPY_AVAILABLE:
        st.error("❌ semopy not installed.")
        return

    df               = st.session_state["df"]
    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])

    if not constructs:
        st.error("❌ No constructs defined.")
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
    st.success(
        "✅ Model comparison complete. "
        "Proceed to **Export Report** to generate your full analysis report."
    )
