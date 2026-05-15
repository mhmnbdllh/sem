"""
mediation.py - Mediation Analysis Module.
Uses R/lavaan bootstrap via r_bridge for methodologically correct mediation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.interpretation import interpret_mediation
from utils.apa_tables import mediation_table

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

def render_mediation_setup(constructs):
    st.subheader("Step 1: Mediation Model Setup")
    st.markdown(
        "Mediation tests whether the effect of **X** on **Y** "
        "operates through an intervening **mediator M**.\n\n"
        "Model: X --> M --> Y (with direct path X --> Y)"
    )

    construct_names = list(constructs.keys())
    if len(construct_names) < 3:
        st.warning("Mediation analysis requires at least 3 constructs (X, M, Y).")
        return None

    # Check if pre-configured in Data Input
    adv = st.session_state.get("advanced_options", {})
    default_x = adv.get("mediator_x", construct_names[0])
    default_m = adv.get("mediator_m", construct_names[1])
    default_y = adv.get("mediator_y", construct_names[2])

    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Predictor (X)", construct_names,
                             index=construct_names.index(default_x) if default_x in construct_names else 0,
                             key="med_x")
    with c2:
        m_opts = [c for c in construct_names if c != x_var]
        m_var  = st.selectbox("Mediator (M)", m_opts,
                              index=m_opts.index(default_m) if default_m in m_opts else 0,
                              key="med_m")
    with c3:
        y_opts = [c for c in construct_names if c not in [x_var, m_var]]
        if not y_opts:
            st.warning("Need at least 3 different constructs.")
            return None
        y_var = st.selectbox("Outcome (Y)", y_opts,
                             index=y_opts.index(default_y) if default_y in y_opts else 0,
                             key="med_y")

    # Visual diagram
    st.markdown(
        f'<div style="text-align:center;padding:16px;background:#f0f4f8;'
        f'border-radius:8px;color:#1a1a1a;margin:10px 0;border:1px solid #dde3ea">'
        f'<b style="color:#1a6fa8">{x_var}</b> '
        f'<span style="color:#555"> --(a)--> </span>'
        f'<b style="color:#b7770d">{m_var}</b>'
        f'<span style="color:#555"> --(b)--> </span>'
        f'<b style="color:#1a7a4a">{y_var}</b>'
        f'<br><span style="color:#888;font-size:0.85rem">'
        f'Direct path (c prime): {x_var} --> {y_var}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    n_boot = st.select_slider(
        "Bootstrap resamples",
        options=[1000, 2000, 5000, 10000],
        value=5000,
        help="5,000 is standard; 10,000 for publication-quality results."
    )
    ci_level = st.select_slider(
        "Confidence interval level",
        options=[0.90, 0.95, 0.99],
        value=0.95
    )

    return x_var, m_var, y_var, n_boot, ci_level


def render_mediation_results(df, constructs, x_var, m_var, y_var, n_boot, ci_level):
    st.subheader("Step 2: Mediation Analysis Results")
    st.markdown(
        f"Bootstrap mediation via R/lavaan ({n_boot:,} resamples, "
        f"{int(ci_level*100)}% BCa CI). "
        "Significance criterion: CI does not contain zero."
    )

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))

    try:
        from r_scripts.r_bridge import run_mediation, check_r_available
        r_check = check_r_available()
        if not r_check["available"]:
            st.error(f"R is not available: {r_check['message']}")
            return

        with st.spinner(f"Running bootstrap mediation via R/lavaan ({n_boot:,} resamples)... this may take 1-2 minutes."):
            result = run_mediation(
                df         = df,
                all_cols   = all_items,
                x_var      = x_var,
                m_var      = m_var,
                y_var      = y_var,
                constructs = {k: v for k, v in constructs.items()},
                n_boot     = n_boot,
                estimator  = st.session_state.get("recommended_estimator", "MLR")
            )

        if "error" in result:
            st.error(f"Mediation failed: {result['error']}")
            return

        st.session_state["mediation_results"] = result
        st.session_state["mediation_vars"]    = {"x": x_var, "m": m_var, "y": y_var}
        st.success("Mediation analysis complete.")

    except Exception as e:
        st.error(f"Mediation error: {str(e)}")
        return

    # ── Path Coefficients ──────────────────────────────────────
    st.markdown("Path Coefficients:")
    path_rows = []
    for key, label in [
        ("a_path",  f"{x_var} --> {m_var} (a path)"),
        ("b_path",  f"{m_var} --> {y_var} | {x_var} (b path)"),
        ("cp_path", f"{x_var} --> {y_var} direct (c prime)"),
        ("total",   f"{x_var} --> {y_var} total (c)"),
    ]:
        data = result.get(key, {})
        if not isinstance(data, dict): continue
        est   = _safe_float(data.get("est"))
        se    = _safe_float(data.get("se"))
        p_val = _safe_float(data.get("p"))
        path_rows.append({
            "Path":  label,
            "beta":  round(est, 3) if est is not None else "—",
            "SE":    round(se, 3)  if se  is not None else "—",
            "p":     round(p_val, 4) if p_val is not None else "—",
            "Sig.":  "Yes" if p_val is not None and p_val < 0.05 else "No" if p_val is not None else "—",
        })

    path_df = pd.DataFrame(path_rows)
    def color_sig(val):
        if val == "Yes": return "color:#1a7a4a;font-weight:700"
        elif val == "No": return "color:#c0392b"
        return ""
    st.dataframe(
        path_df.style.map(color_sig, subset=["Sig."])
                     .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                     .set_table_styles([{
                         "selector":"th",
                         "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                     }]),
        use_container_width=True, hide_index=True
    )

    # ── Effects Summary ────────────────────────────────────────
    st.markdown("**Effects Summary (Bootstrap BCa CI):**")
    med_table_df = mediation_table(result)
    if not med_table_df.empty:
        def color_sig2(val):
            if "Significant" in str(val) and "Not" not in str(val): return "color:#1a7a4a;font-weight:700"
            if "Not" in str(val): return "color:#c0392b"
            if val == "Sig.": return "color:#1a7a4a"
            return ""
        st.dataframe(
            med_table_df.style.map(color_sig2, subset=["Sig."])
                              .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                              .set_table_styles([{
                                  "selector":"th",
                                  "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                              }]),
            use_container_width=True, hide_index=True
        )
        st.caption(
            f"Note: Indirect effect significance based on {n_boot:,}-resample bootstrap "
            f"{int(ci_level*100)}% BCa CI. CI not containing zero = significant mediation."
        )

    # ── Extract key values ────────────────────────────────────
    indirect_data = result.get("indirect", {})
    if not isinstance(indirect_data, dict): indirect_data = {}
    indirect = _safe_float(indirect_data.get("est"))
    ci_lo    = _safe_float(indirect_data.get("ci_lo"))
    ci_hi    = _safe_float(indirect_data.get("ci_hi"))

    cp_data  = result.get("cp_path", {})
    if not isinstance(cp_data, dict): cp_data = {}
    direct   = _safe_float(cp_data.get("est"))

    # ── Mediation type determination ──────────────────────────
    st.markdown("**Mediation Type (Zhao et al., 2010):**")
    if indirect is not None and ci_lo is not None and ci_hi is not None:
        sig_indirect = not (ci_lo <= 0 <= ci_hi)
        sig_direct   = direct is not None and abs(direct) > 0.05

        if sig_indirect and not sig_direct:
            med_type  = "Full (Indirect-Only) Mediation"
            med_level = "excellent"
            med_desc  = f"{m_var} fully mediates the {x_var} to {y_var} relationship. The direct effect is negligible."
        elif sig_indirect and sig_direct:
            med_type  = "Partial Mediation"
            med_level = "ok"
            med_desc  = f"{m_var} partially mediates the relationship. Both direct and indirect paths are significant."
        elif not sig_indirect and direct is not None:
            med_type  = "No Mediation (Direct Effect Only)"
            med_level = "warning"
            med_desc  = f"The indirect path through {m_var} is not significant."
        else:
            med_type  = "No Effect"
            med_level = "critical"
            med_desc  = "Neither direct nor indirect effect is significant."

        badge(med_level, f"**{med_type}:** {med_desc}")

        # ── Full interpretation ──────────────────────────────
        st.markdown("**Full Interpretation:**")
        r = interpret_mediation(
            indirect  = indirect,
            ci_lo     = ci_lo,
            ci_hi     = ci_hi,
            direct    = direct,
            mediator  = m_var,
            predictor = x_var,
            outcome   = y_var,
        )
        badge(r["level"], r["message"])

        # ── VAF ───────────────────────────────────────────────
        total_data = result.get("total", {})
        if not isinstance(total_data, dict): total_data = {}
        total = _safe_float(total_data.get("est"))
        if total and abs(total) > 0.001 and indirect is not None:
            vaf = indirect / total
            st.markdown(f"**Variance Accounted For (VAF):** {vaf:.1%}")
            if vaf >= 0.80:
                badge("ok", f"VAF = {vaf:.1%} — suggests **full mediation** (VAF >= 80%).")
            elif vaf >= 0.20:
                badge("ok", f"VAF = {vaf:.1%} — suggests **partial mediation** (20% <= VAF < 80%).")
            else:
                badge("warning", f"VAF = {vaf:.1%} — mediation effect is very small (VAF < 20%).")

        # ── CI visualization ─────────────────────────────────
        with st.expander("Confidence Interval Visualization"):
            effects = []
            vals    = []
            ci_lows = []
            ci_highs= []

            for key, label in [
                ("indirect", "Indirect (a x b)"),
                ("cp_path",  "Direct (c prime)"),
                ("total",    "Total (c)"),
            ]:
                d = result.get(key, {})
                if not isinstance(d, dict): continue
                est = _safe_float(d.get("est"))
                lo  = _safe_float(d.get("ci_lo"))
                hi  = _safe_float(d.get("ci_hi"))
                if est is not None:
                    effects.append(label)
                    vals.append(est)
                    ci_lows.append(lo if lo is not None else est)
                    ci_highs.append(hi if hi is not None else est)

            if effects:
                fig = go.Figure()
                colors_ci = ["#c0392b" if sig_indirect and e == "Indirect (a x b)" else "#1a6fa8" for e in effects]
                for i, (eff, val, lo, hi, col) in enumerate(zip(effects, vals, ci_lows, ci_highs, colors_ci)):
                    fig.add_trace(go.Scatter(
                        x=[lo, hi], y=[eff, eff],
                        mode="lines",
                        line=dict(color=col, width=3),
                        showlegend=False,
                    ))
                    fig.add_trace(go.Scatter(
                        x=[val], y=[eff],
                        mode="markers",
                        marker=dict(color=col, size=10),
                        name=eff,
                        showlegend=False,
                    ))
                fig.add_vline(x=0, line_color="#555", line_dash="dash", line_width=1.5)
                fig.update_layout(
                    template="simple_white", height=280,
                    title=f"{int(ci_level*100,
        margin=dict(t=60, b=40, l=40, r=120),
    )}% Bootstrap BCa Confidence Intervals",
                    xaxis_title="Effect Size (standardized)",
                    font_color="#1a1a1a",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Intervals not crossing zero indicate significant effects.")


def render_mediation():
    st.title("Mediation Analysis")
    st.markdown(
        "Mediation tests whether the effect of a **predictor (X)** on an **outcome (Y)** "
        "operates through an intervening **mediator (M)**. "
        "This module uses **R/lavaan bootstrap** — the gold standard for indirect effects.\n\n"
        "> Significance criterion: Bootstrap BCa CI not containing zero (Hayes, 2018)."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df         = st.session_state["df"]
    constructs = st.session_state.get("constructs", {})

    if len(constructs) < 3:
        st.warning("Mediation requires at least 3 constructs. Define more in Data Input.")
        return

    st.markdown("---")
    setup = render_mediation_setup(constructs)
    if setup is None:
        return

    x_var, m_var, y_var, n_boot, ci_level = setup
    st.markdown("---")

    run_clicked = st.button("Run Mediation Analysis via R/lavaan", type="primary",
                             key="run_med_btn", use_container_width=True)

    if run_clicked:
        render_mediation_results(df, constructs, x_var, m_var, y_var, n_boot, ci_level)

    if st.session_state.get("mediation_results"):
        vars_stored = st.session_state.get("mediation_vars", {})
        if (vars_stored.get("x") == x_var and
            vars_stored.get("m") == m_var and
            vars_stored.get("y") == y_var and
            not run_clicked):
            render_mediation_results(df, constructs, x_var, m_var, y_var, n_boot, ci_level)

    st.markdown("---")
    badge("ok", "Mediation analysis complete. Proceed to Moderation Analysis or Export Report.")
