"""
moderation.py - Moderation Analysis Module.
Uses R via r_bridge for methodologically correct moderation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.interpretation import interpret_moderation

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


def render_moderation_setup(constructs):
    st.subheader("Step 1: Moderation Model Setup")
    st.markdown(
        "Moderation tests whether a **moderator (W)** changes the strength or direction "
        "of the relationship between **X** and **Y**.\n\n"
        "Model: X x W --> Y (interaction effect)"
    )

    construct_names = list(constructs.keys())
    if len(construct_names) < 3:
        st.warning("Moderation requires at least 3 constructs (X, W, Y).")
        return None

    adv       = st.session_state.get("advanced_options", {})
    default_x = adv.get("mod_x", construct_names[0])
    default_w = adv.get("mod_w", construct_names[1])
    default_y = adv.get("mod_y", construct_names[2])

    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox(
            "Predictor (X)", construct_names,
            index=construct_names.index(default_x) if default_x in construct_names else 0,
            key="mod_x_sel"
        )
    with c2:
        w_opts = [c for c in construct_names if c != x_var]
        w_var  = st.selectbox(
            "Moderator (W)", w_opts,
            index=w_opts.index(default_w) if default_w in w_opts else 0,
            key="mod_w_sel"
        )
    with c3:
        y_opts = [c for c in construct_names if c not in [x_var, w_var]]
        if not y_opts:
            st.warning("Need at least 3 different constructs.")
            return None
        y_var = st.selectbox(
            "Outcome (Y)", y_opts,
            index=y_opts.index(default_y) if default_y in y_opts else 0,
            key="mod_y_sel"
        )

    # Visual diagram
    st.markdown(
        f'<div style="text-align:center;padding:16px;background:#f0f4f8;'
        f'border-radius:8px;color:#1a1a1a;margin:10px 0;border:1px solid #dde3ea">'
        f'<b style="color:#1a6fa8">{x_var}</b> x '
        f'<b style="color:#b7770d">{w_var}</b>'
        f'<span style="color:#555"> --> </span>'
        f'<b style="color:#1a7a4a">{y_var}</b><br>'
        f'<span style="color:#888;font-size:0.85rem">'
        f'Does {w_var} moderate the {x_var} to {y_var} relationship?</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("Moderator levels for simple slope plots:")
    c1, c2, c3 = st.columns(3)
    with c1:
        low_sd  = st.number_input("Low W (-1 SD)", value=-1.0, step=0.5, key="mod_low")
    with c2:
        st.metric("Mean W", "0 (mean-centered)")
    with c3:
        high_sd = st.number_input("High W (+1 SD)", value=1.0, step=0.5, key="mod_high")

    badge("ok",
        "Variables are **mean-centered** before computing the interaction term "
        "to reduce multicollinearity (Aiken & West, 1991)."
    )

    return x_var, w_var, y_var, low_sd, high_sd


def render_moderation_results(df, constructs, x_var, w_var, y_var, low_sd, high_sd):
    st.subheader("Step 2: Moderation Analysis Results")

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))

    try:
        from r_scripts.r_bridge import run_moderation, check_r_available
        r_check = check_r_available()
        if not r_check["available"]:
            st.error(f"R is not available: {r_check['message']}")
            return

        with st.spinner("Running moderation analysis via R..."):
            result = run_moderation(
                df         = df,
                all_cols   = all_items,
                x_var      = x_var,
                w_var      = w_var,
                y_var      = y_var,
                constructs = {k: v for k, v in constructs.items()},
            )

        if "error" in result:
            st.error(f"Moderation failed: {result['error']}")
            return

        st.session_state["moderation_results"] = result
        st.session_state["moderation_vars"]    = {"x": x_var, "w": w_var, "y": y_var}
        st.success("Moderation analysis complete.")

    except Exception as e:
        st.error(f"Moderation error: {str(e)}")
        return

    # ── Regression Coefficients ────────────────────────────────
    st.markdown("**Regression Coefficients (standardized, mean-centered):**")

    b1   = _safe_float(result.get("b1"))
    b1_se= _safe_float(result.get("b1_se"))
    b1_t = _safe_float(result.get("b1_t"))
    b1_p = _safe_float(result.get("b1_p"))
    b2   = _safe_float(result.get("b2"))
    b2_se= _safe_float(result.get("b2_se"))
    b2_t = _safe_float(result.get("b2_t"))
    b2_p = _safe_float(result.get("b2_p"))
    b3   = _safe_float(result.get("b3"))
    b3_se= _safe_float(result.get("b3_se"))
    b3_t = _safe_float(result.get("b3_t"))
    b3_p = _safe_float(result.get("b3_p"))
    r2_1 = _safe_float(result.get("r2_1"))
    r2_2 = _safe_float(result.get("r2_2"))
    delta_r2 = _safe_float(result.get("delta_r2"))
    n    = _safe_float(result.get("n"))

    coef_rows = [
        {"Term": f"{x_var} (X)",      "beta": round(b1,3) if b1 is not None else "—",
         "SE": round(b1_se,3) if b1_se else "—",
         "t":  round(b1_t,3)  if b1_t  else "—",
         "p":  round(b1_p,4)  if b1_p is not None else "—",
         "Sig.": "Yes" if b1_p is not None and b1_p < 0.05 else "No"},
        {"Term": f"{w_var} (W)",      "beta": round(b2,3) if b2 is not None else "—",
         "SE": round(b2_se,3) if b2_se else "—",
         "t":  round(b2_t,3)  if b2_t  else "—",
         "p":  round(b2_p,4)  if b2_p is not None else "—",
         "Sig.": "Yes" if b2_p is not None and b2_p < 0.05 else "No"},
        {"Term": f"{x_var} x {w_var} (interaction)",
         "beta": round(b3,3) if b3 is not None else "—",
         "SE":   round(b3_se,3) if b3_se else "—",
         "t":    round(b3_t,3)  if b3_t  else "—",
         "p":    round(b3_p,4)  if b3_p is not None else "—",
         "Sig.": "Yes" if b3_p is not None and b3_p < 0.05 else "No"},
    ]
    coef_df = pd.DataFrame(coef_rows)

    def color_sig(val):
        if val == "Yes": return "color:#1a7a4a;font-weight:700"
        elif val == "No": return "color:#c0392b"
        return ""

    st.dataframe(
        coef_df.style.map(color_sig, subset=["Sig."])
                     .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                     .set_table_styles([{
                         "selector":"th",
                         "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                     }]),
        use_container_width=True, hide_index=True
    )
    st.caption("Note: Variables mean-centered prior to interaction computation (Aiken & West, 1991).")

    # ── R2 Summary ─────────────────────────────────────────────
    st.markdown("Model R2 Summary:")
    r2_rows = [
        {"Model": f"Model 1: {x_var} + {w_var} (no interaction)",
         "R2": round(r2_1, 4) if r2_1 else "—", "Delta R2": "—"},
        {"Model": f"Model 2: {x_var} + {w_var} + {x_var}x{w_var}",
         "R2": round(r2_2, 4) if r2_2 else "—",
         "Delta R2": round(delta_r2, 4) if delta_r2 else "—"},
    ]
    st.dataframe(
        pd.DataFrame(r2_rows).style
          .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
          .set_table_styles([{
              "selector":"th",
              "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
          }]),
        use_container_width=True, hide_index=True
    )

    # ── Interaction significance ───────────────────────────────
    if b3 is not None and b3_p is not None and delta_r2 is not None:
        r = interpret_moderation(b3, b3_p, delta_r2, x_var, w_var, y_var)
        badge(r["level"], r["message"])

    # ── Simple Slopes ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Simple Slope Analysis:**")
    st.markdown(
        "Simple slopes show the X --> Y relationship at different levels of W. "
        "Significant simple slopes indicate the effect exists at that W level."
    )

    slopes_raw = result.get("simple_slopes")
    if slopes_raw and isinstance(slopes_raw, dict):
        slope_rows = []
        for label, slope_data in slopes_raw.items():
            if not isinstance(slope_data, dict): continue
            slope = _safe_float(slope_data.get("slope"))
            se    = _safe_float(slope_data.get("se"))
            t_val = _safe_float(slope_data.get("t"))
            p_val = _safe_float(slope_data.get("p"))
            slope_rows.append({
                "W Level":          label,
                "Simple Slope (b)": round(slope, 4) if slope else "—",
                "SE":               round(se, 4)    if se    else "—",
                "t":                round(t_val, 3) if t_val is not None else "—",
                "p":                round(p_val, 4) if p_val is not None else "—",
                "Sig.":             "Yes" if p_val is not None and p_val < 0.05 else "No",
            })

        slope_df = pd.DataFrame(slope_rows)
        st.dataframe(
            slope_df.style.map(color_sig, subset=["Sig."])
                          .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                          .set_table_styles([{
                              "selector":"th",
                              "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                          }]),
            use_container_width=True, hide_index=True
        )

        # Simple slope interpretations
        for row in slope_rows:
            p_val = row.get("p")
            slope = row.get("Simple Slope (b)")
            try:
                p_val = float(p_val)
                slope = float(slope)
                w_lvl = row["W Level"]
                if p_val < 0.05:
                    direction = "positive" if slope > 0 else "negative"
                    badge("ok",
                        f"At **{w_lvl}** of {w_var}: simple slope = {slope:.4f} — "
                        f"**significant {direction} effect** of {x_var} on {y_var} (p = {p_val:.4f}). ✅"
                    )
                else:
                    badge("warning",
                        f"At **{w_lvl}** of {w_var}: simple slope = {slope:.4f} — "
                        f"**not significant** (p = {p_val:.4f}). ❌"
                    )
            except (TypeError, ValueError):
                pass

    # ── Interaction Plot ────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Interaction Plot:**")

    if b1 is not None and b3 is not None:
        b0 = _safe_float(result.get("b0"), 0)
        b2_val = b2 if b2 else 0

        x_range = np.linspace(-2, 2, 100)
        fig     = go.Figure()

        level_map = {
            "Low (-1 SD)":  low_sd,
            "Mean (0)":     0.0,
            "High (+1 SD)": high_sd,
        }
        colors_plot = {
            "Low (-1 SD)":  "#c0392b",
            "Mean (0)":     "#b7770d",
            "High (+1 SD)": "#1a7a4a",
        }

        for label, w_level in level_map.items():
            y_pred = b0 + b1 * x_range + b2_val * w_level + b3 * x_range * w_level
            fig.add_trace(go.Scatter(
                x=x_range, y=y_pred,
                mode="lines",
                name=f"{w_var}: {label}",
                line=dict(color=colors_plot[label], width=2.5),
            ))

        fig.update_layout(
            template="simple_white", height=420,
            title=f"Interaction Plot: {x_var} x {w_var} --> {y_var}",
            xaxis_title=f"{x_var} (standardized, mean-centered)",
            yaxis_title=f"{y_var} (standardized)",
            legend=dict(title=f"{w_var} level", x=0.02, y=0.98),
            font_color="#1a1a1a",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(t=60, b=60, l=60, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Pattern interpretation
        if b3 is not None and b3_p is not None and b3_p < 0.05:
            if b3 > 0:
                badge("ok",
                    f"The interaction plot shows the positive effect of **{x_var}** on **{y_var}** "
                    f"**strengthens as {w_var} increases** — an **enhancing moderation** pattern."
                )
            else:
                badge("ok",
                    f"The interaction plot shows the effect of **{x_var}** on **{y_var}** "
                    f"**weakens as {w_var} increases** — a **buffering moderation** pattern."
                )


def render_moderation():
    st.title("Moderation Analysis")
    st.markdown(
        "Moderation tests whether a **third variable (W)** changes the strength "
        "or direction of the relationship between **X** and **Y**.\n\n"
        "> Variables are **mean-centered** before creating the interaction term "
        "to reduce multicollinearity (Aiken & West, 1991)."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df         = st.session_state["df"]
    constructs = st.session_state.get("constructs", {})

    if len(constructs) < 3:
        st.warning("Moderation requires at least 3 constructs. Define more in Data Input.")
        return

    st.markdown("---")
    setup = render_moderation_setup(constructs)
    if setup is None:
        return

    x_var, w_var, y_var, low_sd, high_sd = setup
    st.markdown("---")

    if st.button("Run Moderation Analysis via R", type="primary",
                 key="run_mod_btn", use_container_width=True):
        render_moderation_results(df, constructs, x_var, w_var, y_var, low_sd, high_sd)
    elif st.session_state.get("moderation_results"):
        vars_stored = st.session_state.get("moderation_vars", {})
        if (vars_stored.get("x") == x_var and
            vars_stored.get("w") == w_var and
            vars_stored.get("y") == y_var):
            st.info("Showing previously computed results. Click the button above to re-run.")
            render_moderation_results(df, constructs, x_var, w_var, y_var, low_sd, high_sd)

    st.markdown("---")
    badge("ok", "Moderation analysis complete. Proceed to Measurement Invariance or Export Report.")
