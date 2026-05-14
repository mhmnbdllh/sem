"""
visualization.py - Path Diagram & Visualization Module.
Interactive path diagram using Plotly.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import math

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


def compute_layout(constructs, structural_paths):
    """Compute x,y positions for all nodes."""
    construct_names = list(constructs.keys())
    endogenous  = set(out for _, out in structural_paths)
    exogenous   = set(construct_names) - endogenous

    positions = {}

    ex_list = sorted(list(exogenous))
    en_list = sorted(list(endogenous))

    if not structural_paths:
        for i, name in enumerate(construct_names):
            angle = 2 * math.pi * i / max(len(construct_names), 1)
            positions[name] = (3 * math.cos(angle), 3 * math.sin(angle))
    else:
        for i, name in enumerate(ex_list):
            y = (i - (len(ex_list) - 1) / 2) * 3.0
            positions[name] = (-4.0, y)
        for i, name in enumerate(en_list):
            y = (i - (len(en_list) - 1) / 2) * 3.0
            positions[name] = (4.0, y)

    # Indicators fan out from constructs
    for cname, items in constructs.items():
        if cname not in positions:
            positions[cname] = (0, 0)
        cx, cy = positions[cname]
        n_items = len(items)
        direction = -1 if cx <= 0 else 1

        for j, item in enumerate(items):
            y_offset = (j - (n_items - 1) / 2) * 1.1
            positions[item] = (cx + direction * 3.2, cy + y_offset)

    return positions


def build_path_diagram(constructs, structural_paths,
                       sem_paths=None, cfa_loadings=None,
                       r2_data=None, show_indicators=True):
    """Build interactive Plotly path diagram."""

    positions = compute_layout(constructs, structural_paths)

    path_dict = {}
    if sem_paths:
        for p in sem_paths:
            key = (p.get("predictor",""), p.get("outcome",""))
            path_dict[key] = p

    loading_dict = cfa_loadings or {}

    r2_dict = {}
    if r2_data:
        for row in r2_data:
            if isinstance(row, dict) and isinstance(row.get("R2"), (int, float)):
                r2_dict[row["Construct"]] = row["R2"]

    fig = go.Figure()

    # ── Measurement paths (construct → indicator) ──────────────
    if show_indicators:
        for cname, items in constructs.items():
            if cname not in positions: continue
            cx, cy = positions[cname]
            for item in items:
                if item not in positions: continue
                ix, iy = positions[item]

                loading = None
                if cname in loading_dict and item in loading_dict[cname]:
                    loading = loading_dict[cname][item]

                color = (
                    "#1a7a4a" if loading and abs(float(loading)) >= 0.70 else
                    "#1a6fa8" if loading and abs(float(loading)) >= 0.50 else
                    "#b7770d" if loading else
                    "#aaaaaa"
                )
                label = f"lambda={loading:.2f}" if loading else ""

                fig.add_trace(go.Scatter(
                    x=[cx, ix], y=[cy, iy],
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ))
                if label:
                    fig.add_annotation(
                        x=(cx+ix)/2, y=(cy+iy)/2,
                        text=label, showarrow=False,
                        font=dict(size=8, color=color),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor=color, borderwidth=1,
                    )

    # ── Structural paths ────────────────────────────────────────
    for pred, out in structural_paths:
        if pred not in positions or out not in positions: continue
        px_, py_ = positions[pred]
        ox, oy  = positions[out]

        p_info = path_dict.get((pred, out), {})
        beta   = p_info.get("beta")
        p_val  = p_info.get("p")

        if beta is not None and p_val is not None:
            sig    = float(p_val) < 0.05
            color  = "#1a7a4a" if (sig and float(beta) > 0) else "#c0392b" if (sig and float(beta) < 0) else "#aaaaaa"
            width  = 3.0 if sig else 1.5
            dash   = "solid" if sig else "dash"
            stars  = "***" if float(p_val)<0.001 else "**" if float(p_val)<0.01 else "*" if float(p_val)<0.05 else "ns"
            label  = f"beta={float(beta):.2f}{stars}"
        else:
            color, width, dash, label = "#aaaaaa", 2.0, "solid", f"{pred}->{out}"

        # Line drawn inside arrow section below (shortened to avoid covering nodes)
        fig.add_trace(go.Scatter(
            x=[px_, ox], y=[py_, oy],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            hovertext=f"{pred} --> {out}<br>{label}",
            hoverinfo="text",
            showlegend=False,
        ))

        # Arrowhead - stop before node center to avoid covering text
        dx, dy = ox - px_, oy - py_
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            ux, uy = dx/dist, dy/dist
            # node_offset: arrow tip stops 0.9 units before node center
            node_offset = 0.9
            # arrow start: 0.9 units before source node too
            tip_x   = ox - ux * node_offset
            tip_y   = oy - uy * node_offset
            start_x = px_ + ux * node_offset
            start_y = py_ + uy * node_offset
            # Draw line from start to just before tip
            fig.add_trace(go.Scatter(
                x=[start_x, tip_x], y=[start_y, tip_y],
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                hoverinfo="skip", showlegend=False,
            ))
            # Arrowhead annotation at tip
            fig.add_annotation(
                x=tip_x, y=tip_y,
                ax=tip_x - ux*0.3, ay=tip_y - uy*0.3,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.2, arrowwidth=2,
                arrowcolor=color, showarrow=True, text="",
            )

        # Label at midpoint - offset perpendicular to path to avoid overlap
        mx = (px_ + ox) / 2
        my = (py_ + oy) / 2
        # Perpendicular offset so label doesn't sit on arrow line
        if dist > 0:
            perp_x = -dy/dist * 0.35
            perp_y =  dx/dist * 0.35
        else:
            perp_x, perp_y = 0, 0.35
        fig.add_annotation(
            x=mx + perp_x, y=my + perp_y,
            text=label, showarrow=False,
            font=dict(size=10, color=color, family="monospace"),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=color, borderwidth=1,
            borderpad=3,
        )

    # ── Construct nodes (ovals) ─────────────────────────────────
    for cname in constructs:
        if cname not in positions: continue
        cx, cy = positions[cname]
        r2 = r2_dict.get(cname)
        r2_label = f"  R2={r2:.2f}" if r2 else ""
        display_label = f"{cname}{r2_label}"

        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(
                symbol="circle", size=55,
                color="#e8f4fc",
                line=dict(color="#2E86AB", width=2.5),
            ),
            text=[cname],
            textposition="middle center",
            textfont=dict(size=11, color="#1a1a1a", family="Arial Black"),
            hovertext=display_label,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Indicator nodes (rectangles) ────────────────────────────
    if show_indicators:
        all_items = [item for items in constructs.values() for item in items]
        for item in all_items:
            if item not in positions: continue
            ix, iy = positions[item]
            fig.add_trace(go.Scatter(
                x=[ix], y=[iy],
                mode="markers+text",
                marker=dict(
                    symbol="square", size=34,
                    color="#f5f5f5",
                    line=dict(color="#555555", width=1.5),
                ),
                text=[item],
                textposition="middle center",
                textfont=dict(size=8, color="#333333"),
                hovertext=item,
                hoverinfo="text",
                showlegend=False,
            ))

    all_x = [v[0] for v in positions.values()]
    all_y = [v[1] for v in positions.values()]
    pad   = 1.5

    fig.update_layout(
        template="simple_white",
        height=max(500, len([i for items in constructs.values() for i in items]) * 32 + 150),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_x)-pad, max(all_x)+pad],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_y)-pad, max(all_y)+pad],
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=50, b=20),
        title=dict(
            text="SEM Path Diagram",
            font=dict(size=16, color="#1a1a1a"),
        ),
        font_color="#1a1a1a",
    )

    return fig


def render_path_diagram():
    st.subheader("SEM Path Diagram")
    st.markdown(
        "Interactive path diagram of your complete SEM model. "
        "**Oval nodes** = latent constructs. **Square nodes** = observed indicators. "
        "**Green** = significant positive path. **Red** = significant negative. "
        "**Gray dashed** = non-significant."
    )

    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])
    sem_paths        = st.session_state.get("sem_paths", [])
    cfa_loadings     = st.session_state.get("cfa_loadings", {})
    r2_data          = st.session_state.get("sem_r2", [])

    if not constructs:
        st.warning("No constructs defined. Complete Data Input first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        show_loadings   = st.checkbox("Show factor loadings", value=True, key="diag_loadings")
    with c2:
        show_r2         = st.checkbox("Show R2 values", value=True, key="diag_r2")
    with c3:
        show_indicators = st.checkbox("Show indicators", value=True, key="diag_indicators")

    fig = build_path_diagram(
        constructs       = constructs,
        structural_paths = structural_paths,
        sem_paths        = sem_paths,
        cfa_loadings     = cfa_loadings if show_loadings else {},
        r2_data          = r2_data if show_r2 else [],
        show_indicators  = show_indicators,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div style="font-size:0.82rem;color:#555;padding:4px 0">'
        'Green solid = significant positive | '
        'Red solid = significant negative | '
        'Gray dashed = non-significant | '
        'Dotted = measurement path'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Note: Path labels show standardized beta. * p < .05; ** p < .01; *** p < .001; ns = not significant.")
    badge("ok",
        "Tip: Hover over nodes and arrows for details. "
        "Use the camera icon in the top-right to save as PNG."
    )


def render_fit_dashboard():
    st.subheader("Model Fit Dashboard")

    cfa_fit = st.session_state.get("cfa_fit", {})
    sem_fit = st.session_state.get("sem_fit", {})

    if not cfa_fit and not sem_fit:
        st.info("Run CFA and/or SEM first to see the fit dashboard.")
        return

    tab1, tab2 = st.tabs(["CFA Fit", "SEM Fit"])

    for tab, fit, label in [(tab1, cfa_fit, "CFA"), (tab2, sem_fit, "SEM")]:
        with tab:
            if not fit:
                st.info(f"Run {label} first.")
                continue

            indices = {
                "RMSEA": {"val": fit.get("rmsea"), "good": 0.06, "ok": 0.08, "direction": "lower"},
                "CFI":   {"val": fit.get("cfi"),   "good": 0.95, "ok": 0.90, "direction": "higher"},
                "TLI":   {"val": fit.get("tli"),   "good": 0.95, "ok": 0.90, "direction": "higher"},
                "SRMR":  {"val": fit.get("srmr"),  "good": 0.05, "ok": 0.08, "direction": "lower"},
            }

            cols = st.columns(len(indices))
            for col, (name, info) in zip(cols, indices.items()):
                val = info["val"]
                if val is None:
                    col.metric(name, "—")
                    continue
                if info["direction"] == "lower":
                    status = "Good" if val <= info["good"] else "Acceptable" if val <= info["ok"] else "Poor"
                else:
                    status = "Good" if val >= info["good"] else "Acceptable" if val >= info["ok"] else "Poor"
                col.metric(label=f"{name}", value=f"{val:.3f}", delta=status)

            # Bar chart of fit indices
            valid_indices = {k: v["val"] for k, v in indices.items() if v["val"] is not None}
            if valid_indices:
                bar_colors = []
                for name, info in indices.items():
                    val = info["val"]
                    if val is None: continue
                    if info["direction"] == "lower":
                        c = "#1a7a4a" if val <= info["good"] else "#b7770d" if val <= info["ok"] else "#c0392b"
                    else:
                        c = "#1a7a4a" if val >= info["good"] else "#b7770d" if val >= info["ok"] else "#c0392b"
                    bar_colors.append(c)

                fig = go.Figure(go.Bar(
                    x=list(valid_indices.keys()),
                    y=list(valid_indices.values()),
                    marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in valid_indices.values()],
                    textposition="outside",
                ))
                fig.update_layout(
                    template="simple_white", height=280,
                    title=f"{label} Fit Indices",
                    yaxis=dict(range=[0, 1.1]),
                    font_color="#1a1a1a",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                )
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Green = Good | Orange = Acceptable | Red = Poor")


def render_effect_sizes():
    st.subheader("Effect Size Visualization")

    sem_paths = st.session_state.get("sem_paths", [])
    if not sem_paths:
        st.info("Run SEM first to see effect sizes.")
        return

    valid = [p for p in sem_paths if p.get("beta") is not None]
    if not valid:
        return

    labels  = [f"{p['predictor']} to {p['outcome']}" for p in valid]
    betas   = [float(p["beta"]) for p in valid]
    p_vals  = [float(p.get("p", 1.0)) for p in valid]
    colors  = [
        "#1a7a4a" if (pv < 0.05 and b > 0) else
        "#c0392b" if (pv < 0.05 and b < 0) else
        "#aaaaaa"
        for b, pv in zip(betas, p_vals)
    ]

    fig = go.Figure(go.Bar(
        x=betas, y=labels, orientation="h",
        marker_color=colors,
        text=[f"beta={b:.3f}" + ("*" if p < 0.05 else "") for b, p in zip(betas, p_vals)],
        textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#333", line_width=1)
    fig.add_vline(x=0.10,  line_dash="dot", line_color="#b7770d", annotation_text="small")
    fig.add_vline(x=-0.10, line_dash="dot", line_color="#b7770d")
    fig.add_vline(x=0.30,  line_dash="dash", line_color="#1a7a4a", annotation_text="medium")
    fig.add_vline(x=-0.30, line_dash="dash", line_color="#1a7a4a")
    fig.update_layout(
        template="simple_white",
        height=max(300, len(valid)*55+100),
        title="Standardized Path Coefficients (beta)",
        xaxis_title="Standardized Coefficient (beta)",
        xaxis=dict(range=[-1, 1]),
        margin=dict(l=200, r=80, t=60, b=40),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note: * p < .05; ** p < .01; *** p < .001; ns = not significant. Green = significant positive | Red = significant negative | Gray = non-significant.")


def render_construct_correlations():
    st.subheader("Latent Construct Correlations")

    df         = st.session_state.get("df")
    constructs = st.session_state.get("constructs", {})

    if df is None or not constructs:
        st.info("Complete Data Input and run CFA first.")
        return

    parcel_df = pd.DataFrame()
    for cname, items in constructs.items():
        valid = [c for c in items if c in df.columns]
        if valid:
            parcel_df[cname] = df[valid].mean(axis=1)

    parcel_df = parcel_df.dropna()
    if parcel_df.shape[1] < 2:
        st.warning("Not enough constructs with valid data.")
        return

    corr = parcel_df.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="simple_white",
        text_auto=".3f",
        title="Inter-Construct Correlation Matrix",
        aspect="auto",
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=max(320, len(constructs)*70),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        coloraxis_colorbar=dict(thickness=10, len=0.75, title="", tickfont=dict(size=9)),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note: Correlations based on construct parcel scores (mean of indicators). "
        "Values > .85 may indicate discriminant validity concerns."
    )


def render_visualization():
    st.title("Path Diagram and Visualizations")
    st.markdown(
        "Visual representations of your SEM model results. "
        "All charts are **interactive** — hover for details, zoom, and pan. "
        "Use the camera icon to export as PNG."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Path Diagram",
        "Fit Dashboard",
        "Effect Sizes",
        "Construct Correlations",
    ])

    with tab1:
        render_path_diagram()
    with tab2:
        render_fit_dashboard()
    with tab3:
        render_effect_sizes()
    with tab4:
        render_construct_correlations()
