"""
visualization.py
================
Sprint 5 — Path Diagram & Visualization Module for SEM Studio.

Covers:
- Interactive path diagram (HTML/SVG via networkx + plotly)
- Standardized path coefficients on diagram
- Significance indicators on paths
- R² displayed on endogenous constructs
- Factor loadings on measurement paths
- Color-coded paths (significant/non-significant/negative)
- Exportable diagram (PNG, SVG)
- Correlation heatmap
- Fit index dashboard
- Effect size visualization

References:
    - Kline (2016). Principles and Practice of SEM (4th ed.)
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import math


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


# ─── PATH DIAGRAM LAYOUT ─────────────────────────────────────────

def compute_layout(constructs: dict, structural_paths: list) -> dict:
    """
    Compute x,y positions for constructs and indicators.
    Returns dict: {node_name: (x, y)}
    """
    construct_names = list(constructs.keys())
    n_constructs    = len(construct_names)

    # Identify endogenous (has incoming structural path)
    endogenous  = set(out for _, out in structural_paths)
    exogenous   = set(construct_names) - endogenous

    positions = {}

    # Exogenous: left column
    ex_list = sorted(list(exogenous))
    for i, name in enumerate(ex_list):
        y = (i - (len(ex_list) - 1) / 2) * 3.0
        positions[name] = (-4.0, y)

    # Endogenous: right column
    en_list = sorted(list(endogenous))
    for i, name in enumerate(en_list):
        y = (i - (len(en_list) - 1) / 2) * 3.0
        positions[name] = (4.0, y)

    # If no structural paths defined, arrange in circle
    if not structural_paths:
        for i, name in enumerate(construct_names):
            angle = 2 * math.pi * i / n_constructs
            positions[name] = (3 * math.cos(angle), 3 * math.sin(angle))

    # Indicators: fan out from constructs
    for cname, items in constructs.items():
        if cname not in positions:
            positions[cname] = (0, 0)
        cx, cy = positions[cname]
        n_items = len(items)
        direction = -1 if cx < 0 else 1  # indicators go further out

        for j, item in enumerate(items):
            y_offset = (j - (n_items - 1) / 2) * 1.2
            x_offset = direction * 3.5
            positions[item] = (cx + x_offset, cy + y_offset)

    return positions


def build_path_diagram(
    constructs: dict,
    structural_paths: list,
    sem_paths: list = None,
    cfa_loadings: dict = None,
    r2_data: list = None,
) -> go.Figure:
    """
    Build an interactive Plotly path diagram.

    Parameters
    ----------
    constructs       : {construct: [items]}
    structural_paths : [(predictor, outcome)]
    sem_paths        : list of dicts with beta, p per path
    cfa_loadings     : {construct: {item: loading}}
    r2_data          : list of dicts with construct, R²
    """
    positions = compute_layout(constructs, structural_paths)

    # Build lookup dicts
    path_dict = {}
    if sem_paths:
        for p in sem_paths:
            key = (p.get("predictor"), p.get("outcome"))
            path_dict[key] = p

    loading_dict = cfa_loadings or {}

    r2_dict = {}
    if r2_data:
        for row in r2_data:
            if isinstance(row.get("R²"), (int, float)):
                r2_dict[row["Construct"]] = row["R²"]

    fig = go.Figure()

    # ── Draw measurement paths (construct → indicator) ────────────
    for cname, items in constructs.items():
        if cname not in positions:
            continue
        cx, cy = positions[cname]
        for item in items:
            if item not in positions:
                continue
            ix, iy = positions[item]

            loading = None
            if cname in loading_dict and item in loading_dict[cname]:
                loading = loading_dict[cname][item]

            color = "#4a90d9"
            if loading is not None:
                color = "#2ecc71" if abs(loading) >= 0.70 else \
                        "#f39c12" if abs(loading) >= 0.50 else "#e74c3c"

            label = f"λ={loading:.2f}" if loading is not None else ""

            # Arrow line
            fig.add_trace(go.Scatter(
                x=[cx, ix], y=[cy, iy],
                mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                hoverinfo="skip",
                showlegend=False,
            ))

            # Loading label at midpoint
            if label:
                mx, my = (cx + ix) / 2, (cy + iy) / 2
                fig.add_annotation(
                    x=mx, y=my, text=label,
                    showarrow=False,
                    font=dict(size=8, color=color),
                    bgcolor="rgba(15,17,23,0.7)",
                )

    # ── Draw structural paths ─────────────────────────────────────
    for pred, out in structural_paths:
        if pred not in positions or out not in positions:
            continue
        px_, py = positions[pred]
        ox, oy  = positions[out]

        # Get path stats
        p_info = path_dict.get((pred, out), {})
        beta   = p_info.get("beta")
        p_val  = p_info.get("p")

        if beta is not None and p_val is not None:
            sig   = p_val < 0.05
            color = "#2ecc71" if sig and beta > 0 else \
                    "#e74c3c" if sig and beta < 0 else "#888888"
            width = 3.0 if sig else 1.5
            dash  = "solid" if sig else "dash"
            stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else \
                    "*" if p_val < 0.05 else "ns"
            label = f"β={beta:.2f}{stars}"
        else:
            color = "#888888"
            width = 2.0
            dash  = "solid"
            label = f"{pred}→{out}"

        # Arrow body
        fig.add_trace(go.Scatter(
            x=[px_, ox], y=[py, oy],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            hovertext=f"{pred} → {out}<br>{label}",
            hoverinfo="text",
            showlegend=False,
        ))

        # Arrowhead
        dx = ox - px_
        dy = oy - py
        dist = math.sqrt(dx**2 + dy**2)
        if dist > 0:
            ux, uy = dx / dist, dy / dist
            ax = ox - ux * 0.6
            ay = oy - uy * 0.6
            fig.add_annotation(
                x=ox, y=oy, ax=ax, ay=ay,
                xref="x", yref="y", axref="x", ayref="y",
                arrowhead=2, arrowsize=1.2,
                arrowwidth=2, arrowcolor=color,
                showarrow=True, text="",
            )

        # Label at midpoint
        mx = (px_ + ox) / 2
        my = (py + oy) / 2 + 0.25
        fig.add_annotation(
            x=mx, y=my, text=label,
            showarrow=False,
            font=dict(size=10, color=color, family="monospace"),
            bgcolor="rgba(15,17,23,0.85)",
            bordercolor=color, borderwidth=1,
        )

    # ── Draw construct nodes (ovals) ──────────────────────────────
    for cname, (cx, cy) in positions.items():
        if cname in constructs:
            r2 = r2_dict.get(cname)
            r2_label = f"<br><sub>R²={r2:.2f}</sub>" if r2 else ""
            label    = f"<b>{cname}</b>{r2_label}"

            fig.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=52,
                    color="#1a2a3a",
                    line=dict(color="#2E86AB", width=2.5),
                ),
                text=[f"<b>{cname}</b>"],
                textposition="middle center",
                textfont=dict(size=11, color="#ffffff"),
                hovertext=f"<b>{cname}</b>{r2_label}",
                hoverinfo="text",
                showlegend=False,
            ))

    # ── Draw indicator nodes (rectangles via scatter) ─────────────
    all_items = [item for items in constructs.values() for item in items]
    for item in all_items:
        if item not in positions:
            continue
        ix, iy = positions[item]
        fig.add_trace(go.Scatter(
            x=[ix], y=[iy],
            mode="markers+text",
            marker=dict(
                symbol="square",
                size=36,
                color="#111827",
                line=dict(color="#4a90d9", width=1.5),
            ),
            text=[item],
            textposition="middle center",
            textfont=dict(size=8, color="#c0d0e0"),
            hovertext=item,
            hoverinfo="text",
            showlegend=False,
        ))

    # ── Layout ───────────────────────────────────────────────────
    all_x = [v[0] for v in positions.values()]
    all_y = [v[1] for v in positions.values()]
    pad   = 1.5

    fig.update_layout(
        template="plotly_dark",
        height=max(500, len(all_items) * 35 + 150),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_x) - pad, max(all_x) + pad],
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=[min(all_y) - pad, max(all_y) + pad],
            scaleanchor="x", scaleratio=1,
        ),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text="SEM Path Diagram", font=dict(size=16, color="#2E86AB")),
    )

    return fig


# ─── SECTION 1: PATH DIAGRAM ─────────────────────────────────────

def render_path_diagram():
    st.subheader("🗺️ SEM Path Diagram")
    st.markdown(
        "Interactive path diagram showing your complete SEM model. "
        "**Oval nodes** = latent constructs. **Square nodes** = observed indicators. "
        "**Solid arrows** = significant paths. **Dashed arrows** = non-significant. "
        "**Green** = positive significant. **Red** = negative significant. **Gray** = non-significant."
    )

    constructs       = st.session_state.get("constructs", {})
    structural_paths = st.session_state.get("structural_paths", [])
    sem_paths        = st.session_state.get("sem_paths", [])
    cfa_loadings     = st.session_state.get("cfa_loadings", {})
    r2_data          = st.session_state.get("sem_r2", [])

    if not constructs:
        st.warning("⚠️ No constructs defined. Complete Data Input first.")
        return

    # Display options
    col1, col2, col3 = st.columns(3)
    with col1:
        show_loadings = st.checkbox("Show factor loadings", value=True, key="diag_loadings")
    with col2:
        show_r2 = st.checkbox("Show R² values", value=True, key="diag_r2")
    with col3:
        show_indicators = st.checkbox("Show indicators", value=True, key="diag_indicators")

    constructs_to_show = constructs if show_indicators else {k: [] for k in constructs}

    fig = build_path_diagram(
        constructs=constructs_to_show,
        structural_paths=structural_paths,
        sem_paths=sem_paths if sem_paths else [],
        cfa_loadings=cfa_loadings if show_loadings else {},
        r2_data=r2_data if show_r2 else [],
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display:flex;gap:20px;font-size:0.8rem;color:#aaa;padding:8px 0">
        <span><span style="color:#2ecc71">━━</span> Significant positive path</span>
        <span><span style="color:#e74c3c">━━</span> Significant negative path</span>
        <span><span style="color:#888">╌╌</span> Non-significant path</span>
        <span><span style="color:#4a90d9">╌╌</span> Factor loading</span>
    </div>
    """, unsafe_allow_html=True)

    _badge("ok",
        "💡 **Tip:** Hover over nodes and arrows for details. "
        "The path diagram is interactive — zoom, pan, and hover for more information. "
        "Use the export button (camera icon) in the top-right of the chart to save as PNG."
    )


# ─── SECTION 2: FIT DASHBOARD ────────────────────────────────────

def render_fit_dashboard():
    st.subheader("📊 Model Fit Dashboard")

    cfa_fit = st.session_state.get("cfa_fit", {})
    sem_fit = st.session_state.get("sem_fit", {})

    if not cfa_fit and not sem_fit:
        st.info("ℹ️ Run CFA and/or SEM first to see the fit dashboard.")
        return

    tab1, tab2 = st.tabs(["CFA Fit", "SEM Fit"])

    for tab, fit, label in [(tab1, cfa_fit, "CFA"), (tab2, sem_fit, "SEM")]:
        with tab:
            if not fit:
                st.info(f"ℹ️ Run {label} first.")
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
                    status = "✅" if val <= info["good"] else "⚠️" if val <= info["ok"] else "❌"
                    delta_color = "inverse"
                else:
                    status = "✅" if val >= info["good"] else "⚠️" if val >= info["ok"] else "❌"
                    delta_color = "normal"

                col.metric(
                    label=f"{name} {status}",
                    value=f"{val:.3f}",
                    delta=f"criterion: {'≤' if info['direction']=='lower' else '≥'} {info['good']}",
                    delta_color=delta_color,
                )

            # Gauge charts
            gauge_data = [(k, v["val"], v["direction"], v["good"], v["ok"])
                         for k, v in indices.items() if v["val"] is not None]

            if gauge_data:
                fig = go.Figure()
                for i, (name, val, direction, good, ok) in enumerate(gauge_data):
                    if direction == "lower":
                        normalized = max(0, min(1, 1 - val / 0.15))
                    else:
                        normalized = max(0, min(1, val))

                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=val,
                        title={"text": name, "font": {"size": 13}},
                        gauge={
                            "axis": {"range": [0, 1] if direction == "higher" else [0.15, 0],
                                     "tickfont": {"size": 9}},
                            "bar": {"color": "#2ecc71" if normalized >= 0.7 else
                                            "#f39c12" if normalized >= 0.4 else "#e74c3c"},
                            "steps": [
                                {"range": [0, ok] if direction == "higher" else [ok, 0.15],
                                 "color": "#2a0a0a"},
                                {"range": [ok, good] if direction == "higher" else [good, ok],
                                 "color": "#2a1a0a"},
                                {"range": [good, 1] if direction == "higher" else [0, good],
                                 "color": "#0a2a0a"},
                            ],
                        },
                        domain={"row": 0, "column": i},
                        number={"font": {"size": 16}},
                    ))

                fig.update_layout(
                    grid={"rows": 1, "columns": len(gauge_data)},
                    template="plotly_dark",
                    height=220,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="#0d1117",
                )
                st.plotly_chart(fig, use_container_width=True)


# ─── SECTION 3: EFFECT SIZE VISUALIZATION ───────────────────────

def render_effect_sizes():
    st.subheader("📏 Effect Size Visualization")

    sem_paths = st.session_state.get("sem_paths", [])
    if not sem_paths:
        st.info("ℹ️ Run SEM first to see effect sizes.")
        return

    valid_paths = [p for p in sem_paths if p.get("beta") is not None]
    if not valid_paths:
        return

    path_labels = [f"{p['predictor']} → {p['outcome']}" for p in valid_paths]
    betas       = [p["beta"] for p in valid_paths]
    p_vals      = [p.get("p", 1.0) for p in valid_paths]
    colors      = [
        "#2ecc71" if (p < 0.05 and b > 0) else
        "#e74c3c" if (p < 0.05 and b < 0) else "#888"
        for b, p in zip(betas, p_vals)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=betas,
        y=path_labels,
        orientation="h",
        marker_color=colors,
        text=[f"β={b:.3f}" + ("*" if p < 0.05 else "") for b, p in zip(betas, p_vals)],
        textposition="outside",
        hovertext=[
            f"{path_labels[i]}<br>β={betas[i]:.3f}<br>p={p_vals[i]:.4f}"
            for i in range(len(valid_paths))
        ],
        hoverinfo="text",
    ))

    fig.add_vline(x=0, line_color="#888", line_width=1)
    fig.add_vline(x=0.10,  line_dash="dot", line_color="#f39c12", annotation_text="small")
    fig.add_vline(x=-0.10, line_dash="dot", line_color="#f39c12")
    fig.add_vline(x=0.30,  line_dash="dash", line_color="#2ecc71", annotation_text="medium")
    fig.add_vline(x=-0.30, line_dash="dash", line_color="#2ecc71")

    fig.update_layout(
        template="plotly_dark",
        height=max(300, len(valid_paths) * 50 + 100),
        title="Standardized Path Coefficients (β)",
        xaxis_title="Standardized Coefficient (β)",
        xaxis=dict(range=[-1, 1]),
        margin=dict(l=180, r=80, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Note. Green = significant positive; Red = significant negative; Gray = non-significant. * p < .05.")


# ─── SECTION 4: CORRELATION HEATMAP ─────────────────────────────

def render_construct_correlations():
    st.subheader("🔗 Latent Construct Correlations")

    df         = st.session_state.get("df")
    constructs = st.session_state.get("constructs", {})

    if df is None or not constructs:
        st.info("ℹ️ Complete Data Input and run CFA first.")
        return

    # Parcel scores
    parcel_df = pd.DataFrame()
    for cname, items in constructs.items():
        valid = [c for c in items if c in df.columns]
        if valid:
            parcel_df[cname] = df[valid].mean(axis=1)

    parcel_df = parcel_df.dropna()
    if parcel_df.shape[1] < 2:
        st.warning("⚠️ Not enough constructs with data.")
        return

    corr = parcel_df.corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        template="plotly_dark",
        text_auto=".3f",
        title="Inter-Construct Correlation Matrix",
        aspect="auto",
    )
    fig.update_layout(height=max(350, len(constructs) * 60))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Note. Correlations based on construct parcel scores (mean of indicators). "
        "Values > .85 may indicate discriminant validity concerns."
    )


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_visualization():
    st.title("🗺️ Path Diagram & Visualizations")
    st.markdown(
        "Visual representations of your SEM model results. "
        "All charts are **interactive** — hover for details, zoom, and pan. "
        "Use the camera icon (🔗) in any chart to export as PNG."
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Path Diagram",
        "📊 Fit Dashboard",
        "📏 Effect Sizes",
        "🔗 Correlations",
    ])

    with tab1:
        render_path_diagram()

    with tab2:
        render_fit_dashboard()

    with tab3:
        render_effect_sizes()

    with tab4:
        render_construct_correlations()
