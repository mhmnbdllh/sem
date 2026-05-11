"""
efa.py
======
Sprint 2 — Exploratory Factor Analysis (EFA) Module for SEM Studio.

Covers:
- KMO & Bartlett's Test of Sphericity
- Parallel Analysis for factor number determination
- Factor extraction (PAF, ML, PCA)
- Rotation methods (Oblimin, Varimax, Promax)
- Factor loading matrix with cross-loading detection
- Communalities
- Total variance explained
- Scree plot
- Full auto-interpretation per output

References:
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
    - Brown (2015). CFA for Applied Research (2nd ed.)
    - Fabrigar et al. (1999). Evaluating the use of EFA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity

from utils.thresholds import EFA as EFA_THRESH, CFA as CFA_THRESH
from utils.interpretation import interpret_kmo, interpret_bartlett, interpret_factor_loading
from utils.apa_tables import style_df

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


# ─── SECTION 1: FACTORABILITY ───────────────────────────────────

def render_factorability(data: pd.DataFrame):
    st.subheader("🔬 Step 1: Factorability Tests")
    st.markdown(
        "Before extracting factors, we must confirm the data are **suitable for factor analysis** "
        "using the **Kaiser-Meyer-Olkin (KMO)** measure and **Bartlett's Test of Sphericity**."
    )

    try:
        kmo_all, kmo_model = calculate_kmo(data)
        chi2, p = calculate_bartlett_sphericity(data)

        c1, c2, c3 = st.columns(3)
        c1.metric("KMO Overall", f"{kmo_model:.3f}")
        c2.metric("Bartlett's χ²", f"{chi2:.3f}")
        c3.metric("Bartlett's p-value", f"{p:.4f}" if p >= 0.0001 else "< .0001")

        kmo_result = interpret_kmo(kmo_model)
        bart_result = interpret_bartlett(p)
        _badge(kmo_result["level"], kmo_result["message"])
        _badge(bart_result["level"], bart_result["message"])

        # Per-item KMO
        with st.expander("🔍 KMO Per Item (MSA — Measures of Sampling Adequacy)"):
            st.markdown("Items with MSA < .50 should be considered for removal.")
            msa_df = pd.DataFrame({
                "Item": data.columns,
                "MSA": kmo_all,
                "Status": ["✅" if v >= 0.60 else ("⚠️" if v >= 0.50 else "❌") for v in kmo_all]
            }).sort_values("MSA")
            st.dataframe(msa_df.style.background_gradient(subset=["MSA"], cmap="RdYlGn"),
                        use_container_width=True)

            low_msa = [data.columns[i] for i, v in enumerate(kmo_all) if v < 0.50]
            if low_msa:
                _badge("warning", f"Items with inadequate MSA (< .50): **{', '.join(low_msa)}**. Consider removing them.")

        st.session_state["efa_factorable"] = kmo_model >= 0.60 and p < 0.05
        return kmo_model >= 0.60 and p < 0.05

    except Exception as e:
        st.error(f"❌ Error computing factorability tests: {str(e)}")
        return False


# ─── SECTION 2: NUMBER OF FACTORS ───────────────────────────────

def render_factor_number(data: pd.DataFrame) -> int:
    st.subheader("📊 Step 2: Determining Number of Factors")
    st.markdown(
        "Three criteria are used: **(1) Kaiser criterion** (eigenvalue > 1), "
        "**(2) Scree plot** (elbow point), and **(3) Parallel Analysis** (gold standard). "
        "Convergence across criteria provides stronger justification."
    )

    try:
        fa = FactorAnalyzer(n_factors=min(data.shape[1], data.shape[0]-1), rotation=None)
        fa.fit(data)
        ev, _ = fa.get_eigenvalues()

        # Kaiser criterion
        kaiser_n = int((ev > 1).sum())

        # Scree plot with parallel analysis
        n_items = data.shape[1]
        n_obs   = data.shape[0]
        n_iter  = 100

        # Parallel analysis: simulate random data eigenvalues
        sim_evs = np.zeros((n_iter, n_items))
        rng = np.random.default_rng(42)
        for i in range(n_iter):
            sim_data = rng.standard_normal((n_obs, n_items))
            sim_corr = np.corrcoef(sim_data.T)
            sim_evs[i] = np.linalg.eigvalsh(sim_corr)[::-1]

        pa_95 = np.percentile(sim_evs, 95, axis=0)[:n_items]
        pa_n  = int((ev > pa_95).sum())

        # Plot
        x = list(range(1, n_items + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=ev[:n_items], mode="lines+markers",
                                 name="Actual Eigenvalues", line=dict(color="#2E86AB", width=2),
                                 marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=x, y=pa_95, mode="lines+markers",
                                 name="Parallel Analysis (95th %ile)", line=dict(color="#e74c3c", width=2, dash="dash"),
                                 marker=dict(size=5, symbol="diamond")))
        fig.add_hline(y=1, line_dash="dot", line_color="#f39c12", annotation_text="Eigenvalue = 1 (Kaiser)")
        fig.update_layout(
            template="plotly_dark", height=380,
            title="Scree Plot with Parallel Analysis",
            xaxis_title="Factor Number", yaxis_title="Eigenvalue",
            legend=dict(x=0.6, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary
        c1, c2, c3 = st.columns(3)
        c1.metric("Kaiser Criterion", f"{kaiser_n} factor(s)")
        c2.metric("Parallel Analysis", f"{pa_n} factor(s)")
        c3.metric("Recommended", f"{pa_n} factor(s)", help="Parallel analysis is preferred over Kaiser criterion")

        _badge("ok",
            f"**Kaiser criterion** suggests **{kaiser_n}** factor(s) (eigenvalue > 1). "
            f"**Parallel analysis** (gold standard) suggests **{pa_n}** factor(s). "
            f"The final number of factors should also be guided by **theoretical considerations**."
        )

        # User override
        st.markdown("**Select number of factors to extract:**")
        n_factors = st.number_input(
            "Number of factors", min_value=1,
            max_value=min(n_items - 1, 10),
            value=pa_n, step=1, key="efa_n_factors"
        )

        return int(n_factors)

    except Exception as e:
        st.error(f"❌ Error in factor number determination: {str(e)}")
        return 1


# ─── SECTION 3: FACTOR EXTRACTION & ROTATION ────────────────────

def render_factor_extraction(data: pd.DataFrame, n_factors: int) -> FactorAnalyzer | None:
    st.subheader("⚙️ Step 3: Factor Extraction & Rotation")

    col1, col2 = st.columns(2)
    with col1:
        method = st.selectbox(
            "Extraction Method",
            options=["minres", "ml", "principal"],
            format_func=lambda x: {
                "minres": "Minimum Residual (PAF — recommended)",
                "ml":     "Maximum Likelihood",
                "principal": "Principal Components (PCA)",
            }[x],
            help="PAF (minres) is recommended for SEM/CFA preparation.",
            key="efa_method"
        )
    with col2:
        rotation = st.selectbox(
            "Rotation Method",
            options=["oblimin", "varimax", "promax", "None"],
            format_func=lambda x: {
                "oblimin": "Oblimin (oblique — recommended if constructs correlated)",
                "varimax": "Varimax (orthogonal — assumes uncorrelated factors)",
                "promax":  "Promax (oblique — alternative)",
                "None":    "No rotation",
            }[x],
            help="Oblique rotation (oblimin/promax) is generally preferred in social science.",
            key="efa_rotation"
        )

    rotation_arg = None if rotation == "None" else rotation

    _badge("ok",
        "**Recommendation:** Use **PAF (minres)** with **oblimin rotation** for SEM preparation. "
        "Oblique rotation allows factors to correlate, which is realistic in social/behavioral research."
    )

    if st.button("▶️ Run EFA", type="primary", key="run_efa"):
        with st.spinner("Running Exploratory Factor Analysis..."):
            try:
                fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation_arg, method=method)
                fa.fit(data)
                st.session_state["efa_model"]    = fa
                st.session_state["efa_n_factors"] = n_factors
                st.session_state["efa_rotation"]  = rotation
                st.session_state["efa_data_cols"] = list(data.columns)
                st.success(f"✅ EFA completed: {n_factors} factor(s), {method} extraction, {rotation} rotation.")
            except Exception as e:
                st.error(f"❌ EFA failed: {str(e)}")
                return None

    return st.session_state.get("efa_model")


# ─── SECTION 4: FACTOR LOADINGS ─────────────────────────────────

def render_loadings(fa: FactorAnalyzer, item_names: list, n_factors: int):
    st.subheader("📋 Step 4: Factor Loading Matrix")
    st.markdown(
        "Factor loadings represent the correlation between each item and its factor. "
        f"**Criterion:** |λ| ≥ **{EFA_THRESH['factor_loading_good']}** (good); "
        f"≥ **{EFA_THRESH['factor_loading_acceptable']}** (acceptable). "
        f"Cross-loadings > **{EFA_THRESH['cross_loading_max']}** are flagged."
    )

    loadings = fa.loadings_
    cols = [f"F{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(loadings[:len(item_names)], index=item_names, columns=cols)

    # Communalities
    communalities = fa.get_communalities()
    loadings_df["h²"] = communalities[:len(item_names)]

    # Color-code
    def color_loading(val):
        try:
            v = float(val)
            if abs(v) >= EFA_THRESH["factor_loading_good"]:    return "color:#2ecc71;font-weight:bold"
            elif abs(v) >= EFA_THRESH["factor_loading_acceptable"]: return "color:#f39c12"
            elif abs(v) >= EFA_THRESH["cross_loading_max"]:    return "color:#e67e22"
            else:                                               return "color:#888"
        except: return ""

    styled = loadings_df.style.map(color_loading)
    st.dataframe(styled, use_container_width=True)

    # Variance explained
    var = fa.get_factor_variance()
    var_df = pd.DataFrame({
        "Factor": cols,
        "SS Loadings":      np.round(var[0], 3),
        "Proportion Var":   np.round(var[1], 3),
        "Cumulative Var":   np.round(var[2], 3),
    })
    st.markdown("**Variance Explained:**")
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    total_var = var[2][-1]
    if total_var >= EFA_THRESH["variance_explained_min"]:
        _badge("ok", f"Total variance explained: **{total_var:.1%}** ✅ (criterion: ≥ 50%). The factor solution accounts for sufficient variance.")
    else:
        _badge("warning", f"Total variance explained: **{total_var:.1%}** ⚠️ (criterion: ≥ 50%). Consider adding more factors or revising items.")

    # Cross-loading detection
    st.markdown("**Cross-Loading Check:**")
    cross_loadings = []
    for item in item_names:
        row = loadings_df.loc[item, cols].abs()
        above = row[row >= EFA_THRESH["cross_loading_max"]]
        if len(above) > 1:
            cross_loadings.append({
                "Item": item,
                "Factors with |λ| > .32": ", ".join([f"{f}={loadings_df.loc[item, f]:.3f}" for f in above.index])
            })

    if cross_loadings:
        st.warning("⚠️ **Cross-loadings detected** (item loads on multiple factors above threshold):")
        st.dataframe(pd.DataFrame(cross_loadings), use_container_width=True, hide_index=True)
        _badge("warning",
            "Cross-loading items may lack discriminant validity. "
            "Consider dropping these items or rewriting them to be more construct-specific."
        )
    else:
        _badge("excellent", "✅ No cross-loadings detected. Each item loads clearly on one factor.")

    # Low communalities
    low_comm = [(item_names[i], communalities[i]) for i in range(len(item_names))
                if communalities[i] < EFA_THRESH["communality_min"]]
    if low_comm:
        st.warning(f"⚠️ Items with low communality (h² < .40): " +
                   ", ".join([f"**{i}** (h²={v:.3f})" for i, v in low_comm]))

    # Heatmap
    with st.expander("🎨 Factor Loading Heatmap"):
        fig = px.imshow(
            loadings_df[cols].T,
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            template="plotly_dark", text_auto=".2f",
            title="Factor Loading Heatmap", aspect="auto",
        )
        fig.update_layout(height=max(200, n_factors * 60))
        st.plotly_chart(fig, use_container_width=True)

    # Item-level interpretation
    with st.expander("🔍 Item-by-Item Loading Interpretation"):
        for item in item_names:
            best_factor = loadings_df.loc[item, cols].abs().idxmax()
            best_loading = loadings_df.loc[item, best_factor]
            result = interpret_factor_loading(best_loading, item)
            _badge(result["level"], f"{result['message']} → Primary factor: **{best_factor}**")

    return loadings_df


# ─── SECTION 5: FACTOR NAMING ────────────────────────────────────

def render_factor_naming(loadings_df: pd.DataFrame, n_factors: int) -> dict:
    st.subheader("🏷️ Step 5: Factor Naming & Structure Confirmation")
    st.markdown(
        "Based on the items loading on each factor, assign a **theoretical name** to each factor. "
        "This step bridges EFA (data-driven) and CFA (theory-driven)."
    )

    factor_names = {}
    cols = [f"F{i+1}" for i in range(n_factors)]

    for f in cols:
        top_items = loadings_df[f].abs().nlargest(5).index.tolist()
        c1, c2 = st.columns([1, 2])
        with c1:
            name = st.text_input(f"Name for {f}", value=f"Factor_{f[1:]}", key=f"fname_{f}")
            factor_names[f] = name
        with c2:
            st.markdown(f"**Top items loading on {f}:**")
            for item in top_items:
                val = loadings_df.loc[item, f]
                st.markdown(f"  - {item}: λ = `{val:.3f}`")

    st.session_state["efa_factor_names"] = factor_names

    _badge("ok",
        "✅ Factor names defined. Proceed to **CFA** to confirm this factor structure "
        "using a theory-driven confirmatory approach."
    )

    return factor_names


# ─── SECTION 6: EFA SUMMARY ─────────────────────────────────────

def render_efa_summary(fa: FactorAnalyzer, item_names: list, n_factors: int, factor_names: dict):
    st.subheader("📄 EFA Summary & CFA Preparation")

    var = fa.get_factor_variance()
    loadings = fa.loadings_
    cols = [f"F{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(loadings[:len(item_names)], index=item_names, columns=cols)

    st.markdown("**Suggested CFA Model based on EFA results:**")
    st.markdown("Each factor below becomes a **latent construct** in your CFA model.")

    suggested_constructs = {}
    for f in cols:
        fname = factor_names.get(f, f)
        # Items with primary loading ≥ .40 on this factor
        primary_items = [item for item in item_names
                        if loadings_df.loc[item, cols].abs().idxmax() == f
                        and abs(loadings_df.loc[item, f]) >= EFA_THRESH["factor_loading_acceptable"]]
        suggested_constructs[fname] = primary_items

        st.markdown(f"**{fname}** ({len(primary_items)} items): {', '.join(primary_items)}")
        if len(primary_items) < 3:
            st.warning(f"⚠️ **{fname}** has only {len(primary_items)} item(s) with adequate loading. CFA requires ≥ 3 indicators.")

    st.session_state["efa_suggested_constructs"] = suggested_constructs

    # Option to use EFA results in CFA
    if st.button("📐 Use EFA Results for CFA Model", type="primary", key="efa_to_cfa"):
        # Merge into existing constructs or replace
        st.session_state["constructs"] = suggested_constructs
        st.success("✅ EFA factor structure transferred to CFA. Navigate to **CFA** in the sidebar.")

    _badge("ok",
        "💡 **Note:** EFA is data-driven. Before running CFA, ensure the suggested factor structure "
        "is **theoretically justified**. Do not proceed to CFA solely based on EFA statistics."
    )


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_efa():
    st.title("🔍 Exploratory Factor Analysis (EFA)")
    st.markdown(
        "EFA is used when the **factor structure is unknown or unconfirmed**. "
        "It explores how items cluster together empirically, providing the foundation "
        "for a subsequent **Confirmatory Factor Analysis (CFA)**.\n\n"
        "> 💡 *If your instrument has been validated in prior research, you may skip EFA and proceed directly to CFA.*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    df          = st.session_state["df"]
    assignments = st.session_state.get("assignments", {})
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if len(indicator_cols) < 3:
        st.error("❌ At least 3 indicator variables are required for EFA.")
        return

    data = df[indicator_cols].dropna()

    st.info(f"📊 Analyzing **{len(indicator_cols)} items** across **{len(data)} complete cases**.")
    st.markdown("---")

    # Step 1
    factorable = render_factorability(data)
    if not factorable:
        st.error("❌ Data are not suitable for factor analysis. Address the issues above before proceeding.")
        return

    st.markdown("---")

    # Step 2
    n_factors = render_factor_number(data)
    st.markdown("---")

    # Step 3
    fa = render_factor_extraction(data, n_factors)
    if fa is None:
        return

    st.markdown("---")

    # Step 4
    loadings_df = render_loadings(fa, indicator_cols, n_factors)
    st.markdown("---")

    # Step 5
    factor_names = render_factor_naming(loadings_df, n_factors)
    st.markdown("---")

    # Step 6
    render_efa_summary(fa, indicator_cols, n_factors, factor_names)

    st.session_state["efa_complete"] = True
    st.markdown("---")
    st.success("✅ EFA complete. Proceed to **Confirmatory Factor Analysis (CFA)**.")
