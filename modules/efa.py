"""
efa.py
======
Sprint 2 — Exploratory Factor Analysis (EFA) Module for SEM Studio.
Uses numpy/scipy only — no factor_analyzer dependency.

References:
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
    - Brown (2015). CFA for Applied Research (2nd ed.)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.linalg import svd

from utils.thresholds import EFA as EFA_THRESH, CFA as CFA_THRESH
from utils.interpretation import interpret_factor_loading

LEVEL_COLOR = {
    "excellent": "#2ecc71", "good": "#27ae60",
    "ok": "#3498db", "warning": "#f39c12", "critical": "#e74c3c",
}

def _badge(level, message):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>', unsafe_allow_html=True
    )


# ── Numpy-based EFA utilities ────────────────────────────────────

def compute_kmo(data: pd.DataFrame):
    """Compute KMO measure of sampling adequacy."""
    X = data.values
    corr = np.corrcoef(X.T)
    np.fill_diagonal(corr, 1)
    try:
        corr_inv = np.linalg.inv(corr)
    except np.linalg.LinAlgError:
        return np.array([0.5] * data.shape[1]), 0.5

    # Partial correlations
    d = np.diag(corr_inv)
    pcorr = -corr_inv / np.sqrt(np.outer(d, d))
    np.fill_diagonal(pcorr, 0)
    np.fill_diagonal(corr, 0)

    corr2  = corr  ** 2
    pcorr2 = pcorr ** 2

    kmo_per = corr2.sum(axis=0) / (corr2.sum(axis=0) + pcorr2.sum(axis=0) + 1e-10)
    kmo_total = corr2.sum() / (corr2.sum() + pcorr2.sum() + 1e-10)
    np.fill_diagonal(corr, 1)
    return kmo_per, float(kmo_total)


def compute_bartlett(data: pd.DataFrame):
    """Compute Bartlett's Test of Sphericity."""
    n, p = data.shape
    corr = data.corr().values
    det  = np.linalg.det(corr)
    det  = max(det, 1e-300)
    chi2 = -(n - 1 - (2*p + 5)/6) * np.log(det)
    df   = p * (p - 1) / 2
    p_val = 1 - stats.chi2.cdf(chi2, df)
    return float(chi2), float(p_val)


def paf_extraction(corr: np.ndarray, n_factors: int, max_iter: int = 100) -> np.ndarray:
    """Principal Axis Factoring (PAF) extraction. Always returns exactly n_factors columns."""
    p = corr.shape[0]
    n_factors = min(n_factors, p - 1)  # safety cap

    try:
        communalities = 1 - 1 / np.diag(np.linalg.inv(corr + 1e-6 * np.eye(p)))
    except Exception:
        communalities = np.full(p, 0.5)
    communalities = np.clip(communalities, 0.1, 0.99)

    loadings = np.zeros((p, n_factors))  # default fallback

    for _ in range(max_iter):
        R = corr.copy()
        np.fill_diagonal(R, communalities)
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take only positive eigenvalues, cap at n_factors
        n_pos = int((eigenvalues > 0).sum())
        n_use = min(n_factors, n_pos)
        if n_use == 0:
            break

        ev_use  = eigenvalues[:n_use]
        vec_use = eigenvectors[:, :n_use]
        L       = vec_use * np.sqrt(np.abs(ev_use))

        # Pad to exactly n_factors columns
        if L.shape[1] < n_factors:
            pad = np.zeros((p, n_factors - L.shape[1]))
            L = np.hstack([L, pad])
        loadings = L

        new_comm = np.clip(np.sum(loadings ** 2, axis=1), 0.1, 0.99)
        if np.max(np.abs(new_comm - communalities)) < 1e-6:
            break
        communalities = new_comm

    # Final guarantee: exactly n_factors columns
    if loadings.shape[1] != n_factors:
        new_L = np.zeros((p, n_factors))
        cols  = min(loadings.shape[1], n_factors)
        new_L[:, :cols] = loadings[:, :cols]
        loadings = new_L

    return loadings


def oblimin_rotation(loadings: np.ndarray, gamma: float = 0, max_iter: int = 1000) -> np.ndarray:
    """Simplified oblimin rotation using gradient algorithm."""
    p, k = loadings.shape
    if k < 2:
        return loadings

    T = np.eye(k)
    L = loadings @ T

    for _ in range(max_iter):
        L2 = L ** 2
        # Gradient
        u = np.ones((p, p)) / p
        Cmat = (np.eye(p) - gamma * u) @ L2
        grad = loadings.T @ (L * Cmat) - (loadings.T @ L * (L * Cmat).sum(axis=0))
        # Update
        T_new = T - 0.01 * grad.T
        # Re-orthogonalize columns
        for j in range(k):
            T_new[:, j] /= (np.linalg.norm(T_new[:, j]) + 1e-10)
        if np.max(np.abs(T_new - T)) < 1e-6:
            break
        T = T_new
        L = loadings @ T

    return L


def varimax_rotation(loadings: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """Varimax rotation."""
    p, k = loadings.shape
    if k < 2:
        return loadings

    T = np.eye(k)
    for _ in range(max_iter):
        L = loadings @ T
        u, s, vt = svd(loadings.T @ (L**3 - L @ np.diag(np.sum(L**2, axis=0)) / p))
        T_new = u @ vt
        if np.max(np.abs(T_new - T)) < 1e-10:
            break
        T = T_new

    return loadings @ T


def run_efa(data: pd.DataFrame, n_factors: int, rotation: str) -> dict:
    """Run full EFA. Returns dict with loadings, communalities, variance."""
    corr = data.corr().values
    p    = corr.shape[0]
    n_factors = min(n_factors, p - 1)

    # Extraction (PAF)
    loadings = paf_extraction(corr, n_factors)

    # Rotation
    if rotation == "varimax" and n_factors >= 2:
        loadings = varimax_rotation(loadings)
    elif rotation == "oblimin" and n_factors >= 2:
        loadings = oblimin_rotation(loadings)

    communalities = np.sum(loadings ** 2, axis=1)
    ss_loadings   = np.sum(loadings ** 2, axis=0)
    total_var     = p
    prop_var      = ss_loadings / total_var
    cum_var       = np.cumsum(prop_var)

    return {
        "loadings":      loadings,
        "communalities": communalities,
        "ss_loadings":   ss_loadings,
        "prop_var":      prop_var,
        "cum_var":       cum_var,
    }


# ── SECTION 1: FACTORABILITY ────────────────────────────────────

def render_factorability(data: pd.DataFrame):
    st.subheader("🔬 Step 1: Factorability Tests")
    st.markdown(
        "Before extracting factors, we confirm the data are **suitable for factor analysis** "
        "using the **KMO** measure and **Bartlett's Test of Sphericity**."
    )

    try:
        kmo_all, kmo_model = compute_kmo(data)
        chi2, p            = compute_bartlett(data)

        c1, c2, c3 = st.columns(3)
        c1.metric("KMO Overall",       f"{kmo_model:.3f}")
        c2.metric("Bartlett's χ²",     f"{chi2:.3f}")
        c3.metric("Bartlett's p-value", "< .0001" if p < 0.0001 else f"{p:.4f}")

        from utils.interpretation import interpret_kmo, interpret_bartlett
        _badge(**interpret_kmo(kmo_model))
        _badge(**interpret_bartlett(p))

        with st.expander("🔍 KMO Per Item (MSA)"):
            msa_df = pd.DataFrame({
                "Item":   list(data.columns),
                "MSA":    [round(v, 3) for v in kmo_all],
                "Status": ["✅" if v >= 0.60 else ("⚠️" if v >= 0.50 else "❌")
                           for v in kmo_all],
            }).sort_values("MSA")
            st.dataframe(msa_df, use_container_width=True, hide_index=True)

        st.session_state["efa_factorable"] = kmo_model >= 0.60 and p < 0.05
        return kmo_model >= 0.60 and p < 0.05

    except Exception as e:
        st.error(f"❌ Error computing factorability tests: {str(e)}")
        return False


# ── SECTION 2: NUMBER OF FACTORS ────────────────────────────────

def render_factor_number(data: pd.DataFrame) -> int:
    st.subheader("📊 Step 2: Determining Number of Factors")
    st.markdown(
        "Three criteria: **(1) Kaiser criterion** (eigenvalue > 1), "
        "**(2) Scree plot**, and **(3) Parallel Analysis** (gold standard)."
    )

    try:
        corr_matrix = data.corr().values
        ev = np.linalg.eigvalsh(corr_matrix)[::-1]
        ev = ev[ev > 0]

        kaiser_n = int((ev > 1).sum())
        n_items  = len(ev)
        n_obs    = data.shape[0]

        # Parallel analysis via random correlation matrices
        rng     = np.random.default_rng(42)
        sim_evs = np.zeros((100, n_items))
        for i in range(100):
            sim_data = rng.standard_normal((n_obs, n_items))
            sim_corr = np.corrcoef(sim_data.T)
            sim_ev   = np.linalg.eigvalsh(sim_corr)[::-1]
            sim_evs[i] = sim_ev[:n_items]

        pa_95 = np.percentile(sim_evs, 95, axis=0)
        pa_n  = int((ev > pa_95[:n_items]).sum())
        pa_n  = max(1, pa_n)

        # Scree plot
        x = list(range(1, n_items + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=ev, mode="lines+markers",
            name="Actual Eigenvalues",
            line=dict(color="#2E86AB", width=2), marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=x, y=pa_95[:n_items], mode="lines+markers",
            name="Parallel Analysis (95th %ile)",
            line=dict(color="#e74c3c", width=2, dash="dash"),
            marker=dict(size=5, symbol="diamond")
        ))
        fig.add_hline(y=1, line_dash="dot", line_color="#f39c12",
                      annotation_text="Eigenvalue = 1 (Kaiser)")
        fig.update_layout(
            template="plotly_dark", height=380,
            title="Scree Plot with Parallel Analysis",
            xaxis_title="Factor Number", yaxis_title="Eigenvalue",
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Kaiser Criterion",  f"{kaiser_n} factor(s)")
        c2.metric("Parallel Analysis", f"{pa_n} factor(s)")
        c3.metric("Recommended",       f"{pa_n} factor(s)")

        _badge("ok",
            f"**Kaiser** suggests **{kaiser_n}** factor(s). "
            f"**Parallel analysis** (gold standard) suggests **{pa_n}** factor(s). "
            "Always consider theoretical justification alongside statistical criteria."
        )

        n_factors = st.number_input(
            "Number of factors to extract:",
            min_value=1, max_value=min(n_items - 1, 10),
            value=pa_n, step=1, key="efa_n_factors_widget"
        )
        return int(n_factors)

    except Exception as e:
        st.error(f"❌ Error in factor number determination: {str(e)}")
        return 1


# ── SECTION 3: EXTRACTION & ROTATION ────────────────────────────

def render_factor_extraction(data: pd.DataFrame, n_factors: int):
    st.subheader("⚙️ Step 3: Factor Extraction & Rotation")

    col1, col2 = st.columns(2)
    with col1:
        rotation = st.selectbox(
            "Rotation Method",
            options=["oblimin", "varimax", "none"],
            format_func=lambda x: {
                "oblimin": "Oblimin (oblique — recommended)",
                "varimax": "Varimax (orthogonal)",
                "none":    "No rotation",
            }[x],
            key="efa_rotation_widget"
        )
    with col2:
        st.markdown("**Extraction:** Principal Axis Factoring (PAF)")
        st.markdown("*(recommended for SEM/CFA preparation)*")

    _badge("ok",
        "**Recommendation:** PAF with **oblimin rotation** is preferred for social science — "
        "it allows factors to correlate, which is realistic."
    )

    if st.button("▶️ Run EFA", type="primary", key="run_efa"):
        with st.spinner("Running EFA..."):
            try:
                result = run_efa(data, n_factors, rotation)
                st.session_state["efa_result"]      = result
                st.session_state["efa_n_factors_val"] = n_factors
                st.session_state["efa_rotation"]      = rotation
                st.session_state["efa_data_cols"]     = list(data.columns)
                st.success(f"✅ EFA complete: {n_factors} factor(s), PAF, {rotation} rotation.")
            except Exception as e:
                st.error(f"❌ EFA failed: {str(e)}")
                return None

    return st.session_state.get("efa_result")


# ── SECTION 4: FACTOR LOADINGS ───────────────────────────────────

def render_loadings(result: dict, item_names: list, n_factors: int):
    st.subheader("📋 Step 4: Factor Loading Matrix")

    loadings      = result["loadings"]
    communalities = result["communalities"]
    prop_var      = result["prop_var"]
    cum_var       = result["cum_var"]
    cols          = [f"F{i+1}" for i in range(n_factors)]

    loadings_df = pd.DataFrame(
        loadings[:len(item_names)],
        index=item_names, columns=cols
    )
    loadings_df["h²"] = communalities[:len(item_names)].round(3)

    def color_loading(val):
        try:
            v = float(val)
            if abs(v) >= 0.70: return "color:#2ecc71;font-weight:bold"
            elif abs(v) >= 0.50: return "color:#f39c12"
            elif abs(v) >= 0.32: return "color:#e67e22"
            else: return "color:#888"
        except: return ""

    st.dataframe(
        loadings_df.round(3).style.map(color_loading),
        use_container_width=True
    )

    # Variance explained
    var_df = pd.DataFrame({
        "Factor":         cols,
        "SS Loadings":    result["ss_loadings"].round(3),
        "Proportion Var": prop_var.round(3),
        "Cumulative Var": cum_var.round(3),
    })
    st.markdown("**Variance Explained:**")
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    total_var = cum_var[-1]
    if total_var >= EFA_THRESH["variance_explained_min"]:
        _badge("ok", f"Total variance explained: **{total_var:.1%}** ✅ (criterion: ≥ 50%)")
    else:
        _badge("warning", f"Total variance explained: **{total_var:.1%}** ⚠️ (criterion: ≥ 50%)")

    # Cross-loading check
    cross = []
    for item in item_names:
        row = loadings_df.loc[item, cols].abs()
        if (row >= EFA_THRESH["cross_loading_max"]).sum() > 1:
            cross.append(item)
    if cross:
        _badge("warning", f"⚠️ Cross-loadings detected: **{', '.join(cross)}**. Consider removing these items.")
    else:
        _badge("excellent", "✅ No cross-loadings detected.")

    # Low communalities
    low = [item_names[i] for i, h in enumerate(communalities[:len(item_names)]) if h < 0.40]
    if low:
        _badge("warning", f"⚠️ Low communality (h² < .40): **{', '.join(low)}**")

    # Heatmap
    with st.expander("🎨 Factor Loading Heatmap"):
        fig = px.imshow(
            loadings_df[cols].T, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, template="plotly_dark",
            text_auto=".2f", title="Factor Loading Heatmap", aspect="auto",
        )
        fig.update_layout(height=max(200, n_factors * 60))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔍 Item-by-Item Interpretation"):
        for item in item_names:
            best_f   = loadings_df.loc[item, cols].abs().idxmax()
            best_val = loadings_df.loc[item, best_f]
            r = interpret_factor_loading(best_val, item)
            _badge(r["level"], f"{r['message']} → Primary: **{best_f}**")

    return loadings_df


# ── SECTION 5: FACTOR NAMING ─────────────────────────────────────

def render_factor_naming(loadings_df: pd.DataFrame, n_factors: int) -> dict:
    st.subheader("🏷️ Step 5: Factor Naming")

    cols = [f"F{i+1}" for i in range(n_factors)]
    factor_names = {}

    for f in cols:
        top_items = loadings_df[f].abs().nlargest(5).index.tolist()
        c1, c2 = st.columns([1, 2])
        with c1:
            name = st.text_input(f"Name for {f}", value=f"Factor_{f[1:]}", key=f"fname_{f}")
            factor_names[f] = name
        with c2:
            st.markdown(f"**Top items on {f}:**")
            for item in top_items:
                st.markdown(f"  - {item}: λ = `{loadings_df.loc[item, f]:.3f}`")

    st.session_state["efa_factor_names"] = factor_names
    return factor_names


# ── SECTION 6: EFA SUMMARY ───────────────────────────────────────

def render_efa_summary(result: dict, item_names: list, n_factors: int,
                       factor_names: dict, loadings_df: pd.DataFrame):
    st.subheader("📄 Step 6: EFA Summary & CFA Preparation")

    cols = [f"F{i+1}" for i in range(n_factors)]
    suggested = {}

    for f in cols:
        fname = factor_names.get(f, f)
        items = [
            item for item in item_names
            if loadings_df.loc[item, cols].abs().idxmax() == f
            and abs(loadings_df.loc[item, f]) >= EFA_THRESH["factor_loading_acceptable"]
        ]
        suggested[fname] = items
        n = len(items)
        st.markdown(f"**{fname}** ({n} items): {', '.join(items)}")
        if n < 3:
            st.warning(f"⚠️ **{fname}** has only {n} item(s). CFA requires ≥ 3.")

    st.session_state["efa_suggested_constructs"] = suggested

    if st.button("📐 Use EFA Results for CFA Model", type="primary", key="efa_to_cfa"):
        st.session_state["constructs"] = suggested
        st.success("✅ EFA structure transferred to CFA. Navigate to **CFA** in the sidebar.")

    _badge("ok",
        "💡 **Note:** EFA is data-driven. Ensure the factor structure is "
        "**theoretically justified** before proceeding to CFA."
    )


# ── MAIN RENDER ──────────────────────────────────────────────────

def render_efa():
    st.title("🔍 Exploratory Factor Analysis (EFA)")
    st.markdown(
        "EFA explores how items cluster empirically when the factor structure is unknown. "
        "Results guide the subsequent CFA.\n\n"
        "> 💡 *If your instrument is already validated, you may skip EFA and go directly to CFA.*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    df             = st.session_state["df"]
    assignments    = st.session_state.get("assignments", {})
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if len(indicator_cols) < 3:
        st.error("❌ At least 3 indicator variables required for EFA.")
        return

    data = df[indicator_cols].dropna()
    st.info(f"📊 Analyzing **{len(indicator_cols)} items** across **{len(data)} complete cases**.")
    st.markdown("---")

    factorable = render_factorability(data)
    if not factorable:
        st.error("❌ Data not suitable for factor analysis. Address issues above.")
        return
    st.markdown("---")

    n_factors = render_factor_number(data)
    st.markdown("---")

    result = render_factor_extraction(data, n_factors)
    if result is None:
        return
    st.markdown("---")

    loadings_df = render_loadings(result, indicator_cols, n_factors)
    st.markdown("---")

    factor_names = render_factor_naming(loadings_df, n_factors)
    st.markdown("---")

    render_efa_summary(result, indicator_cols, n_factors, factor_names, loadings_df)

    st.session_state["efa_complete"] = True
    st.markdown("---")
    st.success("✅ EFA complete. Proceed to **Confirmatory Factor Analysis (CFA)**.")
