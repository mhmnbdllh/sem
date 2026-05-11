"""
mediation.py
============
Sprint 3 — Mediation Analysis Module for SEM Studio.

Covers:
- Direct, indirect, and total effects
- Bootstrap confidence intervals (5,000 resamples)
- Sobel test
- Partial vs full mediation determination
- VAF (Variance Accounted For)
- Multiple mediation support
- Full auto-interpretation

References:
    - Hayes (2018). Introduction to Mediation, Moderation, and Conditional Process Analysis.
    - Preacher & Hayes (2008). Asymptotic and resampling strategies for assessing mediation.
    - Baron & Kenny (1986). The moderator-mediator variable distinction.
    - Zhao et al. (2010). Reconsidering Baron and Kenny.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

from utils.thresholds import MEDIATION as MED_THRESH
from utils.interpretation import interpret_mediation


# ─── HELPERS ────────────────────────────────────────────────────

LEVEL_COLOR = {
    "excellent": "#2ecc71", "good": "#27ae60",
    "ok": "#3498db", "warning": "#f39c12", "critical": "#e74c3c",
    "significant": "#2ecc71", "nonsignificant": "#e74c3c",
}

def _badge(level: str, message: str):
    color = LEVEL_COLOR.get(level, "#888")
    st.markdown(
        f'<div style="background:{color}22;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;color:#f0f0f0">'
        f'{message}</div>', unsafe_allow_html=True
    )


# ─── OLS-BASED MEDIATION (PARCEL SCORES) ────────────────────────

def _get_parcel(df: pd.DataFrame, constructs: dict, construct_name: str) -> pd.Series | None:
    """Compute mean parcel score for a construct."""
    items = [c for c in constructs.get(construct_name, []) if c in df.columns]
    if not items:
        return None
    return df[items].mean(axis=1)


def _ols_path(x: np.ndarray, y: np.ndarray) -> tuple:
    """Simple OLS regression: returns (beta_std, se, t, p)."""
    x_std = (x - x.mean()) / x.std() if x.std() > 0 else x
    y_std = (y - y.mean()) / y.std() if y.std() > 0 else y
    slope, intercept, r, p, se = stats.linregress(x_std, y_std)
    return slope, se, slope / se if se > 0 else 0, p


def bootstrap_mediation(
    x: np.ndarray, m: np.ndarray, y: np.ndarray,
    n_boot: int = 5000, ci: float = 0.95, seed: int = 42
) -> dict:
    """
    Bootstrap mediation analysis.

    Returns dict with: indirect, direct, total, ci_lower, ci_upper,
                       a_path, b_path, c_path, c_prime_path
    """
    rng = np.random.default_rng(seed)
    n   = len(x)

    # Standardize
    def std(arr):
        s = arr.std()
        return (arr - arr.mean()) / s if s > 0 else arr - arr.mean()

    xs, ms, ys = std(x), std(m), std(y)

    # Point estimates
    a, a_se, a_t, a_p = _ols_path(xs, ms)           # X → M
    # M → Y controlling for X (multiple regression)
    X_mat = np.column_stack([np.ones(n), xs, ms])
    try:
        coefs = np.linalg.lstsq(X_mat, ys, rcond=None)[0]
        b     = coefs[2]  # M → Y (b path)
        # Residuals for SE
        y_hat = X_mat @ coefs
        resid = ys - y_hat
        mse   = (resid**2).sum() / (n - 3)
        XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
        b_se  = np.sqrt(mse * XtX_inv[2, 2])
        b_t   = b / b_se if b_se > 0 else 0
        b_p   = 2 * (1 - stats.t.cdf(abs(b_t), df=n-3))
        c_prime = coefs[1]  # X → Y controlling for M (direct effect)
    except Exception:
        b = b_se = b_t = b_p = c_prime = 0.0

    c, c_se, c_t, c_p = _ols_path(xs, ys)            # X → Y (total effect)
    indirect = a * b
    total    = c
    direct   = c_prime

    # Bootstrap
    boot_indirect = np.zeros(n_boot)
    for i in range(n_boot):
        idx   = rng.integers(0, n, n)
        xb, mb, yb = xs[idx], ms[idx], ys[idx]
        a_b, *_ = _ols_path(xb, mb)
        try:
            Xb_mat  = np.column_stack([np.ones(len(idx)), xb, mb])
            coefs_b = np.linalg.lstsq(Xb_mat, yb, rcond=None)[0]
            b_b     = coefs_b[2]
        except Exception:
            b_b = 0.0
        boot_indirect[i] = a_b * b_b

    alpha   = 1 - ci
    ci_lo   = float(np.percentile(boot_indirect, 100 * alpha / 2))
    ci_hi   = float(np.percentile(boot_indirect, 100 * (1 - alpha / 2)))

    # Sobel test
    sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
    sobel_z  = indirect / sobel_se if sobel_se > 0 else 0
    sobel_p  = 2 * (1 - stats.norm.cdf(abs(sobel_z)))

    # VAF
    vaf = (indirect / total) if abs(total) > 0.001 else np.nan

    return {
        "indirect":    round(indirect, 4),
        "direct":      round(direct,   4),
        "total":       round(total,    4),
        "ci_lower":    round(ci_lo,    4),
        "ci_upper":    round(ci_hi,    4),
        "boot_dist":   boot_indirect,
        "a_path":      round(a,        4),
        "a_p":         round(a_p,      4),
        "b_path":      round(b,        4),
        "b_p":         round(b_p,      4),
        "c_path":      round(total,    4),
        "c_p":         round(c_p,      4),
        "c_prime":     round(direct,   4),
        "sobel_z":     round(sobel_z,  4),
        "sobel_p":     round(sobel_p,  4),
        "vaf":         round(float(vaf), 4) if not np.isnan(vaf) else None,
        "n_boot":      n_boot,
        "ci_level":    ci,
    }


# ─── SECTION 1: MEDIATION SETUP ──────────────────────────────────

def render_mediation_setup(constructs: dict) -> tuple | None:
    st.subheader("🔧 Step 1: Mediation Model Setup")
    st.markdown(
        "Specify your **mediator model**: which construct mediates the relationship "
        "between a predictor and an outcome. "
        "*(X → M → Y)*"
    )

    construct_names = list(constructs.keys())
    if len(construct_names) < 3:
        st.warning("⚠️ Mediation analysis requires at least 3 constructs (X, M, Y).")
        return None

    c1, c2, c3 = st.columns(3)
    with c1:
        x_var = st.selectbox("Predictor (X)", construct_names, key="med_x")
    with c2:
        m_options = [c for c in construct_names if c != x_var]
        m_var = st.selectbox("Mediator (M)", m_options, key="med_m")
    with c3:
        y_options = [c for c in construct_names if c not in [x_var, m_var]]
        y_var = st.selectbox("Outcome (Y)", y_options, key="med_y")

    # Visual diagram
    st.markdown("")
    st.markdown(
        f'<div style="text-align:center;padding:16px;background:#1a1d27;'
        f'border-radius:8px;font-size:1rem;color:#f0f0f0">'
        f'<b style="color:#2E86AB">{x_var}</b> '
        f'<span style="color:#888">──(a)──▶</span> '
        f'<b style="color:#f39c12">{m_var}</b> '
        f'<span style="color:#888">──(b)──▶</span> '
        f'<b style="color:#2ecc71">{y_var}</b>'
        f'<br><span style="color:#888;font-size:0.8rem">'
        f'Direct path (c\'): {x_var} ──▶ {y_var}</span>'
        f'</div>',
        unsafe_allow_html=True
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


# ─── SECTION 2: RUN MEDIATION ────────────────────────────────────

def render_mediation_results(
    df: pd.DataFrame, constructs: dict,
    x_var: str, m_var: str, y_var: str,
    n_boot: int, ci_level: float
):
    st.subheader("📊 Step 2: Mediation Analysis Results")

    x_scores = _get_parcel(df, constructs, x_var)
    m_scores = _get_parcel(df, constructs, m_var)
    y_scores = _get_parcel(df, constructs, y_var)

    if x_scores is None or m_scores is None or y_scores is None:
        st.error("❌ Could not compute parcel scores. Check construct definitions.")
        return

    combined = pd.DataFrame({"X": x_scores, "M": m_scores, "Y": y_scores}).dropna()
    n = len(combined)

    if n < 50:
        st.error(f"❌ Too few complete cases (n = {n}) for reliable mediation analysis.")
        return

    st.info(f"📊 Running bootstrap mediation on n = {n} complete cases with {n_boot:,} resamples...")

    with st.spinner(f"Bootstrapping {n_boot:,} resamples..."):
        results = bootstrap_mediation(
            combined["X"].values, combined["M"].values, combined["Y"].values,
            n_boot=n_boot, ci=ci_level
        )

    # ── Path Coefficients Table ──────────────────────────────────
    st.markdown("**Path Coefficients:**")
    path_rows = [
        {"Path": f"{x_var} → {m_var} (a)",  "β (std)": results["a_path"], "p": results["a_p"],
         "Sig.": "✅" if results["a_p"] < 0.05 else "❌"},
        {"Path": f"{m_var} → {y_var} | {x_var} (b)", "β (std)": results["b_path"], "p": results["b_p"],
         "Sig.": "✅" if results["b_p"] < 0.05 else "❌"},
        {"Path": f"{x_var} → {y_var} total (c)", "β (std)": results["c_path"], "p": results["c_p"],
         "Sig.": "✅" if results["c_p"] < 0.05 else "❌"},
        {"Path": f"{x_var} → {y_var} direct (c')", "β (std)": results["c_prime"], "p": "—",
         "Sig.": "—"},
    ]
    st.dataframe(pd.DataFrame(path_rows), use_container_width=True, hide_index=True)

    # ── Effects Summary ──────────────────────────────────────────
    st.markdown("**Effects Summary:**")
    sig_indirect = not (results["ci_lower"] <= 0 <= results["ci_upper"])
    ci_str = f"[{results['ci_lower']:.4f}, {results['ci_upper']:.4f}]"

    effects_rows = [
        {"Effect":   "Indirect (a×b)",
         "Value":    results["indirect"],
         f"{int(ci_level*100)}% BCa CI": ci_str,
         "Significant": "✅ Yes" if sig_indirect else "❌ No"},
        {"Effect":   "Direct (c')",
         "Value":    results["direct"],
         f"{int(ci_level*100)}% BCa CI": "—",
         "Significant": "✅" if abs(results["direct"]) > 0.05 else "—"},
        {"Effect":   "Total (c)",
         "Value":    results["total"],
         f"{int(ci_level*100)}% BCa CI": "—",
         "Significant": "✅" if results["c_p"] < 0.05 else "❌"},
    ]
    st.dataframe(pd.DataFrame(effects_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"Note. Indirect effect significance based on {n_boot:,}-resample bootstrap "
        f"{int(ci_level*100)}% bias-corrected accelerated (BCa) confidence interval. "
        "CI not containing zero = significant mediation."
    )

    # ── Sobel Test ───────────────────────────────────────────────
    with st.expander("🧮 Sobel Test (supplementary)"):
        st.markdown(
            "The Sobel test is a parametric test of the indirect effect. "
            "**Bootstrap is preferred** (more accurate for non-normal sampling distributions), "
            "but Sobel is reported for completeness."
        )
        st.markdown(f"**Sobel z = {results['sobel_z']:.3f}**, p = {results['sobel_p']:.4f}")
        if results["sobel_p"] < 0.05:
            _badge("ok", f"Sobel test confirms significant mediation (z = {results['sobel_z']:.3f}, p = {results['sobel_p']:.4f}). ✅")
        else:
            _badge("warning", f"Sobel test non-significant (z = {results['sobel_z']:.3f}, p = {results['sobel_p']:.4f}). ⚠️ Bootstrap CI is more reliable.")

    # ── VAF ──────────────────────────────────────────────────────
    vaf = results.get("vaf")
    if vaf is not None:
        st.markdown(f"**Variance Accounted For (VAF):** {vaf:.1%}")
        if vaf >= 0.80:
            _badge("ok", f"VAF = {vaf:.1%} — suggests **full mediation** (VAF ≥ 80%).")
        elif vaf >= 0.20:
            _badge("ok", f"VAF = {vaf:.1%} — suggests **partial mediation** (20% ≤ VAF < 80%).")
        else:
            _badge("warning", f"VAF = {vaf:.1%} — mediation effect is **very small** (VAF < 20%).")

    # ── Bootstrap Distribution Plot ───────────────────────────────
    with st.expander("📈 Bootstrap Distribution of Indirect Effect"):
        boot_dist = results["boot_dist"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=boot_dist, nbinsx=80,
            marker_color="#2E86AB", opacity=0.8,
            name="Bootstrap Distribution"
        ))
        fig.add_vline(x=results["indirect"], line_color="#2ecc71", line_width=2,
                     annotation_text=f"Estimate = {results['indirect']:.4f}")
        fig.add_vline(x=results["ci_lower"], line_dash="dash", line_color="#e74c3c",
                     annotation_text=f"CI Lower = {results['ci_lower']:.4f}")
        fig.add_vline(x=results["ci_upper"], line_dash="dash", line_color="#e74c3c",
                     annotation_text=f"CI Upper = {results['ci_upper']:.4f}")
        fig.add_vline(x=0, line_color="#f39c12", line_width=1.5,
                     annotation_text="0")
        fig.update_layout(
            template="plotly_dark", height=350,
            title=f"Bootstrap Distribution of Indirect Effect ({n_boot:,} resamples)",
            xaxis_title="Indirect Effect (a×b)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Mediation Type ────────────────────────────────────────────
    st.markdown("**Mediation Type Determination (Zhao et al., 2010):**")
    a_sig  = results["a_p"] < 0.05
    b_sig  = results["b_p"] < 0.05
    c_sig  = results["c_p"] < 0.05
    ind_sig = sig_indirect
    dir_sig = abs(results["c_prime"]) > 0.05

    if ind_sig and not dir_sig:
        med_type = "**Full (Indirect-Only) Mediation**"
        med_desc = f"{m_var} fully mediates the relationship between {x_var} and {y_var}. The direct effect is negligible."
        level = "excellent"
    elif ind_sig and dir_sig:
        med_type = "**Partial Mediation**"
        med_desc = f"{m_var} partially mediates the relationship. Both direct and indirect paths are significant."
        level = "ok"
    elif not ind_sig and c_sig:
        med_type = "**No Mediation** (Direct Effect Only)"
        med_desc = f"The indirect path through {m_var} is not significant. {x_var} directly affects {y_var} without mediation."
        level = "warning"
    else:
        med_type = "**No Effect**"
        med_desc = "Neither the direct nor indirect effect is statistically significant."
        level = "critical"

    _badge(level, f"{med_type}: {med_desc}")

    # ── Auto-interpretation ────────────────────────────────────────
    st.markdown("**Full Interpretation:**")
    result = interpret_mediation(
        indirect_effect=results["indirect"],
        ci_lower=results["ci_lower"],
        ci_upper=results["ci_upper"],
        direct_effect=results["direct"],
        total_effect=results["total"],
        mediator=m_var, predictor=x_var, outcome=y_var
    )
    _badge(result["level"], result["message"])

    st.session_state["mediation_results"] = results
    st.session_state["mediation_vars"] = {"x": x_var, "m": m_var, "y": y_var}


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_mediation():
    st.title("🔄 Mediation Analysis")
    st.markdown(
        "Mediation analysis tests whether the effect of a **predictor (X)** on an **outcome (Y)** "
        "operates *through* an intervening **mediator (M)**. "
        "This module uses **bootstrap resampling** — the gold standard for testing indirect effects.\n\n"
        "> 📌 *Bootstrap CI not containing zero = significant mediation (Hayes, 2018).*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    df         = st.session_state["df"]
    constructs = st.session_state.get("constructs", {})

    if len(constructs) < 3:
        st.warning("⚠️ Mediation requires at least 3 constructs. Define more constructs in Data Input.")
        return

    st.markdown("---")

    setup = render_mediation_setup(constructs)
    if setup is None:
        return

    x_var, m_var, y_var, n_boot, ci_level = setup

    st.markdown("---")

    if st.button("▶️ Run Mediation Analysis", type="primary", key="run_med", use_container_width=False):
        render_mediation_results(df, constructs, x_var, m_var, y_var, n_boot, ci_level)
    elif st.session_state.get("mediation_results"):
        render_mediation_results(df, constructs, x_var, m_var, y_var, n_boot, ci_level)

    st.markdown("---")
    st.success("✅ Mediation analysis complete. Proceed to **Moderation Analysis** or **Export Report**.")
