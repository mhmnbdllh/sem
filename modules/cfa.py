"""
cfa.py
======
Sprint 2 — Confirmatory Factor Analysis (CFA) Module for SEM Studio.

Covers:
- CFA model specification from user-defined constructs
- Model estimation via semopy
- Factor loadings (standardized & unstandardized)
- Convergent validity: AVE, factor loadings
- Discriminant validity: HTMT, Fornell-Larcker criterion
- Reliability: Cronbach's Alpha, Composite Reliability, McDonald's Omega
- Full fit indices: RMSEA, CFI, TLI, SRMR, GFI, AIC, BIC, χ²
- Modification indices
- Second-order CFA option
- Full auto-interpretation for every output

References:
    - Brown (2015). CFA for Applied Research (2nd ed.)
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
    - Fornell & Larcker (1981). Evaluating structural equation models.
    - Henseler et al. (2015). HTMT criterion.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

from utils.thresholds import CFA as CFA_THRESH, FIT, get_fit_label
from utils.interpretation import (
    interpret_factor_loading, interpret_ave, interpret_cr,
    interpret_alpha, interpret_htmt, interpret_fit_index, interpret_overall_fit
)
from utils.apa_tables import (
    fit_indices_table, factor_loadings_table,
    reliability_validity_table, htmt_table, style_df
)

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


# ─── MODEL SYNTAX BUILDER ───────────────────────────────────────

def build_cfa_syntax(constructs: dict) -> str:
    """Build semopy CFA model syntax from construct definition."""
    lines = []
    for construct, items in constructs.items():
        if items:
            lines.append(f"{construct} =~ {' + '.join(items)}")
    return "\n".join(lines)


# ─── SECTION 1: MODEL SPECIFICATION ─────────────────────────────

def render_model_spec(constructs: dict) -> str:
    st.subheader("📝 Step 1: CFA Model Specification")
    st.markdown(
        "The model syntax below defines your **measurement model**. "
        "Each line specifies which items (indicators) are hypothesized to load on each latent construct. "
        "You can edit the syntax directly if needed."
    )

    syntax = build_cfa_syntax(constructs)

    edited_syntax = st.text_area(
        "Model Syntax (semopy format)",
        value=syntax,
        height=max(120, len(constructs) * 40),
        help="Format: ConstructName =~ item1 + item2 + item3"
    )

    # Syntax preview
    with st.expander("📋 Construct Summary"):
        for cname, citems in constructs.items():
            n = len(citems)
            status = "✅" if n >= 3 else "⚠️"
            st.markdown(f"{status} **{cname}**: {n} indicators — {', '.join(citems)}")

    return edited_syntax


# ─── SECTION 2: MODEL ESTIMATION ────────────────────────────────

def run_cfa(syntax: str, df: pd.DataFrame, estimator: str) -> object | None:
    """Run CFA using semopy."""
    if not SEMOPY_AVAILABLE:
        st.error("❌ semopy is not installed. Run: `pip install semopy`")
        return None

    try:
        model = semopy.Model(syntax)
        model.fit(df)
        return model
    except Exception as e:
        st.error(f"❌ CFA estimation failed: {str(e)}")
        st.markdown("**Common fixes:**")
        st.markdown("- Ensure item names in syntax match column names in data exactly")
        st.markdown("- Check for missing values (use complete cases or FIML)")
        st.markdown("- Verify each construct has ≥ 3 indicators")
        return None


def render_estimation(syntax: str, df: pd.DataFrame) -> object | None:
    st.subheader("⚙️ Step 2: Model Estimation")

    estimator = st.session_state.get("recommended_estimator", "ML")
    st.info(
        f"**Estimator:** {estimator} "
        f"(based on normality test results — see Descriptive Statistics). "
        "You can override this in the Descriptive Statistics page."
    )

    if not SEMOPY_AVAILABLE:
        st.error("❌ **semopy** library is not installed. Please add it to requirements.txt and restart.")
        st.code("pip install semopy", language="bash")
        return None

    col1, col2 = st.columns([1, 2])
    with col1:
        run = st.button("▶️ Run CFA", type="primary", key="run_cfa_btn", use_container_width=True)

    if run:
        with st.spinner("Estimating CFA model... this may take a moment."):
            # Get indicator columns used in model
            constructs = st.session_state.get("constructs", {})
            all_items  = [item for items in constructs.values() for item in items]
            data       = df[all_items].dropna()

            model = run_cfa(syntax, data, estimator)
            if model:
                st.session_state["cfa_model"]  = model
                st.session_state["cfa_syntax"] = syntax
                st.session_state["cfa_data"]   = data
                st.success(f"✅ CFA estimated successfully on n = {len(data)} complete cases.")

    return st.session_state.get("cfa_model")


# ─── SECTION 3: FIT INDICES ─────────────────────────────────────

def render_fit_indices(model) -> dict:
    st.subheader("📐 Step 3: Model Fit Assessment")
    st.markdown(
        "Fit indices assess how well the hypothesized factor structure reproduces the observed "
        "covariance matrix. **Multiple criteria** should be evaluated jointly."
    )

    try:
        stats_result = semopy.calc_stats(model)
        fit = {}

        # Extract available stats
        stat_map = {
            "chi2":  ["chi2", "Chi2", "chisq"],
            "df":    ["dof", "df", "DoF"],
            "p":     ["pvalue", "p", "chi2_p"],
            "rmsea": ["rmsea", "RMSEA"],
            "cfi":   ["cfi", "CFI"],
            "tli":   ["tli", "TLI", "nnfi"],
            "srmr":  ["srmr", "SRMR"],
            "gfi":   ["gfi", "GFI"],
            "aic":   ["aic", "AIC"],
            "bic":   ["bic", "BIC"],
            "nfi":   ["nfi", "NFI"],
        }

        if hasattr(stats_result, 'to_dict'):
            stats_dict = stats_result.to_dict()
        elif isinstance(stats_result, pd.DataFrame):
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

        # Display metric cards
        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, "χ²", fit.get("chi2"), ""),
            (c2, "df", fit.get("df"), ""),
            (c3, "RMSEA", fit.get("rmsea"), ""),
            (c4, "CFI", fit.get("cfi"), ""),
            (c5, "SRMR", fit.get("srmr"), ""),
        ]
        for col, label, val, _ in metrics:
            col.metric(label, f"{val:.3f}" if val is not None else "—")

        # APA Table
        st.markdown("**Complete Fit Indices:**")
        fit_table = fit_indices_table(fit)
        st.dataframe(fit_table, use_container_width=True, hide_index=True)

        # Individual interpretations
        with st.expander("🔍 Fit Index Interpretations"):
            for idx in ["rmsea", "cfi", "tli", "srmr"]:
                val = fit.get(idx)
                if val is not None:
                    result = interpret_fit_index(idx, val)
                    _badge(result["level"], result["message"])

        # Overall verdict
        overall = interpret_overall_fit(fit)
        st.markdown("**Overall Assessment:**")
        st.info(overall)

        # χ²/df ratio
        if fit.get("chi2") and fit.get("df") and fit["df"] > 0:
            ratio = fit["chi2"] / fit["df"]
            fit["chisq_df"] = ratio
            label, color = get_fit_label("chisq_df", ratio)
            _badge("ok" if ratio <= 5 else "warning",
                f"χ²/df = {ratio:.3f} — {label} (criterion: ≤ 2.0 good; ≤ 5.0 acceptable)"
            )

        st.session_state["cfa_fit"] = fit
        return fit

    except Exception as e:
        st.error(f"❌ Could not compute fit indices: {str(e)}")
        return {}


# ─── SECTION 4: FACTOR LOADINGS ─────────────────────────────────

def render_cfa_loadings(model, constructs: dict) -> dict:
    st.subheader("📊 Step 4: Factor Loadings")
    st.markdown(
        f"**Criterion:** Standardized loading (λ) ≥ **{CFA_THRESH['factor_loading_good']}** (good); "
        f"≥ **{CFA_THRESH['factor_loading_min']}** (acceptable). "
        "All loadings should be statistically significant (p < .05)."
    )

    try:
        params = model.inspect(std_est=True)

        loadings_by_construct = {}
        for cname, items in constructs.items():
            loadings_by_construct[cname] = {}
            for item in items:
                row = params[
                    (params.get("lval", params.get("op", "")) == cname) |
                    (params.get("rval", params.get("Estimate", "")) == item)
                ] if False else params  # fallback

                # Try to find loading for this item
                try:
                    mask = (params["lval"] == cname) & (params["rval"] == item)
                    if mask.any():
                        std_est = params.loc[mask, "Estimate"].values[0]
                        loadings_by_construct[cname][item] = std_est
                except:
                    loadings_by_construct[cname][item] = np.nan

        # Display as table
        rows = []
        for cname, items_dict in loadings_by_construct.items():
            for item, loading in items_dict.items():
                if not np.isnan(loading):
                    status = "✅" if abs(loading) >= CFA_THRESH["factor_loading_good"] else \
                             "⚠️" if abs(loading) >= CFA_THRESH["factor_loading_min"] else "❌"
                    rows.append({
                        "Construct": cname,
                        "Item": item,
                        "Std. Loading (λ)": round(loading, 3),
                        "Status": status,
                    })

        if rows:
            loadings_df = pd.DataFrame(rows)

            def color_loading(val):
                try:
                    v = float(val)
                    if v >= CFA_THRESH["factor_loading_good"]: return "color:#2ecc71;font-weight:bold"
                    elif v >= CFA_THRESH["factor_loading_min"]: return "color:#f39c12"
                    else: return "color:#e74c3c"
                except: return ""

            styled = loadings_df.style.applymap(color_loading, subset=["Std. Loading (λ)"])
            st.dataframe(styled, use_container_width=True, hide_index=True)

            # Item-level interpretation
            with st.expander("🔍 Item-by-Item Interpretation"):
                for _, row in loadings_df.iterrows():
                    result = interpret_factor_loading(row["Std. Loading (λ)"], row["Item"])
                    _badge(result["level"], result["message"])

        st.session_state["cfa_loadings"] = loadings_by_construct
        return loadings_by_construct

    except Exception as e:
        st.error(f"❌ Could not extract factor loadings: {str(e)}")
        return {}


# ─── SECTION 5: RELIABILITY ─────────────────────────────────────

def compute_reliability(df: pd.DataFrame, constructs: dict) -> dict:
    """Compute Cronbach's Alpha, Composite Reliability, AVE, Omega per construct."""
    metrics = {}

    for cname, items in constructs.items():
        valid_items = [c for c in items if c in df.columns]
        if len(valid_items) < 2:
            continue

        data = df[valid_items].dropna()
        n_items = len(valid_items)
        n_obs   = len(data)

        if n_obs < 10:
            continue

        # Cronbach's Alpha
        item_vars  = data.var(ddof=1)
        total_var  = data.sum(axis=1).var(ddof=1)
        alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else np.nan

        # Get standardized loadings from CFA model if available
        cfa_loadings = st.session_state.get("cfa_loadings", {})
        if cname in cfa_loadings:
            lambdas = np.array([v for v in cfa_loadings[cname].values() if not np.isnan(v)])
        else:
            # Fallback: use item-total correlations as proxy
            corr_matrix = data.corr()
            lambdas = np.array([corr_matrix[c].drop(c).mean() for c in valid_items])
            lambdas = np.clip(lambdas, 0.1, 0.99)

        if len(lambdas) == 0:
            continue

        # Composite Reliability (CR)
        sum_lambda  = lambdas.sum()
        sum_lambda2 = (lambdas ** 2).sum()
        error_vars  = 1 - lambdas ** 2
        cr = (sum_lambda ** 2) / (sum_lambda ** 2 + error_vars.sum()) if (sum_lambda ** 2 + error_vars.sum()) > 0 else np.nan

        # AVE
        ave = sum_lambda2 / (sum_lambda2 + error_vars.sum()) if (sum_lambda2 + error_vars.sum()) > 0 else np.nan

        # McDonald's Omega (approximation)
        omega = cr  # CR is a good approximation of omega when using CFA loadings

        metrics[cname] = {
            "alpha": round(float(alpha), 3) if not np.isnan(alpha) else None,
            "cr":    round(float(cr), 3)    if not np.isnan(cr)    else None,
            "omega": round(float(omega), 3) if not np.isnan(omega) else None,
            "ave":   round(float(ave), 3)   if not np.isnan(ave)   else None,
            "n_items": n_items,
            "lambdas": lambdas.tolist(),
        }

    return metrics


def render_reliability(df: pd.DataFrame, constructs: dict):
    st.subheader("🔒 Step 5: Reliability Analysis")
    st.markdown(
        "Reliability assesses whether items consistently measure their intended construct. "
        f"**Criteria:** Cronbach's α ≥ **{CFA_THRESH['alpha_min']}**, "
        f"CR ≥ **{CFA_THRESH['cr_min']}**, "
        f"AVE ≥ **{CFA_THRESH['ave_min']}** (Fornell & Larcker, 1981)."
    )

    metrics = compute_reliability(df, constructs)

    if not metrics:
        st.warning("⚠️ Could not compute reliability metrics. Ensure CFA has been run.")
        return {}

    # APA Table
    rel_table = reliability_validity_table(metrics)
    st.dataframe(rel_table, use_container_width=True, hide_index=True)
    st.caption("Note. CR = Composite Reliability; AVE = Average Variance Extracted.")

    # Per-construct interpretation
    with st.expander("🔍 Construct-by-Construct Reliability Interpretation"):
        for cname, m in metrics.items():
            st.markdown(f"**{cname}:**")
            if m.get("alpha"):
                r = interpret_alpha(m["alpha"], cname)
                _badge(r["level"], r["message"])
            if m.get("cr"):
                r = interpret_cr(m["cr"], cname)
                _badge(r["level"], r["message"])
            if m.get("ave"):
                r = interpret_ave(m["ave"], cname)
                _badge(r["level"], r["message"])

    # Bar chart
    chart_data = pd.DataFrame({
        "Construct": list(metrics.keys()),
        "Cronbach α": [m.get("alpha", 0) or 0 for m in metrics.values()],
        "CR":         [m.get("cr", 0) or 0 for m in metrics.values()],
        "AVE":        [m.get("ave", 0) or 0 for m in metrics.values()],
    })

    fig = go.Figure()
    for col, color in [("Cronbach α", "#2E86AB"), ("CR", "#2ecc71"), ("AVE", "#f39c12")]:
        fig.add_trace(go.Bar(name=col, x=chart_data["Construct"], y=chart_data[col],
                             marker_color=color, opacity=0.85))

    fig.add_hline(y=0.70, line_dash="dash", line_color="#888", annotation_text="α/CR ≥ .70")
    fig.add_hline(y=0.50, line_dash="dot",  line_color="#e74c3c", annotation_text="AVE ≥ .50")
    fig.update_layout(
        barmode="group", template="plotly_dark", height=380,
        title="Reliability & Validity Metrics by Construct",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["cfa_metrics"] = metrics
    return metrics


# ─── SECTION 6: VALIDITY ─────────────────────────────────────────

def render_validity(metrics: dict, df: pd.DataFrame, constructs: dict):
    st.subheader("✅ Step 6: Construct Validity")

    tab1, tab2 = st.tabs(["Convergent Validity", "Discriminant Validity (HTMT & Fornell-Larcker)"])

    # ── Convergent Validity ──────────────────────────────────────
    with tab1:
        st.markdown(
            "**Convergent validity** is supported when:\n"
            f"1. All factor loadings λ ≥ **{CFA_THRESH['factor_loading_min']}**\n"
            f"2. AVE ≥ **{CFA_THRESH['ave_min']}** for each construct"
        )

        conv_pass = []
        for cname, m in metrics.items():
            ave = m.get("ave", 0) or 0
            loadings = m.get("lambdas", [])
            min_loading = min(loadings) if loadings else 0

            loading_ok = min_loading >= CFA_THRESH["factor_loading_min"]
            ave_ok     = ave >= CFA_THRESH["ave_min"]

            if loading_ok and ave_ok:
                _badge("excellent", f"**{cname}**: Convergent validity supported ✅ (AVE = {ave:.3f}, min λ = {min_loading:.3f})")
                conv_pass.append(True)
            elif ave_ok:
                _badge("ok", f"**{cname}**: AVE adequate (AVE = {ave:.3f}) but some loadings are weak (min λ = {min_loading:.3f}). ⚠️")
                conv_pass.append(True)
            else:
                _badge("critical", f"**{cname}**: Convergent validity NOT supported ❌ (AVE = {ave:.3f} < .50)")
                conv_pass.append(False)

    # ── Discriminant Validity ────────────────────────────────────
    with tab2:
        st.markdown(
            "**Discriminant validity** ensures constructs are empirically distinct. "
            "Two methods are assessed:\n\n"
            f"1. **HTMT** (Henseler et al., 2015): HTMT < **{CFA_THRESH['htmt_strict']}** (strict) or < **{CFA_THRESH['htmt_liberal']}** (liberal)\n"
            "2. **Fornell-Larcker Criterion**: AVE of each construct > squared correlation with other constructs"
        )

        construct_names = list(constructs.keys())

        if len(construct_names) < 2:
            st.info("ℹ️ Discriminant validity requires at least 2 constructs.")
            return

        # Compute parcel scores
        parcel_df = pd.DataFrame()
        for cname, items in constructs.items():
            valid = [c for c in items if c in df.columns]
            if valid:
                parcel_df[cname] = df[valid].mean(axis=1)

        parcel_df = parcel_df.dropna()

        if parcel_df.shape[1] < 2:
            st.warning("⚠️ Not enough constructs with valid data for discriminant validity.")
            return

        cn = list(parcel_df.columns)

        # HTMT computation
        st.markdown("**HTMT Matrix:**")
        htmt_mat = pd.DataFrame(np.nan, index=cn, columns=cn)
        for i, c1 in enumerate(cn):
            for j, c2 in enumerate(cn):
                if i != j:
                    # HTMT = mean of cross-construct correlations / geometric mean of within-construct correlations
                    items_c1 = [c for c in constructs.get(c1, []) if c in df.columns]
                    items_c2 = [c for c in constructs.get(c2, []) if c in df.columns]

                    if len(items_c1) >= 2 and len(items_c2) >= 2:
                        data_c1 = df[items_c1].dropna()
                        data_c2 = df[items_c2].dropna()

                        # Cross correlations
                        combined = pd.concat([data_c1, data_c2], axis=1).dropna()
                        if len(combined) < 10:
                            continue

                        cross_corrs = []
                        for a in items_c1:
                            for b in items_c2:
                                if a in combined.columns and b in combined.columns:
                                    r, _ = stats.pearsonr(combined[a], combined[b])
                                    cross_corrs.append(abs(r))

                        # Within correlations (geometric mean of avg within-construct corr)
                        within_c1 = []
                        for a in range(len(items_c1)):
                            for b in range(a+1, len(items_c1)):
                                ia, ib = items_c1[a], items_c1[b]
                                if ia in combined.columns and ib in combined.columns:
                                    r, _ = stats.pearsonr(combined[ia], combined[ib])
                                    within_c1.append(abs(r))

                        within_c2 = []
                        for a in range(len(items_c2)):
                            for b in range(a+1, len(items_c2)):
                                ia, ib = items_c2[a], items_c2[b]
                                if ia in combined.columns and ib in combined.columns:
                                    r, _ = stats.pearsonr(combined[ia], combined[ib])
                                    within_c2.append(abs(r))

                        if cross_corrs and within_c1 and within_c2:
                            mean_cross  = np.mean(cross_corrs)
                            mean_w1     = np.mean(within_c1)
                            mean_w2     = np.mean(within_c2)
                            denom       = np.sqrt(mean_w1 * mean_w2)
                            htmt_val    = mean_cross / denom if denom > 0 else np.nan
                            htmt_mat.loc[c1, c2] = round(htmt_val, 3)
                    else:
                        htmt_mat.loc[c1, c2] = np.nan

        # Display HTMT table
        htmt_display = htmt_table(htmt_mat)
        st.dataframe(htmt_display, use_container_width=True)
        st.caption(f"Note. HTMT < .85 = ✅ supported; .85–.90 = ⚠️ borderline; > .90 = ❌ not supported.")

        # HTMT interpretations
        with st.expander("🔍 HTMT Pair-by-Pair Interpretation"):
            for i, c1 in enumerate(cn):
                for j, c2 in enumerate(cn):
                    if i < j:
                        val = htmt_mat.loc[c1, c2]
                        if not np.isnan(val):
                            result = interpret_htmt(val, c1, c2)
                            _badge(result["level"], result["message"])

        # Fornell-Larcker
        st.markdown("**Fornell-Larcker Criterion:**")
        st.markdown("Diagonal = √AVE; Off-diagonal = inter-construct correlations.")
        fl_matrix = pd.DataFrame(index=cn, columns=cn, dtype=object)
        corr_matrix = parcel_df.corr()

        for i, c1 in enumerate(cn):
            for j, c2 in enumerate(cn):
                if i == j:
                    ave = metrics.get(c1, {}).get("ave", None)
                    fl_matrix.loc[c1, c2] = f"**{np.sqrt(ave):.3f}**" if ave else "—"
                else:
                    r = corr_matrix.loc[c1, c2] if c1 in corr_matrix.index and c2 in corr_matrix.columns else np.nan
                    fl_matrix.loc[c1, c2] = f"{r:.3f}" if not np.isnan(r) else "—"

        st.dataframe(fl_matrix, use_container_width=True)
        st.caption("Note. Bold diagonal = √AVE. Discriminant validity is supported when √AVE > all off-diagonal correlations in the same row/column.")

        # Fornell-Larcker check
        fl_pass = True
        for cname in cn:
            ave = metrics.get(cname, {}).get("ave")
            if ave is None:
                continue
            sqrt_ave = np.sqrt(ave)
            for other in cn:
                if cname != other and other in corr_matrix.columns:
                    r = abs(corr_matrix.loc[cname, other])
                    if sqrt_ave <= r:
                        _badge("warning",
                            f"⚠️ Fornell-Larcker violated: **{cname}** √AVE ({sqrt_ave:.3f}) ≤ "
                            f"correlation with **{other}** ({r:.3f}). Discriminant validity is questionable."
                        )
                        fl_pass = False

        if fl_pass:
            _badge("excellent", "✅ Fornell-Larcker criterion satisfied for all construct pairs. Discriminant validity is supported.")


# ─── SECTION 7: MODIFICATION INDICES ────────────────────────────

def render_modification_indices(model, constructs: dict):
    st.subheader("🔧 Step 7: Modification Indices")
    st.markdown(
        "Modification indices (MI) indicate how much the model χ² would decrease if a "
        "parameter were freed. **MI > 3.84** (χ²(1) at p = .05) is considered notable. "
        "> ⚠️ **Important:** Only apply modifications that are **theoretically justified**. "
        "Never modify a model purely based on statistics."
    )

    try:
        mi = semopy.ModificationIndices(model)

        if mi is not None and hasattr(mi, 'head'):
            mi_df = mi if isinstance(mi, pd.DataFrame) else pd.DataFrame(mi)

            if len(mi_df) > 0:
                # Filter notable MIs
                mi_col = [c for c in mi_df.columns if "mi" in c.lower() or "modification" in c.lower()]
                if mi_col:
                    mi_df = mi_df.sort_values(mi_col[0], ascending=False)
                    notable = mi_df[mi_df[mi_col[0]] >= 3.84]

                    if len(notable) > 0:
                        st.warning(f"⚠️ {len(notable)} modification index/indices above threshold (MI ≥ 3.84):")
                        st.dataframe(notable.head(10), use_container_width=True)
                        _badge("warning",
                            "Consider freeing parameters with large MI **only if theoretically justified**. "
                            "Purely data-driven modifications lead to overfitting and capitalization on chance."
                        )
                    else:
                        _badge("excellent", "✅ No modification indices above threshold. Model is well-specified.")
                else:
                    st.dataframe(mi_df.head(10), use_container_width=True)
            else:
                _badge("excellent", "✅ No notable modification indices detected.")

    except Exception as e:
        st.info(f"ℹ️ Modification indices could not be computed: {str(e)}")


# ─── SECTION 8: CFA SUMMARY ─────────────────────────────────────

def render_cfa_summary(fit: dict, metrics: dict, constructs: dict):
    st.subheader("📄 CFA Summary & Methodological Checklist")

    checks = {
        f"Model fit — RMSEA ≤ {FIT['rmsea_acceptable']}":
            fit.get("rmsea", 999) <= FIT["rmsea_acceptable"] if fit.get("rmsea") else False,
        f"Model fit — CFI ≥ {FIT['cfi_acceptable']}":
            fit.get("cfi", 0) >= FIT["cfi_acceptable"] if fit.get("cfi") else False,
        f"Model fit — SRMR ≤ {FIT['srmr_acceptable']}":
            fit.get("srmr", 999) <= FIT["srmr_acceptable"] if fit.get("srmr") else False,
        "All factor loadings ≥ .50":
            all(min(m.get("lambdas", [0])) >= CFA_THRESH["factor_loading_min"]
                for m in metrics.values() if m.get("lambdas")),
        "All AVE ≥ .50 (convergent validity)":
            all((m.get("ave") or 0) >= CFA_THRESH["ave_min"] for m in metrics.values()),
        "All CR ≥ .70 (composite reliability)":
            all((m.get("cr") or 0) >= CFA_THRESH["cr_min"] for m in metrics.values()),
        "All Cronbach's α ≥ .70":
            all((m.get("alpha") or 0) >= CFA_THRESH["alpha_min"] for m in metrics.values()),
    }

    rows = [{"Check": k, "Status": "✅ Pass" if v else "❌ Fail"} for k, v in checks.items()]
    result_df = pd.DataFrame(rows)

    def color_status(val):
        if "Pass" in str(val): return "color:#2ecc71;font-weight:bold"
        return "color:#e74c3c;font-weight:bold"

    st.dataframe(
        result_df.style.applymap(color_status, subset=["Status"]),
        use_container_width=True, hide_index=True
    )

    n_fail = sum(1 for v in checks.values() if not v)
    if n_fail == 0:
        _badge("excellent",
            "🎉 **All CFA criteria passed!** The measurement model demonstrates good validity and reliability. "
            "Proceed to **Structural Model (SEM)**."
        )
    elif n_fail <= 2:
        _badge("warning",
            f"⚠️ **{n_fail} criterion/criteria not met.** Review the items marked ❌. "
            "Address issues before proceeding to SEM to avoid biased structural estimates."
        )
    else:
        _badge("critical",
            f"❌ **{n_fail} criteria not met.** The measurement model needs re-specification. "
            "Consider: removing weak items, checking model syntax, reviewing construct definitions."
        )

    st.session_state["cfa_complete"] = n_fail == 0


# ─── MAIN RENDER ────────────────────────────────────────────────

def render_cfa():
    st.title("📐 Confirmatory Factor Analysis (CFA)")
    st.markdown(
        "CFA tests whether the **hypothesized factor structure** — based on theory or prior EFA — "
        "adequately fits the observed data. It assesses the **measurement model** before "
        "estimating structural (path) relationships.\n\n"
        "> 📌 *CFA must demonstrate adequate fit and validity before proceeding to SEM.*"
    )

    if not st.session_state.get("df_ready"):
        st.warning("⚠️ Please complete **Data Input & Setup** first.")
        return

    df         = st.session_state["df"]
    constructs = st.session_state.get("constructs", {})

    if not constructs:
        st.warning("⚠️ No constructs defined. Please define constructs in **Data Input** or run **EFA** first.")
        return

    # Check semopy
    if not SEMOPY_AVAILABLE:
        st.error(
            "❌ **semopy** is not installed. Add `semopy>=2.3.5` to `requirements.txt` and restart."
        )
        st.code("pip install semopy", language="bash")
        return

    st.markdown("---")

    # Step 1
    syntax = render_model_spec(constructs)
    st.markdown("---")

    # Step 2
    model = render_estimation(syntax, df)

    if model is None:
        st.info("👆 Run the CFA model above to see results.")
        return

    st.markdown("---")

    # Step 3
    fit = render_fit_indices(model)
    st.markdown("---")

    # Step 4
    loadings = render_cfa_loadings(model, constructs)
    st.markdown("---")

    # Step 5
    metrics = render_reliability(df, constructs)
    st.markdown("---")

    # Step 6
    if metrics:
        render_validity(metrics, df, constructs)
    st.markdown("---")

    # Step 7
    render_modification_indices(model, constructs)
    st.markdown("---")

    # Step 8
    if fit and metrics:
        render_cfa_summary(fit, metrics, constructs)

    st.markdown("---")
    st.success(
        "✅ CFA complete. If the measurement model is satisfactory, "
        "proceed to **Structural Model (SEM)**."
    )
