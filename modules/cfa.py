"""
cfa.py - Confirmatory Factor Analysis Module.
Uses R/lavaan via r_bridge for methodologically correct CFA.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.interpretation import (
    interpret_loading, interpret_ave, interpret_cr,
    interpret_alpha, interpret_htmt, interpret_fit_index,
    interpret_overall_fit
)
from utils.apa_tables import fit_indices_table, reliability_validity_table, htmt_table

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


def render_model_spec(constructs):
    st.subheader("Step 1: CFA Model Specification")
    st.markdown(
        "The syntax below defines your **measurement model**. "
        "Each line specifies which items load on each latent construct. "
        "You can edit it directly if needed."
    )

    syntax = st.session_state.get("cfa_syntax", "")
    if not syntax:
        lines = [f"{c} =~ {' + '.join(items)}" for c, items in constructs.items() if items]
        syntax = "\n".join(lines)

    edited = st.text_area(
        "lavaan CFA Syntax",
        value=syntax,
        height=max(120, len(constructs) * 40),
        help="Format: ConstructName =~ item1 + item2 + item3"
    )
    st.session_state["cfa_syntax"] = edited

    with st.expander("Construct Summary"):
        for cname, items in constructs.items():
            n = len(items)
            ok = "OK" if n >= 3 else "Warning"
            st.markdown(f"- **{cname}**: {n} indicators — {', '.join(items)}")
            if n < 3:
                badge("warning", f"{cname} has only {n} indicator(s). Minimum is 3 for reliable CFA.")

    return edited


def render_estimation(syntax, df, constructs):
    st.subheader("Step 2: Model Estimation")

    estimator = st.session_state.get("recommended_estimator", "MLR")
    st.info(
        f"**Estimator:** {estimator} "
        f"(based on Mardia's normality test — see Descriptive Statistics). "
        "Override in Descriptive Statistics if needed."
    )

    all_items = list(set(
        item for items in constructs.values() for item in items
        if item in df.columns
    ))

    if st.button("Run CFA via R/lavaan", type="primary", key="run_cfa_btn", use_container_width=True):
        try:
            from r_scripts.r_bridge import run_cfa, check_r_available
            r_check = check_r_available()
            if not r_check["available"]:
                st.error(f"R is not available: {r_check['message']}")
                return None

            with st.spinner("Estimating CFA via R/lavaan... this may take a moment."):
                result = run_cfa(
                    df             = df,
                    indicator_cols = all_items,
                    model_syntax   = syntax,
                    estimator      = estimator
                )

            if "error" in result:
                st.error(f"CFA estimation failed: {result['error']}")
                st.markdown("**Common fixes:**")
                st.markdown("- Ensure item names in syntax match column names exactly")
                st.markdown("- Check for missing values")
                st.markdown("- Verify each construct has at least 3 indicators")
                return None

            n = result.get("n")
            if isinstance(n, list): n = n[0]
            converged = result.get("converged")
            if isinstance(converged, list): converged = converged[0]

            st.session_state["cfa_result"]    = result
            st.session_state["cfa_all_items"] = all_items
            st.success(f"CFA estimated successfully on n = {int(float(n)) if n else 'N/A'} complete cases. Converged: {converged}")

        except Exception as e:
            st.error(f"CFA error: {str(e)}")
            return None

    return st.session_state.get("cfa_result")


def render_fit_indices(result):
    st.subheader("Step 3: Model Fit Assessment")
    st.markdown(
        "Fit indices assess how well the hypothesized factor structure reproduces "
        "the observed covariance matrix. **Multiple criteria** should be evaluated jointly."
    )

    fit_raw = result.get("fit_indices")
    if fit_raw is None:
        st.warning("No fit indices available.")
        return {}

    if isinstance(fit_raw, dict):
        fit = {k: (v[0] if isinstance(v, list) else v) for k, v in fit_raw.items()}
    else:
        fit = {}

    # Normalize keys
    key_map = {
        "chisq": "chi2", "pvalue": "p", "dof": "df",
        "rmsea.robust": "rmsea", "cfi.robust": "cfi",
        "tli.robust": "tli", "srmr": "srmr",
    }
    normalized = {}
    for k, v in fit.items():
        nk = key_map.get(k, k.lower().replace(".", "_"))
        if v is not None:
            try:
                normalized[nk] = float(v)
            except (TypeError, ValueError):
                pass

    # Prefer scaled/robust versions
    for base in ["rmsea", "cfi", "tli", "chisq", "pvalue"]:
        scaled_key = f"{base}_scaled"
        if scaled_key in normalized and base not in normalized:
            normalized[base] = normalized[scaled_key]

    # Metric cards
    indices = ["rmsea", "cfi", "tli", "srmr"]
    cols    = st.columns(len(indices) + 1)

    chi2 = normalized.get("chi2") or normalized.get("chisq_scaled")
    df_  = normalized.get("df") or normalized.get("df_scaled")
    if chi2 and df_ and float(df_) > 0:
        cols[0].metric("chi2/df", f"{chi2/df_:.3f}")

    for i, idx in enumerate(indices):
        val = normalized.get(idx)
        if val is not None:
            label, color, level = _fit_label(idx, val)
            cols[i+1].metric(idx.upper(), f"{val:.3f}", label)

    # Full APA table
    st.markdown("**Complete Fit Indices:**")
    fit_df = fit_indices_table(normalized)
    if not fit_df.empty:
        st.dataframe(
            fit_df.style.set_properties(**{"color": "#1a1a1a", "background-color": "#ffffff"})
                        .set_table_styles([{
                            "selector": "th",
                            "props": [("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                        }]),
            use_container_width=True, hide_index=True
        )

    # Individual interpretations
    with st.expander("Fit Index Interpretations"):
        for idx in ["rmsea", "cfi", "tli", "srmr"]:
            val = normalized.get(idx)
            if val is not None:
                r = interpret_fit_index(idx, val)
                badge(r["level"], r["message"])

    # Overall verdict
    overall = interpret_overall_fit(normalized)
    st.markdown("**Overall Assessment:**")
    badge(
        "ok" if "acceptable" in overall.lower() or "good" in overall.lower() else
        "warning" if "marginal" in overall.lower() else "critical",
        overall
    )

    st.session_state["cfa_fit"] = normalized
    return normalized


def _fit_label(idx, val):
    from utils.thresholds import interpret_fit
    label, color, level = interpret_fit(idx, val)
    return label, color, level


def render_factor_loadings(result, constructs):
    st.subheader("Step 4: Factor Loadings")
    st.markdown(
        f"Standardized factor loadings (lambda). "
        f"**Criterion:** lambda >= .70 (strong); lambda >= .50 (acceptable)."
    )

    loadings_raw = result.get("loadings")
    if loadings_raw is None:
        st.warning("No loadings available.")
        return {}

    # Convert loadings to DataFrame - handles all formats
    try:
        if isinstance(loadings_raw, pd.DataFrame):
            loadings_df = loadings_raw
        elif isinstance(loadings_raw, list):
            loadings_df = pd.DataFrame(loadings_raw)
        elif isinstance(loadings_raw, dict):
            loadings_df = pd.DataFrame(loadings_raw)
        else:
            st.warning("Could not parse loadings.")
            return {}
    except Exception as e:
        st.warning(f"Could not parse loadings: {str(e)}")
        return {}

    if loadings_df.empty:
        st.warning("Loadings table is empty.")
        return {}

    # Ensure correct column names
    # Direct column mapping - columns from lavaan are:
    # construct, item, unstd, se, z, p, std
    col_map = {
        "construct": "construct",
        "item":      "item",
        "std":       "std",
        "std.all":   "std",
        "unstd":     "unstd",
        "est":       "unstd",
        "se":        "se",
        "z":         "z",
        "z-value":   "z",
        "p":         "p",
        "pvalue":    "p",
        "p-value":   "p",
        "lhs":       "construct",
        "rhs":       "item",
    }
    loadings_df = loadings_df.rename(columns=col_map)

    rows = []
    for _, row in loadings_df.iterrows():
        construct = str(row.get("construct", "—"))
        item      = str(row.get("item", "—"))
        std       = row.get("std")
        p_val     = row.get("p")
        try: std   = float(std)
        except: std = None
        try: p_val = float(p_val)
        except: p_val = None

        status = (
            "Strong"      if std and abs(std) >= 0.70 else
            "Acceptable"  if std and abs(std) >= 0.50 else
            "Weak"
        )
        rows.append({
            "Construct":     construct,
            "Item":          item,
            "Std Loading":   round(std, 3) if std else "—",
            "p":             round(p_val, 3) if p_val is not None else "—",
            "Status":        status,
        })

    if not rows:
        st.warning("No loadings to display.")
        return {}

    disp_df = pd.DataFrame(rows)

    def color_loading_status(val):
        if val == "Strong":     return "color:#1a7a4a;font-weight:700"
        elif val == "Acceptable": return "color:#1a6fa8"
        else: return "color:#c0392b"

    def color_std(val):
        try:
            v = abs(float(val))
            if v >= 0.70: return "color:#1a7a4a;font-weight:700"
            elif v >= 0.50: return "color:#1a6fa8"
            else: return "color:#c0392b"
        except: return ""

    st.dataframe(
        disp_df.style
               .map(color_loading_status, subset=["Status"])
               .map(color_std, subset=["Std Loading"])
               .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
               .set_table_styles([{
                   "selector":"th",
                   "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
               }]),
        use_container_width=True, hide_index=True
    )

    # Item interpretations
    with st.expander("Item-by-Item Interpretation"):
        for row in rows:
            std = row.get("Std Loading")
            try:
                std = float(std)
                r = interpret_loading(std, row["Item"])
                badge(r["level"], r["message"])
            except (TypeError, ValueError):
                pass

    # Save for other modules
    loadings_by_construct = {}
    for row in rows:
        c = row["Construct"]
        if c not in loadings_by_construct:
            loadings_by_construct[c] = {}
        try:
            loadings_by_construct[c][row["Item"]] = float(row["Std Loading"])
        except (TypeError, ValueError):
            pass

    st.session_state["cfa_loadings"] = loadings_by_construct
    return loadings_by_construct


def render_reliability(result, constructs, df):
    st.subheader("Step 5: Reliability Analysis")
    st.markdown(
        "**Criteria:** Cronbach's alpha >= .70, CR >= .70, AVE >= .50 "
        "(Fornell & Larcker, 1981; Hair et al., 2019)."
    )

    rel_raw = result.get("reliability")
    ave_raw = result.get("ave")

    metrics = {}
    cfa_loadings = st.session_state.get("cfa_loadings", {})

    for cname, items in constructs.items():
        valid_items = [c for c in items if c in df.columns]
        if len(valid_items) < 2:
            continue

        data = df[valid_items].dropna()
        n_items = len(valid_items)

        # Cronbach's alpha from data
        if n_items >= 2 and len(data) >= 10:
            item_vars  = data.var(ddof=1)
            total_var  = data.sum(axis=1).var(ddof=1)
            alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var) if total_var > 0 else None
        else:
            alpha = None

        # CR and AVE from lavaan reliability output
        cr = omega = ave_val = None

        if rel_raw and isinstance(rel_raw, dict):
            # semTools reliability returns matrix
            for key in ["omega2", "omega3", "omega"]:
                val = rel_raw.get(key)
                if val:
                    if isinstance(val, dict) and cname in val:
                        try: cr = float(val[cname])
                        except: pass
                    elif isinstance(val, list) and len(val) > 0:
                        try: cr = float(val[0])
                        except: pass

        if ave_raw and isinstance(ave_raw, dict) and cname in ave_raw:
            try:
                av = ave_raw[cname]
                ave_val = float(av[0] if isinstance(av, list) else av)
            except: pass

        # Compute CR and AVE from loadings if not from R
        lambdas = list(cfa_loadings.get(cname, {}).values())
        if lambdas and (cr is None or ave_val is None):
            lam = np.array([float(l) for l in lambdas if l is not None])
            if len(lam) > 0:
                sum_lam  = lam.sum()
                sum_lam2 = (lam**2).sum()
                err_vars = 1 - lam**2
                if cr is None:
                    denom = sum_lam**2 + err_vars.sum()
                    cr = float(sum_lam**2 / denom) if denom > 0 else None
                if ave_val is None:
                    denom2 = sum_lam2 + err_vars.sum()
                    ave_val = float(sum_lam2 / denom2) if denom2 > 0 else None

        metrics[cname] = {
            "alpha":   round(float(alpha), 3) if alpha and not np.isnan(alpha) else None,
            "cr":      round(float(cr), 3)    if cr    else None,
            "omega":   round(float(cr), 3)    if cr    else None,
            "ave":     round(float(ave_val), 3) if ave_val else None,
            "n_items": n_items,
            "lambdas": [float(l) for l in lambdas if l is not None],
        }

    if not metrics:
        st.warning("Could not compute reliability metrics.")
        return {}

    # Display table
    rel_df = reliability_validity_table(metrics)
    if not rel_df.empty:
        def color_pass(val):
            if val == "Pass": return "color:#1a7a4a;font-weight:700"
            elif val == "Fail": return "color:#c0392b;font-weight:700"
            return ""
        pass_cols = [c for c in rel_df.columns if ">=" in c]
        st.dataframe(
            rel_df.style.map(color_pass, subset=pass_cols)
                        .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                        .set_table_styles([{
                            "selector":"th",
                            "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                        }]),
            use_container_width=True, hide_index=True
        )
        st.caption("Note: CR = Composite Reliability; AVE = Average Variance Extracted.")

    # Per-construct interpretations
    with st.expander("Construct-by-Construct Interpretation"):
        for cname, m in metrics.items():
            st.markdown(f"**{cname}:**")
            if m.get("alpha"): badge(**interpret_alpha(m["alpha"], cname))
            if m.get("cr"):    badge(**interpret_cr(m["cr"], cname))
            if m.get("ave"):   badge(**interpret_ave(m["ave"], cname))

    # Bar chart
    chart_data = pd.DataFrame({
        "Construct": list(metrics.keys()),
        "Alpha":     [m.get("alpha", 0) or 0 for m in metrics.values()],
        "CR":        [m.get("cr", 0) or 0 for m in metrics.values()],
        "AVE":       [m.get("ave", 0) or 0 for m in metrics.values()],
    })
    fig = go.Figure()
    for col, color in [("Alpha","#2E86AB"),("CR","#1a7a4a"),("AVE","#b7770d")]:
        fig.add_trace(go.Bar(name=col, x=chart_data["Construct"], y=chart_data[col],
                             marker_color=color, opacity=0.85))
    fig.add_hline(y=0.70, line_dash="dash", line_color="#555", annotation_text="alpha/CR >= .70")
    fig.add_hline(y=0.50, line_dash="dot",  line_color="#c0392b", annotation_text="AVE >= .50")
    fig.update_layout(
        barmode="group", template="simple_white", height=350,
        title="Reliability and Validity Metrics by Construct",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(orientation="h", y=1.1),
        font_color="#1a1a1a",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.session_state["cfa_metrics"] = metrics
    return metrics


def render_validity(metrics, df, constructs):
    st.subheader("Step 6: Construct Validity")

    tab1, tab2 = st.tabs(["Convergent Validity", "Discriminant Validity (HTMT and Fornell-Larcker)"])

    with tab1:
        st.markdown(
            "**Convergent validity** is supported when:\n"
            "1. All factor loadings lambda >= .50\n"
            "2. AVE >= .50 for each construct (Fornell & Larcker, 1981)"
        )
        for cname, m in metrics.items():
            ave = m.get("ave", 0) or 0
            lambdas = m.get("lambdas", [])
            min_loading = min(lambdas) if lambdas else 0
            if ave >= 0.50 and min_loading >= 0.50:
                badge("excellent", f"**{cname}**: Convergent validity supported (AVE = {ave:.3f}, min lambda = {min_loading:.3f}). ✅")
            elif ave >= 0.50:
                badge("ok", f"**{cname}**: AVE adequate ({ave:.3f}) but some loadings are weak (min = {min_loading:.3f}). ⚠️")
            else:
                badge("critical", f"**{cname}**: Convergent validity NOT supported (AVE = {ave:.3f} < .50). ❌")

    with tab2:
        st.markdown(
            "**HTMT criterion** (Henseler et al., 2015): HTMT < .85 (strict) or < .90 (liberal).\n\n"
            "**Fornell-Larcker criterion**: sqrt(AVE) > inter-construct correlations."
        )
        construct_names = list(constructs.keys())
        if len(construct_names) < 2:
            st.info("At least 2 constructs needed for discriminant validity.")
            return

        # Compute parcel scores
        parcel_df = pd.DataFrame()
        for cname, items in constructs.items():
            valid = [c for c in items if c in df.columns]
            if valid:
                parcel_df[cname] = df[valid].mean(axis=1)
        parcel_df = parcel_df.dropna()
        cn = list(parcel_df.columns)

        if len(cn) < 2:
            st.warning("Not enough constructs with valid data.")
            return

        # Correlation matrix for Fornell-Larcker
        corr = parcel_df.corr()

        # HTMT matrix (compute from items)
        htmt_mat = pd.DataFrame(np.nan, index=cn, columns=cn)
        for i, c1 in enumerate(cn):
            for j, c2 in enumerate(cn):
                if i == j: continue
                items1 = [x for x in constructs.get(c1, []) if x in df.columns]
                items2 = [x for x in constructs.get(c2, []) if x in df.columns]
                if len(items1) >= 2 and len(items2) >= 2:
                    combined = df[items1 + items2].dropna()
                    if len(combined) < 10: continue
                    from scipy import stats as scipy_stats
                    cross_corrs = []
                    for a in items1:
                        for b in items2:
                            if a in combined and b in combined:
                                r, _ = scipy_stats.pearsonr(combined[a], combined[b])
                                cross_corrs.append(abs(r))
                    within1 = [abs(scipy_stats.pearsonr(combined[items1[a]], combined[items1[b]])[0])
                               for a in range(len(items1)) for b in range(a+1, len(items1))
                               if items1[a] in combined and items1[b] in combined]
                    within2 = [abs(scipy_stats.pearsonr(combined[items2[a]], combined[items2[b]])[0])
                               for a in range(len(items2)) for b in range(a+1, len(items2))
                               if items2[a] in combined and items2[b] in combined]
                    if cross_corrs and within1 and within2:
                        mean_cross = np.mean(cross_corrs)
                        denom = np.sqrt(np.mean(within1) * np.mean(within2))
                        if denom > 0:
                            htmt_mat.loc[c1, c2] = round(mean_cross / denom, 3)

        st.markdown("**HTMT Matrix:**")
        htmt_display = htmt_table(htmt_mat)
        st.dataframe(
            htmt_display.style.set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                              .set_table_styles([{
                                  "selector":"th",
                                  "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                              }]),
            use_container_width=True
        )
        st.caption("Note: HTMT < .85 = OK; .85-.90 = Warning; > .90 = Fail (Henseler et al., 2015)")

        with st.expander("HTMT Pair-by-Pair Interpretation"):
            for i, c1 in enumerate(cn):
                for j, c2 in enumerate(cn):
                    if i < j:
                        val = htmt_mat.loc[c1, c2]
                        if not pd.isna(val):
                            r = interpret_htmt(float(val), c1, c2)
                            badge(r["level"], r["message"])

        # Fornell-Larcker
        st.markdown("**Fornell-Larcker Criterion:**")
        st.markdown("Diagonal = sqrt(AVE). Off-diagonal = inter-construct correlations.")
        fl_df = pd.DataFrame(index=cn, columns=cn, dtype=object)
        for i, c1 in enumerate(cn):
            for j, c2 in enumerate(cn):
                if i == j:
                    ave = metrics.get(c1, {}).get("ave")
                    fl_df.loc[c1, c2] = f"**{np.sqrt(ave):.3f}**" if ave else "—"
                else:
                    r = corr.loc[c1, c2] if c1 in corr.index and c2 in corr.columns else np.nan
                    fl_df.loc[c1, c2] = f"{r:.3f}" if not np.isnan(r) else "—"

        st.dataframe(fl_df, use_container_width=True)
        st.caption("Note: Bold diagonal = sqrt(AVE). Discriminant validity supported when sqrt(AVE) > all off-diagonal values in same row/column.")

        # Fornell-Larcker check
        fl_pass = True
        for cname in cn:
            ave = metrics.get(cname, {}).get("ave")
            if not ave: continue
            sqrt_ave = np.sqrt(ave)
            for other in cn:
                if cname != other and other in corr.columns:
                    r = abs(corr.loc[cname, other])
                    if sqrt_ave <= r:
                        badge("warning", f"Fornell-Larcker violated: **{cname}** sqrt(AVE) ({sqrt_ave:.3f}) <= correlation with **{other}** ({r:.3f}).")
                        fl_pass = False
        if fl_pass:
            badge("excellent", "Fornell-Larcker criterion satisfied for all construct pairs. Discriminant validity supported. ✅")


def render_modification_indices(result):
    st.subheader("Step 7: Modification Indices")
    st.markdown(
        "Modification indices (MI) indicate how much chi-square would decrease "
        "if a parameter were freed. MI > 3.84 is notable.\n\n"
        "> **Important:** Only apply modifications that are **theoretically justified**. "
        "Never modify a model purely based on statistics."
    )

    mi_raw = result.get("mod_indices")
    if mi_raw is None:
        st.info("No modification indices available.")
        return

    if isinstance(mi_raw, pd.DataFrame):
        mi_df = mi_raw
    elif isinstance(mi_raw, dict):
        mi_df = pd.DataFrame(mi_raw)
    else:
        st.info("Could not parse modification indices.")
        return

    if mi_df.empty:
        badge("excellent", "No notable modification indices. Model is well-specified. ✅")
        return

    # Filter notable MIs
    mi_col = next((c for c in mi_df.columns if "mi" in c.lower()), None)
    if mi_col:
        notable = mi_df[mi_df[mi_col] >= 3.84].head(10)
        if not notable.empty:
            st.warning(f"{len(notable)} modification index/indices above threshold (MI >= 3.84):")
            st.dataframe(notable.round(3), use_container_width=True, hide_index=True)
            badge("warning",
                "Consider freeing parameters with large MI **only if theoretically justified**. "
                "Purely data-driven modifications lead to overfitting."
            )
        else:
            badge("excellent", "No modification indices above threshold (MI >= 3.84). Model is well-specified. ✅")
    else:
        st.dataframe(mi_df.head(10).round(3), use_container_width=True, hide_index=True)


def render_cfa_checklist(fit, metrics):
    st.subheader("Step 8: CFA Methodological Checklist")

    from utils.thresholds import FIT, CFA as CFA_T
    checks = {
        f"Model fit — RMSEA <= {FIT['rmsea_acceptable']}":
            (fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"],
        f"Model fit — CFI >= {FIT['cfi_acceptable']}":
            (fit.get("cfi") or 0) >= FIT["cfi_acceptable"],
        f"Model fit — SRMR <= {FIT['srmr_acceptable']}":
            (fit.get("srmr") or 999) <= FIT["srmr_acceptable"],
        "All factor loadings >= .50":
            all(min(m.get("lambdas",[0])) >= CFA_T["loading_min"]
                for m in metrics.values() if m.get("lambdas")),
        "All AVE >= .50 (convergent validity)":
            all((m.get("ave") or 0) >= CFA_T["ave_min"] for m in metrics.values()),
        "All CR >= .70 (composite reliability)":
            all((m.get("cr") or 0) >= CFA_T["cr_min"] for m in metrics.values()),
        "All Cronbach's alpha >= .70":
            all((m.get("alpha") or 0) >= CFA_T["alpha_min"] for m in metrics.values()),
    }

    rows = [{"Check": k, "Status": "Pass" if v else "Fail"} for k, v in checks.items()]
    cdf  = pd.DataFrame(rows)

    def color_status(val):
        if val == "Pass": return "color:#1a7a4a;font-weight:700"
        return "color:#c0392b;font-weight:700"

    st.dataframe(
        cdf.style.map(color_status, subset=["Status"])
                 .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                 .set_table_styles([{
                     "selector":"th",
                     "props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]
                 }]),
        use_container_width=True, hide_index=True
    )

    n_fail = sum(1 for v in checks.values() if not v)
    if n_fail == 0:
        badge("excellent", "All CFA criteria passed! The measurement model is valid and reliable. Proceed to Structural Model (SEM). ✅")
        st.session_state["cfa_complete"] = True
    elif n_fail <= 2:
        badge("warning", f"{n_fail} criterion/criteria not met. Review items marked Fail before proceeding to SEM.")
        st.session_state["cfa_complete"] = False
    else:
        badge("critical", f"{n_fail} criteria not met. The measurement model needs re-specification before SEM.")
        st.session_state["cfa_complete"] = False


def render_cfa():
    st.title("Confirmatory Factor Analysis (CFA)")
    st.markdown(
        "CFA tests whether the **hypothesized factor structure** adequately fits the observed data. "
        "It assesses the **measurement model** before estimating structural relationships.\n\n"
        "> CFA must demonstrate adequate fit and validity **before** proceeding to SEM."
    )

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df         = st.session_state["df"]
    constructs = st.session_state.get("constructs", {})

    if not constructs:
        st.warning("No constructs defined. Please define constructs in Data Input or run EFA first.")
        return

    st.markdown("---")
    syntax = render_model_spec(constructs)
    st.markdown("---")
    result = render_estimation(syntax, df, constructs)

    if result is None:
        st.info("Run the CFA model above to see results.")
        return

    st.markdown("---")
    fit = render_fit_indices(result)
    st.markdown("---")
    loadings = render_factor_loadings(result, constructs)
    st.markdown("---")
    metrics = render_reliability(result, constructs, df)
    st.markdown("---")
    if metrics:
        render_validity(metrics, df, constructs)
    st.markdown("---")
    render_modification_indices(result)
    st.markdown("---")
    if fit and metrics:
        render_cfa_checklist(fit, metrics)

    st.markdown("---")
    badge("ok", "CFA complete. If the measurement model is satisfactory, proceed to Structural Model (SEM).")
