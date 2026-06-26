"""
efa.py - Exploratory Factor Analysis Module.
Uses R/psych via r_bridge for methodologically correct EFA.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.interpretation import interpret_kmo, interpret_bartlett, interpret_loading

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


def render_factorability(result):
    st.subheader("Step 3: Factorability Tests")
    st.markdown(
        "**KMO** and **Bartlett's Test** confirm whether data are suitable for factor analysis."
    )

    kmo = result.get("kmo")
    if isinstance(kmo, list): kmo = kmo[0]
    kmo = float(kmo) if kmo is not None else None

    bart_chi2 = result.get("bartlett_chi2")
    bart_p    = result.get("bartlett_p")
    if isinstance(bart_chi2, list): bart_chi2 = bart_chi2[0]
    if isinstance(bart_p, list):    bart_p    = bart_p[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("KMO Overall",       f"{kmo:.3f}" if kmo is not None else "N/A")
    c2.metric("Bartlett's chi2",   f"{float(bart_chi2):.3f}" if bart_chi2 else "N/A")
    c3.metric("Bartlett's p-value","< .001" if bart_p is not None and float(bart_p) < 0.001 else f"{float(bart_p):.4f}" if bart_p is not None else "N/A")

    if kmo is not None:
        r = interpret_kmo(kmo)
        badge(r["level"], r["message"])
    if bart_p is not None:
        r = interpret_bartlett(float(bart_p))
        badge(r["level"], r["message"])

    # Per-item KMO
    kmo_per = result.get("kmo_per_item")
    if kmo_per and isinstance(kmo_per, dict):
        with st.expander("KMO Per Item (MSA — Measures of Sampling Adequacy)"):
            st.markdown("Items with MSA < .50 should be considered for removal.")
            msa_rows = []
            for item, val in kmo_per.items():
                v = float(val) if val is not None else 0
                msa_rows.append({
                    "Item":   item,
                    "MSA":    round(v, 3),
                    "Status": "Good" if v >= 0.70 else "Acceptable" if v >= 0.60 else "Warning" if v >= 0.50 else "Poor",
                })
            msa_df = pd.DataFrame(msa_rows).sort_values("MSA")

            def color_msa(val):
                try:
                    v = float(val)
                    if v >= 0.70: return "color:#1a7a4a;font-weight:600"
                    elif v >= 0.60: return "color:#1a6fa8"
                    elif v >= 0.50: return "color:#b7770d"
                    else: return "color:#c0392b;font-weight:600"
                except: return ""

            st.dataframe(
                msa_df.style.map(color_msa, subset=["MSA"]),
                use_container_width=True, hide_index=True
            )

    factorable = (kmo is not None and float(kmo) >= 0.60) and (bart_p is not None and float(bart_p) < 0.05)
    if not factorable and kmo is not None and float(kmo) < 0.60:
        badge("warning",
            f"KMO = {kmo:.3f} — data tidak cocok untuk EFA. "
            "Saran: Lewati EFA dan langsung ke CFA menggunakan struktur konstruk dari Data Input. "
            "Klik 'Skip EFA → Go to CFA' di bagian atas halaman ini."
        )
    return factorable


def render_factor_number(result):
    st.subheader("Step 4: Scree Plot and Factor Number Confirmation")
    st.markdown(
        "**Parallel Analysis** (gold standard via R/psych) + **Kaiser criterion** "
        "(eigenvalue > 1) + **Scree plot**."
    )

    suggested = result.get("suggested_factors")
    if isinstance(suggested, list): suggested = suggested[0]
    suggested = int(suggested) if suggested else 1

    # Scree plot using loadings SS
    var_explained = result.get("var_explained")
    if var_explained and isinstance(var_explained, dict):
        # Extract eigenvalues from SS loadings row
        ss_row = var_explained.get("SS loadings", var_explained.get("ss_loadings"))
        if ss_row:
            if isinstance(ss_row, dict):
                ev_vals = [float(v) for v in ss_row.values()]
            elif isinstance(ss_row, list):
                ev_vals = [float(v) for v in ss_row]
            else:
                ev_vals = []

            if ev_vals:
                ev_df = pd.DataFrame({
                    "Factor":     range(1, len(ev_vals)+1),
                    "SS Loading": ev_vals
                })
                fig = px.line(
                    ev_df, x="Factor", y="SS Loading",
                    markers=True, template="simple_white",
                    title="Scree Plot (SS Loadings from R/psych PAF)",
                )
                fig.add_hline(y=1, line_dash="dash", line_color="#b7770d",
                             annotation_text="Eigenvalue = 1 (Kaiser)")
                fig.update_layout(
                    height=350, font_color="#1a1a1a",
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    margin=dict(t=60, b=40, l=40, r=120),
                )
                st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Parallel Analysis Suggests", f"{suggested} factor(s)")
    c2.metric("Recommendation", f"{suggested} factor(s)")

    badge("ok",
        f"Parallel analysis (R/psych, 100 iterations) suggests **{suggested}** factor(s). "
        "Always combine statistical criteria with theoretical justification."
    )

    # Display suggestion only - user already set n_factors in Step 2 before Run EFA
    n_factors = int(st.session_state.get("efa_n_factors_widget", suggested))
    st.info(
        f"Factors extracted: {n_factors} "
        f"(parallel analysis suggested: {suggested}). "
        f"To change, update Step 2 above and click Run EFA again."
    )
    return n_factors


def render_loadings_table(result, item_names, n_factors):
    st.subheader("Step 5: Factor Loading Matrix")
    st.markdown(
        f"Factor loadings (PAF extraction via R/psych). "
        f"**Strong:** lambda >= .70 | **Acceptable:** lambda >= .50 | **Weak:** lambda < .50. "
        f"Cross-loadings > .32 are flagged."
    )

    loadings_raw = result.get("loadings")
    if loadings_raw is None:
        st.warning("No loadings available.")
        return None

    # Convert to DataFrame
    if isinstance(loadings_raw, pd.DataFrame):
        loadings_mat = loadings_raw
    elif isinstance(loadings_raw, dict):
        loadings_mat = pd.DataFrame(loadings_raw)
    elif isinstance(loadings_raw, list):
        if len(loadings_raw) > 0 and isinstance(loadings_raw[0], list):
            actual_cols = len(loadings_raw[0])
        else:
            actual_cols = n_factors
        loadings_mat = pd.DataFrame(
            np.array(loadings_raw),
            index=item_names[:len(loadings_raw)],
            columns=[f"F{i+1}" for i in range(actual_cols)]
        )
    else:
        st.warning("Could not parse loadings.")
        return None

    # Ensure index is item names
    if list(loadings_mat.index) != item_names[:len(loadings_mat)]:
        loadings_mat.index = item_names[:len(loadings_mat)]

    # Rename columns to F1, F2, ... using ACTUAL number of columns
    actual_n_factors = loadings_mat.shape[1]
    factor_cols = [f"F{i+1}" for i in range(actual_n_factors)]
    loadings_mat.columns = factor_cols

    # Communalities
    communalities_raw = result.get("communalities")
    if communalities_raw:
        if isinstance(communalities_raw, dict):
            comm_vals = [float(v) for v in communalities_raw.values()]
        elif isinstance(communalities_raw, list):
            comm_vals = [float(v) for v in communalities_raw]
        else:
            comm_vals = []
        if comm_vals:
            loadings_mat["h2"] = comm_vals[:len(loadings_mat)]

    # Color code
    def color_loading(val):
        try:
            v = float(val)
            if abs(v) >= 0.70: return "color:#1a7a4a;font-weight:700"
            elif abs(v) >= 0.50: return "color:#1a6fa8;font-weight:600"
            elif abs(v) >= 0.32: return "color:#b7770d"
            else: return "color:#aaaaaa"
        except: return ""

    display_cols = factor_cols[:loadings_mat.shape[1]]
    st.dataframe(
        loadings_mat.round(3).style.map(color_loading, subset=display_cols),
        use_container_width=True
    )

    # Variance explained
    var_explained = result.get("var_explained")
    if var_explained and isinstance(var_explained, dict):
        st.markdown("Variance Explained:")
        var_rows = []
        row_names = list(var_explained.keys())
        for rn in row_names:
            row_vals = var_explained[rn]
            if isinstance(row_vals, dict):
                vals = list(row_vals.values())
            elif isinstance(row_vals, list):
                vals = row_vals
            else:
                vals = [row_vals]
            var_rows.append({"Statistic": rn, **{f"F{i+1}": round(float(v), 3) for i, v in enumerate(vals)}})
        st.dataframe(pd.DataFrame(var_rows), use_container_width=True, hide_index=True)

        # Total variance
        cum_row = var_explained.get("Cumulative Var") or var_explained.get("cumulative_var")
        if cum_row:
            if isinstance(cum_row, dict): vals = list(cum_row.values())
            elif isinstance(cum_row, list): vals = cum_row
            else: vals = [cum_row]
            if vals:
                total_var = float(vals[-1]) if vals else 0
                if total_var >= 0.50:
                    badge("ok", f"Total variance explained: **{total_var:.1%}** (criterion: >= 50%). ✅")
                else:
                    badge("warning", f"Total variance explained: **{total_var:.1%}** (criterion: >= 50%). ⚠️ Consider adding factors.")

    # Cross-loading detection
    cross = []
    for item in loadings_mat.index:
        row = loadings_mat.loc[item, display_cols].abs()
        if (row >= 0.32).sum() > 1:
            cross.append(item)
    if cross:
        badge("warning", f"Cross-loadings detected: **{', '.join(cross)}**. These items load on multiple factors. Consider removing or rewriting.")
    else:
        badge("excellent", "No cross-loadings detected. Each item loads primarily on one factor. ✅")

    # Low communalities
    if "h2" in loadings_mat.columns:
        low_h2 = [str(idx) for idx in loadings_mat.index if loadings_mat.loc[idx, "h2"] < 0.40]
        if low_h2:
            badge("warning", f"Low communality (h2 < .40): **{', '.join(low_h2)}**. These items share little variance with their factors.")

    # Heatmap
    with st.expander("Factor Loading Heatmap"):
        fig = px.imshow(
            loadings_mat[display_cols].T,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            template="simple_white",
            text_auto=".2f",
            title="Factor Loading Heatmap",
            aspect="auto",
        )
        fig.update_layout(
            height=max(300, n_factors * 120),
            margin=dict(t=60, b=40, l=40, r=120),
            font_color="#1a1a1a",
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Item-by-item interpretation
    with st.expander("Item-by-Item Loading Interpretation"):
        for item in loadings_mat.index:
            row = loadings_mat.loc[item, display_cols].abs()
            best_f   = row.idxmax()
            best_val = loadings_mat.loc[item, best_f]
            r = interpret_loading(float(best_val), str(item))
            badge(r["level"], f"{r['message']} — Primary factor: **{best_f}**")

    return loadings_mat


def render_factor_naming_setup(n_factors_expected):
    constructs      = st.session_state.get("constructs", {})
    construct_names = list(constructs.keys())

    
    st.markdown(
        "Assign a name to each factor before running EFA. "
        "Use the same names as Data Input to ensure consistency across all modules."
    )

    if not construct_names:
        st.warning("No constructs defined in Data Input. Please complete Data Input first.")
        return {}

    # Reminder box showing constructs from Data Input
    reminder_parts = ["Constructs from Data Input (for reference):"]
    for cname, items in constructs.items():
        reminder_parts.append(f"  - {cname}: {', '.join(items)}")
    st.info("\n".join(reminder_parts))
    st.markdown("---")

    factor_names = {}
    OPTIONS      = construct_names + ["[ Custom name... ]"]

    for i in range(n_factors_expected):
        f     = f"F{i+1}"
        saved = st.session_state.get(f"efa_fname_{f}", "")

        if saved and saved in construct_names:
            default_idx = construct_names.index(saved)
        else:
            default_idx = min(i, len(construct_names) - 1)

        c1, c2 = st.columns([2, 2])
        with c1:
            selected = st.selectbox(
                f"Factor {i+1} name",
                options=OPTIONS,
                index=default_idx,
                key=f"efa_sel_{f}",
            )
        with c2:
            if selected == "[ Custom name... ]":
                prev_custom = saved if saved and saved not in construct_names else ""
                custom = st.text_input(
                    f"Type name for Factor {i+1}",
                    value=prev_custom,
                    key=f"efa_custom_{f}",
                    placeholder="e.g., NewFactor",
                )
                name = custom.strip() if custom.strip() else f"Factor{i+1}"
            else:
                name = selected
                items_list = constructs.get(name, [])
                if items_list:
                    st.markdown(f"Items: {', '.join(items_list)}")

        factor_names[f] = name

    values = list(factor_names.values())
    if len(set(values)) < len(values):
        badge("warning", "Duplicate names detected. Each factor should have a unique name.")
    elif all(v in construct_names for v in values):
        badge("ok", "All factor names match Data Input constructs. Consistency ensured.")
    else:
        badge("ok", "Factor names set. Ensure custom names are consistent with structural paths.")

    st.session_state["efa_factor_names"] = factor_names
    return factor_names


def render_factor_naming(loadings_mat, n_factors):
    """
    Step shown AFTER seeing factor loading results.
    User assigns construct names based on which items load on each factor.
    """
    st.subheader("Step 6: Assign Construct Names to Factors")
    st.markdown(
        "Based on the loading matrix above, assign a construct name to each factor. "
        "Look at which items load highest on each factor, then select the matching construct. "
        "Use the same names as defined in Data Input."
    )

    # Auto-suggest: show which items meet loading threshold per factor
    threshold = 0.40
    st.caption(f"Items with |λ| ≥ {threshold} are highlighted as strong candidates for each factor.")
    for f in [c for c in loadings_mat.columns if c.startswith("F")]:
        strong = loadings_mat[f][loadings_mat[f].abs() >= threshold].sort_values(key=abs, ascending=False)
        if not strong.empty:
            items_str = ", ".join([f"{item} (λ={v:.3f})" for item, v in strong.items()])
            st.caption(f"  {f} strong items: {items_str}")

    constructs      = st.session_state.get("constructs", {})
    construct_names = list(constructs.keys())
    factor_cols     = [f"F{i+1}" for i in range(n_factors) if f"F{i+1}" in loadings_mat.columns]

    if construct_names:
        reminder_lines = ["Constructs from Data Input (for reference):"]
        for cname, items in constructs.items():
            reminder_lines.append(f"  - {cname}: {', '.join(items)}")
        st.info("\n".join(reminder_lines))

    OPTIONS      = construct_names + ["[ Custom name... ]"]
    factor_names = {}

    for i, f in enumerate(factor_cols):
        top_items = loadings_mat[f].abs().nlargest(3).index.tolist()
        top_str   = ", ".join([f"{item} (λ={loadings_mat.loc[item,f]:.3f})" for item in top_items])

        c1, c2 = st.columns([2, 3])
        with c1:
            saved       = st.session_state.get(f"efa_fname_{f}", "")
            default_idx = construct_names.index(saved) if saved in construct_names else min(i, max(0, len(construct_names)-1))
            selected    = st.selectbox(
                f"{f} name",
                options=OPTIONS,
                index=default_idx,
                key=f"efa_fname_{f}",
            )
            if selected == "[ Custom name... ]":
                prev   = saved if saved and saved not in construct_names else ""
                custom = st.text_input(f"Custom name for {f}", value=prev, key=f"efa_custom_{f}")
                name   = custom.strip() if custom.strip() else f"Factor{i+1}"
            else:
                name = selected
        with c2:
            st.markdown(f"**Top items loading on {f}:** {top_str}")

        factor_names[f] = name

    values = list(factor_names.values())
    if len(set(values)) < len(values):
        badge("warning", "Duplicate names detected. Each factor should have a unique name.")
    elif all(v in construct_names for v in values):
        badge("ok", "All factor names match Data Input constructs. Consistency ensured.")
    else:
        badge("ok", "Factor names set. Ensure custom names are consistent with structural paths.")

    st.session_state["efa_factor_names"] = factor_names
    return factor_names


def render_efa_summary(loadings_mat, n_factors, factor_names, item_names):
    st.subheader("Step 7: EFA Summary and CFA Preparation")
    st.markdown("Suggested construct structure for CFA based on EFA results.")

    factor_cols = [f"F{i+1}" for i in range(n_factors) if f"F{i+1}" in loadings_mat.columns]
    suggested   = {}

    for f in factor_cols:
        fname = factor_names.get(f, f)
        primary_items = [
            str(item) for item in loadings_mat.index
            if loadings_mat.loc[item, factor_cols].abs().idxmax() == f
            and abs(loadings_mat.loc[item, f]) >= 0.40
        ]
        suggested[fname] = primary_items
        n = len(primary_items)
        status = "OK" if n >= 3 else "Warning"
        st.markdown(f"**{fname}** ({n} items): {', '.join(primary_items)}")
        if n < 3:
            badge("warning", f"{fname} has only {n} item(s). CFA requires at least 3 indicators.")

    st.session_state["efa_suggested_constructs"] = suggested

    # Check if all constructs have at least 3 items
    all_ok = all(len(v) >= 3 for v in suggested.values())

    if not all_ok:
        badge("warning",
            "Some constructs have fewer than 3 items based on EFA results. "
            "This usually means cross-loadings are high or items don't cluster well. "
            "Options: (1) Use original Data Input structure instead, "
            "(2) Accept EFA structure and edit CFA syntax manually."
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Use EFA Results for CFA", type="primary", key="efa_to_cfa", use_container_width=True):
            # Use original constructs from Data Input for items, EFA for factor assignment
            # This ensures all items are preserved
            original_constructs = st.session_state.get("constructs", {})
            factor_names_map = st.session_state.get("efa_factor_names", {})

            # Build new constructs: map original items to EFA-assigned factor names
            new_constructs = {}
            for fname, construct_name in factor_names_map.items():
                # Get original items for this construct
                orig_items = original_constructs.get(construct_name, [])
                if orig_items:
                    new_constructs[construct_name] = orig_items

            # Fallback: if mapping fails, use original constructs
            if not new_constructs:
                new_constructs = original_constructs

            st.session_state["constructs"] = new_constructs
            cfa_lines = [f"{c} =~ {' + '.join(items)}" for c, items in new_constructs.items() if items]
            st.session_state["cfa_syntax"] = "\n".join(cfa_lines)
            sem_lines = cfa_lines.copy()
            for pred, out in st.session_state.get("structural_paths", []):
                sem_lines.append(f"{out} ~ {pred}")
            st.session_state["sem_syntax"] = "\n".join(sem_lines)
            # Save JSON backup
            import json
            cfg = {"constructs": {k: list(v) for k,v in new_constructs.items()},
                   "structural_paths": [list(p) for p in st.session_state.get("structural_paths", [])],
                   "cfa_syntax": st.session_state["cfa_syntax"],
                   "sem_syntax": st.session_state["sem_syntax"]}
            st.session_state["_model_config_json"] = json.dumps(cfg)
            st.session_state["current_page"] = "cfa"
            badge("excellent", "Structure transferred. Navigating to CFA automatically...")
            st.rerun()

    with col2:
        if st.button("📋 Keep Data Input Structure", type="secondary", key="efa_keep_original", use_container_width=True):
            # Use original constructs from Data Input unchanged
            original_constructs = st.session_state.get("constructs", {})
            cfa_lines = [f"{c} =~ {' + '.join(items)}" for c, items in original_constructs.items() if items]
            st.session_state["cfa_syntax"] = "\n".join(cfa_lines)
            sem_lines = cfa_lines.copy()
            for pred, out in st.session_state.get("structural_paths", []):
                sem_lines.append(f"{out} ~ {pred}")
            st.session_state["sem_syntax"] = "\n".join(sem_lines)
            st.session_state["current_page"] = "cfa"
            badge("excellent", "Using original Data Input structure. Navigating to CFA...")
            st.rerun()

    badge("ok",
        "Note: EFA is data-driven. Always verify the factor structure is "
        "theoretically justified before proceeding to CFA."
    )


def render_efa():
    st.title("Exploratory Factor Analysis (EFA)")
    st.markdown("EFA explores how items cluster empirically.")

    with st.expander("❓ Do I need EFA? (Click to read)", expanded=not st.session_state.get("efa_complete")):
        st.markdown("""
**EFA adalah OPSIONAL.** Lewati EFA dan langsung ke CFA jika:
- Instrumen kamu sudah divalidasi dari penelitian sebelumnya
- Kamu sudah tahu item mana yang masuk ke konstruk mana

**Jalankan EFA jika:**
- Kamu mengembangkan instrumen baru
- Kamu ingin memverifikasi struktur konstruk secara empiris
        """)
        if st.button("Lewati EFA → Langsung ke CFA", key="skip_efa_btn", type="secondary"):
            st.session_state["efa_complete"] = True
            st.session_state["current_page"] = "cfa"
            st.rerun()

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    df             = st.session_state["df"]
    assignments    = st.session_state.get("assignments", {})
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if len(indicator_cols) < 3:
        st.error("At least 3 indicator variables are required for EFA.")
        return

    data = df[indicator_cols].dropna()
    st.info(f"Analyzing {len(indicator_cols)} items across {len(data)} complete cases.")
    st.markdown("---")

    # EFA settings
    st.subheader("Step 1: Extraction and Rotation Settings")
    col1, col2 = st.columns(2)
    with col1:
        rotation = st.selectbox(
            "Rotation Method",
            options=["oblimin", "varimax", "promax", "none"],
            format_func=lambda x: {
                "oblimin": "Oblimin (oblique — recommended for social science)",
                "varimax": "Varimax (orthogonal — assumes uncorrelated factors)",
                "promax":  "Promax (oblique — alternative)",
                "none":    "No rotation",
            }[x],
            key="efa_rotation_widget"
        )
    with col2:
        st.markdown("**Extraction Method:** Principal Axis Factoring (PAF)")
        st.markdown("*Standard for SEM/CFA preparation (Hair et al., 2019)*")

    badge("ok",
        "Recommendation: Use PAF with oblimin rotation for social science. "
        "Oblique rotation allows factors to correlate, which is realistic in behavioral research."
    )

    st.markdown("---")

    # Number of factors - BEFORE Run EFA
    st.subheader("Step 2: Number of Factors and Construct Names")
    constructs_preview = st.session_state.get("constructs", {})
    suggested_default  = len(constructs_preview) if constructs_preview else 2

    n_factors_setup = st.number_input(
        "How many factors to extract?",
        min_value=1,
        max_value=20,
        value=st.session_state.get("efa_n_factors_widget", suggested_default),
        step=1,
        key="efa_n_factors_widget",
        help="Default = number of constructs defined in Data Input. Adjust if needed."
    )
    st.caption(
        f"Default = {suggested_default} (number of constructs in Data Input). "
        "You can change this after seeing the scree plot — but that requires running EFA again."
    )

    st.markdown("---")

    if st.button("Run EFA via R/psych", type="primary", key="run_efa_btn"):
        try:
            from r_scripts.r_bridge import run_efa, check_r_available
            r_check = check_r_available()

            if not r_check["available"]:
                st.error(f"R is not available: {r_check['message']}")
                return

            n_factors_preview = st.session_state.get("efa_n_factors_widget", 2)

            with st.spinner("Running EFA via R/psych... (PAF extraction, parallel analysis)"):
                result = run_efa(
                    df          = df,
                    indicator_cols = indicator_cols,
                    n_factors   = n_factors_preview,
                    rotation    = rotation
                )

            if "error" in result:
                st.error(f"EFA failed: {result['error']}")
                return

            st.session_state["efa_result"]     = result
            st.session_state["efa_rotation"]   = rotation
            st.session_state["efa_data_cols"]  = indicator_cols
            st.success("EFA completed successfully via R/psych.")

        except Exception as e:
            st.error(f"EFA error: {str(e)}")
            return

    result = st.session_state.get("efa_result")
    if result is None:
        st.info("Configure settings above and click Run EFA to begin.")
        return

    st.markdown("---")
    factorable = render_factorability(result)
    if not factorable:
        badge("critical", "Data may not be suitable for factor analysis. Review KMO and Bartlett results.")
    st.markdown("---")

    n_factors = render_factor_number(result)
    st.markdown("---")

    # Re-run if n_factors changed
    stored_result = st.session_state.get("efa_result", {})
    stored_nf     = stored_result.get("n_factors")
    if isinstance(stored_nf, list): stored_nf = stored_nf[0]
    if stored_nf and int(stored_nf) != n_factors:
        badge("warning", f"You changed the number of factors to {n_factors}. Click Run EFA again to update.")

    loadings_mat = render_loadings_table(result, indicator_cols, n_factors)
    if loadings_mat is None:
        return
    st.markdown("---")

    factor_names = render_factor_naming(loadings_mat, n_factors)
    st.markdown("---")

    render_efa_summary(loadings_mat, n_factors, factor_names, indicator_cols)

    st.session_state["efa_complete"] = True
    st.markdown("---")
    badge("excellent", "EFA complete. Proceed to Confirmatory Factor Analysis (CFA).")
