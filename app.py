"""
app.py
======
SEM Studio - Main Streamlit Application
Level 100 | Full-Featured SEM & CFA Analysis Suite
Backend: R/lavaan | Frontend: Streamlit

Sprint 1 : Data Input, Descriptive Statistics, Assumption Testing
Sprint 2 : EFA, CFA
Sprint 3 : Structural Model, Mediation, Moderation
Sprint 4 : Measurement Invariance, Model Comparison
Sprint 5 : Path Diagram, Export Report
"""

import streamlit as st

st.set_page_config(
    page_title="SEM Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help":     "https://github.com/YOUR_USERNAME/sem",
        "Report a bug": "https://github.com/YOUR_USERNAME/sem/issues",
        "About":        "SEM Studio — Level 100 SEM & CFA Suite powered by R/lavaan",
    },
)

from modules.data_input       import render_data_input
from modules.descriptive      import render_descriptive
from modules.efa              import render_efa
from modules.cfa              import render_cfa
from modules.sem_model        import render_sem
from modules.mediation        import render_mediation
from modules.moderation       import render_moderation
from modules.invariance       import render_invariance
from modules.model_comparison import render_model_comparison
from modules.visualization    import render_visualization
from modules.export           import render_export
from modules.pls              import render_pls

# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', Arial, sans-serif;
}

/* Metric values */
[data-testid="stMetricValue"] {
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #1a6fa8 !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2E86AB, #1a5f7a);
    color: white;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 10px 20px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #dde3ea;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #f0f4f8 !important;
    border-radius: 6px !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: #f0f4f8;
    border-radius: 6px 6px 0 0;
    color: #1a1a1a;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background-color: #2E86AB !important;
    color: white !important;
}

/* Dividers */
hr {
    border-color: #dde3ea !important;
    margin: 20px 0 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #2E86AB;
}

/* Info/warning/error boxes */
.stAlert {
    color: #1a1a1a !important;
}

/* DataFrame tables */
[data-testid="stDataFrame"] {
    border-radius: 6px;
    overflow: hidden;
}

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton > button {
    text-align: left !important;
    background: transparent !important;
    border: none !important;
    color: #1a1a1a !important;
    padding: 6px 8px !important;
    font-size: 0.88rem !important;
    width: 100% !important;
    border-radius: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #e8f4fc !important;
    color: #1a6fa8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Navigation ───────────────────────────────────────────────────
PAGES = {
    "🏠 Home":                          "home",
    "── SETUP (shared) ──":             "divider",
    "📂 Data Input and Setup":          "data_input",
    "📊 Descriptive Statistics":        "descriptive",
    "── CB-SEM (lavaan) ──":            "divider",
    "🔍 EFA":                           "efa",
    "📐 CFA":                           "cfa",
    "🔗 SEM (Structural Model)":        "sem",
    "🔄 Mediation Analysis":            "mediation",
    "⚖️ Moderation Analysis":           "moderation",
    "👥 Measurement Invariance":        "invariance",
    "📑 Model Comparison":              "model_comparison",
    "🗺️ Path Diagram":                  "visualization",
    "── PLS-SEM (separate method) ──":  "divider",
    "🔬 PLS-SEM Analysis":              "pls",
    "── REPORT ──":                     "divider",
    "📤 Export Report":                 "export",
}

RENDER = {
    "data_input":       render_data_input,
    "descriptive":      render_descriptive,
    "efa":              render_efa,
    "cfa":              render_cfa,
    "sem":              render_sem,
    "mediation":        render_mediation,
    "moderation":       render_moderation,
    "invariance":       render_invariance,
    "model_comparison": render_model_comparison,
    "visualization":    render_visualization,
    "pls":              render_pls,
    "export":           render_export,
}

# ── Session State ────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "current_page":          "home",
        "df_ready":              False,
        "df":                    None,
        "assignments":           {},
        "constructs":            {},
        "structural_paths":      [],
        "recommended_estimator": "MLR",
        "descriptive_complete":  False,
        "efa_complete":          False,
        "cfa_complete":          False,
        "sem_complete":          False,
        "cfa_syntax":            "",
        "sem_syntax":            "",
        "cfa_fit":               {},
        "sem_fit":               {},
        "cfa_metrics":           {},
        "cfa_loadings":          {},
        "cfa_result":            None,
        "sem_result":            None,
        "sem_paths":             [],
        "sem_r2":                [],
        "sem_endogenous":        [],
        "sem_exogenous":         [],
        "mediation_results":     None,
        "mediation_vars":        {},
        "moderation_results":    None,
        "moderation_vars":       {},
        "invariance_results":    None,
        "invariance_level":      None,
        "comparison_results":    {},
        "best_model":            None,
        "advanced_options":      {},
        "efa_result":            None,
        "efa_factor_names":      {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Progress tracker ─────────────────────────────────────────────
def get_progress():
    ss = st.session_state
    return {
        "data_input":       ss.get("df_ready", False),
        "descriptive":      ss.get("descriptive_complete", False),
        "efa":              ss.get("efa_complete", False),
        "cfa":              ss.get("cfa_complete", False),
        "sem":              ss.get("sem_complete", False),
        "mediation":        bool(ss.get("mediation_results")),
        "moderation":       bool(ss.get("moderation_results")),
        "invariance":       bool(ss.get("invariance_results")),
        "model_comparison": bool(ss.get("comparison_results")),
        "visualization":    ss.get("df_ready", False),
        "pls":              bool(ss.get("pls_complete")),
        "export":           ss.get("df_ready", False),
    }


# ── Sidebar ──────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="text-align:center;padding:16px 0 10px">
            <div style="font-size:2.4rem">🧠</div>
            <div style="font-size:1.3rem;font-weight:800;color:#1a6fa8">
                SEM Studio
            </div>
            <div style="font-size:0.7rem;color:#888;margin-top:2px">
                Level 100 · R/lavaan Backend
            </div>
        </div>
        <hr style="margin:8px 0;border-color:#dde3ea"/>
        """, unsafe_allow_html=True)

        # Dataset info
        if st.session_state.get("df_ready"):
            n   = len(st.session_state["df"])
            n_c = len(st.session_state.get("constructs", {}))
            n_p = len(st.session_state.get("structural_paths", []))
            st.markdown(
                f'<div style="background:#e8f4fc;border-radius:6px;'
                f'padding:7px 10px;font-size:0.75rem;color:#1a6fa8;'
                f'margin-bottom:8px;text-align:center;border:1px solid #bee3f8">'
                f'n = <b>{n:,}</b> &nbsp;·&nbsp; '
                f'<b>{n_c}</b> constructs &nbsp;·&nbsp; '
                f'<b>{n_p}</b> paths</div>',
                unsafe_allow_html=True,
            )

        progress = get_progress()

        # Two SEPARATE method tracks. Next-step suggestion only applies
        # within whichever track the user has already started.
        # This prevents pointing PLS-SEM users back toward CB-SEM steps.
        cbsem_track = ["data_input", "descriptive", "efa", "cfa", "sem"]
        pls_track   = ["data_input", "descriptive", "pls"]

        started_cbsem = any(progress.get(s) for s in ["efa","cfa","sem"])
        started_pls   = bool(progress.get("pls"))

        if started_pls and not started_cbsem:
            active_track = pls_track
        elif started_cbsem and not started_pls:
            active_track = cbsem_track
        else:
            # Neither started, or both started: don't force a direction.
            # Only suggest the shared setup steps.
            active_track = ["data_input", "descriptive"]

        next_step = None
        for step in active_track:
            if not progress.get(step, False):
                next_step = step
                break

        # Step labels reflecting the two separate tracks
        step_labels = {
            "data_input":       "Setup Step 1 (shared)",
            "descriptive":      "Setup Step 2 (shared)",
            "efa":              "CB-SEM — optional",
            "cfa":              "CB-SEM — required",
            "sem":              "CB-SEM — required",
            "mediation":        "CB-SEM — optional",
            "moderation":       "CB-SEM — optional",
            "invariance":       "CB-SEM — optional",
            "model_comparison": "CB-SEM — optional",
            "visualization":    "CB-SEM — optional",
            "pls":              "PLS-SEM — separate method",
            "export":           "Final step (either method)",
        }

        for label, key in PAGES.items():
            if key == "divider":
                st.markdown(
                    f'<div style="color:#888;font-size:0.65rem;font-weight:700;'
                    f'letter-spacing:1.5px;padding:10px 4px 2px;'
                    f'text-transform:uppercase">{label}</div>',
                    unsafe_allow_html=True,
                )
                continue

            done       = progress.get(key, False)
            is_current = st.session_state.get("current_page") == key
            is_next    = key == next_step
            step_label = step_labels.get(key, "")

            if done:
                icon = "✅"
                color = "#1a7a4a"
                bg = "#f0fff4"
                border = "#1a7a4a"
            elif is_next:
                icon = "👉"
                color = "#1a6fa8"
                bg = "#e8f4fc"
                border = "#2E86AB"
            else:
                icon = "⚪"
                color = "#888"
                bg = "transparent"
                border = "transparent"

            if is_current:
                st.markdown(
                    f'<div style="background:#e8f4fc;border-left:3px solid #2E86AB;'
                    f'padding:6px 10px;border-radius:0 4px 4px 0;margin:1px 0">'
                    f'<div style="font-size:0.88rem;color:#1a6fa8;font-weight:700">{icon} {label}</div>'
                    f'<div style="font-size:0.68rem;color:#888">{step_label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"{icon} {label}",
                    key=f"nav_{key}",
                    use_container_width=True,
                    help=step_label,
                ):
                    st.session_state["current_page"] = key
                    st.rerun()

        # Show next step guidance
        if next_step and next_step != "data_input":
            next_label = [l for l,k in PAGES.items() if k == next_step]
            if next_label:
                st.markdown(
                    f'<div style="background:#fff8e1;border:1px solid #ffd54f;'
                    f'border-radius:6px;padding:8px 10px;margin-top:8px;font-size:0.75rem;color:#7a6000">'
                    f'👉 <b>Next:</b> {next_label[0]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<hr style='border-color:#dde3ea;margin:12px 0'/>", unsafe_allow_html=True)

        with st.expander("References"):
            st.markdown(
                "- Hair et al. (2019). Multivariate Data Analysis  \n"
                "- Kline (2016). Principles and Practice of SEM  \n"
                "- Brown (2015). CFA for Applied Research  \n"
                "- Rosseel (2012). lavaan: R Package for SEM  \n"
                "- Hu & Bentler (1999). Cutoff criteria for fit  \n"
                "- Fornell & Larcker (1981). Evaluating SEM  \n"
                "- Henseler et al. (2015). HTMT criterion  \n"
                "- Hayes (2018). Mediation and Moderation  \n"
                "- Aiken & West (1991). Multiple Regression  \n"
                "- Vandenberg & Lance (2000). MI literature  \n"
                "- Burnham & Anderson (2002). Model Selection"
            )

        st.markdown(
            '<div style="text-align:center;color:#aaa;font-size:0.68rem;padding:6px 0">'
            'SEM Studio v2.0 · R/lavaan Backend · MIT</div>',
            unsafe_allow_html=True,
        )


# ── Home Page ────────────────────────────────────────────────────
def render_home():
    st.markdown("""
    <div style="text-align:center;padding:40px 0 20px">
        <div style="font-size:4rem">🧠</div>
        <h1 style="font-size:2.8rem;font-weight:900;color:#1a6fa8;margin:8px 0">
            SEM Studio
        </h1>
        <p style="font-size:1.05rem;color:#555;max-width:640px;
                  margin:0 auto;line-height:1.7">
            A complete, methodologically rigorous suite for
            <strong>Structural Equation Modeling</strong> and
            <strong>Confirmatory Factor Analysis</strong>.
            Powered by <strong>R/lavaan</strong> — the gold standard for SEM.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature cards
    card_style = (
        "background:#f8fafc;border-radius:10px;padding:18px;"
        "border:1px solid #dde3ea;margin-bottom:16px"
    )
    features = [
        ("📂", "Data Input",
         "Upload CSV/Excel, validate data, assign variable roles, define constructs and hypotheses. Built-in demo dataset included."),
        ("📊", "Descriptive Statistics",
         "Full descriptive stats, missing value analysis, Mahalanobis outlier detection, Mardia's normality test via R/psych."),
        ("🔍", "EFA",
         "PAF extraction, parallel analysis, oblimin/varimax/promax rotation — all via R/psych. Methodologically correct."),
        ("📐", "CFA",
         "Full lavaan CFA with fit indices (RMSEA, CFI, TLI, SRMR), AVE, HTMT, Fornell-Larcker, CR, Cronbach's alpha."),
        ("🔗", "Structural SEM",
         "Full lavaan SEM with standardized path coefficients, R2, Cohen's f2 effect sizes, hypothesis testing table."),
        ("🔄", "Mediation",
         "Bootstrap (5,000 resamples) via lavaan. BCa CI, VAF, full/partial/no mediation determination (Zhao et al., 2010)."),
        ("⚖️", "Moderation",
         "Mean-centered interaction via R. Simple slope analysis, interaction plots, enhancing/buffering pattern detection."),
        ("👥", "Measurement Invariance",
         "Configural, metric, scalar invariance via lavaan. ΔCFI and ΔRMSEA difference tests with practical implications."),
        ("📑", "Model Comparison",
         "Rival model estimation via lavaan. AIC/BIC, Akaike weights, Δchi2 for nested models, automatic recommendation."),
        ("🗺️", "Path Diagram",
         "Interactive Plotly path diagram with loadings, beta coefficients, R2, color-coded significance."),
        ("📤", "Export",
         "APA 7th edition narrative, Excel workbook (9 sheets), methodological checklist."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div style="{card_style}">'
                f'<span style="font-size:1.6rem">{icon}</span>'
                f'<h4 style="color:#1a6fa8;margin:8px 0 5px;font-size:0.95rem">{title}</h4>'
                f'<p style="color:#555;font-size:0.83rem;line-height:1.5;margin:0">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Explicit method choice — shown prominently BEFORE the sidebar order
    # implies any sequence. Sidebar position (CB-SEM listed above PLS-SEM)
    # is purely alphabetical/visual and does NOT mean PLS-SEM comes "after".
    if not st.session_state.get("df_ready"):
        st.subheader("Choose Your Method First")
        st.markdown(
            "**CB-SEM and PLS-SEM are two independent methods.** "
            "Their position in the sidebar does not imply an order — "
            "pick the one that fits your research goal."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(
                "**CB-SEM (lavaan)**\n\n"
                "Theory confirmation, n ≥ 100, reflective constructs only.\n\n"
                "Path: Data Input → Descriptive → EFA *(optional)* → CFA → SEM"
            )
        with col_b:
            st.info(
                "**PLS-SEM**\n\n"
                "Prediction focus, small n OK, formative constructs allowed.\n\n"
                "Path: Data Input → Descriptive → PLS-SEM (skip EFA/CFA/SEM entirely)"
            )
        st.markdown("---")

    # CB-SEM vs PLS-SEM guide
    with st.expander("📖 Analysis Flow Guide — CB-SEM vs PLS-SEM", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**CB-SEM Flow (lavaan) — Theory Confirmation:**
1. Data Input
2. Descriptive Statistics
3. EFA *(optional)*
4. CFA
5. SEM
6. Mediation / Moderation *(optional)*
7. Export Report

✅ Use when: testing theory, n ≥ 100, normal-ish data
            """)
        with col2:
            st.markdown("""
**PLS-SEM Flow — Prediction:**
1. Data Input
2. Descriptive Statistics
3. PLS-SEM *(skip EFA/CFA/SEM)*
4. Export Report

✅ Use when: prediction focus, small n (≥ 30), non-normal data

⚠️ CB-SEM and PLS-SEM results are **not directly comparable**.
Quick Text and HTML Report show each method separately.
            """)

    st.markdown("---")

    # Pipeline progress — CB-SEM and PLS-SEM tracked SEPARATELY
    # These are two distinct statistical methods (lavaan vs composite-based PLS)
    # and must never be presented as one linear sequence or one combined percentage.
    st.subheader("Your Analysis Pipeline")
    progress = get_progress()

    st.markdown("**Shared Setup:**")
    setup_keys  = ["data_input", "descriptive"]
    setup_icons = ["📂", "📊"]
    setup_names = ["Data", "Desc"]
    cols = st.columns(len(setup_keys))
    for i, key in enumerate(setup_keys):
        done = progress.get(key, False)
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center;padding:4px 0">'
                f'<div style="font-size:1.4rem">{setup_icons[i]}</div>'
                f'<div style="font-size:0.62rem;color:#1a6fa8;font-weight:700">{setup_names[i]}</div>'
                f'<div style="font-size:0.7rem">{"🟢" if done else "⚪"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    col_cb, col_pls = st.columns(2)

    with col_cb:
        st.markdown("**CB-SEM Track (lavaan):**")
        cb_keys  = ["efa", "cfa", "sem", "mediation", "moderation", "invariance", "model_comparison", "visualization"]
        cb_icons = ["🔍", "📐", "🔗", "🔄", "⚖️", "👥", "📑", "🗺️"]
        cb_names = ["EFA", "CFA", "SEM", "Mediation", "Moderation", "Invariance", "Compare", "Diagram"]
        sub_cols = st.columns(4)
        for i, key in enumerate(cb_keys):
            done = progress.get(key, False)
            with sub_cols[i % 4]:
                st.markdown(
                    f'<div style="text-align:center;padding:4px 0">'
                    f'<div style="font-size:1.2rem">{cb_icons[i]}</div>'
                    f'<div style="font-size:0.58rem;color:#1a6fa8;font-weight:700">{cb_names[i]}</div>'
                    f'<div style="font-size:0.65rem">{"🟢" if done else "⚪"}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        cb_required = ["cfa", "sem"]
        cb_done = sum(1 for k in cb_required if progress.get(k))
        st.caption(f"Required steps: {cb_done}/{len(cb_required)} (CFA, SEM)")

    with col_pls:
        st.markdown("**PLS-SEM Track (separate method):**")
        done_pls = progress.get("pls", False)
        st.markdown(
            f'<div style="text-align:center;padding:20px 0">'
            f'<div style="font-size:2rem">🔬</div>'
            f'<div style="font-size:0.8rem;color:#1a6fa8;font-weight:700">PLS-SEM Analysis</div>'
            f'<div style="font-size:0.9rem">{"🟢 Complete" if done_pls else "⚪ Not started"}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "PLS-SEM does not require EFA/CFA/SEM. "
            "Go directly to PLS-SEM in the sidebar if this is your chosen method."
        )

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Quick Start")
        st.markdown("""
1. Click **📂 Data Input and Setup** in the sidebar
2. Upload your CSV/Excel file or use the built-in **demo dataset**
3. Assign variable roles and define your latent constructs
4. Define your structural hypotheses (paths)
5. Follow the pipeline step by step

Every output includes **automatic plain-language interpretation** following
Hair et al. (2019), Kline (2016), Hu & Bentler (1999), and Fornell & Larcker (1981).

All statistical computations use **R/lavaan** — the same engine used in top academic journals.
        """)

    with col2:
        st.subheader("Data Requirements")
        st.markdown("""
**Format:** CSV or Excel

**Minimum sample:**
- n ≥ 100 for CFA
- n ≥ 200 for SEM

**Per construct:**
- Minimum 3 indicators
- Ideally 4–5 indicators

**Scale:** Likert 1–5 or 1–7

**Missing:** Leave blank (not -99)
        """)

    st.markdown("---")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button(
            "🚀  Get Started  →  Data Input and Setup",
            type="primary",
            use_container_width=True,
            key="home_cta",
        ):
            st.session_state["current_page"] = "data_input"
            st.rerun()


# ── Main Router ───────────────────────────────────────────────────
def main():
    init_session_state()
    render_sidebar()

    page = st.session_state.get("current_page", "home")

    if page == "home":
        render_home()
    elif page in RENDER:
        RENDER[page]()
    else:
        render_home()


if __name__ == "__main__":
    main()
