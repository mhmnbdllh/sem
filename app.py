"""
app.py
======
SEM Studio — Main Streamlit Application
Level 100 | Full-Featured SEM & CFA Analysis Suite

Sprint 1 : Data Input, Descriptive Statistics, Assumption Testing
Sprint 2 : Exploratory Factor Analysis (EFA), Confirmatory Factor Analysis (CFA)
Sprint 3 : Structural Equation Model (SEM), Mediation, Moderation
Sprint 4 : Measurement Invariance, Model Comparison
Sprint 5 : Path Diagram & Visualization, Export Report
Sprint 6 : Final app.py — all modules connected

Author  : SEM Studio
License : MIT
"""

# ── Page config — must be the very first Streamlit call ──────────
import streamlit as st

st.set_page_config(
    page_title  = "SEM Studio",
    page_icon   = "🧠",
    layout      = "wide",
    initial_sidebar_state = "expanded",
    menu_items  = {
        "Get Help"    : "https://github.com/YOUR_USERNAME/sem-studio",
        "Report a bug": "https://github.com/YOUR_USERNAME/sem-studio/issues",
        "About"       : "SEM Studio — Level 100 SEM & CFA Analysis Suite",
    },
)

# ── Module imports ────────────────────────────────────────────────
from modules.data_input       import render_data_input
from modules.descriptive      import render_descriptive
from modules.assumptions      import render_assumptions
from modules.efa              import render_efa
from modules.cfa              import render_cfa
from modules.sem_model        import render_sem
from modules.mediation        import render_mediation
from modules.moderation       import render_moderation
from modules.invariance       import render_invariance
from modules.model_comparison import render_model_comparison
from modules.visualization    import render_visualization
from modules.export           import render_export


# ── Global CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #2E86AB !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2E86AB, #1a5f7a);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 24px;
    font-size: 0.95rem;
}
.stTabs [data-baseweb="tab"] {
    background: #1a1d27;
    border-radius: 6px 6px 0 0;
    padding: 8px 18px;
}
.streamlit-expanderHeader {
    background: #1a1d27 !important;
    border-radius: 6px !important;
}
hr {
    border-color: #21262d !important;
    margin: 24px 0 !important;
}
.stProgress > div > div > div > div {
    background-color: #2E86AB;
}
</style>
""", unsafe_allow_html=True)


# ── Navigation definition ─────────────────────────────────────────
PAGES = {
    "🏠 Home":                         "home",

    "── SPRINT 1 ──":                  "divider",
    "📂 Data Input & Setup":           "data_input",
    "📊 Descriptive Statistics":       "descriptive",
    "🧪 Assumption Testing":           "assumptions",

    "── SPRINT 2 ──":                  "divider",
    "🔍 Exploratory Factor Analysis":  "efa",
    "📐 Confirmatory Factor Analysis": "cfa",

    "── SPRINT 3 ──":                  "divider",
    "🔗 Structural Model (SEM)":       "sem",
    "🔄 Mediation Analysis":           "mediation",
    "⚖️ Moderation Analysis":          "moderation",

    "── SPRINT 4 ──":                  "divider",
    "👥 Measurement Invariance":       "invariance",
    "📑 Model Comparison":             "model_comparison",

    "── SPRINT 5 ──":                  "divider",
    "🗺️ Path Diagram & Visuals":       "visualization",
    "📤 Export Report":                "export",
}

# page_key → render function
RENDER = {
    "data_input":       render_data_input,
    "descriptive":      render_descriptive,
    "assumptions":      render_assumptions,
    "efa":              render_efa,
    "cfa":              render_cfa,
    "sem":              render_sem,
    "mediation":        render_mediation,
    "moderation":       render_moderation,
    "invariance":       render_invariance,
    "model_comparison": render_model_comparison,
    "visualization":    render_visualization,
    "export":           render_export,
}


# ── Session state initialisation ─────────────────────────────────
def init_session_state():
    defaults = {
        "current_page":          "home",
        "df_ready":              False,
        "df":                    None,
        "assignments":           {},
        "constructs":            {},
        "structural_paths":      [],
        "recommended_estimator": "ML",
        "descriptive_complete":  False,
        "efa_complete":          False,
        "cfa_complete":          False,
        "sem_complete":          False,
        "cfa_fit":               {},
        "sem_fit":               {},
        "cfa_metrics":           {},
        "cfa_loadings":          {},
        "sem_paths":             [],
        "sem_r2":                [],
        "sem_endogenous":        [],
        "mediation_results":     None,
        "mediation_vars":        {},
        "moderation_results":    None,
        "moderation_vars":       {},
        "invariance_results":    None,
        "invariance_level":      None,
        "comparison_results":    {},
        "best_model":            None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Progress tracker ──────────────────────────────────────────────
def get_progress() -> dict:
    ss = st.session_state
    return {
        "data_input":       ss.get("df_ready", False),
        "descriptive":      ss.get("descriptive_complete", False),
        "assumptions":      ss.get("df_ready", False),
        "efa":              ss.get("efa_complete", False),
        "cfa":              ss.get("cfa_complete", False),
        "sem":              ss.get("sem_complete", False),
        "mediation":        bool(ss.get("mediation_results")),
        "moderation":       bool(ss.get("moderation_results")),
        "invariance":       bool(ss.get("invariance_results")),
        "model_comparison": bool(ss.get("comparison_results")),
        "visualization":    ss.get("df_ready", False),
        "export":           ss.get("df_ready", False),
    }


# ── Sidebar ───────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:

        # Logo block
        st.markdown("""
        <div style="text-align:center;padding:18px 0 10px">
            <div style="font-size:2.8rem">🧠</div>
            <div style="font-size:1.4rem;font-weight:800;
                        color:#2E86AB;letter-spacing:1px">SEM Studio by Muhaimin Abdullah</div>
            <div style="font-size:0.70rem;color:#555;margin-top:3px">
                Level 100 · Full SEM Suite
            </div>
        </div>
        <hr style="margin:8px 0;border-color:#21262d"/>
        """, unsafe_allow_html=True)

        # Dataset info pill
        if st.session_state.get("df_ready"):
            n   = len(st.session_state["df"])
            n_c = len(st.session_state.get("constructs", {}))
            n_p = len(st.session_state.get("structural_paths", []))
            st.markdown(
                f'<div style="background:#1a1d27;border-radius:6px;'
                f'padding:7px 10px;font-size:0.74rem;color:#aaa;'
                f'margin-bottom:8px;text-align:center">'
                f'n = <b style="color:#2E86AB">{n:,}</b> &nbsp;·&nbsp; '
                f'<b style="color:#2E86AB">{n_c}</b> constructs &nbsp;·&nbsp; '
                f'<b style="color:#2E86AB">{n_p}</b> paths</div>',
                unsafe_allow_html=True,
            )

        progress = get_progress()

        for label, key in PAGES.items():

            # Section header (divider)
            if key == "divider":
                st.markdown(
                    f'<div style="color:#444;font-size:0.64rem;font-weight:700;'
                    f'letter-spacing:1.5px;padding:10px 4px 2px">{label}</div>',
                    unsafe_allow_html=True,
                )
                continue

            done       = progress.get(key, False)
            is_current = st.session_state.get("current_page") == key
            dot        = "🟢 " if done else "⚪ "

            if is_current:
                # Highlight active page — render label styled, button still needed for rerun
                st.markdown(
                    f'<div style="background:#0d2233;border-left:3px solid #2E86AB;'
                    f'padding:5px 10px;border-radius:0 4px 4px 0;margin:1px 0;'
                    f'font-size:0.85rem;color:#2E86AB;font-weight:600">'
                    f'{dot}{label}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"{dot}{label}",
                    key=f"nav_{key}",
                    use_container_width=True,
                ):
                    st.session_state["current_page"] = key
                    st.rerun()

        # References
        st.markdown(
            "<hr style='border-color:#21262d;margin:12px 0'/>",
            unsafe_allow_html=True,
        )
        with st.expander("📚 References"):
            st.markdown("""
- Hair et al. (2019). *Multivariate Data Analysis*
- Kline (2016). *Principles & Practice of SEM*
- Brown (2015). *CFA for Applied Research*
- Hu & Bentler (1999). *Cutoff criteria for fit indexes*
- Fornell & Larcker (1981). *Evaluating SEM models*
- Henseler et al. (2015). *HTMT criterion*
- Hayes (2018). *Mediation & Moderation*
- Aiken & West (1991). *Multiple Regression*
- Vandenberg & Lance (2000). *Measurement Invariance*
- Burnham & Anderson (2002). *Model Selection*
            """)

        st.markdown(
            '<div style="text-align:center;color:#444;font-size:0.68rem;padding:6px 0">'
            'SEM Studio v1.0 · MIT License</div>',
            unsafe_allow_html=True,
        )


# ── Home page ─────────────────────────────────────────────────────
def render_home():

    st.markdown("""
    <div style="text-align:center;padding:48px 0 24px">
        <div style="font-size:4.5rem">🧠</div>
        <h1 style="font-size:3rem;font-weight:900;color:#2E86AB;margin:10px 0 6px">
            SEM Studio
        </h1>
        <p style="font-size:1.1rem;color:#888;max-width:640px;
                  margin:0 auto;line-height:1.7">
            A complete, methodologically rigorous suite for
            <strong style="color:#ccc">Structural Equation Modeling</strong> and
            <strong style="color:#ccc">Confirmatory Factor Analysis</strong> —
            built for researchers, academics, and advanced analysts
            in social science and education.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature grid (3 columns × 4 rows)
    card = (
        "background:#1a1d27;border-radius:12px;padding:18px;"
        "border:1px solid #21262d;margin-bottom:16px"
    )
    features = [
        ("📂", "Data Input & Validation",
         "Upload CSV/Excel, validate sample size, assign variable roles, "
         "define constructs and structural hypotheses."),
        ("📊", "Descriptives & Assumptions",
         "Full descriptive stats, missing value analysis, Mahalanobis outlier "
         "detection, Mardia's normality test, estimator recommendation."),
        ("🧪", "Assumption Testing",
         "Harman's single factor test (CMB), linearity assessment, "
         "and full methodological assumption checklist."),
        ("🔍", "Exploratory Factor Analysis",
         "KMO & Bartlett's test, parallel analysis, PAF/ML extraction, "
         "oblique/orthogonal rotation, cross-loading detection."),
        ("📐", "Confirmatory Factor Analysis",
         "Full fit indices (RMSEA, CFI, TLI, SRMR), AVE, HTMT, "
         "Fornell-Larcker, CR, Cronbach's α, McDonald's ω."),
        ("🔗", "Structural SEM",
         "Standardized path coefficients, hypothesis testing table, "
         "R² per endogenous construct, Cohen's f² effect sizes."),
        ("🔄", "Mediation Analysis",
         "Bootstrap 5,000 resamples, BCa CI, direct/indirect/total effects, "
         "VAF, full vs partial mediation determination (Zhao et al., 2010)."),
        ("⚖️", "Moderation Analysis",
         "Mean-centered interaction terms, simple slope analysis, "
         "interaction plots, ΔR², enhancing/buffering pattern detection."),
        ("👥", "Measurement Invariance",
         "Configural, metric, and scalar invariance testing, "
         "Δχ², ΔCFI, practical implications for group comparisons."),
        ("📑", "Model Comparison",
         "Rival model estimation, AIC/BIC + Akaike weights, "
         "Δχ² for nested models, automatic model selection recommendation."),
        ("🗺️", "Path Diagram & Visuals",
         "Interactive Plotly path diagram with β, loadings, R², "
         "fit index dashboard, effect size bar chart, correlation heatmap."),
        ("📤", "Export Report",
         "APA 7th edition narrative, Excel workbook (9 sheets), "
         "PDF report, full methodological checklist."),
    ]

    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f'<div style="{card}">'
                f'<span style="font-size:1.6rem">{icon}</span>'
                f'<h4 style="color:#2E86AB;margin:8px 0 5px;font-size:0.95rem">{title}</h4>'
                f'<p style="color:#777;font-size:0.82rem;line-height:1.5;margin:0">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Analysis pipeline progress
    st.subheader("🗺️ Your Analysis Pipeline")
    progress = get_progress()
    keys     = list(RENDER.keys())
    icons    = ["📂","📊","🧪","🔍","📐","🔗","🔄","⚖️","👥","📑","🗺️","📤"]
    names    = ["Data","Desc.","Assume.","EFA","CFA","SEM",
                "Mediation","Moderation","Invariance","Compare","Diagram","Export"]

    cols = st.columns(len(keys))
    for i, key in enumerate(keys):
        done = progress.get(key, False)
        with cols[i]:
            st.markdown(
                f'<div style="text-align:center;padding:4px 0">'
                f'<div style="font-size:1.4rem">{icons[i]}</div>'
                f'<div style="font-size:0.62rem;color:#2E86AB;font-weight:700">{names[i]}</div>'
                f'<div style="font-size:0.7rem">{"🟢" if done else "⚪"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    completed = sum(1 for v in progress.values() if v)
    total     = len(progress)
    pct       = completed / total
    st.progress(pct)
    st.caption(f"{completed}/{total} steps completed ({pct:.0%})")

    st.markdown("---")

    # Quick start + requirements
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🚀 Quick Start")
        st.markdown("""
1. Click **📂 Data Input & Setup** in the sidebar
2. Upload your CSV/Excel file (or use the built-in **demo dataset**)
3. Assign variable roles and define your latent constructs
4. Define your structural hypotheses (directed paths)
5. Follow the pipeline step by step — every output includes
   **automatic interpretation** in plain language

All standards follow *Hair et al. (2019)*, *Kline (2016)*,
*Hu & Bentler (1999)*, and *Fornell & Larcker (1981)*.
        """)

    with col2:
        st.subheader("📋 Requirements")
        st.markdown("""
**Data format:**
- CSV or Excel (.xlsx)
- Rows = respondents
- Columns = Likert items (1–5 or 1–7)

**Minimum sample:**
- n ≥ 100 for CFA
- n ≥ 200 for SEM

**Per construct:**
- ≥ 3 indicators (ideally 4–5)
        """)

    st.markdown("---")
    cols = st.columns([1, 2, 1])
    with cols[1]:
        if st.button(
            "🚀  Get Started  →  Data Input & Setup",
            type="primary",
            use_container_width=True,
            key="home_cta",
        ):
            st.session_state["current_page"] = "data_input"
            st.rerun()


# ── Main router ───────────────────────────────────────────────────
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
