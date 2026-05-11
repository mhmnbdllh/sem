"""
data_input.py
=============
Sprint 1 — Data Upload & Validation Module for SEM Studio.

Handles:
- CSV and Excel file upload
- Data type detection and validation
- Variable role assignment (indicator, demographic, etc.)
- Data preview and summary
- Session state management
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO


# ─── CONSTANTS ──────────────────────────────────────────────────

SUPPORTED_FORMATS = ["csv", "xlsx", "xls"]
MAX_FILE_SIZE_MB  = 200

VARIABLE_ROLES = {
    "indicator":    "📊 Indicator (questionnaire item)",
    "demographic":  "👤 Demographic / Control variable",
    "id":           "🔑 Respondent ID (exclude from analysis)",
    "exclude":      "🚫 Exclude from analysis",
}


# ─── FILE LOADING ───────────────────────────────────────────────

def load_file(uploaded_file) -> pd.DataFrame | None:
    """
    Load a CSV or Excel file into a DataFrame.

    Parameters
    ----------
    uploaded_file : streamlit UploadedFile

    Returns
    -------
    pd.DataFrame or None
    """
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".csv"):
            # Try multiple encodings
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("❌ Could not decode the CSV file. Please save it as UTF-8 and try again.")
            return None

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
            return df

        else:
            st.error(f"❌ Unsupported format: {name.split('.')[-1]}. Please upload CSV or Excel.")
            return None

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None


# ─── VALIDATION ─────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame) -> dict:
    """
    Run basic validation checks on the uploaded DataFrame.

    Returns
    -------
    dict with keys: valid (bool), warnings (list), errors (list), info (list)
    """
    errors   = []
    warnings = []
    info     = []

    n_rows, n_cols = df.shape

    # Minimum rows
    if n_rows < 30:
        errors.append(f"Dataset has only {n_rows} rows. SEM requires at least 200 observations.")
    elif n_rows < 100:
        warnings.append(f"Dataset has {n_rows} rows — below the minimum recommended for CFA (n ≥ 100).")
    elif n_rows < 200:
        warnings.append(f"Dataset has {n_rows} rows — adequate for basic CFA but below recommended for SEM (n ≥ 200).")
    else:
        info.append(f"✅ Sample size: n = {n_rows}")

    # Minimum columns
    if n_cols < 3:
        errors.append("Dataset has fewer than 3 variables. SEM requires multiple indicators per construct.")

    # Numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric  = [c for c in df.columns if c not in numeric_cols]

    if len(numeric_cols) == 0:
        errors.append("No numeric columns detected. Questionnaire items must be numeric (e.g., Likert scale).")
    else:
        info.append(f"✅ Numeric columns detected: {len(numeric_cols)}")

    if non_numeric:
        warnings.append(f"Non-numeric columns detected: {non_numeric}. These will be treated as demographic/ID variables.")

    # Duplicate rows
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"⚠️ {n_dupes} duplicate row(s) detected. Consider removing them.")

    # Constant columns (zero variance)
    const_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if const_cols:
        errors.append(f"Zero-variance columns detected (constant values): {const_cols}. These must be removed.")

    # Scale check (Likert detection)
    likert_cols = []
    for c in numeric_cols:
        unique_vals = sorted(df[c].dropna().unique())
        if len(unique_vals) <= 10 and df[c].min() >= 1 and df[c].max() <= 10:
            likert_cols.append(c)

    if likert_cols:
        info.append(f"✅ Likely Likert-scale items detected: {len(likert_cols)} columns")

    valid = len(errors) == 0

    return {
        "valid":        valid,
        "errors":       errors,
        "warnings":     warnings,
        "info":         info,
        "n_rows":       n_rows,
        "n_cols":       n_cols,
        "numeric_cols": numeric_cols,
        "non_numeric":  non_numeric,
        "likert_cols":  likert_cols,
    }


# ─── VARIABLE ROLE ASSIGNMENT ───────────────────────────────────

def render_variable_assignment(df: pd.DataFrame) -> dict:
    """
    Render an interactive UI for assigning roles to each variable.

    Returns
    -------
    dict: {column_name: role_key}
    """
    st.subheader("🏷️ Step 2: Assign Variable Roles")
    st.markdown(
        "Assign a role to each column in your dataset. "
        "**Indicators** are your questionnaire items (Likert-scale responses). "
        "You will group indicators into constructs in the next step."
    )

    numeric_cols    = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols= [c for c in df.columns if c not in numeric_cols]

    assignments = {}

    # Auto-detect ID columns
    id_hints = ["id", "respondent", "no", "number", "code", "resp"]

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Numeric Columns**")
        for col in numeric_cols:
            default_idx = 0  # indicator
            if any(hint in col.lower() for hint in id_hints):
                default_idx = 2  # id
            role = st.selectbox(
                label=col,
                options=list(VARIABLE_ROLES.keys()),
                format_func=lambda x: VARIABLE_ROLES[x],
                index=default_idx,
                key=f"role_{col}"
            )
            assignments[col] = role

    with col2:
        st.markdown("**Non-Numeric Columns**")
        for col in non_numeric_cols:
            default_idx = 1  # demographic
            if any(hint in col.lower() for hint in id_hints):
                default_idx = 2  # id
            role = st.selectbox(
                label=col,
                options=list(VARIABLE_ROLES.keys()),
                format_func=lambda x: VARIABLE_ROLES[x],
                index=default_idx,
                key=f"role_{col}"
            )
            assignments[col] = role

    # Summary
    indicators  = [c for c, r in assignments.items() if r == "indicator"]
    demographics= [c for c, r in assignments.items() if r == "demographic"]
    ids         = [c for c, r in assignments.items() if r == "id"]
    excluded    = [c for c, r in assignments.items() if r == "exclude"]

    st.markdown("---")
    st.markdown("**Assignment Summary:**")
    mcols = st.columns(4)
    mcols[0].metric("📊 Indicators",    len(indicators))
    mcols[1].metric("👤 Demographics",  len(demographics))
    mcols[2].metric("🔑 ID columns",    len(ids))
    mcols[3].metric("🚫 Excluded",       len(excluded))

    if len(indicators) < 3:
        st.warning("⚠️ You need at least 3 indicator variables to run CFA/SEM.")

    return assignments


# ─── CONSTRUCT DEFINITION ───────────────────────────────────────

def render_construct_definition(df: pd.DataFrame, assignments: dict) -> dict:
    """
    Render UI for grouping indicators into latent constructs.

    Returns
    -------
    dict: {construct_name: [list of indicator columns]}
    """
    st.subheader("🏗️ Step 3: Define Latent Constructs")
    st.markdown(
        "Group your indicator variables into **latent constructs** (factors). "
        "Each construct should have **at least 3 indicators** (ideally 4–5)."
    )

    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]

    if len(indicator_cols) < 3:
        st.error("❌ Not enough indicator variables. Assign at least 3 columns as 'Indicator'.")
        return {}

    n_constructs = st.number_input(
        "How many latent constructs do you have?",
        min_value=1, max_value=20, value=2, step=1
    )

    constructs = {}
    remaining  = indicator_cols.copy()

    for i in range(int(n_constructs)):
        st.markdown(f"**Construct {i+1}**")
        c1, c2 = st.columns([1, 3])

        with c1:
            name = st.text_input(
                f"Construct name",
                value=f"Construct_{i+1}",
                key=f"construct_name_{i}"
            )
        with c2:
            selected = st.multiselect(
                f"Select indicators for {name}",
                options=indicator_cols,
                key=f"construct_items_{i}"
            )

        if name and selected:
            constructs[name] = selected

    # Validation
    all_assigned = [item for items in constructs.values() for item in items]
    unassigned   = [c for c in indicator_cols if c not in all_assigned]

    if unassigned:
        st.warning(f"⚠️ Unassigned indicators: {unassigned}. Assign them to a construct or mark as 'Exclude'.")

    for cname, citems in constructs.items():
        if len(citems) < 3:
            st.warning(f"⚠️ **{cname}** has only {len(citems)} indicator(s). Minimum is 3 for reliable CFA.")

    return constructs


# ─── STRUCTURAL PATH DEFINITION ─────────────────────────────────

def render_structural_paths(constructs: dict) -> list:
    """
    Render UI for defining hypothesized structural paths between constructs.

    Returns
    -------
    list of tuples: [(predictor, outcome), ...]
    """
    st.subheader("➡️ Step 4: Define Structural Paths (Hypotheses)")
    st.markdown(
        "Specify the **directional relationships** between your latent constructs. "
        "Each path represents a hypothesis (e.g., H1: Motivation → Performance)."
    )

    construct_names = list(constructs.keys())

    if len(construct_names) < 2:
        st.warning("⚠️ You need at least 2 constructs to define structural paths.")
        return []

    n_paths = st.number_input("How many structural paths (hypotheses)?", min_value=1, max_value=30, value=1)

    paths = []
    for i in range(int(n_paths)):
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            predictor = st.selectbox(f"Predictor (H{i+1})", construct_names, key=f"pred_{i}")
        with c2:
            st.markdown("<br><center>→</center>", unsafe_allow_html=True)
        with c3:
            outcome_options = [c for c in construct_names if c != predictor]
            outcome = st.selectbox(f"Outcome (H{i+1})", outcome_options, key=f"out_{i}")

        if predictor and outcome:
            paths.append((predictor, outcome))

    return paths


# ─── MAIN RENDER FUNCTION ───────────────────────────────────────

def render_data_input():
    """Main render function for the Data Input page."""

    st.title("📂 Data Input & Model Setup")
    st.markdown(
        "Upload your dataset and define your measurement model. "
        "This is the foundation of your entire SEM analysis."
    )

    # ── Upload ──────────────────────────────────────────────────
    st.subheader("📁 Step 1: Upload Your Data")

    uploaded = st.file_uploader(
        label="Upload CSV or Excel file",
        type=SUPPORTED_FORMATS,
        help="Rows = respondents, Columns = questionnaire items (Likert scale). Max 200MB."
    )

    # Demo data option
    use_demo = st.checkbox("🎓 Use demo dataset (n=300, 3 constructs, 9 items)", value=False)

    if use_demo:
        df = _generate_demo_data()
        st.info("✅ Demo dataset loaded: 300 respondents, 3 constructs (Motivation, SelfEfficacy, Performance), 3 items each.")
    elif uploaded:
        df = load_file(uploaded)
        if df is None:
            return
    else:
        st.info("👆 Upload your data file above, or use the demo dataset to explore the app.")
        _render_data_format_guide()
        return

    # ── Preview ─────────────────────────────────────────────────
    with st.expander("👁️ Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Validation ──────────────────────────────────────────────
    st.subheader("🔍 Data Validation")
    validation = validate_dataframe(df)

    for msg in validation["info"]:
        st.success(msg)
    for msg in validation["warnings"]:
        st.warning(msg)
    for msg in validation["errors"]:
        st.error(msg)

    if not validation["valid"]:
        st.error("❌ Please fix the errors above before proceeding.")
        return

    st.success(f"✅ Data validation passed. Ready for analysis.")

    st.markdown("---")

    # ── Variable Assignment ──────────────────────────────────────
    assignments = render_variable_assignment(df)
    st.session_state["assignments"] = assignments

    st.markdown("---")

    # ── Construct Definition ─────────────────────────────────────
    constructs = render_construct_definition(df, assignments)
    st.session_state["constructs"] = constructs

    st.markdown("---")

    # ── Structural Paths ─────────────────────────────────────────
    if constructs:
        paths = render_structural_paths(constructs)
        st.session_state["structural_paths"] = paths

    st.markdown("---")

    # ── Mediator/Moderator ───────────────────────────────────────
    if constructs and len(constructs) >= 3:
        with st.expander("🔗 Optional: Define Mediator / Moderator Variables"):
            construct_names = list(constructs.keys())

            has_mediator = st.checkbox("Include mediation analysis?")
            if has_mediator:
                mediator = st.selectbox("Select mediator variable", construct_names, key="mediator_sel")
                st.session_state["mediator"] = mediator

            has_moderator = st.checkbox("Include moderation analysis?")
            if has_moderator:
                moderator = st.selectbox("Select moderator variable", construct_names, key="moderator_sel")
                st.session_state["moderator"] = moderator

    # ── Save & Confirm ───────────────────────────────────────────
    st.markdown("---")
    if constructs and st.button("✅ Confirm Model Setup & Proceed", type="primary", use_container_width=True):
        st.session_state["df"]          = df
        st.session_state["df_ready"]    = True
        st.session_state["validation"]  = validation
        st.balloons()
        st.success(
            "🎉 Model setup complete! Navigate to **Descriptive Statistics** in the sidebar to continue."
        )


# ─── DEMO DATA ──────────────────────────────────────────────────

def _generate_demo_data(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic demo dataset for SEM demo purposes."""
    rng = np.random.default_rng(seed)

    # Latent factors
    motivation    = rng.normal(0, 1, n)
    self_efficacy = 0.6 * motivation + rng.normal(0, 0.8, n)
    performance   = 0.5 * motivation + 0.4 * self_efficacy + rng.normal(0, 0.7, n)

    def likert(latent, noise=0.5):
        raw = latent + rng.normal(0, noise, n)
        return np.clip(np.round(raw * 0.8 + 3.5).astype(int), 1, 5)

    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "age":           rng.integers(20, 55, n),
        "gender":        rng.choice(["Male", "Female"], n),
        # Motivation items
        "MOT1": likert(motivation),
        "MOT2": likert(motivation),
        "MOT3": likert(motivation),
        # Self-Efficacy items
        "SE1":  likert(self_efficacy),
        "SE2":  likert(self_efficacy),
        "SE3":  likert(self_efficacy),
        # Performance items
        "PERF1": likert(performance),
        "PERF2": likert(performance),
        "PERF3": likert(performance),
    })

    # Introduce a small % of missing values (~3%)
    for col in ["MOT1", "SE2", "PERF3"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ─── FORMAT GUIDE ───────────────────────────────────────────────

def _render_data_format_guide():
    """Display the expected data format guide."""
    st.markdown("---")
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your data should be in **CSV or Excel** format with the following structure:

    | respondent_id | X1_1 | X1_2 | X1_3 | X2_1 | X2_2 | Y1_1 | Y1_2 | age | gender |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    | 1 | 4 | 5 | 3 | 4 | 5 | 3 | 4 | 25 | Male |
    | 2 | 3 | 4 | 4 | 5 | 4 | 4 | 5 | 30 | Female |

    **Rules:**
    - Each **row** = one respondent
    - Each **column** = one questionnaire item OR demographic variable
    - Questionnaire items should use **Likert scale** (e.g., 1–5 or 1–7)
    - No special characters in column names (use underscores instead of spaces)
    - Missing values should be left **blank** (not coded as -99 or 999)

    **Minimum requirements:**
    - n ≥ 200 for SEM (n ≥ 100 for CFA only)
    - At least 3 items per latent construct
    """)
