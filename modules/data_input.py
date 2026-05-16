"""
data_input.py - Data Input & Model Setup Module for SEM Studio.
"""
import streamlit as st
import pandas as pd
import numpy as np

VARIABLE_ROLES = {
    "indicator":   "Indicator (questionnaire item)",
    "demographic": "Demographic / Control variable",
    "id":          "Respondent ID (exclude from analysis)",
    "exclude":     "Exclude from analysis",
}

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

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc)
                except UnicodeDecodeError:
                    continue
            st.error("Cannot decode CSV. Please save as UTF-8.")
            return None
        elif name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported format. Please upload CSV or Excel.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def generate_demo_data(n=300, seed=42):
    rng = np.random.default_rng(seed)
    motivation    = rng.normal(0, 1, n)
    self_efficacy = 0.55 * motivation + rng.normal(0, 0.84, n)
    performance   = 0.40 * motivation + 0.45 * self_efficacy + rng.normal(0, 0.71, n)
    def to_likert(latent, noise=0.5):
        raw = latent + rng.normal(0, noise, n)
        return np.clip(np.round(raw * 0.9 + 3.5).astype(int), 1, 5)
    df = pd.DataFrame({
        "respondent_id": range(1, n + 1),
        "age":           rng.integers(20, 56, n),
        "gender":        rng.choice(["Male", "Female"], n),
        "MOT1":          to_likert(motivation),
        "MOT2":          to_likert(motivation),
        "MOT3":          to_likert(motivation),
        "SE1":           to_likert(self_efficacy),
        "SE2":           to_likert(self_efficacy),
        "SE3":           to_likert(self_efficacy),
        "PERF1":         to_likert(performance),
        "PERF2":         to_likert(performance),
        "PERF3":         to_likert(performance),
    })
    for col in ["MOT2", "SE1", "PERF3"]:
        mask = rng.random(n) < 0.03
        df.loc[mask, col] = np.nan
    return df

def validate_dataframe(df):
    errors, warnings, info = [], [], []
    n_rows, n_cols = df.shape
    if n_rows < 30:
        errors.append(f"Only {n_rows} rows. SEM requires at least 200 observations.")
    elif n_rows < 100:
        warnings.append(f"n = {n_rows} is below minimum for CFA (n >= 100).")
    elif n_rows < 200:
        warnings.append(f"n = {n_rows} is adequate for CFA but below recommended for SEM (n >= 200).")
    else:
        info.append(f"Sample size: n = {n_rows}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        errors.append("No numeric columns detected.")
    else:
        info.append(f"{len(numeric_cols)} numeric columns detected.")
    const_cols = [c for c in numeric_cols if df[c].nunique() <= 1]
    if const_cols:
        errors.append(f"Zero-variance columns: {const_cols}. Must be removed.")
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        warnings.append(f"{n_dupes} duplicate row(s) detected.")
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "info": info,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_cols": numeric_cols,
    }

def render_data_upload():
    st.subheader("Step 1: Upload Your Data")

    # If data already loaded in this session, show it without requiring re-upload
    cached_df = st.session_state.get("df")
    if cached_df is not None and st.session_state.get("df_ready"):
        st.success(f"Data already loaded: {cached_df.shape[0]:,} rows × {cached_df.shape[1]} columns.")
        if st.button("Upload Different Dataset", key="reupload_btn"):
            for key in ["df", "df_ready", "constructs", "structural_paths", "assignments",
                        "cfa_result", "cfa_fit", "cfa_loadings", "cfa_metrics",
                        "sem_result", "sem_fit", "sem_paths", "sem_r2",
                        "mediation_results", "moderation_results", "invariance_results",
                        "efa_result", "comparison_results"]:
                st.session_state.pop(key, None)
            st.rerun()
        return cached_df

    use_demo = st.checkbox("Use demo dataset (n=300, 3 constructs, 9 items)", value=False, key="use_demo_checkbox")
    if use_demo:
        df = generate_demo_data()
        st.success("Demo dataset loaded: 300 respondents, 3 constructs (Motivation, SelfEfficacy, Performance), 3 items each.")
        return df
    uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"],
                                 help="Rows = respondents. Columns = questionnaire items.")
    if uploaded:
        df = load_file(uploaded)
        if df is not None:
            st.session_state["df"] = df
        return df
    with st.expander("Expected Data Format"):
        st.markdown("""
- Each row = one respondent
- Each column = one questionnaire item or demographic variable
- Items should use Likert scale (e.g., 1-5 or 1-7)
- Leave missing values blank
- Minimum: n >= 200 for SEM, n >= 100 for CFA, at least 3 items per construct
        """)
    return None

def render_variable_assignment(df):
    st.subheader("Step 2: Assign Variable Roles")
    st.markdown("Assign a role to each column. **Indicators** are your Likert-scale questionnaire items.")
    numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]
    id_hints         = ["id", "respondent", "no", "number", "code", "resp"]
    assignments = {}
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric Columns**")
        for col in numeric_cols:
            default = 2 if any(h in col.lower() for h in id_hints) else 0
            # Restore from session state if available
            saved_role = st.session_state.get(f"role_{col}")
            if saved_role is not None and saved_role in VARIABLE_ROLES:
                default = list(VARIABLE_ROLES.keys()).index(saved_role)
            role = st.selectbox(
                label=col,
                options=list(VARIABLE_ROLES.keys()),
                format_func=lambda x: VARIABLE_ROLES[x],
                index=default,
                key=f"role_{col}",
            )
            assignments[col] = role
    with col2:
        st.markdown("**Non-Numeric Columns**")
        for col in non_numeric_cols:
            default = 2 if any(h in col.lower() for h in id_hints) else 1
            saved_role = st.session_state.get(f"role_{col}")
            if saved_role is not None and saved_role in VARIABLE_ROLES:
                default = list(VARIABLE_ROLES.keys()).index(saved_role)
            role = st.selectbox(
                label=col,
                options=list(VARIABLE_ROLES.keys()),
                format_func=lambda x: VARIABLE_ROLES[x],
                index=default,
                key=f"role_{col}",
            )
            assignments[col] = role
    indicators   = [c for c, r in assignments.items() if r == "indicator"]
    demographics = [c for c, r in assignments.items() if r == "demographic"]
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indicators",   len(indicators))
    c2.metric("Demographics", len(demographics))
    c3.metric("ID columns",   sum(1 for r in assignments.values() if r == "id"))
    c4.metric("Excluded",     sum(1 for r in assignments.values() if r == "exclude"))
    if len(indicators) < 3:
        st.warning("You need at least 3 indicator variables to run CFA/SEM.")
    return assignments

def render_construct_definition(df, assignments):
    st.subheader("Step 3: Define Latent Constructs")
    st.markdown("Group your indicator variables into latent constructs. Each needs at least 3 indicators.")
    indicator_cols = [c for c, r in assignments.items() if r == "indicator"]
    if len(indicator_cols) < 3:
        st.error("Not enough indicator variables. Assign at least 3 columns as Indicator.")
        return {}
    n_constructs = st.number_input("How many latent constructs?", min_value=1, max_value=20, value=2, step=1, key="n_constructs_input")
    constructs = {}
    for i in range(int(n_constructs)):
        c1, c2 = st.columns([1, 3])
        with c1:
            default_name = f"Construct{i+1}"
        if f"construct_name_{i}" not in st.session_state:
            st.session_state[f"construct_name_{i}"] = default_name
        name = st.text_input(f"Construct {i+1} name", key=f"construct_name_{i}")
        with c2:
            selected = st.multiselect(f"Indicators for {name}", options=indicator_cols, key=f"construct_items_{i}")
        if name and selected:
            constructs[name] = selected
            if len(selected) < 3:
                st.warning(f"{name} has only {len(selected)} indicator(s). Minimum is 3 for reliable CFA.")
    assigned = [item for items in constructs.values() for item in items]
    unassigned = [c for c in indicator_cols if c not in assigned]
    if unassigned:
        st.warning(f"Unassigned indicators: {', '.join(unassigned)}")
    return constructs

def render_structural_paths(constructs):
    st.subheader("Step 4: Define Structural Paths (Hypotheses)")
    st.markdown("Specify the directional relationships between latent constructs.")
    construct_names = list(constructs.keys())
    if len(construct_names) < 2:
        st.warning("You need at least 2 constructs to define structural paths.")
        return []

    # Restore saved paths
    saved_paths = st.session_state.get("structural_paths", [])
    saved_n     = len(saved_paths) if saved_paths else 1

    n_paths = st.number_input("How many structural paths (hypotheses)?", min_value=1, max_value=30, value=1, step=1, key="n_paths_input")
    paths   = []
    for i in range(int(n_paths)):
        c1, c2, c3 = st.columns([2, 1, 2])

        # Restore saved pred/out for this path
        saved_pred = saved_paths[i][0] if i < len(saved_paths) else construct_names[0]
        saved_out  = saved_paths[i][1] if i < len(saved_paths) else (construct_names[1] if len(construct_names) > 1 else construct_names[0])

        with c1:
            pred = st.selectbox(f"Predictor (H{i+1})", construct_names, key=f"pred_{i}")
        with c2:
            st.markdown("<br><div style='text-align:center;font-size:1.4rem'>to</div>", unsafe_allow_html=True)
        with c3:
            out_opts = [c for c in construct_names if c != pred]
            if not out_opts:
                continue
            out = st.selectbox(f"Outcome (H{i+1})", out_opts, key=f"out_{i}")
        if pred and out:
            paths.append((pred, out))
    return paths

def render_advanced_options(constructs, existing_paths=None):
    if existing_paths is None:
        existing_paths = []
    options = {}
    construct_names = list(constructs.keys())
    with st.expander("Optional: Mediation and Moderation Setup"):
        if len(construct_names) >= 3:
            has_med = st.checkbox("Include mediation analysis?", key="has_med")
            if has_med:
                c1, c2, c3 = st.columns(3)
                with c1: options["mediator_x"] = st.selectbox("Predictor (X)", construct_names, key="med_x_setup")
                with c2: options["mediator_m"] = st.selectbox("Mediator (M)", construct_names, key="med_m_setup")
                with c3: options["mediator_y"] = st.selectbox("Outcome (Y)", construct_names, key="med_y_setup")
                options["has_mediation"] = True
            has_mod = st.checkbox("Include moderation analysis?", key="has_mod")
            if has_mod:
                c1, c2, c3 = st.columns(3)
                with c1: options["mod_x"] = st.selectbox("Predictor (X)", construct_names, key="mod_x_setup")
                with c2: options["mod_w"] = st.selectbox("Moderator (W)", construct_names, key="mod_w_setup")
                with c3: options["mod_y"] = st.selectbox("Outcome (Y)", construct_names, key="mod_y_setup")
                options["has_moderation"] = True
        else:
            st.info("Define at least 3 constructs to set up mediation/moderation.")
    return options

def render_data_input():
    st.title("Data Input and Model Setup")
    st.markdown("Upload your dataset and define your measurement and structural model.")
    st.markdown("---")
    df = render_data_upload()
    if df is None:
        return

    # Add Reset Session button in sidebar-like area
    with st.expander("🔄 Session Management", expanded=False):
        st.markdown("Start a completely new analysis:")
        if st.button("Reset Session (Clear All Data)", key="reset_session_btn", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    with st.expander("Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    st.markdown("---")
    st.subheader("Data Validation")
    validation = validate_dataframe(df)
    for msg in validation["info"]:
        badge("ok", msg)
    for msg in validation["warnings"]:
        badge("warning", f"Warning: {msg}")
    for msg in validation["errors"]:
        badge("critical", f"Error: {msg}")
    if not validation["valid"]:
        st.error("Please fix the errors above before proceeding.")
        return
    st.markdown("---")
    assignments = render_variable_assignment(df)
    st.session_state["assignments"] = assignments
    st.markdown("---")
    constructs = render_construct_definition(df, assignments)
    st.session_state["constructs"] = constructs
    st.markdown("---")
    paths = []
    if constructs:
        paths = render_structural_paths(constructs)
        st.session_state["structural_paths"] = paths
    st.markdown("---")
    if constructs:
        options = render_advanced_options(constructs, paths)
        st.session_state["advanced_options"] = options
        # Update paths after auto-add from mediation/moderation setup
        st.session_state["structural_paths"] = paths
    st.markdown("---")
    # Model Preview
    if constructs and paths:
        st.markdown("---")
        st.subheader("Model Preview")
        st.markdown("Review your model before proceeding:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Measurement Model:**")
            for cname, items in constructs.items():
                st.markdown(f"- **{cname}**: {', '.join(items)}")
        with col2:
            st.markdown("**Structural Paths:**")
            for i, (pred, out) in enumerate(paths, 1):
                st.markdown(f"- H{i}: {pred} → {out}")
        adv = st.session_state.get("advanced_options", {})
        if adv.get("has_mediation"):
            st.markdown(f"**Mediation:** {adv.get('mediator_x')} → {adv.get('mediator_m')} → {adv.get('mediator_y')}")
        if adv.get("has_moderation"):
            st.markdown(f"**Moderation:** {adv.get('mod_x')} × {adv.get('mod_w')} → {adv.get('mod_y')}")

    if constructs:
        if st.button("Confirm Model Setup and Proceed", type="primary", use_container_width=True, key="confirm_setup_btn"):
            st.session_state["df"]       = df
            st.session_state["df_ready"] = True
            st.session_state["validation"] = validation
            cfa_lines = [f"{c} =~ {' + '.join(items)}" for c, items in constructs.items() if items]
            st.session_state["cfa_syntax"] = "\n".join(cfa_lines)
            sem_lines = cfa_lines.copy()
            for pred, out in st.session_state.get("structural_paths", []):
                sem_lines.append(f"{out} ~ {pred}")
            st.session_state["sem_syntax"] = "\n".join(sem_lines)
            st.balloons()
            badge("excellent", "Model setup complete! Navigate to Descriptive Statistics in the sidebar to continue.")
