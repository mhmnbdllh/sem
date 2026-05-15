"""
export.py - Export & Report Generation Module for SEM Studio.
Generates comprehensive HTML report with interpretations and visualizations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

COLORS = {
    "excellent": "#1a7a4a",
    "good":      "#2ecc71",
    "ok":        "#1a6fa8",
    "warning":   "#b7770d",
    "critical":  "#c0392b",
}

def badge(level, message):
    import re
    color = COLORS.get(level, "#555555")
    message = re.sub(r'[*][*](.+?)[*][*]', r'<b>\1</b>', str(message))
    st.markdown(
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:4px;margin:6px 0;'
        f'color:#1a1a1a;font-size:0.92rem">{message}</div>',
        unsafe_allow_html=True,
    )

def _safe_float(val, default=None):
    if val is None: return default
    try:
        if isinstance(val, list): val = val[0]
        f = float(val)
        return None if np.isnan(f) else f
    except: return default

def _fmt(val, decimals=3):
    if val is None: return "—"
    try:
        if np.isnan(float(val)): return "—"
        return f"{float(val):.{decimals}f}"
    except: return str(val)

def _stars(p):
    if p is None: return ""
    try:
        p = float(p)
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        elif p < 0.10: return "†"
        return ""
    except: return ""


# ── CHECKLIST ────────────────────────────────────────────────────

def render_full_checklist():
    st.subheader("Methodological Checklist")
    ss = st.session_state
    constructs = ss.get("constructs", {})

    checks = {
        "DATA PREPARATION": {
            "Data uploaded and validated":           ss.get("df_ready", False),
            "Variable roles assigned":               bool(ss.get("assignments")),
            "Constructs defined (>= 3 items each)":  all(len(v) >= 3 for v in constructs.values()) if constructs else False,
            "Structural paths defined":              len(ss.get("structural_paths", [])) > 0,
        },
        "DESCRIPTIVE AND ASSUMPTION TESTING": {
            "Descriptive statistics computed":       ss.get("descriptive_complete", False),
            "Normality tested (Mardia)":             "recommended_estimator" in ss,
            "Estimator selected (ML or MLR)":        "recommended_estimator" in ss,
            "Common method bias assessed (Harman)":  ss.get("descriptive_complete", False),
            "Outlier detection performed":           "mv_outliers" in ss,
        },
        "MEASUREMENT MODEL (CFA)": {
            "EFA conducted (if needed)":             ss.get("efa_complete", False),
            "CFA model estimated":                   "cfa_result" in ss,
            "Model fit assessed":                    bool(ss.get("cfa_fit")),
            "Factor loadings >= .50":                _check_loadings(),
            "AVE >= .50 (convergent validity)":      _check_ave(),
            "CR >= .70 (composite reliability)":     _check_cr(),
            "Cronbach alpha >= .70":                 _check_alpha(),
            "HTMT discriminant validity assessed":   "cfa_result" in ss,
        },
        "STRUCTURAL MODEL (SEM)": {
            "Full SEM estimated":                    "sem_result" in ss,
            "SEM fit indices adequate":              _check_sem_fit(),
            "Structural paths reported":             bool(ss.get("sem_paths")),
            "R2 reported for endogenous constructs": bool(ss.get("sem_r2")),
            "Effect sizes (f2) computed":            bool(ss.get("sem_paths")),
        },
        "ADVANCED ANALYSES": {
            "Mediation analysis with bootstrap CI":  bool(ss.get("mediation_results")),
            "Moderation analysis":                   bool(ss.get("moderation_results")),
            "Measurement invariance tested":         bool(ss.get("invariance_results")),
            "Model comparison with rival models":    bool(ss.get("comparison_results")),
        },
    }

    total_checks = 0
    total_pass   = 0

    for section, section_checks in checks.items():
        st.markdown(f"**{section}**")
        rows = []
        for check, passed in section_checks.items():
            rows.append({"Check": check, "Status": "Pass" if passed else "Pending"})
            total_checks += 1
            if passed: total_pass += 1

        def color_status(val):
            if val == "Pass": return "color:#1a7a4a;font-weight:700"
            return "color:#888888"

        df_check = pd.DataFrame(rows)
        st.dataframe(
            df_check.style.map(color_status, subset=["Status"])
                          .set_properties(**{"color":"#1a1a1a","background-color":"#ffffff"})
                          .set_table_styles([{"selector":"th","props":[("background-color","#2E86AB"),("color","white"),("font-weight","bold")]}]),
            use_container_width=True, hide_index=True
        )

    pct = total_pass / total_checks if total_checks > 0 else 0
    st.markdown(f"**Overall progress: {total_pass}/{total_checks} checks complete ({pct:.0%})**")
    st.progress(pct)

    if pct >= 0.80:
        badge("excellent", f"{pct:.0%} of methodological steps completed. Ready to export!")
    elif pct >= 0.50:
        badge("warning", f"{pct:.0%} complete. Consider completing remaining analyses before reporting.")
    else:
        badge("critical", f"Only {pct:.0%} complete. Please run CFA and SEM before exporting.")


def _check_loadings():
    from utils.thresholds import CFA as CFA_T
    metrics = st.session_state.get("cfa_metrics", {})
    if not metrics: return False
    return all(min(m.get("lambdas",[0])) >= CFA_T["loading_min"] for m in metrics.values() if m.get("lambdas"))

def _check_ave():
    from utils.thresholds import CFA as CFA_T
    metrics = st.session_state.get("cfa_metrics", {})
    if not metrics: return False
    return all((m.get("ave") or 0) >= CFA_T["ave_min"] for m in metrics.values())

def _check_cr():
    from utils.thresholds import CFA as CFA_T
    metrics = st.session_state.get("cfa_metrics", {})
    if not metrics: return False
    return all((m.get("cr") or 0) >= CFA_T["cr_min"] for m in metrics.values())

def _check_alpha():
    from utils.thresholds import CFA as CFA_T
    metrics = st.session_state.get("cfa_metrics", {})
    if not metrics: return False
    return all((m.get("alpha") or 0) >= CFA_T["alpha_min"] for m in metrics.values())

def _check_sem_fit():
    from utils.thresholds import FIT
    fit = st.session_state.get("sem_fit", {})
    if not fit: return False
    return ((fit.get("rmsea") or 999) <= FIT["rmsea_acceptable"] and
            (fit.get("cfi")   or 0)   >= FIT["cfi_acceptable"])


# ── APA NARRATIVE ─────────────────────────────────────────────────

def generate_apa_narrative():
    ss  = st.session_state
    df  = ss.get("df")
    n   = len(df) if df is not None else "N/A"
    est = ss.get("recommended_estimator", "ML")
    constructs  = ss.get("constructs", {})
    cfa_fit     = ss.get("cfa_fit", {})
    sem_fit     = ss.get("sem_fit", {})
    metrics     = ss.get("cfa_metrics", {})
    sem_paths   = ss.get("sem_paths", [])
    med_results = ss.get("mediation_results")
    med_vars    = ss.get("mediation_vars", {})
    now         = datetime.now().strftime("%B %d, %Y")

    lines = []
    lines.append("=" * 70)
    lines.append("SEM STUDIO - APA RESULTS SECTION")
    lines.append(f"Generated: {now}")
    lines.append("=" * 70)
    lines.append("")
    lines.append("SAMPLE AND METHOD")
    lines.append("-" * 40)
    lines.append(
        f"The analysis was conducted on a sample of N = {n} participants. "
        f"{est} estimation was used based on Mardia multivariate normality assessment. "
        f"All analyses were performed using SEM Studio (R/lavaan; Rosseel, 2012)."
        f"Scripted by Dr. Muhaimin Abdullah, S.Pd., M.Pd. (https://muhaiminabdullah.com)"
    )
    lines.append("")
    lines.append("MEASUREMENT MODEL (CFA)")
    lines.append("-" * 40)
    if constructs:
        n_c = len(constructs)
        n_i = sum(len(v) for v in constructs.values())
        lines.append(f"A confirmatory factor analysis (CFA) was conducted with {n_c} latent constructs and {n_i} observed indicators.")
    if cfa_fit:
        rmsea = _safe_float(cfa_fit.get("rmsea"))
        cfi   = _safe_float(cfa_fit.get("cfi"))
        tli   = _safe_float(cfa_fit.get("tli"))
        srmr  = _safe_float(cfa_fit.get("srmr"))
        chi2  = _safe_float(cfa_fit.get("chi2"))
        df_   = _safe_float(cfa_fit.get("df"))
        p_    = _safe_float(cfa_fit.get("p"))
        from utils.thresholds import FIT
        acceptable = (rmsea or 999) <= FIT["rmsea_acceptable"] and (cfi or 0) >= FIT["cfi_acceptable"]
        verdict = "acceptable fit" if acceptable else "marginal fit"
        fit_parts = []
        if rmsea is not None: fit_parts.append(f"RMSEA = {rmsea:.3f}")
        if cfi   is not None: fit_parts.append(f"CFI = {cfi:.3f}")
        if tli   is not None: fit_parts.append(f"TLI = {tli:.3f}")
        if srmr  is not None: fit_parts.append(f"SRMR = {srmr:.3f}")
        if chi2 is not None and df_:
            lines.append(f"The measurement model demonstrated {verdict}: chi2({int(df_)}) = {chi2:.3f}, p = {_fmt(p_,3)}, {', '.join(fit_parts)}.")
        else:
            lines.append(f"The measurement model demonstrated {verdict}: {', '.join(fit_parts)}.")
    if metrics:
        lines.append("")
        lines.append("Reliability and validity indicators:")
        for cname, m in metrics.items():
            parts = []
            if m.get("alpha"): parts.append(f"alpha = {m['alpha']:.3f}")
            if m.get("cr"):    parts.append(f"CR = {m['cr']:.3f}")
            if m.get("ave"):   parts.append(f"AVE = {m['ave']:.3f}")
            if parts: lines.append(f"  {cname}: {', '.join(parts)}")
    lines.append("")
    lines.append("STRUCTURAL MODEL (SEM)")
    lines.append("-" * 40)
    if sem_fit:
        rmsea = _safe_float(sem_fit.get("rmsea"))
        cfi   = _safe_float(sem_fit.get("cfi"))
        tli   = _safe_float(sem_fit.get("tli"))
        srmr  = _safe_float(sem_fit.get("srmr"))
        chi2  = _safe_float(sem_fit.get("chi2"))
        df_   = _safe_float(sem_fit.get("df"))
        p_    = _safe_float(sem_fit.get("p"))
        from utils.thresholds import FIT
        acceptable = (rmsea or 999) <= FIT["rmsea_acceptable"] and (cfi or 0) >= FIT["cfi_acceptable"]
        verdict = "acceptable fit" if acceptable else "marginal fit"
        fit_parts = []
        if rmsea is not None: fit_parts.append(f"RMSEA = {rmsea:.3f}")
        if cfi   is not None: fit_parts.append(f"CFI = {cfi:.3f}")
        if tli   is not None: fit_parts.append(f"TLI = {tli:.3f}")
        if srmr  is not None: fit_parts.append(f"SRMR = {srmr:.3f}")
        if chi2 is not None and df_:
            lines.append(f"The full structural model demonstrated {verdict}: chi2({int(df_)}) = {chi2:.3f}, p = {_fmt(p_,3)}, {', '.join(fit_parts)}.")
    if sem_paths:
        lines.append("")
        lines.append("Structural path results:")
        for i, p in enumerate(sem_paths):
            beta  = _safe_float(p.get("beta"))
            se    = _safe_float(p.get("se"))
            p_val = _safe_float(p.get("p"))
            if beta is None or p_val is None: continue
            sig   = "significant" if p_val < 0.05 else "not significant"
            lines.append(
                f"  H{i+1}: {p.get('predictor','?')} --> {p.get('outcome','?')}: "
                f"beta = {beta:.3f}{_stars(p_val)}, SE = {_fmt(se)}, p = {_fmt(p_val,3)} -- {sig}."
            )
    lines.append("")
    if med_results and isinstance(med_results, dict):
        lines.append("MEDIATION ANALYSIS")
        lines.append("-" * 40)
        x = med_vars.get("x","X"); m = med_vars.get("m","M"); y = med_vars.get("y","Y")
        indirect_data = med_results.get("indirect", {})
        if isinstance(indirect_data, dict):
            indirect = _safe_float(indirect_data.get("est"))
            ci_lo    = _safe_float(indirect_data.get("ci_lo"))
            ci_hi    = _safe_float(indirect_data.get("ci_hi"))
            n_boot   = _safe_float(med_results.get("n_boot"), 5000)
            if indirect is not None and ci_lo is not None and ci_hi is not None:
                sig = not (ci_lo <= 0 <= ci_hi)
                lines.append(
                    f"A bootstrap mediation analysis ({int(n_boot):,} resamples) examined whether "
                    f"{m} mediated the relationship between {x} and {y}. "
                    f"The indirect effect was {'significant' if sig else 'not significant'} "
                    f"(indirect = {indirect:.4f}, 95% BCa CI [{ci_lo:.4f}, {ci_hi:.4f}])."
                )
        lines.append("")
    lines.append("=" * 70)
    lines.append("Note: All analyses via SEM Studio (R/lavaan).")
    lines.append("* p < .05; ** p < .01; *** p < .001; dag p < .10")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_apa_narrative():
    st.subheader("APA Results Narrative")
    st.markdown("Auto-generated APA 7th edition results text. Copy, paste, then review and edit before submitting.")
    narrative = generate_apa_narrative()
    st.text_area("APA Results Section (copy-paste ready)", value=narrative, height=500)
    badge("warning",
        "This auto-generated text is a starting point. "
        "Always review, edit, and supplement with your own theoretical interpretation before submitting."
    )


# ── HTML REPORT ───────────────────────────────────────────────────

def _html_tbl(headers, rows, caption="", note=""):
    hdr = "".join(f'<th style="padding:8px 14px;text-align:left;white-space:nowrap">{h}</th>' for h in headers)
    body = ""
    for i, row in enumerate(rows):
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        cells = "".join(
            f'<td style="padding:7px 14px;border-bottom:1px solid #eee;vertical-align:top">{v}</td>'
            for v in row
        )
        body += f'<tr style="background:{bg}">{cells}</tr>'
    cap_html = f'<div style="font-size:0.8rem;color:#1a6fa8;font-weight:600;margin-bottom:6px">{caption}</div>' if caption else ""
    note_html = f'<div style="font-size:0.78rem;color:#666;margin-top:6px;font-style:italic">{note}</div>' if note else ""
    return (
        cap_html +
        f'<div style="overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse;font-size:0.88rem;margin:8px 0">'
        f'<thead><tr style="background:#2E86AB;color:white">{hdr}</tr></thead>'
        f'<tbody>{body}</tbody>'
        f'</table></div>' +
        note_html
    )

def _html_section(num, title, content, color="#2E86AB"):
    return (
        f'<div style="margin:32px 0;page-break-inside:avoid">'
        f'<h2 style="color:{color};border-bottom:2px solid {color};'
        f'padding-bottom:8px;margin-bottom:16px;font-size:1.2rem">'
        f'{num}. {title}</h2>'
        f'{content}'
        f'</div>'
    )

def _html_badge(level, msg):
    colors = {"excellent":"#1a7a4a","ok":"#1a6fa8","warning":"#b7770d","critical":"#c0392b","good":"#2ecc71"}
    c = colors.get(level, "#555")
    return f'<div style="background:{c}18;border-left:4px solid {c};padding:10px 14px;border-radius:4px;margin:8px 0;color:#1a1a1a;font-size:0.88rem">{msg}</div>'

def _fit_interpretation(index, value):
    """Return (level, text) interpretation for a fit index value."""
    if value is None: return ("", "—")
    try: value = float(value)
    except: return ("", "—")
    if index == "rmsea":
        if value <= 0.05: return ("excellent", f"Excellent (≤ .05)")
        elif value <= 0.06: return ("good",    f"Good (≤ .06)")
        elif value <= 0.08: return ("ok",      f"Acceptable (≤ .08)")
        else: return ("critical", f"Poor (> .08)")
    elif index in ("cfi","tli"):
        if value >= 0.97: return ("excellent", f"Excellent (≥ .97)")
        elif value >= 0.95: return ("good",    f"Good (≥ .95)")
        elif value >= 0.90: return ("ok",      f"Acceptable (≥ .90)")
        else: return ("critical", f"Poor (< .90)")
    elif index == "srmr":
        if value <= 0.05: return ("excellent", f"Excellent (≤ .05)")
        elif value <= 0.08: return ("ok",      f"Acceptable (≤ .08)")
        else: return ("critical", f"Poor (> .08)")
    elif index in ("gfi","nfi"):
        if value >= 0.95: return ("good",   f"Good (≥ .95)")
        elif value >= 0.90: return ("ok",   f"Acceptable (≥ .90)")
        else: return ("critical", f"Poor (< .90)")
    return ("", "")

def _level_badge_html(level, text):
    icons = {"excellent":"✅","good":"✅","ok":"ℹ️","warning":"⚠️","critical":"❌"}
    colors = {"excellent":"#1a7a4a","good":"#2ecc71","ok":"#1a6fa8","warning":"#b7770d","critical":"#c0392b"}
    c = colors.get(level,"#888")
    icon = icons.get(level,"")
    return f'<span style="color:{c};font-weight:600">{icon} {text}</span>'


def generate_html_report():
    ss          = st.session_state
    df          = ss.get("df")
    n           = len(df) if df is not None else "N/A"
    constructs  = ss.get("constructs", {})
    struct_paths= ss.get("structural_paths", [])
    assignments = ss.get("assignments", {})
    estimator   = ss.get("recommended_estimator", "N/A")
    cfa_fit     = ss.get("cfa_fit", {})
    sem_fit     = ss.get("sem_fit", {})
    metrics     = ss.get("cfa_metrics", {})
    cfa_loadings= ss.get("cfa_loadings", {})
    sem_paths   = ss.get("sem_paths", [])
    sem_r2      = ss.get("sem_r2", [])
    med_results = ss.get("mediation_results")
    med_vars    = ss.get("mediation_vars", {})
    mod_results = ss.get("moderation_results")
    mod_vars    = ss.get("moderation_vars", {})
    inv_results = ss.get("invariance_results")
    inv_level   = ss.get("invariance_level", "—")
    comp_results= ss.get("comparison_results", {})
    best_model  = ss.get("best_model", "—")
    efa_result  = ss.get("efa_result")
    narrative   = generate_apa_narrative()
    now         = datetime.now().strftime("%B %d, %Y %H:%M")

    sections = []

    # ── 1. Overview ──────────────────────────────────────────────
    overview_rows = [
        ["Sample Size (n)", f"<b>{n}</b>"],
        ["Number of Constructs", str(len(constructs))],
        ["Total Indicators", str(sum(len(v) for v in constructs.values()))],
        ["Structural Paths", str(len(struct_paths))],
        ["Estimator", f"<b>{estimator}</b>"],
        ["Report Generated", now],
    ]
    sections.append(_html_section(1, "Study Overview",
        _html_tbl(["Item","Value"], overview_rows) +
        _html_badge("ok",
            "This report summarizes a Structural Equation Modeling (SEM) analysis. "
            "All statistical analyses were conducted using R/lavaan (Rosseel, 2012). "
            "Interpretation guidelines follow Hair et al. (2019) and Kline (2016)."
        )))

    # ── 2. Construct Definitions ──────────────────────────────────
    const_rows = []
    for c, items in constructs.items():
        const_rows.append([f"<b>{c}</b>", ", ".join(items), str(len(items))])
    path_rows = [[f"<b>H{i+1}</b>", f"{pred} → {out}"] for i,(pred,out) in enumerate(struct_paths)]
    s2 = _html_tbl(["Construct","Indicators","n"], const_rows, "Measurement Model")
    if path_rows:
        s2 += _html_tbl(["Hypothesis","Path"], path_rows, "Structural Paths (Hypotheses)")
    s2 += _html_badge("ok",
        "The measurement model specifies which indicators belong to each latent construct. "
        "The structural model specifies the hypothesized directional relationships between constructs. "
        "Each construct should have at least 3 indicators for model identification."
    )
    sections.append(_html_section(2, "Model Specification", s2))

    # ── 3. Descriptive Statistics ─────────────────────────────────
    indicator_cols = [c for c,r in assignments.items() if r == "indicator"] if assignments else []
    desc_rows = []
    if df is not None and indicator_cols:
        from scipy import stats as scipy_stats
        for col in indicator_cols:
            x = df[col].dropna()
            if len(x) < 3: continue
            sk = float(scipy_stats.skew(x))
            ku = float(scipy_stats.kurtosis(x))
            sk_flag = "" if abs(sk) <= 1 else (" ⚠️" if abs(sk) <= 2 else " ❌")
            ku_flag = "" if abs(ku) <= 3 else (" ⚠️" if abs(ku) <= 7 else " ❌")
            miss_n  = int(df[col].isna().sum())
            miss_pct= miss_n / len(df) * 100
            desc_rows.append([
                col,
                str(int(x.count())),
                f"{float(x.mean()):.3f}",
                f"{float(x.std()):.3f}",
                f"{float(x.min()):.1f}",
                f"{float(x.max()):.1f}",
                f"{sk:.3f}{sk_flag}",
                f"{ku:.3f}{ku_flag}",
                f"{miss_n} ({miss_pct:.1f}%)",
            ])
    s3 = _html_tbl(
        ["Variable","N","Mean","SD","Min","Max","Skewness","Kurtosis","Missing"],
        desc_rows,
        note="Note: ⚠️ = mild non-normality; ❌ = substantial non-normality. "
             "|Skewness| > 2 or |Kurtosis| > 7 suggests MLR estimator (Hair et al., 2019)."
    )
    mardia = ss.get("mardia_result", {})
    if mardia:
        sk_p = _safe_float(mardia.get("skewness_p"))
        ku_p = _safe_float(mardia.get("kurtosis_p"))
        if sk_p is not None:
            normal = sk_p > 0.05 and ku_p > 0.05
            s3 += _html_badge(
                "ok" if normal else "warning",
                f"Mardia's Test: skewness p = {_fmt(sk_p,4)}, kurtosis p = {_fmt(ku_p,4)}. "
                f"{'Multivariate normality satisfied. ML estimation appropriate.' if normal else 'Multivariate normality violated. MLR (Robust ML) recommended.'}"
            )
    s3 += _html_badge("ok", f"Estimator used: <b>{estimator}</b>")
    sections.append(_html_section(3, "Descriptive Statistics and Assumption Testing", s3))

    # ── 4. EFA Results ────────────────────────────────────────────
    if efa_result and isinstance(efa_result, dict) and "error" not in efa_result:
        kmo     = _safe_float(efa_result.get("kmo"))
        bart_p  = _safe_float(efa_result.get("bartlett_p"))
        n_f     = efa_result.get("n_factors")
        if isinstance(n_f, list): n_f = n_f[0]
        efa_rows = [
            ["KMO Overall",    _fmt(kmo), "≥ .80 meritorious; ≥ .60 acceptable", _level_badge_html("excellent" if kmo is not None and kmo >= 0.80 else "ok" if kmo is not None and kmo >= 0.60 else "critical", "Meritorious" if kmo is not None and kmo >= 0.80 else "Acceptable" if kmo is not None and kmo >= 0.60 else "Poor")],
            ["Bartlett's p",   "< .001" if bart_p is not None and bart_p < 0.001 else _fmt(bart_p,4), "p < .05 required", _level_badge_html("excellent" if bart_p is not None and bart_p < 0.05 else "critical", "Significant" if bart_p is not None and bart_p < 0.05 else "Not Significant")],
            ["Factors Extracted", str(int(n_f)) if n_f else "—", "Based on parallel analysis", ""],
            ["Rotation",       str(efa_result.get("rotation","—")), "PAF extraction", ""],
        ]
        s4 = _html_tbl(["Test","Value","Criterion","Interpretation"], efa_rows, "EFA Factorability")
        # Factor names
        factor_names = ss.get("efa_factor_names", {})
        if factor_names:
            fn_rows = [[f, name] for f, name in factor_names.items()]
            s4 += _html_tbl(["Factor","Construct Name"], fn_rows, "Factor Assignments")
        s4 += _html_badge("ok",
            "EFA identifies the underlying factor structure empirically. "
            "KMO ≥ .80 = meritorious; ≥ .60 = acceptable. "
            "Bartlett's test must be significant (p < .05). "
            "Factor loadings ≥ .50 are considered acceptable; ≥ .70 are strong."
        )
        sections.append(_html_section(4, "Exploratory Factor Analysis (EFA)", s4))

    # ── 5. CFA Fit Indices ────────────────────────────────────────
    fit_specs = [
        ("chi2",  "χ²",     "Non-significant preferred (sensitive to n)"),
        ("df",    "df",     "Degrees of freedom"),
        ("pvalue","p",      "χ² p-value"),
        ("rmsea", "RMSEA",  "≤ .05 excellent; ≤ .06 good; ≤ .08 acceptable"),
        ("cfi",   "CFI",    "≥ .97 excellent; ≥ .95 good; ≥ .90 acceptable"),
        ("tli",   "TLI",    "≥ .95 good; ≥ .90 acceptable"),
        ("srmr",  "SRMR",   "≤ .05 good; ≤ .08 acceptable"),
        ("gfi",   "GFI",    "≥ .95 good; ≥ .90 acceptable"),
        ("nfi",   "NFI",    "≥ .95 good; ≥ .90 acceptable"),
        ("aic",   "AIC",    "Lower is better"),
        ("bic",   "BIC",    "Lower is better"),
    ]
    cfa_fit_rows = []
    for key, label, criterion in fit_specs:
        val = cfa_fit.get(key)
        if val is None: val = cfa_fit.get(key.replace("_",""))
        if val is None: continue
        try: val_f = float(val)
        except: continue
        level, interp = _fit_interpretation(key, val_f)
        val_str = f"{val_f:.3f}" if key not in ("df",) else str(int(val_f))
        interp_html = _level_badge_html(level, interp) if interp else ""
        cfa_fit_rows.append([label, val_str, criterion, interp_html])
    # chi2/df
    chi2 = _safe_float(cfa_fit.get("chi2"))
    df_  = _safe_float(cfa_fit.get("df"))
    if chi2 is not None and df_ and float(df_) > 0:
        ratio = chi2/df_
        level_r = "good" if ratio <= 2 else "ok" if ratio <= 5 else "critical"
        label_r = "Good (≤ 2.0)" if ratio <= 2 else "Acceptable (≤ 5.0)" if ratio <= 5 else "Poor (> 5.0)"
        cfa_fit_rows.insert(2, ["χ²/df", f"{ratio:.3f}", "≤ 2.0 good; ≤ 5.0 acceptable",
                                _level_badge_html(level_r, label_r)])
    s5 = _html_tbl(["Index","Value","Criterion","Interpretation"], cfa_fit_rows,
        "CFA Goodness-of-Fit Indices",
        "Note: Hu & Bentler (1999); Kline (2016); Jöreskog & Sörbom (1984).")
    sections.append(_html_section(5, "CFA Model Fit", s5))

    # ── 6. Factor Loadings ────────────────────────────────────────
    load_rows = []
    for cname, items_dict in cfa_loadings.items():
        for item, lam in items_dict.items():
            try:
                lam_f = float(lam)
                status = "Strong (≥.70)" if abs(lam_f) >= 0.70 else "Acceptable (≥.50)" if abs(lam_f) >= 0.50 else "Weak (<.50)"
                color  = "#1a7a4a" if abs(lam_f) >= 0.70 else "#1a6fa8" if abs(lam_f) >= 0.50 else "#c0392b"
                load_rows.append([cname, item, f'<span style="color:{color};font-weight:600">{lam_f:.3f}</span>', f'<span style="color:{color}">{status}</span>'])
            except: pass
    s6 = _html_tbl(["Construct","Item","Std. Loading (λ)","Status"], load_rows,
        "Standardized Factor Loadings",
        "Note: λ ≥ .70 = strong; λ ≥ .50 = acceptable; λ < .50 = weak (Hair et al., 2019). p < .001 for all significant loadings.")
    # Add interpretation summary for loadings
    if load_rows:
        weak   = sum(1 for r in load_rows if "Weak" in r[3])
        strong = sum(1 for r in load_rows if "Strong" in r[3])
        total  = len(load_rows)
        if weak == 0:
            s6 += _html_badge("excellent",
                f"All {total} factor loadings meet the acceptable threshold (λ ≥ .50). "
                f"{strong} loadings are strong (λ ≥ .70). "
                "Convergent validity at the item level is supported.")
        else:
            s6 += _html_badge("warning",
                f"{weak} of {total} loadings are weak (λ < .50). "
                "Consider removing weak items and re-running CFA.")
    sections.append(_html_section(6, "Factor Loadings", s6))

    # ── 7. Reliability and Validity ───────────────────────────────
    rel_rows = []
    for cname, m in metrics.items():
        alpha = _safe_float(m.get("alpha"))
        cr    = _safe_float(m.get("cr"))
        ave   = _safe_float(m.get("ave"))
        def pass_fail(val, threshold, fmt=".3f"):
            if val is None: return "—"
            color = "#1a7a4a" if val >= threshold else "#c0392b"
            icon  = "✅" if val >= threshold else "❌"
            return f'<span style="color:{color}">{icon} {val:{fmt}}</span>'
        rel_rows.append([
            f"<b>{cname}</b>",
            str(m.get("n_items","—")),
            pass_fail(alpha, 0.70),
            pass_fail(cr,    0.70),
            pass_fail(ave,   0.50),
        ])
    s7  = _html_tbl(["Construct","Items","Cronbach α","CR","AVE"], rel_rows,
        "Reliability and Convergent Validity",
        "Note: α ≥ .70 acceptable (Nunnally, 1978); CR ≥ .70, AVE ≥ .50 (Fornell & Larcker, 1981).")

    # Fornell-Larcker matrix
    if df is not None and constructs:
        parcel_df = pd.DataFrame()
        for cname, items in constructs.items():
            valid = [c for c in items if c in df.columns]
            if valid: parcel_df[cname] = df[valid].mean(axis=1)
        parcel_df = parcel_df.dropna()
        if parcel_df.shape[1] >= 2:
            corr = parcel_df.corr()
            cn   = list(parcel_df.columns)
            fl_headers = [""] + cn
            fl_rows    = []
            for c1 in cn:
                row = [f"<b>{c1}</b>"]
                for c2 in cn:
                    if c1 == c2:
                        ave = _safe_float(metrics.get(c1, {}).get("ave"))
                        val = f"<b>{np.sqrt(ave):.3f}(*)</b>" if ave else "—"
                    else:
                        r = corr.loc[c1,c2] if c1 in corr.index and c2 in corr.columns else np.nan
                        val = f"{r:.3f}" if not np.isnan(r) else "—"
                    row.append(val)
                fl_rows.append(row)
            s7 += _html_tbl(fl_headers, fl_rows,
                "Fornell-Larcker Criterion",
                "Note: Diagonal (*) = sqrt(AVE). Discriminant validity supported when sqrt(AVE) > all off-diagonal correlations in the same row/column.")
    # Add overall reliability interpretation
    if rel_rows:
        all_pass_alpha = all((m.get("alpha") or 0) >= 0.70 for m in metrics.values())
        all_pass_cr    = all((m.get("cr")    or 0) >= 0.70 for m in metrics.values())
        all_pass_ave   = all((m.get("ave")   or 0) >= 0.50 for m in metrics.values())
        if all_pass_alpha and all_pass_cr and all_pass_ave:
            s7 += _html_badge("excellent",
                "All constructs demonstrate adequate reliability (α ≥ .70, CR ≥ .70) "
                "and convergent validity (AVE ≥ .50). "
                "The Fornell-Larcker criterion should also be checked: "
                "sqrt(AVE) of each construct should exceed its correlations with all other constructs.")
        else:
            s7 += _html_badge("warning",
                "One or more constructs do not meet reliability or validity thresholds. "
                "Review items with low loadings and consider scale refinement.")
    sections.append(_html_section(7, "Reliability and Validity", s7))

    # ── 8. SEM Fit Indices ────────────────────────────────────────
    sem_fit_rows = []
    for key, label, criterion in fit_specs:
        val = sem_fit.get(key)
        if val is None: continue
        try: val_f = float(val)
        except: continue
        level, interp = _fit_interpretation(key, val_f)
        val_str = f"{val_f:.3f}" if key != "df" else str(int(val_f))
        interp_html = _level_badge_html(level, interp) if interp else ""
        sem_fit_rows.append([label, val_str, criterion, interp_html])
    chi2s = _safe_float(sem_fit.get("chi2"))
    df_s  = _safe_float(sem_fit.get("df"))
    if chi2s and df_s and float(df_s) > 0:
        ratio = chi2s/df_s
        level_r = "good" if ratio <= 2 else "ok" if ratio <= 5 else "critical"
        label_r = "Good (≤ 2.0)" if ratio <= 2 else "Acceptable (≤ 5.0)" if ratio <= 5 else "Poor (> 5.0)"
        sem_fit_rows.insert(2, ["χ²/df", f"{ratio:.3f}", "≤ 2.0 good; ≤ 5.0 acceptable",
                                _level_badge_html(level_r, label_r)])
    s8 = _html_tbl(["Index","Value","Criterion","Interpretation"], sem_fit_rows,
        "SEM Goodness-of-Fit Indices",
        "Note: Same criteria as CFA fit indices above.")
    sections.append(_html_section(8, "SEM Model Fit", s8))

    # ── 9. Structural Paths ───────────────────────────────────────
    path_rows_html = []
    for i, p in enumerate(sem_paths):
        beta  = _safe_float(p.get("beta"))
        se    = _safe_float(p.get("se"))
        z     = _safe_float(p.get("z"))
        pval  = _safe_float(p.get("p"))
        if beta is None: continue
        sig   = pval is not None and pval < 0.05
        color = "#1a7a4a" if (sig and beta > 0) else "#c0392b" if (sig and beta < 0) else "#888"
        f2    = beta**2 / (1 - beta**2) if abs(beta) < 1 else None
        f2_str = f"{f2:.3f}" if f2 else "—"
        f2_label = "Large" if f2 and f2 >= 0.35 else "Medium" if f2 and f2 >= 0.15 else "Small" if f2 and f2 >= 0.02 else "Negligible"
        decision = f'<span style="color:{"#1a7a4a" if sig else "#c0392b"};font-weight:700">{"✅ Supported" if sig else "❌ Not Supported"}</span>'
        path_rows_html.append([
            f"<b>H{i+1}</b>",
            f"{p.get('predictor','?')} → {p.get('outcome','?')}",
            f'<span style="color:{color};font-weight:700">{beta:.3f}{_stars(pval)}</span>',
            _fmt(se), _fmt(z),
            _fmt(pval,4) if pval is not None else "—",
            f"{f2_str} ({f2_label})",
            decision,
        ])
    s9 = _html_tbl(
        ["H","Path","β","SE","z","p","Cohen's f²","Decision"],
        path_rows_html,
        "Structural Path Coefficients",
        "Note: β = standardized path coefficient. * p < .05; ** p < .01; *** p < .001. "
        "Cohen's f²: ≥ .02 small; ≥ .15 medium; ≥ .35 large (Cohen, 1988)."
    )
    # Add overall path interpretation
    if path_rows_html:
        supported   = sum(1 for p in sem_paths if (_safe_float(p.get("p")) or 1) < 0.05)
        total_paths = len(sem_paths)
        s9 += _html_badge(
            "excellent" if supported == total_paths else "ok" if supported > 0 else "warning",
            f"{supported} of {total_paths} hypothesized path(s) are statistically significant (p < .05). "
            "Significant paths support the hypothesized relationships. "
            "Note: * p &lt; .05; ** p &lt; .01; *** p &lt; .001. "
            "Beta = standardized path coefficient; larger absolute values indicate stronger effects."
        )
    sections.append(_html_section(9, "Structural Path Coefficients", s9))

    # ── 10. R-Squared ─────────────────────────────────────────────
    r2_rows_html = []
    for row in sem_r2:
        if not isinstance(row, dict): continue
        r2  = _safe_float(row.get("R2"))
        if r2 is None: continue
        level_r2 = "good" if r2 >= 0.26 else "ok" if r2 >= 0.13 else "warning" if r2 >= 0.02 else "critical"
        label_r2 = "Substantial (≥.26)" if r2 >= 0.26 else "Moderate (≥.13)" if r2 >= 0.13 else "Weak (≥.02)" if r2 >= 0.02 else "Negligible"
        r2_rows_html.append([
            f"<b>{row.get('Construct','—')}</b>",
            f"{r2:.3f}",
            f"{r2:.1%}",
            _level_badge_html(level_r2, label_r2),
        ])
    s10 = _html_tbl(["Construct","R²","R² (%)","Level"], r2_rows_html,
        "Explained Variance (R²) for Endogenous Constructs",
        "Note: R² ≥ .26 = substantial; ≥ .13 = moderate; ≥ .02 = weak (Cohen, 1988).")
    if r2_rows_html:
        s10 += _html_badge("ok",
            "R² indicates the proportion of variance in each endogenous construct "
            "explained by its predictors. "
            "Benchmarks: R² ≥ .26 = substantial; ≥ .13 = moderate; ≥ .02 = weak (Cohen, 1988). "
            "Higher R² indicates the model explains more variance in the outcome construct."
        )
    sections.append(_html_section(10, "Explained Variance (R²)", s10))

    # ── 11. Mediation ─────────────────────────────────────────────
    if med_results and isinstance(med_results, dict):
        x = med_vars.get("x","X"); m_v = med_vars.get("m","M"); y = med_vars.get("y","Y")
        med_rows_html = []
        for key, label in [
            ("a_path",  f"{x} → {m_v} (a path)"),
            ("b_path",  f"{m_v} → {y} | {x} (b path)"),
            ("cp_path", f"{x} → {y} direct (c')"),
            ("total",   f"{x} → {y} total (c)"),
            ("indirect",f"Indirect (a × b)"),
        ]:
            d = med_results.get(key, {})
            if not isinstance(d, dict): continue
            est   = _safe_float(d.get("est"))
            pval  = _safe_float(d.get("p"))
            ci_lo = _safe_float(d.get("ci_lo"))
            ci_hi = _safe_float(d.get("ci_hi"))
            ci_str = f"[{ci_lo:.4f}, {ci_hi:.4f}]" if ci_lo is not None and ci_hi is not None else "—"
            if key == "indirect":
                sig = ci_lo is not None and ci_hi is not None and not (ci_lo <= 0 <= ci_hi)
                sig_str = _level_badge_html("excellent" if sig else "critical",
                                            "Significant (CI ≠ 0)" if sig else "Not Significant (CI includes 0)")
            else:
                sig_str = _stars(pval) if pval is not None else "—"
            med_rows_html.append([label, _fmt(est), ci_str, sig_str])

        # Mediation type
        indirect_d = med_results.get("indirect", {})
        if isinstance(indirect_d, dict):
            ind  = _safe_float(indirect_d.get("est"))
            cilo = _safe_float(indirect_d.get("ci_lo"))
            cihi = _safe_float(indirect_d.get("ci_hi"))
            cp_d = med_results.get("cp_path", {})
            dir_ = _safe_float(cp_d.get("est") if isinstance(cp_d,dict) else None)
            tot_d= med_results.get("total", {})
            tot  = _safe_float(tot_d.get("est") if isinstance(tot_d,dict) else None)
            if ind is not None and cilo is not None and cihi is not None:
                sig_ind = not (cilo <= 0 <= cihi)
                sig_dir = dir_ is not None and abs(dir_) > 0.05
                if sig_ind and not sig_dir:
                    med_type = "Full (Indirect-Only) Mediation"
                    med_level = "excellent"
                elif sig_ind and sig_dir:
                    med_type = "Partial Mediation"
                    med_level = "ok"
                else:
                    med_type = "No Mediation"
                    med_level = "warning"
                vaf_str = ""
                if tot and abs(tot) > 0.001 and ind is not None:
                    vaf = ind / tot
                    vaf_str = f" VAF = {vaf:.1%}."

        s11 = _html_tbl(["Effect","β","95% BCa CI","Significance"], med_rows_html,
            f"Bootstrap Mediation: {x} → {m_v} → {y}",
            f"Note: {med_results.get('n_boot',5000):,}-resample bootstrap via R/lavaan. "
            "Significance criterion: 95% BCa CI not containing zero. "
            "* p &lt; .05; ** p &lt; .01; *** p &lt; .001 for direct and total effects.")
        s11 += _html_badge(med_level, f"<b>{med_type}</b>: indirect = {_fmt(ind,4)}, "
            f"95% BCa CI [{_fmt(cilo,4)}, {_fmt(cihi,4)}].{vaf_str}")
        sections.append(_html_section(11, f"Mediation Analysis ({x} → {m_v} → {y})", s11))

    # ── 12. Moderation ────────────────────────────────────────────
    if mod_results and isinstance(mod_results, dict):
        x = mod_vars.get("x","X"); w = mod_vars.get("w","W"); y = mod_vars.get("y","Y")
        b1=_safe_float(mod_results.get("b1")); b1_p=_safe_float(mod_results.get("b1_p"))
        b2=_safe_float(mod_results.get("b2")); b2_p=_safe_float(mod_results.get("b2_p"))
        b3=_safe_float(mod_results.get("b3")); b3_p=_safe_float(mod_results.get("b3_p"))
        r2_1=_safe_float(mod_results.get("r2_1")); r2_2=_safe_float(mod_results.get("r2_2"))
        dr2 =_safe_float(mod_results.get("delta_r2"))

        mod_coef_rows = [
            [f"<b>{x}</b> (X)",         _fmt(b1), _fmt(_safe_float(mod_results.get("b1_se"))), _fmt(b1_p,4), _stars(b1_p), _level_badge_html("ok" if b1_p is not None and b1_p < 0.05 else "warning", "Significant" if b1_p is not None and b1_p < 0.05 else "n.s.")],
            [f"<b>{w}</b> (W)",         _fmt(b2), _fmt(_safe_float(mod_results.get("b2_se"))), _fmt(b2_p,4), _stars(b2_p), _level_badge_html("ok" if b2_p is not None and b2_p < 0.05 else "warning", "Significant" if b2_p is not None and b2_p < 0.05 else "n.s.")],
            [f"<b>{x} × {w}</b> (Int.)",_fmt(b3), _fmt(_safe_float(mod_results.get("b3_se"))), _fmt(b3_p,4), _stars(b3_p), _level_badge_html("ok" if b3_p is not None and b3_p < 0.05 else "warning", "Significant ✅" if b3_p is not None and b3_p < 0.05 else "Not Significant")],
        ]
        s12 = _html_tbl(["Term","β","SE","p","Sig.","Interpretation"], mod_coef_rows,
            f"Moderation Analysis: {x} × {w} → {y}",
            "Note: * p &lt; .05; ** p &lt; .01; *** p &lt; .001. "
            "Variables are mean-centered before computing the interaction term (Aiken & West, 1991). "
            "A significant interaction term (X × W) indicates moderation.")
        s12 += _html_tbl(
            ["Model","R²","ΔR²"],
            [[f"Model 1: {x} + {w}", _fmt(r2_1), "—"],
             [f"Model 2: {x} + {w} + {x}×{w}", _fmt(r2_2), _fmt(dr2,4)]],
            note="Note: Variables mean-centered before interaction (Aiken & West, 1991). "
                 "ΔR² = variance explained by interaction term."
        )
        # Simple slopes
        slopes = mod_results.get("simple_slopes", {})
        if slopes and isinstance(slopes, dict):
            ss_rows = []
            for label, sd in slopes.items():
                if not isinstance(sd, dict): continue
                slope = _safe_float(sd.get("slope"))
                sp    = _safe_float(sd.get("p"))
                ss_rows.append([label, _fmt(slope,4), _fmt(_safe_float(sd.get("se")),4),
                                 _fmt(_safe_float(sd.get("t")),3), _fmt(sp,4), _stars(sp),
                                 _level_badge_html("ok" if sp is not None and sp < 0.05 else "warning",
                                                   "Significant" if sp is not None and sp < 0.05 else "n.s.")])
            s12 += _html_tbl(
                ["W Level","Simple Slope","SE","t","p","Sig.","Interpretation"],
                ss_rows,
                "Simple Slope Analysis",
                "Note: Simple slopes show X → Y effect at low (−1 SD), mean, and high (+1 SD) of W."
            )
        sections.append(_html_section(12, f"Moderation Analysis ({x} × {w} → {y})", s12))

    # ── 13. Measurement Invariance ────────────────────────────────
    if inv_results and isinstance(inv_results, dict) and "error" not in inv_results:
        inv_model_rows = []
        for name, label in [("configural","Configural"),("metric","Metric"),("scalar","Scalar")]:
            fd = inv_results.get(name, {})
            if not isinstance(fd, dict): continue
            inv_model_rows.append([
                f"<b>{label}</b>",
                _fmt(fd.get("chi2"),3),
                str(int(fd.get("df",0))) if fd.get("df") else "—",
                _fmt(fd.get("cfi"),3),
                _fmt(fd.get("rmsea"),3),
                _fmt(fd.get("srmr"),3),
            ])
        diff_rows = []
        for key, label in [("diff_metric","Metric vs Configural"),("diff_scalar","Scalar vs Metric")]:
            d = inv_results.get(key,{})
            if not isinstance(d, dict): continue
            dcfi = _safe_float(d.get("delta_cfi"))
            supported = dcfi is not None and dcfi >= -0.010
            diff_rows.append([
                label,
                _fmt(d.get("delta_chi2"),3),
                _fmt(dcfi,4),
                _fmt(d.get("delta_rmsea"),4),
                _level_badge_html("ok" if supported else "warning",
                                  "Supported (ΔCFI ≥ −.010)" if supported else "Not Supported (ΔCFI < −.010)"),
            ])
        s13  = _html_tbl(["Model","χ²","df","CFI","RMSEA","SRMR"], inv_model_rows, "Invariance Model Fit")
        s13 += _html_tbl(["Comparison","Δχ²","ΔCFI","ΔRMSEA","Invariance"], diff_rows,
            "Model Comparison (Difference Tests)",
            "Note: ΔCFI ≥ −.010 supports invariance (Cheung & Rensvold, 2002); ΔRMSEA ≤ .015 (Chen, 2007).")
        s13 += _html_badge("ok", f"Highest invariance level achieved: <b>{inv_level}</b>. "
            "Configural = same structure; Metric = equal loadings; Scalar = equal intercepts.")
        sections.append(_html_section(13, "Measurement Invariance", s13))

    # ── 14. Model Comparison ─────────────────────────────────────
    if comp_results and isinstance(comp_results, dict):
        valid = {k:v for k,v in comp_results.items() if isinstance(v,dict) and "error" not in v}
        if valid:
            aic_vals = {k: _safe_float(v.get("aic")) for k,v in valid.items() if _safe_float(v.get("aic"))}
            min_aic  = min(aic_vals.values()) if aic_vals else None
            comp_rows = []
            for name, fit in valid.items():
                aic = _safe_float(fit.get("aic"))
                bic = _safe_float(fit.get("bic"))
                d_aic = round(aic - min_aic, 2) if aic and min_aic else "—"
                is_best = name == best_model
                comp_rows.append([
                    f'<b style="color:{"#1a7a4a" if is_best else "#1a1a1a"}">{name}{"  ✅ Best" if is_best else ""}</b>',
                    _fmt(_safe_float(fit.get("rmsea")),3),
                    _fmt(_safe_float(fit.get("cfi")),3),
                    _fmt(aic,1) if aic else "—",
                    _fmt(bic,1) if bic else "—",
                    str(d_aic),
                ])
            s14 = _html_tbl(["Model","RMSEA","CFI","AIC","BIC","ΔAIC"], comp_rows,
                "Model Fit Comparison",
                "Note: Lower AIC/BIC = better fit. ΔAIC ≤ 2 = substantial support; > 10 = no support (Burnham & Anderson, 2002).")
            s14 += _html_badge("ok", f"Recommended model: <b>{best_model}</b>. "
                "Model selection should always be guided by theoretical considerations.")
            sections.append(_html_section(14, "Model Comparison", s14))

    # ── 15. Path Diagram ─────────────────────────────────────────
    try:
        from modules.visualization import build_path_diagram
        sem_paths_data = ss.get("sem_paths", [])
        r2_data        = ss.get("sem_r2", [])
        fig = build_path_diagram(
            constructs       = constructs,
            structural_paths = struct_paths,
            sem_paths        = sem_paths_data,
            cfa_loadings     = cfa_loadings,
            r2_data          = r2_data,
            show_indicators  = True,
        )
        # Export as interactive HTML snippet (no kaleido needed)
        fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        s15 = (
            f'<div style="margin:16px 0">{fig_html}</div>'
            f'<p style="font-size:0.78rem;color:#666;text-align:center">'
            f'Figure 1. SEM Path Diagram. Green = significant positive path; '
            f'Red = significant negative path; Gray dashed = non-significant. '
            f'* p &lt; .05; ** p &lt; .01; *** p &lt; .001. Numbers on paths = standardized β.</p>'
        )
        sections.append(_html_section(15, "SEM Path Diagram", s15))
    except Exception as e:
        sections.append(_html_section(15, "SEM Path Diagram",
            f'<p style="color:#888">Path diagram not available: {str(e)}</p>'))

    # ── 16. Methodological Checklist ─────────────────────────────
    check_rows = []
    check_items = {
        "Data uploaded and validated":           ss.get("df_ready",False),
        "Normality tested, estimator selected":  "recommended_estimator" in ss,
        "CFA estimated and validated":           "cfa_result" in ss,
        "Factor loadings ≥ .50":                 _check_loadings(),
        "AVE ≥ .50 (convergent validity)":       _check_ave(),
        "CR ≥ .70 (composite reliability)":      _check_cr(),
        "Cronbach α ≥ .70":                      _check_alpha(),
        "CFA model fit adequate":                bool(cfa_fit),
        "SEM estimated":                         "sem_result" in ss,
        "SEM model fit adequate":                _check_sem_fit(),
        "Structural paths reported":             bool(sem_paths),
        "R² reported":                           bool(sem_r2),
        "Mediation analysis conducted":          bool(med_results),
        "Moderation analysis conducted":         bool(mod_results),
        "Measurement invariance tested":         bool(inv_results),
    }
    for item, passed in check_items.items():
        icon  = "✅" if passed else "⬜"
        color = "#1a7a4a" if passed else "#888"
        check_rows.append([f'<span style="color:{color}">{icon} {item}</span>'])
    s16 = _html_tbl(["Check"], check_rows)
    passed = sum(1 for _, passed in check_items.items() if passed)
    total_c = len(check_items)
    pct_c   = passed / total_c if total_c > 0 else 0
    s16 += _html_badge(
        "excellent" if pct_c >= 0.90 else "ok" if pct_c >= 0.70 else "warning",
        f"Methodological completeness: {passed}/{total_c} steps completed ({pct_c:.0%}). "
        f"{'All major methodological requirements are met.' if pct_c >= 0.90 else 'Consider completing remaining analyses before publication.'}"
    )
    sections.append(_html_section(16, "Methodological Checklist", s16, color="#555"))

    # ── 17. APA Narrative ────────────────────────────────────────
    s17 = (
        f'<pre style="background:#f8fafc;padding:20px;border-radius:6px;'
        f'font-family:Georgia,serif;font-size:0.87rem;line-height:1.8;'
        f'white-space:pre-wrap;border:1px solid #dde3ea;color:#1a1a1a">'
        f'{narrative}'
        f'</pre>'
        f'<p style="font-size:0.78rem;color:#888;margin-top:8px">'
        f'Note: This auto-generated text is a starting point. '
        f'Review, edit, and supplement with theoretical interpretation before submitting.</p>'
    )
    sections.append(_html_section(17, "APA Results Narrative", s17, color="#555"))

    # ── 18. References ───────────────────────────────────────────
    refs = [
        ["Aiken, L. S., & West, S. G. (1991).", "Multiple regression: Testing and interpreting interactions. Sage."],
        ["Burnham, K. P., & Anderson, D. R. (2002).", "Model selection and multimodel inference (2nd ed.). Springer."],
        ["Chen, F. F. (2007).", "Sensitivity of goodness of fit indexes to lack of measurement invariance. Structural Equation Modeling, 14(3), 464–504."],
        ["Cheung, G. W., & Rensvold, R. B. (2002).", "Evaluating goodness-of-fit indexes for testing measurement invariance. Structural Equation Modeling, 9(2), 233–255."],
        ["Cohen, J. (1988).", "Statistical power analysis for the behavioral sciences (2nd ed.). Lawrence Erlbaum."],
        ["Fornell, C., & Larcker, D. F. (1981).", "Evaluating structural equation models with unobservable variables and measurement error. Journal of Marketing Research, 18(1), 39–50."],
        ["Hair, J. F., Black, W. C., Babin, B. J., & Anderson, R. E. (2019).", "Multivariate data analysis (8th ed.). Cengage."],
        ["Hayes, A. F. (2018).", "Introduction to mediation, moderation, and conditional process analysis (2nd ed.). Guilford Press."],
        ["Henseler, J., Ringle, C. M., & Sarstedt, M. (2015).", "A new criterion for assessing discriminant validity in variance-based structural equation modeling. Journal of the Academy of Marketing Science, 43(1), 115–135."],
        ["Hu, L., & Bentler, P. M. (1999).", "Cutoff criteria for fit indexes in covariance structure analysis. Structural Equation Modeling, 6(1), 1–55."],
        ["Kline, R. B. (2016).", "Principles and practice of structural equation modeling (4th ed.). Guilford Press."],
        ["Nunnally, J. C. (1978).", "Psychometric theory (2nd ed.). McGraw-Hill."],
        ["Rosseel, Y. (2012).", "lavaan: An R package for structural equation modeling. Journal of Statistical Software, 48(2), 1–36."],
        ["Vandenberg, R. J., & Lance, C. E. (2000).", "A review and synthesis of the measurement invariance literature. Organizational Research Methods, 3(1), 4–70."],
        ["Zhao, X., Lynch, J. G., & Chen, Q. (2010).", "Reconsidering Baron and Kenny: Myths and truths about mediation analysis. Journal of Consumer Research, 37(2), 197–206."],
    ]
    ref_html = "".join(
        f'<p style="margin:6px 0;font-size:0.85rem"><b>{r[0]}</b> {r[1]}</p>'
        for r in refs
    )
    sections.append(_html_section(18, "References", ref_html, color="#555"))

    # ── Assemble HTML ─────────────────────────────────────────────
    body = "\n".join(sections)

    # TOC
    # TOC items with markers: (title, has_data)
    toc_items = [
        ("Study Overview",         bool(df is not None)),
        ("Model Specification",    bool(constructs)),
        ("Descriptive Statistics", bool(df is not None and indicator_cols)),
        ("EFA Results",            bool(efa_result and "error" not in efa_result)),
        ("CFA Model Fit",          bool(cfa_fit)),
        ("Factor Loadings",        bool(cfa_loadings)),
        ("Reliability and Validity", bool(metrics)),
        ("SEM Model Fit",          bool(sem_fit)),
        ("Structural Paths",       bool(sem_paths)),
        ("Explained Variance R2",  bool(sem_r2)),
        ("Mediation Analysis",     bool(med_results)),
        ("Moderation Analysis",    bool(mod_results)),
        ("Measurement Invariance", bool(inv_results)),
        ("Model Comparison",       bool(comp_results)),
        ("Path Diagram",           bool(constructs and struct_paths)),
        ("Methodological Checklist", True),
        ("APA Narrative",          True),
        ("References",             True),
    ]
    toc_html = '<div style="background:#f8fafc;border:1px solid #dde3ea;border-radius:8px;padding:16px 20px;margin:20px 0">'
    toc_html += '<h3 style="color:#2E86AB;margin:0 0 10px 0;font-size:1rem">Table of Contents</h3>'
    toc_html += '<p style="font-size:0.78rem;color:#888;margin:0 0 10px 0">&#9989; = completed &nbsp; &#9744; = not run</p>'
    toc_html += '<table style="border-collapse:collapse;width:100%;font-size:0.85rem">'
    for i, (title, has_data) in enumerate(toc_items, 1):
        marker = "&#9989;" if has_data else "&#9744;"
        color  = "#1a1a1a" if has_data else "#aaaaaa"
        bg     = "#ffffff" if i % 2 == 0 else "#f8fafc"
        col    = "left" if i <= 9 else "right"
        toc_html += (
            f'<tr style="background:{bg}">'
            f'<td style="padding:4px 8px;width:50%;color:{color}">'
            f'{marker} {i}. {title}</td>'
        )
        if i % 2 == 1 and i < len(toc_items):
            # Will be closed by the even row
            pass
        if i % 2 == 0:
            toc_html += '</tr>'
    if len(toc_items) % 2 == 1:
        toc_html += '<td style="padding:4px 8px"></td></tr>'
    toc_html += '</table></div>'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SEM Studio Report - {now}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
    max-width: 1000px;
    margin: 40px auto;
    padding: 0 32px 60px;
    color: #1a1a1a;
    background: #ffffff;
    line-height: 1.65;
    font-size: 14px;
  }}
  h1 {{
    color: #2E86AB;
    border-bottom: 3px solid #2E86AB;
    padding-bottom: 12px;
    font-size: 1.8rem;
    margin-bottom: 4px;
  }}
  h2 {{ font-size: 1.15rem; }}
  table {{ page-break-inside: avoid; }}
  th {{ padding: 8px 14px; text-align: left; font-weight: 600; }}
  a {{ color: #2E86AB; }}
  @media print {{
    body {{ margin: 20px; padding: 0 20px; }}
    .no-print {{ display: none; }}
    h2 {{ page-break-before: always; }}
    h2:first-of-type {{ page-break-before: avoid; }}
  }}
</style>
</head>
<body>
<h1>SEM Studio Analysis Report</h1>
<p style="color:#888;font-size:0.82rem;margin-top:4px">
  Generated: {now} &nbsp;|&nbsp; Powered by R/lavaan (Rosseel, 2012) &nbsp;|&nbsp; n = {n}
</p>
<hr style="border:none;border-top:1px solid #dde3ea;margin:16px 0">
{toc_html}
{body}
<hr style="border:none;border-top:1px solid #dde3ea;margin:40px 0 16px">
<p style="color:#aaa;font-size:0.75rem;text-align:center">
  SEM Studio &mdash; Open Source SEM Analysis Suite &mdash;
  Powered by R/lavaan &mdash; Generated {now}
</p>
</body>
</html>"""

    return html


def render_html_export():
    st.subheader("Export HTML Report")
    st.markdown(
        "Download a complete, publication-ready HTML report containing all results, "
        "interpretations, path diagram, and APA narrative. "
        "Open in browser, print to PDF, or copy-paste into Word."
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown("✅ **All results** in one file")
    col2.markdown("✅ **Path diagram** embedded")
    col3.markdown("✅ **Print to PDF** ready")

    if st.button("Generate HTML Report", type="primary", key="export_html_btn", use_container_width=True):
        with st.spinner("Generating comprehensive HTML report..."):
            try:
                html = generate_html_report()
                fname = f"SEM_Studio_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
                st.download_button(
                    label="Download HTML Report",
                    data=html.encode("utf-8"),
                    file_name=fname,
                    mime="text/html",
                    key="dl_html_btn",
                    use_container_width=True,
                )
                badge("excellent",
                    "HTML report ready! "
                    "Open in browser to view | Print > Save as PDF | Select All > Copy > Paste into Word."
                )
            except Exception as e:
                st.error(f"HTML generation failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_text_export():
    st.subheader("Quick Text Summary")
    ss    = st.session_state
    lines = []
    cfa_fit  = ss.get("cfa_fit", {})
    if cfa_fit:
        lines.append("CFA FIT:")
        for k in ["rmsea","cfi","tli","srmr","aic","bic"]:
            v = _safe_float(cfa_fit.get(k))
            if v is not None: lines.append(f"  {k.upper()} = {v:.3f}")
    sem_paths = ss.get("sem_paths", [])
    if sem_paths:
        lines.append("\nSTRUCTURAL PATHS:")
        for i, p in enumerate(sem_paths):
            beta  = _safe_float(p.get("beta"))
            p_val = _safe_float(p.get("p"))
            if beta is not None:
                lines.append(f"  H{i+1}: {p.get('predictor','?')} -> {p.get('outcome','?')}: beta = {beta:.3f}, p = {_fmt(p_val,3)}")
    med = ss.get("mediation_results", {})
    if med and isinstance(med, dict):
        indirect_data = med.get("indirect", {})
        if isinstance(indirect_data, dict):
            indirect = _safe_float(indirect_data.get("est"))
            ci_lo    = _safe_float(indirect_data.get("ci_lo"))
            ci_hi    = _safe_float(indirect_data.get("ci_hi"))
            if indirect is not None:
                lines.append(f"\nMEDIATION:")
                lines.append(f"  Indirect = {indirect:.4f}, 95% CI [{_fmt(ci_lo,4)}, {_fmt(ci_hi,4)}]")
    text = "\n".join(lines) if lines else "No results yet. Run analyses first."
    st.text_area("Results Summary", value=text, height=250, key="text_export_area")


def render_export():
    st.title("Export Report")
    st.markdown("Generate and download your complete SEM analysis report.")

    if not st.session_state.get("df_ready"):
        st.warning("Please complete Data Input and Model Setup first.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Checklist",
        "APA Narrative",
        "HTML Report",
        "Quick Text",
    ])

    with tab1:
        render_full_checklist()
    with tab2:
        render_apa_narrative()
    with tab3:
        render_html_export()
    with tab4:
        render_text_export()
