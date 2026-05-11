"""
apa_tables.py
=============
APA 7th edition table formatting utilities for SEM Studio.
Generates publication-ready tables as pandas DataFrames and HTML.
"""

import pandas as pd
import numpy as np


def style_df(df: pd.DataFrame, highlight_cols: list = None) -> pd.io.formats.style.Styler:
    """Apply minimal APA-like styling to a DataFrame."""
    styler = df.style.set_properties(**{
        'font-size': '13px',
        'text-align': 'center',
        'border': '1px solid #444',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1e2130'),
            ('color', '#ffffff'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('padding', '8px 12px'),
            ('border-bottom', '2px solid #2E86AB'),
        ]},
        {'selector': 'td', 'props': [
            ('padding', '6px 12px'),
            ('color', '#f0f0f0'),
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#252a3d'),
        ]},
    ])

    if highlight_cols:
        for col in highlight_cols:
            if col in df.columns:
                styler = styler.background_gradient(subset=[col], cmap='RdYlGn')

    return styler


def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate APA-style descriptive statistics table.

    Returns columns: Variable, N, Mean, SD, Min, Max, Skewness, Kurtosis, Missing%
    """
    from scipy import stats

    records = []
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col].dropna()
        records.append({
            "Variable":  col,
            "N":         int(s.count()),
            "Mean":      round(s.mean(), 3),
            "SD":        round(s.std(), 3),
            "Min":       round(s.min(), 3),
            "Max":       round(s.max(), 3),
            "Skewness":  round(float(stats.skew(s)), 3),
            "Kurtosis":  round(float(stats.kurtosis(s)), 3),
            "Missing %": f"{df[col].isna().mean():.1%}",
        })

    return pd.DataFrame(records)


def correlation_table(df: pd.DataFrame, vars: list = None) -> pd.DataFrame:
    """
    Generate APA-style correlation matrix.
    Lower triangle only, with significance markers.
    """
    from scipy import stats

    if vars:
        df = df[vars]
    numeric = df.select_dtypes(include=[np.number])
    cols = numeric.columns.tolist()
    n = len(cols)

    corr_df = pd.DataFrame(index=cols, columns=cols, dtype=object)

    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                corr_df.loc[c1, c2] = "—"
            elif i > j:
                combined = numeric[[c1, c2]].dropna()
                r, p = stats.pearsonr(combined[c1], combined[c2])
                stars = ""
                if p < 0.001: stars = "***"
                elif p < 0.01: stars = "**"
                elif p < 0.05: stars = "*"
                corr_df.loc[c1, c2] = f"{r:.2f}{stars}"
            else:
                corr_df.loc[c1, c2] = ""

    return corr_df


def fit_indices_table(fit_dict: dict) -> pd.DataFrame:
    """
    Generate APA-style fit indices summary table.
    fit_dict: {'chi2': x, 'df': x, 'p': x, 'rmsea': x, 'cfi': x, ...}
    """
    from utils.thresholds import FIT

    rows = []

    if "chi2" in fit_dict and "df" in fit_dict:
        ratio = fit_dict["chi2"] / fit_dict["df"] if fit_dict["df"] > 0 else None
        rows.append({
            "Index": "χ²/df",
            "Value": f"{ratio:.3f}" if ratio else "—",
            "Criterion": "≤ 2.00 (good); ≤ 5.00 (acceptable)",
            "Reference": "Kline (2016)",
        })

    index_map = {
        "rmsea": ("RMSEA", f"≤ {FIT['rmsea_excellent']:.2f} (excellent); ≤ {FIT['rmsea_acceptable']:.2f} (acceptable)", "Hu & Bentler (1999)"),
        "cfi":   ("CFI",   f"≥ {FIT['cfi_good']:.2f} (good); ≥ {FIT['cfi_acceptable']:.2f} (acceptable)", "Hu & Bentler (1999)"),
        "tli":   ("TLI",   f"≥ {FIT['tli_good']:.2f} (good)", "Tucker & Lewis (1973)"),
        "srmr":  ("SRMR",  f"≤ {FIT['srmr_acceptable']:.2f}", "Hu & Bentler (1999)"),
        "gfi":   ("GFI",   f"≥ {FIT['gfi_acceptable']:.2f}", "Jöreskog & Sörbom (1984)"),
        "nfi":   ("NFI",   f"≥ {FIT['nfi_acceptable']:.2f}", "Bentler & Bonett (1980)"),
        "agfi":  ("AGFI",  f"≥ {FIT['agfi_acceptable']:.2f}", "Jöreskog & Sörbom (1984)"),
        "aic":   ("AIC",   "Lower is better (model comparison)", "Akaike (1974)"),
        "bic":   ("BIC",   "Lower is better (model comparison)", "Schwarz (1978)"),
    }

    for key, (label, criterion, ref) in index_map.items():
        if key in fit_dict:
            val = fit_dict[key]
            rows.append({
                "Index": label,
                "Value": f"{val:.3f}" if isinstance(val, float) else str(val),
                "Criterion": criterion,
                "Reference": ref,
            })

    return pd.DataFrame(rows)


def factor_loadings_table(loadings_dict: dict) -> pd.DataFrame:
    """
    Generate APA-style factor loadings table.
    loadings_dict: {'ConstructName': {'Item1': 0.75, 'Item2': 0.82}, ...}
    """
    rows = []
    for construct, items in loadings_dict.items():
        first = True
        for item, loading in items.items():
            rows.append({
                "Construct": construct if first else "",
                "Item": item,
                "Standardized Loading (λ)": f"{loading:.3f}",
                "Note": "✅" if abs(loading) >= 0.70 else ("⚠️" if abs(loading) >= 0.50 else "❌"),
            })
            first = False

    return pd.DataFrame(rows)


def reliability_validity_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Generate reliability and validity summary table.
    metrics_dict: {'ConstructName': {'alpha': x, 'cr': x, 'omega': x, 'ave': x}, ...}
    """
    rows = []
    for construct, m in metrics_dict.items():
        rows.append({
            "Construct":           construct,
            "Cronbach's α":        f"{m.get('alpha', None):.3f}" if m.get('alpha') else "—",
            "Composite Reliability (CR)": f"{m.get('cr', None):.3f}" if m.get('cr') else "—",
            "McDonald's ω":        f"{m.get('omega', None):.3f}" if m.get('omega') else "—",
            "AVE":                 f"{m.get('ave', None):.3f}" if m.get('ave') else "—",
            "α ≥ .70":             "✅" if m.get('alpha', 0) >= 0.70 else "❌",
            "CR ≥ .70":            "✅" if m.get('cr', 0) >= 0.70 else "❌",
            "AVE ≥ .50":           "✅" if m.get('ave', 0) >= 0.50 else "❌",
        })

    return pd.DataFrame(rows)


def htmt_table(htmt_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Format HTMT matrix with pass/fail indicators.
    """
    from utils.thresholds import CFA

    result = htmt_matrix.copy().astype(object)
    for i in result.index:
        for j in result.columns:
            val = htmt_matrix.loc[i, j]
            if pd.isna(val) or i == j:
                result.loc[i, j] = "—"
            elif isinstance(val, float):
                flag = "✅" if val < CFA["htmt_strict"] else ("⚠️" if val < CFA["htmt_liberal"] else "❌")
                result.loc[i, j] = f"{val:.3f} {flag}"

    return result


def structural_paths_table(paths: list) -> pd.DataFrame:
    """
    Generate APA-style structural paths table.
    paths: list of dicts with keys: predictor, outcome, beta, se, z, p, ci_lower, ci_upper
    """
    rows = []
    for p in paths:
        pval = p.get("p", 1.0)
        stars = ""
        if pval < 0.001: stars = "***"
        elif pval < 0.01: stars = "**"
        elif pval < 0.05: stars = "*"
        elif pval < 0.10: stars = "†"

        rows.append({
            "Path":          f"{p.get('predictor', '?')} → {p.get('outcome', '?')}",
            "β (std)":       f"{p.get('beta', 0):.3f}{stars}",
            "SE":            f"{p.get('se', 0):.3f}",
            "z":             f"{p.get('z', 0):.3f}",
            "p":             f"{pval:.3f}",
            "95% CI":        f"[{p.get('ci_lower', 0):.3f}, {p.get('ci_upper', 0):.3f}]"
                             if p.get('ci_lower') is not None else "—",
            "Sig.":          "✅" if pval < 0.05 else ("⚠️" if pval < 0.10 else "❌"),
        })

    note_df = pd.DataFrame(rows)
    return note_df
