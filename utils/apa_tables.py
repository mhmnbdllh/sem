"""
apa_tables.py - APA 7th edition table formatting for SEM Studio.
"""
import pandas as pd
import numpy as np
from utils.thresholds import FIT, CFA


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

def _fmt(val, decimals=3):
    if val is None: return "—"
    try:
        if np.isnan(float(val)): return "—"
        return f"{float(val):.{decimals}f}"
    except: return str(val)

def _safe_float(val):
    if val is None: return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except: return None


def descriptive_table(desc_results: dict) -> pd.DataFrame:
    rows = []
    for var, stats in desc_results.items():
        if isinstance(stats, dict):
            rows.append({
                "Variable": var,
                "N":        stats.get("n", "—"),
                "Mean":     _fmt(stats.get("mean")),
                "SD":       _fmt(stats.get("sd")),
                "Min":      _fmt(stats.get("min")),
                "Max":      _fmt(stats.get("max")),
                "Skewness": _fmt(stats.get("skewness")),
                "Kurtosis": _fmt(stats.get("kurtosis")),
                "Missing":  stats.get("missing", 0),
            })
    return pd.DataFrame(rows)


def correlation_table(df: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    from scipy import stats as scipy_stats
    if cols:
        df = df[cols].copy()
    numeric = df.select_dtypes(include=[np.number])
    col_list = numeric.columns.tolist()
    result = pd.DataFrame("", index=col_list, columns=col_list)
    for i, c1 in enumerate(col_list):
        for j, c2 in enumerate(col_list):
            if i == j:
                result.loc[c1, c2] = "—"
            elif i > j:
                combined = numeric[[c1, c2]].dropna()
                if len(combined) < 3:
                    result.loc[c1, c2] = "—"
                    continue
                r, p = scipy_stats.pearsonr(combined[c1], combined[c2])
                result.loc[c1, c2] = f"{r:.2f}{_stars(p)}"
            else:
                result.loc[c1, c2] = ""
    return result


def fit_indices_table(fit_dict: dict) -> pd.DataFrame:
    key_map = {
        "chisq": "chi2", "Chi2": "chi2",
        "pvalue": "p", "p.value": "p",
        "dof": "df", "DoF": "df",
    }
    fit = {}
    for k, v in fit_dict.items():
        normalized = key_map.get(k, k.lower().replace(".", "_"))
        fit[normalized] = v

    rows = []
    chi2 = fit.get("chi2") or fit.get("chisq_scaled")
    df_  = fit.get("df")
    p_   = fit.get("p") or fit.get("pvalue")

    if chi2 is not None and df_ is not None:
        rows.append({
            "Index":     "chi2",
            "Value":     f"{float(chi2):.3f}",
            "Criterion": f"df = {int(float(df_))}, p = {_fmt(p_, 3)}",
            "Reference": "Kline (2016)",
        })
        if float(df_) > 0:
            ratio = float(chi2) / float(df_)
            label = "Good" if ratio <= 2 else "Acceptable" if ratio <= 5 else "Poor"
            rows.append({
                "Index":     "chi2/df",
                "Value":     f"{ratio:.3f}",
                "Criterion": f"<=2.0 (good); <=5.0 (acceptable) [{label}]",
                "Reference": "Kline (2016)",
            })

    index_map = [
        ("rmsea", "RMSEA", "<=.05 (excellent); <=.06 (good); <=.08 (acceptable)", "Hu & Bentler (1999)"),
        ("cfi",   "CFI",   ">=.97 (excellent); >=.95 (good); >=.90 (acceptable)",  "Hu & Bentler (1999)"),
        ("tli",   "TLI",   ">=.95 (good); >=.90 (acceptable)",                     "Tucker & Lewis (1973)"),
        ("srmr",  "SRMR",  "<=.05 (good); <=.08 (acceptable)",                     "Hu & Bentler (1999)"),
        ("gfi",   "GFI",   ">=.95 (good); >=.90 (acceptable)",                     "Joreskog & Sorbom (1984)"),
        ("nfi",   "NFI",   ">=.95 (good); >=.90 (acceptable)",                     "Bentler & Bonett (1980)"),
        ("aic",   "AIC",   "Lower is better (model comparison)",                    "Akaike (1974)"),
        ("bic",   "BIC",   "Lower is better (model comparison)",                    "Schwarz (1978)"),
    ]

    for key, label, criterion, ref in index_map:
        val = fit.get(key) or fit.get(f"{key}_scaled") or fit.get(f"{key}scaled")
        if val is not None:
            try:
                val_f = float(val)
                from utils.thresholds import interpret_fit
                _, _, level = interpret_fit(key, val_f)
                indicator = "OK" if level in ("excellent","good","info") else "Warning" if level == "acceptable" else "Poor" if level == "poor" else "-"
                rows.append({
                    "Index":     label,
                    "Value":     f"{val_f:.3f} [{indicator}]",
                    "Criterion": criterion,
                    "Reference": ref,
                })
            except: pass

    return pd.DataFrame(rows)


def factor_loadings_table(loadings_df, source: str = "cfa") -> pd.DataFrame:
    rows = []
    if source == "cfa" and isinstance(loadings_df, pd.DataFrame):
        if "construct" in loadings_df.columns:
            for _, row in loadings_df.iterrows():
                loading = row.get("std", row.get("unstd", None))
                p_val   = row.get("p", None)
                status = (
                    "Strong" if loading is not None and abs(float(loading)) >= CFA["loading_good"]
                    else "Acceptable" if loading is not None and abs(float(loading)) >= CFA["loading_min"]
                    else "Weak"
                )
                rows.append({
                    "Construct":        row.get("construct", "—"),
                    "Item":             row.get("item", "—"),
                    "Std. Loading (lambda)": _fmt(loading),
                    "p":                _fmt(p_val, 3),
                    "Sig.":             _stars(p_val),
                    "Status":           status,
                })
    elif source == "efa" and isinstance(loadings_df, pd.DataFrame):
        factor_cols = [c for c in loadings_df.columns if c.startswith("F")]
        for item in loadings_df.index:
            row_data = {"Item": item}
            for f in factor_cols:
                row_data[f] = _fmt(loadings_df.loc[item, f])
            abs_vals = loadings_df.loc[item, factor_cols].abs() if factor_cols else None
            row_data["Primary"] = abs_vals.idxmax() if abs_vals is not None else "—"
            rows.append(row_data)
    return pd.DataFrame(rows)


def reliability_validity_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for construct, m in metrics.items():
        if not isinstance(m, dict): continue
        alpha = _safe_float(m.get("alpha"))
        cr    = _safe_float(m.get("cr"))
        omega = _safe_float(m.get("omega"))
        ave   = _safe_float(m.get("ave"))
        rows.append({
            "Construct":    construct,
            "Items":        m.get("n_items", "—"),
            "Cronbach a":   _fmt(alpha),
            "CR":           _fmt(cr),
            "McDonald w":   _fmt(omega),
            "AVE":          _fmt(ave),
            "a>=.70":      "Pass" if alpha and alpha >= CFA["alpha_min"] else "Fail",
            "CR>=.70":     "Pass" if cr    and cr    >= CFA["cr_min"]    else "Fail",
            "AVE>=.50":    "Pass" if ave   and ave   >= CFA["ave_min"]   else "Fail",
        })
    return pd.DataFrame(rows)


def htmt_table(htmt_matrix) -> pd.DataFrame:
    if not isinstance(htmt_matrix, pd.DataFrame):
        try: htmt_matrix = pd.DataFrame(htmt_matrix)
        except: return pd.DataFrame()
    result = htmt_matrix.copy().astype(object)
    for i in result.index:
        for j in result.columns:
            val = htmt_matrix.loc[i, j]
            if i == j or pd.isna(val):
                result.loc[i, j] = "—"
            else:
                try:
                    v = float(val)
                    flag = "OK" if v < CFA["htmt_strict"] else "Warning" if v < CFA["htmt_liberal"] else "Fail"
                    result.loc[i, j] = f"{v:.3f} [{flag}]"
                except: result.loc[i, j] = "—"
    return result


def structural_paths_table(paths_data, hypotheses: list = None) -> pd.DataFrame:
    rows = []
    if isinstance(paths_data, pd.DataFrame):
        for i, row in paths_data.iterrows():
            pred  = row.get("predictor", row.get("rhs", "?"))
            out   = row.get("outcome",   row.get("lhs", "?"))
            beta  = row.get("beta", row.get("std", row.get("std.all", None)))
            se    = row.get("se", None)
            z     = row.get("z", None)
            p_val = row.get("p", row.get("pvalue", None))
            stars = _stars(p_val)
            rows.append({
                "H":        f"H{i+1}",
                "Path":     f"{pred} -> {out}",
                "Beta":     f"{float(beta):.3f}{stars}" if beta is not None else "—",
                "SE":       _fmt(se),
                "z":        _fmt(z),
                "p":        _fmt(p_val),
                "Decision": "Supported" if p_val is not None and float(p_val) < 0.05 else "Not Supported",
            })
    elif isinstance(paths_data, list):
        for i, p in enumerate(paths_data):
            p_val = p.get("p")
            beta  = p.get("beta")
            rows.append({
                "H":        f"H{i+1}",
                "Path":     f"{p.get('predictor','?')} -> {p.get('outcome','?')}",
                "Beta":     f"{float(beta):.3f}{_stars(p_val)}" if beta is not None else "—",
                "SE":       _fmt(p.get("se")),
                "z":        _fmt(p.get("z")),
                "p":        _fmt(p_val),
                "Decision": "Supported" if p_val is not None and float(p_val) < 0.05 else "Not Supported",
            })
    return pd.DataFrame(rows)


def mediation_table(med_results: dict) -> pd.DataFrame:
    rows = []
    effects = [
        ("a_path",  "a path (X -> M)"),
        ("b_path",  "b path (M -> Y | X)"),
        ("cp_path", "c prime path (direct)"),
        ("total",   "Total effect (c)"),
        ("indirect","Indirect effect (a x b)"),
    ]
    for key, label in effects:
        data = med_results.get(key, {})
        if not isinstance(data, dict): continue
        est   = data.get("est")
        se    = data.get("se")
        p_val = data.get("p")
        ci_lo = data.get("ci_lo")
        ci_hi = data.get("ci_hi")
        ci_str = f"[{float(ci_lo):.4f}, {float(ci_hi):.4f}]" if ci_lo is not None and ci_hi is not None else "—"
        sig = ""
        if key == "indirect":
            if ci_lo is not None and ci_hi is not None:
                sig = "Significant" if not (float(ci_lo) <= 0 <= float(ci_hi)) else "Not Significant"
        elif p_val is not None:
            sig = "Sig." if float(p_val) < 0.05 else "n.s."
        rows.append({
            "Effect": label,
            "Beta":   _fmt(est),
            "SE":     _fmt(se),
            "p":      _fmt(p_val) if p_val is not None else "—",
            "95% CI": ci_str,
            "Sig.":   sig,
        })
    return pd.DataFrame(rows)
