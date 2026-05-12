"""
thresholds.py - Methodological cut-off standards for SEM Studio.
"""

SAMPLE = {
    "minimum_cfa": 100, "recommended_cfa": 200,
    "minimum_sem": 200, "recommended_sem": 300,
    "optimal_sem": 500, "per_parameter": 10,
}
DESCRIPTIVE = {
    "skewness_strict": 1.0, "skewness_acceptable": 2.0,
    "kurtosis_strict": 3.0, "kurtosis_acceptable": 7.0,
    "missing_acceptable": 0.05, "missing_critical": 0.10,
}
OUTLIER = {"mahalanobis_p": 0.001, "z_threshold": 3.29}
EFA = {
    "kmo_unacceptable": 0.50, "kmo_mediocre": 0.60,
    "kmo_middling": 0.70, "kmo_meritorious": 0.80, "kmo_marvelous": 0.90,
    "bartlett_p": 0.05, "eigenvalue_min": 1.0,
    "loading_poor": 0.32, "loading_acceptable": 0.40,
    "loading_good": 0.50, "loading_strong": 0.70,
    "cross_loading_max": 0.32, "communality_min": 0.40, "variance_min": 0.50,
}
CFA = {
    "loading_min": 0.50, "loading_good": 0.70,
    "ave_min": 0.50, "cr_min": 0.70,
    "alpha_min": 0.70, "alpha_acceptable": 0.60, "omega_min": 0.70,
    "htmt_strict": 0.85, "htmt_liberal": 0.90,
}
FIT = {
    "chisq_df_good": 2.0, "chisq_df_acceptable": 5.0,
    "rmsea_excellent": 0.05, "rmsea_good": 0.06,
    "rmsea_acceptable": 0.08, "rmsea_poor": 0.10,
    "cfi_excellent": 0.97, "cfi_good": 0.95, "cfi_acceptable": 0.90,
    "tli_good": 0.95, "tli_acceptable": 0.90,
    "srmr_good": 0.05, "srmr_acceptable": 0.08,
    "gfi_good": 0.95, "gfi_acceptable": 0.90,
    "nfi_good": 0.95, "nfi_acceptable": 0.90,
}
SEM = {
    "p_significant": 0.05, "p_marginal": 0.10,
    "r2_weak": 0.02, "r2_moderate": 0.13, "r2_strong": 0.26,
    "f2_small": 0.02, "f2_medium": 0.15, "f2_large": 0.35,
}
MEDIATION = {"bootstrap_min": 1000, "bootstrap_rec": 5000, "ci_level": 0.95}
INVARIANCE = {
    "delta_cfi_criterion": -0.010, "delta_rmsea_criterion": 0.015,
    "mi_threshold": 3.84, "mi_notable": 10.0,
}
CORRELATION = {
    "multicollinearity_warning": 0.85,
    "multicollinearity_critical": 0.90,
    "item_total_min": 0.30,
}

def interpret_fit(index: str, value: float) -> tuple:
    if value is None:
        return ("—", "#999999", "unknown")
    COLORS = {
        "excellent": "#1a7a4a", "good": "#2ecc71",
        "acceptable": "#f39c12", "poor": "#e74c3c",
    }
    def res(level, label):
        return (label, COLORS[level], level)
    f = FIT
    if index == "rmsea":
        if value <= f["rmsea_excellent"]:    return res("excellent", "✅ Excellent (≤ .05)")
        elif value <= f["rmsea_good"]:       return res("good",      "✅ Good (≤ .06)")
        elif value <= f["rmsea_acceptable"]: return res("acceptable","⚠️ Acceptable (≤ .08)")
        else:                                return res("poor",      "❌ Poor (> .08)")
    elif index in ("cfi", "tli"):
        if value >= f["cfi_excellent"]:      return res("excellent", "✅ Excellent (≥ .97)")
        elif value >= f["cfi_good"]:         return res("good",      "✅ Good (≥ .95)")
        elif value >= f["cfi_acceptable"]:   return res("acceptable","⚠️ Acceptable (≥ .90)")
        else:                                return res("poor",      "❌ Poor (< .90)")
    elif index == "srmr":
        if value <= f["srmr_good"]:          return res("excellent", "✅ Excellent (≤ .05)")
        elif value <= f["srmr_acceptable"]:  return res("acceptable","⚠️ Acceptable (≤ .08)")
        else:                                return res("poor",      "❌ Poor (> .08)")
    elif index == "chisq_df":
        if value <= f["chisq_df_good"]:      return res("good",      "✅ Good (≤ 2.0)")
        elif value <= f["chisq_df_acceptable"]: return res("acceptable","⚠️ Acceptable (≤ 5.0)")
        else:                                return res("poor",      "❌ Poor (> 5.0)")
    return ("—", "#999999", "unknown")
