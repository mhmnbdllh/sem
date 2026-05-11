"""
thresholds.py
=============
Methodological cut-off standards for SEM/CFA analyses.
All thresholds are sourced from peer-reviewed literature.

References:
    - Hair et al. (2019). Multivariate Data Analysis (8th ed.)
    - Hu & Bentler (1999). Cutoff criteria for fit indexes.
    - Fornell & Larcker (1981). Evaluating structural equation models.
    - Kline (2016). Principles and Practice of SEM (4th ed.).
    - Brown (2015). CFA for Applied Research (2nd ed.).
"""

# ─── SAMPLE SIZE ────────────────────────────────────────────────
SAMPLE_SIZE = {
    "minimum_cfa": 100,
    "recommended_cfa": 200,
    "minimum_sem": 200,
    "recommended_sem": 300,
    "optimal_sem": 500,
    "rule_of_thumb_per_parameter": 10,
}

# ─── DESCRIPTIVE STATISTICS ─────────────────────────────────────
DESCRIPTIVE = {
    "skewness_acceptable": 2.0,       # |skewness| < 2 (Hair et al., 2019)
    "skewness_strict": 1.0,           # |skewness| < 1 (strict)
    "kurtosis_acceptable": 7.0,       # |kurtosis| < 7 (Hair et al., 2019)
    "kurtosis_strict": 3.0,           # |kurtosis| < 3 (strict)
    "missing_acceptable": 0.05,       # < 5% missing per variable
    "missing_critical": 0.10,         # > 10% = critical problem
}

# ─── MULTIVARIATE NORMALITY ─────────────────────────────────────
NORMALITY = {
    "mardia_skewness_pvalue": 0.05,
    "mardia_kurtosis_pvalue": 0.05,
}

# ─── OUTLIER DETECTION ──────────────────────────────────────────
OUTLIER = {
    "mahalanobis_pvalue": 0.001,      # p < .001 = outlier
    "z_score_threshold": 3.29,        # |z| > 3.29 = univariate outlier
}

# ─── FACTOR ANALYSIS ────────────────────────────────────────────
EFA = {
    "kmo_acceptable": 0.60,           # KMO > .60 acceptable
    "kmo_good": 0.70,
    "kmo_great": 0.80,
    "bartlett_pvalue": 0.05,          # Bartlett's test p < .05
    "eigenvalue_min": 1.0,            # Kaiser criterion
    "factor_loading_acceptable": 0.40,
    "factor_loading_good": 0.50,
    "cross_loading_max": 0.32,        # Cross-loading threshold
    "communality_min": 0.40,
    "variance_explained_min": 0.50,   # Total variance explained > 50%
}

# ─── CFA / MEASUREMENT MODEL ────────────────────────────────────
CFA = {
    "factor_loading_min": 0.50,       # Minimum acceptable (Hair et al.)
    "factor_loading_good": 0.70,      # Recommended (Hair et al.)
    "ave_min": 0.50,                  # AVE ≥ .50 (Fornell & Larcker, 1981)
    "cr_min": 0.70,                   # Composite Reliability ≥ .70
    "alpha_min": 0.70,                # Cronbach's α ≥ .70 (Nunnally, 1978)
    "alpha_acceptable": 0.60,         # α ≥ .60 (exploratory research)
    "omega_min": 0.70,
    "htmt_strict": 0.85,              # HTMT < .85 (Henseler et al., 2015)
    "htmt_liberal": 0.90,             # HTMT < .90 (Gold et al., 2001)
}

# ─── GOODNESS OF FIT INDICES ────────────────────────────────────
FIT = {
    # Chi-square
    "chisq_df_ratio_good": 2.0,       # χ²/df ≤ 2 (good)
    "chisq_df_ratio_acceptable": 5.0, # χ²/df ≤ 5 (acceptable)

    # RMSEA — Root Mean Square Error of Approximation
    "rmsea_excellent": 0.05,          # ≤ .05 (Browne & Cudeck, 1993)
    "rmsea_good": 0.06,               # ≤ .06 (Hu & Bentler, 1999)
    "rmsea_acceptable": 0.08,         # ≤ .08 (Browne & Cudeck, 1993)
    "rmsea_poor": 0.10,               # > .10 = poor fit

    # CFI — Comparative Fit Index
    "cfi_excellent": 0.97,            # ≥ .97 (excellent)
    "cfi_good": 0.95,                 # ≥ .95 (Hu & Bentler, 1999)
    "cfi_acceptable": 0.90,           # ≥ .90 (acceptable)

    # TLI — Tucker-Lewis Index
    "tli_good": 0.95,
    "tli_acceptable": 0.90,

    # SRMR — Standardized Root Mean Square Residual
    "srmr_good": 0.05,
    "srmr_acceptable": 0.08,          # ≤ .08 (Hu & Bentler, 1999)

    # GFI — Goodness of Fit Index
    "gfi_good": 0.95,
    "gfi_acceptable": 0.90,

    # NFI — Normed Fit Index
    "nfi_good": 0.95,
    "nfi_acceptable": 0.90,

    # AGFI — Adjusted GFI
    "agfi_acceptable": 0.85,

    # PNFI — Parsimony NFI
    "pnfi_acceptable": 0.50,
}

# ─── STRUCTURAL MODEL ───────────────────────────────────────────
SEM = {
    "path_significant_pvalue": 0.05,
    "path_marginal_pvalue": 0.10,
    "r2_weak": 0.02,                  # Cohen (1988)
    "r2_moderate": 0.13,
    "r2_strong": 0.26,
    "effect_size_small": 0.02,
    "effect_size_medium": 0.15,
    "effect_size_large": 0.35,
}

# ─── MEDIATION ──────────────────────────────────────────────────
MEDIATION = {
    "bootstrap_samples": 5000,
    "ci_level": 0.95,
    "significance_level": 0.05,
}

# ─── MODIFICATION INDICES ───────────────────────────────────────
MODIFICATION = {
    "mi_threshold": 3.84,             # χ²(1, p=.05) = 3.84
    "mi_notable": 10.0,               # MI > 10 = notable
}

# ─── CORRELATION ────────────────────────────────────────────────
CORRELATION = {
    "multicollinearity_warning": 0.85,
    "multicollinearity_critical": 0.90,
    "item_total_min": 0.30,           # Corrected item-total correlation
}


def get_fit_label(index_name: str, value: float) -> tuple[str, str]:
    """
    Returns (label, color) for a fit index value.

    Parameters
    ----------
    index_name : str
        Name of the fit index (e.g., 'rmsea', 'cfi')
    value : float
        Observed value

    Returns
    -------
    tuple[str, str]
        (label, color_hex)
    """
    labels = {
        "excellent": ("✅ Excellent", "#2ecc71"),
        "good":      ("✅ Good",      "#27ae60"),
        "acceptable":("⚠️ Acceptable", "#f39c12"),
        "poor":      ("❌ Poor",       "#e74c3c"),
    }

    f = FIT
    if index_name == "rmsea":
        if value <= f["rmsea_excellent"]:   return labels["excellent"]
        elif value <= f["rmsea_good"]:      return labels["good"]
        elif value <= f["rmsea_acceptable"]:return labels["acceptable"]
        else:                               return labels["poor"]

    elif index_name in ("cfi", "tli", "nfi", "gfi"):
        threshold_good = f.get(f"{index_name}_good", 0.95)
        threshold_acc  = f.get(f"{index_name}_acceptable", 0.90)
        if value >= 0.97:                   return labels["excellent"]
        elif value >= threshold_good:       return labels["good"]
        elif value >= threshold_acc:        return labels["acceptable"]
        else:                               return labels["poor"]

    elif index_name == "srmr":
        if value <= f["srmr_good"]:         return labels["excellent"]
        elif value <= f["srmr_acceptable"]: return labels["acceptable"]
        else:                               return labels["poor"]

    elif index_name == "chisq_df":
        if value <= f["chisq_df_ratio_good"]:       return labels["good"]
        elif value <= f["chisq_df_ratio_acceptable"]:return labels["acceptable"]
        else:                                        return labels["poor"]

    return ("—", "#999999")
