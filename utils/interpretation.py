"""
interpretation.py
=================
Auto-narrative engine for SEM Studio.
Generates plain-language interpretations of all statistical outputs.

All interpretations follow APA 7th edition reporting standards.
"""

from utils.thresholds import DESCRIPTIVE, CFA, FIT, SEM, SAMPLE_SIZE, EFA


# ─── SAMPLE SIZE ────────────────────────────────────────────────

def interpret_sample_size(n: int, n_params: int = None) -> dict:
    s = SAMPLE_SIZE
    if n < s["minimum_cfa"]:
        level = "critical"
        msg = (
            f"Your sample size (n = {n}) is **below the minimum recommended** for CFA (n ≥ 100). "
            "Results should be interpreted with extreme caution. Consider collecting more data before proceeding."
        )
    elif n < s["recommended_cfa"]:
        level = "warning"
        msg = (
            f"Your sample size (n = {n}) meets the **minimum threshold** for CFA but is below the recommended "
            f"n ≥ 200. Proceed with caution and interpret fit indices conservatively."
        )
    elif n < s["minimum_sem"]:
        level = "warning"
        msg = (
            f"Your sample size (n = {n}) is adequate for CFA but **borderline for full SEM** (recommended n ≥ 200). "
            "Full SEM analyses can still be performed, but bootstrap procedures are especially recommended."
        )
    elif n < s["recommended_sem"]:
        level = "ok"
        msg = (
            f"Your sample size (n = {n}) meets the **minimum requirement for SEM** (n ≥ 200). "
            "Proceed with standard analyses."
        )
    elif n < s["optimal_sem"]:
        level = "good"
        msg = (
            f"Your sample size (n = {n}) is **adequate for full SEM** analyses. "
            "All planned analyses, including mediation with bootstrapping, can be conducted reliably."
        )
    else:
        level = "excellent"
        msg = (
            f"Your sample size (n = {n}) is **excellent** for SEM. "
            "All analyses, including complex multi-group and higher-order models, can be conducted with high statistical power."
        )

    if n_params:
        ratio = n / n_params
        param_note = (
            f" The observed ratio of observations-to-parameters is {ratio:.1f}:1 "
            f"({'✅ adequate' if ratio >= 10 else '⚠️ below recommended 10:1'})."
        )
        msg += param_note

    return {"level": level, "message": msg}


# ─── MISSING VALUES ─────────────────────────────────────────────

def interpret_missing(pct_missing: float, variable: str = None) -> dict:
    d = DESCRIPTIVE
    var_ref = f"Variable **{variable}**" if variable else "Your data"

    if pct_missing == 0:
        return {"level": "excellent", "message": f"{var_ref} has **no missing values**. ✅"}
    elif pct_missing <= d["missing_acceptable"]:
        return {
            "level": "ok",
            "message": (
                f"{var_ref} has {pct_missing:.1%} missing values, which is **within the acceptable range** (< 5%). "
                "Listwise or pairwise deletion is appropriate. FIML estimation is also an option."
            )
        }
    elif pct_missing <= d["missing_critical"]:
        return {
            "level": "warning",
            "message": (
                f"{var_ref} has {pct_missing:.1%} missing values (5–10%). "
                "This is **moderately concerning**. Consider using **Full Information Maximum Likelihood (FIML)** "
                "or multiple imputation rather than listwise deletion."
            )
        }
    else:
        return {
            "level": "critical",
            "message": (
                f"{var_ref} has {pct_missing:.1%} missing values — **above the critical 10% threshold**. "
                "Listwise deletion will substantially reduce your effective sample size. "
                "Multiple imputation or FIML is strongly recommended. "
                "Consider whether this variable should be retained in the analysis."
            )
        }


# ─── NORMALITY ──────────────────────────────────────────────────

def interpret_skewness(value: float, variable: str = None) -> dict:
    d = DESCRIPTIVE
    var_ref = f"**{variable}**" if variable else "This variable"
    abs_val = abs(value)

    if abs_val <= d["skewness_strict"]:
        return {"level": "excellent", "message": f"{var_ref}: skewness = {value:.3f} — **normally distributed** (|skewness| < 1.0). ✅"}
    elif abs_val <= d["skewness_acceptable"]:
        return {"level": "ok", "message": f"{var_ref}: skewness = {value:.3f} — **mildly skewed** but within acceptable limits (|skewness| < 2.0) for ML estimation."}
    else:
        return {"level": "warning", "message": f"{var_ref}: skewness = {value:.3f} — **substantially skewed** (|skewness| > 2.0). Consider robust estimation (MLR) or data transformation."}


def interpret_kurtosis(value: float, variable: str = None) -> dict:
    d = DESCRIPTIVE
    var_ref = f"**{variable}**" if variable else "This variable"
    abs_val = abs(value)

    if abs_val <= d["kurtosis_strict"]:
        return {"level": "excellent", "message": f"{var_ref}: kurtosis = {value:.3f} — **mesokurtic** (near normal). ✅"}
    elif abs_val <= d["kurtosis_acceptable"]:
        return {"level": "ok", "message": f"{var_ref}: kurtosis = {value:.3f} — **moderate kurtosis**, acceptable for ML estimation (|kurtosis| < 7.0)."}
    else:
        return {"level": "warning", "message": f"{var_ref}: kurtosis = {value:.3f} — **extreme kurtosis** (|kurtosis| > 7.0). Multivariate normality is likely violated. Use MLR or WLSMV estimator."}


def interpret_mardia(skewness_p: float, kurtosis_p: float) -> dict:
    sk_fail = skewness_p < 0.05
    ku_fail = kurtosis_p < 0.05

    if not sk_fail and not ku_fail:
        return {
            "level": "ok",
            "message": (
                "Mardia's test indicates that **multivariate normality is satisfied** "
                f"(skewness p = {skewness_p:.3f}, kurtosis p = {kurtosis_p:.3f}). "
                "**Maximum Likelihood (ML)** estimation is appropriate."
            ),
            "recommended_estimator": "ML"
        }
    elif sk_fail and ku_fail:
        return {
            "level": "warning",
            "message": (
                "Mardia's test indicates **multivariate normality is violated** on both skewness "
                f"(p = {skewness_p:.3f}) and kurtosis (p = {kurtosis_p:.3f}). "
                "**Robust Maximum Likelihood (MLR)** is recommended, which provides Satorra-Bentler corrected "
                "chi-square and robust standard errors."
            ),
            "recommended_estimator": "MLR"
        }
    else:
        failed = "skewness" if sk_fail else "kurtosis"
        p_val = skewness_p if sk_fail else kurtosis_p
        return {
            "level": "warning",
            "message": (
                f"Mardia's test indicates a violation of multivariate {failed} (p = {p_val:.3f}). "
                "**MLR (Robust ML)** estimation is recommended as a precaution."
            ),
            "recommended_estimator": "MLR"
        }


# ─── OUTLIERS ───────────────────────────────────────────────────

def interpret_outliers(n_outliers: int, n_total: int) -> dict:
    pct = n_outliers / n_total if n_total > 0 else 0
    if n_outliers == 0:
        return {"level": "excellent", "message": "**No multivariate outliers** detected (Mahalanobis D², p < .001). ✅"}
    elif pct <= 0.02:
        return {
            "level": "ok",
            "message": (
                f"**{n_outliers} multivariate outlier(s)** detected ({pct:.1%} of sample). "
                "This is within an acceptable range. Review these cases individually. "
                "Consider running analyses with and without outliers to assess their influence."
            )
        }
    else:
        return {
            "level": "warning",
            "message": (
                f"**{n_outliers} multivariate outlier(s)** detected ({pct:.1%} of sample). "
                "This is a notable proportion. Carefully examine these cases — they may represent "
                "legitimate heterogeneity or data entry errors. "
                "Robust estimation (MLR) is recommended."
            )
        }


# ─── EFA ────────────────────────────────────────────────────────

def interpret_kmo(kmo_value: float) -> dict:
    if kmo_value >= 0.90:
        label = "marvelous"
        level = "excellent"
    elif kmo_value >= 0.80:
        label = "meritorious"
        level = "good"
    elif kmo_value >= 0.70:
        label = "middling"
        level = "ok"
    elif kmo_value >= 0.60:
        label = "mediocre"
        level = "warning"
    else:
        label = "unacceptable"
        level = "critical"

    return {
        "level": level,
        "message": (
            f"Kaiser-Meyer-Olkin (KMO) measure = **{kmo_value:.3f}** — classified as **'{label}'** "
            f"(Kaiser, 1974). {'✅ The data are suitable for factor analysis.' if kmo_value >= 0.60 else '❌ Factor analysis is not recommended with this data.'}"
        )
    }


def interpret_bartlett(p_value: float) -> dict:
    if p_value < 0.001:
        return {"level": "excellent", "message": f"Bartlett's Test of Sphericity: χ²(p < .001) — **significant** ✅. The correlation matrix is not an identity matrix; factor analysis is appropriate."}
    elif p_value < 0.05:
        return {"level": "ok", "message": f"Bartlett's Test of Sphericity: p = {p_value:.3f} — **significant** ✅. Factor analysis is appropriate."}
    else:
        return {"level": "critical", "message": f"Bartlett's Test of Sphericity: p = {p_value:.3f} — **NOT significant** ❌. The correlation matrix is close to an identity matrix. Factor analysis is not appropriate."}


# ─── CFA ────────────────────────────────────────────────────────

def interpret_factor_loading(loading: float, item: str = None) -> dict:
    item_ref = f"**{item}**" if item else "This item"
    abs_l = abs(loading)

    if abs_l >= CFA["factor_loading_good"]:
        return {"level": "excellent", "message": f"{item_ref}: λ = {loading:.3f} — **strong loading** (≥ .70). ✅"}
    elif abs_l >= CFA["factor_loading_min"]:
        return {"level": "ok", "message": f"{item_ref}: λ = {loading:.3f} — **acceptable loading** (≥ .50). Consider retaining."}
    elif abs_l >= 0.40:
        return {"level": "warning", "message": f"{item_ref}: λ = {loading:.3f} — **weak loading** (.40–.50). Consider dropping this item if AVE is below threshold."}
    else:
        return {"level": "critical", "message": f"{item_ref}: λ = {loading:.3f} — **inadequate loading** (< .40). ❌ Strongly consider removing this item."}


def interpret_ave(ave: float, construct: str = None) -> dict:
    c_ref = f"Construct **{construct}**" if construct else "This construct"
    if ave >= CFA["ave_min"]:
        return {"level": "excellent", "message": f"{c_ref}: AVE = {ave:.3f} — **convergent validity is supported** (AVE ≥ .50). ✅ More than 50% of the variance in indicators is accounted for by the latent construct."}
    else:
        return {"level": "critical", "message": f"{c_ref}: AVE = {ave:.3f} — **convergent validity is NOT supported** (AVE < .50). ❌ The construct explains less variance in its indicators than the error terms. Consider revising or dropping weak items."}


def interpret_cr(cr: float, construct: str = None) -> dict:
    c_ref = f"Construct **{construct}**" if construct else "This construct"
    if cr >= CFA["cr_min"]:
        return {"level": "excellent", "message": f"{c_ref}: CR = {cr:.3f} — **composite reliability is good** (CR ≥ .70). ✅"}
    else:
        return {"level": "critical", "message": f"{c_ref}: CR = {cr:.3f} — **composite reliability is insufficient** (CR < .70). ❌ The construct indicators are not internally consistent enough."}


def interpret_alpha(alpha: float, construct: str = None) -> dict:
    c_ref = f"Construct **{construct}**" if construct else "This construct"
    if alpha >= 0.90:
        return {"level": "excellent", "message": f"{c_ref}: α = {alpha:.3f} — **excellent internal consistency**. ✅ (Note: values > .95 may indicate item redundancy.)"}
    elif alpha >= CFA["alpha_min"]:
        return {"level": "good", "message": f"{c_ref}: α = {alpha:.3f} — **good internal consistency** (α ≥ .70). ✅"}
    elif alpha >= CFA["alpha_acceptable"]:
        return {"level": "warning", "message": f"{c_ref}: α = {alpha:.3f} — **marginally acceptable** (α ≥ .60). ⚠️ Acceptable for exploratory research only."}
    else:
        return {"level": "critical", "message": f"{c_ref}: α = {alpha:.3f} — **insufficient reliability** (α < .60). ❌ Revise or replace items."}


def interpret_htmt(htmt: float, c1: str = None, c2: str = None) -> dict:
    pair = f"(**{c1}** ↔ **{c2}**)" if c1 and c2 else ""
    if htmt < CFA["htmt_strict"]:
        return {"level": "excellent", "message": f"HTMT {pair} = {htmt:.3f} — **discriminant validity is supported** (HTMT < .85). ✅ The two constructs are empirically distinct."}
    elif htmt < CFA["htmt_liberal"]:
        return {"level": "warning", "message": f"HTMT {pair} = {htmt:.3f} — **borderline discriminant validity** (.85 ≤ HTMT < .90). ⚠️ Use the liberal threshold (.90) as justification, but consider conceptual overlap."}
    else:
        return {"level": "critical", "message": f"HTMT {pair} = {htmt:.3f} — **discriminant validity is NOT supported** (HTMT ≥ .90). ❌ These constructs are too similar. Consider merging them or revising indicators."}


# ─── FIT INDICES ────────────────────────────────────────────────

def interpret_fit_index(index: str, value: float) -> dict:
    f = FIT
    templates = {
        "rmsea": {
            "name": "RMSEA",
            "direction": "lower",
            "levels": [
                (f["rmsea_excellent"], "excellent", f"RMSEA = {value:.3f} — **excellent model fit** (≤ .05). ✅"),
                (f["rmsea_good"],      "good",      f"RMSEA = {value:.3f} — **good model fit** (≤ .06). ✅"),
                (f["rmsea_acceptable"],"acceptable", f"RMSEA = {value:.3f} — **acceptable model fit** (≤ .08). ⚠️"),
                (f["rmsea_poor"],      "poor",       f"RMSEA = {value:.3f} — **poor model fit** (≤ .10). ❌ Consider model re-specification."),
                (999,                  "critical",   f"RMSEA = {value:.3f} — **very poor model fit** (> .10). ❌ Model re-specification is necessary."),
            ]
        },
        "cfi": {
            "name": "CFI",
            "direction": "higher",
            "levels": [
                (0.97, "excellent", f"CFI = {value:.3f} — **excellent model fit** (≥ .97). ✅"),
                (f["cfi_good"], "good", f"CFI = {value:.3f} — **good model fit** (≥ .95). ✅"),
                (f["cfi_acceptable"], "acceptable", f"CFI = {value:.3f} — **acceptable model fit** (≥ .90). ⚠️"),
                (0, "poor", f"CFI = {value:.3f} — **poor model fit** (< .90). ❌"),
            ]
        },
        "tli": {
            "name": "TLI",
            "direction": "higher",
            "levels": [
                (0.97, "excellent", f"TLI = {value:.3f} — **excellent model fit** (≥ .97). ✅"),
                (f["tli_good"], "good", f"TLI = {value:.3f} — **good model fit** (≥ .95). ✅"),
                (f["tli_acceptable"], "acceptable", f"TLI = {value:.3f} — **acceptable model fit** (≥ .90). ⚠️"),
                (0, "poor", f"TLI = {value:.3f} — **poor model fit** (< .90). ❌"),
            ]
        },
        "srmr": {
            "name": "SRMR",
            "direction": "lower",
            "levels": [
                (f["srmr_good"], "excellent", f"SRMR = {value:.3f} — **excellent model fit** (≤ .05). ✅"),
                (f["srmr_acceptable"], "acceptable", f"SRMR = {value:.3f} — **acceptable model fit** (≤ .08). ⚠️"),
                (999, "poor", f"SRMR = {value:.3f} — **poor model fit** (> .08). ❌"),
            ]
        },
    }

    if index not in templates:
        return {"level": "info", "message": f"{index.upper()} = {value:.3f}"}

    t = templates[index]
    if t["direction"] == "lower":
        for threshold, level, msg in t["levels"]:
            if value <= threshold:
                return {"level": level, "message": msg}
    else:
        for threshold, level, msg in reversed(t["levels"]):
            if value >= threshold:
                return {"level": level, "message": msg}
        return {"level": "poor", "message": t["levels"][-1][2]}

    return {"level": "info", "message": f"{index.upper()} = {value:.3f}"}


def interpret_overall_fit(fit_dict: dict) -> str:
    """
    Generates a holistic narrative of model fit based on multiple indices.
    fit_dict: {'rmsea': 0.06, 'cfi': 0.95, 'tli': 0.94, 'srmr': 0.07}
    """
    rmsea = fit_dict.get("rmsea", None)
    cfi   = fit_dict.get("cfi",   None)
    tli   = fit_dict.get("tli",   None)
    srmr  = fit_dict.get("srmr",  None)

    good_count = 0
    total = 0

    checks = [
        (rmsea, lambda v: v <= FIT["rmsea_acceptable"]),
        (cfi,   lambda v: v >= FIT["cfi_acceptable"]),
        (tli,   lambda v: v >= FIT["tli_acceptable"]),
        (srmr,  lambda v: v <= FIT["srmr_acceptable"]),
    ]

    for val, check_fn in checks:
        if val is not None:
            total += 1
            if check_fn(val):
                good_count += 1

    if total == 0:
        return "Insufficient fit indices available for overall assessment."

    pct = good_count / total
    if pct == 1.0:
        verdict = "✅ **Overall, the model demonstrates acceptable to good fit** across all evaluated indices. The hypothesized factor structure is supported by the data."
    elif pct >= 0.75:
        verdict = "⚠️ **Overall, the model demonstrates marginally acceptable fit**. Most indices meet threshold criteria, but attention to the failing indices and possible model re-specification is recommended."
    else:
        verdict = "❌ **Overall, the model fit is inadequate**. Multiple fit indices fail to meet minimum criteria. Inspect modification indices, re-examine the theoretical model, and consider re-specification before interpreting structural paths."

    return verdict


# ─── STRUCTURAL PATHS ───────────────────────────────────────────

def interpret_path(beta: float, se: float, p_value: float,
                   predictor: str = None, outcome: str = None) -> dict:
    pred = f"**{predictor}**" if predictor else "the predictor"
    out  = f"**{outcome}**"   if outcome   else "the outcome"

    direction = "positively" if beta > 0 else "negatively"
    sig = p_value < SEM["path_significant_pvalue"]
    marginal = p_value < SEM["path_marginal_pvalue"]

    if sig:
        return {
            "level": "significant",
            "message": (
                f"The path from {pred} to {out} is **statistically significant** "
                f"(β = {beta:.3f}, SE = {se:.3f}, p = {p_value:.3f}). "
                f"{pred.capitalize()} **{direction} predicts** {out}. ✅"
            )
        }
    elif marginal:
        return {
            "level": "marginal",
            "message": (
                f"The path from {pred} to {out} is **marginally significant** "
                f"(β = {beta:.3f}, SE = {se:.3f}, p = {p_value:.3f}). "
                "Interpret with caution. ⚠️"
            )
        }
    else:
        return {
            "level": "nonsignificant",
            "message": (
                f"The path from {pred} to {out} is **not statistically significant** "
                f"(β = {beta:.3f}, SE = {se:.3f}, p = {p_value:.3f}). "
                f"There is insufficient evidence that {pred} predicts {out}. ❌"
            )
        }


def interpret_r2(r2: float, construct: str = None) -> dict:
    c_ref = f"**{construct}**" if construct else "This endogenous variable"
    if r2 >= SEM["r2_strong"]:
        return {"level": "strong", "message": f"{c_ref}: R² = {r2:.3f} — **substantial explained variance** (R² ≥ .26). ✅ The model explains {r2:.1%} of the variance."}
    elif r2 >= SEM["r2_moderate"]:
        return {"level": "moderate", "message": f"{c_ref}: R² = {r2:.3f} — **moderate explained variance** (.13 ≤ R² < .26). ⚠️ The model explains {r2:.1%} of variance."}
    elif r2 >= SEM["r2_weak"]:
        return {"level": "weak", "message": f"{c_ref}: R² = {r2:.3f} — **weak explained variance** (.02 ≤ R² < .13). The model explains only {r2:.1%} of variance. Consider adding predictors."}
    else:
        return {"level": "negligible", "message": f"{c_ref}: R² = {r2:.3f} — **negligible explained variance** (< .02). ❌ The predictors do not explain meaningful variance in this outcome."}


# ─── MEDIATION ──────────────────────────────────────────────────

def interpret_mediation(indirect_effect: float, ci_lower: float, ci_upper: float,
                         direct_effect: float = None, total_effect: float = None,
                         mediator: str = None, predictor: str = None, outcome: str = None) -> dict:
    med  = f"**{mediator}**"  if mediator  else "**the mediator**"
    pred = f"**{predictor}**" if predictor else "**the predictor**"
    out  = f"**{outcome}**"   if outcome   else "**the outcome**"

    # CI does not contain zero = significant mediation
    sig = not (ci_lower <= 0 <= ci_upper)

    if sig:
        if direct_effect is not None:
            if abs(direct_effect) < 0.05:
                mediation_type = "**full mediation**"
                note = f"The direct effect of {pred} on {out} is negligible (β = {direct_effect:.3f}), suggesting full mediation."
            else:
                mediation_type = "**partial mediation**"
                note = f"The direct effect remains significant (β = {direct_effect:.3f}), indicating partial mediation."
        else:
            mediation_type = "**significant mediation**"
            note = ""

        return {
            "level": "significant",
            "message": (
                f"The indirect effect of {pred} on {out} through {med} is {mediation_type} "
                f"(indirect effect = {indirect_effect:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
                f"The confidence interval does not include zero, confirming significance. {note} ✅"
            )
        }
    else:
        return {
            "level": "nonsignificant",
            "message": (
                f"The indirect effect of {pred} on {out} through {med} is **not significant** "
                f"(indirect effect = {indirect_effect:.3f}, 95% CI [{ci_lower:.3f}, {ci_upper:.3f}]). "
                "The confidence interval includes zero. Mediation is not supported. ❌"
            )
        }
