"""
interpretation.py - Auto-narrative engine for SEM Studio.
"""
from utils.thresholds import (
    SAMPLE, DESCRIPTIVE, EFA, CFA, FIT, SEM, MEDIATION, INVARIANCE, OUTLIER
)

def _safe(value, default=None):
    if value is None: return default
    try:
        import math
        if math.isnan(float(value)): return default
    except (TypeError, ValueError): pass
    return value

def interpret_sample_size(n):
    s = SAMPLE
    if n < s["minimum_cfa"]:
        return {"level":"critical","message":f"Sample size (n = {n}) is **below the minimum** for CFA (n ≥ 100). Collect more data before proceeding."}
    elif n < s["recommended_cfa"]:
        return {"level":"warning","message":f"Sample size (n = {n}) meets **minimum for CFA** but below recommended n ≥ 200. Proceed with caution."}
    elif n < s["minimum_sem"]:
        return {"level":"warning","message":f"Sample size (n = {n}) is adequate for CFA but **borderline for SEM** (recommended n ≥ 200)."}
    elif n < s["recommended_sem"]:
        return {"level":"ok","message":f"Sample size (n = {n}) **meets the minimum for SEM** (n ≥ 200). Proceed with standard analyses."}
    elif n < s["optimal_sem"]:
        return {"level":"good","message":f"Sample size (n = {n}) is **adequate for full SEM** analyses including mediation with bootstrapping."}
    else:
        return {"level":"excellent","message":f"Sample size (n = {n}) is **excellent** for SEM. All analyses can be conducted with high statistical power."}

def interpret_missing(pct, variable=None):
    ref = f"**{variable}**" if variable else "Your data"
    d = DESCRIPTIVE
    if pct == 0:
        return {"level":"excellent","message":f"{ref} has **no missing values**. ✅"}
    elif pct <= d["missing_acceptable"]:
        return {"level":"ok","message":f"{ref} has {pct:.1%} missing — **within acceptable range** (< 5%). Listwise deletion or FIML are appropriate."}
    elif pct <= d["missing_critical"]:
        return {"level":"warning","message":f"{ref} has {pct:.1%} missing (5–10%). **FIML or multiple imputation** is recommended."}
    else:
        return {"level":"critical","message":f"{ref} has {pct:.1%} missing — **above critical 10% threshold**. FIML or multiple imputation strongly recommended."}

def interpret_skewness(value, variable=None):
    ref = f"**{variable}**" if variable else "This variable"
    av = abs(value)
    if av <= DESCRIPTIVE["skewness_strict"]:
        return {"level":"excellent","message":f"{ref}: skewness = {value:.3f} — **normally distributed** (|skew| < 1.0). ✅"}
    elif av <= DESCRIPTIVE["skewness_acceptable"]:
        return {"level":"ok","message":f"{ref}: skewness = {value:.3f} — **mildly skewed** but acceptable for ML estimation."}
    else:
        return {"level":"warning","message":f"{ref}: skewness = {value:.3f} — **substantially skewed** (|skew| > 2.0). Consider MLR estimator."}

def interpret_kurtosis(value, variable=None):
    ref = f"**{variable}**" if variable else "This variable"
    av = abs(value)
    if av <= DESCRIPTIVE["kurtosis_strict"]:
        return {"level":"excellent","message":f"{ref}: kurtosis = {value:.3f} — **near normal** (|kurt| < 3.0). ✅"}
    elif av <= DESCRIPTIVE["kurtosis_acceptable"]:
        return {"level":"ok","message":f"{ref}: kurtosis = {value:.3f} — **moderate kurtosis**, acceptable for ML estimation."}
    else:
        return {"level":"warning","message":f"{ref}: kurtosis = {value:.3f} — **extreme kurtosis** (|kurt| > 7.0). Use MLR estimator."}

def interpret_mardia(skewness_p, kurtosis_p):
    sk_fail = _safe(skewness_p, 1.0) < 0.05
    ku_fail = _safe(kurtosis_p, 1.0) < 0.05
    if not sk_fail and not ku_fail:
        return {"level":"ok","message":f"Mardia's test: **multivariate normality satisfied** (skewness p = {skewness_p:.3f}, kurtosis p = {kurtosis_p:.3f}). **ML** estimation is appropriate.","estimator":"ML"}
    else:
        return {"level":"warning","message":"Mardia's test: **multivariate normality violated**. **MLR** (Robust ML) is recommended — provides Satorra-Bentler corrected chi-square and robust standard errors.","estimator":"MLR"}

def interpret_kmo(kmo):
    e = EFA
    if kmo >= e["kmo_marvelous"]: label,level = "marvelous","excellent"
    elif kmo >= e["kmo_meritorious"]: label,level = "meritorious","good"
    elif kmo >= e["kmo_middling"]: label,level = "middling","ok"
    elif kmo >= e["kmo_mediocre"]: label,level = "mediocre","warning"
    else: label,level = "unacceptable","critical"
    suitable = kmo >= e["kmo_mediocre"]
    return {"level":level,"message":f"KMO = **{kmo:.3f}** — classified as **'{label}'** (Kaiser, 1974). {'✅ Data are suitable for factor analysis.' if suitable else '❌ Factor analysis is not recommended.'}"}

def interpret_bartlett(p):
    if p < 0.001: return {"level":"excellent","message":f"Bartlett's Test: p < .001 — **significant** ✅. Factor analysis is appropriate."}
    elif p < 0.05: return {"level":"ok","message":f"Bartlett's Test: p = {p:.3f} — **significant** ✅. Factor analysis is appropriate."}
    else: return {"level":"critical","message":f"Bartlett's Test: p = {p:.3f} — **NOT significant** ❌. Factor analysis is not appropriate."}

def interpret_loading(loading, item=None):
    ref = f"**{item}**" if item else "This item"
    av = abs(loading)
    if av >= CFA["loading_good"]: return {"level":"excellent","message":f"{ref}: λ = {loading:.3f} — **strong loading** (≥ .70). ✅"}
    elif av >= CFA["loading_min"]: return {"level":"ok","message":f"{ref}: λ = {loading:.3f} — **acceptable loading** (≥ .50). Consider retaining."}
    elif av >= EFA["loading_acceptable"]: return {"level":"warning","message":f"{ref}: λ = {loading:.3f} — **weak loading** (.40–.50). ⚠️ Consider dropping if AVE is below threshold."}
    else: return {"level":"critical","message":f"{ref}: λ = {loading:.3f} — **inadequate loading** (< .40). ❌ Strongly consider removing this item."}

def interpret_ave(ave, construct=None):
    ref = f"**{construct}**" if construct else "This construct"
    if _safe(ave, 0) >= CFA["ave_min"]:
        return {"level":"excellent","message":f"{ref}: AVE = {ave:.3f} — **convergent validity supported** (AVE ≥ .50). ✅"}
    else:
        return {"level":"critical","message":f"{ref}: AVE = {ave:.3f} — **convergent validity NOT supported** (AVE < .50). ❌ Revise or drop weak items."}

def interpret_cr(cr, construct=None):
    ref = f"**{construct}**" if construct else "This construct"
    if _safe(cr, 0) >= CFA["cr_min"]:
        return {"level":"excellent","message":f"{ref}: CR = {cr:.3f} — **composite reliability good** (CR ≥ .70). ✅"}
    else:
        return {"level":"critical","message":f"{ref}: CR = {cr:.3f} — **composite reliability insufficient** (CR < .70). ❌"}

def interpret_alpha(alpha, construct=None):
    ref = f"**{construct}**" if construct else "This construct"
    a = _safe(alpha, 0)
    if a >= 0.90: return {"level":"excellent","message":f"{ref}: α = {alpha:.3f} — **excellent internal consistency**. ✅"}
    elif a >= CFA["alpha_min"]: return {"level":"good","message":f"{ref}: α = {alpha:.3f} — **good internal consistency** (α ≥ .70). ✅"}
    elif a >= CFA["alpha_acceptable"]: return {"level":"warning","message":f"{ref}: α = {alpha:.3f} — **marginally acceptable** (α ≥ .60). ⚠️"}
    else: return {"level":"critical","message":f"{ref}: α = {alpha:.3f} — **insufficient reliability** (α < .60). ❌"}

def interpret_htmt(htmt, c1=None, c2=None):
    pair = f"(**{c1}** ↔ **{c2}**)" if c1 and c2 else ""
    h = _safe(htmt, 1.0)
    if h < CFA["htmt_strict"]: return {"level":"excellent","message":f"HTMT {pair} = {htmt:.3f} — **discriminant validity supported** (HTMT < .85). ✅"}
    elif h < CFA["htmt_liberal"]: return {"level":"warning","message":f"HTMT {pair} = {htmt:.3f} — **borderline discriminant validity** (.85–.90). ⚠️"}
    else: return {"level":"critical","message":f"HTMT {pair} = {htmt:.3f} — **discriminant validity NOT supported** (HTMT ≥ .90). ❌"}

def interpret_fit_index(index, value):
    from utils.thresholds import interpret_fit
    label, color, level = interpret_fit(index, value)
    msgs = {"rmsea":f"RMSEA = {value:.3f} — {label}","cfi":f"CFI = {value:.3f} — {label}","tli":f"TLI = {value:.3f} — {label}","srmr":f"SRMR = {value:.3f} — {label}","chisq_df":f"χ²/df = {value:.3f} — {label}"}
    return {"level":level,"message":msgs.get(index,f"{index.upper()} = {value:.3f} — {label}"),"color":color}

def interpret_overall_fit(fit):
    checks = [(fit.get("rmsea"),lambda v:v<=FIT["rmsea_acceptable"]),(fit.get("cfi"),lambda v:v>=FIT["cfi_acceptable"]),(fit.get("tli"),lambda v:v>=FIT["tli_acceptable"]),(fit.get("srmr"),lambda v:v<=FIT["srmr_acceptable"])]
    good = sum(1 for v,fn in checks if v is not None and fn(v))
    total = sum(1 for v,_ in checks if v is not None)
    if total == 0: return "Insufficient fit indices for overall assessment."
    pct = good/total
    if pct == 1.0: return "✅ **Overall, the model demonstrates acceptable to good fit** across all evaluated indices."
    elif pct >= 0.75: return "⚠️ **Overall, the model demonstrates marginally acceptable fit**. Most indices meet threshold criteria."
    else: return "❌ **Overall, the model fit is inadequate**. Multiple fit indices fail minimum criteria. Consider re-specification."

def interpret_path(beta, se, p, predictor=None, outcome=None):
    pred = f"**{predictor}**" if predictor else "the predictor"
    out = f"**{outcome}**" if outcome else "the outcome"
    direction = "positively" if beta > 0 else "negatively"
    if p < SEM["p_significant"]:
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*"
        return {"level":"significant","message":f"Path {pred} → {out}: **significant** (β = {beta:.3f}{stars}, SE = {se:.3f}, p = {p:.3f}). {predictor or pred} {direction} predicts {outcome or out}. ✅"}
    elif p < SEM["p_marginal"]:
        return {"level":"marginal","message":f"Path {pred} → {out}: **marginally significant** (β = {beta:.3f}, p = {p:.3f}). ⚠️ Interpret with caution."}
    else:
        return {"level":"nonsignificant","message":f"Path {pred} → {out}: **not significant** (β = {beta:.3f}, p = {p:.3f}). Insufficient evidence. ❌"}

def interpret_r2(r2, construct=None):
    ref = f"**{construct}**" if construct else "This endogenous variable"
    rv = _safe(r2, 0)
    if rv >= SEM["r2_strong"]: return {"level":"strong","message":f"{ref}: R² = {r2:.3f} — **substantial explained variance** (≥ .26). ✅ Model explains {r2:.1%} of variance."}
    elif rv >= SEM["r2_moderate"]: return {"level":"moderate","message":f"{ref}: R² = {r2:.3f} — **moderate explained variance** (.13–.26). Model explains {r2:.1%} of variance."}
    elif rv >= SEM["r2_weak"]: return {"level":"weak","message":f"{ref}: R² = {r2:.3f} — **weak explained variance** (.02–.13). ⚠️ Consider adding predictors."}
    else: return {"level":"negligible","message":f"{ref}: R² = {r2:.3f} — **negligible explained variance** (< .02). ❌"}

def interpret_mediation(indirect, ci_lo, ci_hi, direct=None, mediator=None, predictor=None, outcome=None):
    med = f"**{mediator}**" if mediator else "**the mediator**"
    pred = f"**{predictor}**" if predictor else "**the predictor**"
    out = f"**{outcome}**" if outcome else "**the outcome**"
    sig = not (ci_lo <= 0 <= ci_hi)
    if sig:
        if direct is not None and abs(direct) < 0.05:
            med_type,note = "**full mediation**",f"Direct effect is negligible (β = {direct:.3f})."
        elif direct is not None:
            med_type,note = "**partial mediation**",f"Direct effect remains (β = {direct:.3f})."
        else:
            med_type,note = "**significant mediation**",""
        return {"level":"significant","message":f"Indirect effect of {pred} → {out} through {med}: {med_type} (indirect = {indirect:.4f}, 95% BCa CI [{ci_lo:.4f}, {ci_hi:.4f}]). CI does not contain zero. {note} ✅"}
    else:
        return {"level":"nonsignificant","message":f"Indirect effect through {med}: **not significant** (indirect = {indirect:.4f}, 95% BCa CI [{ci_lo:.4f}, {ci_hi:.4f}]). CI contains zero. ❌"}

def interpret_moderation(b3, p, delta_r2, x=None, w=None, y=None):
    xv = f"**{x}**" if x else "**X**"
    wv = f"**{w}**" if w else "**W**"
    yv = f"**{y}**" if y else "**Y**"
    if p < SEM["p_significant"]:
        return {"level":"significant","message":f"Interaction term is **significant** (β = {b3:.3f}, p = {p:.3f}). {wv} **significantly moderates** the {xv} → {yv} relationship. ΔR² = {delta_r2:.4f}. ✅"}
    else:
        return {"level":"nonsignificant","message":f"Interaction term is **not significant** (β = {b3:.3f}, p = {p:.3f}). {wv} does **not** significantly moderate the {xv} → {yv} relationship. ❌"}

def interpret_invariance_level(configural_ok, metric_ok, scalar_ok, group1=None, group2=None):
    g1 = f"**{group1}**" if group1 else "**Group 1**"
    g2 = f"**{group2}**" if group2 else "**Group 2**"
    results = []
    if configural_ok:
        results.append({"level":"ok","message":f"**Configural Invariance: ✅ Supported** — Same factor structure holds for {g1} and {g2}."})
    else:
        results.append({"level":"critical","message":f"**Configural Invariance: ❌ Not Supported** — Factor structure differs between groups. Cross-group comparisons are not meaningful."})
        return results
    if metric_ok:
        results.append({"level":"ok","message":f"**Metric Invariance: ✅ Supported** (ΔCFI ≥ −.010) — Factor loadings are equivalent across groups. You can compare **relationships** between groups."})
    else:
        results.append({"level":"warning","message":f"**Metric Invariance: ❌ Not Supported** (ΔCFI < −.010) — Factor loadings differ. Partial metric invariance may allow limited comparisons."})
    if scalar_ok:
        results.append({"level":"ok","message":f"**Scalar Invariance: ✅ Supported** — Item intercepts are equivalent. You can compare **latent means** between groups."})
    else:
        results.append({"level":"warning","message":f"**Scalar Invariance: ❌ Not Supported** — Item intercepts differ. Latent mean comparisons are not valid."})
    return results
