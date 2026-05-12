# sem_analysis.R
# ============================================================
# SEM Studio — Core R Analysis Script
# All SEM/CFA/EFA computations using lavaan & psych
#
# Libraries: lavaan, semTools, psych, GPArotation
# Called from Python via rpy2
#
# References:
#   Rosseel (2012). lavaan: An R Package for SEM. JSS.
#   Hair et al. (2019). Multivariate Data Analysis (8th ed.)
#   Kline (2016). Principles and Practice of SEM (4th ed.)
#   Hu & Bentler (1999). Cutoff criteria for fit indexes.
#   Fornell & Larcker (1981). Evaluating SEM models.
# ============================================================

suppressPackageStartupMessages({
  library(lavaan)
  library(semTools)
  library(psych)
  library(GPArotation)
  library(jsonlite)
})


# ── UTILITY: Data prep ───────────────────────────────────────────

prepare_data <- function(data) {
  # Convert all indicator columns to numeric
  data[] <- lapply(data, function(x) suppressWarnings(as.numeric(as.character(x))))
  return(data)
}


# ── 1. DESCRIPTIVE STATISTICS ────────────────────────────────────

run_descriptives <- function(data) {
  data <- prepare_data(data)

  results <- list()
  for (col in colnames(data)) {
    x <- data[[col]]
    x <- x[!is.na(x)]
    if (length(x) < 3) next

    results[[col]] <- list(
      n        = length(x),
      mean     = round(mean(x), 3),
      sd       = round(sd(x), 3),
      min      = round(min(x), 3),
      max      = round(max(x), 3),
      skewness = round(psych::skew(x), 3),
      kurtosis = round(psych::kurtosi(x), 3),
      missing  = sum(is.na(data[[col]]))
    )
  }
  return(results)
}


# ── 2. MULTIVARIATE NORMALITY (Mardia's Test) ────────────────────

run_mardia <- function(data) {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  if (nrow(data) < 10 || ncol(data) < 2) {
    return(list(
      skewness   = NA, skewness_p = NA,
      kurtosis   = NA, kurtosis_p = NA,
      estimator  = "ML",
      normal     = FALSE
    ))
  }

  tryCatch({
    result <- suppressMessages(suppressWarnings(psych::mardia(data, plot = FALSE)))

    sk_p <- result$p.skew
    ku_p <- result$p.kurt

    normal    <- (sk_p > 0.05) && (ku_p > 0.05)
    estimator <- if (normal) "ML" else "MLR"

    return(list(
      skewness   = round(result$b1p, 4),
      skewness_p = round(sk_p, 4),
      kurtosis   = round(result$b2p, 4),
      kurtosis_p = round(ku_p, 4),
      estimator  = estimator,
      normal     = normal
    ))
  }, error = function(e) {
    return(list(
      skewness = NA, skewness_p = NA,
      kurtosis = NA, kurtosis_p = NA,
      estimator = "MLR", normal = FALSE,
      error = conditionMessage(e)
    ))
  })
}


# ── 3. EXPLORATORY FACTOR ANALYSIS ───────────────────────────────

run_efa <- function(data, n_factors, rotation = "oblimin") {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  if (nrow(data) < 10) {
    return(list(error = "Not enough complete cases for EFA."))
  }

  tryCatch({
    # KMO & Bartlett
    kmo_result <- suppressMessages(suppressWarnings(psych::KMO(data)))
    bart       <- suppressMessages(suppressWarnings(psych::cortest.bartlett(data)))

    # Parallel analysis via eigenvalue simulation (no console output)
    n_obs   <- nrow(data)
    n_items <- ncol(data)
    obs_ev  <- eigen(cor(data))$values

    # Simulate random eigenvalues (100 iterations)
    set.seed(42)
    sim_ev <- matrix(0, nrow=100, ncol=n_items)
    for (i in 1:100) {
      rand_data <- matrix(rnorm(n_obs * n_items), nrow=n_obs, ncol=n_items)
      sim_ev[i,] <- eigen(cor(rand_data))$values
    }
    pa_95 <- apply(sim_ev, 2, quantile, 0.95)
    suggested_factors <- max(1, sum(obs_ev > pa_95))

    # Factor analysis
    fa_result <- suppressMessages(suppressWarnings(
      psych::fa(
        data,
        nfactors = n_factors,
        rotate   = rotation,
        fm       = "pa",
        scores   = FALSE
      )
    ))

    # Extract loadings matrix
    loadings_mat <- unclass(fa_result$loadings)

    # Variance explained
    var_explained <- fa_result$Vaccounted

    # Communalities
    communalities <- fa_result$communality

    return(list(
      kmo              = round(kmo_result$MSA, 3),
      kmo_per_item     = round(kmo_result$MSAi, 3),
      bartlett_chi2    = round(bart$chisq, 3),
      bartlett_df      = bart$df,
      bartlett_p       = round(bart$p.value, 4),
      suggested_factors= suggested_factors,
      loadings         = round(loadings_mat, 3),
      communalities    = round(communalities, 3),
      var_explained    = round(var_explained, 3),
      n_factors        = n_factors,
      rotation         = rotation,
      n                = nrow(data)
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 4. CONFIRMATORY FACTOR ANALYSIS ──────────────────────────────

run_cfa <- function(data, model_syntax, estimator = "MLR") {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  if (nrow(data) < 50) {
    return(list(error = "Not enough complete cases for CFA (minimum 50)."))
  }

  tryCatch({
    fit <- lavaan::cfa(
      model     = model_syntax,
      data      = data,
      estimator = estimator,
      std.lv    = TRUE
    )

    # Fit indices
    fit_indices <- lavaan::fitMeasures(fit, c(
      "chisq", "df", "pvalue",
      "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
      "cfi", "tli", "nfi", "ifi",
      "srmr", "gfi", "agfi",
      "aic", "bic",
      "chisq.scaled", "df.scaled", "pvalue.scaled",
      "rmsea.scaled", "cfi.scaled", "tli.scaled"
    ))

    # Parameter estimates
    params <- lavaan::parameterEstimates(fit, standardized = TRUE)

    # Factor loadings (standardized)
    loadings <- params[params$op == "=~", c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")]
    colnames(loadings) <- c("construct", "item", "unstd", "se", "z", "p", "std")
    loadings <- loadings[order(loadings$construct), ]

    # Reliability & Validity via semTools
    rel <- tryCatch({
      semTools::reliability(fit)
    }, error = function(e) NULL)

    # AVE
    ave <- tryCatch({
      semTools::AVE(fit)
    }, error = function(e) NULL)

    # HTMT
    htmt <- tryCatch({
      semTools::htmt(model_syntax, data = data)
    }, error = function(e) NULL)

    # Modification indices
    mi <- tryCatch({
      lavaan::modindices(fit, sort. = TRUE, maximum.number = 10)
    }, error = function(e) NULL)

    return(list(
      fit_indices  = round(fit_indices, 4),
      loadings     = loadings,
      reliability  = rel,
      ave          = ave,
      htmt         = htmt,
      mod_indices  = mi,
      n            = nrow(data),
      estimator    = estimator,
      converged    = lavaan::lavInspect(fit, "converged")
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 5. FULL SEM ───────────────────────────────────────────────────

run_sem <- function(data, model_syntax, estimator = "MLR") {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  if (nrow(data) < 100) {
    return(list(error = "Not enough complete cases for SEM (minimum 100)."))
  }

  tryCatch({
    fit <- lavaan::sem(
      model     = model_syntax,
      data      = data,
      estimator = estimator,
      std.lv    = TRUE
    )

    # Fit indices
    fit_indices <- lavaan::fitMeasures(fit, c(
      "chisq", "df", "pvalue",
      "rmsea", "rmsea.ci.lower", "rmsea.ci.upper",
      "cfi", "tli", "nfi", "srmr",
      "aic", "bic",
      "chisq.scaled", "rmsea.scaled", "cfi.scaled"
    ))

    # All parameter estimates
    params <- lavaan::parameterEstimates(fit, standardized = TRUE)

    # Structural paths only
    paths <- params[params$op == "~", c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")]
    colnames(paths) <- c("outcome", "predictor", "unstd", "se", "z", "p", "beta")

    # Factor loadings
    loadings <- params[params$op == "=~", c("lhs", "rhs", "est", "se", "z", "pvalue", "std.all")]
    colnames(loadings) <- c("construct", "item", "unstd", "se", "z", "p", "std")

    # R-squared for endogenous variables
    r2 <- tryCatch({
      lavaan::lavInspect(fit, "r2")
    }, error = function(e) NULL)

    return(list(
      fit_indices = round(fit_indices, 4),
      paths       = paths,
      loadings    = loadings,
      r2          = r2,
      n           = nrow(data),
      estimator   = estimator,
      converged   = lavaan::lavInspect(fit, "converged")
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 6. MEDIATION ANALYSIS (Bootstrap) ────────────────────────────

run_mediation <- function(data, x_var, m_var, y_var,
                          constructs,
                          n_boot = 5000, estimator = "MLR") {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  if (nrow(data) < 100) {
    return(list(error = "Not enough complete cases for mediation (minimum 100)."))
  }

  tryCatch({
    # Build parcel scores (mean of indicators per construct)
    for (cname in names(constructs)) {
      items <- constructs[[cname]]
      items <- items[items %in% colnames(data)]
      if (length(items) > 0) {
        data[[cname]] <- rowMeans(data[, items, drop = FALSE], na.rm = TRUE)
      }
    }

    # Build lavaan mediation syntax
    med_syntax <- paste0(
      "# a path\n",
      m_var, " ~ a * ", x_var, "\n",
      "# b and c' paths\n",
      y_var, " ~ b * ", m_var, " + cp * ", x_var, "\n",
      "# Indirect effect\n",
      "indirect := a * b\n",
      "# Total effect\n",
      "total := cp + a * b\n"
    )

    fit <- lavaan::sem(
      model     = med_syntax,
      data      = data,
      estimator = estimator,
      se        = "bootstrap",
      bootstrap = n_boot
    )

    params <- lavaan::parameterEstimates(
      fit,
      boot.ci.type = "bca.simple",
      level        = 0.95,
      standardized = TRUE
    )

    # Extract key paths
    get_param <- function(label) {
      row <- params[!is.na(params$label) & params$label == label, ]
      if (nrow(row) == 0) return(NULL)
      list(
        est    = round(row$est[1], 4),
        se     = round(row$se[1], 4),
        z      = round(row$z[1], 4),
        p      = round(row$pvalue[1], 4),
        ci_lo  = round(row$ci.lower[1], 4),
        ci_hi  = round(row$ci.upper[1], 4),
        std    = round(row$std.all[1], 4)
      )
    }

    return(list(
      a_path   = get_param("a"),
      b_path   = get_param("b"),
      cp_path  = get_param("cp"),
      indirect = get_param("indirect"),
      total    = get_param("total"),
      n        = nrow(data),
      n_boot   = n_boot,
      syntax   = med_syntax
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 7. MODERATION ANALYSIS ───────────────────────────────────────

run_moderation <- function(data, x_var, w_var, y_var, constructs) {
  data <- prepare_data(data)

  tryCatch({
    # Build parcel scores
    for (cname in names(constructs)) {
      items <- constructs[[cname]]
      items <- items[items %in% colnames(data)]
      if (length(items) > 0) {
        data[[cname]] <- rowMeans(data[, items, drop = FALSE], na.rm = TRUE)
      }
    }

    data <- data[complete.cases(data[, c(x_var, w_var, y_var)]), ]

    if (nrow(data) < 50) {
      return(list(error = "Not enough complete cases."))
    }

    # Mean-center
    x <- scale(data[[x_var]], scale = FALSE)[, 1]
    w <- scale(data[[w_var]], scale = FALSE)[, 1]
    y <- scale(data[[y_var]], scale = FALSE)[, 1]
    xw <- x * w

    # Model without interaction
    m1 <- lm(y ~ x + w)
    r2_1 <- summary(m1)$r.squared

    # Model with interaction
    m2 <- lm(y ~ x + w + xw)
    s2  <- summary(m2)
    r2_2 <- s2$r.squared
    delta_r2 <- r2_2 - r2_1

    coefs <- coef(s2)

    # Simple slopes at -1 SD, mean, +1 SD of W
    w_sd   <- sd(data[[w_var]], na.rm = TRUE)
    slopes <- list()
    for (level in c(-1, 0, 1)) {
      w_val  <- level * w_sd
      slope  <- coefs["x", "Estimate"] + coefs["xw", "Estimate"] * w_val
      se_val <- sqrt(
        coefs["x", "Std. Error"]^2 +
        (w_val^2) * coefs["xw", "Std. Error"]^2
      )
      t_val  <- slope / se_val
      p_val  <- 2 * pt(abs(t_val), df = nrow(data) - 4, lower.tail = FALSE)
      label  <- if (level == -1) "Low (-1 SD)" else if (level == 0) "Mean" else "High (+1 SD)"
      slopes[[label]] <- list(
        slope = round(slope, 4),
        se    = round(se_val, 4),
        t     = round(t_val, 4),
        p     = round(p_val, 4)
      )
    }

    return(list(
      b0          = round(coefs["(Intercept)", "Estimate"], 4),
      b1          = round(coefs["x", "Estimate"], 4),
      b1_se       = round(coefs["x", "Std. Error"], 4),
      b1_t        = round(coefs["x", "t value"], 4),
      b1_p        = round(coefs["x", "Pr(>|t|)"], 4),
      b2          = round(coefs["w", "Estimate"], 4),
      b2_se       = round(coefs["w", "Std. Error"], 4),
      b2_t        = round(coefs["w", "t value"], 4),
      b2_p        = round(coefs["w", "Pr(>|t|)"], 4),
      b3          = round(coefs["xw", "Estimate"], 4),
      b3_se       = round(coefs["xw", "Std. Error"], 4),
      b3_t        = round(coefs["xw", "t value"], 4),
      b3_p        = round(coefs["xw", "Pr(>|t|)"], 4),
      r2_1        = round(r2_1, 4),
      r2_2        = round(r2_2, 4),
      delta_r2    = round(delta_r2, 4),
      simple_slopes = slopes,
      n           = nrow(data)
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 8. MEASUREMENT INVARIANCE ────────────────────────────────────

run_invariance <- function(data, model_syntax, group_var, estimator = "MLR") {
  data <- prepare_data(data)

  tryCatch({
    extract_fit <- function(fit) {
      fi <- lavaan::fitMeasures(fit, c(
        "chisq", "df", "pvalue", "rmsea", "cfi", "tli", "srmr", "aic", "bic"
      ))
      round(fi, 4)
    }

    # Configural
    conf <- lavaan::cfa(model_syntax, data = data,
                        group = group_var, estimator = estimator, std.lv = TRUE)

    # Metric
    metr <- lavaan::cfa(model_syntax, data = data,
                        group = group_var, group.equal = "loadings",
                        estimator = estimator, std.lv = TRUE)

    # Scalar
    scal <- lavaan::cfa(model_syntax, data = data,
                        group = group_var,
                        group.equal = c("loadings", "intercepts"),
                        estimator = estimator, std.lv = TRUE)

    # Difference tests
    diff_metric <- lavaan::compareFit(conf, metr)
    diff_scalar <- lavaan::compareFit(metr, scal)

    return(list(
      configural = as.list(extract_fit(conf)),
      metric     = as.list(extract_fit(metr)),
      scalar     = as.list(extract_fit(scal)),
      diff_metric = summary(diff_metric),
      diff_scalar = summary(diff_scalar)
    ))

  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}


# ── 9. MODEL COMPARISON ───────────────────────────────────────────

run_model_comparison <- function(data, models, estimator = "MLR") {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  results <- list()
  fitted  <- list()

  for (name in names(models)) {
    tryCatch({
      fit <- lavaan::sem(models[[name]], data = data,
                         estimator = estimator, std.lv = TRUE)
      fi  <- lavaan::fitMeasures(fit, c(
        "chisq", "df", "pvalue", "rmsea", "cfi", "tli", "srmr", "aic", "bic"
      ))
      results[[name]] <- round(fi, 4)
      fitted[[name]]  <- fit
    }, error = function(e) {
      results[[name]] <- list(error = conditionMessage(e))
    })
  }

  # Chi-square difference test between first two models
  diff_test <- NULL
  if (length(fitted) >= 2) {
    tryCatch({
      diff_test <- lavaan::compareFit(fitted[[1]], fitted[[2]])
    }, error = function(e) NULL)
  }

  return(list(
    fit_results = results,
    diff_test   = diff_test
  ))
}


# ── 10. HARMAN'S SINGLE FACTOR TEST ──────────────────────────────

run_harman <- function(data) {
  data <- prepare_data(data)
  data <- data[complete.cases(data), ]

  tryCatch({
    fa_1 <- suppressMessages(suppressWarnings(
      psych::fa(data, nfactors = 1, rotate = "none", fm = "pa")
    ))
    var   <- fa_1$Vaccounted
    prop  <- var["Proportion Var", 1]

    # All eigenvalues for scree plot
    corr_mat <- cor(data, use = "complete.obs")
    ev       <- eigen(corr_mat)$values

    return(list(
      single_factor_var = round(prop, 4),
      eigenvalues       = round(ev, 4),
      cmb_concern       = prop > 0.50
    ))
  }, error = function(e) {
    return(list(error = conditionMessage(e)))
  })
}
