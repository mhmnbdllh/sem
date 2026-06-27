# =============================================================================
# pls_analysis.R
# PLS-SEM Analysis via seminr package
# Methodological reference:
#   Hair et al. (2022) A Primer on Partial Least Squares SEM, 3rd ed.
#   Rademaker & Schuberth (2020) seminr: Building and Estimating SEM in R
# =============================================================================

suppressPackageStartupMessages({
  library(seminr)
  library(jsonlite)
})

# ── Helper: safe numeric ──────────────────────────────────────────────────────
safe_num <- function(x, digits = 4) {
  v <- tryCatch(as.numeric(x), error = function(e) NA_real_)
  if (is.null(v) || length(v) == 0) return(NA_real_)
  v <- v[1]
  if (!is.finite(v)) return(NA_real_)
  round(v, digits)
}

safe_mat <- function(mat, digits = 4) {
  if (is.null(mat)) return(NULL)
  m <- apply(mat, c(1, 2), safe_num, digits = digits)
  as.data.frame(m)
}

# ── Build measurement model from constructs list ──────────────────────────────
build_mm <- function(constructs_list, construct_types) {
  # construct_types: named list, values "reflective" or "formative"
  args <- list()
  for (cname in names(constructs_list)) {
    items <- constructs_list[[cname]]
    ctype <- if (!is.null(construct_types[[cname]])) construct_types[[cname]] else "reflective"

    if (ctype == "reflective") {
      args[[length(args) + 1]] <- composite(cname, item_names = items, weights = mode_A)
    } else {
      # Formative: Mode B weights
      args[[length(args) + 1]] <- composite(cname, item_names = items, weights = mode_B)
    }
  }
  do.call(constructs, args)
}

# ── Build structural model from paths list ────────────────────────────────────
build_sm <- function(paths_list) {
  # paths_list: list of c(from, to)
  # Group by 'from' for efficiency
  from_map <- list()
  for (p in paths_list) {
    frm <- p[1]; to <- p[2]
    from_map[[frm]] <- c(from_map[[frm]], to)
  }
  args <- list()
  for (frm in names(from_map)) {
    args[[length(args) + 1]] <- paths(from = frm, to = from_map[[frm]])
  }
  do.call(relationships, args)
}

# ── Main PLS-SEM function ─────────────────────────────────────────────────────
run_plssem <- function(data, constructs_list, paths_list,
                       construct_types = NULL,
                       n_boot = 1000, seed = 42) {

  tryCatch({

    # Validate inputs
    if (nrow(data) < 30) stop("Sample size too small for PLS-SEM (minimum n = 30).")
    all_items <- unlist(constructs_list)
    missing   <- setdiff(all_items, colnames(data))
    if (length(missing) > 0) stop(paste("Missing items:", paste(missing, collapse = ", ")))

    # Use only complete cases
    data_clean <- data[complete.cases(data[, all_items]), all_items]
    n_obs <- nrow(data_clean)

    if (n_obs < 30) stop(paste("Only", n_obs, "complete cases. Minimum is 30."))

    # Convert to numeric
    data_clean <- as.data.frame(lapply(data_clean, as.numeric))

    # Build measurement and structural models
    if (is.null(construct_types)) {
      construct_types <- setNames(
        rep(list("reflective"), length(constructs_list)),
        names(constructs_list)
      )
    }

    mm <- build_mm(constructs_list, construct_types)
    sm <- build_sm(paths_list)

    # Estimate PLS model
    set.seed(seed)
    pls_model <- estimate_pls(
      data               = data_clean,
      measurement_model  = mm,
      structural_model   = sm,
      inner_weights      = path_weighting,  # path weighting scheme (Hair et al., 2022)
      missing            = mean_replacement  # mean replacement for missing
    )

    # Bootstrap for significance testing
    set.seed(seed)
    boot_model <- bootstrap_model(
      seminr_model = pls_model,
      nboot        = n_boot,
      cores        = 1,
      seed         = seed
    )

    # ── Extract outer loadings ─────────────────────────────────────────────
    outer_loads <- pls_model$outer_loadings
    loadings_list <- list()
    for (cname in names(constructs_list)) {
      items <- constructs_list[[cname]]
      ctype <- if (!is.null(construct_types[[cname]])) construct_types[[cname]] else "reflective"
      for (item in items) {
        lam <- tryCatch(outer_loads[item, cname], error = function(e) NA_real_)
        loadings_list[[length(loadings_list) + 1]] <- list(
          construct = cname,
          item      = item,
          type      = ctype,
          loading   = safe_num(lam),
          status    = if (!is.na(lam) && abs(lam) >= 0.70) "Strong"
                      else if (!is.na(lam) && abs(lam) >= 0.50) "Acceptable"
                      else "Weak"
        )
      }
    }

    # ── Extract path coefficients ──────────────────────────────────────────
    path_coef   <- pls_model$path_coef
    boot_paths  <- boot_model$bootstrapped_paths
    paths_out   <- list()

    for (p in paths_list) {
      frm <- p[1]; to <- p[2]
      beta <- tryCatch(path_coef[frm, to], error = function(e) NA_real_)

      # Bootstrap CI and t-statistic
      boot_row_name <- paste0(frm, " -> ", to)
      t_val  <- NA_real_; p_val <- NA_real_
      ci_lo  <- NA_real_; ci_hi <- NA_real_

      # Try different naming conventions in seminr
      boot_names <- rownames(boot_paths)
      possible_names <- c(
        paste0(frm, " -> ", to),
        paste0(to, " ~ ", frm),
        paste0(frm, "->", to)
      )
      matched_name <- intersect(possible_names, boot_names)

      if (length(matched_name) > 0) {
        row <- boot_paths[matched_name[1], ]
        t_val <- tryCatch(as.numeric(row["T Stat."]), error = function(e) NA_real_)
        p_val <- tryCatch(as.numeric(row["p-value"]), error = function(e) {
          # Two-tailed p from t
          if (!is.na(t_val)) 2 * pt(abs(t_val), df = n_boot - 1, lower.tail = FALSE)
          else NA_real_
        })
        ci_lo <- tryCatch(as.numeric(row["2.5% CI"]), error = function(e) NA_real_)
        ci_hi <- tryCatch(as.numeric(row["97.5% CI"]), error = function(e) NA_real_)
      }

      sig <- !is.na(p_val) && p_val < 0.05
      paths_out[[length(paths_out) + 1]] <- list(
        predictor = frm,
        outcome   = to,
        beta      = safe_num(beta),
        t_stat    = safe_num(t_val, 3),
        p_value   = safe_num(p_val, 4),
        ci_lo     = safe_num(ci_lo),
        ci_hi     = safe_num(ci_hi),
        supported = sig,
        decision  = if (sig) "Supported" else "Not Supported"
      )
    }

    # ── R² for endogenous constructs ───────────────────────────────────────
    r2_vals <- pls_model$rSquared
    r2_list <- list()
    endo <- unique(sapply(paths_list, function(p) p[2]))
    for (cname in endo) {
      r2 <- tryCatch(as.numeric(r2_vals[cname, 1]), error = function(e) NA_real_)
      r2_adj <- tryCatch(as.numeric(r2_vals[cname, "Adj. R-Squared"]),
                          error = function(e) NA_real_)
      r2_list[[length(r2_list) + 1]] <- list(
        construct   = cname,
        r2          = safe_num(r2),
        r2_adj      = safe_num(r2_adj),
        level       = if (!is.na(r2) && r2 >= 0.75) "Substantial"
                      else if (!is.na(r2) && r2 >= 0.50) "Moderate"
                      else if (!is.na(r2) && r2 >= 0.25) "Weak"
                      else "Very Weak"
      )
    }

    # ── Reliability & Validity ─────────────────────────────────────────────
    rel <- pls_model$reliability
    rel_list <- list()
    for (cname in names(constructs_list)) {
      alpha  <- tryCatch(as.numeric(rel[cname, "alpha"]),     error = function(e) NA_real_)
      cr     <- tryCatch(as.numeric(rel[cname, "rho_c"]),     error = function(e) NA_real_)
      ave    <- tryCatch(as.numeric(rel[cname, "AVE"]),        error = function(e) NA_real_)
      rho_a  <- tryCatch(as.numeric(rel[cname, "rho_A"]),     error = function(e) NA_real_)

      # Validate ranges
      alpha <- if (!is.na(alpha) && alpha >= 0 && alpha <= 1) alpha else NA_real_
      cr    <- if (!is.na(cr)    && cr    >= 0 && cr    <= 1) cr    else NA_real_
      ave   <- if (!is.na(ave)   && ave   >= 0 && ave   <= 1) ave   else NA_real_

      rel_list[[cname]] <- list(
        alpha   = safe_num(alpha),
        cr      = safe_num(cr),        # rho_c = composite reliability
        rho_a   = safe_num(rho_a),     # rho_A = Dijkstra-Henseler reliability
        ave     = safe_num(ave),
        alpha_ok = !is.na(alpha) && alpha >= 0.70,
        cr_ok    = !is.na(cr)    && cr    >= 0.70,
        ave_ok   = !is.na(ave)   && ave   >= 0.50
      )
    }

    # ── HTMT discriminant validity ─────────────────────────────────────────
    htmt_mat <- tryCatch(
      as.data.frame(as.matrix(htmt(pls_model))),
      error = function(e) NULL
    )

    # ── Fornell-Larcker criterion ──────────────────────────────────────────
    fl_mat <- tryCatch({
      cors   <- cor(pls_model$scores, use = "pairwise.complete.obs")
      n_cons <- length(constructs_list)
      fl     <- matrix(NA_real_, n_cons, n_cons,
                       dimnames = list(names(constructs_list), names(constructs_list)))
      for (i in seq_along(names(constructs_list))) {
        cn_i <- names(constructs_list)[i]
        ave_i <- rel_list[[cn_i]]$ave
        fl[cn_i, cn_i] <- if (!is.na(ave_i)) sqrt(ave_i) else NA_real_
        for (j in seq_along(names(constructs_list))) {
          if (i != j) {
            cn_j <- names(constructs_list)[j]
            fl[cn_i, cn_j] <- tryCatch(abs(cors[cn_i, cn_j]), error = function(e) NA_real_)
          }
        }
      }
      as.data.frame(fl)
    }, error = function(e) NULL)

    # ── VIF for inner model (collinearity) ────────────────────────────────
    vif_vals <- tryCatch({
      vif_list <- all_vifs(pls_model)
      lapply(vif_list, function(v) {
        if (is.numeric(v)) round(v, 3) else v
      })
    }, error = function(e) NULL)

    # ── Model fit (SRMR) ───────────────────────────────────────────────────
    # PLS-SEM uses SRMR as primary fit index (Henseler et al., 2015)
    # Threshold: SRMR < 0.08 (Hair et al., 2022)
    fit_indices <- tryCatch({
      fit <- model_fit(pls_model)
      list(
        srmr    = safe_num(fit$Saturated$SRMR),
        rms_theta = safe_num(tryCatch(fit$Estimated$RMS_theta, error=function(e) NA_real_)),
        nfi     = safe_num(tryCatch(fit$Saturated$NFI, error=function(e) NA_real_))
      )
    }, error = function(e) {
      list(srmr = NA_real_, rms_theta = NA_real_, nfi = NA_real_)
    })

    # ── Predictive relevance Q² ────────────────────────────────────────────
    # Q² > 0 = model has predictive relevance for that construct
    q2_vals <- tryCatch({
      pa   <- predictive_accuracy(pls_model)
      q2_out <- list()
      for (cname in endo) {
        q2 <- tryCatch(as.numeric(pa[cname, "q_square"]), error = function(e) NA_real_)
        q2_out[[cname]] <- list(
          construct = cname,
          q2        = safe_num(q2),
          relevant  = !is.na(q2) && q2 > 0
        )
      }
      q2_out
    }, error = function(e) list())

    # ── Final result ───────────────────────────────────────────────────────
    return(list(
      n           = n_obs,
      n_boot      = n_boot,
      loadings    = loadings_list,
      paths       = paths_out,
      r2          = r2_list,
      reliability = rel_list,
      htmt        = htmt_mat,
      fl_criterion = fl_mat,
      vif         = vif_vals,
      fit         = fit_indices,
      q2          = q2_vals,
      construct_types = construct_types
    ))

  }, error = function(e) {
    list(error = conditionMessage(e))
  })
}
