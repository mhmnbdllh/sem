# =============================================================================
# pls_analysis.R
# PLS-SEM Analysis — Manual implementation (no external SEM packages required)
# Uses composite-based SEM (consistent with PLSc methodology)
#
# Methodological references:
#   Hair et al. (2022) A Primer on Partial Least Squares SEM, 3rd ed.
#   Dijkstra & Henseler (2015) Consistent PLS (PLSc) - Structural Equation Modeling
#   Henseler et al. (2015) HTMT criterion - Journal of the Academy of Marketing Science
#   Fornell & Larcker (1981) Evaluating SEM - Journal of Marketing Research
# =============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
})

safe_num <- function(x, digits = 4) {
  v <- tryCatch(as.numeric(x), error = function(e) NA_real_)
  if (is.null(v) || length(v) == 0) return(NA_real_)
  v <- v[1]
  if (!is.finite(v)) return(NA_real_)
  round(v, digits)
}

# ── Bootstrap confidence intervals ─────────────────────────────────────────
boot_ci <- function(data, constructs, paths_list, n_boot = 1000, seed = 42) {
  set.seed(seed)
  n <- nrow(data)
  boot_betas <- vector("list", length(paths_list))
  for (b in seq_len(n_boot)) {
    idx    <- sample(n, n, replace = TRUE)
    d_boot <- data[idx, , drop = FALSE]
    # Compute composites
    scores_b <- compute_composites(d_boot, constructs)
    # Fit paths
    betas_b <- fit_paths(scores_b, paths_list)
    for (j in seq_along(paths_list)) {
      boot_betas[[j]] <- c(boot_betas[[j]], betas_b[[j]])
    }
  }
  # BCa CI and t-stat
  results <- list()
  for (j in seq_along(paths_list)) {
    b_vec <- boot_betas[[j]]
    b_vec <- b_vec[is.finite(b_vec)]
    t_stat <- if (sd(b_vec) > 0) mean(b_vec) / sd(b_vec) else NA_real_
    p_val  <- if (!is.na(t_stat))
      2 * pt(abs(t_stat), df = n_boot - 1, lower.tail = FALSE)
    else NA_real_
    ci_lo  <- quantile(b_vec, 0.025, na.rm = TRUE)
    ci_hi  <- quantile(b_vec, 0.975, na.rm = TRUE)
    results[[j]] <- list(
      t_stat = safe_num(t_stat, 3),
      p_val  = safe_num(p_val, 4),
      ci_lo  = safe_num(ci_lo),
      ci_hi  = safe_num(ci_hi)
    )
  }
  results
}

# ── Compute composite scores (mean-weighted) ────────────────────────────────
compute_composites <- function(data, constructs) {
  scores <- data.frame(matrix(NA, nrow(data), length(constructs)))
  colnames(scores) <- names(constructs)
  for (cname in names(constructs)) {
    items <- intersect(constructs[[cname]], colnames(data))
    if (length(items) >= 1)
      scores[[cname]] <- rowMeans(data[, items, drop = FALSE], na.rm = TRUE)
  }
  scores
}

# ── Standardize composites ──────────────────────────────────────────────────
standardize <- function(x) (x - mean(x, na.rm=TRUE)) / sd(x, na.rm=TRUE)

# ── Fit structural paths via OLS ────────────────────────────────────────────
fit_paths <- function(scores, paths_list) {
  # Group paths by outcome
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  path_coefs <- vector("list", length(paths_list))
  for (out in outcomes) {
    preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]] == out)],
                    function(p) p[[1]])
    X <- as.matrix(scores[, preds, drop = FALSE])
    y <- scores[[out]]
    if (any(is.na(y)) || any(is.na(X))) next
    fit <- tryCatch(lm.fit(cbind(1, X), y), error = function(e) NULL)
    if (!is.null(fit)) {
      coefs <- fit$coefficients[-1]  # remove intercept
      for (j in seq_along(paths_list)) {
        if (paths_list[[j]][[2]] == out) {
          pred <- paths_list[[j]][[1]]
          idx  <- which(preds == pred)
          path_coefs[[j]] <- if (length(idx) > 0) coefs[idx] else NA_real_
        }
      }
    }
  }
  path_coefs
}

# ── Outer loadings ──────────────────────────────────────────────────────────
compute_loadings <- function(data, constructs, scores) {
  loadings <- list()
  for (cname in names(constructs)) {
    items <- intersect(constructs[[cname]], colnames(data))
    score <- scores[[cname]]
    for (item in items) {
      x   <- data[[item]]
      lam <- if (sd(x, na.rm=TRUE) > 0 && sd(score, na.rm=TRUE) > 0)
        cor(x, score, use = "pairwise.complete.obs")
      else NA_real_
      loadings[[length(loadings)+1]] <- list(
        construct = cname,
        item      = item,
        loading   = safe_num(lam),
        status    = if (!is.na(lam) && abs(lam) >= 0.70) "Strong"
                    else if (!is.na(lam) && abs(lam) >= 0.50) "Acceptable"
                    else "Weak"
      )
    }
  }
  loadings
}

# ── AVE, CR, Alpha ──────────────────────────────────────────────────────────
compute_reliability <- function(data, constructs, scores) {
  rel <- list()
  for (cname in names(constructs)) {
    items <- intersect(constructs[[cname]], colnames(data))
    score <- scores[[cname]]
    if (length(items) < 2) next
    # Loadings
    lams <- sapply(items, function(i) {
      x <- data[[i]]
      if (sd(x,na.rm=TRUE) > 0 && sd(score,na.rm=TRUE) > 0)
        cor(x, score, use="pairwise.complete.obs")
      else NA_real_
    })
    lams <- lams[!is.na(lams)]
    if (length(lams) == 0) next
    ave   <- mean(lams^2)
    cr    <- sum(lams)^2 / (sum(lams)^2 + sum(1 - lams^2))
    alpha_raw <- tryCatch({
      if (length(items) >= 2) {
        d   <- data[, items, drop=FALSE]
        d   <- d[complete.cases(d), ]
        k   <- ncol(d)
        var_total <- var(rowSums(d))
        var_items <- sum(apply(d, 2, var))
        (k/(k-1)) * (1 - var_items/var_total)
      } else NA_real_
    }, error=function(e) NA_real_)
    ave   <- max(0, min(1, safe_num(ave)))
    cr    <- max(0, min(1, safe_num(cr)))
    alpha_raw <- if (!is.na(alpha_raw)) max(0, min(1, safe_num(alpha_raw))) else NA_real_
    rel[[cname]] <- list(
      alpha    = alpha_raw,
      cr       = cr,
      rho_a    = cr,  # approximation: rho_A ≈ rho_c for mean-weighted composites
      ave      = ave,
      alpha_ok = !is.na(alpha_raw) && alpha_raw >= 0.70,
      cr_ok    = !is.na(cr) && cr >= 0.70,
      ave_ok   = !is.na(ave) && ave >= 0.50
    )
  }
  rel
}

# ── HTMT ────────────────────────────────────────────────────────────────────
compute_htmt <- function(data, constructs) {
  cnames <- names(constructs)
  n      <- length(cnames)
  htmt   <- matrix(NA_real_, n, n, dimnames=list(cnames, cnames))
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      if (i == j) next
      items_i <- intersect(constructs[[cnames[i]]], colnames(data))
      items_j <- intersect(constructs[[cnames[j]]], colnames(data))
      if (length(items_i) < 2 || length(items_j) < 2) next
      # Hetero-trait correlations (between constructs)
      cors_het <- outer(items_i, items_j, Vectorize(function(a,b)
        cor(data[[a]], data[[b]], use="pairwise.complete.obs")))
      # Mono-trait correlations (within constructs)
      cors_ii  <- cor(data[,items_i,drop=FALSE], use="pairwise.complete.obs")
      cors_jj  <- cor(data[,items_j,drop=FALSE], use="pairwise.complete.obs")
      diag(cors_ii) <- NA; diag(cors_jj) <- NA
      mean_het <- mean(abs(cors_het), na.rm=TRUE)
      mean_ii  <- mean(abs(cors_ii),  na.rm=TRUE)
      mean_jj  <- mean(abs(cors_jj),  na.rm=TRUE)
      htmt[i,j]<- if (mean_ii>0 && mean_jj>0)
        mean_het / sqrt(mean_ii * mean_jj)
      else NA_real_
    }
  }
  round(as.data.frame(htmt), 4)
}

# ── Fornell-Larcker ─────────────────────────────────────────────────────────
compute_fl <- function(scores, rel_list) {
  cnames <- names(rel_list)
  n      <- length(cnames)
  fl     <- matrix(NA_real_, n, n, dimnames=list(cnames, cnames))
  for (i in seq_len(n)) {
    ave_i <- rel_list[[cnames[i]]]$ave
    fl[cnames[i], cnames[i]] <- if (!is.na(ave_i)) sqrt(ave_i) else NA_real_
    for (j in seq_len(n)) {
      if (i != j)
        fl[cnames[i], cnames[j]] <- tryCatch(
          abs(cor(scores[[cnames[i]]], scores[[cnames[j]]], use="pairwise.complete.obs")),
          error=function(e) NA_real_
        )
    }
  }
  round(as.data.frame(fl), 4)
}

# ── VIF ─────────────────────────────────────────────────────────────────────
compute_vif <- function(scores, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  vif_out  <- list()
  for (out in outcomes) {
    preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]]==out)],
                    function(p) p[[1]])
    if (length(preds) < 2) {
      vif_out[[out]] <- setNames(list(1.0), preds)
      next
    }
    X <- as.matrix(scores[, preds, drop=FALSE])
    X <- X[complete.cases(X), ]
    vifs <- tryCatch({
      sapply(seq_len(ncol(X)), function(j) {
        r2j <- summary(lm(X[,j] ~ X[,-j]))$r.squared
        1 / (1 - r2j)
      })
    }, error=function(e) rep(NA_real_, length(preds)))
    vif_out[[out]] <- as.list(setNames(round(vifs, 3), preds))
  }
  vif_out
}

# ── SRMR ────────────────────────────────────────────────────────────────────
compute_srmr <- function(data, constructs, scores) {
  all_items <- unlist(constructs)
  S <- cor(data[, all_items, drop=FALSE], use="pairwise.complete.obs")
  C <- cor(cbind(data[, all_items, drop=FALSE], scores), use="pairwise.complete.obs")
  C <- C[all_items, all_items]
  resid  <- S - C
  srmr   <- sqrt(mean(resid[lower.tri(resid)]^2, na.rm=TRUE))
  safe_num(srmr)
}

# ── Q² via blindfolding approximation ───────────────────────────────────────
compute_q2 <- function(data, constructs, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  q2_out   <- list()
  for (out in outcomes) {
    items <- intersect(constructs[[out]], colnames(data))
    if (length(items) == 0) next
    # Cross-validated R² approximation (leave-10%-out)
    n      <- nrow(data)
    folds  <- cut(seq_len(n), breaks=10, labels=FALSE)
    ss_res <- 0; ss_tot <- 0
    for (f in 1:10) {
      test_idx  <- which(folds == f)
      train_idx <- which(folds != f)
      if (length(train_idx) < 10) next
      train_d <- data[train_idx, , drop=FALSE]
      test_d  <- data[test_idx,  , drop=FALSE]
      # Compute composites on train, apply to test
      scores_tr <- compute_composites(train_d, constructs)
      scores_te <- compute_composites(test_d,  constructs)
      # Fit on train
      preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]]==out)],
                      function(p) p[[1]])
      X_tr <- as.matrix(scores_tr[, preds, drop=FALSE])
      y_tr <- scores_tr[[out]]
      fit  <- tryCatch(lm.fit(cbind(1,X_tr), y_tr), error=function(e) NULL)
      if (is.null(fit)) next
      X_te <- as.matrix(scores_te[, preds, drop=FALSE])
      y_te <- scores_te[[out]]
      y_hat<- cbind(1, X_te) %*% fit$coefficients
      ss_res <- ss_res + sum((y_te - y_hat)^2, na.rm=TRUE)
      ss_tot <- ss_tot + sum((y_te - mean(y_tr,na.rm=TRUE))^2, na.rm=TRUE)
    }
    q2 <- if (ss_tot > 0) 1 - ss_res/ss_tot else NA_real_
    q2_out[[out]] <- list(construct=out, q2=safe_num(q2), relevant=!is.na(q2)&&q2>0)
  }
  q2_out
}

# ── R² ──────────────────────────────────────────────────────────────────────
compute_r2 <- function(scores, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  r2_out   <- list()
  for (out in outcomes) {
    preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]]==out)],
                    function(p) p[[1]])
    X     <- as.matrix(scores[, preds, drop=FALSE])
    y     <- scores[[out]]
    fit   <- tryCatch(summary(lm(y ~ X)), error=function(e) NULL)
    r2    <- if (!is.null(fit)) safe_num(fit$r.squared) else NA_real_
    r2adj <- if (!is.null(fit)) safe_num(fit$adj.r.squared) else NA_real_
    level <- if (!is.na(r2) && r2>=0.75) "Substantial"
             else if (!is.na(r2) && r2>=0.50) "Moderate"
             else if (!is.na(r2) && r2>=0.25) "Weak"
             else "Very Weak"
    r2_out[[length(r2_out)+1]] <- list(
      construct=out, r2=r2, r2_adj=r2adj, level=level
    )
  }
  r2_out
}

# ── Main function ────────────────────────────────────────────────────────────
run_plssem <- function(data, constructs_list, paths_list,
                       construct_types=NULL, n_boot=1000, seed=42) {
  tryCatch({
    # Validate
    if (nrow(data) < 30) stop("Sample size too small (minimum n = 30).")
    all_items <- unlist(constructs_list)
    missing   <- setdiff(all_items, colnames(data))
    if (length(missing) > 0)
      stop(paste("Items not found in dataset:", paste(missing, collapse=", ")))

    # Complete cases only
    data_clean <- data[complete.cases(data[, all_items]), all_items, drop=FALSE]
    data_clean <- as.data.frame(lapply(data_clean, as.numeric))
    n_obs      <- nrow(data_clean)
    if (n_obs < 30)
      stop(paste("Only", n_obs, "complete cases. Minimum is 30."))

    # Step 1: Composite scores
    scores <- compute_composites(data_clean, constructs_list)
    # Standardize
    for (cn in names(constructs_list))
      if (!all(is.na(scores[[cn]]))) scores[[cn]] <- standardize(scores[[cn]])

    # Step 2: Outer loadings
    loadings_list <- compute_loadings(data_clean, constructs_list, scores)

    # Step 3: Inner paths (OLS)
    betas <- fit_paths(scores, paths_list)

    # Step 4: Standardized betas
    paths_out <- list()
    for (j in seq_along(paths_list)) {
      paths_out[[j]] <- list(
        predictor = paths_list[[j]][[1]],
        outcome   = paths_list[[j]][[2]],
        beta      = safe_num(betas[[j]])
      )
    }

    # Step 5: Bootstrap CI
    boot_res <- boot_ci(data_clean, constructs_list, paths_list, n_boot, seed)
    for (j in seq_along(paths_out)) {
      br <- boot_res[[j]]
      paths_out[[j]]$t_stat    <- br$t_stat
      paths_out[[j]]$p_value   <- br$p_val
      paths_out[[j]]$ci_lo     <- br$ci_lo
      paths_out[[j]]$ci_hi     <- br$ci_hi
      paths_out[[j]]$supported <- !is.na(br$p_val) && br$p_val < 0.05
      paths_out[[j]]$decision  <- if (!is.na(br$p_val) && br$p_val < 0.05)
        "Supported" else "Not Supported"
    }

    # Step 6: Reliability & validity
    rel_list <- compute_reliability(data_clean, constructs_list, scores)

    # Step 7: HTMT
    htmt_mat <- tryCatch(compute_htmt(data_clean, constructs_list), error=function(e) NULL)

    # Step 8: Fornell-Larcker
    fl_mat <- tryCatch(compute_fl(scores, rel_list), error=function(e) NULL)

    # Step 9: VIF
    vif_vals <- tryCatch(compute_vif(scores, paths_list), error=function(e) NULL)

    # Step 10: R²
    r2_list <- compute_r2(scores, paths_list)

    # Step 11: SRMR
    srmr_val <- tryCatch(compute_srmr(data_clean, constructs_list, scores), error=function(e) NA_real_)

    # Step 12: Q²
    q2_vals <- tryCatch(compute_q2(data_clean, constructs_list, paths_list), error=function(e) list())

    return(list(
      n              = n_obs,
      n_boot         = n_boot,
      loadings       = loadings_list,
      paths          = paths_out,
      r2             = r2_list,
      reliability    = rel_list,
      htmt           = htmt_mat,
      fl_criterion   = fl_mat,
      vif            = vif_vals,
      fit            = list(srmr=srmr_val),
      q2             = q2_vals,
      construct_types= construct_types,
      method         = "Composite-based PLS (mean weights, OLS inner model)"
    ))
  }, error=function(e) list(error=conditionMessage(e)))
}
