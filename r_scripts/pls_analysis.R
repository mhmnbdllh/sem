# =============================================================================
# pls_analysis.R
# PLS-SEM Analysis — NIPALS Algorithm (standard PLS-SEM procedure)
#
# Implements the classic PLS-SEM algorithm exactly as described in:
#   Wold, H. (1982). Soft modeling: The basic design and some extensions.
#   Lohmoller, J.B. (1989). Latent Variable Path Modeling with Partial Least Squares.
#   Hair, J.F., Hult, G.T.M., Ringle, C.M., Sarstedt, M. (2022).
#     A Primer on Partial Least Squares Structural Equation Modeling (PLS-SEM), 3rd ed.
#   Henseler, J., Ringle, C.M., Sarstedt, M. (2015). HTMT criterion.
#   Dijkstra, T.K., Henseler, J. (2015). Consistent PLS (rho_A reliability).
#   Fornell, C., Larcker, D.F. (1981). AVE and discriminant validity.
#
# Zero-dependency implementation using only base R for reliable deployment.
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

standardize <- function(x) {
  s <- sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(x - mean(x, na.rm = TRUE))
  (x - mean(x, na.rm = TRUE)) / s
}

build_adjacency <- function(constructs_list, paths_list) {
  cnames <- names(constructs_list)
  predecessors <- setNames(vector("list", length(cnames)), cnames)
  successors   <- setNames(vector("list", length(cnames)), cnames)
  for (cn in cnames) { predecessors[[cn]] <- character(0); successors[[cn]] <- character(0) }
  for (p in paths_list) {
    frm <- p[[1]]; to <- p[[2]]
    successors[[frm]]  <- union(successors[[frm]], to)
    predecessors[[to]] <- union(predecessors[[to]], frm)
  }
  list(predecessors = predecessors, successors = successors)
}

# NIPALS: iterative outer weight estimation (the core PLS-SEM algorithm)
nipals_pls <- function(data, constructs_list, construct_types,
                       adjacency, max_iter = 300, tol = 1e-7) {
  cnames <- names(constructs_list)
  n      <- nrow(data)

  X_list <- list()
  for (cn in cnames) {
    items <- constructs_list[[cn]]
    Xk    <- as.matrix(data[, items, drop = FALSE])
    Xk    <- apply(Xk, 2, standardize)
    X_list[[cn]] <- Xk
  }

  weights <- list()
  for (cn in cnames) {
    p <- ncol(X_list[[cn]])
    weights[[cn]] <- rep(1 / sqrt(p), p)
  }

  Y <- matrix(NA_real_, n, length(cnames), dimnames = list(NULL, cnames))
  for (cn in cnames) {
    Y[, cn] <- standardize(as.vector(X_list[[cn]] %*% weights[[cn]]))
  }

  converged  <- FALSE
  iter_count <- 0

  for (iter in seq_len(max_iter)) {
    iter_count  <- iter
    weights_old <- weights

    Z <- matrix(NA_real_, n, length(cnames), dimnames = list(NULL, cnames))
    for (cn in cnames) {
      neighbors <- union(adjacency$predecessors[[cn]], adjacency$successors[[cn]])
      if (length(neighbors) == 0) { Z[, cn] <- Y[, cn]; next }
      inner_sum <- rep(0, n)
      for (nb in neighbors) {
        r <- cor(Y[, cn], Y[, nb], use = "pairwise.complete.obs")
        sign_r <- if (is.na(r)) 0 else sign(r)
        inner_sum <- inner_sum + sign_r * Y[, nb]
      }
      Z[, cn] <- standardize(inner_sum)
    }

    for (cn in cnames) {
      Xk    <- X_list[[cn]]
      ztgt  <- Z[, cn]
      ctype <- if (!is.null(construct_types[[cn]])) construct_types[[cn]] else "reflective"
      if (ctype == "reflective") {
        w <- as.vector(t(Xk) %*% ztgt) / n
      } else {
        w <- tryCatch({ lm.fit(Xk, ztgt)$coefficients },
                      error = function(e) as.vector(t(Xk) %*% ztgt) / n)
      }
      raw_score <- as.vector(Xk %*% w)
      sdv <- sd(raw_score, na.rm = TRUE)
      if (!is.na(sdv) && sdv > 0) w <- w / sdv
      weights[[cn]] <- w
    }

    for (cn in cnames) {
      Y[, cn] <- standardize(as.vector(X_list[[cn]] %*% weights[[cn]]))
    }

    max_delta <- 0
    for (cn in cnames) {
      delta <- max(abs(weights[[cn]] - weights_old[[cn]]))
      if (!is.na(delta)) max_delta <- max(max_delta, delta)
    }
    if (max_delta < tol) { converged <- TRUE; break }
  }

  list(weights = weights, scores = Y, X_list = X_list,
       converged = converged, iterations = iter_count)
}

compute_loadings <- function(constructs_list, X_list, scores, construct_types) {
  loadings <- list()
  for (cn in names(constructs_list)) {
    items <- constructs_list[[cn]]
    Xk    <- X_list[[cn]]
    yk    <- scores[, cn]
    ctype <- if (!is.null(construct_types[[cn]])) construct_types[[cn]] else "reflective"
    for (j in seq_along(items)) {
      lam <- tryCatch(cor(Xk[, j], yk, use = "pairwise.complete.obs"), error = function(e) NA_real_)
      loadings[[length(loadings) + 1]] <- list(
        construct = cn, item = items[j], type = ctype, loading = safe_num(lam),
        status = if (!is.na(lam) && abs(lam) >= 0.70) "Strong"
                 else if (!is.na(lam) && abs(lam) >= 0.50) "Acceptable" else "Weak"
      )
    }
  }
  loadings
}

fit_inner_paths <- function(scores, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  betas    <- vector("list", length(paths_list))
  for (out in outcomes) {
    idxs  <- which(sapply(paths_list, function(p) p[[2]] == out))
    preds <- sapply(paths_list[idxs], function(p) p[[1]])
    X     <- as.matrix(scores[, preds, drop = FALSE])
    y     <- scores[, out]
    fit   <- tryCatch(lm.fit(cbind(1, X), y), error = function(e) NULL)
    if (!is.null(fit)) {
      coefs <- fit$coefficients[-1]
      for (k in seq_along(idxs)) {
        betas[[idxs[k]]] <- if (length(preds) == 1) coefs else coefs[k]
      }
    }
  }
  betas
}

boot_ci <- function(data, constructs_list, construct_types, adjacency,
                    paths_list, n_boot = 1000, seed = 42) {
  set.seed(seed)
  n <- nrow(data)
  boot_betas <- vector("list", length(paths_list))
  for (b in seq_len(n_boot)) {
    idx    <- sample(n, n, replace = TRUE)
    d_boot <- data[idx, , drop = FALSE]
    nip_b  <- tryCatch(nipals_pls(d_boot, constructs_list, construct_types, adjacency, max_iter = 100),
                       error = function(e) NULL)
    if (is.null(nip_b)) next
    betas_b <- fit_inner_paths(nip_b$scores, paths_list)
    for (j in seq_along(paths_list)) {
      val <- betas_b[[j]]
      if (!is.null(val) && length(val) == 1 && is.finite(val))
        boot_betas[[j]] <- c(boot_betas[[j]], val)
    }
  }
  results <- list()
  for (j in seq_along(paths_list)) {
    b_vec <- boot_betas[[j]]
    b_vec <- b_vec[is.finite(b_vec)]
    if (length(b_vec) < 10) {
      results[[j]] <- list(t_stat = NA_real_, p_val = NA_real_, ci_lo = NA_real_, ci_hi = NA_real_)
      next
    }
    se_boot <- sd(b_vec)
    t_stat  <- if (se_boot > 0) mean(b_vec) / se_boot else NA_real_
    p_val   <- if (!is.na(t_stat)) 2 * pt(abs(t_stat), df = length(b_vec) - 1, lower.tail = FALSE) else NA_real_
    results[[j]] <- list(
      t_stat = safe_num(t_stat, 3), p_val = safe_num(p_val, 4),
      ci_lo  = safe_num(quantile(b_vec, 0.025, na.rm = TRUE)),
      ci_hi  = safe_num(quantile(b_vec, 0.975, na.rm = TRUE))
    )
  }
  results
}

compute_reliability <- function(constructs_list, X_list, weights, scores) {
  rel <- list()
  for (cn in names(constructs_list)) {
    items <- constructs_list[[cn]]
    if (length(items) < 2) next
    Xk <- X_list[[cn]]; yk <- scores[, cn]; w <- weights[[cn]]

    lams <- sapply(seq_along(items), function(j)
      tryCatch(cor(Xk[, j], yk, use = "pairwise.complete.obs"), error = function(e) NA_real_))
    lams_valid <- lams[!is.na(lams)]
    if (length(lams_valid) == 0) next

    ave <- mean(lams_valid^2)
    cr  <- (sum(lams_valid))^2 / ((sum(lams_valid))^2 + sum(1 - lams_valid^2))

    k <- ncol(Xk)
    alpha <- tryCatch({
      cov_mat   <- cov(Xk, use = "pairwise.complete.obs")
      var_sum   <- sum(diag(cov_mat))
      total_var <- sum(cov_mat)
      (k / (k - 1)) * (1 - var_sum / total_var)
    }, error = function(e) NA_real_)

    rho_a <- tryCatch({
      S <- cor(Xk, use = "pairwise.complete.obs")
      diag(S) <- 0
      numerator   <- (sum(w))^2 - sum(w^2)
      denominator <- as.numeric(t(w) %*% S %*% w)
      if (!is.na(denominator) && denominator > 0) numerator / denominator else NA_real_
    }, error = function(e) NA_real_)

    clamp01 <- function(v) if (is.na(v)) NA_real_ else max(0, min(1, v))

    rel[[cn]] <- list(
      alpha = safe_num(clamp01(alpha)), cr = safe_num(clamp01(cr)),
      rho_a = safe_num(clamp01(rho_a)), ave = safe_num(clamp01(ave)),
      alpha_ok = !is.na(alpha) && clamp01(alpha) >= 0.70,
      cr_ok    = !is.na(cr)    && clamp01(cr)    >= 0.70,
      ave_ok   = !is.na(ave)   && clamp01(ave)   >= 0.50
    )
  }
  rel
}

compute_htmt <- function(constructs_list, X_list) {
  cnames <- names(constructs_list)
  k      <- length(cnames)
  htmt   <- matrix(NA_real_, k, k, dimnames = list(cnames, cnames))
  for (i in seq_len(k)) {
    for (j in seq_len(k)) {
      if (i == j) next
      Xi <- X_list[[cnames[i]]]; Xj <- X_list[[cnames[j]]]
      if (ncol(Xi) < 2 || ncol(Xj) < 2) next
      het <- outer(seq_len(ncol(Xi)), seq_len(ncol(Xj)),
                   Vectorize(function(a, b) cor(Xi[, a], Xj[, b], use = "pairwise.complete.obs")))
      cii <- cor(Xi, use = "pairwise.complete.obs"); diag(cii) <- NA
      cjj <- cor(Xj, use = "pairwise.complete.obs"); diag(cjj) <- NA
      m_het <- mean(abs(het), na.rm = TRUE)
      m_ii  <- mean(abs(cii), na.rm = TRUE)
      m_jj  <- mean(abs(cjj), na.rm = TRUE)
      htmt[i, j] <- if (m_ii > 0 && m_jj > 0) m_het / sqrt(m_ii * m_jj) else NA_real_
    }
  }
  round(as.data.frame(htmt), 4)
}

compute_fl <- function(scores, rel_list) {
  cnames <- names(rel_list)
  k      <- length(cnames)
  fl     <- matrix(NA_real_, k, k, dimnames = list(cnames, cnames))
  for (i in seq_len(k)) {
    ave_i <- rel_list[[cnames[i]]]$ave
    fl[cnames[i], cnames[i]] <- if (!is.na(ave_i)) sqrt(ave_i) else NA_real_
    for (j in seq_len(k)) {
      if (i != j)
        fl[cnames[i], cnames[j]] <- tryCatch(
          abs(cor(scores[, cnames[i]], scores[, cnames[j]], use = "pairwise.complete.obs")),
          error = function(e) NA_real_)
    }
  }
  round(as.data.frame(fl), 4)
}

compute_vif <- function(scores, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  vif_out  <- list()
  for (out in outcomes) {
    preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]] == out)], function(p) p[[1]])
    if (length(preds) < 2) { vif_out[[out]] <- setNames(list(1.0), preds); next }
    X <- as.matrix(scores[, preds, drop = FALSE])
    vifs <- tryCatch(
      sapply(seq_len(ncol(X)), function(j) {
        r2j <- summary(lm(X[, j] ~ X[, -j]))$r.squared
        1 / (1 - r2j)
      }), error = function(e) rep(NA_real_, length(preds)))
    vif_out[[out]] <- as.list(setNames(round(vifs, 3), preds))
  }
  vif_out
}

compute_r2 <- function(scores, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  r2_out   <- list()
  for (out in outcomes) {
    preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]] == out)], function(p) p[[1]])
    X     <- as.matrix(scores[, preds, drop = FALSE])
    y     <- scores[, out]
    fit   <- tryCatch(summary(lm(y ~ X)), error = function(e) NULL)
    r2    <- if (!is.null(fit)) safe_num(fit$r.squared) else NA_real_
    r2adj <- if (!is.null(fit)) safe_num(fit$adj.r.squared) else NA_real_
    level <- if (!is.na(r2) && r2 >= 0.75) "Substantial"
             else if (!is.na(r2) && r2 >= 0.50) "Moderate"
             else if (!is.na(r2) && r2 >= 0.25) "Weak" else "Very Weak"
    r2_out[[length(r2_out) + 1]] <- list(construct = out, r2 = r2, r2_adj = r2adj, level = level)
  }
  r2_out
}

compute_srmr <- function(data, constructs_list, scores) {
  all_items <- unlist(constructs_list)
  S <- cor(data[, all_items, drop = FALSE], use = "pairwise.complete.obs")
  C <- cor(cbind(data[, all_items, drop = FALSE], scores), use = "pairwise.complete.obs")
  C <- C[all_items, all_items]
  resid <- S - C
  safe_num(sqrt(mean(resid[lower.tri(resid)]^2, na.rm = TRUE)))
}

compute_q2 <- function(data, constructs_list, construct_types, adjacency, paths_list) {
  outcomes <- unique(sapply(paths_list, function(p) p[[2]]))
  q2_out   <- list()
  n <- nrow(data)
  n_folds <- min(10, n)
  folds <- cut(seq_len(n), breaks = n_folds, labels = FALSE)
  for (out in outcomes) {
    ss_res <- 0; ss_tot <- 0
    for (f in seq_len(n_folds)) {
      test_idx  <- which(folds == f)
      train_idx <- which(folds != f)
      if (length(train_idx) < 15 || length(test_idx) < 1) next
      train_d <- data[train_idx, , drop = FALSE]
      test_d  <- data[test_idx, , drop = FALSE]
      nip_tr  <- tryCatch(nipals_pls(train_d, constructs_list, construct_types, adjacency, max_iter = 100),
                          error = function(e) NULL)
      if (is.null(nip_tr)) next
      preds <- sapply(paths_list[sapply(paths_list, function(p) p[[2]] == out)], function(p) p[[1]])
      X_tr  <- as.matrix(nip_tr$scores[, preds, drop = FALSE])
      y_tr  <- nip_tr$scores[, out]
      fit   <- tryCatch(lm.fit(cbind(1, X_tr), y_tr), error = function(e) NULL)
      if (is.null(fit)) next
      X_te_list <- list()
      for (cn in names(constructs_list)) {
        items <- constructs_list[[cn]]
        Xte   <- as.matrix(test_d[, items, drop = FALSE])
        Xte   <- apply(Xte, 2, standardize)
        X_te_list[[cn]] <- Xte
      }
      scores_te <- sapply(names(constructs_list), function(cn)
        as.vector(X_te_list[[cn]] %*% nip_tr$weights[[cn]]))
      X_te  <- as.matrix(scores_te[, preds, drop = FALSE])
      y_te  <- scores_te[, out]
      y_hat <- cbind(1, X_te) %*% fit$coefficients
      ss_res <- ss_res + sum((y_te - y_hat)^2, na.rm = TRUE)
      ss_tot <- ss_tot + sum((y_te - mean(y_tr, na.rm = TRUE))^2, na.rm = TRUE)
    }
    q2 <- if (ss_tot > 0) 1 - ss_res / ss_tot else NA_real_
    q2_out[[out]] <- list(construct = out, q2 = safe_num(q2), relevant = !is.na(q2) && q2 > 0)
  }
  q2_out
}

run_plssem <- function(data, constructs_list, paths_list,
                       construct_types = NULL, n_boot = 1000, seed = 42) {
  tryCatch({
    if (nrow(data) < 30) stop("Sample size too small (minimum n = 30).")
    all_items <- unlist(constructs_list)
    missing   <- setdiff(all_items, colnames(data))
    if (length(missing) > 0)
      stop(paste("Items not found in dataset:", paste(missing, collapse = ", ")))

    data_clean <- data[complete.cases(data[, all_items]), all_items, drop = FALSE]
    data_clean <- as.data.frame(lapply(data_clean, as.numeric))
    n_obs <- nrow(data_clean)
    if (n_obs < 30) stop(paste("Only", n_obs, "complete cases. Minimum is 30."))

    if (is.null(construct_types)) {
      construct_types <- setNames(as.list(rep("reflective", length(constructs_list))),
                                   names(constructs_list))
    }

    adjacency <- build_adjacency(constructs_list, paths_list)

    nip <- nipals_pls(data_clean, constructs_list, construct_types, adjacency,
                      max_iter = 300, tol = 1e-7)

    loadings_list <- compute_loadings(constructs_list, nip$X_list, nip$scores, construct_types)

    betas <- fit_inner_paths(nip$scores, paths_list)
    paths_out <- list()
    for (j in seq_along(paths_list)) {
      paths_out[[j]] <- list(
        predictor = paths_list[[j]][[1]], outcome = paths_list[[j]][[2]],
        beta = safe_num(betas[[j]])
      )
    }

    boot_res <- boot_ci(data_clean, constructs_list, construct_types, adjacency,
                        paths_list, n_boot, seed)
    for (j in seq_along(paths_out)) {
      br <- boot_res[[j]]
      paths_out[[j]]$t_stat    <- br$t_stat
      paths_out[[j]]$p_value   <- br$p_val
      paths_out[[j]]$ci_lo     <- br$ci_lo
      paths_out[[j]]$ci_hi     <- br$ci_hi
      paths_out[[j]]$supported <- !is.na(br$p_val) && br$p_val < 0.05
      paths_out[[j]]$decision  <- if (!is.na(br$p_val) && br$p_val < 0.05) "Supported" else "Not Supported"
    }

    rel_list <- compute_reliability(constructs_list, nip$X_list, nip$weights, nip$scores)
    htmt_mat <- tryCatch(compute_htmt(constructs_list, nip$X_list), error = function(e) NULL)
    fl_mat   <- tryCatch(compute_fl(nip$scores, rel_list), error = function(e) NULL)
    vif_vals <- tryCatch(compute_vif(nip$scores, paths_list), error = function(e) NULL)
    r2_list  <- compute_r2(nip$scores, paths_list)
    srmr_val <- tryCatch(compute_srmr(data_clean, constructs_list, nip$scores), error = function(e) NA_real_)
    q2_vals  <- tryCatch(compute_q2(data_clean, constructs_list, construct_types, adjacency, paths_list),
                         error = function(e) list())

    return(list(
      n = n_obs, n_boot = n_boot, loadings = loadings_list, paths = paths_out,
      r2 = r2_list, reliability = rel_list, htmt = htmt_mat, fl_criterion = fl_mat,
      vif = vif_vals, fit = list(srmr = srmr_val), q2 = q2_vals,
      construct_types = construct_types,
      algorithm = "NIPALS (Wold, 1982; Lohmoller, 1989)",
      converged = nip$converged, iterations = nip$iterations,
      method = "PLS-SEM (NIPALS algorithm, centroid inner weighting)"
    ))
  }, error = function(e) list(error = conditionMessage(e)))
}
