#' Internal Non-Negative Elastic-Net Stacking Engine
#'
#' Fits a non-negative elastic-net meta-learner over out-of-fold island predictions
#' and produces an honest nested cross-validated estimate of the deployed stacking
#' procedure (scalar normalized island weights + weighted prediction averaging).
#'
#' @keywords internal
#' @noRd
.stack_select <- function(y_true, val_preds_list, task, metric,
                          num_class = NULL, classes = NULL,
                          stack_folds = 5L, fold_partition = NULL,
                          alpha = 0.5, seed = NULL, verbose = FALSE) {
  n_candidates <- length(val_preds_list)
  candidate_names <- names(val_preds_list)
  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])

  fam <- switch(task,
    regression = "gaussian",
    classification = "binomial",
    multiclass = "multinomial",
    stop(sprintf("Unsupported task '%s' for stacking.", task))
  )

  # Local seed wrapper preserving the user's global RNG state
  run_with_seed <- function(seed_val, code) {
    with_seed(seed_val, code)
  }

  # Metric helper (higher is better); NA predictions are masked out
  eval_fitness <- function(y, p) {
    if (is.null(p) || length(p) == 0) return(-Inf)
    if (is.matrix(p)) {
      mask <- !is.na(p[, 1]) & !is.na(y)
      if (!any(mask)) return(-Inf)
      res <- tryCatch(
        compute_metric(y[mask], p[mask, , drop = FALSE], task = task, metric = metric, num_class = num_class),
        error = function(e) -Inf
      )
      if (is.na(res) || is.nan(res)) -Inf else res
    } else {
      mask <- !is.na(p) & !is.na(y)
      if (!any(mask)) return(-Inf)
      res <- tryCatch(
        compute_metric(y[mask], p[mask], task = task, metric = metric),
        error = function(e) -Inf
      )
      if (is.na(res) || is.nan(res)) -Inf else res
    }
  }

  # Level-2 design matrix: one block of columns per candidate island
  blocks <- lapply(val_preds_list, function(p) {
    if (is.matrix(p)) p else matrix(p, ncol = 1)
  })
  block_ncols <- vapply(blocks, ncol, integer(1))
  col_end <- cumsum(block_ncols)
  col_start <- col_end - block_ncols + 1L

  X_fit <- do.call(cbind, blocks)
  X_fit[!is.finite(X_fit)] <- 0

  if (task == "regression") {
    y_fit <- as.numeric(y_true)
  } else {
    lv <- if (!is.null(classes)) classes else sort(unique(as.character(y_true)))
    # Multiclass/classification targets may arrive integer-encoded; map back to labels
    y_lab <- if (is.numeric(y_true) && length(lv) > 0 &&
                 all(stats::na.omit(y_true) %in% (seq_along(lv) - 1))) {
      lv[as.integer(y_true) + 1]
    } else {
      as.character(y_true)
    }
    y_fit <- factor(y_lab, levels = lv)
  }

  # Nesting folds: reuse the evolution partition when provided, else internal balanced folds
  folds <- if (!is.null(fold_partition) && length(fold_partition) == n_obs && anyDuplicated(fold_partition) > 0 &&
               length(unique(fold_partition)) >= 2L) {
    as.integer(factor(fold_partition))
  } else {
    run_with_seed(if (!is.null(seed)) seed + 7919L else NULL, function() {
      sample(rep(seq_len(stack_folds), length.out = n_obs))
    })
  }
  nest_folds <- sort(unique(folds))
  k_fmt <- nchar(as.character(length(nest_folds)))

  aggregate_weights <- function(fit) {
    if (is.null(fit)) return(NULL)
    raw_w <- if (fam == "multinomial") {
      cls_coefs <- tryCatch(glmnet::coef.glmnet(fit, s = "lambda.min"), error = function(e) NULL)
      if (is.null(cls_coefs)) return(NULL)
      mat <- do.call(cbind, lapply(cls_coefs, function(m) as.numeric(m)[-1]))
      mat <- pmax(mat, 0)
      vapply(seq_len(n_candidates), function(j) {
        sum(mat[col_start[j]:col_end[j], , drop = FALSE]) / ncol(mat)
      }, numeric(1))
    } else {
      cf <- tryCatch(as.numeric(glmnet::coef.glmnet(fit, s = "lambda.min"))[-1], error = function(e) NULL)
      if (is.null(cf)) return(NULL)
      cf <- pmax(cf, 0)
      vapply(seq_len(n_candidates), function(j) sum(cf[col_start[j]:col_end[j]]), numeric(1))
    }
    total <- sum(raw_w)
    if (!is.finite(total) || total <= 0) {
      return(NULL)
    }
    stats::setNames(raw_w / total, candidate_names)
  }

  blend_with <- function(w) {
    if (!is.matrix(val_preds_list[[1]])) {
      as.vector(X_fit %*% w)
    } else {
      out <- NULL
      for (j in seq_len(n_candidates)) {
        if (w[j] <= 0) next
        p <- val_preds_list[[j]]
        p[!is.finite(p)] <- 0
        out <- if (is.null(out)) w[j] * p else out + w[j] * p
      }
      if (!is.null(out)) {
        rs <- rowSums(out)
        pos <- rs > 0
        if (any(pos)) {
          out[pos, ] <- out[pos, , drop = FALSE] / rs[pos]
        }
      }
      out
    }
  }

  fallback_weights <- function(rows = seq_len(n_obs)) {
    y_sub <- if (is.matrix(y_true)) y_true[rows, , drop = FALSE] else y_true[rows]
    solo <- vapply(candidate_names, function(nm) {
      p_sub <- if (is.matrix(val_preds_list[[nm]])) val_preds_list[[nm]][rows, , drop = FALSE] else val_preds_list[[nm]][rows]
      eval_fitness(y_sub, p_sub)
    }, double(1))
    best <- names(which.max(solo))[1]
    if (is.na(best) || length(best) == 0) {
      return(stats::setNames(rep(1 / n_candidates, n_candidates), candidate_names))
    }
    stats::setNames(ifelse(candidate_names == best, 1, 0), candidate_names)
  }

  # Class-balanced (or random, for regression) internal CV fold ids for cv.glmnet.
  sample_counter <- 0L
  seeded_sample <- function(...) {
    sample_counter <<- sample_counter + 1L
    run_with_seed(if (!is.null(seed)) seed + 1000003L * sample_counter else NULL, function() {
      sample(...)
    })
  }

  make_foldid <- function(y_sub, k) {
    k <- max(2L, min(k, length(y_sub)))
    if (fam == "gaussian") {
      return(seeded_sample(rep(seq_len(k), length.out = length(y_sub))))
    }
    fid <- integer(length(y_sub))
    for (lv in levels(droplevels(factor(y_sub)))) {
      ids <- which(as.character(y_sub) == lv)
      if (length(ids) == 0) next
      fid[ids] <- seeded_sample(rep(seq_len(min(k, length(ids))), length.out = length(ids)))
    }
    un <- which(fid == 0)
    if (length(un) > 0) {
      fid[un] <- seeded_sample(rep(seq_len(k), length.out = length(un)))
    }
    if (length(unique(fid[fid > 0])) < min(2L, length(y_sub))) {
      fid <- seeded_sample(rep(seq_len(k), length.out = length(y_sub)))
    }
    fid
  }

  fit_net <- function(rows) {
    y_tr <- y_fit[rows]
    if (fam != "gaussian") {
      tab <- table(y_tr)
      if (length(tab[tab > 0]) < 2L || any(tab[tab > 0] < 2L)) {
        fit_direct <- tryCatch({
          glmnet::glmnet(
            x = X_fit[rows, , drop = FALSE],
            y = y_tr,
            family = fam,
            alpha = alpha,
            standardize = FALSE,
            lambda = 0.01
          )
        }, error = function(e) NULL)
        return(fit_direct)
      }
      nf <- max(3L, min(10L, length(rows) %/% 4L))
      nf <- min(nf, min(tab[tab > 0]))
      nf <- max(2L, min(nf, length(rows)))
    } else {
      nf <- max(3L, min(10L, length(rows) %/% 4L))
      nf <- max(2L, min(nf, length(rows)))
    }
    args <- list(
      x = X_fit[rows, , drop = FALSE],
      y = y_tr,
      family = fam,
      alpha = alpha,
      standardize = FALSE,
      foldid = make_foldid(y_tr, nf)
    )
    if (fam != "multinomial") args$lower.limits <- 0
    tryCatch({
      do.call(glmnet::cv.glmnet, args)
    }, error = function(e) {
      tryCatch({
        glmnet::glmnet(
          x = X_fit[rows, , drop = FALSE],
          y = y_tr,
          family = fam,
          alpha = alpha,
          standardize = FALSE,
          lambda = 0.01
        )
      }, error = function(e2) NULL)
    })
  }

  if (verbose) {
    message(sprintf("  Stacking %d island models with non-negative elastic net (alpha = %.2f), nested over %d folds...",
                    n_candidates, alpha, length(nest_folds)))
  }

  # Honest nested evaluation of the deployed procedure (scalar weights + weighted blending)
  pooled_preds <- NULL
  pooled_y <- NULL
  fold_fitnesses <- numeric(length(nest_folds))

  for (fi in seq_along(nest_folds)) {
    f <- nest_folds[fi]
    te <- which(folds == f)
    tr <- which(folds != f)

    w_f <- local({
      fit <- fit_net(tr)
      w <- aggregate_weights(fit)
      if (is.null(w)) fallback_weights(tr) else w
    })

    is_mat <- is.matrix(val_preds_list[[1]])
    blended <- blend_with(w_f)
    preds_te <- if (is_mat) blended[te, , drop = FALSE] else blended[te]
    fold_fitnesses[fi] <- eval_fitness(y_true[te], preds_te)

    if (is.null(pooled_preds)) {
      pooled_preds <- if (is_mat) {
        matrix(NA_real_, nrow = n_obs, ncol = num_class)
      } else {
        rep(NA_real_, n_obs)
      }
      pooled_y <- y_true
    }
    if (is_mat) {
      pooled_preds[te, ] <- preds_te
    } else {
      pooled_preds[te] <- preds_te
    }

    if (verbose) {
      message(sprintf("  [Nest Fold %*d/%*d] Fitness: %.4f", k_fmt, fi, k_fmt, length(nest_folds), fold_fitnesses[fi]))
    }
  }

  stack_cv_fitness <- eval_fitness(pooled_y, pooled_preds)

  # Final weights on all level-2 rows
  final_w <- local({
    fit <- fit_net(seq_len(n_obs))
    w <- aggregate_weights(fit)
    if (is.null(w)) fallback_weights(seq_len(n_obs)) else w
  })
  final_fitness <- eval_fitness(y_true, blend_with(final_w))

  if (verbose) {
    message(sprintf("  Stack CV Fitness (honest): %.4f  |  Full-Fit Ensemble Fitness: %.4f", stack_cv_fitness, final_fitness))
  }

  list(
    weights = final_w,
    history = NULL,
    final_fitness = final_fitness,
    stack_cv_fitness = stack_cv_fitness
  )
}
