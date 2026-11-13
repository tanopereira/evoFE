#' Convert Probabilities to Logit Margins
#'
#' Clamps probabilities to \code{[eps, 1 - eps]} (binary) or \code{[eps, 1]}
#' (multiclass, returning zero-sum centered log-probabilities).
#'
#' @keywords internal
#' @noRd
.prob_to_logit <- function(p, eps = 1e-6) {
  if (is.matrix(p)) {
    p[!is.finite(p)] <- eps
    p <- pmax(p, eps)
    lp <- log(p)
    out <- lp - rowMeans(lp)
    colnames(out) <- colnames(p)
    return(out)
  }
  p[!is.finite(p)] <- 0.5
  p <- pmin(pmax(p, eps), 1 - eps)
  stats::qlogis(p)
}

#' Convert Logit Margins Back to Probabilities
#'
#' @keywords internal
#' @noRd
.logit_to_prob <- function(z) {
  if (is.matrix(z)) {
    max_r <- z[cbind(seq_len(nrow(z)), max.col(z, ties.method = "first"))]
    ez <- exp(z - max_r)
    out <- ez / rowSums(ez)
    colnames(out) <- colnames(z)
    return(out)
  }
  stats::plogis(z)
}

#' Resolve Ensemble Blending Space
#'
#' Resolves \code{"prob"}, \code{"logit"}, or \code{"auto"} into a concrete
#' blending space (\code{"prob"} or \code{"logit"}).
#'
#' @keywords internal
#' @noRd
.resolve_stack_space <- function(stack_space = c("prob", "logit", "auto"),
                                 task = "classification",
                                 metric = NULL,
                                 rank_average = FALSE) {
  if (is.null(stack_space)) stack_space <- "prob"
  stack_space <- match.arg(stack_space, c("prob", "logit", "auto"))
  if (!identical(stack_space, "auto")) {
    return(stack_space)
  }
  if (task %in% c("classification", "multiclass") && !isTRUE(rank_average)) {
    "logit"
  } else {
    "prob"
  }
}

#' Blend Ensemble Candidate Predictions in Probability or Logit Space
#'
#' @keywords internal
#' @noRd
.blend_ensemble_preds <- function(preds_list, weights, stack_space = "prob") {
  is_mat <- is.matrix(preds_list[[1L]])
  use_logit <- identical(stack_space, "logit")
  w_names <- names(weights)
  p_names <- names(preds_list)
  out <- NULL

  for (j in seq_along(preds_list)) {
    nm <- if (!is.null(p_names)) p_names[j] else NULL
    wj <- if (!is.null(nm) && nzchar(nm) && !is.null(w_names) && nm %in% w_names) {
      weights[[nm]]
    } else {
      weights[j]
    }
    if (!is.finite(wj) || wj <= 0) next
    pj <- preds_list[[j]]
    if (use_logit) {
      pj <- .prob_to_logit(pj)
    } else {
      pj[!is.finite(pj)] <- 0
    }
    out <- if (is.null(out)) wj * pj else out + wj * pj
  }

  if (!is_mat) {
    if (is.null(out)) {
      return(numeric(length(preds_list[[1L]])))
    }
    return(if (use_logit) .logit_to_prob(out) else as.vector(out))
  }
  if (is.null(out)) return(NULL)
  if (use_logit) {
    return(.logit_to_prob(out))
  }
  rs <- rowSums(out)
  pos <- is.finite(rs) & rs > 0
  if (any(pos)) {
    out[pos, ] <- out[pos, , drop = FALSE] / rs[pos]
  }
  out
}

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
                          alpha = 0.5, seed = NULL, rank_average = FALSE,
                          stack_space = "prob", verbose = FALSE) {
  n_candidates <- length(val_preds_list)
  candidate_names <- names(val_preds_list)
  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])

  if (isTRUE(rank_average)) {
    val_preds_list <- lapply(val_preds_list, rank_transform_predictions)
  }

  resolved_space <- .resolve_stack_space(stack_space, task, metric, rank_average)

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
  preds_for_fit <- if (identical(resolved_space, "logit") && task %in% c("classification", "multiclass")) {
    lapply(val_preds_list, .prob_to_logit)
  } else {
    val_preds_list
  }
  blocks <- lapply(preds_for_fit, function(p) {
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
    .blend_ensemble_preds(val_preds_list, w, stack_space = resolved_space)
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
          d_args <- list(
            x = X_fit[rows, , drop = FALSE],
            y = y_tr,
            family = fam,
            alpha = alpha,
            standardize = FALSE,
            lambda = 0.01
          )
          if (fam != "multinomial") d_args$lower.limits <- 0
          do.call(glmnet::glmnet, d_args)
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
        fb_args <- list(
          x = X_fit[rows, , drop = FALSE],
          y = y_tr,
          family = fam,
          alpha = alpha,
          standardize = FALSE,
          lambda = 0.01
        )
        if (fam != "multinomial") fb_args$lower.limits <- 0
        do.call(glmnet::glmnet, fb_args)
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

#' Build Feature Matrix for Residual Blender
#'
#' Extracts the union of active recipe features on \code{data}, sanitizes and
#' median-imputes non-finite entries, and optionally expands numeric features
#' into B-spline bases when \code{mode = "spline_lm"}.
#'
#' @keywords internal
#' @noRd
.build_residual_design <- function(active_recipes, data, target_col = NULL,
                                   active_cat_schemas = NULL,
                                   mode = c("lm", "spline_lm"),
                                   fit_spec = NULL) {
  mode <- match.arg(mode)
  if (is.null(active_recipes) || length(active_recipes) == 0L) return(NULL)

  dt_base <- shallow_copy_dt(data.table::as.data.table(data))
  out_dt <- shallow_copy_dt(dt_base)
  all_features <- character(0)

  for (name in names(active_recipes)) {
    ind_i <- active_recipes[[name]]
    dt_i <- shallow_copy_dt(dt_base)
    res_i <- apply_individual(
      ind_i, dt_i, val_data = NULL, target_col = NULL,
      allow_prune = TRUE, state_cache = NULL, full_data = NULL
    )
    sub_dt <- res_i$train
    new_cols <- setdiff(names(sub_dt), names(out_dt))
    if (length(new_cols) > 0L) {
      for (col in new_cols) {
        out_dt[[col]] <- sub_dt[[col]]
      }
    }
    feats_i <- extract_individual_features(res_i$ind, target_col = target_col, unique = TRUE)
    all_features <- union(all_features, feats_i)
  }

  feature_names <- if (is.null(fit_spec)) {
    setdiff(all_features, target_col)
  } else {
    fit_spec$feature_names
  }
  if (length(feature_names) == 0L) return(NULL)

  missing_feats <- setdiff(feature_names, names(out_dt))
  if (length(missing_feats) > 0L) {
    for (mf in missing_feats) {
      out_dt[, (mf) := 0.0]
    }
  }

  feat_dt <- out_dt[, feature_names, with = FALSE]
  cs <- NULL
  if (!is.null(active_cat_schemas) && length(active_cat_schemas) > 0L) {
    for (item in active_cat_schemas) {
      if (!is.null(item)) {
        cs <- item
        break
      }
    }
  }
  x_mat <- .sanitize_feature_matrix(feat_dt, cat_schema = cs)
  if (is.null(x_mat) || ncol(x_mat) == 0L) return(NULL)

  col_means <- NULL
  spline_state <- NULL
  if (identical(mode, "spline_lm") &&
      requireNamespace("splines", quietly = TRUE) &&
      requireNamespace("Matrix", quietly = TRUE)) {
    if (is.null(fit_spec)) {
      sp <- .build_sparse_spline_design(
        x_mat,
        cat_schema = cs,
        n_knots = 3L,
        degree = 3L,
        include_linear = TRUE,
        max_cat_levels = 32L
      )
      x_mat <- sp$matrix
      spline_state <- sp$state
    } else if (!is.null(fit_spec$spline_state)) {
      x_mat <- .build_sparse_spline_design(
        x_mat,
        cat_schema = cs,
        state = fit_spec$spline_state
      )
      spline_state <- fit_spec$spline_state
    }
  } else {
    imp <- .impute_and_col_stats(
      x_mat,
      means = if (!is.null(fit_spec)) fit_spec$col_means else NULL,
      compute_sd = FALSE
    )
    x_mat <- imp$mat
    col_means <- imp$col_means
    if (requireNamespace("Matrix", quietly = TRUE)) {
      x_mat <- Matrix::Matrix(x_mat, sparse = TRUE)
    }
  }

  list(
    x = x_mat,
    spec = list(
      feature_names = feature_names,
      col_means = col_means,
      spline_state = spline_state,
      mode = mode
    )
  )
}

#' Fit Regularized Residual Blender on Top of Base Ensemble OOF Margin
#'
#' Fits a damped ridge (\code{alpha = 0}) \code{glmnet} model with
#' \code{offset = base_margin} over the union of active recipe features and
#' verifies via honest out-of-fold predictions that it improves over the base
#' ensemble. Returns \code{NULL} if no improvement is observed.
#'
#' @keywords internal
#' @noRd
.fit_residual_blender <- function(y_true, val_preds_list, weights,
                                  active_recipes, data, target_col,
                                  task, metric, num_class = NULL, classes = NULL,
                                  stack_space = "prob",
                                  residual_blender = c("none", "lm", "spline_lm"),
                                  fold_partition = NULL,
                                  active_cat_schemas = NULL,
                                  seed = NULL, threads = NULL, verbose = FALSE) {
  residual_blender <- match.arg(residual_blender)
  if (identical(residual_blender, "none")) return(NULL)
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    warning("Package 'glmnet' is required for residual_blender; skipping.")
    return(NULL)
  }

  base_oof <- .blend_ensemble_preds(val_preds_list, weights, stack_space = stack_space)
  if (is.null(base_oof)) return(NULL)

  fam <- switch(task,
    regression = "gaussian",
    classification = "binomial",
    multiclass = "multinomial",
    return(NULL)
  )

  if (task == "regression") {
    y_fit <- as.numeric(y_true)
    base_margin <- as.numeric(base_oof)
  } else {
    lv <- if (!is.null(classes)) classes else sort(unique(as.character(y_true)))
    y_lab <- if (is.numeric(y_true) && length(lv) > 0L &&
                 all(stats::na.omit(y_true) %in% (seq_along(lv) - 1))) {
      lv[as.integer(y_true) + 1]
    } else {
      as.character(y_true)
    }
    y_fit <- factor(y_lab, levels = lv)
    if (task == "classification") {
      base_margin <- .prob_to_logit(as.numeric(base_oof))
    } else {
      base_margin <- .prob_to_logit(base_oof)
      colnames(base_margin) <- lv
    }
  }

  design <- .build_residual_design(
    active_recipes = active_recipes,
    data = data,
    target_col = target_col,
    active_cat_schemas = active_cat_schemas,
    mode = residual_blender
  )
  if (is.null(design) || ncol(design$x) == 0L || nrow(design$x) != length(y_fit)) {
    return(NULL)
  }

  n_obs <- length(y_fit)
  foldid <- if (!is.null(fold_partition) && length(fold_partition) == n_obs &&
                length(unique(fold_partition)) >= 2L) {
    as.integer(factor(fold_partition))
  } else {
    with_seed(if (!is.null(seed)) seed + 17919L else NULL, function() {
      sample(rep(seq_len(min(5L, max(2L, n_obs %/% 4L))), length.out = n_obs))
    })
  }

  fit <- tryCatch({
    glmnet::cv.glmnet(
      x = design$x,
      y = y_fit,
      family = fam,
      alpha = 0,
      offset = base_margin,
      foldid = foldid,
      keep = TRUE,
      standardize = TRUE
    )
  }, error = function(e) NULL)
  if (is.null(fit) || is.null(fit$fit.preval)) return(NULL)

  eval_fn <- function(y, p) {
    if (is.null(p) || length(p) == 0L) return(-Inf)
    mask <- if (is.matrix(p)) !is.na(p[, 1L]) & !is.na(y) else !is.na(p) & !is.na(y)
    if (!any(mask)) return(-Inf)
    res <- tryCatch(
      if (is.matrix(p)) compute_metric(y[mask], p[mask, , drop = FALSE], task = task, metric = metric, num_class = num_class)
      else compute_metric(y[mask], p[mask], task = task, metric = metric),
      error = function(e) -Inf
    )
    if (is.na(res) || is.nan(res)) -Inf else res
  }

  idx_min <- which.min(abs(fit$lambda - fit$lambda.min))
  oof_pred <- if (fam == "gaussian") {
    as.numeric(fit$fit.preval[, idx_min])
  } else if (fam == "binomial") {
    stats::plogis(as.numeric(fit$fit.preval[, idx_min]))
  } else {
    .logit_to_prob(fit$fit.preval[, , idx_min])
  }

  base_score <- eval_fn(y_true, base_oof)
  blend_score <- eval_fn(y_true, oof_pred)
  if (!is.finite(blend_score) || blend_score <= base_score + 1e-6) {
    if (verbose) {
      message(sprintf("  Residual blender (%s) did not improve OOF (%.4f <= %.4f); keeping base ensemble.",
                      residual_blender, blend_score, base_score))
    }
    return(NULL)
  }

  if (verbose) {
    message(sprintf("  Residual blender (%s) improved OOF: %.4f -> %.4f",
                    residual_blender, base_score, blend_score))
  }

  structure(
    list(
      fit = fit,
      spec = design$spec,
      active_recipes = active_recipes,
      active_cat_schemas = active_cat_schemas,
      target_col = target_col,
      task = task,
      classes = if (task == "multiclass") levels(y_fit) else classes,
      stack_space = stack_space,
      mode = residual_blender,
      base_score = base_score,
      blender_score = blend_score
    ),
    class = "evo_residual_blender"
  )
}

#' Predict with Fitted Residual Blender Using Base Ensemble Predictions as Offset
#'
#' @keywords internal
#' @noRd
.predict_residual_blender <- function(blender, base_preds, newdata) {
  if (is.null(blender) || is.null(blender$fit)) return(base_preds)

  design <- .build_residual_design(
    active_recipes = blender$active_recipes,
    data = newdata,
    target_col = blender$target_col,
    active_cat_schemas = blender$active_cat_schemas,
    mode = blender$mode,
    fit_spec = blender$spec
  )
  if (is.null(design)) return(base_preds)

  newoffset <- if (blender$task == "regression") {
    as.numeric(base_preds)
  } else if (blender$task == "classification") {
    .prob_to_logit(as.numeric(base_preds))
  } else {
    m <- .prob_to_logit(base_preds)
    if (!is.null(blender$classes)) colnames(m) <- blender$classes
    m
  }

  pred <- tryCatch({
    stats::predict(
      blender$fit,
      newx = design$x,
      s = "lambda.min",
      newoffset = newoffset,
      type = "response"
    )
  }, error = function(e) NULL)
  if (is.null(pred)) return(base_preds)

  if (blender$task == "multiclass") {
    p_mat <- pred[, , 1L, drop = TRUE]
    if (!is.matrix(p_mat)) {
      p_mat <- matrix(p_mat, nrow = nrow(design$x), ncol = length(blender$classes))
    }
    return(format_multiclass_predictions(p_mat, blender$classes))
  }
  as.numeric(pred)
}
