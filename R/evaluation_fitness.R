#' Evaluate the fitness of an individual
#'
#' Trains a model using the features specified by the individual's recipe and evaluates
#' performance using cross-validation or train/val split.
#'
#' @param ind An evo_individual object.
#' @param data A data.frame or data.table containing the dataset.
#' @param target_col Name of the target column.
#' @param task "classification" or "regression".
#' @param cv_folds Number of cross-validation folds.
#' @param evaluation_strategy Character string, either "cv" (cross-validation) or "split" (train/validation split).
#' @param split_ids Optional vector of pre-defined split assignments (e.g. \code{c("train", "train", "val", "holdout", "train")}). Must have the same length as the number of rows in \code{data} and contain only "train", "val", or "holdout" labels.
#' @param shared_splits Optional list of shared data.table splits for in-place caching.
#' @param evaluator Character string specifying the model backend: "lightgbm", "xgboost", "catboost", "rf", or "lm".
#' @param fold_ids Optional integer vector of pre-assigned fold indices.
#' @param shared_folds Optional list of shared data.table fold splits for in-place caching.
#' @param shared_full Optional shared full data.table.
#' @param state_cache Optional environment used to cache transformer training states across evaluations.
#' @param threads Number of threads for model training.
#' @param metric Character string or evaluation metric.
#' @param verbose Logical.
#' @param allow_prune Logical.
#' @param complexity_penalty Numeric. Dimensionless penalty multiplier (default 0).
#' @param complexity_mode Character. Complexity penalty strategy: "bic_dynamic" (default),
#'   "bic", "pac_bayes_dynamic", "pac_bayes", or "none".
#' @param complexity_floor Numeric in \code{[0, 1]}. Minimum safety floor factor for dynamic penalty (default \code{0.20}).
#' @param complexity_target Character. "all_features" (default, penalizes total active features) or "genes" (penalizes only derived genes).
#' @param running_best_fitness Optional numeric. Current running best fitness for dynamic BIC.
#' @param baseline_fitness Optional numeric. Generation 0 baseline fitness for dynamic BIC.
#' @param n_samples Optional integer. Dataset sample size N for BIC calculations. Defaults to nrow(data).
#' @param cv_strategy Fold construction strategy for CV: \code{"random"} (default), \code{"time"}, or \code{"group"}.
#' @param time_col Column name used when \code{cv_strategy = "time"}.
#' @param group_col Column name used when \code{cv_strategy = "group"}.
#' @param global_unsupervised Logical. If TRUE (default), unsupervised stateful transformers (e.g. UMAP, PCA, Lumbermark, Genie) are fit globally on the full input feature matrix X to ensure invariant cluster IDs and manifold coordinates across CV folds with zero target leakage, while supervised transformers remain strictly per-fold. Set to FALSE for strict per-fold unsupervised fitting.
#' @param ... Additional arguments passed to the underlying evaluator training functions.
#' @return The input \code{evo_individual} with its \code{fitness} field set to
#'   the computed score (higher is better), \code{importances} set to a named
#'   numeric vector of feature importances, \code{holdout_fitness} set to
#'   \code{NULL}, and \code{genes} updated with fitted transformer states.
#' @export
evaluate_fitness <- function(ind, data, target_col, task = "classification",
                             cv_folds = 3, evaluation_strategy = "cv",
                             split_ids = NULL, shared_splits = NULL,
                             evaluator = "lightgbm", fold_ids = NULL,
                             shared_folds = NULL, shared_full = NULL,
                             state_cache = NULL, threads = 2,
                             metric = "default", verbose = FALSE, allow_prune = TRUE,
                             complexity_penalty = 0, complexity_mode = "bic_dynamic",
                             complexity_floor = 0.20, complexity_target = "all_features",
                             running_best_fitness = NULL, baseline_fitness = NULL,
                             n_samples = NULL, cv_strategy = "random",
                             time_col = NULL, group_col = NULL,
                             global_unsupervised = getOption("evoFE.global_unsupervised", TRUE), ...) {
  if (!is.na(ind$fitness)) {
    return(ind)
  }

  num_class <- NULL
  classes <- NULL
  if (task == "multiclass") {
    target_factor <- as.factor(data[[target_col]])
    classes <- levels(target_factor)
    num_class <- length(classes)
  }

  if (evaluation_strategy %in% c("split", "metacv", "meta_cv")) {
    # Train / Validation split strategy
    if (!is.null(shared_splits)) {
      train_fold <- shared_splits$train
      val_fold <- shared_splits$val
    } else {
      train_fold <- data.table::as.data.table(data[split_ids == "train", ])
      val_fold <- data.table::as.data.table(data[split_ids == "val", ])
    }

    # Copy train/val folds so we can modify them when applying recipe
    train_fold <- data.table::copy(train_fold)
    val_fold <- data.table::copy(val_fold)
    full_split_data <- if (!is.null(shared_full)) shared_full else data

    # Apply genes
    res <- tryCatch(
      {
        apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache,
                         allow_prune = allow_prune, full_data = full_split_data,
                         global_unsupervised = global_unsupervised)
      },
      error = function(e) {
        NULL
      }
    )

    if (is.null(res)) {
      # Lethal mutation: invalid gene dependency graph
      ind$raw_fitness <- -Inf
      ind$penalty <- 0.0
      ind$fitness <- -Inf
      ind$holdout_fitness <- NULL
      return(ind)
    }

    train_fold_feat <- res$train
    val_fold_feat <- res$val

    features <- extract_individual_features(res$ind, target_col = target_col)
    dt_sub_tr <- train_fold_feat[, features, with = FALSE]
    dt_sub_va <- val_fold_feat[, features, with = FALSE]
    x_train <- .sanitize_feature_matrix(dt_sub_tr)
    x_val <- .sanitize_feature_matrix(dt_sub_va)
    y_train <- train_fold_feat[[target_col]]
    y_val <- val_fold_feat[[target_col]]
    if (task == "multiclass") {
      y_train <- encode_multiclass_target(y_train, classes)
      y_val <- encode_multiclass_target(y_val, classes)
    } else if (task == "classification") {
      if (is.factor(y_train)) {
        y_train <- as.integer(y_train) - 1L
        y_val <- as.integer(y_val) - 1L
      } else if (is.character(y_train)) {
        y_fac <- as.factor(y_train)
        y_train <- as.integer(y_fac) - 1L
        y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
      } else if (is.logical(y_train)) {
        y_train <- as.integer(y_train)
        y_val <- as.integer(y_val)
      } else if (is.numeric(y_train) && !all(stats::na.omit(y_train) %in% c(0, 1))) {
        y_fac <- as.factor(y_train)
        y_train <- as.integer(y_fac) - 1L
        y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
      }
    }

    res_model <- train_model(x_train, y_train, x_val,
      y_val = y_val, task = task,
      evaluator = evaluator, threads = threads,
      num_class = num_class, metric = metric,
      verbose = verbose, ...
    )
    preds <- res_model$predictions

    # Store importances (single fold)
    if (!is.null(res_model$importances) && length(res_model$importances) > 0) {
      imp <- res_model$importances
      imp_sum <- sum(imp, na.rm = TRUE)
      if (imp_sum > 0) {
        imp <- imp / imp_sum
        threshold <- getOption("evoFE.importance_threshold", 0.001)
        imp[imp < threshold] <- 0
      }
      ind$importances <- imp
    } else {
      ind$importances <- numeric(0)
    }

    if (!is.null(res_model$best_params)) {
      ind$best_params <- res_model$best_params
    }

    iter_val <- if (!is.null(res_model$best_iteration)) {
      res_model$best_iteration
    } else if (!is.null(res_model$best_epoch)) {
      res_model$best_epoch
    } else if (!is.null(res_model$model$best_iteration)) {
      res_model$model$best_iteration
    } else if (!is.null(res_model$model$best_epoch)) {
      res_model$model$best_epoch
    } else if (!is.null(res_model$model$best_iter)) {
      res_model$model$best_iter
    } else {
      NULL
    }
    if (!is.null(iter_val) && is.numeric(iter_val) && is.finite(iter_val) && iter_val > 0) {
      ind$best_iteration <- as.integer(round(iter_val))
    }
    ind$train_size <- nrow(train_fold)

    # Validation score -> raw_fitness
    if (task == "multiclass") {
      y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
      raw_score <- compute_metric(y_val_encoded, preds, task, metric, num_class)
      ind$val_preds <- preds
      ind$y_val <- y_val_encoded
    } else if (task == "classification") {
      y_val_encoded <- y_val
      raw_score <- compute_metric(y_val_encoded, preds, task, metric)
      ind$val_preds <- preds
      ind$y_val <- y_val_encoded
    } else {
      raw_score <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
      ind$val_preds <- preds
      ind$y_val <- val_fold_feat[[target_col]]
    }

    ind$raw_fitness <- raw_score
    ind$penalty <- 0.0

    # Complexity penalty: discourage long recipes (parsimony pressure)
    if (complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
      n_samp <- if (!is.null(n_samples)) n_samples else if (!is.null(data)) nrow(data) else if (!is.null(shared_full)) nrow(shared_full) else 100
      n_active_raw <- length(res$ind$numeric_cols) + length(res$ind$categorical_cols) + length(res$ind$datetime_cols)
      n_genes <- length(res$ind$genes)
      pen <- compute_complexity_penalty(
        n_genes = n_genes,
        n_samples = n_samp,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        metric = metric,
        task = task,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        n_features = n_active_raw + n_genes
      )
      ind$penalty <- pen
      if (task == "regression") {
        ind$fitness <- if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen)
      } else {
        error_gap <- max(0, min(1.0, 1.0 - raw_score))
        ind$fitness <- raw_score - error_gap * pen
      }
    } else {
      ind$fitness <- raw_score
    }

    ind$holdout_fitness <- NULL

    # Propagate the updated genes (with state)
    ind$genes <- res$ind$genes
  } else {
    # --- Existing CV strategy ---
    use_shared <- !is.null(shared_folds) && !is.null(shared_full)

    if (use_shared) {
      dt <- shared_full
      folds <- if (!is.null(fold_ids)) fold_ids else rep(seq_along(shared_folds), length.out = nrow(dt))
    } else {
      dt <- data.table::as.data.table(data)
      if (is.null(fold_ids)) {
        folds <- .build_cv_folds(dt, cv_folds, cv_strategy, time_col, group_col)
      } else {
        folds <- fold_ids
      }
    }

    unique_folds <- sort(unique(folds))
    k_folds <- length(unique_folds)
    metrics <- rep(NA_real_, k_folds)
    fold_importances <- list()
    last_fit_genes <- NULL

    n_total <- nrow(dt)
    if (task == "multiclass") {
      oof_preds <- matrix(NA_real_, nrow = n_total, ncol = num_class)
      oof_y <- integer(n_total)
    } else {
      oof_preds <- rep(NA_real_, n_total)
      oof_y <- vector(mode = typeof(dt[[target_col]]), length = n_total)
    }

    fold_best_iters <- integer(0)
    fold_train_sizes <- integer(0)
    is_tuned <- is_tuned_evaluator(evaluator)
    fold_data <- if (is_tuned) vector("list", k_folds) else NULL

    for (fi in seq_along(unique_folds)) {
      f <- unique_folds[fi]
      if (use_shared) {
        train_fold <- data.table::copy(shared_folds[[f]]$train)
        val_fold <- data.table::copy(shared_folds[[f]]$val)
        val_idx <- if (!is.null(fold_ids)) which(fold_ids == f) else seq_len(nrow(val_fold))
      } else {
        train_idx <- which(folds != f)
        val_idx <- which(folds == f)
        train_fold <- dt[train_idx, ]
        val_fold <- dt[val_idx, ]
      }
      fold_train_sizes <- c(fold_train_sizes, nrow(train_fold))

      res <- tryCatch(
        {
          apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache,
                           allow_prune = allow_prune, full_data = dt,
                           global_unsupervised = global_unsupervised)
        },
        error = function(e) {
          NULL
        }
      )

      if (is.null(res)) {
        # This fold failed (e.g. constant column on this split) — skip it.
        # The individual is only killed if every fold fails (handled below).
        next
      }

      train_fold_feat <- res$train
      val_fold_feat <- res$val
      if (!is.null(res$ind)) last_fit_genes <- res$ind$genes

      features <- extract_individual_features(res$ind, target_col = target_col)
      dt_sub_tr <- train_fold_feat[, features, with = FALSE]
      dt_sub_va <- val_fold_feat[, features, with = FALSE]
      x_train <- .sanitize_feature_matrix(dt_sub_tr)
      x_val <- .sanitize_feature_matrix(dt_sub_va)
      y_train <- train_fold_feat[[target_col]]
      y_val <- val_fold_feat[[target_col]]
      if (task == "multiclass") {
        y_train <- encode_multiclass_target(y_train, classes)
        y_val <- encode_multiclass_target(y_val, classes)
      } else if (task == "classification") {
        if (is.factor(y_train)) {
          y_train <- as.integer(y_train) - 1L
          y_val <- as.integer(y_val) - 1L
        } else if (is.character(y_train)) {
          y_fac <- as.factor(y_train)
          y_train <- as.integer(y_fac) - 1L
          y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
        } else if (is.logical(y_train)) {
          y_train <- as.integer(y_train)
          y_val <- as.integer(y_val)
        } else if (is.numeric(y_train) && !all(stats::na.omit(y_train) %in% c(0, 1))) {
          y_fac <- as.factor(y_train)
          y_train <- as.integer(y_fac) - 1L
          y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
        }
      }

      if (is_tuned) {
        fold_data[[fi]] <- list(
          x_train = x_train,
          y_train = y_train,
          x_val = x_val,
          y_val = y_val,
          val_idx = val_idx,
          val_fold_feat = val_fold_feat,
          fold_idx = fi
        )
      } else {
        res_model <- train_model(x_train, y_train, x_val,
          y_val = y_val, task = task,
          evaluator = evaluator, threads = threads,
          num_class = num_class, metric = metric,
          verbose = verbose, ...
        )
        preds <- res_model$predictions
        if (!is.null(res_model$importances)) {
          fold_importances[[fi]] <- res_model$importances
        }
        if (!is.null(res_model$best_params)) {
          ind$best_params <- res_model$best_params
        }

        iter_val <- if (!is.null(res_model$best_iteration)) {
          res_model$best_iteration
        } else if (!is.null(res_model$best_epoch)) {
          res_model$best_epoch
        } else if (!is.null(res_model$model$best_iteration)) {
          res_model$model$best_iteration
        } else if (!is.null(res_model$model$best_epoch)) {
          res_model$model$best_epoch
        } else if (!is.null(res_model$model$best_iter)) {
          res_model$model$best_iter
        } else {
          NULL
        }
        if (!is.null(iter_val) && is.numeric(iter_val) && is.finite(iter_val) && iter_val > 0) {
          fold_best_iters <- c(fold_best_iters, as.integer(round(iter_val)))
        }

        if (task == "multiclass") {
          y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
          metrics[fi] <- compute_metric(y_val_encoded, preds, task, metric, num_class)
          if (length(val_idx) == length(y_val_encoded)) {
            if (is.matrix(preds)) {
              oof_preds[val_idx, ] <- preds
            } else {
              oof_preds[val_idx, ] <- matrix(preds, ncol = num_class, byrow = FALSE)
            }
            oof_y[val_idx] <- y_val_encoded
          }
        } else if (task == "classification") {
          y_val_encoded <- y_val
          metrics[fi] <- compute_metric(y_val_encoded, preds, task, metric)
          if (length(val_idx) == length(preds)) {
            oof_preds[val_idx] <- preds
            oof_y[val_idx] <- y_val_encoded
          }
        } else {
          metrics[fi] <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
          if (length(val_idx) == length(preds)) {
            oof_preds[val_idx] <- preds
            oof_y[val_idx] <- val_fold_feat[[target_col]]
          }
        }

        # Clean up the model if the evaluator provides a cleanup function (e.g. to prevent TF memory leaks)
        eval_entry <- get_evaluator(evaluator)
        if (!is.null(eval_entry$cleanup_func)) {
          eval_entry$cleanup_func(res_model$model)
        }
      }
    }

    if (is_tuned) {
      valid_fold_data <- fold_data[!vapply(fold_data, is.null, logical(1))]
      if (length(valid_fold_data) > 0) {
        res_model <- train_model(
          x_train = NULL, y_train = NULL, x_val = NULL, y_val = NULL,
          task = task, evaluator = evaluator, threads = threads,
          num_class = num_class, metric = metric, verbose = verbose,
          best_params = ind$best_params, fold_data = valid_fold_data, ...
        )

        if (!is.null(res_model$best_params)) {
          ind$best_params <- res_model$best_params
        }

        eval_entry <- get_evaluator(evaluator)
        for (i in seq_along(valid_fold_data)) {
          fd <- valid_fold_data[[i]]
          orig_fi <- fd$fold_idx
          res_m <- res_model$fold_res[[i]]
          preds <- res_m$predictions
          val_idx <- fd$val_idx

          if (!is.null(res_m$importances)) {
            fold_importances[[orig_fi]] <- res_m$importances
          }

          iter_val <- if (!is.null(res_m$best_iteration)) {
            res_m$best_iteration
          } else if (!is.null(res_m$best_epoch)) {
            res_m$best_epoch
          } else if (!is.null(res_m$model$best_iteration)) {
            res_m$model$best_iteration
          } else if (!is.null(res_m$model$best_epoch)) {
            res_m$model$best_epoch
          } else if (!is.null(res_m$model$best_iter)) {
            res_m$model$best_iter
          } else {
            NULL
          }
          if (!is.null(iter_val) && is.numeric(iter_val) && is.finite(iter_val) && iter_val > 0) {
            fold_best_iters <- c(fold_best_iters, as.integer(round(iter_val)))
          }

          if (task == "multiclass") {
            y_val_encoded <- as.integer(factor(fd$val_fold_feat[[target_col]], levels = classes)) - 1
            metrics[orig_fi] <- compute_metric(y_val_encoded, preds, task, metric, num_class)
            if (length(val_idx) == length(y_val_encoded)) {
              if (is.matrix(preds)) {
                oof_preds[val_idx, ] <- preds
              } else {
                oof_preds[val_idx, ] <- matrix(preds, ncol = num_class, byrow = FALSE)
              }
              oof_y[val_idx] <- y_val_encoded
            }
          } else if (task == "classification") {
            y_val_encoded <- fd$y_val
            metrics[orig_fi] <- compute_metric(y_val_encoded, preds, task, metric)
            if (length(val_idx) == length(preds)) {
              oof_preds[val_idx] <- preds
              oof_y[val_idx] <- y_val_encoded
            }
          } else {
            metrics[orig_fi] <- compute_metric(fd$val_fold_feat[[target_col]], preds, task, metric)
            if (length(val_idx) == length(preds)) {
              oof_preds[val_idx] <- preds
              oof_y[val_idx] <- fd$val_fold_feat[[target_col]]
            }
          }

          if (!is.null(eval_entry$cleanup_func)) {
            eval_entry$cleanup_func(res_m$model)
          }
        }
      }
    }

    # Fitness: average over successful folds only; -Inf only when every fold failed.
    finite_metrics <- metrics[!is.na(metrics)]
    raw_score <- if (length(finite_metrics) == 0) -Inf else mean(finite_metrics)

    ind$raw_fitness <- raw_score
    ind$penalty <- 0.0
    ind$val_preds <- oof_preds
    ind$y_val <- oof_y
    if (length(fold_best_iters) > 0) {
      ind$best_iteration <- as.integer(round(mean(fold_best_iters)))
    }
    if (length(fold_train_sizes) > 0) {
      ind$train_size <- as.integer(round(mean(fold_train_sizes)))
    }

    # Complexity penalty: discourage long recipes (parsimony pressure)
    if (complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
      n_samp <- if (!is.null(n_samples)) n_samples else if (!is.null(data)) nrow(data) else if (!is.null(shared_full)) nrow(shared_full) else 100
      n_active_raw <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols)
      n_genes <- length(ind$genes)
      pen <- compute_complexity_penalty(
        n_genes = n_genes,
        n_samples = n_samp,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        metric = metric,
        task = task,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        n_features = n_active_raw + n_genes
      )
      ind$penalty <- pen
      if (task == "regression") {
        ind$fitness <- if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen)
      } else {
        error_gap <- max(0, min(1.0, 1.0 - raw_score))
        ind$fitness <- raw_score - error_gap * pen
      }
    } else {
      ind$fitness <- raw_score
    }

    # Aggregate importances across folds
    all_feats <- unique(unlist(lapply(fold_importances, names)))
    if (length(all_feats) > 0) {
      avg_imp <- sapply(all_feats, function(feat) {
        vals <- sapply(fold_importances, function(fold) {
          if (feat %in% names(fold)) fold[[feat]] else 0
        })
        mean(vals)
      })

      imp_sum <- sum(avg_imp, na.rm = TRUE)
      if (imp_sum > 0) {
        avg_imp <- avg_imp / imp_sum
        threshold <- getOption("evoFE.importance_threshold", 0.001)
        avg_imp[avg_imp < threshold] <- 0
      }
      ind$importances <- avg_imp
    } else {
      ind$importances <- numeric(0)
    }

    # Propagate fitted gene states from the last successful fold
    if (!is.null(last_fit_genes)) ind$genes <- last_fit_genes

    ind$holdout_fitness <- NULL
  }

  ind
}
