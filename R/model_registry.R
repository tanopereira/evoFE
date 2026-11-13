#' Global environment for registered model evaluators
#'
#' @return An \code{environment} containing registered model evaluators.
#' @export
evo_evaluators <- new.env(parent = emptyenv())

# Internal helper to read the best iteration from a trained xgboost model,
# supporting both the legacy API (attribute on the object) and xgboost >= 3.0
# (where early stopping info lives in model$early_stop).
.xgb_best_iter <- function(model) {
  if (!is.null(model$best_iteration)) {
    return(model$best_iteration)
  }
  if (!is.null(model$early_stop) && !is.null(model$early_stop$best_iteration)) {
    bi <- model$early_stop$best_iteration
    if (is.numeric(bi) && is.finite(bi) && bi > 0) return(bi)
    return(NULL)
  }
  val <- tryCatch(xgboost::xgb.attr(model, "best_iteration"), error = function(e) NULL)
  if (!is.null(val)) as.numeric(val) else NULL
}

# Internal helper to unify SHAP values across models
.extract_shap_importances <- function(sh, num_feats, feature_names) {
  if (length(dim(sh)) == 3) {
    # 3D Array: [N, num_classes, num_feats + 1] (CatBoost/XGBoost Multiclass)
    sh_feats <- sh[, , 1:num_feats, drop = FALSE]
    sh_sum <- apply(abs(sh_feats), c(1, 3), sum)
    imp <- colMeans(sh_sum)
  } else if (length(dim(sh)) == 2) {
    if (ncol(sh) == num_feats + 1) {
      # 2D Matrix: [N, num_feats + 1] (Regression/Binary)
      sh_feats <- sh[, 1:num_feats, drop = FALSE]
      imp <- colMeans(abs(sh_feats))
    } else {
      # 2D Matrix: [N, (num_feats + 1) * num_classes] (LightGBM Multiclass)
      num_classes <- ncol(sh) / (num_feats + 1)
      imp <- numeric(num_feats)
      for (c in seq_len(num_classes)) {
        cols <- (c - 1) * (num_feats + 1) + seq_len(num_feats)
        sh_feats <- sh[, cols, drop = FALSE]
        imp <- imp + colMeans(abs(sh_feats))
      }
    }
  } else {
    imp <- numeric(num_feats)
  }
  stats::setNames(as.numeric(imp), feature_names)
}
#' Register a model evaluator
#'
#' @param name Name of the evaluator.
#' @param train_func Function to train the model. Must accept \code{x_train},
#'   \code{y_train}, \code{x_val}, \code{task}, \code{threads}, \code{num_class},
#'   and any additional parameters, and return a list with \code{model},
#'   \code{predictions}, and \code{importances}.
#' @param predict_func Function to make predictions. Must accept \code{model},
#'   \code{x_new}, \code{task}, and any additional parameters, and return a
#'   vector or matrix of predictions.
#' @param base_evaluator Optional character name of the base registered model.
#' @param cleanup_func Optional function to clean up model resources/states after evaluation.
#' @return Invisible NULL. Called for the side effect of registering the evaluator in \code{evo_evaluators}.
#' @importFrom lightgbm lgb.train
#' @importFrom xgboost xgb.train
#' @examples
#' # Register a simple mock evaluator
#' register_evaluator(
#'   "mock_eval",
#'   train_func = function(x_train, y_train, x_val = NULL,
#'                         task = "regression", ...) {
#'     list(
#'       model = list(weights = colMeans(x_train)),
#'       predictions = if (!is.null(x_val)) rowMeans(x_val) else NULL,
#'       importances = stats::setNames(
#'         rep(1, ncol(x_train)), colnames(x_train)
#'       )
#'     )
#'   },
#'   predict_func = function(model, x_new, task, ...) {
#'     rowMeans(x_new)
#'   }
#' )
#'
#' # Verify it is registered
#' exists("mock_eval", envir = evo_evaluators)
#' @param traits Optional named list of evaluator capabilities/traits.
#' @return Invisible entry list. Called for the side effect of registering the evaluator in \code{evo_evaluators}.
#' @export
register_evaluator <- function(name, train_func, predict_func, base_evaluator = NULL, cleanup_func = NULL, traits = list()) {
  entry <- list(
    train_func = train_func,
    predict_func = predict_func,
    base_evaluator = base_evaluator,
    cleanup_func = cleanup_func,
    traits = traits,
    is_tuned = !is.null(base_evaluator) && is.character(base_evaluator) && length(base_evaluator) == 1 && nzchar(base_evaluator) && base_evaluator != name
  )
  assign(name, entry, envir = evo_evaluators)
  invisible(entry)
}

#' Check if an Evaluator is Registered
#'
#' @param name Character name of the evaluator.
#' @return Logical scalar indicating if the evaluator is present in \code{evo_evaluators}.
#' @export
has_evaluator <- function(name) {
  if (missing(name) || is.null(name) || !is.character(name) || length(name) != 1 || is.na(name)) {
    return(FALSE)
  }
  exists(name, envir = evo_evaluators)
}

#' List All Registered Evaluators
#'
#' @return Character vector of names of registered evaluators.
#' @export
list_evaluators <- function() {
  names(evo_evaluators)
}

#' Retrieve a Registered Model Evaluator
#'
#' @param name Character name of the evaluator.
#' @return List containing evaluator entry components (\code{train_func},
#'   \code{predict_func}, \code{base_evaluator}, \code{cleanup_func}, \code{traits}).
#' @export
get_evaluator <- function(name) {
  if (missing(name) || is.null(name) || !is.character(name) || length(name) != 1 || is.na(name) || !exists(name, envir = evo_evaluators)) {
    stop(sprintf("Model '%s' is not registered in evo_evaluators. Registered models: %s",
                 if (missing(name) || is.null(name) || length(name) == 0 || is.na(name)) "" else name,
                 paste(names(evo_evaluators), collapse = ", ")))
  }
  get(name, envir = evo_evaluators)
}

#' Get Base Evaluator Name
#'
#' Resolves tuner wrappers to find their underlying base evaluator name.
#'
#' @param evaluator Character name or evaluator list object.
#' @return Character name of the base evaluator (or the original evaluator name if not tuned).
#' @export
get_base_evaluator <- function(evaluator) {
  if (is.list(evaluator) && !is.null(evaluator$base_evaluator)) {
    return(evaluator$base_evaluator)
  }
  if (is.character(evaluator) && length(evaluator) == 1 && !is.na(evaluator) && has_evaluator(evaluator)) {
    ev <- get_evaluator(evaluator)
    if (!is.null(ev$base_evaluator)) return(ev$base_evaluator)
    return(evaluator)
  }
  evaluator
}

#' Recursively Unwrap Wrapped or Tuned Evaluators
#'
#' Resolves tuning or wrapping layers to find the canonical base evaluator name.
#'
#' @param evaluator Character name of evaluator or tuner.
#' @return Character name of the base evaluator.
#' @export
unwrap_evaluator <- function(evaluator) {
  if (!is.character(evaluator) || length(evaluator) != 1 || is.na(evaluator)) {
    return(evaluator)
  }
  curr <- evaluator
  visited <- character(0)
  while (is.character(curr) && length(curr) == 1 && !is.na(curr) && has_evaluator(curr) && !(curr %in% visited)) {
    visited <- c(visited, curr)
    entry <- get_evaluator(curr)
    if (!is.null(entry$base_evaluator) && is.character(entry$base_evaluator) &&
        length(entry$base_evaluator) == 1 && !is.na(entry$base_evaluator) &&
        entry$base_evaluator != curr) {
      curr <- entry$base_evaluator
    } else {
      break
    }
  }
  curr
}

#' Check if an Evaluator is Tuned / Wrapped
#'
#' @param evaluator Character name or evaluator list object.
#' @return Logical scalar TRUE if evaluator wraps a base evaluator via tuning.
#' @export
is_tuned_evaluator <- function(evaluator) {
  if (is.list(evaluator) && !is.null(evaluator$base_evaluator)) {
    return(TRUE)
  }
  if (is.character(evaluator) && length(evaluator) == 1 && !is.na(evaluator) && has_evaluator(evaluator)) {
    ev <- get_evaluator(evaluator)
    return(!is.null(ev$base_evaluator) && ev$base_evaluator != evaluator)
  }
  FALSE
}

#' Resolve Tuned Evaluator and Merge Pre-Tuned Hyperparameters
#'
#' When an individual has already been tuned (non-empty \code{best_params}),
#' unwraps the tunable evaluator to its base model and merges \code{best_params}
#' into \code{extra_args} so post-evolution re-evaluations (tournaments, OOF
#' harmonization, holdout scoring, and final refits) never re-run Bayesian
#' optimization from scratch.
#'
#' @param evaluator Character name of evaluator or tuner.
#' @param best_params Optional named list of pre-tuned hyperparameters.
#' @param extra_args List of additional arguments to pass to the evaluator.
#' @param always_unwrap Logical. If TRUE, unwraps \code{evaluator} even when
#'   \code{best_params} is empty (used for holdout and final full-data refits).
#' @return A list with \code{evaluator} and \code{extra_args}.
#' @keywords internal
#' @noRd
resolve_tuned_evaluator <- function(evaluator, best_params = NULL, extra_args = list(), always_unwrap = FALSE) {
  has_params <- !is.null(best_params) && length(best_params) > 0
  list(
    evaluator = if (always_unwrap || has_params) unwrap_evaluator(evaluator) else evaluator,
    extra_args = if (has_params) utils::modifyList(as.list(extra_args), as.list(best_params)) else as.list(extra_args)
  )
}

#' Check if an Evaluator is Tree-Based
#'
#' Inspects the traits of the evaluator (or its unwrapped base evaluator) to determine
#' if it is a tree-based algorithm (e.g. LightGBM, XGBoost, CatBoost).
#'
#' @param evaluator Character name or evaluator list object.
#' @return Logical scalar TRUE if tree-based, FALSE otherwise.
#' @export
is_tree_evaluator <- function(evaluator) {
  if (is.list(evaluator)) {
    if (!is.null(evaluator$traits) && !is.null(evaluator$traits$is_tree)) {
      return(isTRUE(evaluator$traits$is_tree))
    }
    if (!is.null(evaluator$base_evaluator)) {
      return(is_tree_evaluator(evaluator$base_evaluator))
    }
    return(FALSE)
  }
  if (!is.character(evaluator) || length(evaluator) != 1 || is.na(evaluator)) {
    return(FALSE)
  }
  base_name <- unwrap_evaluator(evaluator)
  if (has_evaluator(base_name)) {
    ev <- get_evaluator(base_name)
    if (!is.null(ev$traits) && !is.null(ev$traits$is_tree)) {
      return(isTRUE(ev$traits$is_tree))
    }
  }
  grepl("lightgbm|xgboost|catboost", tolower(evaluator))
}

#' Scale Evaluator Iterations with Dataset Size
#'
#' Scales tree-based iteration/round counts proportionally when moving from validation fold
#' to the full dataset during final model fitting.
#'
#' @param evaluator Character name or evaluator list object.
#' @param iters Integer initial number of iterations (e.g. \code{best_iteration}).
#' @param train_size Optional numeric size of training partition.
#' @param total_size Optional numeric total dataset size.
#' @param factor Optional numeric explicit scaling factor (defaults to \code{total_size / train_size}).
#' @return Integer scaled iterations, or original \code{iters} if not scaled.
#' @export
scale_evaluator_iterations <- function(evaluator, iters, train_size = NULL, total_size = NULL, factor = NULL) {
  if (is.null(iters) || is.na(iters) || !is.numeric(iters) || iters <= 0) return(iters)
  
  if (is.null(factor)) {
    if (!is.null(train_size) && !is.null(total_size) &&
        is.numeric(train_size) && is.numeric(total_size) &&
        train_size > 0 && total_size > train_size) {
      factor <- as.numeric(total_size) / as.numeric(train_size)
    } else {
      factor <- 1.0
    }
  }
  
  if (factor <= 1.0) return(as.integer(iters))
  
  should_scale <- FALSE
  if (is.character(evaluator) && length(evaluator) == 1 && !is.na(evaluator)) {
    base_name <- unwrap_evaluator(evaluator)
    if (has_evaluator(base_name)) {
      ev <- get_evaluator(base_name)
      if (!is.null(ev$traits$scale_iterations_with_data)) {
        should_scale <- isTRUE(ev$traits$scale_iterations_with_data)
      } else {
        should_scale <- is_tree_evaluator(evaluator)
      }
    } else {
      should_scale <- is_tree_evaluator(evaluator)
    }
  } else if (is.list(evaluator)) {
    if (!is.null(evaluator$traits$scale_iterations_with_data)) {
      should_scale <- isTRUE(evaluator$traits$scale_iterations_with_data)
    } else {
      should_scale <- is_tree_evaluator(evaluator)
    }
  } else {
    should_scale <- is_tree_evaluator(evaluator)
  }
  
  if (should_scale) {
    as.integer(max(1L, round(iters * factor)))
  } else {
    as.integer(iters)
  }
}

# --- Register Default Evaluators ---

.evaluator_control_params <- c(
  "verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design",
  "mbo_folds", "mbo_infill_opt", "early_stopping_rounds", "early_stopping_round",
  "early_stop_round", "early_stopping", "od_wait", "od_type", "fold_data",
  "bag_folds", "supervised_stage4",
  "realmlp_device", "n_cycles", "realmlp_n_cycles", "realmlp_lr",
  "realmlp_epochs", "realmlp_seed", "hidden_dim", "stop_epoch",
  "nthread", "nthreads", "num_threads", "n_jobs", "threads",
  "nrounds", "num_rounds", "n_rounds", "num_round", "nround", "epochs",
  "n_epochs", "iterations", "n_iterations", "seed",
  "init_score", "base_margin", "baseline", "offset", "train_offset",
  "val_init_score", "val_base_margin", "val_baseline", "val_offset", "offset_val",
  "newoffset", "new_base_margin",
  "cat_schema", "cat_cols", "cat_cardinalities", "enable_categorical",
  "use_cv", "s", "lm_s", "evofe_threads", "num_iteration",
  "tabm_epochs", "tabm_lr", "tabm_k", "tabm_hidden_dim", "tabm_n_blocks",
  "tabm_dropout", "tabm_weight_decay", "k_ensemble", "d_block", "n_blocks",
  "dropout", "num_emb", "n_bins", "d_embedding", "d_cat", "lr", "weight_decay",
  "batch_size", "lr_warmup_frac",
  "n_knots", "degree", "include_linear", "max_cat_levels",
  "approx_method", "feature_map", "n_components", "n_features", "sigma",
  "kernel", "max_irls_iter", "irls_tol",
  "mtry_ratio", "num.trees", "splitrule", "min.node.size", "sample.fraction",
  "num.random.splits", "respect.unordered.factors", "save.memory", "replace",
  "max.depth"
)

# Combine raw margin predictions with offset and apply link inverse
.apply_raw_margin_offset <- function(raw_preds, offset_vec, task, n_rows, num_class = NULL) {
  if (task == "regression") {
    return(as.numeric(raw_preds) + as.numeric(offset_vec))
  }
  if (task == "classification") {
    return(stats::plogis(as.numeric(raw_preds) + as.numeric(offset_vec)))
  }
  raw_mat <- if (is.matrix(raw_preds)) {
    raw_preds
  } else {
    matrix(as.numeric(raw_preds), nrow = as.integer(n_rows), ncol = as.integer(num_class))
  }
  z <- raw_mat + offset_vec
  row_max <- z[cbind(seq_len(nrow(z)), max.col(z, ties.method = "first"))]
  ez <- exp(z - row_max)
  ez / rowSums(ez)
}

# Build dense numeric + bounded OHE matrix for RFF (zero-copy when purely numeric)
.build_categorical_ohe_or_matrix <- function(x, cat_schema = NULL, state = NULL,
                                             max_cat_levels = 32L, threads = 1L) {
  n <- nrow(x)
  if (is.null(state)) {
    p_in <- ncol(x)
    cat_info <- .resolve_cat_cols(x, cat_schema = cat_schema)
    if (length(cat_info$cols) == 0L) {
      return(list(
        matrix = x,
        col_group = seq_len(p_in),
        state = list(cat_cols = integer(0L), p_in = p_in)
      ))
    }
    max_cat_levels <- max(1L, as.integer(max_cat_levels))
    num_cols <- setdiff(seq_len(p_in), cat_info$cols)
    num_means <- numeric(0L)
    if (length(num_cols) > 0L) {
      imp <- .impute_and_col_stats(x[, num_cols, drop = FALSE], compute_sd = FALSE, threads = threads)
      num_means <- imp$col_means
      x_num <- imp$mat
    } else {
      x_num <- matrix(0.0, nrow = n, ncol = 0L)
    }

    cat_specs <- list()
    ohe_widths <- integer(length(cat_info$cols))
    for (i in seq_along(cat_info$cols)) {
      K_keep <- min(cat_info$cardinalities[i], max_cat_levels)
      ohe_widths[i] <- K_keep + 1L
      cat_specs[[i]] <- list(orig_col = cat_info$cols[i], K_keep = K_keep)
    }
    total_cols <- length(num_cols) + sum(ohe_widths)
    out <- matrix(0.0, nrow = n, ncol = total_cols)
    if (length(num_cols) > 0L) {
      out[, seq_along(num_cols)] <- x_num
    }
    col_group <- integer(total_cols)
    if (length(num_cols) > 0L) col_group[seq_along(num_cols)] <- num_cols

    offset <- length(num_cols)
    for (i in seq_along(cat_specs)) {
      cs <- cat_specs[[i]]
      w <- cs$K_keep + 1L
      if (n > 0L) {
        code <- suppressWarnings(as.integer(floor(x[, cs$orig_col])))
        bad <- is.na(code) | code < 1L | code > cs$K_keep
        code[bad] <- w
        out[cbind(seq_len(n), offset + code)] <- 1.0
      }
      col_group[(offset + 1L):(offset + w)] <- cs$orig_col
      offset <- offset + w
    }

    return(list(
      matrix = out,
      col_group = col_group,
      state = list(
        p_in = p_in,
        num_cols = num_cols,
        num_means = num_means,
        cat_cols = cat_info$cols,
        cat_specs = cat_specs,
        total_cols = total_cols,
        col_group = col_group
      )
    ))
  }

  if (length(state$cat_cols) == 0L) {
    return(x)
  }
  out <- matrix(0.0, nrow = n, ncol = state$total_cols)
  if (length(state$num_cols) > 0L) {
    x_num <- .impute_and_col_stats(
      x[, state$num_cols, drop = FALSE],
      means = state$num_means,
      compute_sd = FALSE,
      threads = threads
    )$mat
    out[, seq_along(state$num_cols)] <- x_num
  }
  offset <- length(state$num_cols)
  for (cs in state$cat_specs) {
    w <- cs$K_keep + 1L
    if (n > 0L) {
      code <- suppressWarnings(as.integer(floor(x[, cs$orig_col])))
      bad <- is.na(code) | code < 1L | code > cs$K_keep
      code[bad] <- w
      out[cbind(seq_len(n), offset + code)] <- 1.0
    }
    offset <- offset + w
  }
  out
}

# Shared LightGBM Training Core (used by lightgbm and lightgbm_rf)
.lightgbm_fit_core <- function(x_train, y_train, x_val = NULL, task = "classification",
                               threads = default_threads(), num_class = NULL, nrounds = 50,
                               default_params = list(num_leaves = 15, learning_rate = 0.1),
                               enforce_params = list(), ...) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("The 'lightgbm' package is required to use the 'lightgbm' evaluator. Please install it.")
  }
  extra_params <- list(...)
  if (!is.null(extra_params$threads)) threads <- as.integer(extra_params$threads)
  if (!is.null(extra_params$nthreads)) threads <- as.integer(extra_params$nthreads)
  if (!is.null(extra_params$nthread)) threads <- as.integer(extra_params$nthread)
  if (!is.null(extra_params$num_threads)) threads <- as.integer(extra_params$num_threads)
  if (!is.null(extra_params$n_jobs)) threads <- as.integer(extra_params$n_jobs)
  if (!is.null(extra_params$epochs)) nrounds <- as.integer(extra_params$epochs)
  if (!is.null(extra_params$n_epochs)) nrounds <- as.integer(extra_params$n_epochs)
  if (!is.null(extra_params$iterations)) nrounds <- as.integer(extra_params$iterations)
  if (!is.null(extra_params$n_iterations)) nrounds <- as.integer(extra_params$n_iterations)
  if (!is.null(extra_params$num_round)) nrounds <- as.integer(extra_params$num_round)
  if (!is.null(extra_params$nround)) nrounds <- as.integer(extra_params$nround)
  if (!is.null(extra_params$num_rounds)) nrounds <- as.integer(extra_params$num_rounds)
  if (!is.null(extra_params$n_rounds)) nrounds <- as.integer(extra_params$n_rounds)
  if (!is.null(extra_params$nrounds)) nrounds <- as.integer(extra_params$nrounds)

  x_train <- .sanitize_feature_matrix(x_train, threads = threads)
  cs <- attr(x_train, "cat_schema", exact = TRUE)
  x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)
  cat_info <- .resolve_cat_cols(x_train, extra_args = extra_params, cat_schema = cs)
  cat_cols <- cat_info$cols

  offs <- .resolve_offset_args(
    extra_params,
    n_train = nrow(x_train),
    n_val = if (!is.null(x_val)) nrow(x_val) else NULL,
    task = task,
    num_class = num_class
  )

  dtrain <- if (length(cat_cols) > 0L) {
    lightgbm::lgb.Dataset(data = x_train, label = y_train, categorical_feature = cat_cols)
  } else {
    lightgbm::lgb.Dataset(data = x_train, label = y_train)
  }
  if (!is.null(offs$train)) {
    lightgbm::set_field(dtrain, "init_score", as.numeric(offs$train))
  }

  y_val <- extra_params$y_val
  metric_arg <- extra_params$metric
  early_stopping_rounds <- extra_params$early_stopping_rounds

  cm_arg <- if (!is.null(metric_arg) && is.character(metric_arg) && length(metric_arg) == 1L) {
    canonical_metric_name(metric_arg, task)
  } else {
    NULL
  }
  is_mae_metric <- !is.null(cm_arg) && cm_arg %in% c("mae", "cal_mae")
  reg_obj <- if (is_mae_metric) "regression_l1" else "regression"
  reg_metric <- if (is_mae_metric) "mae" else if (identical(cm_arg, "mape")) "mape" else "rmse"
  cls_metric <- if (identical(cm_arg, "auc")) {
    "auc"
  } else if (identical(cm_arg, "aucpr")) {
    "average_precision"
  } else if (identical(cm_arg, "accuracy")) {
    "binary_error"
  } else {
    "binary_logloss"
  }
  mc_metric <- if (identical(cm_arg, "accuracy")) "multi_error" else "multi_logloss"

  learner_verbose <- is_learner_verbose(extra_params$verbose)
  seed_val <- if (!is.null(extra_params$seed) && is.numeric(extra_params$seed) && is.finite(extra_params$seed)) {
    as.integer(extra_params$seed)
  } else {
    42L
  }

  params <- list(
    objective = switch(task,
      classification = "binary",
      multiclass     = "multiclass",
      reg_obj
    ),
    metric = switch(task,
      classification = cls_metric,
      multiclass     = mc_metric,
      reg_metric
    ),
    verbose = if (learner_verbose) 1L else -1L,
    num_threads = threads,
    seed = seed_val
  )
  for (dp_name in names(default_params)) {
    params[[dp_name]] <- default_params[[dp_name]]
  }
  if (task == "multiclass") params$num_class <- num_class

  use_custom_eval <- FALSE
  custom_eval_type <- NULL
  if (!is.null(metric_arg) && is.character(metric_arg)) {
    metric_lower <- tolower(metric_arg)
    if (metric_lower %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      use_custom_eval <- TRUE
      custom_eval_type <- "ts_refinement"
    } else if (metric_lower %in% c("cal_rmse", "cal-rmse")) {
      use_custom_eval <- TRUE
      custom_eval_type <- "cal_rmse"
    } else if (metric_lower %in% c("cal_mae", "cal-mae")) {
      use_custom_eval <- TRUE
      custom_eval_type <- "cal_mae"
    }
  }

  if (use_custom_eval) {
    params$metric <- "None"
  }

  control_params <- .evaluator_control_params
  extra_params <- extra_params[!names(extra_params) %in% control_params]
  for (name in names(extra_params)) {
    params[[name]] <- extra_params[[name]]
  }
  for (ep_name in names(enforce_params)) {
    params[[ep_name]] <- enforce_params[[ep_name]]
  }
  if (identical(params$boosting, "rf")) {
    if (is.null(params$bagging_freq) || !is.finite(params$bagging_freq) || as.integer(params$bagging_freq) < 1L) {
      params$bagging_freq <- 1L
    } else {
      params$bagging_freq <- as.integer(params$bagging_freq)
    }
    bf <- if (!is.null(params$bagging_fraction) && is.finite(params$bagging_fraction)) {
      as.numeric(params$bagging_fraction)
    } else {
      0.8
    }
    ff <- if (!is.null(params$feature_fraction) && is.finite(params$feature_fraction)) {
      as.numeric(params$feature_fraction)
    } else {
      0.8
    }
    if (bf <= 0 || bf > 1.0) bf <- 0.8
    if (ff <= 0 || ff > 1.0) ff <- 0.8
    if (bf >= 1.0 && ff >= 1.0) {
      bf <- 0.8
      ff <- 0.8
    }
    params$bagging_fraction <- bf
    params$feature_fraction <- ff
  }

  valids <- list()
  needs_val <- (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) || learner_verbose
  if (needs_val && !is.null(x_val) && !is.null(y_val)) {
    dval <- if (length(cat_cols) > 0L) {
      lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain, categorical_feature = cat_cols)
    } else {
      lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain)
    }
    if (!is.null(offs$val)) {
      lightgbm::set_field(dval, "init_score", as.numeric(offs$val))
    }
    valids$val <- dval
  }

  cal_eval_score <- if (use_custom_eval && length(valids) > 0) {
    make_calibration_eval_fn(y_val, custom_eval_type, task = task, num_class = num_class, is_logits = FALSE, threads = threads)
  } else {
    NULL
  }

  # early_stopping_rounds requires at least one validation dataset
  esr <- if (length(valids) > 0) early_stopping_rounds else NULL
  eval_fn <- if (!is.null(cal_eval_score)) {
    function(preds, dtrain) {
      score <- cal_eval_score(preds, if (is.null(y_val)) lightgbm::get_field(dtrain, "label") else NULL)
      list(name = custom_eval_type, value = score, higher_better = FALSE)
    }
  } else {
    NULL
  }

  if (learner_verbose) {
    model <- lightgbm::lgb.train(
      params = params,
      data = dtrain,
      nrounds = nrounds,
      valids = valids,
      eval = eval_fn,
      early_stopping_rounds = esr,
      verbose = 1L
    )
  } else {
    utils::capture.output({
      model <- lightgbm::lgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        valids = valids,
        eval = eval_fn,
        early_stopping_rounds = esr,
        verbose = -1L
      )
    })
  }

  attr(model, "evofe_cat_schema") <- cs
  attr(model, "evofe_num_class") <- if (task == "multiclass") as.integer(num_class) else NULL

  best_it <- if (!is.null(model$best_iter) && model$best_iter > 0) model$best_iter else NULL
  preds <- if (!is.null(x_val)) {
    if (!is.null(offs$val)) {
      # predict.lgb.Booster does not add test init_score automatically; combine in raw margin space
      raw_p <- if (!is.null(best_it)) {
        stats::predict(model, x_val, num_iteration = best_it, type = "raw", params = list(num_threads = threads))
      } else {
        stats::predict(model, x_val, type = "raw", params = list(num_threads = threads))
      }
      .apply_raw_margin_offset(raw_p, offs$val, task = task, n_rows = nrow(x_val), num_class = num_class)
    } else if (!is.null(best_it)) {
      stats::predict(model, x_val, num_iteration = best_it, params = list(num_threads = threads))
    } else {
      stats::predict(model, x_val, params = list(num_threads = threads))
    }
  } else {
    NULL
  }

  importances <- tryCatch(
    {
      n_tr <- nrow(x_train)
      if (n_tr > 2000L) {
        sub_idx <- with_seed(seed_val, sample.int(n_tr, 2000L))
        x_sub <- as.matrix(.slice_feature_matrix(x_train, sub_idx))
        sh <- stats::predict(model, x_sub, type = "contrib", num_iteration = best_it, params = list(num_threads = threads))
      } else {
        sh <- stats::predict(model, as.matrix(x_train), type = "contrib", num_iteration = best_it, params = list(num_threads = threads))
      }
      .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
    },
    error = function(e) {
      tryCatch({
        df_imp <- lightgbm::lgb.importance(model)
        imp <- stats::setNames(numeric(ncol(x_train)), colnames(x_train))
        if (is.data.frame(df_imp) && nrow(df_imp) > 0 && "Feature" %in% names(df_imp) && "Gain" %in% names(df_imp)) {
          imp[df_imp$Feature] <- df_imp$Gain
        }
        imp
      }, error = function(e2) NULL)
    }
  )

  rm(dtrain)
  list(model = model, predictions = preds, importances = importances, best_iteration = best_it)
}

.lightgbm_predict_core <- function(model, x_new, task, ...) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("The 'lightgbm' package is required to use the 'lightgbm' evaluator. Please install it.")
  }
  extra_args <- list(...)
  th <- resolve_param_aliases(extra_args)$threads
  if (is.null(th)) {
    th <- if (!is.null(model$params$num_threads)) as.integer(model$params$num_threads) else default_threads()
  }
  cs <- attr(model, "evofe_cat_schema", exact = TRUE)
  x_new <- .sanitize_feature_matrix(x_new, threads = th, cat_schema = cs)
  nc <- attr(model, "evofe_num_class", exact = TRUE)
  if (is.null(nc) && identical(task, "multiclass")) nc <- model$params$num_class
  off_new <- .resolve_offset_args(extra_args, n_new = nrow(x_new), task = task, num_class = nc)$new

  num_iter <- if (!is.null(extra_args$num_iteration)) {
    as.integer(extra_args$num_iteration)
  } else if (!is.null(model$best_iter) && model$best_iter > 0) {
    model$best_iter
  } else {
    NULL
  }

  if (!is.null(off_new)) {
    raw_p <- if (!is.null(num_iter) && num_iter > 0) {
      stats::predict(model, x_new, num_iteration = num_iter, type = "raw", params = list(num_threads = th))
    } else {
      stats::predict(model, x_new, type = "raw", params = list(num_threads = th))
    }
    return(.apply_raw_margin_offset(raw_p, off_new, task = task, n_rows = nrow(x_new), num_class = nc))
  }

  if (!is.null(num_iter) && num_iter > 0) {
    stats::predict(model, x_new, num_iteration = num_iter, params = list(num_threads = th))
  } else {
    stats::predict(model, x_new, params = list(num_threads = th))
  }
}

# 1. LightGBM Evaluator
register_evaluator(
  "lightgbm",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    .lightgbm_fit_core(
      x_train = x_train, y_train = y_train, x_val = x_val,
      task = task, threads = threads, num_class = num_class, nrounds = nrounds,
      default_params = list(num_leaves = 15, learning_rate = 0.1),
      ...
    )
  },
  predict_func = function(model, x_new, task, ...) {
    .lightgbm_predict_core(model = model, x_new = x_new, task = task, ...)
  },
  traits = list(is_tree = TRUE, iteration_param = "nrounds", scale_iterations_with_data = TRUE)
)

# 1b. LightGBM Extra-Trees Random Forest Evaluator
register_evaluator(
  "lightgbm_rf",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    .lightgbm_fit_core(
      x_train = x_train, y_train = y_train, x_val = x_val,
      task = task, threads = threads, num_class = num_class, nrounds = nrounds,
      default_params = list(
        boosting = "rf",
        extra_trees = TRUE,
        num_leaves = 63L,
        bagging_freq = 1L,
        bagging_fraction = 0.8,
        feature_fraction = 0.7,
        min_data_in_leaf = 10L
      ),
      enforce_params = list(boosting = "rf", extra_trees = TRUE),
      ...
    )
  },
  predict_func = function(model, x_new, task, ...) {
    .lightgbm_predict_core(model = model, x_new = x_new, task = task, ...)
  },
  traits = list(is_tree = TRUE, iteration_param = "nrounds", scale_iterations_with_data = FALSE)
)

.xgb_make_dmatrix <- function(data, label = NULL, cat_cols = integer(0L),
                             feature_types = NULL, base_margin = NULL, nthread = 1L) {
  if (!is.null(cat_cols) && length(cat_cols) > 0L) {
    for (j in cat_cols) {
      if (j >= 1L && j <= ncol(data)) {
        col_v <- data[, j]
        neg <- !is.na(col_v) & col_v < 0
        if (any(neg)) data[neg, j] <- NA_real_
      }
    }
  }
  args <- list(data = data, missing = NA, nthread = as.integer(nthread))
  if (!is.null(label)) args$label <- label
  if (!is.null(feature_types)) args$feature_types <- feature_types
  if (!is.null(base_margin)) args$base_margin <- base_margin
  do.call(xgboost::xgb.DMatrix, args)
}

# 2. XGBoost Evaluator
register_evaluator(
  "xgboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("The 'xgboost' package is required to use the 'xgboost' evaluator. Please install it.")
    }
    extra_params <- list(...)
    if (!is.null(extra_params$threads)) threads <- as.integer(extra_params$threads)
    if (!is.null(extra_params$nthreads)) threads <- as.integer(extra_params$nthreads)
    if (!is.null(extra_params$nthread)) threads <- as.integer(extra_params$nthread)
    if (!is.null(extra_params$num_threads)) threads <- as.integer(extra_params$num_threads)
    if (!is.null(extra_params$n_jobs)) threads <- as.integer(extra_params$n_jobs)
    if (!is.null(extra_params$epochs)) nrounds <- as.integer(extra_params$epochs)
    if (!is.null(extra_params$n_epochs)) nrounds <- as.integer(extra_params$n_epochs)
    if (!is.null(extra_params$iterations)) nrounds <- as.integer(extra_params$iterations)
    if (!is.null(extra_params$n_iterations)) nrounds <- as.integer(extra_params$n_iterations)
    if (!is.null(extra_params$num_round)) nrounds <- as.integer(extra_params$num_round)
    if (!is.null(extra_params$nround)) nrounds <- as.integer(extra_params$nround)
    if (!is.null(extra_params$num_rounds)) nrounds <- as.integer(extra_params$num_rounds)
    if (!is.null(extra_params$n_rounds)) nrounds <- as.integer(extra_params$n_rounds)
    if (!is.null(extra_params$nrounds)) nrounds <- as.integer(extra_params$nrounds)

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)
    cat_info <- .resolve_cat_cols(x_train, extra_args = extra_params, cat_schema = cs)
    cat_cols <- cat_info$cols
    ft <- if (length(cat_cols) > 0L) {
      f_vec <- rep("q", ncol(x_train))
      f_vec[cat_cols] <- "c"
      f_vec
    } else {
      NULL
    }

    offs <- .resolve_offset_args(
      extra_params,
      n_train = nrow(x_train),
      n_val = if (!is.null(x_val)) nrow(x_val) else NULL,
      task = task,
      num_class = num_class
    )

    dtrain <- .xgb_make_dmatrix(
      data = x_train,
      label = y_train,
      cat_cols = cat_cols,
      feature_types = ft,
      base_margin = offs$train,
      nthread = threads
    )
    y_val <- extra_params$y_val
    metric_arg <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    cm_arg <- if (!is.null(metric_arg) && is.character(metric_arg) && length(metric_arg) == 1L) {
      canonical_metric_name(metric_arg, task)
    } else {
      NULL
    }
    is_mae_metric <- !is.null(cm_arg) && cm_arg %in% c("mae", "cal_mae")
    reg_obj <- if (is_mae_metric) "reg:absoluteerror" else "reg:squarederror"
    reg_eval_metric <- if (is_mae_metric) "mae" else if (identical(cm_arg, "mape")) "mape" else "rmse"
    cls_eval_metric <- if (identical(cm_arg, "auc")) {
      "auc"
    } else if (identical(cm_arg, "aucpr")) {
      "aucpr"
    } else if (identical(cm_arg, "accuracy")) {
      "error"
    } else {
      "logloss"
    }
    mc_eval_metric <- if (identical(cm_arg, "accuracy")) "merror" else "mlogloss"

    learner_verbose <- is_learner_verbose(extra_params$verbose)
    seed_val <- if (!is.null(extra_params$seed) && is.numeric(extra_params$seed) && is.finite(extra_params$seed)) {
      as.integer(extra_params$seed)
    } else {
      42L
    }

    params <- list(
      objective = switch(task,
        classification = "binary:logistic",
        multiclass     = "multi:softprob",
        reg_obj
      ),
      eval_metric = switch(task,
        classification = cls_eval_metric,
        multiclass     = mc_eval_metric,
        reg_eval_metric
      ),
      nthread = threads,
      max_depth = 6,
      eta = 0.1,
      min_child_weight = 1,
      seed = seed_val,
      verbosity = if (learner_verbose) 1L else 0L,
      silent = if (learner_verbose) 0L else 1L
    )
    if (task == "multiclass") params$num_class <- num_class

    use_custom_eval <- FALSE
    custom_eval_type <- NULL
    if (!is.null(metric_arg) && is.character(metric_arg)) {
      metric_lower <- tolower(metric_arg)
      if (metric_lower %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "ts_refinement"
      } else if (metric_lower %in% c("cal_rmse", "cal-rmse")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_rmse"
      } else if (metric_lower %in% c("cal_mae", "cal-mae")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_mae"
      }
    }

    if (use_custom_eval) {
      params$eval_metric <- NULL
    }

    control_params <- .evaluator_control_params
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    if (length(cat_cols) > 0L) {
      # Categoricals require tree_method "hist" or "approx" in xgboost
      if (is.null(params$tree_method) || identical(params$tree_method, "exact") ||
          identical(params$tree_method, "auto")) {
        params$tree_method <- "hist"
      }
    }

    if (identical(params$booster, "gblinear")) {
      params$nthread <- 1L
    }

    dval_metric <- NULL
    # Early stopping or verbose logging requires a validation set in `evals`
    needs_val <- (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) || learner_verbose
    if (needs_val && !is.null(x_val) && !is.null(y_val)) {
      dval_metric <- .xgb_make_dmatrix(
        data = x_val,
        label = y_val,
        cat_cols = cat_cols,
        feature_types = ft,
        base_margin = offs$val,
        nthread = threads
      )
      evals <- list(val = dval_metric)
    } else {
      evals <- list()
      early_stopping_rounds <- NULL
    }

    cal_eval_score <- if (use_custom_eval && length(evals) > 0) {
      make_calibration_eval_fn(y_val, custom_eval_type, task = task, num_class = num_class, is_logits = TRUE, threads = threads)
    } else {
      NULL
    }

    xgb_feval <- if (!is.null(cal_eval_score)) {
      function(preds, dtrain) {
        score <- cal_eval_score(preds, if (is.null(y_val)) xgboost::getinfo(dtrain, "label") else NULL)
        list(metric = custom_eval_type, value = score)
      }
    } else {
      NULL
    }

    use_feval <- !is.null(xgb_feval)
    xgb_train_call <- function() {
      suppressWarnings(xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        evals = evals,
        custom_metric = if (use_feval) xgb_feval else NULL,
        early_stopping_rounds = early_stopping_rounds,
        maximize = if (use_feval) FALSE else NULL,
        verbose = if (learner_verbose) 1L else 0L
      ))
    }

    model <- if (learner_verbose) {
      xgb_train_call()
    } else {
      utils::capture.output({
        res <- xgb_train_call()
      })
      res
    }

    attr(model, "evoFE_threads") <- as.integer(params$nthread)
    attr(model, "evofe_feature_types") <- ft
    attr(model, "evofe_cat_cols") <- cat_cols
    attr(model, "evofe_cat_schema") <- cs
    attr(model, "evofe_num_class") <- if (task == "multiclass") as.integer(num_class) else NULL
    best_iter <- .xgb_best_iter(model)

    preds <- if (!is.null(x_val)) {
      dval <- if (!is.null(dval_metric)) {
        dval_metric
      } else {
        .xgb_make_dmatrix(
          data = x_val,
          cat_cols = cat_cols,
          feature_types = ft,
          base_margin = offs$val,
          nthread = threads
        )
      }
      p <- if (!is.null(best_iter) && best_iter >= 1) {
        stats::predict(model, dval, iterationrange = c(1, best_iter + 1))
      } else {
        stats::predict(model, dval)
      }
      if (is.null(dval_metric)) rm(dval)
      p
    } else {
      NULL
    }

    importances <- tryCatch(
      {
        best_it_range <- if (!is.null(best_iter) && best_iter >= 1) c(1, best_iter + 1) else NULL
        n_tr <- nrow(x_train)
        if (n_tr > 2000L) {
          sub_idx <- with_seed(seed_val, sample.int(n_tr, 2000L))
          sub_margin <- if (!is.null(offs$train)) {
            if (is.matrix(offs$train)) offs$train[sub_idx, , drop = FALSE] else offs$train[sub_idx]
          } else {
            NULL
          }
          dsub <- .xgb_make_dmatrix(
            data = .slice_feature_matrix(x_train, sub_idx),
            cat_cols = cat_cols,
            feature_types = ft,
            base_margin = sub_margin,
            nthread = threads
          )
          sh <- stats::predict(model, dsub, predcontrib = TRUE, iterationrange = best_it_range)
          rm(dsub)
        } else {
          sh <- stats::predict(model, dtrain, predcontrib = TRUE, iterationrange = best_it_range)
        }
        .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
      },
      error = function(e) {
        tryCatch({
          df_imp <- xgboost::xgb.importance(model = model)
          imp <- stats::setNames(numeric(ncol(x_train)), colnames(x_train))
          if (is.data.frame(df_imp) && nrow(df_imp) > 0 && "Feature" %in% names(df_imp) && "Gain" %in% names(df_imp)) {
            imp[df_imp$Feature] <- df_imp$Gain
          }
          imp
        }, error = function(e2) NULL)
      }
    )

    rm(dtrain)
    if (!is.null(dval_metric)) rm(dval_metric)
    list(model = model, predictions = preds, importances = importances, best_iteration = best_iter)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("The 'xgboost' package is required to use the 'xgboost' evaluator. Please install it.")
    }
    extra_args <- list(...)
    th <- resolve_param_aliases(extra_args)$threads
    model_th <- attr(model, "evoFE_threads", exact = TRUE)
    if (is.null(th)) {
      th <- if (!is.null(model_th)) as.integer(model_th) else default_threads()
    }
    if (!identical(model_th, th)) {
      tryCatch({
        xgboost::xgb.model.parameters(model) <- list(nthread = th)
      }, error = function(e) NULL)
    }
    cs_model <- attr(model, "evofe_cat_schema", exact = TRUE)
    x_new <- .sanitize_feature_matrix(x_new, threads = th, cat_schema = cs_model)
    ft <- attr(model, "evofe_feature_types", exact = TRUE)
    cat_cols <- attr(model, "evofe_cat_cols", exact = TRUE)
    nc <- attr(model, "evofe_num_class", exact = TRUE)
    off_new <- .resolve_offset_args(extra_args, n_new = nrow(x_new), task = task, num_class = nc)$new

    best_iter <- .xgb_best_iter(model)
    iter_range <- if (!is.null(extra_args$iterationrange)) {
      extra_args$iterationrange
    } else if (!is.null(best_iter) && best_iter >= 1) {
      c(1, best_iter + 1)
    } else {
      NULL
    }
    dmatrix <- .xgb_make_dmatrix(
      data = x_new,
      cat_cols = cat_cols,
      feature_types = ft,
      base_margin = off_new,
      nthread = th
    )
    preds <- if (!is.null(iter_range)) {
      stats::predict(model, dmatrix, iterationrange = iter_range)
    } else {
      stats::predict(model, dmatrix)
    }
    if (identical(task, "multiclass") && !is.matrix(preds)) {
      preds <- matrix(preds, nrow = nrow(x_new), byrow = TRUE)
    }
    rm(dmatrix)
    preds
  },
  traits = list(is_tree = TRUE, iteration_param = "nrounds", scale_iterations_with_data = TRUE)
)

.cb_load_pool <- function(data, label = NULL, cat_cols = integer(0L),
                          baseline = NULL, thread_count = 1L) {
  args <- list(data = data, thread_count = as.integer(thread_count))
  if (!is.null(label)) args$label <- label
  if (!is.null(cat_cols) && length(cat_cols) > 0L) {
    # catboost.load_pool on a numeric matrix strictly requires 0-based indices
    args$cat_features <- as.integer(cat_cols - 1L)
  }
  if (!is.null(baseline)) {
    args$baseline <- if (is.matrix(baseline)) {
      storage.mode(baseline) <- "double"
      baseline
    } else {
      matrix(as.numeric(baseline), ncol = 1L)
    }
  }
  do.call(catboost::catboost.load_pool, args)
}

# 3. CatBoost Evaluator
register_evaluator(
  "catboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    if (system.file(package = "catboost") == "") {
      stop("The 'catboost' package is required to use the 'catboost' evaluator. Please install it.")
    }

    extra_params <- list(...)

    # 1. Thread count
    threads_val <- threads
    for (th_key in c("threads", "nthreads", "nthread", "num_threads", "n_jobs")) {
      if (!is.null(extra_params[[th_key]])) {
        threads_val <- as.integer(extra_params[[th_key]])
        break
      }
    }
    threads <- max(1L, threads_val)

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)
    cat_info <- .resolve_cat_cols(x_train, extra_args = extra_params, cat_schema = cs)
    cat_cols <- cat_info$cols

    offs <- .resolve_offset_args(
      extra_params,
      n_train = nrow(x_train),
      n_val = if (!is.null(x_val)) nrow(x_val) else NULL,
      task = task,
      num_class = num_class
    )

    dtrain <- .cb_load_pool(
      data = x_train,
      label = y_train,
      cat_cols = cat_cols,
      baseline = offs$train,
      thread_count = threads
    )

    # 2. Iterations / nrounds
    iters_val <- nrounds
    for (it_key in c("iterations", "n_iterations", "nrounds", "num_rounds", "n_rounds",
                     "num_round", "nround", "epochs", "n_epochs")) {
      if (!is.null(extra_params[[it_key]])) {
        iters_val <- as.integer(extra_params[[it_key]])
        break
      }
    }
    iters <- max(1L, as.integer(iters_val))

    # 3. Learning rate / eta
    lr_val <- if (!is.null(extra_params$learning_rate)) {
      extra_params$learning_rate
    } else if (!is.null(extra_params$eta)) {
      extra_params$eta
    } else if (!is.null(extra_params$lr)) {
      extra_params$lr
    } else {
      0.1
    }
    lr <- if (is.numeric(lr_val) && is.finite(lr_val) && lr_val > 0) as.numeric(lr_val) else 0.1

    # 4. Depth / max_depth (strictly clamped to [1, 16] for CatBoost C++ constraints)
    depth_val <- if (!is.null(extra_params$depth)) {
      extra_params$depth
    } else if (!is.null(extra_params$max_depth)) {
      extra_params$max_depth
    } else {
      NULL
    }
    depth <- if (!is.null(depth_val) && is.numeric(depth_val) && is.finite(depth_val)) {
      min(16L, max(1L, as.integer(round(depth_val))))
    } else {
      NULL
    }

    # 5. Regularization l2_leaf_reg / lambda
    l2_val <- if (!is.null(extra_params$l2_leaf_reg)) {
      extra_params$l2_leaf_reg
    } else if (!is.null(extra_params$lambda)) {
      extra_params$lambda
    } else if (!is.null(extra_params$reg_lambda)) {
      extra_params$reg_lambda
    } else {
      NULL
    }
    l2_reg <- if (!is.null(l2_val) && is.numeric(l2_val) && is.finite(l2_val) && l2_val >= 0) {
      as.numeric(l2_val)
    } else {
      NULL
    }

    # 6. Column subsampling: rsm / colsample_bytree / feature_fraction (must be in (0, 1])
    rsm_val <- if (!is.null(extra_params$rsm)) {
      extra_params$rsm
    } else if (!is.null(extra_params$colsample_bytree)) {
      extra_params$colsample_bytree
    } else if (!is.null(extra_params$feature_fraction)) {
      extra_params$feature_fraction
    } else {
      NULL
    }
    rsm <- if (!is.null(rsm_val) && is.numeric(rsm_val) && is.finite(rsm_val) && rsm_val > 0 && rsm_val <= 1.0) {
      as.numeric(rsm_val)
    } else {
      NULL
    }

    # 7. Row subsampling / bagging: subsample vs bagging_temperature
    # CatBoost rule: Bayesian bootstrap does NOT support 'subsample'.
    # If bagging_temperature is given, use Bayesian bootstrap and drop subsample.
    # If subsample < 1.0 is given without bagging_temperature, use Bernoulli bootstrap.
    bag_temp <- if (!is.null(extra_params$bagging_temperature) && is.numeric(extra_params$bagging_temperature) && is.finite(extra_params$bagging_temperature)) {
      max(0.0, as.numeric(extra_params$bagging_temperature))
    } else {
      NULL
    }

    sub_val <- if (!is.null(extra_params$subsample)) {
      extra_params$subsample
    } else if (!is.null(extra_params$bagging_fraction)) {
      extra_params$bagging_fraction
    } else if (!is.null(extra_params$subsample_ratio)) {
      extra_params$subsample_ratio
    } else {
      NULL
    }
    subsample <- if (is.null(bag_temp) && !is.null(sub_val) && is.numeric(sub_val) && is.finite(sub_val) && sub_val > 0 && sub_val <= 1.0) {
      as.numeric(sub_val)
    } else {
      NULL
    }

    bootstrap_type <- if (!is.null(extra_params$bootstrap_type)) {
      as.character(extra_params$bootstrap_type)
    } else if (!is.null(bag_temp)) {
      "Bayesian"
    } else if (!is.null(subsample) && subsample < 1.0) {
      "Bernoulli"
    } else {
      NULL
    }

    # 8. Random seed
    seed_val <- if (!is.null(extra_params$random_seed) && is.numeric(extra_params$random_seed) && is.finite(extra_params$random_seed)) {
      as.integer(extra_params$random_seed)
    } else if (!is.null(extra_params$seed) && is.numeric(extra_params$seed) && is.finite(extra_params$seed)) {
      as.integer(extra_params$seed)
    } else {
      42L
    }

    # 9. Loss function
    metric_arg <- extra_params$metric
    is_mae_metric <- !is.null(metric_arg) && is.character(metric_arg) && tolower(metric_arg) %in% c("mae", "cal_mae", "cal-mae")
    reg_loss <- if (is_mae_metric) "MAE" else "RMSE"
    loss_fn <- switch(task,
      classification = "Logloss",
      multiclass     = "MultiClass",
      reg_loss
    )

    # 10. Evaluation metric (validate against task and loss function)
    eval_metric <- NULL
    if (!is.null(extra_params$eval_metric) && is.character(extra_params$eval_metric) && length(extra_params$eval_metric) == 1L) {
      em_cand <- extra_params$eval_metric
      valid_em <- switch(task,
        classification = c("Logloss", "CrossEntropy", "AUC", "Accuracy", "Precision", "Recall", "F1", "TotalF1", "MCC", "BrierScore"),
        multiclass     = c("MultiClass", "MultiClassOneVsAll", "Accuracy", "TotalF1", "Kappa", "WKappa"),
        c("RMSE", "MAE", "Quantile", "MAPE", "MSLE", "MedianAbsoluteError", "SMAPE", "Huber", "R2")
      )
      if (em_cand %in% valid_em) {
        eval_metric <- em_cand
      }
    }

    # 11. Early stopping patience (from any alias)
    es_val <- NULL
    for (es_key in c("early_stopping_rounds", "early_stopping_round", "early_stop_round", "early_stopping", "od_wait")) {
      if (!is.null(extra_params[[es_key]]) && is.numeric(extra_params[[es_key]]) && is.finite(extra_params[[es_key]])) {
        es_val <- as.integer(extra_params[[es_key]])
        break
      }
    }

    y_val <- extra_params$y_val
    has_val_data <- !is.null(x_val) && !is.null(y_val) && (if (is.matrix(x_val) || is.data.frame(x_val)) nrow(x_val) > 0L else length(x_val) > 0L)
    use_early_stopping <- has_val_data && !is.null(es_val) && (es_val > 0L)

    opt_verbose <- getOption("evoFE.verbose", 0)
    verbose_arg <- if (!is.null(extra_params$verbose)) extra_params$verbose else opt_verbose
    learner_verbose <- is_learner_verbose(verbose_arg)

    params <- list(
      loss_function       = loss_fn,
      iterations          = iters,
      learning_rate       = lr,
      thread_count        = threads,
      logging_level       = if (learner_verbose) "Verbose" else "Silent",
      allow_writing_files = FALSE,
      train_dir           = tempdir(),
      random_seed         = seed_val
    )

    if (!is.null(depth)) params$depth <- depth
    if (!is.null(l2_reg)) params$l2_leaf_reg <- l2_reg
    if (!is.null(rsm)) params$rsm <- rsm
    if (!is.null(bootstrap_type)) params$bootstrap_type <- bootstrap_type
    if (!is.null(subsample)) params$subsample <- subsample
    if (!is.null(bag_temp)) params$bagging_temperature <- bag_temp
    if (!is.null(eval_metric)) params$eval_metric <- eval_metric

    if (!is.null(extra_params$random_strength) && is.numeric(extra_params$random_strength) && is.finite(extra_params$random_strength)) {
      params$random_strength <- as.numeric(extra_params$random_strength)
    }
    if (!is.null(extra_params$border_count) && is.numeric(extra_params$border_count) && is.finite(extra_params$border_count)) {
      params$border_count <- min(254L, max(1L, as.integer(extra_params$border_count)))
    }
    if (!is.null(extra_params$min_data_in_leaf) && is.numeric(extra_params$min_data_in_leaf) && is.finite(extra_params$min_data_in_leaf)) {
      params$min_data_in_leaf <- max(1L, as.integer(extra_params$min_data_in_leaf))
    }

    # Add other safe, allowed CatBoost parameters from extra_params
    catboost_allowed_extra <- c(
      "nan_mode", "counter_calc_method", "leaf_estimation_iterations", "leaf_estimation_method",
      "metric_period", "ctr_leaf_count_limit", "store_all_simple_ctr", "max_ctr_complexity",
      "has_time", "classes_count", "class_weights", "auto_class_weights",
      "sampling_unit", "sampling_frequency", "score_function", "fold_len_multiplier",
      "approx_on_full_history", "task_type"
    )
    for (nm in intersect(names(extra_params), catboost_allowed_extra)) {
      params[[nm]] <- extra_params[[nm]]
    }

    # Force CRAN-compliant file safety
    params$allow_writing_files <- FALSE
    params$train_dir <- tempdir()

    # Build validation pool for early stopping when possible
    dval_es <- NULL
    if (use_early_stopping) {
      dval_es <- .cb_load_pool(
        data = x_val,
        label = y_val,
        cat_cols = cat_cols,
        baseline = offs$val,
        thread_count = threads
      )
      params$od_type <- if (!is.null(extra_params$od_type)) as.character(extra_params$od_type) else "Iter"
      params$od_wait <- es_val
    } else {
      params$od_wait <- NULL
      params$od_type <- NULL
      params$od_pval <- NULL
      params$early_stopping_rounds <- NULL
      params$early_stopping_round <- NULL
      params$early_stop_round <- NULL
      params$early_stopping <- NULL
    }

    cb_train_call <- function() {
      suppressWarnings(catboost::catboost.train(dtrain, test_pool = dval_es, params = params))
    }

    model <- if (learner_verbose) {
      cb_train_call()
    } else {
      utils::capture.output({
        res <- cb_train_call()
      })
      res
    }

    attr(model, "evofe_cat_cols") <- cat_cols
    attr(model, "evofe_cat_schema") <- cs
    attr(model, "evofe_num_class") <- if (task == "multiclass") as.integer(num_class) else NULL

    preds <- NULL
    if (!is.null(x_val)) {
      dval <- if (!is.null(dval_es)) {
        dval_es
      } else {
        .cb_load_pool(
          data = x_val,
          cat_cols = cat_cols,
          baseline = offs$val,
          thread_count = threads
        )
      }
      pred_type <- if (task == "regression") "RawFormulaVal" else "Probability"
      preds <- suppressWarnings(catboost::catboost.predict(model, dval, prediction_type = pred_type, thread_count = threads))

      # For multiclass, format shape as probability matrix
      if (task == "multiclass") {
        if (!is.matrix(preds)) {
          preds <- matrix(preds, ncol = num_class, byrow = TRUE)
        }
      }
    }

    # Fetch feature importance and map column names
    importances <- tryCatch(
      {
        n_tr <- nrow(x_train)
        pool_imp <- if (n_tr > 2000L) {
          sub_idx <- with_seed(seed_val, sample.int(n_tr, 2000L))
          sub_base <- if (!is.null(offs$train)) {
            if (is.matrix(offs$train)) offs$train[sub_idx, , drop = FALSE] else offs$train[sub_idx]
          } else {
            NULL
          }
          .cb_load_pool(
            data = .slice_feature_matrix(x_train, sub_idx),
            label = y_train[sub_idx],
            cat_cols = cat_cols,
            baseline = sub_base,
            thread_count = threads
          )
        } else {
          dtrain
        }
        sh <- catboost::catboost.get_feature_importance(model, pool = pool_imp, type = "ShapValues", thread_count = threads)
        .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
      },
      error = function(e) NULL
    )

    best_it <- tryCatch(
      if (!is.null(model$tree_count) && model$tree_count > 0) as.integer(model$tree_count) else NULL,
      error = function(e) NULL
    )
    list(model = model, predictions = preds, importances = importances, best_iteration = best_it)
  },
  predict_func = function(model, x_new, task, ...) {
    if (system.file(package = "catboost") == "") {
      stop("The 'catboost' package is required to use the 'catboost' evaluator. Please install it.")
    }
    user_args <- list(...)
    th <- if (!is.null(user_args$threads)) {
      as.integer(user_args$threads)
    } else if (!is.null(user_args$thread_count)) {
      as.integer(user_args$thread_count)
    } else {
      default_threads()
    }
    cs_model <- attr(model, "evofe_cat_schema", exact = TRUE)
    x_new <- .sanitize_feature_matrix(x_new, threads = th, cat_schema = cs_model)
    cat_cols <- attr(model, "evofe_cat_cols", exact = TRUE)
    nc <- attr(model, "evofe_num_class", exact = TRUE)
    if (is.null(nc) && !is.null(user_args$num_class)) nc <- user_args$num_class
    off_new <- .resolve_offset_args(user_args, n_new = nrow(x_new), task = task, num_class = nc)$new

    dval <- .cb_load_pool(
      data = x_new,
      cat_cols = cat_cols,
      baseline = off_new,
      thread_count = th
    )
    pred_type <- if (task == "regression") "RawFormulaVal" else "Probability"
    preds <- suppressWarnings(catboost::catboost.predict(model, dval, prediction_type = pred_type, thread_count = th))

    if (task == "multiclass" && !is.matrix(preds)) {
      if (!is.null(nc) && nc > 0) {
        preds <- matrix(preds, ncol = nc, byrow = TRUE)
      }
    }
    preds
  },
  traits = list(is_tree = TRUE, iteration_param = "iterations", scale_iterations_with_data = TRUE)
)

## 4. LM/GLM Evaluator (Penalized)
# Vectorized / metric-aware lambda selection from a single glmnet validation path
.select_glmnet_val_lambda <- function(p_all, lambdas, y_val, y_levels, task,
                                      metric_arg, cm_arg, s_choice, num_class, threads) {
  n_lam <- length(lambdas)
  if (n_lam <= 1L) return(1L)
  if (is.numeric(s_choice) && length(s_choice) == 1L && is.finite(s_choice)) {
    return(which.min(abs(lambdas - as.numeric(s_choice))))
  }

  use_1se <- identical(s_choice, "lambda.1se")
  cm_eff <- if (!is.null(cm_arg)) cm_arg else canonical_metric_name("default", task)

  loss_mat <- NULL
  if (!is.function(metric_arg) && identical(task, "regression") && cm_eff %in% c("rmse", "cal_rmse")) {
    yv <- as.numeric(y_val)
    loss_mat <- (p_all - yv)^2
  } else if (!is.function(metric_arg) && identical(task, "regression") && cm_eff %in% c("mae", "cal_mae")) {
    yv <- as.numeric(y_val)
    loss_mat <- abs(p_all - yv)
  } else if (!is.function(metric_arg) && identical(task, "classification") && identical(cm_eff, "logloss")) {
    yb <- as.numeric(encode_binary_target(y_val))
    pc <- pmin(pmax(p_all, 1e-15), 1.0 - 1e-15)
    loss_mat <- -(yb * log(pc) + (1.0 - yb) * log(1.0 - pc))
  } else if (!is.function(metric_arg) && identical(task, "multiclass") && identical(cm_eff, "logloss")) {
    y_fac <- factor(y_val, levels = y_levels)
    y_idx <- as.integer(y_fac)
    n_val <- nrow(p_all)
    k_mat <- dim(p_all)[2L]
    idx_3d <- cbind(
      rep(seq_len(n_val), times = n_lam),
      rep(y_idx, times = n_lam),
      rep(seq_len(n_lam), each = n_val)
    )
    ok <- !is.na(y_idx) & y_idx >= 1L & y_idx <= k_mat
    probs <- numeric(n_val * n_lam)
    if (any(ok)) {
      probs[rep(ok, times = n_lam)] <- p_all[idx_3d[rep(ok, times = n_lam), , drop = FALSE]]
    }
    pc <- matrix(pmax(probs, 1e-15), nrow = n_val, ncol = n_lam)
    loss_mat <- -log(pc)
  }

  if (!is.null(loss_mat)) {
    mean_loss <- base::colMeans(loss_mat, na.rm = TRUE)
    mean_loss[!is.finite(mean_loss)] <- Inf
    best_idx <- which.min(mean_loss)
    if (use_1se && nrow(loss_mat) > 1L && is.finite(mean_loss[best_idx])) {
      se_min <- stats::sd(loss_mat[, best_idx], na.rm = TRUE) / sqrt(sum(is.finite(loss_mat[, best_idx])))
      if (is.finite(se_min) && se_min > 0) {
        cands <- which(mean_loss <= mean_loss[best_idx] + se_min)
        if (length(cands) > 0L) best_idx <- min(cands)
      }
    }
    return(best_idx)
  }

  scores <- numeric(n_lam)
  for (l in seq_len(n_lam)) {
    p_l <- if (identical(task, "multiclass")) {
      mat <- matrix(p_all[, , l], nrow = nrow(p_all), ncol = dim(p_all)[2L])
      colnames(mat) <- dimnames(p_all)[[2L]]
      mat
    } else {
      as.numeric(p_all[, l])
    }
    sc <- tryCatch(
      compute_metric(y_true = y_val, y_pred = p_l, task = task, metric = if (is.null(metric_arg)) "default" else metric_arg,
                     num_class = num_class, threads = threads),
      error = function(e) -Inf
    )
    scores[l] <- if (length(sc) == 1L && is.finite(sc)) sc else -Inf
  }
  which.max(scores)
}

# Strip full lambda path to the single selected lambda and drop embedded call matrices
.trim_glmnet_model <- function(fit, best_idx, alpha) {
  if (inherits(fit, "multnet")) {
    fit$a0 <- fit$a0[, best_idx, drop = FALSE]
    fit$beta <- lapply(fit$beta, function(b) b[, best_idx, drop = FALSE])
    if (!is.null(fit$dfmat)) fit$dfmat <- fit$dfmat[, best_idx, drop = FALSE]
    fit$dim <- c(nrow(fit$beta[[1L]]), 1L)
  } else {
    fit$a0 <- fit$a0[best_idx]
    fit$beta <- fit$beta[, best_idx, drop = FALSE]
    fit$dim <- c(nrow(fit$beta), 1L)
  }
  fit$df <- fit$df[best_idx]
  fit$lambda <- fit$lambda[best_idx]
  fit$dev.ratio <- fit$dev.ratio[best_idx]
  fit$call <- call("glmnet", alpha = alpha)
  fit
}

# Shared Penalized Linear (glmnet) Core (used by lm and spline_lm)
.glmnet_fit_core <- function(x_train, y_train, x_val = NULL, task = "regression",
                             threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
  if (!requireNamespace("glmnet", quietly = TRUE)) {
    stop("The 'glmnet' package is required for the penalized lm evaluator.")
  }

  extra_params <- list(...)
  if (!is.null(extra_params$threads)) threads <- as.integer(extra_params$threads)
  if (!is.null(extra_params$nthreads)) threads <- as.integer(extra_params$nthreads)
  if (!is.null(extra_params$nthread)) threads <- as.integer(extra_params$nthread)
  if (!is.null(extra_params$num_threads)) threads <- as.integer(extra_params$num_threads)
  if (!is.null(extra_params$n_jobs)) threads <- as.integer(extra_params$n_jobs)

  is_sparse <- inherits(x_train, "sparseMatrix")
  cs <- attr(x_train, "cat_schema", exact = TRUE)
  if (is_sparse) {
    x_mat <- x_train
    col_means <- numeric(0L)
    n_r <- nrow(x_mat)
    mu_sp <- Matrix::colMeans(x_mat)
    ex2_sp <- Matrix::colSums(x_mat^2)
    sd_x <- sqrt(pmax(0.0, (ex2_sp - n_r * (mu_sp^2)) / max(1.0, n_r - 1.0)))
    sd_x[!is.finite(sd_x) | sd_x <= 0] <- 1.0
    val_mat <- if (!is.null(x_val)) x_val else NULL
  } else {
    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)
    imp_tr <- .impute_and_col_stats(x_train, compute_sd = TRUE, threads = threads)
    x_mat <- imp_tr$mat
    col_means <- imp_tr$col_means
    sd_x <- imp_tr$col_sds
    if (!is.null(colnames(x_mat))) names(col_means) <- colnames(x_mat)
    val_mat <- if (!is.null(x_val)) {
      .impute_and_col_stats(x_val, means = col_means, compute_sd = FALSE, threads = threads)$mat
    } else {
      NULL
    }
  }

  all_feats <- colnames(x_mat)
  y_val <- extra_params$y_val

  alpha <- if (!is.null(extra_params$alpha) && is.numeric(extra_params$alpha) && is.finite(extra_params$alpha)) {
    min(max(as.numeric(extra_params$alpha), 0.0), 1.0)
  } else {
    0.5
  }

  s_choice <- if (!is.null(extra_params$lm_s) && (is.character(extra_params$lm_s) || (is.numeric(extra_params$lm_s) && is.finite(extra_params$lm_s)))) {
    extra_params$lm_s
  } else if (!is.null(extra_params$s) && (is.character(extra_params$s) || (is.numeric(extra_params$s) && is.finite(extra_params$s)))) {
    extra_params$s
  } else {
    "lambda.min"
  }

  lambda_val <- if (!is.null(extra_params$lambda) && is.numeric(extra_params$lambda) &&
                    length(extra_params$lambda) == 1L && is.finite(extra_params$lambda) &&
                    extra_params$lambda > 0) {
    as.numeric(extra_params$lambda)
  } else {
    NULL
  }

  standardize <- if (!is.null(extra_params$standardize) && is.logical(extra_params$standardize)) {
    isTRUE(extra_params$standardize)
  } else {
    TRUE
  }
  relax <- isTRUE(extra_params$relax)
  use_cv <- isTRUE(extra_params$use_cv)

  metric_arg <- extra_params$metric
  cm_arg <- if (!is.null(metric_arg) && is.character(metric_arg) && length(metric_arg) == 1L) {
    canonical_metric_name(metric_arg, task)
  } else {
    NULL
  }

  type_measure <- if (!is.null(extra_params$type.measure) && is.character(extra_params$type.measure)) {
    extra_params$type.measure
  } else if (!is.null(extra_params$type_measure) && is.character(extra_params$type_measure)) {
    extra_params$type_measure
  } else if (task == "classification") {
    if (!is.null(cm_arg) && cm_arg == "auc") "auc" else "deviance"
  } else if (task == "regression") {
    if (!is.null(cm_arg) && cm_arg %in% c("mae", "cal_mae")) "mae" else "mse"
  } else if (task == "multiclass") {
    if (!is.null(cm_arg) && cm_arg %in% c("accuracy", "class")) "class" else "deviance"
  } else {
    "default"
  }

  y_fit <- if (task == "regression") as.numeric(y_train) else as.factor(y_train)
  y_levels <- if (task == "regression") NULL else levels(y_fit)
  eff_nc <- if (task == "multiclass") {
    max(length(y_levels), if (!is.null(num_class)) as.integer(num_class) else 0L)
  } else {
    NULL
  }

  offs <- .resolve_offset_args(
    extra_params,
    n_train = nrow(x_mat),
    n_val = if (!is.null(val_mat)) nrow(val_mat) else NULL,
    task = task,
    num_class = eff_nc,
    target_levels = y_levels
  )
  off_tr <- offs$train
  off_va <- offs$val
  uses_offset <- !is.null(off_tr)
  if (uses_offset && task == "multiclass") {
    if (ncol(off_tr) != length(y_levels)) {
      stop(sprintf("Multiclass offset columns (%d) must match observed classes (%d).", ncol(off_tr), length(y_levels)))
    }
    colnames(off_tr) <- y_levels
    if (!is.null(off_va)) colnames(off_va) <- y_levels
  }
  if (uses_offset && !is.null(val_mat) && is.null(off_va)) {
    off_va <- if (task == "multiclass") {
      m <- matrix(0.0, nrow = nrow(val_mat), ncol = length(y_levels))
      colnames(m) <- y_levels
      m
    } else {
      numeric(nrow(val_mat))
    }
  }

  # Degeneracy checks: single-class targets, constant regression targets, or 0 features
  u_train <- unique(y_train[!is.na(y_train)])
  is_const_reg <- (task == "regression") && (length(unique(y_train[is.finite(y_train)])) < 2L)
  is_single_cls <- (task %in% c("classification", "multiclass")) && (length(u_train) < 2L)
  if (is_single_cls || is_const_reg || ncol(x_mat) == 0L) {
    single_class <- if (length(u_train) >= 1L) u_train[1] else 0
    reg_const <- if (any(is.finite(y_train))) mean(y_train[is.finite(y_train)]) else 0.0
    model <- list(
      is_degenerate = TRUE,
      degenerate_class = single_class,
      degenerate_reg = reg_const,
      task = task,
      col_means = col_means,
      s = s_choice,
      num_class = eff_nc,
      uses_offset = uses_offset,
      offset_levels = y_levels,
      cat_schema = cs,
      is_sparse = is_sparse,
      call = call("glmnet", alpha = alpha)
    )
    preds <- NULL
    if (!is.null(val_mat)) {
      if (task == "regression") {
        preds <- rep(reg_const, nrow(val_mat))
        if (uses_offset) preds <- preds + as.numeric(off_va)
      } else if (task == "classification") {
        pred_val <- if (as.character(single_class) %in% c("1", "true", "TRUE")) 1.0 else 0.0
        preds <- rep(pred_val, nrow(val_mat))
      } else {
        k <- if (!is.null(eff_nc) && eff_nc > 0L) eff_nc else 2L
        preds <- matrix(0.0, nrow = nrow(val_mat), ncol = k)
        cls_idx <- suppressWarnings(as.integer(as.character(single_class)))
        col_idx <- if (!is.na(cls_idx) && cls_idx >= 1L && cls_idx <= k) cls_idx else 1L
        preds[, col_idx] <- 1.0
      }
    }
    importances <- if (!is.null(all_feats)) stats::setNames(rep(0, length(all_feats)), all_feats) else numeric(0L)
    return(list(model = model, predictions = preds, importances = importances, col_means = col_means))
  }

  is_single_col <- (ncol(x_mat) == 1L)
  fit_mat <- if (is_single_col) cbind(x_mat, .__dummy__ = 0.0) else x_mat
  if (is_single_col && !is.null(val_mat)) val_mat <- cbind(val_mat, .__dummy__ = 0.0)

  family_str <- switch(task,
    regression = "gaussian",
    classification = "binomial",
    multiclass = "multinomial",
    stop(sprintf("Unsupported task '%s' for glmnet.", task))
  )

  preds <- NULL
  best_lambda <- NULL
  s_extract <- s_choice

  if (!is.null(lambda_val) && !isTRUE(relax)) {
    # Case A: scalar lambda supplied (e.g. MBO trial) -> single glmnet fit, zero CV
    model <- glmnet::glmnet(
      x = fit_mat,
      y = y_fit,
      family = family_str,
      alpha = alpha,
      lambda = lambda_val,
      offset = off_tr,
      standardize = standardize
    )
    model$call <- call("glmnet", alpha = alpha)
    best_lambda <- lambda_val
    s_extract <- best_lambda
    if (!is.null(val_mat)) {
      if (task == "regression") {
        preds <- as.numeric(stats::predict(model, newx = val_mat, s = best_lambda, newoffset = off_va))
      } else if (task == "classification") {
        preds <- as.numeric(stats::predict(model, newx = val_mat, s = best_lambda, type = "response", newoffset = off_va))
      } else {
        p_array <- stats::predict(model, newx = val_mat, s = best_lambda, type = "response", newoffset = off_va)
        preds <- matrix(p_array[, , 1L], nrow = nrow(val_mat), ncol = dim(p_array)[2L])
        if (!is.null(dimnames(p_array)[[2L]])) colnames(preds) <- dimnames(p_array)[[2L]]
      }
    }
  } else if (is.null(lambda_val) && !is.null(val_mat) && !is.null(y_val) && !use_cv && !isTRUE(relax)) {
    # Case B: validation set provided -> single glmnet path, score once on val_mat, keep winning lambda
    fit <- glmnet::glmnet(
      x = fit_mat,
      y = y_fit,
      family = family_str,
      alpha = alpha,
      offset = off_tr,
      standardize = standardize
    )
    p_all <- stats::predict(fit, newx = val_mat, type = "response", newoffset = off_va)
    best_idx <- .select_glmnet_val_lambda(
      p_all = p_all,
      lambdas = fit$lambda,
      y_val = y_val,
      y_levels = y_levels,
      task = task,
      metric_arg = metric_arg,
      cm_arg = cm_arg,
      s_choice = s_choice,
      num_class = eff_nc,
      threads = threads
    )
    best_lambda <- fit$lambda[best_idx]
    s_extract <- best_lambda
    if (task == "multiclass") {
      preds <- matrix(p_all[, , best_idx], nrow = nrow(val_mat), ncol = dim(p_all)[2L])
      if (!is.null(dimnames(p_all)[[2L]])) colnames(preds) <- dimnames(p_all)[[2L]]
    } else {
      preds <- as.numeric(p_all[, best_idx])
    }
    model <- .trim_glmnet_model(fit, best_idx, alpha)
  } else {
    # Case C: no validation set (or use_cv / relax requested) -> cv.glmnet fallback
    nfolds <- min(5L, nrow(fit_mat))
    if (!is.null(extra_params$nfolds)) {
      nfolds <- min(as.integer(extra_params$nfolds), nrow(fit_mat))
    }
    if (nfolds < 3L) {
      fallback_lam <- if (!is.null(lambda_val)) lambda_val else 0.01
      model <- glmnet::glmnet(
        x = fit_mat,
        y = y_fit,
        family = family_str,
        alpha = alpha,
        lambda = fallback_lam,
        offset = off_tr,
        standardize = standardize
      )
      model$call <- call("glmnet", alpha = alpha)
      best_lambda <- fallback_lam
      s_extract <- best_lambda
    } else {
      model <- if (isTRUE(relax)) {
        glmnet::cv.glmnet(
          x = fit_mat,
          y = y_fit,
          family = family_str,
          alpha = alpha,
          nfolds = nfolds,
          type.measure = type_measure,
          offset = off_tr,
          standardize = standardize,
          relax = TRUE
        )
      } else {
        glmnet::cv.glmnet(
          x = fit_mat,
          y = y_fit,
          family = family_str,
          alpha = alpha,
          nfolds = nfolds,
          type.measure = type_measure,
          offset = off_tr,
          standardize = standardize
        )
      }
      model$fit.preval <- NULL
      model$call <- call("cv.glmnet", alpha = alpha)
      if (!is.null(model$glmnet.fit)) model$glmnet.fit$call <- call("glmnet", alpha = alpha)
      best_lambda <- if (is.character(s_choice) && !is.null(model[[s_choice]])) {
        as.numeric(model[[s_choice]])
      } else if (is.numeric(s_choice)) {
        as.numeric(s_choice)
      } else {
        model$lambda.min
      }
      s_extract <- s_choice
    }

    if (!is.null(val_mat)) {
      if (task == "regression") {
        preds <- as.numeric(stats::predict(model, newx = val_mat, s = s_extract, newoffset = off_va))
      } else if (task == "classification") {
        preds <- as.numeric(stats::predict(model, newx = val_mat, s = s_extract, type = "response", newoffset = off_va))
      } else {
        p_array <- stats::predict(model, newx = val_mat, s = s_extract, type = "response", newoffset = off_va)
        preds <- matrix(p_array[, , 1L], nrow = nrow(val_mat), ncol = dim(p_array)[2L])
        if (!is.null(dimnames(p_array)[[2L]])) colnames(preds) <- dimnames(p_array)[[2L]]
      }
    }
  }

  if (task == "multiclass") {
    coefs_list <- stats::coef(model, s = s_extract)
    imp_matrix <- do.call(cbind, lapply(coefs_list, function(c_mat) abs(as.numeric(c_mat)[-1L])))
    imp <- rowMeans(imp_matrix)
    names(imp) <- rownames(coefs_list[[1L]])[-1L]
  } else {
    coefs <- as.matrix(stats::coef(model, s = s_extract))
    coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
    imp <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
  }
  if (is_single_col) {
    imp <- imp[seq_len(ncol(x_mat))]
  }
  importances <- imp * sd_x[seq_along(imp)]
  if (!is.null(all_feats)) names(importances) <- all_feats

  if (!is.null(all_feats)) {
    missing <- setdiff(all_feats, names(importances))
    if (length(missing) > 0L) {
      importances <- c(importances, stats::setNames(rep(0, length(missing)), missing))
    }
  }

  model$col_means <- col_means
  model$s <- s_choice
  model$best_lambda <- best_lambda
  model$uses_offset <- uses_offset
  model$offset_levels <- y_levels
  model$num_class <- eff_nc
  model$cat_schema <- cs
  model$is_sparse <- is_sparse

  list(model = model, predictions = preds, importances = importances,
       col_means = col_means, best_lambda = best_lambda)
}

.glmnet_predict_core <- function(model, x_new, task, ...) {
  if (is.null(x_new)) return(numeric(0L))
  is_sparse <- isTRUE(model$is_sparse) || inherits(x_new, "sparseMatrix")
  if (is_sparse) {
    x_mat <- x_new
    if (nrow(x_mat) == 0L || ncol(x_mat) == 0L) return(numeric(0L))
  } else {
    x_mat <- .sanitize_feature_matrix(x_new, cat_schema = model$cat_schema)
    if (is.null(x_mat) || length(x_mat) == 0L) return(numeric(0L))
  }

  extra_args <- list(...)
  n_new <- nrow(x_mat)

  off_new <- NULL
  if (isTRUE(model$uses_offset)) {
    off_new <- .resolve_offset_args(
      extra_args,
      n_new = n_new,
      task = task,
      num_class = model$num_class,
      target_levels = model$offset_levels
    )$new
    if (is.null(off_new)) {
      off_new <- if (identical(task, "multiclass")) {
        k <- if (!is.null(model$offset_levels)) length(model$offset_levels) else if (!is.null(model$num_class)) as.integer(model$num_class) else 2L
        matrix(0.0, nrow = n_new, ncol = k)
      } else {
        numeric(n_new)
      }
    }
    if (identical(task, "multiclass") && !is.null(model$offset_levels)) {
      colnames(off_new) <- model$offset_levels
    }
  }

  if (isTRUE(model$is_degenerate)) {
    if (task == "regression") {
      base_val <- if (!is.null(model$degenerate_reg)) model$degenerate_reg else 0.0
      out <- rep(base_val, n_new)
      return(if (!is.null(off_new)) out + as.numeric(off_new) else out)
    } else if (task == "classification") {
      pred_val <- if (as.character(model$degenerate_class) %in% c("1", "true", "TRUE")) 1.0 else 0.0
      return(rep(pred_val, n_new))
    } else if (task == "multiclass") {
      k <- if (!is.null(model$num_class)) as.integer(model$num_class) else 2L
      out_mat <- matrix(0.0, nrow = n_new, ncol = k)
      cls_idx <- suppressWarnings(as.integer(as.character(model$degenerate_class)))
      col_idx <- if (!is.na(cls_idx) && cls_idx >= 1L && cls_idx <= k) cls_idx else 1L
      out_mat[, col_idx] <- 1.0
      return(out_mat)
    }
  }

  if (!is_sparse) {
    col_means <- if (!is.null(model$col_means) && length(model$col_means) == ncol(x_mat)) {
      if (!is.null(colnames(x_mat)) && !is.null(names(model$col_means)) &&
          all(colnames(x_mat) %in% names(model$col_means))) {
        model$col_means[colnames(x_mat)]
      } else {
        model$col_means
      }
    } else {
      NULL
    }
    x_mat <- .impute_and_col_stats(x_mat, means = col_means, compute_sd = FALSE)$mat
  }

  s_choice <- if (!is.null(extra_args$lm_s) && (is.character(extra_args$lm_s) || (is.numeric(extra_args$lm_s) && is.finite(extra_args$lm_s)))) {
    extra_args$lm_s
  } else if (!is.null(extra_args$s) && (is.character(extra_args$s) || (is.numeric(extra_args$s) && is.finite(extra_args$s)))) {
    extra_args$s
  } else if (!is.null(model$s)) {
    model$s
  } else {
    "lambda.min"
  }

  s_eval <- if (!inherits(model, "cv.glmnet") && is.character(s_choice)) {
    if (!is.null(model$best_lambda) && is.finite(model$best_lambda)) {
      model$best_lambda
    } else {
      model$lambda[1L]
    }
  } else {
    s_choice
  }

  pred_mat <- if (ncol(x_mat) == 1L) cbind(x_mat, .__dummy__ = 0.0) else x_mat
  if (task == "regression") {
    p <- if (isTRUE(model$uses_offset)) {
      stats::predict(model, newx = pred_mat, s = s_eval, newoffset = off_new)
    } else {
      stats::predict(model, newx = pred_mat, s = s_eval)
    }
    as.numeric(p)
  } else if (task == "classification") {
    p <- if (isTRUE(model$uses_offset)) {
      stats::predict(model, newx = pred_mat, s = s_eval, type = "response", newoffset = off_new)
    } else {
      stats::predict(model, newx = pred_mat, s = s_eval, type = "response")
    }
    as.numeric(p)
  } else if (task == "multiclass") {
    p_array <- if (isTRUE(model$uses_offset)) {
      stats::predict(model, newx = pred_mat, s = s_eval, type = "response", newoffset = off_new)
    } else {
      stats::predict(model, newx = pred_mat, s = s_eval, type = "response")
    }
    preds <- matrix(p_array[, , 1L], nrow = n_new, ncol = dim(p_array)[2L])
    if (!is.null(dimnames(p_array)[[2L]])) {
      colnames(preds) <- dimnames(p_array)[[2L]]
    }
    preds
  }
}

register_evaluator(
  "lm",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    .glmnet_fit_core(
      x_train = x_train, y_train = y_train, x_val = x_val,
      task = task, threads = threads, num_class = num_class, nrounds = nrounds,
      ...
    )
  },
  predict_func = function(model, x_new, task, ...) {
    .glmnet_predict_core(model = model, x_new = x_new, task = task, ...)
  },
  traits = list(is_tree = FALSE, iteration_param = NULL, scale_iterations_with_data = FALSE)
)

# 5. Keras 3 Feed-Forward Neural Network Evaluator
register_evaluator(
  "keras3",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("keras3", quietly = TRUE)) {
      stop("The 'keras3' package is required for the keras3 evaluator.")
    }
    
    # Extract tunable parameters from ... with defaults
    extra_params <- list(...)
    seed_val <- if (!is.null(extra_params$seed) && is.numeric(extra_params$seed) && is.finite(extra_params$seed)) {
      as.integer(extra_params$seed)
    } else {
      42L
    }

    # Ensure deterministic weight initialization for stable fitness evaluations,
    # but restore the R global random state so we don't freeze the genetic algorithm!
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
    keras3::set_random_seed(seed_val)
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else {
      rm(".Random.seed", envir = .GlobalEnv)
    }
    if (!is.null(extra_params$nrounds)) nrounds <- as.integer(extra_params$nrounds)
    if (!is.null(extra_params$num_rounds)) nrounds <- as.integer(extra_params$num_rounds)
    if (!is.null(extra_params$n_rounds)) nrounds <- as.integer(extra_params$n_rounds)
    if (!is.null(extra_params$num_round)) nrounds <- as.integer(extra_params$num_round)
    if (!is.null(extra_params$nround)) nrounds <- as.integer(extra_params$nround)
    if (!is.null(extra_params$epochs)) nrounds <- as.integer(extra_params$epochs)
    if (!is.null(extra_params$n_epochs)) nrounds <- as.integer(extra_params$n_epochs)
    if (!is.null(extra_params$iterations)) nrounds <- as.integer(extra_params$iterations)
    if (!is.null(extra_params$n_iterations)) nrounds <- as.integer(extra_params$n_iterations)
    nn_layers <- if (!is.null(extra_params$nn_layers)) extra_params$nn_layers else 2
    nn_units <- if (!is.null(extra_params$nn_units)) extra_params$nn_units else 64
    nn_dropout <- if (!is.null(extra_params$nn_dropout)) extra_params$nn_dropout else 0.2
    nn_lr <- if (!is.null(extra_params$nn_learning_rate)) extra_params$nn_learning_rate else 0.001
    
    # Ensure data is matrix
    x_train <- as.matrix(x_train)
    
    # Define model architecture
    model <- keras3::keras_model_sequential(input_shape = ncol(x_train))
    model <- keras3::layer_dense(model, units = nn_units, activation = "relu")
    if (nn_dropout > 0) model <- keras3::layer_dropout(model, rate = nn_dropout)
    
    if (nn_layers > 1) {
      for (i in 2:nn_layers) {
        # Reduce units in subsequent layers, bounded at 16
        units_next <- max(16, as.integer(nn_units / (2 ^ (i - 1))))
        model <- keras3::layer_dense(model, units = units_next, activation = "relu")
        if (nn_dropout > 0) model <- keras3::layer_dropout(model, rate = nn_dropout)
      }
    }
    
    # Task-specific output layer and compilation
    optimizer <- keras3::optimizer_adam(learning_rate = nn_lr)
    
    if (task == "regression") {
      model <- keras3::layer_dense(model, units = 1)
      keras3::compile(model, optimizer = optimizer, loss = "mse")
    } else if (task == "classification") {
      model <- keras3::layer_dense(model, units = 1, activation = "sigmoid")
      keras3::compile(model, optimizer = optimizer, loss = "binary_crossentropy")
    } else if (task == "multiclass") {
      model <- keras3::layer_dense(model, units = num_class, activation = "softmax")
      # Note: expects y_train to be integer 0 to num_class-1
      keras3::compile(model, optimizer = optimizer, loss = "sparse_categorical_crossentropy")
    }
    
    # Prepare validation data if provided
    y_val <- extra_params$y_val
    early_stopping_rounds <- extra_params$early_stopping_rounds
    val_data <- NULL
    callbacks <- list()
    if (!is.null(x_val) && !is.null(y_val)) {
      val_data <- list(as.matrix(x_val), y_val)
      if (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) {
        callbacks <- list(
          keras3::callback_early_stopping(
            monitor = "val_loss",
            patience = early_stopping_rounds,
            restore_best_weights = TRUE
          )
        )
      }
    }
    
    # Train
    history <- keras3::fit(
      model,
      x = x_train,
      y = y_train,
      epochs = nrounds,
      validation_data = val_data,
      callbacks = callbacks,
      verbose = 0
    )
    
    # Predict on validation
    preds <- NULL
    if (!is.null(x_val)) {
      x_val_mat <- as.matrix(x_val)
      if (task == "regression" || task == "classification") {
        preds <- as.numeric(stats::predict(model, x_val_mat, verbose = 0))
      } else if (task == "multiclass") {
        # Return probability matrix
        preds <- as.matrix(stats::predict(model, x_val_mat, verbose = 0))
      }
    }
    
    # NNs don't have native feature importances, return NULL
    list(model = model, predictions = preds, importances = NULL)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("keras3", quietly = TRUE)) {
      stop("The 'keras3' package is required for the keras3 evaluator.")
    }
    x_mat <- as.matrix(x_new)
    if (task == "regression" || task == "classification") {
      as.numeric(stats::predict(model, x_mat, verbose = 0))
    } else if (task == "multiclass") {
      as.matrix(stats::predict(model, x_mat, verbose = 0))
    }
  },
  cleanup_func = function(model) {
    if (requireNamespace("keras3", quietly = TRUE)) {
      if (exists("clear_session", where = asNamespace("keras3"))) {
        keras3::clear_session()
      } else if (exists("k_clear_session", where = asNamespace("keras3"))) {
        # Fallback for some versions
        tryCatch(keras3::k_clear_session(), error = function(e) NULL)
      }
    }
  },
  traits = list(is_tree = FALSE, iteration_param = "epochs", scale_iterations_with_data = FALSE)
)

# --- RealMLP helpers (keep train_func concise) ---

#' Compute column medians and impute NA/non-finite values in a matrix
#' @param mat Numeric matrix to impute.
#' @param col_meds Precomputed column medians (computed from training set).
#' @return A list with `mat` (imputed matrix) and `col_meds`.
#' @keywords internal
#' @noRd
.realmlp_impute_matrix <- function(mat, col_meds = NULL) {
  if (is.null(mat) || !anyNA(mat)) return(list(mat = mat, col_meds = col_meds))
  if (is.null(col_meds)) {
    col_meds <- vapply(seq_len(ncol(mat)), function(j) {
      col <- mat[, j]
      if (!anyNA(col)) return(stats::median(col))
      fin <- col[!is.na(col)]
      if (length(fin) == 0L) 0.0 else stats::median(fin)
    }, numeric(1))
  }
  for (j in seq_len(ncol(mat))) {
    col <- mat[, j]
    if (anyNA(col)) {
      mat[is.na(col), j] <- if (j <= length(col_meds)) col_meds[j] else 0.0
    }
  }
  list(mat = mat, col_meds = col_meds)
}

#' Encode target variable to numeric for the C++ engine
#' @param y Target vector (factor, numeric, or character).
#' @param task One of "regression", "classification", "multiclass".
#' @param num_class Optional integer number of classes.
#' @param levels_target Optional vector of target class levels.
#' @return A list with `y_num`, `levels_target`, and `num_classes_int`.
#' @keywords internal
#' @noRd
.realmlp_prepare_target <- function(y, task, num_class = NULL, levels_target = NULL) {
  num_classes_int <- 0L
  if (task == "regression") {
    y_num <- as.numeric(y)
  } else if (task == "classification") {
    y_bin <- encode_binary_target(y)
    levels_target <- if (!is.null(levels_target)) levels_target else c("0", "1")
    y_num <- as.numeric(y_bin)
  } else if (task == "multiclass") {
    if (!is.null(levels_target)) {
      # Use provided levels_target
    } else if (is.factor(y)) {
      levels_target <- levels(y)
    } else if (is.character(y)) {
      levels_target <- sort(unique(stats::na.omit(y)))
    } else if (is.numeric(y)) {
      u_vals <- sort(unique(stats::na.omit(y)))
      if (!is.null(num_class) && all(u_vals %in% seq(0, num_class - 1))) {
        levels_target <- seq(0, num_class - 1)
      } else if (!is.null(num_class) && all(u_vals %in% seq(1, num_class))) {
        levels_target <- seq(1, num_class)
      } else {
        levels_target <- if (!is.null(num_class)) seq(0, num_class - 1) else u_vals
      }
    } else {
      levels_target <- if (!is.null(num_class)) seq(0, num_class - 1) else sort(unique(y))
    }
    num_classes_int <- as.integer(length(levels_target))
    if (!is.null(num_class) && num_classes_int < as.integer(num_class)) {
      num_classes_int <- as.integer(num_class)
    }
    y_num <- as.numeric(factor(y, levels = levels_target)) - 1.0
  } else {
    stop(sprintf("Unsupported task '%s' for RealMLP evaluator.", task))
  }
  list(y_num = y_num, levels_target = levels_target, num_classes_int = num_classes_int)
}

# 6. RealMLP Evaluator (Native C++ with PBLD Embeddings)
register_evaluator(
  "realmlp",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    extra_params <- list(...)
    if (!is.null(extra_params$threads)) threads <- as.integer(extra_params$threads)
    if (!is.null(extra_params$nthreads)) threads <- as.integer(extra_params$nthreads)
    if (!is.null(extra_params$nthread)) threads <- as.integer(extra_params$nthread)
    if (!is.null(extra_params$num_threads)) threads <- as.integer(extra_params$num_threads)
    if (!is.null(extra_params$n_jobs)) threads <- as.integer(extra_params$n_jobs)
    if (!is.null(extra_params$nrounds)) nrounds <- as.integer(extra_params$nrounds)
    if (!is.null(extra_params$num_rounds)) nrounds <- as.integer(extra_params$num_rounds)
    if (!is.null(extra_params$n_rounds)) nrounds <- as.integer(extra_params$n_rounds)
    if (!is.null(extra_params$num_round)) nrounds <- as.integer(extra_params$num_round)
    if (!is.null(extra_params$nround)) nrounds <- as.integer(extra_params$nround)
    if (!is.null(extra_params$iterations)) nrounds <- as.integer(extra_params$iterations)
    if (!is.null(extra_params$n_iterations)) nrounds <- as.integer(extra_params$n_iterations)
    if (!is.null(extra_params$epochs)) nrounds <- as.integer(extra_params$epochs)
    if (!is.null(extra_params$n_epochs)) nrounds <- as.integer(extra_params$n_epochs)
    if (!is.null(extra_params$realmlp_epochs)) nrounds <- as.integer(extra_params$realmlp_epochs)
    y_val <- extra_params$y_val
    early_stopping_rounds <- if (!is.null(extra_params$early_stopping_rounds)) as.integer(extra_params$early_stopping_rounds) else 0L

    seed_val <- if (!is.null(extra_params$realmlp_seed)) {
      as.integer(extra_params$realmlp_seed)
    } else if (!is.null(extra_params$seed)) {
      as.integer(extra_params$seed)
    } else {
      42L
    }
    n_epochs <- as.integer(nrounds)
    batch_size <- if (!is.null(extra_params$batch_size)) as.integer(extra_params$batch_size) else -1L
    lr_val <- if (!is.null(extra_params$realmlp_lr)) as.numeric(extra_params$realmlp_lr) else if (!is.null(extra_params$lr)) as.numeric(extra_params$lr) else -1.0

    verbose_int <- .nn_verbose_level(
      extra_params,
      learner_opt = "evoFE.realmlp_verbose",
      param_keys = c("realmlp_verbose", "nn_verbose")
    )
    show_log <- verbose_int >= 1L
    is_detail <- verbose_int >= 2L

    use_es <- early_stopping_rounds > 0L && !is.null(x_val) && !is.null(y_val)
    es_rounds_to_pass <- if (use_es) early_stopping_rounds else 0L
    stop_epoch_val <- if (!use_es && !is.null(extra_params$stop_epoch) &&
                          is.numeric(extra_params$stop_epoch) && extra_params$stop_epoch > 0) {
      as.integer(min(extra_params$stop_epoch, n_epochs))
    } else {
      0L
    }

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    x_val   <- .sanitize_feature_matrix(x_val, threads = threads)

    imp <- .realmlp_impute_matrix(x_train)
    x_train  <- imp$mat
    col_meds <- imp$col_meds
    x_val <- .realmlp_impute_matrix(x_val, col_meds)$mat

    t0 <- Sys.time()

    target <- .realmlp_prepare_target(y_train, task, num_class)
    y_train_num   <- target$y_num
    levels_target <- target$levels_target
    y_val_num <- if (!is.null(y_val)) .realmlp_prepare_target(y_val, task, num_class, levels_target = target$levels_target)$y_num else NULL

    metric_arg <- if (!is.null(extra_params$metric)) as.character(extra_params$metric) else "default"
    hidden_dim_val <- if (!is.null(extra_params$hidden_dim)) as.integer(extra_params$hidden_dim) else 256L
    n_cycles_val <- if (!is.null(extra_params$realmlp_n_cycles)) {
      as.integer(extra_params$realmlp_n_cycles)
    } else if (!is.null(extra_params$n_cycles)) {
      if (identical(extra_params$n_cycles, "auto")) 0L else as.integer(extra_params$n_cycles)
    } else {
      4L
    }

    fit_res <- rcpp_realmlp_train(
      x_train = x_train, y_train = y_train_num,
      x_val = x_val,
      y_val = if (use_es) y_val_num else NULL,
      task = task, n_epochs = n_epochs, batch_size = batch_size, lr = lr_val,
      early_stopping_rounds = es_rounds_to_pass, seed = seed_val,
      verbose = verbose_int, num_classes = target$num_classes_int,
      threads = as.integer(threads), metric = metric_arg,
      hidden_dim = hidden_dim_val,
      n_cycles = n_cycles_val,
      stop_epoch = stop_epoch_val
    )

    preds <- NULL
    if (!is.null(x_val)) {
      raw_preds <- if (!is.null(fit_res$val_preds)) {
        fit_res$val_preds
      } else {
        rcpp_realmlp_predict(fit_res$model_state, x_val, threads = as.integer(threads))
      }
      if (task == "multiclass") {
        probs <- as.matrix(raw_preds)
        colnames(probs) <- as.character(levels_target)
        preds <- probs
      } else {
        preds <- as.numeric(raw_preds)
      }
    }

    best_it <- if (use_es && !is.null(fit_res$best_epoch) && fit_res$best_epoch > 0) {
      as.integer(fit_res$best_epoch)
    } else if (use_es && !is.null(fit_res$stopped_epoch) && fit_res$stopped_epoch > 0) {
      as.integer(fit_res$stopped_epoch)
    } else if (stop_epoch_val > 0L) {
      stop_epoch_val
    } else {
      NULL
    }

    if (show_log) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      es_status <- if (isTRUE(fit_res$stopped_early)) {
        sprintf("stopped at epoch %d, best epoch %d (patience=%d)", fit_res$stopped_epoch, fit_res$best_epoch, early_stopping_rounds)
      } else if (use_es) {
        sprintf("best epoch %d (patience=%d)", fit_res$best_epoch, early_stopping_rounds)
      } else if (stop_epoch_val > 0L && stop_epoch_val < n_epochs) {
        sprintf("target best epoch %d/%d", stop_epoch_val, n_epochs)
      } else {
        "OFF"
      }
      msg <- sprintf("    [RealMLP C++ %s] Fitted %d rows (Early stopping: %s, %.3fs)",
                     task, nrow(x_train), es_status, elapsed)
      if (is_detail) msg <- sprintf("%s [Features: %d]", msg, ncol(x_train))
      message(msg)
    }

    importances <- as.numeric(fit_res$importances)
    if (length(importances) == ncol(x_train)) {
      names(importances) <- colnames(x_train)
    } else {
      importances <- NULL
    }

    wrapped_model <- list(
      model_state = fit_res$model_state,
      col_meds = col_meds,
      levels_target = levels_target,
      best_iteration = best_it,
      best_epoch = best_it,
      n_epochs = as.integer(n_epochs),
      n_cycles = as.integer(fit_res$n_cycles),
      base_lr = as.numeric(fit_res$base_lr),
      threads = as.integer(threads)
    )

    list(model = wrapped_model, predictions = preds, importances = importances,
         best_iteration = best_it, best_epoch = best_it)
  },

  predict_func = function(model, x_new, task, ...) {
    x_new <- .sanitize_feature_matrix(x_new)

    if (anyNA(x_new)) {
      col_meds <- model$col_meds
      if (is.null(col_meds) && !is.null(model$model_state)) {
        col_meds <- model$model_state$x_mean
      }
      if (!is.null(col_meds)) {
        for (j in seq_len(ncol(x_new))) {
          if (j <= length(col_meds)) {
            col <- x_new[, j]
            if (anyNA(col)) x_new[is.na(col), j] <- col_meds[j]
          }
        }
      }
    }

    extra_args <- list(...)
    th <- resolve_param_aliases(extra_args)$threads
    n_threads <- if (!is.null(th)) {
      as.integer(th)
    } else if (!is.null(model$threads)) {
      as.integer(model$threads)
    } else {
      default_threads()
    }
    raw_preds <- rcpp_realmlp_predict(model$model_state, x_new, threads = n_threads)

    if (task == "regression") {
      as.numeric(raw_preds)
    } else if (task == "classification") {
      as.numeric(raw_preds)
    } else if (task == "multiclass") {
      probs <- as.matrix(raw_preds)
      if (!is.null(model$levels_target)) {
        colnames(probs) <- as.character(model$levels_target)
      }
      probs
    }
  },

  cleanup_func = NULL,
  traits = list(is_tree = FALSE, iteration_param = "epochs", scale_iterations_with_data = FALSE)
)

# Shared parameter alias helper for new evaluators
.pick_first_param <- function(extra_params, keys, default = NULL) {
  for (k in keys) {
    v <- extra_params[[k]]
    if (!is.null(v) && length(v) > 0L) return(v)
  }
  default
}

# Map expanded basis/OHE importances back to original input columns
.aggregate_grouped_importances <- function(raw_imp, col_group, p_in, in_names = NULL) {
  imp_orig <- numeric(p_in)
  if (!is.null(raw_imp) && length(raw_imp) == length(col_group) && length(raw_imp) > 0L) {
    rs <- rowsum(abs(as.numeric(raw_imp)), group = col_group, reorder = FALSE)
    idx_g <- suppressWarnings(as.integer(rownames(rs)))
    valid_g <- !is.na(idx_g) & idx_g >= 1L & idx_g <= p_in
    imp_orig[idx_g[valid_g]] <- as.numeric(rs[valid_g, 1L])
  }
  if (!is.null(in_names) && length(in_names) == p_in) {
    names(imp_orig) <- in_names
  }
  imp_orig
}

# 7. Spline-Basis Penalized Linear Evaluator (Python-free, sparse Matrix + glmnet + native offsets)
register_evaluator(
  "spline_lm",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    extra_params <- list(...)
    th <- .pick_first_param(extra_params, c("threads", "nthreads", "nthread", "num_threads", "n_jobs"), threads)
    threads <- as.integer(th)

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)

    n_knots <- as.integer(.pick_first_param(extra_params, "n_knots", 5L))
    degree <- as.integer(.pick_first_param(extra_params, "degree", 2L))
    include_linear <- .parse_bool_flag(.pick_first_param(extra_params, "include_linear", TRUE), default = TRUE)
    max_cat_levels <- as.integer(.pick_first_param(extra_params, "max_cat_levels", 64L))

    sp_tr <- .build_sparse_spline_design(
      x_train,
      cat_schema = cs,
      n_knots = n_knots,
      degree = degree,
      include_linear = include_linear,
      max_cat_levels = max_cat_levels
    )
    sp_va <- if (!is.null(x_val)) {
      .build_sparse_spline_design(x_val, state = sp_tr$state)
    } else {
      NULL
    }

    fit_res <- .glmnet_fit_core(
      x_train = sp_tr$matrix,
      y_train = y_train,
      x_val = sp_va,
      task = task,
      threads = threads,
      num_class = num_class,
      nrounds = nrounds,
      ...
    )

    importances <- .aggregate_grouped_importances(
      fit_res$importances,
      sp_tr$state$col_group,
      ncol(x_train),
      colnames(x_train)
    )

    wrapped <- list(
      glmnet_model = fit_res$model,
      spline_state = sp_tr$state,
      cat_schema = cs
    )
    list(
      model = wrapped,
      predictions = fit_res$predictions,
      importances = importances,
      best_lambda = fit_res$best_lambda
    )
  },
  predict_func = function(model, x_new, task, ...) {
    x_san <- .sanitize_feature_matrix(x_new, cat_schema = model$cat_schema)
    x_sp <- .build_sparse_spline_design(x_san, state = model$spline_state)
    .glmnet_predict_core(model = model$glmnet_model, x_new = x_sp, task = task, ...)
  },
  traits = list(is_tree = FALSE, iteration_param = NULL, scale_iterations_with_data = FALSE)
)

# 8. Random Fourier Features / Nystrom Kernel Evaluator (Native C++17)
register_evaluator(
  "rff",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    fn_train <- get0("rcpp_rff_train", mode = "function")
    fn_predict <- get0("rcpp_rff_predict", mode = "function")
    if (is.null(fn_train) || is.null(fn_predict)) {
      stop("Native C++ entrypoints 'rcpp_rff_train'/'rcpp_rff_predict' are not available in this build.")
    }

    extra_params <- list(...)
    th <- .pick_first_param(extra_params, c("threads", "nthreads", "nthread", "num_threads", "n_jobs"), threads)
    threads <- max(1L, as.integer(th))

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)

    max_cat_levels <- as.integer(.pick_first_param(extra_params, "max_cat_levels", 32L))
    ohe_tr <- .build_categorical_ohe_or_matrix(
      x_train,
      cat_schema = cs,
      max_cat_levels = max_cat_levels,
      threads = threads
    )
    x_tr_num <- ohe_tr$matrix
    x_va_num <- if (!is.null(x_val)) {
      .build_categorical_ohe_or_matrix(x_val, state = ohe_tr$state, threads = threads)
    } else {
      NULL
    }

    target <- .realmlp_prepare_target(y_train, task, num_class)
    y_val_num <- if (!is.null(extra_params$y_val)) {
      .realmlp_prepare_target(extra_params$y_val, task, num_class, levels_target = target$levels_target)$y_num
    } else {
      NULL
    }

    sigma_val <- if (!is.null(extra_params$sigma) && is.numeric(extra_params$sigma) && is.finite(extra_params$sigma)) {
      as.numeric(extra_params$sigma)
    } else if (!is.null(extra_params$gamma) && is.numeric(extra_params$gamma) &&
               is.finite(extra_params$gamma) && extra_params$gamma > 0) {
      1.0 / sqrt(2.0 * as.numeric(extra_params$gamma))
    } else {
      -1.0
    }
    lambda_val <- as.numeric(.pick_first_param(extra_params, c("lambda", "l2_reg"), 1e-2))
    n_features <- max(16L, as.integer(.pick_first_param(extra_params, c("n_features", "n_components"), 512L)))
    approx_method <- as.character(.pick_first_param(extra_params, c("approx_method", "feature_map"), "rff"))
    kernel_str <- as.character(.pick_first_param(extra_params, "kernel", "rbf"))
    max_irls_iter <- max(1L, as.integer(.pick_first_param(extra_params, "max_irls_iter", 25L)))
    irls_tol <- as.numeric(.pick_first_param(extra_params, "irls_tol", 1e-6))
    seed_val <- as.integer(.pick_first_param(extra_params, "seed", 42L))
    metric_arg <- as.character(.pick_first_param(extra_params, "metric", "default"))

    offs <- .resolve_offset_args(
      extra_params,
      n_train = nrow(x_tr_num),
      n_val = if (!is.null(x_va_num)) nrow(x_va_num) else NULL,
      task = task,
      num_class = if (task == "multiclass") target$num_classes_int else NULL,
      target_levels = target$levels_target
    )

    fit_res <- fn_train(
      x_train = x_tr_num,
      y_train = target$y_num,
      x_val = x_va_num,
      y_val = y_val_num,
      task = task,
      sigma = sigma_val,
      lambda = lambda_val,
      n_features = n_features,
      kernel = kernel_str,
      approx_method = approx_method,
      threads = threads,
      seed = seed_val,
      num_classes = target$num_classes_int,
      metric = metric_arg,
      offset_train = if (!is.null(offs$train)) as.numeric(offs$train) else NULL,
      offset_val = if (!is.null(offs$val)) as.numeric(offs$val) else NULL,
      max_irls_iter = max_irls_iter,
      irls_tol = irls_tol
    )

    preds <- NULL
    if (!is.null(x_va_num)) {
      raw_preds <- if (!is.null(fit_res$val_preds)) {
        fit_res$val_preds
      } else if (!is.null(fit_res$predictions)) {
        fit_res$predictions
      } else {
        fn_predict(
          fit_res$model_state,
          x_va_num,
          offset_new = if (!is.null(offs$val)) as.numeric(offs$val) else NULL,
          threads = threads
        )
      }
      if (task == "multiclass") {
        probs <- as.matrix(raw_preds)
        colnames(probs) <- as.character(target$levels_target)
        preds <- probs
      } else {
        preds <- as.numeric(raw_preds)
      }
    }

    importances <- .aggregate_grouped_importances(
      fit_res$importances,
      ohe_tr$col_group,
      ncol(x_train),
      colnames(x_train)
    )

    wrapped <- list(
      model_state = fit_res$model_state,
      ohe_state = ohe_tr$state,
      cat_schema = cs,
      levels_target = target$levels_target,
      num_class = target$num_classes_int,
      uses_offset = !is.null(offs$train),
      threads = threads
    )
    list(model = wrapped, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    fn_predict <- get0("rcpp_rff_predict", mode = "function")
    if (is.null(fn_predict)) {
      stop("Native C++ entrypoint 'rcpp_rff_predict' is not available in this build.")
    }
    extra_args <- list(...)
    th <- resolve_param_aliases(extra_args)$threads
    n_threads <- if (!is.null(th)) {
      as.integer(th)
    } else if (!is.null(model$threads)) {
      as.integer(model$threads)
    } else {
      default_threads()
    }
    x_san <- .sanitize_feature_matrix(x_new, threads = n_threads, cat_schema = model$cat_schema)
    x_num <- .build_categorical_ohe_or_matrix(x_san, state = model$ohe_state, threads = n_threads)
    off_new <- .resolve_offset_args(
      extra_args,
      n_new = nrow(x_num),
      task = task,
      num_class = if (identical(task, "multiclass")) model$num_class else NULL,
      target_levels = model$levels_target
    )$new
    raw_preds <- fn_predict(
      model$model_state,
      x_num,
      offset_new = if (!is.null(off_new)) as.numeric(off_new) else NULL,
      threads = n_threads
    )
    if (identical(task, "multiclass")) {
      probs <- as.matrix(raw_preds)
      if (!is.null(model$levels_target)) colnames(probs) <- as.character(model$levels_target)
      probs
    } else {
      as.numeric(raw_preds)
    }
  },
  traits = list(is_tree = FALSE, iteration_param = NULL, scale_iterations_with_data = FALSE)
)

# 9. TabM BatchEnsemble MLP + PLE Evaluator (Native C++17)
register_evaluator(
  "tabm",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    fn_train <- get0("rcpp_tabm_train", mode = "function")
    fn_predict <- get0("rcpp_tabm_predict", mode = "function")
    if (is.null(fn_train) || is.null(fn_predict)) {
      stop("Native C++ entrypoints 'rcpp_tabm_train'/'rcpp_tabm_predict' are not available in this build.")
    }

    extra_params <- list(...)
    th <- .pick_first_param(extra_params, c("threads", "nthreads", "nthread", "num_threads", "n_jobs"), threads)
    threads <- max(1L, as.integer(th))

    n_epochs <- as.integer(.pick_first_param(
      extra_params,
      c("tabm_epochs", "epochs", "n_epochs", "iterations", "n_iterations",
        "nrounds", "num_rounds", "n_rounds", "num_round", "nround"),
      nrounds
    ))
    n_epochs <- max(1L, n_epochs)

    lr_val <- as.numeric(.pick_first_param(extra_params, c("tabm_lr", "lr"), 0.002))
    k_ens <- max(1L, as.integer(.pick_first_param(extra_params, c("k_ensemble", "tabm_k"), 16L)))
    d_block <- max(8L, as.integer(.pick_first_param(extra_params, c("d_block", "tabm_hidden_dim", "hidden_dim"), 128L)))
    n_blocks <- max(1L, as.integer(.pick_first_param(extra_params, c("n_blocks", "tabm_n_blocks"), 2L)))
    dropout <- min(0.9, max(0.0, as.numeric(.pick_first_param(extra_params, c("dropout", "tabm_dropout"), 0.1))))
    weight_decay <- max(0.0, as.numeric(.pick_first_param(extra_params, c("weight_decay", "tabm_weight_decay"), 1e-4)))
    num_emb <- as.character(.pick_first_param(extra_params, "num_emb", "pwl"))
    n_bins <- max(2L, as.integer(.pick_first_param(extra_params, "n_bins", 48L)))
    d_embedding <- max(1L, as.integer(.pick_first_param(extra_params, c("d_embedding", "d_cat"), 8L)))
    batch_size <- as.integer(.pick_first_param(extra_params, "batch_size", -1L))
    lr_warmup_frac <- min(0.5, max(0.0, as.numeric(.pick_first_param(extra_params, "lr_warmup_frac", 0.05))))
    seed_val <- as.integer(.pick_first_param(extra_params, c("tabm_seed", "seed"), 42L))
    metric_arg <- as.character(.pick_first_param(extra_params, "metric", "default"))

    y_val <- extra_params$y_val
    es_raw <- .pick_first_param(
      extra_params,
      c("early_stopping_rounds", "early_stopping_round", "early_stop_round", "early_stopping"),
      16L
    )
    use_es <- !is.null(x_val) && !is.null(y_val) && is.numeric(es_raw) && is.finite(es_raw) && es_raw > 0
    es_rounds <- if (use_es) as.integer(es_raw) else 0L

    verbose_int <- .nn_verbose_level(
      extra_params,
      learner_opt = "evoFE.tabm_verbose",
      param_keys = c("tabm_verbose", "nn_verbose")
    )

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)

    cat_info <- .resolve_cat_cols(x_train, extra_args = extra_params, cat_schema = cs)
    cat_cols_arg <- if (length(cat_info$cols) > 0L) as.integer(cat_info$cols) else NULL
    cat_cards_arg <- if (length(cat_info$cols) > 0L) as.integer(cat_info$cardinalities) else NULL

    target <- .realmlp_prepare_target(y_train, task, num_class)
    y_val_num <- if (!is.null(y_val)) {
      .realmlp_prepare_target(y_val, task, num_class, levels_target = target$levels_target)$y_num
    } else {
      NULL
    }

    fit_res <- fn_train(
      x_train = x_train,
      y_train = target$y_num,
      x_val = x_val,
      y_val = if (use_es) y_val_num else NULL,
      task = task,
      n_epochs = n_epochs,
      batch_size = batch_size,
      lr = lr_val,
      weight_decay = weight_decay,
      early_stopping_rounds = es_rounds,
      seed = seed_val,
      verbose = verbose_int,
      num_classes = target$num_classes_int,
      threads = threads,
      metric = metric_arg,
      k_ensemble = k_ens,
      d_block = d_block,
      n_blocks = n_blocks,
      dropout = dropout,
      num_emb = num_emb,
      n_bins = n_bins,
      cat_cols = cat_cols_arg,
      cat_cardinalities = cat_cards_arg,
      d_embedding = d_embedding,
      lr_warmup_frac = lr_warmup_frac
    )

    preds <- NULL
    if (!is.null(x_val)) {
      raw_preds <- if (!is.null(fit_res$val_preds)) {
        fit_res$val_preds
      } else if (!is.null(fit_res$predictions)) {
        fit_res$predictions
      } else {
        fn_predict(fit_res$model_state, x_val, threads = threads)
      }
      if (task == "multiclass") {
        probs <- as.matrix(raw_preds)
        colnames(probs) <- as.character(target$levels_target)
        preds <- probs
      } else {
        preds <- as.numeric(raw_preds)
      }
    }

    importances <- as.numeric(fit_res$importances)
    if (length(importances) == ncol(x_train)) {
      names(importances) <- colnames(x_train)
    } else {
      importances <- NULL
    }

    best_it <- if (use_es && !is.null(fit_res$best_epoch) && fit_res$best_epoch > 0) {
      as.integer(fit_res$best_epoch)
    } else if (!is.null(fit_res$best_iteration) && fit_res$best_iteration > 0) {
      as.integer(fit_res$best_iteration)
    } else {
      NULL
    }

    wrapped <- list(
      model_state = fit_res$model_state,
      cat_schema = cs,
      levels_target = target$levels_target,
      best_iteration = best_it,
      best_epoch = best_it,
      threads = threads
    )
    list(model = wrapped, predictions = preds, importances = importances,
         best_iteration = best_it, best_epoch = best_it)
  },
  predict_func = function(model, x_new, task, ...) {
    fn_predict <- get0("rcpp_tabm_predict", mode = "function")
    if (is.null(fn_predict)) {
      stop("Native C++ entrypoint 'rcpp_tabm_predict' is not available in this build.")
    }
    extra_args <- list(...)
    th <- resolve_param_aliases(extra_args)$threads
    n_threads <- if (!is.null(th)) {
      as.integer(th)
    } else if (!is.null(model$threads)) {
      as.integer(model$threads)
    } else {
      default_threads()
    }
    x_san <- .sanitize_feature_matrix(x_new, threads = n_threads, cat_schema = model$cat_schema)
    raw_preds <- fn_predict(model$model_state, x_san, threads = n_threads)
    if (identical(task, "multiclass")) {
      probs <- as.matrix(raw_preds)
      if (!is.null(model$levels_target)) colnames(probs) <- as.character(model$levels_target)
      probs
    } else {
      as.numeric(raw_preds)
    }
  },
  traits = list(is_tree = FALSE, iteration_param = "epochs", scale_iterations_with_data = FALSE)
)

# Helper to prepare NA-free ranger inputs (zero-copy numeric matrix fast path, factor DF when cats exist)
.ranger_prepare_input <- function(x, cat_info = NULL, state = NULL, threads = 1L) {
  p <- ncol(x)
  if (is.null(state)) {
    in_names <- colnames(x)
    if (is.null(in_names) || length(in_names) != p) {
      in_names <- paste0("V", seq_len(p))
    } else {
      in_names <- make.unique(in_names)
    }
    cat_cols <- if (!is.null(cat_info$cols)) cat_info$cols else integer(0L)
    if (length(cat_cols) == 0L) {
      imp <- .impute_and_col_stats(x, compute_sd = FALSE, threads = threads)
      mat <- imp$mat
      if (is.null(colnames(mat)) || !identical(colnames(mat), in_names)) {
        colnames(mat) <- in_names
      }
      return(list(
        data = mat,
        state = list(
          has_cats = FALSE,
          in_names = in_names,
          num_means = imp$col_means,
          p_in = p
        )
      ))
    }

    num_cols <- setdiff(seq_len(p), cat_cols)
    num_means <- numeric(0L)
    x_num <- NULL
    if (length(num_cols) > 0L) {
      imp <- .impute_and_col_stats(x[, num_cols, drop = FALSE], compute_sd = FALSE, threads = threads)
      num_means <- imp$col_means
      x_num <- imp$mat
    }

    cs_levels <- if (!is.null(cat_info$schema$levels)) cat_info$schema$levels else NULL
    cat_specs <- vector("list", length(cat_cols))
    cols_list <- vector("list", p)
    names(cols_list) <- in_names

    num_pos <- 0L
    cat_pos <- 0L
    for (j in seq_len(p)) {
      idx_c <- match(j, cat_cols)
      if (is.na(idx_c)) {
        num_pos <- num_pos + 1L
        cols_list[[j]] <- x_num[, num_pos]
      } else {
        cat_pos <- cat_pos + 1L
        k_orig <- cat_info$cardinalities[idx_c]
        lv <- if (!is.null(cs_levels) && idx_c <= length(cs_levels) && length(cs_levels[[idx_c]]) >= k_orig) {
          as.character(cs_levels[[idx_c]][seq_len(k_orig)])
        } else {
          paste0("L", seq_len(k_orig))
        }
        lv_full <- if (".__NA__" %in% lv) lv else c(lv, ".__NA__")
        na_code <- match(".__NA__", lv_full)
        code <- suppressWarnings(as.integer(floor(x[, j])))
        bad <- is.na(code) | code < 1L | code > k_orig
        if (any(bad)) code[bad] <- na_code
        cols_list[[j]] <- structure(code, levels = lv_full, class = "factor")
        cat_specs[[cat_pos]] <- list(col = j, k_orig = k_orig, lv_full = lv_full, na_code = na_code)
      }
    }
    df_out <- as.data.frame(cols_list, optional = TRUE, stringsAsFactors = FALSE)
    return(list(
      data = df_out,
      state = list(
        has_cats = TRUE,
        in_names = in_names,
        num_cols = num_cols,
        num_means = num_means,
        cat_cols = cat_cols,
        cat_specs = cat_specs,
        p_in = p
      )
    ))
  }

  # Predict time
  if (!isTRUE(state$has_cats)) {
    imp <- .impute_and_col_stats(x, means = state$num_means, compute_sd = FALSE, threads = threads)
    mat <- imp$mat
    if (ncol(mat) == length(state$in_names)) {
      colnames(mat) <- state$in_names
    }
    return(mat)
  }

  x_num <- if (length(state$num_cols) > 0L) {
    .impute_and_col_stats(x[, state$num_cols, drop = FALSE], means = state$num_means, compute_sd = FALSE, threads = threads)$mat
  } else {
    NULL
  }
  cols_list <- vector("list", state$p_in)
  names(cols_list) <- state$in_names
  num_pos <- 0L
  for (j in seq_len(state$p_in)) {
    idx_c <- match(j, state$cat_cols)
    if (is.na(idx_c)) {
      num_pos <- num_pos + 1L
      cols_list[[j]] <- x_num[, num_pos]
    } else {
      spec <- state$cat_specs[[idx_c]]
      code <- suppressWarnings(as.integer(floor(x[, j])))
      bad <- is.na(code) | code < 1L | code > spec$k_orig
      if (any(bad)) code[bad] <- spec$na_code
      cols_list[[j]] <- structure(code, levels = spec$lv_full, class = "factor")
    }
  }
  as.data.frame(cols_list, optional = TRUE, stringsAsFactors = FALSE)
}

# Format ranger predictions identically across validation and test paths
.format_ranger_preds <- function(raw_p, n_rows, task, levels_target = NULL) {
  if (identical(task, "regression")) {
    return(as.numeric(raw_p))
  }
  p_mat <- as.matrix(raw_p)
  if (nrow(p_mat) != n_rows && ncol(p_mat) == n_rows) {
    p_mat <- t(p_mat)
  }
  if (identical(task, "classification")) {
    if ("1" %in% colnames(p_mat)) {
      return(as.numeric(p_mat[, "1"]))
    }
    return(if (ncol(p_mat) >= 2L) as.numeric(p_mat[, 2L]) else as.numeric(p_mat[, 1L]))
  }
  out <- matrix(0.0, nrow = n_rows, ncol = length(levels_target))
  colnames(out) <- levels_target
  common <- intersect(colnames(p_mat), levels_target)
  if (length(common) > 0L) {
    out[, common] <- p_mat[, common, drop = FALSE]
  }
  out
}

# 10. Ranger Extra-Trees Random Forest Evaluator (Python-free)
register_evaluator(
  "ranger",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("ranger", quietly = TRUE)) {
      stop("The 'ranger' package is required to use the 'ranger' evaluator.")
    }
    extra_params <- list(...)
    th <- .pick_first_param(extra_params, c("threads", "nthreads", "nthread", "num_threads", "n_jobs"), threads)
    threads <- max(1L, as.integer(th))

    num_trees <- max(1L, as.integer(.pick_first_param(
      extra_params,
      c("num.trees", "nrounds", "num_rounds", "n_rounds", "num_round", "nround",
        "iterations", "n_iterations", "epochs", "n_epochs"),
      nrounds
    )))

    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    cs <- attr(x_train, "cat_schema", exact = TRUE)
    x_val <- .sanitize_feature_matrix(x_val, threads = threads, cat_schema = cs)

    cat_info <- .resolve_cat_cols(x_train, extra_args = extra_params, cat_schema = cs)
    prep_tr <- .ranger_prepare_input(x_train, cat_info = cat_info, threads = threads)
    x_tr_prep <- prep_tr$data
    prep_state <- prep_tr$state
    x_va_prep <- if (!is.null(x_val)) .ranger_prepare_input(x_val, state = prep_state, threads = threads) else NULL

    orig_names <- colnames(x_train)
    p_in <- ncol(x_train)

    # Degeneracy check for classification/multiclass
    if (task %in% c("classification", "multiclass")) {
      target <- .realmlp_prepare_target(y_train, task, num_class)
      u_obs <- unique(target$y_num[!is.na(target$y_num)])
      if (length(u_obs) < 2L) {
        single_cls <- if (length(u_obs) == 1L) u_obs[1L] else 0.0
        wrapped <- list(
          is_degenerate = TRUE,
          degenerate_class = single_cls,
          task = task,
          levels_target = as.character(target$levels_target),
          num_class = target$num_classes_int,
          prep_state = prep_state,
          cat_schema = cs,
          threads = threads
        )
        preds <- NULL
        if (!is.null(x_va_prep)) {
          n_va <- nrow(x_va_prep)
          if (task == "classification") {
            preds <- rep(if (single_cls >= 0.5) 1.0 else 0.0, n_va)
          } else {
            k <- max(2L, length(target$levels_target))
            preds <- matrix(0.0, nrow = n_va, ncol = k)
            colnames(preds) <- as.character(target$levels_target)
            col_i <- max(1L, min(k, as.integer(single_cls) + 1L))
            preds[, col_i] <- 1.0
          }
        }
        imp <- stats::setNames(numeric(p_in), orig_names)
        return(list(model = wrapped, predictions = preds, importances = imp))
      }
      if (task == "classification") {
        levels_target <- c("0", "1")
        y_fit <- factor(as.character(as.integer(target$y_num >= 0.5)), levels = levels_target)
        prob_flag <- TRUE
      } else {
        levels_target <- as.character(target$levels_target)
        y_fit <- droplevels(factor(levels_target[as.integer(target$y_num) + 1L], levels = levels_target))
        prob_flag <- TRUE
      }
    } else {
      levels_target <- NULL
      y_fit <- as.numeric(y_train)
      prob_flag <- FALSE
    }

    splitrule <- as.character(.pick_first_param(extra_params, "splitrule", "extratrees"))
    mtry_val <- if (!is.null(extra_params$mtry) && is.numeric(extra_params$mtry) && is.finite(extra_params$mtry)) {
      as.integer(extra_params$mtry)
    } else if (!is.null(extra_params$mtry_ratio) && is.numeric(extra_params$mtry_ratio) && is.finite(extra_params$mtry_ratio)) {
      as.integer(round(as.numeric(extra_params$mtry_ratio) * p_in))
    } else {
      NULL
    }
    if (!is.null(mtry_val)) {
      mtry_val <- max(1L, min(p_in, mtry_val))
    }

    seed_val <- as.integer(.pick_first_param(extra_params, "seed", 42L))

    ranger_args <- list(
      x = x_tr_prep,
      y = y_fit,
      num.trees = num_trees,
      splitrule = splitrule,
      probability = prob_flag,
      importance = "impurity",
      save.memory = FALSE,
      num.threads = threads,
      seed = seed_val,
      verbose = is_learner_verbose(extra_params$verbose)
    )
    if (!is.null(mtry_val)) ranger_args$mtry <- mtry_val
    if (length(cat_info$cols) > 0L) {
      ranger_args$respect.unordered.factors <- as.character(
        .pick_first_param(extra_params, "respect.unordered.factors", "order")
      )
    }
    if (!is.null(extra_params$min.node.size) && is.numeric(extra_params$min.node.size) && is.finite(extra_params$min.node.size)) {
      ranger_args$min.node.size <- max(1L, as.integer(extra_params$min.node.size))
    }
    if (!is.null(extra_params$sample.fraction) && is.numeric(extra_params$sample.fraction) && is.finite(extra_params$sample.fraction)) {
      ranger_args$sample.fraction <- min(1.0, max(1e-3, as.numeric(extra_params$sample.fraction)))
    }
    if (identical(splitrule, "extratrees") && !is.null(extra_params$num.random.splits) &&
        is.numeric(extra_params$num.random.splits) && is.finite(extra_params$num.random.splits)) {
      ranger_args$num.random.splits <- max(1L, as.integer(extra_params$num.random.splits))
    }
    if (!is.null(extra_params$replace)) {
      ranger_args$replace <- .parse_bool_flag(extra_params$replace, default = TRUE)
    }
    if (!is.null(extra_params$max.depth) && is.numeric(extra_params$max.depth) && is.finite(extra_params$max.depth)) {
      ranger_args$max.depth <- max(0L, as.integer(extra_params$max.depth))
    }

    fit <- do.call(ranger::ranger, ranger_args)

    preds <- if (!is.null(x_va_prep)) {
      raw_p <- stats::predict(fit, data = x_va_prep, num.threads = threads, verbose = FALSE)$predictions
      .format_ranger_preds(raw_p, nrow(x_va_prep), task, levels_target)
    } else {
      NULL
    }

    imp <- tryCatch({
      ri <- ranger::importance(fit)
      out_imp <- numeric(p_in)
      if (length(ri) == p_in) {
        out_imp <- as.numeric(ri)
      } else if (!is.null(names(ri))) {
        idx <- match(prep_state$in_names, names(ri))
        ok <- !is.na(idx)
        out_imp[ok] <- as.numeric(ri[idx[ok]])
      }
      if (!is.null(orig_names)) names(out_imp) <- orig_names
      out_imp
    }, error = function(e) NULL)

    wrapped <- list(
      ranger_fit = fit,
      task = task,
      levels_target = levels_target,
      prep_state = prep_state,
      cat_schema = cs,
      threads = threads
    )
    list(model = wrapped, predictions = preds, importances = imp)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("ranger", quietly = TRUE)) {
      stop("The 'ranger' package is required to use the 'ranger' evaluator.")
    }
    extra_args <- list(...)
    th <- resolve_param_aliases(extra_args)$threads
    n_threads <- if (!is.null(th)) {
      as.integer(th)
    } else if (!is.null(model$threads)) {
      as.integer(model$threads)
    } else {
      default_threads()
    }
    x_san <- .sanitize_feature_matrix(x_new, threads = n_threads, cat_schema = model$cat_schema)
    x_prep <- .ranger_prepare_input(x_san, state = model$prep_state, threads = n_threads)
    n_rows <- nrow(x_prep)

    if (isTRUE(model$is_degenerate)) {
      if (identical(task, "classification")) {
        return(rep(if (model$degenerate_class >= 0.5) 1.0 else 0.0, n_rows))
      }
      k <- max(2L, length(model$levels_target))
      out <- matrix(0.0, nrow = n_rows, ncol = k)
      if (!is.null(model$levels_target)) colnames(out) <- model$levels_target
      col_i <- max(1L, min(k, as.integer(model$degenerate_class) + 1L))
      out[, col_i] <- 1.0
      return(out)
    }

    raw_p <- stats::predict(model$ranger_fit, data = x_prep, num.threads = n_threads, verbose = FALSE)$predictions
    .format_ranger_preds(raw_p, n_rows, task, model$levels_target)
  },
  traits = list(is_tree = TRUE, iteration_param = "num.trees", scale_iterations_with_data = FALSE)
)

