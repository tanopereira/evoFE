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
  "n_epochs", "iterations", "n_iterations", "seed"
)

# Shared LightGBM Training Core (used by lightgbm and future lightgbm_rf)
.lightgbm_fit_core <- function(x_train, y_train, x_val = NULL, task = "classification",
                               threads = default_threads(), num_class = NULL, nrounds = 50,
                               default_params = list(num_leaves = 15, learning_rate = 0.1), ...) {
  if (!requireNamespace("lightgbm", quietly = TRUE)) {
    stop("The 'lightgbm' package is required to use the 'lightgbm' evaluator. Please install it.")
  }
  x_train <- .sanitize_feature_matrix(x_train, threads = threads)
  x_val   <- .sanitize_feature_matrix(x_val, threads = threads)
  dtrain <- lightgbm::lgb.Dataset(data = x_train, label = y_train)
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

  valids <- list()
  needs_val <- (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) || learner_verbose
  if (needs_val && !is.null(x_val) && !is.null(y_val)) {
    dval <- lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain)
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

  best_it <- if (!is.null(model$best_iter) && model$best_iter > 0) model$best_iter else NULL
  preds <- if (!is.null(x_val)) {
    if (!is.null(best_it)) {
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
  x_new <- .sanitize_feature_matrix(x_new)
  extra_args <- list(...)
  th <- resolve_param_aliases(extra_args)$threads
  if (is.null(th)) {
    th <- if (!is.null(model$params$num_threads)) as.integer(model$params$num_threads) else default_threads()
  }
  num_iter <- if (!is.null(extra_args$num_iteration)) {
    as.integer(extra_args$num_iteration)
  } else if (!is.null(model$best_iter) && model$best_iter > 0) {
    model$best_iter
  } else {
    NULL
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

# 2. XGBoost Evaluator
register_evaluator(
  "xgboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = default_threads(), num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("The 'xgboost' package is required to use the 'xgboost' evaluator. Please install it.")
    }
    x_train <- .sanitize_feature_matrix(x_train, threads = threads)
    x_val   <- .sanitize_feature_matrix(x_val, threads = threads)
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
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train, missing = NA, nthread = threads)
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

    if (identical(params$booster, "gblinear")) {
      params$nthread <- 1L
    }

    dval_metric <- NULL
    # Early stopping or verbose logging requires a validation set in `evals`
    needs_val <- (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) || learner_verbose
    if (needs_val && !is.null(x_val) && !is.null(y_val)) {
      dval_metric <- xgboost::xgb.DMatrix(data = x_val, label = y_val, missing = NA, nthread = threads)
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
    best_iter <- .xgb_best_iter(model)

    preds <- if (!is.null(x_val)) {
      dval <- if (!is.null(dval_metric)) dval_metric else xgboost::xgb.DMatrix(data = x_val, missing = NA, nthread = threads)
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
          dsub <- xgboost::xgb.DMatrix(data = .slice_feature_matrix(x_train, sub_idx), missing = NA, nthread = threads)
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
    x_new <- .sanitize_feature_matrix(x_new)
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
    best_iter <- .xgb_best_iter(model)
    iter_range <- if (!is.null(extra_args$iterationrange)) {
      extra_args$iterationrange
    } else if (!is.null(best_iter) && best_iter >= 1) {
      c(1, best_iter + 1)
    } else {
      NULL
    }
    dmatrix <- xgboost::xgb.DMatrix(data = x_new, missing = NA, nthread = th)
    preds <- if (!is.null(iter_range)) {
      stats::predict(model, dmatrix, iterationrange = iter_range)
    } else {
      stats::predict(model, dmatrix)
    }
    rm(dmatrix)
    preds
  },
  traits = list(is_tree = TRUE, iteration_param = "nrounds", scale_iterations_with_data = TRUE)
)

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
    x_val   <- .sanitize_feature_matrix(x_val, threads = threads)

    dtrain <- catboost::catboost.load_pool(data = x_train, label = y_train, thread_count = threads)

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
      dval_es <- catboost::catboost.load_pool(data = x_val, label = y_val, thread_count = threads)
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

    preds <- NULL
    if (!is.null(x_val)) {
      dval <- if (!is.null(dval_es)) dval_es else catboost::catboost.load_pool(data = x_val, thread_count = threads)
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
          catboost::catboost.load_pool(data = .slice_feature_matrix(x_train, sub_idx), label = y_train[sub_idx], thread_count = threads)
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
    x_new <- .sanitize_feature_matrix(x_new, threads = th)
    dval <- catboost::catboost.load_pool(data = x_new, thread_count = th)
    pred_type <- if (task == "regression") "RawFormulaVal" else "Probability"
    preds <- suppressWarnings(catboost::catboost.predict(model, dval, prediction_type = pred_type, thread_count = th))

    if (task == "multiclass" && !is.matrix(preds)) {
      num_class <- if (!is.null(user_args$num_class)) user_args$num_class else NULL
      if (!is.null(num_class) && num_class > 0) {
        preds <- matrix(preds, ncol = num_class, byrow = TRUE)
      }
    }
    preds
  },
  traits = list(is_tree = TRUE, iteration_param = "iterations", scale_iterations_with_data = TRUE)
)

## 4. LM/GLM Evaluator (Penalized)
# Shared Penalized Linear (glmnet) Core (used by lm and future spline_lm)
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

  x_train <- .sanitize_feature_matrix(x_train, threads = threads)
  x_val   <- .sanitize_feature_matrix(x_val, threads = threads)

  x_mat <- as.matrix(x_train)
  col_means <- colMeans(x_mat, na.rm = TRUE)
  col_means[is.na(col_means) | !is.finite(col_means)] <- 0
  for (j in seq_len(ncol(x_mat))) {
    na_idx <- is.na(x_mat[, j])
    if (any(na_idx)) {
      x_mat[na_idx, j] <- col_means[j]
    }
  }

  impute_mat <- function(x_data) {
    if (is.null(x_data)) return(NULL)
    xm <- if (is.matrix(x_data)) x_data else as.matrix(x_data)
    for (j in seq_len(ncol(xm))) {
      col_name <- if (!is.null(colnames(xm))) colnames(xm)[j] else NULL
      c_mean <- if (!is.null(col_name) && col_name %in% names(col_means)) {
        col_means[[col_name]]
      } else if (j <= length(col_means)) {
        col_means[j]
      } else {
        0
      }
      na_idx <- is.na(xm[, j])
      if (any(na_idx)) {
        xm[na_idx, j] <- c_mean
      }
    }
    xm
  }

  all_feats <- colnames(x_train)

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

  standardize <- if (!is.null(extra_params$standardize) && is.logical(extra_params$standardize)) {
    isTRUE(extra_params$standardize)
  } else {
    TRUE
  }

  relax <- isTRUE(extra_params$relax)

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

  # Single-class degeneracy check in classification/multiclass
  u_train <- unique(y_train[!is.na(y_train)])
  if (task %in% c("classification", "multiclass") && length(u_train) < 2L) {
    single_class <- if (length(u_train) == 1L) u_train[1] else 0
    model <- list(
      is_degenerate = TRUE,
      degenerate_class = single_class,
      task = task,
      col_means = col_means,
      s = s_choice,
      num_class = num_class
    )
    preds <- NULL
    if (!is.null(x_val)) {
      val_mat <- impute_mat(x_val)
      if (task == "classification") {
        pred_val <- if (as.character(single_class) %in% c("1", "true", "TRUE")) 1.0 else 0.0
        preds <- rep(pred_val, nrow(val_mat))
      } else {
        k <- if (!is.null(num_class)) as.integer(num_class) else 2L
        preds <- matrix(0.0, nrow = nrow(val_mat), ncol = k)
        cls_idx <- suppressWarnings(as.integer(as.character(single_class)))
        if (!is.na(cls_idx)) {
          col_idx <- if (cls_idx >= 1L && cls_idx <= k) cls_idx else if (cls_idx == 0L) 1L else 1L
          preds[, col_idx] <- 1.0
        } else {
          preds[, 1L] <- 1.0
        }
      }
    }
    importances <- stats::setNames(rep(0, length(all_feats)), all_feats)
    return(list(model = model, predictions = preds, importances = importances, col_means = col_means))
  }

  nfolds <- min(5, nrow(x_mat))
  if (!is.null(extra_params$nfolds)) {
    nfolds <- min(as.integer(extra_params$nfolds), nrow(x_mat))
  }

  # Precompute standard deviations to make importances scale-invariant
  sd_x <- apply(x_mat, 2, stats::sd, na.rm = TRUE)
  sd_x[is.na(sd_x) | sd_x == 0] <- 1

  is_single_col <- (ncol(x_mat) == 1L)
  fit_mat <- if (is_single_col) cbind(x_mat, .__dummy__ = 0.0) else x_mat

  cv_args <- list(
    x = fit_mat,
    y = if (task == "regression") y_train else as.factor(y_train),
    family = switch(task,
      regression = "gaussian",
      classification = "binomial",
      multiclass = "multinomial",
      stop(sprintf("Unsupported task '%s' for glmnet.", task))
    ),
    alpha = alpha,
    nfolds = nfolds,
    type.measure = type_measure,
    standardize = standardize
  )
  if (isTRUE(relax)) {
    cv_args$relax <- TRUE
  }

  model <- do.call(glmnet::cv.glmnet, cv_args)

  preds <- NULL
  if (!is.null(x_val)) {
    val_mat <- impute_mat(x_val)
    if (is_single_col) val_mat <- cbind(val_mat, .__dummy__ = 0.0)
    if (task == "regression") {
      preds <- as.numeric(stats::predict(model, newx = val_mat, s = s_choice))
    } else if (task == "classification") {
      preds <- as.numeric(stats::predict(model, newx = val_mat, s = s_choice, type = "response"))
    } else {
      p_array <- stats::predict(model, newx = val_mat, s = s_choice, type = "response")
      preds <- matrix(p_array[, , 1], nrow = nrow(val_mat))
      if (!is.null(dimnames(p_array)[[2]])) {
        colnames(preds) <- dimnames(p_array)[[2]]
      }
    }
  }

  if (task == "multiclass") {
    coefs_list <- stats::coef(model, s = s_choice)
    imp_matrix <- do.call(cbind, lapply(coefs_list, function(c_mat) abs(as.numeric(c_mat)[-1])))
    imp <- rowMeans(imp_matrix)
    names(imp) <- rownames(coefs_list[[1]])[-1]
  } else {
    coefs <- as.matrix(stats::coef(model, s = s_choice))
    coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
    imp <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
  }
  if (is_single_col) {
    imp <- imp[seq_len(ncol(x_train))]
  }
  importances <- imp * sd_x[seq_along(imp)]
  if (!is.null(all_feats)) names(importances) <- all_feats

  # Ensure missing names map to 0
  if (!is.null(all_feats)) {
    missing <- setdiff(all_feats, names(importances))
    if (length(missing) > 0) {
      importances <- c(importances, stats::setNames(rep(0, length(missing)), missing))
    }
  }

  # Attach training-time imputation means and s to the model so predict-time
  # imputation and evaluation is consistent with training (prevents train/serve skew).
  model$col_means <- col_means
  model$s <- s_choice

  list(model = model, predictions = preds, importances = importances, col_means = col_means)
}

.glmnet_predict_core <- function(model, x_new, task, ...) {
  x_mat <- .sanitize_feature_matrix(x_new)
  if (is.null(x_mat) || length(x_mat) == 0L) return(numeric(0))

  if (isTRUE(model$is_degenerate)) {
    if (task == "classification") {
      pred_val <- if (as.character(model$degenerate_class) %in% c("1", "true", "TRUE")) 1.0 else 0.0
      return(rep(pred_val, nrow(x_mat)))
    } else if (task == "multiclass") {
      k <- if (!is.null(model$num_class)) as.integer(model$num_class) else 2L
      out_mat <- matrix(0.0, nrow = nrow(x_mat), ncol = k)
      cls_idx <- suppressWarnings(as.integer(as.character(model$degenerate_class)))
      if (!is.na(cls_idx)) {
        col_idx <- if (cls_idx >= 1L && cls_idx <= k) cls_idx else if (cls_idx == 0L) 1L else 1L
        out_mat[, col_idx] <- 1.0
      } else {
        out_mat[, 1L] <- 1.0
      }
      return(out_mat)
    }
  }

  col_means <- if (!is.null(model$col_means)) model$col_means else colMeans(x_mat, na.rm = TRUE)
  col_means[is.na(col_means) | !is.finite(col_means)] <- 0
  for (j in seq_len(ncol(x_mat))) {
    col_name <- if (!is.null(colnames(x_mat))) colnames(x_mat)[j] else NULL
    c_mean <- if (!is.null(col_name) && col_name %in% names(col_means)) {
      col_means[[col_name]]
    } else if (j <= length(col_means)) {
      col_means[j]
    } else {
      0
    }
    na_idx <- is.na(x_mat[, j]) | !is.finite(x_mat[, j])
    if (any(na_idx)) {
      x_mat[na_idx, j] <- c_mean
    }
  }

  extra_args <- list(...)
  s_choice <- if (!is.null(extra_args$lm_s) && (is.character(extra_args$lm_s) || (is.numeric(extra_args$lm_s) && is.finite(extra_args$lm_s)))) {
    extra_args$lm_s
  } else if (!is.null(extra_args$s) && (is.character(extra_args$s) || (is.numeric(extra_args$s) && is.finite(extra_args$s)))) {
    extra_args$s
  } else if (!is.null(model$s)) {
    model$s
  } else {
    "lambda.min"
  }

  pred_mat <- if (ncol(x_mat) == 1L) cbind(x_mat, .__dummy__ = 0.0) else x_mat
  if (task == "regression") {
    as.numeric(stats::predict(model, newx = pred_mat, s = s_choice))
  } else if (task == "classification") {
    as.numeric(stats::predict(model, newx = pred_mat, s = s_choice, type = "response"))
  } else if (task == "multiclass") {
    p_array <- stats::predict(model, newx = pred_mat, s = s_choice, type = "response")
    preds <- matrix(p_array[, , 1], nrow = nrow(x_mat))
    if (!is.null(dimnames(p_array)[[2]])) {
      colnames(preds) <- dimnames(p_array)[[2]]
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

    opt_verbose <- getOption("evoFE.verbose", 0)
    verbose_arg <- if (!is.null(extra_params$verbose)) extra_params$verbose else opt_verbose
    show_log <- isTRUE(verbose_arg) || (is.numeric(verbose_arg) && verbose_arg >= 1) || isTRUE(opt_verbose) || (is.numeric(opt_verbose) && opt_verbose >= 1)
    is_detail <- isTRUE(verbose_arg >= 2) || isTRUE(opt_verbose >= 2)
    is_super  <- is_learner_verbose(verbose_arg)
    verbose_int <- if (show_log) (if (is_super) 3L else if (is_detail) 2L else 1L) else 0L

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

