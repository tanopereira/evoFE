#' Evaluate holdout fitness for an individual
#'
#' @param ind An \code{evo_individual} object.
#' @param data A data.frame or data.table.
#' @param split_ids Character vector of split identifiers.
#' @param shared_splits Optional pre-split data tables.
#' @param target_col Target column name.
#' @param task Task type.
#' @param evaluator Evaluator name.
#' @param threads Thread count.
#' @param state_cache State cache environment.
#' @param classes Class levels.
#' @param num_class Number of classes.
#' @param metric Metric name.
#' @param verbose Verbosity.
#' @param ... Additional arguments.
#' @return An updated \code{evo_individual} with \code{holdout_fitness} evaluated on the holdout partition.
#' @keywords internal
evaluate_holdout_fitness <- function(ind, data, split_ids, shared_splits,
                                     target_col, task, evaluator, threads,
                                     state_cache, classes, num_class, metric = "default",
                                     target_transform = getOption("evoFE.target_transform", "none"),
                                     verbose = FALSE, ...) {
  use_tt <- identical(task, "regression") && !is.null(target_transform) && !identical(target_transform, "none")
  use_smear <- !identical(canonical_metric_name(metric, task), "mae")
  if (!is.null(shared_splits)) {
    train_fold <- shared_splits$train
    val_fold <- shared_splits$val
    holdout_fold <- shared_splits$holdout
  } else {
    train_fold <- data.table::as.data.table(data[split_ids == "train", ])
    val_fold <- data.table::as.data.table(data[split_ids == "val", ])
    holdout_fold <- if ("holdout" %in% split_ids) data.table::as.data.table(data[split_ids == "holdout", ]) else NULL
  }

  if (is.null(holdout_fold)) {
    ind$holdout_fitness <- NULL
    return(ind)
  }

  train_fold <- shallow_copy_dt(train_fold)
  val_fold <- shallow_copy_dt(val_fold)
  holdout_fold <- shallow_copy_dt(holdout_fold)

  res <- tryCatch(
    {
      apply_individual(ind, train_fold, val_fold, target_col,
                       state_cache = state_cache, validate_genes = FALSE)
    },
    error = function(e) NULL
  )

  if (is.null(res)) {
    ind$holdout_fitness <- -Inf
    return(ind)
  }

  train_fold_feat <- res$train
  val_fold_feat <- res$val

  features <- extract_individual_features(res$ind, target_col = target_col)

  x_train <- .sanitize_feature_matrix(train_fold_feat[, features, with = FALSE], threads = threads)
  x_val   <- .sanitize_feature_matrix(val_fold_feat[, features, with = FALSE], threads = threads)
  y_train <- train_fold_feat[[target_col]]
  y_val <- val_fold_feat[[target_col]]
  if (task == "multiclass") {
    y_train <- encode_multiclass_target(y_train, classes)
    y_val <- encode_multiclass_target(y_val, classes)
  } else if (task == "classification") {
    if (is.character(y_train) || (is.numeric(y_train) && !all(stats::na.omit(y_train) %in% c(0, 1)))) {
      y_fac <- as.factor(y_train)
      y_train <- as.integer(y_fac) - 1L
      y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
    } else {
      y_train <- encode_binary_target(y_train)
      y_val <- encode_binary_target(y_val)
    }
  }

  tt_state_ho <- NULL
  if (use_tt) {
    tt_fit <- .fit_target_transform(y_train, target_transform)
    y_train <- tt_fit$y_trans
    y_val <- .apply_target_transform(y_val, tt_fit$state)
    tt_state_ho <- tt_fit$state
  }

  # Crucial distinction: The holdout fold is strictly for testing generalization performance on unseen
  # data, and must NEVER be used to drive parameter tuning. Therefore, we bypass the tuner (e.g.
  # lightgbm_mbo) and train the base evaluator (e.g. lightgbm) directly using the best parameters
  # found during evolution.
  resolved_eval <- resolve_tuned_evaluator(evaluator, ind$best_params, list(...), always_unwrap = TRUE)
  final_evaluator <- resolved_eval$evaluator
  final_args <- resolved_eval$extra_args

  res_model <- do.call(train_model, c(
    list(
      x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task,
      evaluator = final_evaluator, threads = threads,
      num_class = num_class, metric = metric,
      verbose = verbose
    ),
    final_args
  ))
  if (use_tt && !is.null(tt_state_ho)) {
    tt_state_ho <- .update_target_transform_smearing(tt_state_ho, val_fold_feat[[target_col]], res_model$predictions)
  }

  if (!is.null(res_model$best_params)) {
    ind$best_params <- res_model$best_params
  }

  iter_val <- extract_best_iteration(res_model)
  if (!is.null(iter_val)) {
    ind$best_iteration <- iter_val
  }
  ind$train_size <- nrow(train_fold)

  res_holdout <- tryCatch(
    {
      apply_individual(res$ind, holdout_fold, NULL, NULL, state_cache = state_cache)
    },
    error = function(e) NULL
  )

  if (!is.null(res_holdout)) {
    x_holdout <- .sanitize_feature_matrix(res_holdout$train[, features, with = FALSE], threads = threads)

    evaluator_entry <- get_evaluator(evaluator)
    preds_holdout <- evaluator_entry$predict_func(res_model$model, x_holdout, task = task, threads = threads)
    if (use_tt && !is.null(tt_state_ho)) {
      preds_holdout <- .invert_target_transform(preds_holdout, tt_state_ho, use_smearing = use_smear)
    }

    if (task == "multiclass") {
      y_holdout_encoded <- encode_multiclass_target(holdout_fold[[target_col]], classes)
      preds_holdout <- format_multiclass_predictions(preds_holdout, classes)
      ind$holdout_fitness <- compute_metric(y_holdout_encoded, preds_holdout, task, metric, num_class, threads = threads)
    } else if (task == "classification") {
      y_holdout_encoded <- encode_binary_target(holdout_fold[[target_col]])
      ind$holdout_fitness <- compute_metric(y_holdout_encoded, preds_holdout, task, metric, threads = threads)
    } else {
      ind$holdout_fitness <- compute_metric(holdout_fold[[target_col]], preds_holdout, task, metric, threads = threads)
    }
  } else {
    ind$holdout_fitness <- -Inf
  }

  # Propagate updated genes
  ind$genes <- res$ind$genes
  ind
}
