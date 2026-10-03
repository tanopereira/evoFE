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
                                     verbose = FALSE, ...) {
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

  train_fold <- data.table::copy(train_fold)
  val_fold <- data.table::copy(val_fold)
  holdout_fold <- data.table::copy(holdout_fold)

  res <- tryCatch(
    {
      apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache)
    },
    error = function(e) NULL
  )

  if (is.null(res)) {
    ind$holdout_fitness <- -Inf
    return(ind)
  }

  train_fold_feat <- res$train
  val_fold_feat <- res$val

  gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(res$ind$numeric_cols, res$ind$categorical_cols, res$ind$datetime_cols, gene_cols)

  x_train <- .sanitize_feature_matrix(train_fold_feat[, features, with = FALSE])
  x_val   <- .sanitize_feature_matrix(val_fold_feat[, features, with = FALSE])
  y_train <- train_fold_feat[[target_col]]
  y_val <- val_fold_feat[[target_col]]
  if (task == "multiclass") {
    y_train <- as.integer(factor(y_train, levels = classes)) - 1
    y_val <- as.integer(factor(y_val, levels = classes)) - 1
  }

  # Crucial distinction: The holdout fold is strictly for testing generalization performance on unseen
  # data, and must NEVER be used to drive parameter tuning. Therefore, we bypass the tuner (e.g.
  # lightgbm_mbo) and train the base evaluator (e.g. lightgbm) directly using the best parameters
  # found during evolution.
  final_evaluator <- unwrap_evaluator(evaluator)

  # Merge best_params into ...
  final_args <- utils::modifyList(list(...), as.list(ind$best_params))

  res_model <- do.call(train_model, c(
    list(
      x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task,
      evaluator = final_evaluator, threads = threads,
      num_class = num_class, metric = metric,
      verbose = verbose
    ),
    final_args
  ))

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

  res_holdout <- tryCatch(
    {
      apply_individual(res$ind, holdout_fold, NULL, NULL, state_cache = state_cache)
    },
    error = function(e) NULL
  )

  if (!is.null(res_holdout)) {
    x_holdout <- .sanitize_feature_matrix(res_holdout$train[, features, with = FALSE])

    evaluator_entry <- get_evaluator(evaluator)
    preds_holdout <- evaluator_entry$predict_func(res_model$model, x_holdout, task = task)

    if (task == "multiclass") {
      y_holdout_encoded <- as.integer(factor(holdout_fold[[target_col]], levels = classes)) - 1
      if (!is.matrix(preds_holdout)) {
        preds_holdout <- matrix(preds_holdout, ncol = num_class, byrow = TRUE)
      }
      ind$holdout_fitness <- compute_metric(y_holdout_encoded, preds_holdout, task, metric, num_class)
    } else {
      ind$holdout_fitness <- compute_metric(holdout_fold[[target_col]], preds_holdout, task, metric)
    }
  } else {
    ind$holdout_fitness <- -Inf
  }

  # Propagate updated genes
  ind$genes <- res$ind$genes
  ind
}
