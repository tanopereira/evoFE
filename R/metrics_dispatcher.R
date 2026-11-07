#' Resolve Canonical Metric Name from Aliases
#'
#' Normalizes metric strings and aliases (e.g. hyphens vs underscores, calibration
#' and refinement prefixes) to their canonical internal identifiers.
#'
#' @param metric Character string or custom metric function.
#' @param task Optional task string ("classification", "multiclass", "regression").
#' @return Canonical metric name string, or the original metric function if a function was provided.
#' @keywords internal
#' @noRd
canonical_metric_name <- function(metric, task = NULL) {
  if (is.null(metric) || is.function(metric)) {
    return(metric)
  }
  m <- tolower(trimws(as.character(metric)))
  if (m %in% c("default", "") && !is.null(task)) {
    return(if (identical(task, "regression")) "rmse" else "logloss")
  }
  if (m %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
    return("ts_refinement")
  }
  if (m %in% c("cal_rmse", "cal-rmse")) {
    return("cal_rmse")
  }
  if (m %in% c("cal_mae", "cal-mae")) {
    return("cal_mae")
  }
  if (m == "binary_logloss" && (is.null(task) || identical(task, "classification"))) {
    return("logloss")
  }
  if (m %in% c("multi_logloss", "mlogloss") && (is.null(task) || identical(task, "multiclass"))) {
    return("logloss")
  }
  if (m %in% c("balanced-accuracy", "bal_acc", "bal-acc")) {
    return("balanced_accuracy")
  }
  if (m %in% c("acc", "accuracy")) {
    return("accuracy")
  }
  if (m %in% c("aucpr", "pr_auc", "pr-auc", "prauc", "average_precision", "ap")) {
    return("aucpr")
  }
  if (m %in% c("macro_f1", "f1_macro", "macro-f1", "f1-macro")) {
    return("macro_f1")
  }
  if (m %in% c("rmsle", "root_mean_squared_log_error")) {
    return("rmsle")
  }
  if (m %in% c("r2", "rsq", "r_squared", "r-squared")) {
    return("r2")
  }
  if (m %in% c("mape", "mean_absolute_percentage_error")) {
    return("mape")
  }
  if (m %in% c("smape", "symmetric_mape")) {
    return("smape")
  }
  if (m %in% c("qwk", "quadratic_weighted_kappa", "kappa")) {
    return("qwk")
  }
  m
}

#' Return supported canonical metric identifiers for a task
#' @keywords internal
#' @noRd
supported_canonical_metrics <- function(task) {
  if (!is.character(task) || length(task) != 1L || is.na(task)) {
    return(character(0))
  }
  switch(task,
    classification = c("logloss", "auc", "aucpr", "f1", "mcc", "balanced_accuracy", "accuracy", "qwk", "ts_refinement"),
    multiclass = c("logloss", "auc", "macro_f1", "accuracy", "qwk", "ts_refinement"),
    regression = c("rmse", "mae", "rmsle", "r2", "mape", "smape", "qwk", "cal_rmse", "cal_mae"),
    character(0)
  )
}

#' Validate that a metric is supported for the given task
#' @keywords internal
#' @noRd
validate_metric <- function(metric, task) {
  if (is.function(metric)) {
    return(invisible(TRUE))
  }
  if (!is.character(metric) || length(metric) != 1L || is.na(metric)) {
    stop("Metric must be a single character string or a custom function.")
  }
  supported <- supported_canonical_metrics(task)
  cm <- canonical_metric_name(metric, task)
  if (!cm %in% supported) {
    stop(sprintf(
      "Metric '%s' is not supported for task '%s'. Supported metrics are: %s",
      metric, task, paste(c("default", supported), collapse = ", ")
    ))
  }
  invisible(TRUE)
}

#' Whether a regression metric expects mean-unbiased Duan smearing on log1p inversion
#' @keywords internal
#' @noRd
metric_uses_smearing <- function(metric, task = "regression") {
  cm <- canonical_metric_name(metric, task)
  if (!is.character(cm) || length(cm) != 1L) return(TRUE)
  !cm %in% c("mae", "rmsle", "mape", "smape", "qwk")
}

#' Ideal ceiling value on the fitness scale (higher-is-better) for headroom and BIC scaling
#' @keywords internal
#' @noRd
metric_ideal_ceiling <- function(metric = "default", task = "classification") {
  if (identical(task, "regression")) {
    cm <- canonical_metric_name(metric, task)
    if (is.character(cm) && length(cm) == 1L && cm %in% c("r2", "qwk")) {
      return(1.0)
    }
    return(0.0)
  }
  1.0
}

#' Whether a metric depends purely on monotonic prediction ranks
#' @keywords internal
#' @noRd
metric_is_rank_invariant <- function(metric, task = "classification") {
  is.character(metric) && length(metric) == 1L &&
    canonical_metric_name(metric, task) %in% c("auc", "aucpr")
}

#' Dispatch evaluation metric computation
#'
#' @param y_true True labels/values
#' @param y_pred Predicted values/probabilities
#' @param task Task type ("classification", "multiclass", "regression")
#' @param metric Metric name or function
#' @param num_class Number of classes for multiclass
#' @param threads Number of threads for parallel metric computation
#' @return Numeric metric score (higher is better)
#' @noRd
compute_metric <- function(y_true, y_pred, task, metric, num_class = NULL, threads = NULL) {
  if (is.function(metric)) {
    return(metric(y_true, y_pred))
  }

  validate_metric(metric, task)
  metric <- canonical_metric_name(metric, task)

  if (task == "classification") {
    switch(metric,
      ts_refinement = exp(-compute_ts_refinement(y_true, y_pred, task = task, is_logits = FALSE, threads = threads)),
      auc = compute_auc(y_true, y_pred),
      aucpr = compute_aucpr(y_true, y_pred),
      f1 = compute_f1(y_true, y_pred),
      mcc = compute_mcc(y_true, y_pred),
      balanced_accuracy = compute_balanced_accuracy(y_true, y_pred),
      accuracy = compute_accuracy(y_true, y_pred),
      qwk = compute_qwk_binary(y_true, y_pred),
      logloss = compute_exp_neg_logloss(y_true, y_pred)
    )
  } else if (task == "multiclass") {
    switch(metric,
      ts_refinement = exp(-compute_ts_refinement(y_true, y_pred, task = task, num_class = num_class, is_logits = FALSE, threads = threads)),
      auc = {
        if (!is.matrix(y_pred)) {
          y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
        }
        compute_multiclass_auc(y_true, y_pred, num_class)
      },
      macro_f1 = compute_macro_f1(y_true, y_pred, num_class),
      accuracy = compute_multiclass_accuracy(y_true, y_pred, num_class),
      qwk = compute_qwk_multiclass(y_true, y_pred, num_class),
      logloss = compute_exp_neg_multiclass_logloss(y_true, y_pred, num_class)
    )
  } else {
    if (metric == "r2") {
      return(compute_r2(y_true, y_pred))
    }
    if (metric == "qwk") {
      return(compute_qwk_regression(y_true, y_pred))
    }
    val_score <- switch(metric,
      mae = compute_mae(y_true, y_pred),
      rmsle = compute_rmsle(y_true, y_pred),
      mape = compute_mape(y_true, y_pred),
      smape = compute_smape(y_true, y_pred),
      cal_rmse = compute_calibrated_rmse(y_true, y_pred),
      cal_mae = compute_calibrated_mae(y_true, y_pred),
      rmse = compute_rmse(y_true, y_pred)
    )
    -val_score
  }
}
