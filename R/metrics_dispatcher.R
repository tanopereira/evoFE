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
  if (m %in% c("binary_logloss", "multi_logloss", "mlogloss")) {
    return("logloss")
  }
  if (m %in% c("balanced-accuracy", "bal_acc", "bal-acc")) {
    return("balanced_accuracy")
  }
  m
}

#' Validate that a metric is supported for the given task
#' @keywords internal
#' @noRd
validate_metric <- function(metric, task) {
  if (is.function(metric)) {
    return(invisible(TRUE))
  }
  metric_lower <- tolower(metric)
  valid_metrics <- list(
    classification = c(
      "default", "auc", "f1", "mcc", "balanced_accuracy", "balanced-accuracy",
      "bal_acc", "bal-acc", "eval-ts-refinement", "ts-refinement", "ts_refinement"
    ),
    multiclass = c("default", "auc", "eval-ts-refinement", "ts-refinement", "ts_refinement"),
    regression = c("default", "mae", "cal_rmse", "cal-rmse", "cal_mae", "cal-mae")
  )
  if (!metric_lower %in% valid_metrics[[task]]) {
    stop(sprintf(
      "Metric '%s' is not supported for task '%s'. Supported metrics are: %s",
      metric, task, paste(valid_metrics[[task]], collapse = ", ")
    ))
  }
  invisible(TRUE)
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

  metric <- canonical_metric_name(metric, task)

  if (task == "classification") {
    if (metric %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      min_loss <- compute_ts_refinement(y_true, y_pred, task = task, is_logits = FALSE, threads = threads)
      return(exp(-min_loss))
    }
    switch(metric,
      auc = compute_auc(y_true, y_pred),
      f1 = compute_f1(y_true, y_pred),
      mcc = compute_mcc(y_true, y_pred),
      balanced_accuracy = compute_balanced_accuracy(y_true, y_pred),
      compute_exp_neg_logloss(y_true, y_pred)
    )
  } else if (task == "multiclass") {
    if (metric %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      min_loss <- compute_ts_refinement(y_true, y_pred, task = task, num_class = num_class, is_logits = FALSE, threads = threads)
      return(exp(-min_loss))
    }
    switch(metric,
      auc = {
        if (!is.matrix(y_pred)) {
          y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
        }
        compute_multiclass_auc(y_true, y_pred, num_class)
      },
      compute_exp_neg_multiclass_logloss(y_true, y_pred, num_class)
    )
  } else {
    val_score <- switch(metric,
      mae = compute_mae(y_true, y_pred),
      cal_rmse = compute_calibrated_rmse(y_true, y_pred),
      `cal-rmse` = compute_calibrated_rmse(y_true, y_pred),
      cal_mae = compute_calibrated_mae(y_true, y_pred),
      `cal-mae` = compute_calibrated_mae(y_true, y_pred),
      compute_rmse(y_true, y_pred)
    )
    -val_score
  }
}
