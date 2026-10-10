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

  metric <- tolower(metric)

  if (task == "classification") {
    if (metric %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      min_loss <- compute_ts_refinement(y_true, y_pred, task = task, is_logits = FALSE, threads = threads)
      return(exp(-min_loss))
    }
    switch(metric,
      auc = compute_auc(y_true, y_pred),
      f1 = compute_f1(y_true, y_pred),
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
