.encode_multiclass_zero_index <- function(y, nc) {
  y_idx <- as.integer(y)
  if (length(y_idx) > 0L && min(y_idx, na.rm = TRUE) >= 1L && max(y_idx, na.rm = TRUE) <= nc) {
    y_idx <- y_idx - 1L
  }
  pmax(0L, pmin(as.integer(y_idx), as.integer(nc - 1L)))
}

#' Temperature Scaled Refinement Metric
#'
#' Computes the Temperature Scaled Refinement (TS-Refinement) metric for binary or multiclass classification.
#' The metric finds the temperature $T$ that minimizes the Laplace-smoothed log-loss of the temperature-scaled prediction margins (logits).
#'
#' @param y_true Numeric vector of true labels (0/1 for classification, or 0 to C-1 for multiclass classification).
#' @param y_pred Numeric vector or matrix of predicted probabilities (or logits, if \code{is_logits = TRUE}).
#' @param task Character. Either \code{"classification"} (binary) or \code{"multiclass"}.
#' @param num_class Integer. Number of classes (required for multiclass).
#' @param alpha Numeric. Laplace smoothing parameter (default is 1).
#' @param is_logits Logical. If \code{TRUE}, the input predictions \code{y_pred} are treated directly as prediction margins (logits). If \code{FALSE}, they are treated as probabilities and converted to logits.
#' @param threads Integer. Number of threads to use for parallel computation (defaults to option \code{"evoFE.threads"} or 1).
#' @return Numeric. The minimized smoothed log-loss.
#' @export
compute_ts_refinement <- function(y_true, y_pred, task = "classification", num_class = NULL, alpha = 1, is_logits = FALSE, threads = NULL) {
  if (!task %in% c("classification", "multiclass")) {
    stop("TS-Refinement metric is only supported for 'classification' and 'multiclass' tasks.")
  }

  th <- if (!is.null(threads)) as.integer(threads) else as.integer(getOption("evoFE.threads", 1L))
  if (is.na(th) || th < 1L) th <- 1L

  if (task == "classification") {
    y_true <- encode_binary_target(y_true)
    y_true <- as.numeric(y_true)
    y_pred <- as.numeric(y_pred)

    rcpp_compute_ts_refinement_binary(
      y_true = y_true,
      y_pred = y_pred,
      alpha = as.numeric(alpha),
      is_logits = isTRUE(is_logits),
      threads = th
    )
  } else if (task == "multiclass") {
    if (is.null(num_class)) {
      stop("num_class must be specified for multiclass TS-Refinement.")
    }

    if (!is.matrix(y_pred)) {
      y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
    }

    y_true_0 <- .encode_multiclass_zero_index(y_true, num_class)

    rcpp_compute_ts_refinement_multiclass(
      y_true = y_true_0,
      y_pred = y_pred,
      num_class = as.integer(num_class),
      alpha = as.numeric(alpha),
      is_logits = isTRUE(is_logits),
      threads = th
    )
  }
}

#' Compute Calibrated RMSE
#'
#' Computes the Root Mean Squared Error (RMSE) of y_pred after optimal linear post-calibration.
#' Mathematically equivalent to SD(y_true) * sqrt(1 - R^2) where R is the Pearson correlation.
#'
#' @param y_true Numeric vector of true target values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric calibrated RMSE score.
#' @export
compute_calibrated_rmse <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)

  if (length(y_true) == 0L || length(y_pred) == 0L) {
    return(0.0)
  }

  rcpp_compute_calibrated_rmse(y_true, y_pred)
}

#' Compute Calibrated MAE
#'
#' Computes the Mean Absolute Error (MAE) of y_pred after optimal L1 linear post-calibration.
#' Finds intercept 'a' and slope 'b' minimizing Mean(|y_true - (a + b * y_pred)|) via fast C++ Brent minimization.
#'
#' @param y_true Numeric vector of true target values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric calibrated MAE score.
#' @export
compute_calibrated_mae <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)

  if (length(y_true) == 0L || length(y_pred) == 0L) {
    return(0.0)
  }

  rcpp_compute_calibrated_mae(y_true, y_pred)
}

#' Build a Pre-Encoded Calibration Evaluation Closure for Iterative Learners
#'
#' Hoists target label encoding outside per-round boosting callbacks (LightGBM / XGBoost)
#' and dispatches directly to the underlying C++ calibration kernels.
#'
#' @param y_val Validation target vector (or NULL to fall back to dataset labels).
#' @param custom_eval_type One of "ts_refinement", "cal_rmse", or "cal_mae".
#' @param task Task type ("classification", "multiclass", or "regression").
#' @param num_class Number of classes for multiclass.
#' @param is_logits Logical; whether learner predictions are raw logits.
#' @param threads Integer thread count.
#' @return A function `f(preds, fallback_labels = NULL)` returning the scalar metric value.
#' @keywords internal
#' @noRd
make_calibration_eval_fn <- function(y_val, custom_eval_type, task = "classification",
                                     num_class = NULL, is_logits = FALSE, threads = 1L) {
  th <- as.integer(if (!is.null(threads) && !is.na(threads) && threads >= 1L) threads else 1L)
  is_log <- isTRUE(is_logits)
  n_cls <- if (!is.null(num_class)) as.integer(num_class) else NULL

  val_labels_num <- NULL
  val_labels_mc <- NULL
  if (!is.null(y_val)) {
    if (custom_eval_type == "ts_refinement" && task == "classification") {
      val_labels_num <- as.numeric(encode_binary_target(y_val))
    } else if (custom_eval_type == "ts_refinement" && task == "multiclass" && !is.null(n_cls)) {
      val_labels_mc <- .encode_multiclass_zero_index(y_val, n_cls)
    } else {
      val_labels_num <- as.numeric(y_val)
    }
  }

  function(preds, fallback_labels = NULL) {
    if (custom_eval_type == "ts_refinement") {
      if (task == "classification") {
        lbl <- if (!is.null(val_labels_num)) val_labels_num else as.numeric(encode_binary_target(fallback_labels))
        rcpp_compute_ts_refinement_binary(lbl, as.numeric(preds), 1.0, is_log, th)
      } else {
        lbl <- if (!is.null(val_labels_mc)) val_labels_mc else .encode_multiclass_zero_index(fallback_labels, n_cls)
        p_mat <- if (is.matrix(preds)) preds else matrix(preds, ncol = n_cls, byrow = FALSE)
        rcpp_compute_ts_refinement_multiclass(lbl, p_mat, n_cls, 1.0, is_log, th)
      }
    } else if (custom_eval_type == "cal_rmse") {
      lbl <- if (!is.null(val_labels_num)) val_labels_num else as.numeric(fallback_labels)
      rcpp_compute_calibrated_rmse(lbl, as.numeric(preds))
    } else {
      lbl <- if (!is.null(val_labels_num)) val_labels_num else as.numeric(fallback_labels)
      rcpp_compute_calibrated_mae(lbl, as.numeric(preds))
    }
  }
}

#' Fit a Post-Hoc Calibrator for Calibration-Refined Metrics
#'
#' Extracts the optimal temperature scaling parameter (\code{beta = 1 / T}) for
#' \code{"ts_refinement"} or optimal affine parameters (\code{a, b}) for
#' \code{"cal_rmse"} / \code{"cal_mae"} on out-of-fold / validation predictions.
#' Returns \code{NULL} for all other metrics.
#'
#' @param y_true Validation target vector.
#' @param y_pred Validation predictions (numeric vector or matrix).
#' @param task Character task type (\code{"classification"}, \code{"multiclass"}, or \code{"regression"}).
#' @param metric Character metric name.
#' @param num_class Integer number of classes for multiclass.
#' @param threads Integer thread count.
#' @return A list describing the fitted calibrator, or \code{NULL}.
#' @keywords internal
#' @noRd
fit_post_calibrator <- function(y_true, y_pred, task = "classification", metric = NULL,
                                num_class = NULL, threads = 1L) {
  if (!is.character(metric) || length(metric) != 1L) {
    return(NULL)
  }
  metric <- canonical_metric_name(metric, task)
  if (!metric %in% c("ts_refinement", "cal_rmse", "cal_mae", "f1", "mcc", "balanced_accuracy")) {
    return(NULL)
  }
  if (is.null(y_true) || is.null(y_pred) || length(y_true) == 0L) {
    return(NULL)
  }
  th <- as.integer(if (!is.null(threads) && !is.na(threads) && threads >= 1L) threads else 1L)

  if (metric == "ts_refinement" && task == "classification") {
    yt <- as.numeric(encode_binary_target(y_true))
    yp <- as.numeric(y_pred)
    keep <- is.finite(yt) & is.finite(yp)
    if (sum(keep) < 2L) return(NULL)
    rcpp_fit_post_calibrator(yt[keep], yp[keep], task = "classification",
                             metric = "ts_refinement", threads = th)
  } else if (metric %in% c("f1", "mcc", "balanced_accuracy") && task == "classification") {
    opt <- optimize_threshold(y_true, y_pred, metric = metric)
    if (is.null(opt) || !is.finite(opt$threshold)) return(NULL)
    list(
      type = "threshold_shift",
      metric = metric,
      threshold = opt$threshold,
      best_score = opt$best_score,
      default_score = opt$default_score
    )
  } else if (metric == "ts_refinement" && task == "multiclass") {
    n_cls <- if (!is.null(num_class)) as.integer(num_class) else if (is.matrix(y_pred)) ncol(y_pred) else 0L
    if (is.na(n_cls) || n_cls < 2L) return(NULL)
    p_mat <- if (is.matrix(y_pred)) y_pred else matrix(y_pred, ncol = n_cls, byrow = FALSE)
    yt <- .encode_multiclass_zero_index(y_true, n_cls)
    keep <- !is.na(yt) & rowSums(!is.finite(p_mat)) == 0L
    if (sum(keep) < 2L) return(NULL)
    rcpp_fit_post_calibrator(yt[keep], p_mat[keep, , drop = FALSE], task = "multiclass",
                             metric = "ts_refinement", num_class = n_cls, threads = th)
  } else if (metric %in% c("cal_rmse", "cal_mae") && task == "regression") {
    yt <- as.numeric(y_true)
    yp <- as.numeric(y_pred)
    keep <- is.finite(yt) & is.finite(yp)
    if (sum(keep) < 2L) return(NULL)
    rcpp_fit_post_calibrator(yt[keep], yp[keep], task = "regression",
                             metric = metric, threads = th)
  } else {
    NULL
  }
}

#' Apply a Fitted Post-Hoc Calibrator to Predictions
#'
#' Applies temperature scaling (\code{ts_binary} / \code{ts_multiclass}),
#' decision-boundary prior-odds shift (\code{threshold_shift}), or affine
#' calibration (\code{linear}) fitted by \code{fit_post_calibrator}.
#'
#' @param preds Numeric vector or matrix of raw model predictions.
#' @param calibrator List returned by \code{fit_post_calibrator}, or \code{NULL}.
#' @return Calibrated predictions with the same dimensions and dimnames as \code{preds}.
#' @keywords internal
#' @noRd
apply_post_calibrator <- function(preds, calibrator) {
  if (is.null(calibrator) || is.null(preds) || is.null(calibrator$type)) {
    return(preds)
  }
  if (calibrator$type == "ts_binary") {
    beta <- as.numeric(calibrator$beta)
    if (!is.finite(beta) || beta <= 0) return(preds)
    p <- pmin(pmax(as.numeric(preds), 1e-15), 1.0 - 1e-15)
    z <- pmin(pmax(log(p / (1.0 - p)), -35.0), 35.0)
    z[!is.finite(z)] <- 0.0
    out <- 1.0 / (1.0 + exp(-beta * z))
    pmin(pmax(out, 1e-15), 1.0 - 1e-15)
  } else if (calibrator$type == "threshold_shift") {
    tau <- as.numeric(calibrator$threshold)
    if (!is.finite(tau) || tau <= 0 || tau >= 1 || abs(tau - 0.5) < 1e-6) {
      return(preds)
    }
    p <- pmin(pmax(as.numeric(preds), 1e-15), 1.0 - 1e-15)
    odds_p <- p / tau
    odds_neg <- (1.0 - p) / (1.0 - tau)
    out <- odds_p / (odds_p + odds_neg)
    pmin(pmax(out, 1e-15), 1.0 - 1e-15)
  } else if (calibrator$type == "ts_multiclass") {
    beta <- as.numeric(calibrator$beta)
    if (!is.finite(beta) || beta <= 0 || !is.matrix(preds)) return(preds)
    p <- pmin(pmax(preds, 1e-15), 1.0 - 1e-15)
    z <- pmin(pmax(log(p), -35.0), 35.0)
    z[!is.finite(z)] <- 0.0
    row_max <- apply(z, 1L, max)
    z_centered <- sweep(z, 1L, row_max, "-")
    exp_z <- exp(beta * z_centered)
    out <- exp_z / rowSums(exp_z)
    dimnames(out) <- dimnames(preds)
    out
  } else if (calibrator$type == "linear") {
    a <- as.numeric(calibrator$a)
    b <- as.numeric(calibrator$b)
    if (!is.finite(a) || !is.finite(b)) return(preds)
    a + b * as.numeric(preds)
  } else {
    preds
  }
}
