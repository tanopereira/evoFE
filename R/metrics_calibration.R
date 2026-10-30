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

    y_idx <- as.integer(y_true)
    if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
      y_true_0 <- y_idx - 1L
    } else {
      y_true_0 <- y_idx
    }
    y_true_0 <- pmax(0L, pmin(as.integer(y_true_0), as.integer(num_class - 1L)))

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

  encode_mc_zero <- function(y, nc) {
    y_idx <- as.integer(y)
    if (length(y_idx) > 0L && min(y_idx, na.rm = TRUE) >= 1L && max(y_idx, na.rm = TRUE) <= nc) {
      y_idx <- y_idx - 1L
    }
    pmax(0L, pmin(as.integer(y_idx), as.integer(nc - 1L)))
  }

  val_labels_num <- NULL
  val_labels_mc <- NULL
  if (!is.null(y_val)) {
    if (custom_eval_type == "ts_refinement" && task == "classification") {
      val_labels_num <- as.numeric(encode_binary_target(y_val))
    } else if (custom_eval_type == "ts_refinement" && task == "multiclass" && !is.null(n_cls)) {
      val_labels_mc <- encode_mc_zero(y_val, n_cls)
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
        lbl <- if (!is.null(val_labels_mc)) val_labels_mc else encode_mc_zero(fallback_labels, n_cls)
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
