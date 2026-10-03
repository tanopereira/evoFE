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
#' @return Numeric. The minimized smoothed log-loss.
#' @export
compute_ts_refinement <- function(y_true, y_pred, task = "classification", num_class = NULL, alpha = 1, is_logits = FALSE) {
  if (!task %in% c("classification", "multiclass")) {
    stop("TS-Refinement metric is only supported for 'classification' and 'multiclass' tasks.")
  }

  if (task == "classification") {
    if (is.factor(y_true)) {
      y_true <- as.integer(y_true) - 1L
    } else if (is.character(y_true)) {
      y_true <- as.integer(as.factor(y_true)) - 1L
    } else if (is.logical(y_true)) {
      y_true <- as.integer(y_true)
    } else if (is.numeric(y_true) && !all(stats::na.omit(y_true) %in% c(0, 1))) {
      y_true <- as.integer(as.factor(y_true)) - 1L
    }
    y_pred <- as.numeric(y_pred)
    if (is_logits) {
      z <- y_pred
    } else {
      # Reconstruct logits (margins) from probabilities
      p <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
      z <- log(p / (1 - p))
    }
    # Clamp infinite values and impute NA/NaN
    z[is.na(z) | is.nan(z)] <- 0
    z <- pmax(pmin(z, 35), -35)

    # Laplace smooth the labels based on true class count
    N1 <- sum(y_true == 1, na.rm = TRUE)
    N0 <- sum(y_true == 0, na.rm = TRUE)
    N_true <- ifelse(y_true == 1, N1, N0)

    y_smooth <- ifelse(y_true == 1,
      (N_true + alpha) / (N_true + 2 * alpha),
      alpha / (N_true + 2 * alpha)
    )

    obj_fn <- function(temp) {
      probs_T <- 1 / (1 + exp(-z / temp))
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
      ll <- -mean(y_smooth * log(probs_T) + (1 - y_smooth) * log(1 - probs_T), na.rm = TRUE)
      ll
    }

    opt <- stats::optimize(f = obj_fn, interval = c(0.001, 10))
    best_temp <- opt$minimum

    # Return the un-smoothed log-loss (Option B) at the optimal temperature
    probs_T <- 1 / (1 + exp(-z / best_temp))
    probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
    ll_unsmoothed <- -mean(y_true * log(probs_T) + (1 - y_true) * log(1 - probs_T), na.rm = TRUE)
    return(ll_unsmoothed)
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

    if (is_logits) {
      z <- y_pred
    } else {
      p <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
      z <- log(p)
    }
    # Clamp infinite values and impute NA/NaN
    z[is.na(z) | is.nan(z)] <- 0
    z <- pmax(pmin(z, 35), -35)

    # Laplace smooth labels based on class-count sweep formulation
    n <- length(y_true_0)
    N_vec <- tabulate(y_true_0 + 1, nbins = num_class)
    N_k_row <- N_vec[y_true_0 + 1]

    true_target <- (N_k_row + alpha) / (N_k_row + 2 * alpha)
    leftover_mass <- alpha / (N_k_row + 2 * alpha)

    denom <- n - N_k_row
    mass_per_item <- ifelse(denom == 0, 0, leftover_mass / pmax(1, denom))

    N_mat <- matrix(N_vec, nrow = n, ncol = num_class, byrow = TRUE)
    y_smooth <- sweep(N_mat, 1, mass_per_item, "*")
    y_smooth[cbind(seq_len(n), y_true_0 + 1)] <- true_target

    obj_fn <- function(temp) {
      z_scaled <- z / temp
      z_max <- apply(z_scaled, 1, max)
      z_stable <- z_scaled - z_max
      exp_z <- exp(z_stable)
      sum_exp_z <- rowSums(exp_z)
      probs_T <- exp_z / sum_exp_z
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)

      ll <- -mean(rowSums(y_smooth * log(probs_T)), na.rm = TRUE)
      ll
    }

    opt <- stats::optimize(f = obj_fn, interval = c(0.001, 10))
    best_temp <- opt$minimum

    # Return the un-smoothed log-loss (Option B) at the optimal temperature
    z_scaled <- z / best_temp
    z_max <- apply(z_scaled, 1, max)
    z_stable <- z_scaled - z_max
    exp_z <- exp(z_stable)
    sum_exp_z <- rowSums(exp_z)
    probs_T <- exp_z / sum_exp_z
    probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)

    y_hard <- matrix(0, nrow = n, ncol = num_class)
    y_hard[cbind(seq_len(n), y_true_0 + 1)] <- 1
    ll_unsmoothed <- -mean(rowSums(y_hard * log(probs_T)), na.rm = TRUE)
    return(ll_unsmoothed)
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

  if (length(y_true) <= 1L) {
    return(0.0)
  }

  sd_true <- stats::sd(y_true)
  sd_pred <- stats::sd(y_pred)

  if (is.na(sd_true) || sd_true < 1e-9) {
    return(0.0)
  }

  if (is.na(sd_pred) || sd_pred < 1e-9) {
    return(sqrt(mean((y_true - mean(y_pred, na.rm = TRUE))^2, na.rm = TRUE)))
  }

  r <- stats::cor(y_pred, y_true, use = "complete.obs")
  if (is.na(r)) {
    return(sqrt(mean((y_true - y_pred)^2, na.rm = TRUE)))
  }

  sd_true * sqrt(pmax(0.0, 1.0 - r^2))
}

#' Compute Calibrated MAE
#'
#' Computes the Mean Absolute Error (MAE) of y_pred after optimal L1 linear post-calibration.
#' Finds intercept 'a' and slope 'b' minimizing Mean(|y_true - (a + b * y_pred)|) via 2D optimization.
#'
#' @param y_true Numeric vector of true target values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric calibrated MAE score.
#' @export
compute_calibrated_mae <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)

  if (length(y_true) == 0L) {
    return(0.0)
  }

  valid <- !is.na(y_true) & !is.na(y_pred)
  y_true <- y_true[valid]
  y_pred <- y_pred[valid]

  if (length(y_true) <= 1L) {
    return(0.0)
  }

  obj_fn <- function(par) {
    a <- par[1]
    b <- par[2]
    mean(abs(y_true - (a + b * y_pred)))
  }

  cov_y <- stats::cov(y_true, y_pred)
  var_pred <- stats::var(y_pred)

  start_b <- if (!is.na(var_pred) && var_pred > 1e-9) cov_y / var_pred else 1.0
  start_a <- mean(y_true) - start_b * mean(y_pred)

  opt <- tryCatch(
    {
      stats::optim(
        par = c(start_a, start_b),
        fn = obj_fn,
        method = "Nelder-Mead"
      )
    },
    error = function(e) {
      list(value = mean(abs(y_true - y_pred)))
    }
  )

  opt$value
}
