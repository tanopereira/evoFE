#' Regression performance metrics and target transformations
#'
#' @noRd
compute_mae <- function(y_true, y_pred) {
  mean(abs(as.numeric(y_true) - as.numeric(y_pred)), na.rm = TRUE)
}

#' @noRd
compute_rmse <- function(y_true, y_pred) {
  sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE))
}

#' Fit a Forward Target Transformation on Regression Training Targets
#'
#' @param y Numeric training target vector.
#' @param method One of \code{"none"}, \code{"log1p"}, \code{"yeo_johnson"}, or \code{"rank_gauss"}.
#' @return A list with \code{y_trans} (transformed target vector) and \code{state} (inverse mapping state).
#' @keywords internal
#' @noRd
.fit_target_transform <- function(y, method = "none") {
  y_num <- as.numeric(y)
  if (is.null(method) || identical(method, "none") || length(y_num) == 0L) {
    return(list(y_trans = y_num, state = NULL))
  }
  method <- match.arg(method, c("none", "log1p", "yeo_johnson", "rank_gauss"))
  if (method == "log1p") {
    fin <- is.finite(y_num)
    min_y <- if (any(fin)) min(y_num[fin]) else 0.0
    shift <- if (min_y < 0.0) (-min_y + 1.0) else 0.0
    y_trans <- log1p(pmax(0.0, y_num + shift))
    list(y_trans = y_trans, state = list(method = "log1p", shift = shift, smearing = 1.0))
  } else if (method == "yeo_johnson") {
    st <- rcpp_fit_yeo_johnson(y_num)
    y_trans <- rcpp_apply_yeo_johnson(y_num, st$lambda, st$mean, st$sd, threads = 1L)
    list(
      y_trans = y_trans,
      state = list(method = "yeo_johnson", lambda = st$lambda, mean = st$mean, sd = st$sd)
    )
  } else if (method == "rank_gauss") {
    knots <- .extract_sorted_knots(y_num, max_knots = 1024L)
    L <- length(knots)
    if (L < 2L) {
      return(list(y_trans = y_num, state = NULL))
    }
    q_vals <- stats::qnorm(pmin(1.0 - 1e-5, pmax(1e-5, (seq_len(L) - 0.5) / L)))
    u <- pmin(1.0 - 1e-5, pmax(1e-5, .eval_midrank_knots(y_num, knots)))
    y_trans <- stats::qnorm(u)
    list(
      y_trans = y_trans,
      state = list(method = "rank_gauss", knots = knots, q_vals = q_vals)
    )
  } else {
    list(y_trans = y_num, state = NULL)
  }
}

#' Apply a Fitted Forward Target Transformation
#'
#' @keywords internal
#' @noRd
.apply_target_transform <- function(y, state) {
  y_num <- as.numeric(y)
  if (is.null(state) || is.null(state$method) || identical(state$method, "none")) {
    return(y_num)
  }
  if (state$method == "log1p") {
    log1p(pmax(0.0, y_num + state$shift))
  } else if (state$method == "yeo_johnson") {
    rcpp_apply_yeo_johnson(y_num, state$lambda, state$mean, state$sd, threads = 1L)
  } else if (state$method == "rank_gauss") {
    u <- pmin(1.0 - 1e-5, pmax(1e-5, .eval_midrank_knots(y_num, state$knots)))
    stats::qnorm(u)
  } else {
    y_num
  }
}

#' Update Duan's Smearing Correction Factor for log1p Target Transform
#'
#' @keywords internal
#' @noRd
.update_target_transform_smearing <- function(state, y_raw, preds_trans) {
  if (is.null(state) || !identical(state$method, "log1p")) {
    return(state)
  }
  z_true <- log1p(pmax(0.0, as.numeric(y_raw) + state$shift))
  z_pred <- as.numeric(preds_trans)
  keep <- is.finite(z_true) & is.finite(z_pred)
  if (sum(keep) >= 5L) {
    resids <- pmin(pmax(z_true[keep] - z_pred[keep], -5.0), 5.0)
    smear <- mean(exp(resids))
    if (is.finite(smear)) {
      state$smearing <- min(2.5, max(0.8, smear))
    }
  }
  state
}

#' Estimate Duan's Smearing from Training-Fold Residuals Without Validation Peeking
#'
#' @keywords internal
#' @noRd
.estimate_model_train_smearing <- function(state, evaluator, model, x_train, y_train_raw,
                                           task = "regression", threads = 1L, seed = 42L) {
  if (is.null(state) || !identical(state$method, "log1p") || is.null(model) || is.null(x_train)) {
    return(state)
  }
  n_tr <- nrow(x_train)
  if (n_tr < 5L) return(state)
  base_eval <- unwrap_evaluator(evaluator)
  eval_entry <- tryCatch(get_evaluator(base_eval), error = function(e) NULL)
  if (is.null(eval_entry) || is.null(eval_entry$predict_func)) return(state)
  if (n_tr > 2000L) {
    s_val <- if (!is.null(seed) && is.numeric(seed) && is.finite(seed)) as.integer(seed) else 42L
    idx <- with_seed(s_val, sample.int(n_tr, 2000L))
    x_sub <- x_train[idx, , drop = FALSE]
    y_sub <- y_train_raw[idx]
  } else {
    x_sub <- x_train
    y_sub <- y_train_raw
  }
  preds_tr <- tryCatch(
    eval_entry$predict_func(model, x_sub, task = task, threads = threads),
    error = function(e) NULL
  )
  if (is.null(preds_tr)) return(state)
  .update_target_transform_smearing(state, y_sub, preds_tr)
}

#' Invert a Target Transformation Back to Original Target Scale
#'
#' @keywords internal
#' @noRd
.invert_target_transform <- function(preds_trans, state, use_smearing = TRUE) {
  if (is.null(preds_trans)) return(NULL)
  p_num <- as.numeric(preds_trans)
  if (is.null(state) || is.null(state$method) || identical(state$method, "none")) {
    return(p_num)
  }
  if (state$method == "log1p") {
    smear <- if (isTRUE(use_smearing) && !is.null(state$smearing) && is.finite(state$smearing)) {
      state$smearing
    } else {
      1.0
    }
    z <- pmin(pmax(p_num, -30.0), 30.0)
    out <- exp(z) * smear - 1.0 - state$shift
    if (state$shift == 0.0) out <- pmax(0.0, out)
    out
  } else if (state$method == "yeo_johnson") {
    w <- p_num * state$sd + state$mean
    lam <- state$lambda
    out <- numeric(length(w))
    pos <- !is.na(w) & w >= 0
    neg <- !is.na(w) & w < 0
    if (any(pos)) {
      wp <- w[pos]
      out[pos] <- if (abs(lam) < 1e-6) {
        exp(pmin(wp, 30.0)) - 1.0
      } else {
        pmax(1e-12, wp * lam + 1.0)^(1.0 / lam) - 1.0
      }
    }
    if (any(neg)) {
      wn <- w[neg]
      lam2 <- 2.0 - lam
      out[neg] <- if (abs(lam2) < 1e-6) {
        1.0 - exp(pmin(-wn, 30.0))
      } else {
        1.0 - pmax(1e-12, -lam2 * wn + 1.0)^(1.0 / lam2)
      }
    }
    out[!is.finite(out)] <- state$mean
    out
  } else if (state$method == "rank_gauss") {
    if (length(state$knots) < 2L) return(p_num)
    stats::approx(state$q_vals, state$knots, xout = p_num, rule = 2, ties = "mean")$y
  } else {
    p_num
  }
}
