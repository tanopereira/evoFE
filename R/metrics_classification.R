#' Classification loss and performance metrics
#'
#' @noRd
compute_exp_neg_logloss <- function(y_true, y_pred) {
  y_true <- encode_binary_target(y_true)
  p <- pmax(pmin(as.numeric(y_pred), 1 - 1e-15), 1e-15)
  ll <- -mean(y_true * log(p) + (1 - y_true) * log(1 - p), na.rm = TRUE)
  exp(-ll)
}

#' @noRd
compute_exp_neg_multiclass_logloss <- function(y_true, y_pred, num_class) {
  n <- length(y_true)
  if (!is.matrix(y_pred)) {
    y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
  }
  y_idx <- as.integer(y_true)
  if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
    idx_col <- y_idx
  } else {
    idx_col <- y_idx + 1L
  }
  idx_col <- pmax(1L, pmin(as.integer(idx_col), as.integer(num_class)))
  probs <- y_pred[cbind(seq_len(n), idx_col)]
  probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
  ll <- -mean(log(probs), na.rm = TRUE)
  exp(-ll)
}

#' @noRd
compute_auc <- function(y_true, y_pred) {
  y_true <- encode_binary_target(y_true)
  valid <- !is.na(y_true) & !is.na(y_pred)
  y_true <- y_true[valid]
  y_pred <- y_pred[valid]
  n_pos <- sum(y_true == 1, na.rm = TRUE)
  n_neg <- sum(y_true == 0, na.rm = TRUE)
  if (n_pos == 0 || n_neg == 0) {
    return(0.5)
  }
  r <- rank(y_pred)
  u <- sum(r[y_true == 1]) - (as.numeric(n_pos) * (n_pos + 1)) / 2
  u / (as.numeric(n_pos) * n_neg)
}

#' @noRd
compute_multiclass_auc <- function(y_true, y_pred_matrix, num_class) {
  if (!is.matrix(y_pred_matrix)) {
    y_pred_matrix <- matrix(y_pred_matrix, ncol = num_class, byrow = FALSE)
  }
  y_idx <- as.integer(y_true)
  if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
    y_true_0 <- y_idx - 1L
  } else {
    y_true_0 <- y_idx
  }
  aucs <- numeric(num_class)
  for (k in 1:num_class) {
    y_true_bin <- as.integer(y_true_0 == (k - 1))
    aucs[k] <- compute_auc(y_true_bin, y_pred_matrix[, k])
  }
  mean(aucs, na.rm = TRUE)
}

#' @noRd
compute_f1 <- function(y_true, y_pred, threshold = 0.5) {
  y_true <- encode_binary_target(y_true)
  preds <- as.integer(as.numeric(y_pred) >= threshold)
  tp <- as.numeric(sum(y_true == 1 & preds == 1, na.rm = TRUE))
  fp <- as.numeric(sum(y_true == 0 & preds == 1, na.rm = TRUE))
  fn <- as.numeric(sum(y_true == 1 & preds == 0, na.rm = TRUE))
  if (2 * tp + fp + fn == 0) 0.0 else (2 * tp) / (2 * tp + fp + fn)
}

#' @noRd
compute_mcc <- function(y_true, y_pred, threshold = 0.5) {
  y_true <- encode_binary_target(y_true)
  preds <- as.integer(as.numeric(y_pred) >= threshold)
  tp <- as.numeric(sum(y_true == 1 & preds == 1, na.rm = TRUE))
  tn <- as.numeric(sum(y_true == 0 & preds == 0, na.rm = TRUE))
  fp <- as.numeric(sum(y_true == 0 & preds == 1, na.rm = TRUE))
  fn <- as.numeric(sum(y_true == 1 & preds == 0, na.rm = TRUE))
  denom <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (!is.finite(denom) || denom <= 0) 0.0 else (tp * tn - fp * fn) / denom
}

#' @noRd
compute_balanced_accuracy <- function(y_true, y_pred, threshold = 0.5) {
  y_true <- encode_binary_target(y_true)
  preds <- as.integer(as.numeric(y_pred) >= threshold)
  tp <- as.numeric(sum(y_true == 1 & preds == 1, na.rm = TRUE))
  tn <- as.numeric(sum(y_true == 0 & preds == 0, na.rm = TRUE))
  fp <- as.numeric(sum(y_true == 0 & preds == 1, na.rm = TRUE))
  fn <- as.numeric(sum(y_true == 1 & preds == 0, na.rm = TRUE))
  tpr <- if (tp + fn == 0) 0.0 else tp / (tp + fn)
  tnr <- if (tn + fp == 0) 0.0 else tn / (tn + fp)
  0.5 * (tpr + tnr)
}

#' Find Optimal Binary Decision Threshold on Out-Of-Fold Predictions
#'
#' Computes the exact global optimum decision threshold \eqn{\tau^* \in [0.02, 0.98]}
#' maximizing \code{"f1"}, \code{"mcc"}, or \code{"balanced_accuracy"} in a single
#' sorted cumulative-sum pass.
#'
#' @param y_true Binary ground-truth labels (0/1, factor, logical, or character).
#' @param y_pred Predicted positive-class probabilities in \eqn{[0, 1]}.
#' @param metric Character string: \code{"f1"}, \code{"mcc"}, or \code{"balanced_accuracy"}.
#' @return A list with \code{threshold} (optimal \eqn{\tau^*}), \code{best_score},
#'   and \code{default_score} (score at \eqn{\tau = 0.5}).
#' @export
optimize_threshold <- function(y_true, y_pred, metric = c("f1", "mcc", "balanced_accuracy")) {
  metric <- match.arg(metric)
  yt <- as.integer(encode_binary_target(y_true))
  yp <- as.numeric(y_pred)
  keep <- !is.na(yt) & is.finite(yp)
  yt <- yt[keep]
  yp <- yp[keep]
  n <- length(yt)
  def_score <- switch(metric,
    f1 = compute_f1(yt, yp, threshold = 0.5),
    mcc = compute_mcc(yt, yp, threshold = 0.5),
    balanced_accuracy = compute_balanced_accuracy(yt, yp, threshold = 0.5)
  )
  if (n < 4L) {
    return(list(threshold = 0.5, best_score = def_score, default_score = def_score))
  }
  total_pos <- as.numeric(sum(yt == 1L))
  total_neg <- as.numeric(n) - total_pos
  if (total_pos == 0 || total_neg == 0) {
    return(list(threshold = 0.5, best_score = def_score, default_score = def_score))
  }

  ord <- order(yp, decreasing = TRUE)
  yp_s <- yp[ord]
  yt_s <- yt[ord]

  tp_vec <- as.numeric(cumsum(yt_s == 1L))
  fp_vec <- as.numeric(seq_len(n)) - tp_vec
  fn_vec <- total_pos - tp_vec
  tn_vec <- total_neg - fp_vec

  scores <- switch(metric,
    f1 = {
      denom <- 2.0 * tp_vec + fp_vec + fn_vec
      ifelse(denom > 0, (2.0 * tp_vec) / denom, 0.0)
    },
    mcc = {
      denom <- sqrt((tp_vec + fp_vec) * (tp_vec + fn_vec) * (tn_vec + fp_vec) * (tn_vec + fn_vec))
      ifelse(denom > 0, (tp_vec * tn_vec - fp_vec * fn_vec) / denom, 0.0)
    },
    balanced_accuracy = {
      0.5 * (tp_vec / total_pos + tn_vec / total_neg)
    }
  )

  # Only evaluate at distinct probability boundaries within [0.02, 0.98]
  is_distinct <- c(yp_s[-n] != yp_s[-1L], TRUE)
  in_bounds <- yp_s >= 0.02 & yp_s <= 0.98
  valid_cuts <- which(is_distinct & in_bounds)
  if (length(valid_cuts) == 0L) {
    return(list(threshold = 0.5, best_score = def_score, default_score = def_score))
  }

  best_idx <- valid_cuts[which.max(scores[valid_cuts])]
  best_score <- scores[best_idx]
  if (!is.finite(best_score) || best_score <= def_score + 1e-12) {
    return(list(threshold = 0.5, best_score = def_score, default_score = def_score))
  }

  tau_star <- if (best_idx < n) {
    0.5 * (yp_s[best_idx] + yp_s[best_idx + 1L])
  } else {
    yp_s[best_idx]
  }
  tau_star <- min(0.98, max(0.02, tau_star))

  list(threshold = tau_star, best_score = best_score, default_score = def_score)
}
