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
compute_f1 <- function(y_true, y_pred) {
  y_true <- encode_binary_target(y_true)
  preds <- as.integer(as.numeric(y_pred) >= 0.5)
  tp <- sum(y_true == 1 & preds == 1, na.rm = TRUE)
  fp <- sum(y_true == 0 & preds == 1, na.rm = TRUE)
  fn <- sum(y_true == 1 & preds == 0, na.rm = TRUE)
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
  if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall)
}
