#' Regression performance metrics
#'
#' @noRd
compute_mae <- function(y_true, y_pred) {
  mean(abs(as.numeric(y_true) - as.numeric(y_pred)), na.rm = TRUE)
}

#' @noRd
compute_rmse <- function(y_true, y_pred) {
  sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE))
}
