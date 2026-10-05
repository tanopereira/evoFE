#' Internal helper to convert input features into a clean numeric matrix
#' safe for C++ ML backends (e.g. XGBoost, LightGBM, CatBoost).
#' Handles data.frames, factors/characters, non-finites (Inf, -Inf, NaN),
#' and numbers exceeding 32-bit single-precision float range (~3.402823e38),
#' converting all out-of-bounds or non-finite values to NA_real_.
#' @noRd
.sanitize_feature_matrix <- function(x) {
  sanitize_feature_matrix(x)
}

#' Train a boosted tree model
#'
#' Internal helper that encapsulates LightGBM / XGBoost parameter construction
#' and training. Returns the fitted model, optional predictions on validation
#' data, and feature importances.
#'
#' @param x_train Numeric matrix of training features.
#' @param y_train Numeric vector of training labels.
#' @param x_val Optional numeric matrix of validation features.
#' @param y_val Optional numeric vector of validation labels.
#' @param task Task type: "classification", "multiclass", or "regression".
#' @param evaluator Model type: "lightgbm" or "xgboost".
#' @param threads Number of threads.
#' @param num_class Number of classes (required for multiclass).
#' @param nrounds Number of boosting rounds.
#' @param ... Additional arguments passed to the evaluator training function.
#' @return A list with elements \code{model},
#'   \code{predictions} (NULL when \code{x_val} is NULL),
#'   and \code{importances} (named numeric vector or NULL).
#' @keywords internal
train_model <- function(x_train = NULL, y_train = NULL, x_val = NULL, y_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {

  evaluator_entry <- get_evaluator(evaluator)

  resolved <- resolve_param_aliases(list(...), defaults = list(threads = threads, nrounds = nrounds))
  threads <- resolved$threads
  nrounds <- resolved$nrounds

  x_train <- .sanitize_feature_matrix(x_train)
  x_val   <- .sanitize_feature_matrix(x_val)

  evaluator_entry$train_func(
    x_train = x_train,
    y_train = y_train,
    x_val = x_val,
    y_val = y_val,
    task = task,
    threads = threads,
    num_class = num_class,
    nrounds = nrounds,
    ...
  )
}
