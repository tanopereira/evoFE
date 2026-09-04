#' Internal helper to convert input features into a clean numeric matrix
#' safe for C++ ML backends (e.g. XGBoost, LightGBM, CatBoost).
#' Handles data.frames, factors/characters, non-finites (Inf, -Inf, NaN),
#' and numbers exceeding 32-bit single-precision float range (~3.402823e38),
#' converting all out-of-bounds or non-finite values to NA_real_.
.sanitize_feature_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  if (!is.matrix(x)) {
    x <- if (is.data.frame(x)) data.matrix(x) else as.matrix(x)
  }
  if (!is.numeric(x)) {
    storage.mode(x) <- "double"
  }
  # 32-bit single-precision float limit is ~3.402823e38.
  # Values exceeding this or non-finite (Inf, -Inf, NaN) cause XGBoost
  # to fail with: Check failed: valid: Input data contains `inf` or a value too large
  max_float <- 3.402823e38
  x[!is.finite(x) | abs(x) > max_float] <- NA_real_
  x
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
train_model <- function(x_train, y_train, x_val = NULL, y_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {

  evaluator_entry <- evo_evaluators[[evaluator]]
  if (is.null(evaluator_entry)) {
    stop(sprintf("Unknown evaluator '%s'. Registered evaluators are: %s", 
                 evaluator, paste(names(evo_evaluators), collapse = ", ")))
  }

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
