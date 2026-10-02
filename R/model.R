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
train_model <- function(x_train, y_train, x_val = NULL, y_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {

  evaluator_entry <- get_evaluator(evaluator)

  extra_args <- list(...)
  if (!is.null(extra_args$threads)) threads <- as.integer(extra_args$threads)
  if (!is.null(extra_args$nthreads)) threads <- as.integer(extra_args$nthreads)
  if (!is.null(extra_args$nthread)) threads <- as.integer(extra_args$nthread)
  if (!is.null(extra_args$num_threads)) threads <- as.integer(extra_args$num_threads)
  if (!is.null(extra_args$n_jobs)) threads <- as.integer(extra_args$n_jobs)
  if (!is.null(extra_args$nrounds)) nrounds <- as.integer(extra_args$nrounds)
  if (!is.null(extra_args$num_rounds)) nrounds <- as.integer(extra_args$num_rounds)
  if (!is.null(extra_args$n_rounds)) nrounds <- as.integer(extra_args$n_rounds)
  if (!is.null(extra_args$num_round)) nrounds <- as.integer(extra_args$num_round)
  if (!is.null(extra_args$nround)) nrounds <- as.integer(extra_args$nround)
  if (!is.null(extra_args$epochs)) nrounds <- as.integer(extra_args$epochs)
  if (!is.null(extra_args$n_epochs)) nrounds <- as.integer(extra_args$n_epochs)
  if (!is.null(extra_args$iterations)) nrounds <- as.integer(extra_args$iterations)
  if (!is.null(extra_args$n_iterations)) nrounds <- as.integer(extra_args$n_iterations)

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
