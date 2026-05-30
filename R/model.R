#' Train a boosted tree model
#'
#' Internal helper that encapsulates LightGBM / XGBoost parameter construction
#' and training. Returns the fitted model, optional predictions on validation
#' data, and feature importances.
#'
#' @param x_train Numeric matrix of training features.
#' @param y_train Numeric vector of training labels.
#' @param x_val Optional numeric matrix of validation features.
#' @param task Task type: "classification", "multiclass", or "regression".
#' @param evaluator Model type: "lightgbm" or "xgboost".
#' @param threads Number of threads.
#' @param num_class Number of classes (required for multiclass).
#' @param nrounds Number of boosting rounds.
#' @param ... Additional arguments passed to the evaluator training function.
#' @return A list with elements \code{model}, \code{predictions} (NULL when
#'   \code{x_val} is NULL), and \code{importances} (named numeric vector or
#'   NULL).
#' @keywords internal
train_model <- function(x_train, y_train, x_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {

  evaluator_entry <- evo_evaluators[[evaluator]]
  if (is.null(evaluator_entry)) {
    stop(sprintf("Unknown evaluator '%s'. Registered evaluators are: %s", 
                 evaluator, paste(names(evo_evaluators), collapse = ", ")))
  }

  evaluator_entry$train_func(
    x_train = x_train,
    y_train = y_train,
    x_val = x_val,
    task = task,
    threads = threads,
    num_class = num_class,
    nrounds = nrounds,
    ...
  )
}
