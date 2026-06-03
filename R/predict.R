#' Apply feature engineering recipe to new data
#'
#' @param object An evo_recipe object
#' @param newdata A data.frame or data.table
#' @param ... Additional arguments
#' @return A \code{data.table} containing the engineered feature columns
#'   (original plus all gene-derived columns) for \code{newdata}, ready for
#'   downstream modelling.
#' @export
predict.evo_recipe <- function(object, newdata, ...) {
  ind <- object$best_individual
  
  res <- apply_individual(ind, newdata, val_data = NULL, target_col = NULL)
  applied_ind <- res$ind
  
  gene_cols <- if (length(applied_ind$genes) > 0) vapply(applied_ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(applied_ind$numeric_cols, applied_ind$categorical_cols, gene_cols)
  
  res$train[, features, with = FALSE]
}

#' Predict target values using the fully evolved model
#'
#' @param object An evo_recipe object containing the trained model and best individual
#' @param newdata A data.frame or data.table to make predictions on
#' @param ... Additional arguments (currently unused)
#' @return For binary classification and regression tasks a numeric vector of
#'   predictions.  For multiclass tasks a numeric matrix with one column per
#'   class (columns named after class levels).
#' @export
predict_model <- function(object, newdata, ...) {
  if (is.null(object$best_model)) {
    stop("No model was trained or saved during evolution.")
  }
  
  # Step 1: Evolve features for newdata
  features_dt <- stats::predict(object, newdata)
  
  # Step 2: Convert to numeric matrix
  x_new <- as.matrix(sapply(features_dt, as.numeric))
  x_new[!is.finite(x_new)] <- NA
  
  # Step 3: Run prediction
  if (object$evaluator == "lightgbm") {
    preds <- stats::predict(object$best_model, x_new)
  } else if (object$evaluator == "xgboost") {
    dmatrix <- xgboost::xgb.DMatrix(data = x_new)
    preds <- stats::predict(object$best_model, dmatrix)
  } else {
    stop("Unknown evaluator.")
  }
  
  if (!is.null(object$classes)) {
    if (!is.matrix(preds)) {
      preds <- matrix(preds, ncol = length(object$classes), byrow = TRUE)
    }
    colnames(preds) <- object$classes
  }
  
  preds
}
