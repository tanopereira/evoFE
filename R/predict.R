#' Apply feature engineering recipe to new data
#'
#' @param object An evo_recipe object
#' @param newdata A data.frame or data.table
#' @param ... Additional arguments
#' @return A \code{data.table} containing the engineered feature
#'   columns (original plus all gene-derived columns) for
#'   \code{newdata}, ready for downstream modelling.
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   seed = 42,
#'   verbose = FALSE
#' )
#'
#' # Extract engineered features
#' engineered_features <- predict(recipe, df[1:5, ])
#' print(engineered_features)
#' }
#' @export
predict.evo_recipe <- function(object, newdata, ...) {
  ind <- object$best_individual
  
  res <- apply_individual(ind, newdata, val_data = NULL, target_col = NULL, allow_prune = TRUE)
  applied_ind <- res$ind
  
  gene_cols <- if (length(applied_ind$genes) > 0) vapply(applied_ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- unique(c(applied_ind$numeric_cols, applied_ind$categorical_cols, applied_ind$datetime_cols, gene_cols))
  
  res$train[, features, with = FALSE]
}

#' Apply engineered features from an ensemble of recipes
#'
#' @param object An evo_ensemble object
#' @param newdata A data.frame or data.table
#' @param ... Additional arguments
#' @return A data.table containing the combined engineered features across
#'   all active recipes in the ensemble.
#' @export
predict.evo_ensemble <- function(object, newdata, ...) {
  if (is.null(object$active_recipes) || length(object$active_recipes) == 0) {
    stop("No active recipes available in evo_ensemble object.")
  }

  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  dt_new <- data.table::as.data.table(newdata)
  out_dt <- data.table::copy(dt_new)

  for (name in names(object$active_recipes)) {
    ind_i <- object$active_recipes[[name]]
    res_i <- apply_individual(ind_i, dt_new, val_data = NULL, target_col = NULL, allow_prune = TRUE, state_cache = state_cache)
    sub_dt <- res_i$train
    new_cols <- setdiff(names(sub_dt), names(out_dt))
    if (length(new_cols) > 0) {
      for (col in new_cols) {
        out_dt[[col]] <- sub_dt[[col]]
      }
    }
  }

  out_dt
}

#' Predict target values using the fully evolved model or ensemble
#'
#' @param object An evo_recipe or evo_ensemble object containing trained model(s)
#' @param newdata A data.frame or data.table to make predictions on
#' @param ... Additional arguments (currently unused)
#' @return For binary classification and regression tasks a numeric vector of
#'   predictions.  For multiclass tasks a numeric matrix with one column per
#'   class (columns named after class levels).
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   seed = 42,
#'   verbose = FALSE
#' )
#'
#' # Get model predictions
#' predictions <- predict_model(recipe, df[1:5, ])
#' print(predictions)
#' }
#' @export
predict_model <- function(object, newdata, ...) {
  UseMethod("predict_model")
}

#' @export
predict_model.evo_recipe <- function(object, newdata, ...) {
  if (is.null(object$best_model)) {
    stop("No model was trained or saved during evolution.")
  }
  
  # Step 1: Evolve features for newdata
  features_dt <- stats::predict(object, newdata)
  
  # Step 2: Convert to numeric matrix
  x_new <- data.matrix(features_dt)
  x_new[!is.finite(x_new)] <- NA
  
  # Step 3: Run prediction
  evaluator_entry <- evo_evaluators[[object$evaluator]]
  if (is.null(evaluator_entry)) {
    stop(sprintf("Unknown evaluator '%s'. Registered evaluators are: %s", 
                 object$evaluator, paste(names(evo_evaluators), collapse = ", ")))
  }
  preds <- evaluator_entry$predict_func(object$best_model, x_new, task = object$task)
  
  if (!is.null(object$classes)) {
    if (!is.matrix(preds)) {
      preds <- matrix(preds, ncol = length(object$classes), byrow = TRUE)
    }
    colnames(preds) <- object$classes
  }
  
  preds
}

#' @export
predict_model.evo_ensemble <- function(object, newdata, ...) {
  if (is.null(object$active_models) || length(object$active_models) == 0) {
    stop("No active models available in evo_ensemble object.")
  }

  weights <- object$weights
  active_names <- names(weights[weights > 0])
  if (length(active_names) == 0) {
    stop("No active island models with positive weights found in evo_ensemble.")
  }

  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  dt_new <- data.table::as.data.table(newdata)

  evaluator_entry <- evo_evaluators[[object$evaluator]]
  if (is.null(evaluator_entry)) {
    stop(sprintf("Unknown evaluator '%s'. Registered evaluators are: %s",
                 object$evaluator, paste(names(evo_evaluators), collapse = ", ")))
  }

  weighted_preds <- NULL

  for (name in active_names) {
    w <- weights[[name]]
    ind_i <- object$active_recipes[[name]]
    mod_i <- object$active_models[[name]]

    res_i <- apply_individual(ind_i, dt_new, val_data = NULL, target_col = NULL, allow_prune = TRUE, state_cache = state_cache)
    applied_ind <- res_i$ind

    gene_cols <- if (length(applied_ind$genes) > 0) vapply(applied_ind$genes, function(g) g$output_col, character(1)) else character(0)
    features <- unique(c(applied_ind$numeric_cols, applied_ind$categorical_cols, applied_ind$datetime_cols, gene_cols))
    if (!is.null(object$target_col)) {
      features <- setdiff(features, object$target_col)
    }

    feat_dt <- res_i$train[, features, with = FALSE]
    x_new <- data.matrix(feat_dt)
    x_new[!is.finite(x_new)] <- NA

    preds_i <- evaluator_entry$predict_func(mod_i, x_new, task = object$task)

    if (!is.null(object$classes)) {
      if (!is.matrix(preds_i)) {
        preds_i <- matrix(preds_i, ncol = length(object$classes), byrow = TRUE)
      }
      colnames(preds_i) <- object$classes
    }

    if (is.null(weighted_preds)) {
      weighted_preds <- w * preds_i
    } else {
      weighted_preds <- weighted_preds + (w * preds_i)
    }
  }

  weighted_preds
}
