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
#' @return A list with elements \code{model}, \code{predictions} (NULL when
#'   \code{x_val} is NULL), and \code{importances} (named numeric vector or
#'   NULL).
#' @keywords internal
train_model <- function(x_train, y_train, x_val = NULL,
                        task = "classification", evaluator = "lightgbm",
                        threads = 2, num_class = NULL, nrounds = 50) {

  if (evaluator == "lightgbm") {
    dtrain <- lightgbm::lgb.Dataset(data = x_train, label = y_train)
    params <- list(
      objective  = switch(task,
        classification = "binary",
        multiclass     = "multiclass",
        "regression"),
      metric     = switch(task,
        classification = "binary_logloss",
        multiclass     = "multi_logloss",
        "rmse"),
      num_leaves    = 15,
      learning_rate = 0.1,
      verbose       = -1,
      num_threads   = threads
    )
    if (task == "multiclass") params$num_class <- num_class

    utils::capture.output({
      model <- lightgbm::lgb.train(
        params = params, data = dtrain, nrounds = nrounds, verbose = -1
      )
    })

    preds <- if (!is.null(x_val)) stats::predict(model, x_val) else NULL

    imp <- tryCatch({
      lightgbm::lgb.importance(model, percentage = TRUE)
    }, error = function(e) {
      data.frame(Feature = character(), Gain = numeric())
    })
    importances <- if (nrow(imp) > 0) stats::setNames(imp$Gain, imp$Feature) else NULL

    # Let R handle garbage collection naturally to prevent macOS pointer race conditions
    rm(dtrain)

  } else if (evaluator == "xgboost") {
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
    params <- list(
      objective   = switch(task,
        classification = "binary:logistic",
        multiclass     = "multi:softprob",
        "reg:squarederror"),
      eval_metric = switch(task,
        classification = "logloss",
        multiclass     = "mlogloss",
        "rmse"),
      nthread   = threads,
      max_depth = 4,
      eta       = 0.1
    )
    if (task == "multiclass") params$num_class <- num_class

    utils::capture.output({
      model <- xgboost::xgb.train(
        params = params, data = dtrain, nrounds = nrounds, verbose = 0
      )
    })

    preds <- if (!is.null(x_val)) {
      dval <- xgboost::xgb.DMatrix(data = x_val)
      p <- stats::predict(model, dval)
      rm(dval)
      p
    } else {
      NULL
    }

    imp <- xgboost::xgb.importance(model = model)
    importances <- if (nrow(imp) > 0) stats::setNames(imp$Gain, imp$Feature) else NULL

    # Let R handle garbage collection naturally to prevent macOS pointer race conditions
    rm(dtrain)

  } else {
    stop("Unknown evaluator specified.")
  }

  list(model = model, predictions = preds, importances = importances)
}
