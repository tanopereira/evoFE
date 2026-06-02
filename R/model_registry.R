#' Global environment for registered model evaluators
#'
#' @export
evo_evaluators <- new.env(parent = emptyenv())

#' Register a model evaluator
#'
#' @param name Name of the evaluator.
#' @param train_func Function to train the model. Must accept \code{x_train},
#'   \code{y_train}, \code{x_val}, \code{task}, \code{threads}, \code{num_class},
#'   and any additional parameters, and return a list with \code{model},
#'   \code{predictions}, and \code{importances}.
#' @param predict_func Function to make predictions. Must accept \code{model},
#'   \code{x_new}, \code{task}, and any additional parameters, and return a
#'   vector or matrix of predictions.
#' @importFrom lightgbm lgb.train
#' @importFrom xgboost xgb.train
#' @examples
#' # Register a simple mock evaluator
#' register_evaluator(
#'   "mock_eval",
#'   train_func = function(x_train, y_train, x_val = NULL, task = "regression", ...) {
#'     list(
#'       model = list(weights = colMeans(x_train)),
#'       predictions = if (!is.null(x_val)) rowMeans(x_val) else NULL,
#'       importances = stats::setNames(rep(1, ncol(x_train)), colnames(x_train))
#'     )
#'   },
#'   predict_func = function(model, x_new, task, ...) {
#'     rowMeans(x_new)
#'   }
#' )
#'
#' # Verify it is registered
#' exists("mock_eval", envir = evo_evaluators)
#' @export
register_evaluator <- function(name, train_func, predict_func, base_evaluator = NULL) {
  assign(name, list(train_func = train_func, predict_func = predict_func, base_evaluator = base_evaluator), envir = evo_evaluators)
}

# --- Register Default Evaluators ---

# 1. LightGBM Evaluator
register_evaluator(
  "lightgbm",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                         threads = 2, num_class = NULL, nrounds = 50, ...) {
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

    extra_params <- list(...)
    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

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

    rm(dtrain)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    stats::predict(model, x_new)
  }
)

# 2. XGBoost Evaluator
register_evaluator(
  "xgboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                         threads = 2, num_class = NULL, nrounds = 50, ...) {
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

    extra_params <- list(...)
    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

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

    rm(dtrain)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    dmatrix <- xgboost::xgb.DMatrix(data = x_new)
    preds <- stats::predict(model, dmatrix)
    rm(dmatrix)
    preds
  }
)

# 3. CatBoost Evaluator
register_evaluator(
  "catboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                         threads = 2, num_class = NULL, nrounds = 50, ...) {
    if (system.file(package = "catboost") == "") {
      stop("The 'catboost' package is required to use the 'catboost' evaluator. Please install it.")
    }

    df_train <- as.data.frame(x_train)
    df_train[] <- lapply(df_train, as.numeric)
    dtrain <- catboost::catboost.load_pool(data = df_train, label = y_train)
    
    params <- list(
      loss_function = switch(task,
        classification = "Logloss",
        multiclass     = "MultiClass",
        "RMSE"),
      thread_count = threads,
      iterations = nrounds,
      learning_rate = 0.1,
      logging_level = "Silent",
      allow_writing_files = FALSE
    )

    extra_params <- list(...)
    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    model <- suppressWarnings(catboost::catboost.train(dtrain, params = params))

    preds <- NULL
    if (!is.null(x_val)) {
      df_val <- as.data.frame(x_val)
      df_val[] <- lapply(df_val, as.numeric)
      dval <- catboost::catboost.load_pool(data = df_val)
      pred_type <- if (task == "regression") "RawFormulaVal" else "Probability"
      preds <- suppressWarnings(catboost::catboost.predict(model, dval, prediction_type = pred_type))
      
      # For multiclass, format shape as probability matrix
      if (task == "multiclass") {
        if (!is.matrix(preds)) {
          preds <- matrix(preds, ncol = num_class, byrow = TRUE)
        }
      }
    }

    # Fetch feature importance and map column names
    importances <- tryCatch({
      scores <- catboost::catboost.get_feature_importance(model)
      stats::setNames(as.numeric(scores), colnames(x_train))
    }, error = function(e) {
      NULL
    })

    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    if (system.file(package = "catboost") == "") {
      stop("The 'catboost' package is required to use the 'catboost' evaluator. Please install it.")
    }
    df_new <- as.data.frame(x_new)
    df_new[] <- lapply(df_new, as.numeric)
    dval <- catboost::catboost.load_pool(data = df_new)
    pred_type <- if (task == "regression") "RawFormulaVal" else "Probability"
    preds <- suppressWarnings(catboost::catboost.predict(model, dval, prediction_type = pred_type))
    
    if (task == "multiclass") {
      # In R predict_model, multiclass format checks are handled at the caller or class level,
      # but returning a matrix with the correct dimensions is standard.
      # The predict_model function checks and reshapes as well.
    }
    preds
  }
)
