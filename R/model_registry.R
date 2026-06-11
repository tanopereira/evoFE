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
#' @param base_evaluator Optional character name of the base registered model.
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
      objective = switch(task,
        classification = "binary",
        multiclass     = "multiclass",
        "regression"
      ),
      metric = switch(task,
        classification = "binary_logloss",
        multiclass     = "multi_logloss",
        "rmse"
      ),
      num_leaves = 15,
      learning_rate = 0.1,
      verbose = -1,
      num_threads = threads,
      seed = 42
    )
    if (task == "multiclass") params$num_class <- num_class

    extra_params <- list(...)
    y_val <- extra_params$y_val
    metric_arg <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    use_ts_refinement <- FALSE
    if (!is.null(metric_arg) && is.character(metric_arg)) {
      if (tolower(metric_arg) %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
        use_ts_refinement <- TRUE
      }
    }

    if (use_ts_refinement) {
      params$metric <- "None"
    }

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    valids <- list()
    if (use_ts_refinement && !is.null(x_val) && !is.null(y_val)) {
      dval <- lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain)
      valids$val <- dval
    }

    lgb_eval <- function(preds, dtrain) {
      labels <- lightgbm::get_field(dtrain, "label")
      score <- compute_ts_refinement(labels, preds, task = task, num_class = num_class, is_logits = FALSE)
      list(name = "ts_refinement", value = score, higher_better = FALSE)
    }

    utils::capture.output({
      model <- lightgbm::lgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        valids = valids,
        eval = if (use_ts_refinement) lgb_eval else NULL,
        early_stopping_rounds = early_stopping_rounds,
        verbose = -1
      )
    })

    preds <- if (!is.null(x_val)) {
      if (!is.null(model$best_iter) && model$best_iter > 0) {
        stats::predict(model, x_val, num_iteration = model$best_iter)
      } else {
        stats::predict(model, x_val)
      }
    } else {
      NULL
    }

    imp <- tryCatch(
      {
        lightgbm::lgb.importance(model, percentage = TRUE)
      },
      error = function(e) {
        data.frame(Feature = character(), Gain = numeric())
      }
    )
    importances <- if (nrow(imp) > 0) stats::setNames(imp$Gain, imp$Feature) else NULL

    rm(dtrain)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!is.null(model$best_iter) && model$best_iter > 0) {
      stats::predict(model, x_new, num_iteration = model$best_iter)
    } else {
      stats::predict(model, x_new)
    }
  }
)

# 2. XGBoost Evaluator
register_evaluator(
  "xgboost",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
    params <- list(
      objective = switch(task,
        classification = "binary:logistic",
        multiclass     = "multi:softprob",
        "reg:squarederror"
      ),
      eval_metric = switch(task,
        classification = "logloss",
        multiclass     = "mlogloss",
        "rmse"
      ),
      nthread = threads,
      max_depth = 6,
      eta = 0.1,
      min_child_weight = 1,
      seed = 42
    )
    if (task == "multiclass") params$num_class <- num_class

    extra_params <- list(...)
    y_val <- extra_params$y_val
    metric_arg <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    use_ts_refinement <- FALSE
    if (!is.null(metric_arg) && is.character(metric_arg)) {
      if (tolower(metric_arg) %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
        use_ts_refinement <- TRUE
      }
    }

    if (use_ts_refinement) {
      params$eval_metric <- NULL
    }

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    evals <- list(train = dtrain)
    dval_metric <- NULL
    if (use_ts_refinement && !is.null(x_val) && !is.null(y_val)) {
      dval_metric <- xgboost::xgb.DMatrix(data = x_val, label = y_val)
      evals$val <- dval_metric
    }

    xgb_feval <- function(preds, dtrain) {
      labels <- xgboost::getinfo(dtrain, "label")
      # XGBoost custom_metric always receives raw margins (logits),
      # unlike LightGBM which passes transformed probabilities.
      score <- compute_ts_refinement(labels, preds, task = task, num_class = num_class, is_logits = TRUE)
      list(metric = "ts_refinement", value = score)
    }

    utils::capture.output({
      model <- xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        evals = evals,
        custom_metric = if (use_ts_refinement) xgb_feval else NULL,
        early_stopping_rounds = early_stopping_rounds,
        maximize = if (use_ts_refinement) FALSE else NULL,
        verbose = 0
      )
    })

    preds <- if (!is.null(x_val)) {
      dval <- xgboost::xgb.DMatrix(data = x_val)
      p <- if (!is.null(model$best_iteration) && model$best_iteration >= 1) {
        stats::predict(model, dval, iterationrange = c(1, model$best_iteration + 1))
      } else {
        stats::predict(model, dval)
      }
      rm(dval)
      p
    } else {
      NULL
    }

    imp <- xgboost::xgb.importance(model = model)
    importances <- if (nrow(imp) > 0) stats::setNames(imp$Gain, imp$Feature) else NULL

    rm(dtrain)
    if (!is.null(dval_metric)) rm(dval_metric)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    dmatrix <- xgboost::xgb.DMatrix(data = x_new)
    preds <- if (!is.null(model$best_iteration) && model$best_iteration >= 1) {
      stats::predict(model, dmatrix, iterationrange = c(1, model$best_iteration + 1))
    } else {
      stats::predict(model, dmatrix)
    }
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
        "RMSE"
      ),
      thread_count = threads,
      iterations = nrounds,
      learning_rate = 0.1,
      logging_level = "Silent",
      allow_writing_files = FALSE,
      random_seed = 42
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
    importances <- tryCatch(
      {
        scores <- catboost::catboost.get_feature_importance(model)
        stats::setNames(as.numeric(scores), colnames(x_train))
      },
      error = function(e) {
        NULL
      }
    )

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

# 4. LM/GLM Evaluator
register_evaluator(
  "lm",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    df_train <- as.data.frame(x_train)
    all_feats <- colnames(x_train)
    
    if (task == "regression") {
      df_train$target_y <- y_train
      model <- suppressWarnings(stats::lm(target_y ~ ., data = df_train))
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- suppressWarnings(stats::predict(model, as.data.frame(x_val)))
      }
      
      coefs <- summary(model)$coefficients
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances_all <- stats::setNames(rep(0, length(all_feats)), all_feats)
      if (nrow(coefs) > 0) {
        val_col <- if (ncol(coefs) >= 3) coefs[, 3] else coefs[, 1]
        valid_names <- intersect(rownames(coefs), all_feats)
        importances_all[valid_names] <- abs(val_col[valid_names])
      }
      importances_all[is.na(importances_all)] <- 0
      importances <- importances_all
      
    } else if (task == "classification") {
      df_train$target_y <- y_train
      model <- suppressWarnings(stats::glm(target_y ~ ., data = df_train, family = stats::binomial(link = "logit")))
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- suppressWarnings(stats::predict(model, as.data.frame(x_val), type = "response"))
      }
      
      coefs <- summary(model)$coefficients
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances_all <- stats::setNames(rep(0, length(all_feats)), all_feats)
      if (nrow(coefs) > 0) {
        val_col <- if (ncol(coefs) >= 3) coefs[, 3] else coefs[, 1]
        valid_names <- intersect(rownames(coefs), all_feats)
        importances_all[valid_names] <- abs(val_col[valid_names])
      }
      importances_all[is.na(importances_all)] <- 0
      importances <- importances_all
      
    } else if (task == "multiclass") {
      classes <- if (is.factor(y_train)) levels(y_train) else unique(y_train)
      models <- lapply(classes, function(cl) {
        y_bin <- as.numeric(y_train == cl)
        df_train_cl <- df_train
        df_train_cl$target_y <- y_bin
        suppressWarnings(stats::glm(target_y ~ ., data = df_train_cl, family = stats::binomial(link = "logit")))
      })
      names(models) <- as.character(classes)
      model <- models
      
      preds <- NULL
      if (!is.null(x_val)) {
        df_val <- as.data.frame(x_val)
        preds_list <- lapply(models, function(mod) {
          suppressWarnings(stats::predict(mod, df_val, type = "response"))
        })
        preds <- do.call(cbind, preds_list)
        row_sums <- rowSums(preds)
        row_sums[row_sums == 0] <- 1
        preds <- preds / row_sums
      }
      
      importances_all <- stats::setNames(rep(0, length(all_feats)), all_feats)
      for (mod in models) {
        coefs <- summary(mod)$coefficients
        coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
        if (nrow(coefs) > 0) {
          val_col <- if (ncol(coefs) >= 3) coefs[, 3] else coefs[, 1]
          valid_names <- intersect(rownames(coefs), all_feats)
          importances_all[valid_names] <- importances_all[valid_names] + abs(val_col[valid_names])
        }
      }
      importances_all <- importances_all / max(1, length(models))
      importances_all[is.na(importances_all)] <- 0
      importances <- importances_all
    }
    
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    df_new <- as.data.frame(x_new)
    if (task == "regression") {
      suppressWarnings(stats::predict(model, df_new))
    } else if (task == "classification") {
      suppressWarnings(stats::predict(model, df_new, type = "response"))
    } else if (task == "multiclass") {
      preds_list <- lapply(model, function(mod) {
        suppressWarnings(stats::predict(mod, df_new, type = "response"))
      })
      preds <- do.call(cbind, preds_list)
      row_sums <- rowSums(preds)
      row_sums[row_sums == 0] <- 1
      preds <- preds / row_sums
      preds
    }
  }
)
