#' Global environment for registered model evaluators
#'
#' @return An \code{environment} containing registered model evaluators.
#' @export
evo_evaluators <- new.env(parent = emptyenv())

# Internal helper to read the best iteration from a trained xgboost model,
# supporting both the legacy API (attribute on the object) and xgboost >= 3.0
# (where early stopping info lives in model$early_stop).
.xgb_best_iter <- function(model) {
  if (!is.null(model$best_iteration)) {
    return(model$best_iteration)
  }
  if (!is.null(model$early_stop) && !is.null(model$early_stop$best_iteration)) {
    bi <- model$early_stop$best_iteration
    if (is.numeric(bi) && is.finite(bi) && bi > 0) return(bi)
    return(NULL)
  }
  val <- tryCatch(xgboost::xgb.attr(model, "best_iteration"), error = function(e) NULL)
  if (!is.null(val)) as.numeric(val) else NULL
}

# Internal helper to unify SHAP values across models
.extract_shap_importances <- function(sh, num_feats, feature_names) {
  if (length(dim(sh)) == 3) {
    # 3D Array: [N, num_classes, num_feats + 1] (CatBoost/XGBoost Multiclass)
    sh_feats <- sh[, , 1:num_feats, drop = FALSE]
    sh_sum <- apply(abs(sh_feats), c(1, 3), sum)
    imp <- colMeans(sh_sum)
  } else if (length(dim(sh)) == 2) {
    if (ncol(sh) == num_feats + 1) {
      # 2D Matrix: [N, num_feats + 1] (Regression/Binary)
      sh_feats <- sh[, 1:num_feats, drop = FALSE]
      imp <- colMeans(abs(sh_feats))
    } else {
      # 2D Matrix: [N, (num_feats + 1) * num_classes] (LightGBM Multiclass)
      num_classes <- ncol(sh) / (num_feats + 1)
      imp <- numeric(num_feats)
      for (c in seq_len(num_classes)) {
        cols <- (c - 1) * (num_feats + 1) + seq_len(num_feats)
        sh_feats <- sh[, cols, drop = FALSE]
        imp <- imp + colMeans(abs(sh_feats))
      }
    }
  } else {
    imp <- numeric(num_feats)
  }
  stats::setNames(as.numeric(imp), feature_names)
}
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
#' @param cleanup_func Optional function to clean up model resources/states after evaluation.
#' @return Invisible NULL. Called for the side effect of registering the evaluator in \code{evo_evaluators}.
#' @importFrom lightgbm lgb.train
#' @importFrom xgboost xgb.train
#' @examples
#' # Register a simple mock evaluator
#' register_evaluator(
#'   "mock_eval",
#'   train_func = function(x_train, y_train, x_val = NULL,
#'                         task = "regression", ...) {
#'     list(
#'       model = list(weights = colMeans(x_train)),
#'       predictions = if (!is.null(x_val)) rowMeans(x_val) else NULL,
#'       importances = stats::setNames(
#'         rep(1, ncol(x_train)), colnames(x_train)
#'       )
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
register_evaluator <- function(name, train_func, predict_func, base_evaluator = NULL, cleanup_func = NULL) {
  assign(name, list(train_func = train_func, predict_func = predict_func, base_evaluator = base_evaluator, cleanup_func = cleanup_func), envir = evo_evaluators)
}

# --- Register Default Evaluators ---

# 1. LightGBM Evaluator
register_evaluator(
  "lightgbm",
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("lightgbm", quietly = TRUE)) {
      stop("The 'lightgbm' package is required to use the 'lightgbm' evaluator. Please install it.")
    }
    x_train <- .sanitize_feature_matrix(x_train)
    x_val   <- .sanitize_feature_matrix(x_val)
    dtrain <- lightgbm::lgb.Dataset(data = x_train, label = y_train)
    extra_params <- list(...)
    y_val <- extra_params$y_val
    metric_arg <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    is_mae_metric <- !is.null(metric_arg) && is.character(metric_arg) && tolower(metric_arg) %in% c("mae", "cal_mae", "cal-mae")
    reg_obj <- if (is_mae_metric) "regression_l1" else "regression"
    reg_metric <- if (is_mae_metric) "mae" else "rmse"

    params <- list(
      objective = switch(task,
        classification = "binary",
        multiclass     = "multiclass",
        reg_obj
      ),
      metric = switch(task,
        classification = "binary_logloss",
        multiclass     = "multi_logloss",
        reg_metric
      ),
      num_leaves = 15,
      learning_rate = 0.1,
      verbose = -1,
      num_threads = threads,
      seed = 42
    )
    if (task == "multiclass") params$num_class <- num_class

    use_custom_eval <- FALSE
    custom_eval_type <- NULL
    if (!is.null(metric_arg) && is.character(metric_arg)) {
      metric_lower <- tolower(metric_arg)
      if (metric_lower %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "ts_refinement"
      } else if (metric_lower %in% c("cal_rmse", "cal-rmse")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_rmse"
      } else if (metric_lower %in% c("cal_mae", "cal-mae")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_mae"
      }
    }

    if (use_custom_eval) {
      params$metric <- "None"
    }

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds", "realmlp_device")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    valids <- list()
    if (!is.null(x_val) && !is.null(y_val)) {
      dval <- lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain)
      valids$val <- dval
    }

    lgb_eval <- function(preds, dtrain) {
      labels <- lightgbm::get_field(dtrain, "label")
      if (custom_eval_type == "ts_refinement") {
        score <- compute_ts_refinement(labels, preds, task = task, num_class = num_class, is_logits = FALSE)
        list(name = "ts_refinement", value = score, higher_better = FALSE)
      } else if (custom_eval_type == "cal_rmse") {
        score <- compute_calibrated_rmse(labels, preds)
        list(name = "cal_rmse", value = score, higher_better = FALSE)
      } else { # cal_mae
        score <- compute_calibrated_mae(labels, preds)
        list(name = "cal_mae", value = score, higher_better = FALSE)
      }
    }

    # early_stopping_rounds requires at least one validation dataset
    esr <- if (length(valids) > 0) early_stopping_rounds else NULL

    utils::capture.output({
      model <- lightgbm::lgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        valids = valids,
        eval = if (use_custom_eval) lgb_eval else NULL,
        early_stopping_rounds = esr,
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

    importances <- tryCatch(
      {
        sh <- stats::predict(model, as.matrix(x_train), type = "contrib")
        .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
      },
      error = function(e) {
        NULL
      }
    )

    rm(dtrain)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("lightgbm", quietly = TRUE)) {
      stop("The 'lightgbm' package is required to use the 'lightgbm' evaluator. Please install it.")
    }
    x_new <- .sanitize_feature_matrix(x_new)
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
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("The 'xgboost' package is required to use the 'xgboost' evaluator. Please install it.")
    }
    x_train <- .sanitize_feature_matrix(x_train)
    x_val   <- .sanitize_feature_matrix(x_val)
    dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train, missing = NA)
    extra_params <- list(...)
    y_val <- extra_params$y_val
    metric_arg <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    is_mae_metric <- !is.null(metric_arg) && is.character(metric_arg) && tolower(metric_arg) %in% c("mae", "cal_mae", "cal-mae")
    reg_obj <- if (is_mae_metric) "reg:absoluteerror" else "reg:squarederror"
    reg_eval_metric <- if (is_mae_metric) "mae" else "rmse"

    params <- list(
      objective = switch(task,
        classification = "binary:logistic",
        multiclass     = "multi:softprob",
        reg_obj
      ),
      eval_metric = switch(task,
        classification = "logloss",
        multiclass     = "mlogloss",
        reg_eval_metric
      ),
      nthread = threads,
      max_depth = 6,
      eta = 0.1,
      min_child_weight = 1,
      seed = 42,
      verbosity = 0,
      silent = 1
    )
    if (task == "multiclass") params$num_class <- num_class

    use_custom_eval <- FALSE
    custom_eval_type <- NULL
    if (!is.null(metric_arg) && is.character(metric_arg)) {
      metric_lower <- tolower(metric_arg)
      if (metric_lower %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "ts_refinement"
      } else if (metric_lower %in% c("cal_rmse", "cal-rmse")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_rmse"
      } else if (metric_lower %in% c("cal_mae", "cal-mae")) {
        use_custom_eval <- TRUE
        custom_eval_type <- "cal_mae"
      }
    }

    if (use_custom_eval) {
      params$eval_metric <- NULL
    }

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters", "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds", "realmlp_device")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    if (identical(params$booster, "gblinear")) {
      params$nthread <- 1
    }

    evals <- list(train = dtrain)
    dval_metric <- NULL
    # Early stopping requires a validation set in `evals`; add one whenever
    # validation data is available and a custom metric or early stopping is used.
    needs_val <- use_custom_eval ||
      (!is.null(early_stopping_rounds) && early_stopping_rounds > 0)
    if (needs_val && !is.null(x_val) && !is.null(y_val)) {
      dval_metric <- xgboost::xgb.DMatrix(data = x_val, label = y_val, missing = NA)
      evals$val <- dval_metric
    }
    if (length(evals) <= 1L) {
      # No validation set available: disable early stopping (mirrors lightgbm path)
      early_stopping_rounds <- NULL
    }

    xgb_feval <- function(preds, dtrain) {
      labels <- xgboost::getinfo(dtrain, "label")
      if (custom_eval_type == "ts_refinement") {
        score <- compute_ts_refinement(labels, preds, task = task, num_class = num_class, is_logits = TRUE)
        list(metric = "ts_refinement", value = score)
      } else if (custom_eval_type == "cal_rmse") {
        score <- compute_calibrated_rmse(labels, preds)
        list(metric = "cal_rmse", value = score)
      } else { # cal_mae
        score <- compute_calibrated_mae(labels, preds)
        list(metric = "cal_mae", value = score)
      }
    }

    utils::capture.output({
      model <- suppressWarnings(xgboost::xgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        evals = evals,
        custom_metric = if (use_custom_eval) xgb_feval else NULL,
        early_stopping_rounds = early_stopping_rounds,
        maximize = if (use_custom_eval) FALSE else NULL,
        verbose = 0
      ))
    })

    best_iter <- .xgb_best_iter(model)

    preds <- if (!is.null(x_val)) {
      dval <- xgboost::xgb.DMatrix(data = x_val, missing = NA)
      p <- if (!is.null(best_iter) && best_iter >= 1) {
        stats::predict(model, dval, iterationrange = c(1, best_iter + 1))
      } else {
        stats::predict(model, dval)
      }
      rm(dval)
      p
    } else {
      NULL
    }

    importances <- tryCatch(
      {
        sh <- stats::predict(model, dtrain, predcontrib = TRUE)
        .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
      },
      error = function(e) {
        NULL
      }
    )

    rm(dtrain)
    if (!is.null(dval_metric)) rm(dval_metric)
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("xgboost", quietly = TRUE)) {
      stop("The 'xgboost' package is required to use the 'xgboost' evaluator. Please install it.")
    }
    x_new <- .sanitize_feature_matrix(x_new)
    best_iter <- .xgb_best_iter(model)
    dmatrix <- xgboost::xgb.DMatrix(data = x_new, missing = NA)
    preds <- if (!is.null(best_iter) && best_iter >= 1) {
      stats::predict(model, dmatrix, iterationrange = c(1, best_iter + 1))
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

    x_train <- .sanitize_feature_matrix(x_train)
    x_val   <- .sanitize_feature_matrix(x_val)

    df_train <- as.data.frame(x_train)
    df_train[] <- lapply(df_train, as.numeric)
    dtrain <- catboost::catboost.load_pool(data = df_train, label = y_train)

    extra_params <- list(...)
    y_val            <- extra_params$y_val
    metric_arg       <- extra_params$metric
    early_stopping_rounds <- extra_params$early_stopping_rounds

    is_mae_metric <- !is.null(metric_arg) && is.character(metric_arg) && tolower(metric_arg) %in% c("mae", "cal_mae", "cal-mae")
    reg_loss <- if (is_mae_metric) "MAE" else "RMSE"

    params <- list(
      loss_function = switch(task,
        classification = "Logloss",
        multiclass     = "MultiClass",
        reg_loss
      ),
      thread_count = threads,
      iterations = nrounds,
      learning_rate = 0.1,
      logging_level = "Silent",
      allow_writing_files = FALSE,
      train_dir = tempdir(),
      random_seed = 42
    )

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters",
                        "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds",
                        "device", "realmlp_device")
    extra_params <- extra_params[!names(extra_params) %in% control_params]
    for (name in names(extra_params)) {
      params[[name]] <- extra_params[[name]]
    }

    # Force no file writing to working directory (required for CRAN compliance)
    params$allow_writing_files <- FALSE
    params$train_dir <- tempdir()

    # Build validation pool for early stopping when possible
    dval_es <- NULL
    if (!is.null(x_val) && !is.null(y_val) && !is.null(early_stopping_rounds) && early_stopping_rounds > 0) {
      df_val_es <- as.data.frame(x_val)
      df_val_es[] <- lapply(df_val_es, as.numeric)
      dval_es <- catboost::catboost.load_pool(data = df_val_es, label = y_val)
      params$od_type <- "Iter"
      params$od_wait <- early_stopping_rounds
    }

    model <- suppressWarnings(catboost::catboost.train(dtrain, test_pool = dval_es, params = params))

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
        sh <- catboost::catboost.get_feature_importance(model, pool = dtrain, type = "ShapValues")
        .extract_shap_importances(sh, ncol(x_train), colnames(x_train))
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
    x_new <- .sanitize_feature_matrix(x_new)
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

## 4. LM/GLM Evaluator (Penalized)
register_evaluator(
  "lm",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("The 'glmnet' package is required for the penalized lm evaluator.")
    }
    
    x_mat <- as.matrix(x_train)
    col_means <- colMeans(x_mat, na.rm = TRUE)
    col_means[is.na(col_means) | !is.finite(col_means)] <- 0
    for (j in seq_len(ncol(x_mat))) {
      na_idx <- is.na(x_mat[, j]) | !is.finite(x_mat[, j])
      if (any(na_idx)) {
        x_mat[na_idx, j] <- col_means[j]
      }
    }

    impute_mat <- function(x_data) {
      if (is.null(x_data)) return(NULL)
      xm <- as.matrix(x_data)
      for (j in seq_len(ncol(xm))) {
        col_name <- colnames(xm)[j]
        c_mean <- if (!is.null(col_name) && col_name %in% names(col_means)) col_means[[col_name]] else 0
        na_idx <- is.na(xm[, j]) | !is.finite(xm[, j])
        if (any(na_idx)) {
          xm[na_idx, j] <- c_mean
        }
      }
      xm
    }

    all_feats <- colnames(x_train)
    nfolds <- min(5, nrow(x_mat))
    
    # Precompute standard deviations to make importances scale-invariant
    sd_x <- apply(x_mat, 2, stats::sd, na.rm = TRUE)
    sd_x[is.na(sd_x) | sd_x == 0] <- 1
    
    if (task == "regression") {
      model <- glmnet::cv.glmnet(x_mat, y_train, family = "gaussian", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- as.numeric(stats::predict(model, newx = impute_mat(x_val), s = "lambda.min"))
      }
      
      coefs <- as.matrix(stats::coef(model, s = "lambda.min"))
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
      importances <- importances * sd_x[names(importances)]
      
    } else if (task == "classification") {
      model <- glmnet::cv.glmnet(x_mat, as.factor(y_train), family = "binomial", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- as.numeric(stats::predict(model, newx = impute_mat(x_val), s = "lambda.min", type = "response"))
      }
      
      coefs <- as.matrix(stats::coef(model, s = "lambda.min"))
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
      importances <- importances * sd_x[names(importances)]
      
    } else if (task == "multiclass") {
      model <- glmnet::cv.glmnet(x_mat, as.factor(y_train), family = "multinomial", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        p_array <- stats::predict(model, newx = impute_mat(x_val), s = "lambda.min", type = "response")
        preds <- p_array[, , 1]
      }
      
      coefs_list <- stats::coef(model, s = "lambda.min")
      imp_matrix <- sapply(coefs_list, function(c_mat) {
        c_mat <- as.matrix(c_mat)
        abs(c_mat[rownames(c_mat) != "(Intercept)", , drop = FALSE])
      })
      importances <- rowMeans(imp_matrix)
      importances <- stats::setNames(importances, rownames(imp_matrix))
      importances <- importances * sd_x[names(importances)]
    }
    
    # Ensure missing names map to 0
    missing <- setdiff(all_feats, names(importances))
    if (length(missing) > 0) {
      importances <- c(importances, stats::setNames(rep(0, length(missing)), missing))
    }
    
    # Attach training-time imputation means to the model so predict-time
    # imputation is consistent with training (prevents train/serve skew).
    model$col_means <- col_means
    
    list(model = model, predictions = preds, importances = importances, col_means = col_means)
  },
  predict_func = function(model, x_new, task, ...) {
    x_mat <- as.matrix(x_new)
    col_means <- if (!is.null(model$col_means)) model$col_means else colMeans(x_mat, na.rm = TRUE)
    col_means[is.na(col_means) | !is.finite(col_means)] <- 0
    for (j in seq_len(ncol(x_mat))) {
      col_name <- colnames(x_mat)[j]
      c_mean <- if (!is.null(col_name) && col_name %in% names(col_means)) col_means[[col_name]] else 0
      na_idx <- is.na(x_mat[, j]) | !is.finite(x_mat[, j])
      if (any(na_idx)) {
        x_mat[na_idx, j] <- c_mean
      }
    }
    if (task == "regression") {
      as.numeric(stats::predict(model, newx = x_mat, s = "lambda.min"))
    } else if (task == "classification") {
      as.numeric(stats::predict(model, newx = x_mat, s = "lambda.min", type = "response"))
    } else if (task == "multiclass") {
      p_array <- stats::predict(model, newx = x_mat, s = "lambda.min", type = "response")
      p_array[, , 1]
    }
  }
)

# 5. Keras 3 Feed-Forward Neural Network Evaluator
register_evaluator(
  "keras3",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("keras3", quietly = TRUE)) {
      stop("The 'keras3' package is required for the keras3 evaluator.")
    }
    
    # Ensure deterministic weight initialization for stable fitness evaluations,
    # but restore the R global random state so we don't freeze the genetic algorithm!
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
    keras3::set_random_seed(42)
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else {
      rm(".Random.seed", envir = .GlobalEnv)
    }
    
    # Extract tunable parameters from ... with defaults
    extra_params <- list(...)
    nn_layers <- if (!is.null(extra_params$nn_layers)) extra_params$nn_layers else 2
    nn_units <- if (!is.null(extra_params$nn_units)) extra_params$nn_units else 64
    nn_dropout <- if (!is.null(extra_params$nn_dropout)) extra_params$nn_dropout else 0.2
    nn_lr <- if (!is.null(extra_params$nn_learning_rate)) extra_params$nn_learning_rate else 0.001
    
    # Ensure data is matrix
    x_train <- as.matrix(x_train)
    
    # Define model architecture
    model <- keras3::keras_model_sequential(input_shape = ncol(x_train))
    model <- keras3::layer_dense(model, units = nn_units, activation = "relu")
    if (nn_dropout > 0) model <- keras3::layer_dropout(model, rate = nn_dropout)
    
    if (nn_layers > 1) {
      for (i in 2:nn_layers) {
        # Reduce units in subsequent layers, bounded at 16
        units_next <- max(16, as.integer(nn_units / (2 ^ (i - 1))))
        model <- keras3::layer_dense(model, units = units_next, activation = "relu")
        if (nn_dropout > 0) model <- keras3::layer_dropout(model, rate = nn_dropout)
      }
    }
    
    # Task-specific output layer and compilation
    optimizer <- keras3::optimizer_adam(learning_rate = nn_lr)
    
    if (task == "regression") {
      model <- keras3::layer_dense(model, units = 1)
      keras3::compile(model, optimizer = optimizer, loss = "mse")
    } else if (task == "classification") {
      model <- keras3::layer_dense(model, units = 1, activation = "sigmoid")
      keras3::compile(model, optimizer = optimizer, loss = "binary_crossentropy")
    } else if (task == "multiclass") {
      model <- keras3::layer_dense(model, units = num_class, activation = "softmax")
      # Note: expects y_train to be integer 0 to num_class-1
      keras3::compile(model, optimizer = optimizer, loss = "sparse_categorical_crossentropy")
    }
    
    # Prepare validation data if provided
    y_val <- extra_params$y_val
    early_stopping_rounds <- extra_params$early_stopping_rounds
    val_data <- NULL
    callbacks <- list()
    if (!is.null(x_val) && !is.null(y_val)) {
      val_data <- list(as.matrix(x_val), y_val)
      if (!is.null(early_stopping_rounds) && early_stopping_rounds > 0) {
        callbacks <- list(
          keras3::callback_early_stopping(
            monitor = "val_loss",
            patience = early_stopping_rounds,
            restore_best_weights = TRUE
          )
        )
      }
    }
    
    # Train
    history <- keras3::fit(
      model,
      x = x_train,
      y = y_train,
      epochs = nrounds,
      validation_data = val_data,
      callbacks = callbacks,
      verbose = 0
    )
    
    # Predict on validation
    preds <- NULL
    if (!is.null(x_val)) {
      x_val_mat <- as.matrix(x_val)
      if (task == "regression" || task == "classification") {
        preds <- as.numeric(stats::predict(model, x_val_mat, verbose = 0))
      } else if (task == "multiclass") {
        # Return probability matrix
        preds <- as.matrix(stats::predict(model, x_val_mat, verbose = 0))
      }
    }
    
    # NNs don't have native feature importances, return NULL
    list(model = model, predictions = preds, importances = NULL)
  },
  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("keras3", quietly = TRUE)) {
      stop("The 'keras3' package is required for the keras3 evaluator.")
    }
    x_mat <- as.matrix(x_new)
    if (task == "regression" || task == "classification") {
      as.numeric(stats::predict(model, x_mat, verbose = 0))
    } else if (task == "multiclass") {
      as.matrix(stats::predict(model, x_mat, verbose = 0))
    }
  },
  cleanup_func = function(model) {
    if (requireNamespace("keras3", quietly = TRUE)) {
      if (exists("clear_session", where = asNamespace("keras3"))) {
        keras3::clear_session()
      } else if (exists("k_clear_session", where = asNamespace("keras3"))) {
        # Fallback for some versions
        tryCatch(keras3::k_clear_session(), error = function(e) NULL)
      }
    }
  }
)

# Internal session environment for RealMLP evaluator state
.realmlp_env <- new.env(parent = emptyenv())

#' Get or create progress-enabled RealMLP classes lazily
#' @noRd
.get_progress_realmlp_classes <- function() {
  if (!is.null(.realmlp_env$ProgressRealMLPRegressor) && !is.null(.realmlp_env$ProgressRealMLPClassifier)) {
    return(list(
      ProgressRealMLPRegressor = .realmlp_env$ProgressRealMLPRegressor,
      ProgressRealMLPClassifier = .realmlp_env$ProgressRealMLPClassifier
    ))
  }

  realmlp_ns <- asNamespace("realmlp")

  ProgressSimpleMLP <- R6::R6Class(
    "ProgressSimpleMLP",
    inherit = realmlp_ns$SimpleMLP,
    public = list(
      show_log = FALSE,
      is_detail = FALSE,
      task_name = "regression",
      early_stopping_rounds = NULL,
      stopped_early = FALSE,
      stopped_epoch = 256L,

      fit = function(X, y, X_val = NULL, y_val = NULL) {
        stopifnot(is.matrix(X))
        X <- as.matrix(X)
        storage.mode(X) <- "double"
        input_dim <- ncol(X)
        is_cls <- self$is_classification

        if (is_cls) {
          y_fac <- as.factor(y)
          self$classes_ <- levels(y_fac)
          y_idx <- as.integer(y_fac)
          output_dim <- length(self$classes_)
        } else {
          y_mat <- if (is.matrix(y) || is.data.frame(y)) as.matrix(y) else matrix(as.numeric(y), ncol = 1)
          storage.mode(y_mat) <- "double"
          self$y_mean_ <- colMeans(y_mat)
          self$y_std_ <- apply(y_mat, 2, stats::sd)
          y_std_safe <- self$y_std_ + 1e-30
          y_mat <- sweep(y_mat, 2, self$y_mean_, "-")
          y_mat <- sweep(y_mat, 2, y_std_safe, "/")
          output_dim <- ncol(y_mat)
          if (!is.null(y_val)) {
            yv <- if (is.matrix(y_val) || is.data.frame(y_val)) as.matrix(y_val) else matrix(as.numeric(y_val), ncol = 1)
            storage.mode(yv) <- "double"
            yv <- sweep(yv, 2, self$y_mean_, "-")
            y_val <- sweep(yv, 2, y_std_safe, "/")
          }
        }

        act <- if (is_cls) torch::nn_selu else realmlp_ns$Mish
        model <- torch::nn_sequential(
          realmlp_ns$ScalingLayer(input_dim),
          realmlp_ns$NTPLinear(input_dim, 256),
          act(),
          realmlp_ns$NTPLinear(256, 256),
          act(),
          realmlp_ns$NTPLinear(256, 256),
          act(),
          realmlp_ns$NTPLinear(256, output_dim, zero_init = TRUE)
        )$to(device = self$device)

        criterion <- if (is_cls) {
          realmlp_ns$make_classification_loss(label_smoothing = 0.1)
        } else {
          function(pred, target) torch::nnf_mse_loss(pred, target, reduction = "mean")
        }

        params <- model$parameters
        scale_params <- list(params[[1]])
        weights <- params[seq(2, length(params), by = 2)]
        biases <- params[seq(3, length(params), by = 2)]
        opt <- torch::optim_adam(
          params = list(list(params = scale_params), list(params = weights), list(params = biases)),
          betas = c(0.9, 0.95)
        )

        x_train <- torch::torch_tensor(X, dtype = torch::torch_float())
        y_train <- if (is_cls) {
          torch::torch_tensor(y_idx, dtype = torch::torch_long())
        } else {
          torch::torch_tensor(y_mat, dtype = torch::torch_float())
        }

        if (!is.null(X_val) && !is.null(y_val)) {
          X_val <- as.matrix(X_val)
          storage.mode(X_val) <- "double"
          x_valid <- torch::torch_tensor(X_val, dtype = torch::torch_float())
          y_valid <- if (is_cls) {
            yv_fac <- factor(y_val, levels = self$classes_)
            yv_idx <- as.integer(yv_fac)
            yv_idx[is.na(yv_idx)] <- 1L
            torch::torch_tensor(yv_idx, dtype = torch::torch_long())
          } else {
            torch::torch_tensor(as.matrix(y_val), dtype = torch::torch_float())
          }
        } else {
          x_valid <- x_train[1:0, ]
          y_valid <- if (is_cls) y_train[1:0] else y_train[1:0, ]
        }

        n_train <- x_train$size()[1]
        n_valid <- x_valid$size()[1]
        n_epochs <- 256L
        train_batch_size <- as.integer(min(256L, n_train))
        n_train_batches <- as.integer(floor(n_train / train_batch_size))
        if (n_train_batches < 1L) n_train_batches <- 1L
        valid_batch_size <- as.integer(max(1L, min(1024L, n_valid)))
        base_lr <- if (is_cls) 0.04 else 0.07

        best_valid_loss <- Inf
        best_valid_params <- NULL
        no_improve <- 0L

        valid_metric <- function(y_pred, y_true) {
          if (is_cls) {
            pred_idx <- y_pred$argmax(dim = 2)
            as.numeric((pred_idx != y_true)$sum()$item()) / max(1L, length(y_true))
          } else {
            as.numeric(torch::nnf_mse_loss(y_pred, y_true, reduction = "mean")$item())
          }
        }

        log_interval <- if (self$is_detail) 32L else 64L

        for (epoch in 0:(n_epochs - 1L)) {
          model$train()
          perm <- sample.int(n_train, size = n_train, replace = FALSE)
          epoch_loss_sum <- 0.0

          for (batch_idx in 0:(n_train_batches - 1L)) {
            start <- batch_idx * train_batch_size + 1L
            end <- min(start + train_batch_size - 1L, n_train)
            if (end > n_train) break
            idx <- perm[start:end]

            x_batch <- x_train[idx, ]$to(device = self$device)
            y_batch <- if (is_cls) y_train[idx]$to(device = self$device) else y_train[idx, ]$to(device = self$device)

            t <- (epoch * n_train_batches + batch_idx) / (n_epochs * n_train_batches)
            lr_sched_value <- 0.5 - 0.5 * cos(2 * pi * log2(1 + 15 * t))
            lr <- base_lr * lr_sched_value
            opt$param_groups[[1]]$lr <- 6 * lr
            opt$param_groups[[2]]$lr <- lr
            opt$param_groups[[3]]$lr <- 0.1 * lr

            opt$zero_grad()
            y_pred <- model(x_batch)
            loss <- criterion(y_pred, y_batch)
            loss$backward()
            opt$step()
            epoch_loss_sum <- epoch_loss_sum + as.numeric(loss$item())
          }

          avg_train_loss <- epoch_loss_sum / n_train_batches
          valid_loss <- NA_real_

          model$eval()
          torch::with_no_grad({
            if (n_valid > 0) {
              preds <- list()
              for (start in seq(1L, n_valid, by = valid_batch_size)) {
                end <- min(n_valid, start + valid_batch_size - 1L)
                xb <- x_valid[start:end, ]$to(device = self$device)
                preds[[length(preds) + 1L]] <- model(xb)$detach()$cpu()
              }
              y_pred_valid <- torch::torch_cat(preds, dim = 1)
              valid_loss <- valid_metric(y_pred_valid, y_valid$to(device = torch::torch_device("cpu")))

              if (valid_loss <= best_valid_loss) {
                best_valid_loss <- valid_loss
                best_valid_params <- lapply(model$parameters, function(p) p$detach()$clone())
                no_improve <- 0L
              } else {
                no_improve <- no_improve + 1L
              }
            }
          })

          cur_epoch <- epoch + 1L
          should_log <- self$show_log && (
            cur_epoch %% log_interval == 0 ||
            cur_epoch == 1L ||
            cur_epoch == n_epochs ||
            (self$is_detail && !is.na(valid_loss) && no_improve == 0L)
          )

          if (should_log) {
            if (!is.na(valid_loss)) {
              metric_label <- if (is_cls) "Val error" else "Val loss"
              best_star <- if (no_improve == 0L) "*" else ""
              message(sprintf("    [RealMLP %s] Epoch %3d/%d | Train loss: %.4f | %s: %.4f (best: %.4f%s)",
                              self$task_name, cur_epoch, n_epochs, avg_train_loss, metric_label, valid_loss, best_valid_loss, best_star))
            } else {
              message(sprintf("    [RealMLP %s] Epoch %3d/%d | Train loss: %.4f",
                              self$task_name, cur_epoch, n_epochs, avg_train_loss))
            }
          }

          # Early stopping
          if (!is.null(self$early_stopping_rounds) && self$early_stopping_rounds > 0 && n_valid > 0) {
            if (no_improve >= self$early_stopping_rounds) {
              self$stopped_early <- TRUE
              self$stopped_epoch <- cur_epoch
              if (self$show_log) {
                message(sprintf("    [RealMLP %s] Early stopping triggered at epoch %d (patience = %d, best val: %.4f)",
                                self$task_name, cur_epoch, self$early_stopping_rounds, best_valid_loss))
              }
              break
            }
          }
        }

        torch::with_no_grad({
          if (!is.null(best_valid_params)) {
            for (i in seq_along(model$parameters)) {
              model$parameters[[i]]$set_(best_valid_params[[i]])
            }
          }
        })

        self$model_ <- model
        invisible(self)
      }
    )
  )

  ProgressRealMLPRegressor <- R6::R6Class(
    "ProgressRealMLPRegressor",
    inherit = realmlp_ns$Standalone_RealMLP_TD_S_Regressor,
    public = list(
      show_log = FALSE,
      is_detail = FALSE,
      early_stopping_rounds = NULL,
      stopped_early = FALSE,
      stopped_epoch = 256L,

      fit = function(X, y, X_val = NULL, y_val = NULL) {
        self$prep_ <- realmlp_ns$get_realmlp_td_s_pipeline()
        self$model_ <- ProgressSimpleMLP$new(is_classification = FALSE, device = self$device)
        self$model_$show_log <- self$show_log
        self$model_$is_detail <- self$is_detail
        self$model_$task_name <- "regression"
        self$model_$early_stopping_rounds <- self$early_stopping_rounds

        Xp <- realmlp_ns$prep_fit_transform(self$prep_, X)
        Xvp <- if (!is.null(X_val)) realmlp_ns$prep_transform(self$prep_, X_val) else NULL

        self$model_$fit(Xp, y, X_val = Xvp, y_val = y_val)
        self$stopped_early <- self$model_$stopped_early
        self$stopped_epoch <- self$model_$stopped_epoch
        invisible(self)
      }
    )
  )

  ProgressRealMLPClassifier <- R6::R6Class(
    "ProgressRealMLPClassifier",
    inherit = realmlp_ns$Standalone_RealMLP_TD_S_Classifier,
    public = list(
      show_log = FALSE,
      is_detail = FALSE,
      task_name = "classification",
      early_stopping_rounds = NULL,
      stopped_early = FALSE,
      stopped_epoch = 256L,

      fit = function(X, y, X_val = NULL, y_val = NULL) {
        self$prep_ <- realmlp_ns$get_realmlp_td_s_pipeline()
        self$model_ <- ProgressSimpleMLP$new(is_classification = TRUE, device = self$device)
        self$model_$show_log <- self$show_log
        self$model_$is_detail <- self$is_detail
        self$model_$task_name <- self$task_name
        self$model_$early_stopping_rounds <- self$early_stopping_rounds

        Xp <- realmlp_ns$prep_fit_transform(self$prep_, X)
        Xvp <- if (!is.null(X_val)) realmlp_ns$prep_transform(self$prep_, X_val) else NULL

        self$model_$fit(Xp, y, X_val = Xvp, y_val = y_val)
        self$classes_ <- self$model_$classes_
        self$stopped_early <- self$model_$stopped_early
        self$stopped_epoch <- self$model_$stopped_epoch
        invisible(self)
      }
    )
  )

  .realmlp_env$ProgressRealMLPRegressor <- ProgressRealMLPRegressor
  .realmlp_env$ProgressRealMLPClassifier <- ProgressRealMLPClassifier

  list(
    ProgressRealMLPRegressor = ProgressRealMLPRegressor,
    ProgressRealMLPClassifier = ProgressRealMLPClassifier
  )
}

# 6. RealMLP Evaluator (frankiethull/realmlp with torch)
register_evaluator(
  "realmlp",
  train_func = function(x_train, y_train, x_val = NULL, task = "regression",
                        threads = 2, num_class = NULL, nrounds = 50, ...) {
    if (!requireNamespace("realmlp", quietly = TRUE)) {
      stop("The 'realmlp' package is required for the 'realmlp' evaluator. Please install it via pak::pak('frankiethull/realmlp').")
    }

    if (!is.null(threads) && is.numeric(threads) && threads >= 1) {
      target_th <- as.integer(threads)
      # Pre-set OMP_NUM_THREADS before torch initializes its native thread pool
      if (!"torch" %in% loadedNamespaces()) {
        Sys.setenv(OMP_NUM_THREADS = as.character(target_th))
      }
    }

    if (!requireNamespace("torch", quietly = TRUE)) {
      stop("The 'torch' package is required for the 'realmlp' evaluator. Please install it via install.packages('torch') and torch::install_torch().")
    }
    if (!"package:torch" %in% search()) {
      suppressPackageStartupMessages(require("torch", quietly = TRUE, character.only = TRUE))
    }

    if (!is.null(threads) && is.numeric(threads) && threads >= 1) {
      target_th <- as.integer(threads)
      # libtorch ParallelNative only allows setting threads once before parallel work starts.
      # Guard against repeatedly calling torch_set_num_threads to avoid ParallelNative C++ warnings.
      if (!identical(.realmlp_env$last_set_threads, target_th)) {
        .realmlp_env$last_set_threads <- target_th
        if (torch::torch_get_num_threads() != target_th) {
          try(torch::torch_set_num_threads(target_th), silent = TRUE)
        }
      }
    }

    extra_params <- list(...)
    y_val <- extra_params$y_val
    early_stopping_rounds <- extra_params$early_stopping_rounds

    # Ensure deterministic torch weight initialization for stable fitness evaluations,
    # and restore torch RNG state on exit so caller session state is unaffected (CRAN-compliant)
    old_torch_state <- if (requireNamespace("torch", quietly = TRUE)) tryCatch(torch::torch_get_rng_state(), error = function(e) NULL) else NULL
    if (!is.null(old_torch_state)) {
      on.exit(tryCatch(torch::torch_set_rng_state(old_torch_state), error = function(e) NULL), add = TRUE)
    }

    seed_val <- if (!is.null(extra_params$seed)) as.integer(extra_params$seed) else 42L
    if (requireNamespace("torch", quietly = TRUE)) {
      torch::torch_manual_seed(seed_val)
    }

    opt_verbose <- getOption("evoFE.verbose", 0)
    verbose_arg <- if (!is.null(extra_params$verbose)) extra_params$verbose else opt_verbose
    show_log <- isTRUE(verbose_arg) || (is.numeric(verbose_arg) && verbose_arg >= 1) || isTRUE(opt_verbose) || (is.numeric(opt_verbose) && opt_verbose >= 1)
    is_detail <- isTRUE(verbose_arg >= 2) || isTRUE(opt_verbose >= 2)

    # Respect early_stopping_rounds consistent with LightGBM and XGBoost
    use_es <- !is.null(early_stopping_rounds) && early_stopping_rounds > 0 && !is.null(x_val) && !is.null(y_val)

    device <- if (!is.null(extra_params$realmlp_device)) {
      extra_params$realmlp_device
    } else if (!is.null(extra_params$device)) {
      extra_params$device
    } else {
      "cpu"
    }

    x_train <- .sanitize_feature_matrix(x_train)
    x_val   <- .sanitize_feature_matrix(x_val)

    df_train <- as.data.frame(x_train)
    df_val   <- if (!is.null(x_val)) as.data.frame(x_val) else NULL

    # Safeguard against residual NAs before torch tensor ingestion
    col_meds <- vapply(df_train, function(col) {
      m <- stats::median(col[!is.na(col) & is.finite(col)])
      if (is.na(m) || !is.finite(m)) 0 else m
    }, numeric(1))

    impute_df <- function(df) {
      if (is.null(df)) return(NULL)
      for (nm in names(df)) {
        na_mask <- is.na(df[[nm]]) | !is.finite(df[[nm]])
        if (any(na_mask)) df[[nm]][na_mask] <- col_meds[[nm]]
      }
      df
    }

    df_train <- impute_df(df_train)
    df_val   <- impute_df(df_val)

    realmlp_ns <- asNamespace("realmlp")
    t0 <- Sys.time()

    if (show_log) {
      es_str <- if (use_es) sprintf(" (early stopping patience=%d)", early_stopping_rounds) else ""
      msg_start <- sprintf("    [RealMLP %s] Starting training: %d rows (%d features) on %s (256 epochs%s)...",
                           task, nrow(df_train), ncol(df_train), device, es_str)
      message(msg_start)
    }

    classes <- .get_progress_realmlp_classes()

    # 1. Regression
    if (task == "regression") {
      net <- classes$ProgressRealMLPRegressor$new(device = device)
      net$show_log <- show_log
      net$is_detail <- is_detail
      net$early_stopping_rounds <- if (use_es) early_stopping_rounds else NULL

      net$fit(
        X = df_train,
        y = as.numeric(y_train),
        X_val = if (use_es) df_val else NULL,
        y_val = if (use_es) as.numeric(y_val) else NULL
      )

      preds <- if (!is.null(df_val)) as.numeric(net$predict(df_val)) else NULL

    # 2. Classification (Binary & Multiclass)
    } else if (task %in% c("classification", "multiclass")) {
      net <- classes$ProgressRealMLPClassifier$new(device = device)
      net$show_log <- show_log
      net$is_detail <- is_detail
      net$task_name <- task
      net$early_stopping_rounds <- if (use_es) early_stopping_rounds else NULL

      levels_target <- if (task == "classification") {
        if (is.factor(y_train)) levels(y_train) else c(0, 1)
      } else if (task == "multiclass") {
        if (!is.null(num_class)) seq(0, num_class - 1) else if (is.factor(y_train)) levels(y_train) else sort(unique(y_train))
      }

      y_train_fac <- factor(y_train, levels = levels_target)
      y_val_fac   <- if (!is.null(y_val)) factor(y_val, levels = levels_target) else NULL

      net$fit(
        X = df_train,
        y = y_train_fac,
        X_val = if (use_es) df_val else NULL,
        y_val = if (use_es) y_val_fac else NULL
      )

      preds <- NULL
      if (!is.null(df_val)) {
        probs <- net$predict_proba(df_val)
        if (task == "classification") {
          pos_idx <- which(as.character(net$classes_) == "1")
          if (length(pos_idx) == 0 && is.matrix(probs) && ncol(probs) >= 2) {
            pos_idx <- 2
          }
          preds <- if (length(pos_idx) == 1 && is.matrix(probs)) {
            probs[, pos_idx]
          } else {
            as.numeric(probs)
          }
        } else {
          probs <- as.matrix(probs)
          expected_cols <- as.character(levels_target)
          cur_cols <- as.character(net$classes_)
          if (length(expected_cols) > 0 && !identical(cur_cols, expected_cols) && all(expected_cols %in% cur_cols)) {
            probs <- probs[, match(expected_cols, cur_cols), drop = FALSE]
          }
          preds <- probs
        }
      }
    } else {
      stop(sprintf("Unsupported task '%s' for RealMLP evaluator.", task))
    }

    if (show_log) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      es_status <- if (isTRUE(net$stopped_early)) {
        sprintf("stopped at epoch %d (patience=%d)", net$stopped_epoch, early_stopping_rounds)
      } else if (use_es) {
        sprintf("patience=%d", early_stopping_rounds)
      } else {
        "OFF"
      }
      msg <- sprintf("    [RealMLP %s] Fitted %d rows on %s (Early stopping: %s, %.3fs)",
                     task, nrow(df_train), device, es_status, elapsed)
      if (is_detail) {
        msg <- sprintf("%s [Features: %d]", msg, ncol(df_train))
      }
      message(msg)
    }

    importances <- tryCatch({
      m <- net$model_$model_
      s <- as.numeric(m[[1]]$scale$abs()$cpu())
      w1 <- as.matrix(m[[2]]$weight$cpu())
      imp <- s * sqrt(rowSums(w1^2))
      if (length(imp) == ncol(df_train)) {
        names(imp) <- colnames(df_train)
        sum_imp <- sum(imp)
        if (is.finite(sum_imp) && sum_imp > 0) {
          imp <- imp / sum_imp
        }
        imp
      } else {
        NULL
      }
    }, error = function(e) NULL)

    wrapped_model <- list(
      net = net,
      col_meds = col_meds,
      levels_target = if (task %in% c("classification", "multiclass")) levels_target else NULL
    )

    list(model = wrapped_model, predictions = preds, importances = importances)
  },

  predict_func = function(model, x_new, task, ...) {
    if (!requireNamespace("realmlp", quietly = TRUE)) {
      stop("The 'realmlp' package is required for the 'realmlp' evaluator.")
    }
    if (!"package:torch" %in% search()) {
      suppressPackageStartupMessages(require("torch", quietly = TRUE, character.only = TRUE))
    }
    net <- model$net
    col_meds <- model$col_meds
    x_new <- .sanitize_feature_matrix(x_new)
    df_new <- as.data.frame(x_new)

    if (!is.null(col_meds)) {
      for (nm in names(df_new)) {
        if (nm %in% names(col_meds)) {
          na_mask <- is.na(df_new[[nm]]) | !is.finite(df_new[[nm]])
          if (any(na_mask)) df_new[[nm]][na_mask] <- col_meds[[nm]]
        }
      }
    }

    if (task == "regression") {
      as.numeric(net$predict(df_new))
    } else if (task == "classification") {
      probs <- net$predict_proba(df_new)
      pos_idx <- which(as.character(net$classes_) == "1")
      if (length(pos_idx) == 0 && is.matrix(probs) && ncol(probs) >= 2) {
        pos_idx <- 2
      }
      if (length(pos_idx) == 1 && is.matrix(probs)) {
        probs[, pos_idx]
      } else {
        as.numeric(probs)
      }
    } else if (task == "multiclass") {
      probs <- as.matrix(net$predict_proba(df_new))
      expected_cols <- as.character(model$levels_target)
      cur_cols <- as.character(net$classes_)
      if (length(expected_cols) > 0 && !identical(cur_cols, expected_cols) && all(expected_cols %in% cur_cols)) {
        probs <- probs[, match(expected_cols, cur_cols), drop = FALSE]
      }
      probs
    }
  },

  cleanup_func = function(model) {
    if (requireNamespace("torch", quietly = TRUE)) {
      if (torch::cuda_is_available()) torch::cuda_empty_cache()
      gc(verbose = FALSE)
    }
  }
)

