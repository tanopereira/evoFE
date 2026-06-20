#' Global environment for registered model evaluators
#'
#' @export
evo_evaluators <- new.env(parent = emptyenv())

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
    if (!is.null(x_val) && !is.null(y_val)) {
      dval <- lightgbm::lgb.Dataset(data = x_val, label = y_val, reference = dtrain)
      valids$val <- dval
    }

    lgb_eval <- function(preds, dtrain) {
      labels <- lightgbm::get_field(dtrain, "label")
      score <- compute_ts_refinement(labels, preds, task = task, num_class = num_class, is_logits = FALSE)
      list(name = "ts_refinement", value = score, higher_better = FALSE)
    }

    # early_stopping_rounds requires at least one validation dataset
    esr <- if (length(valids) > 0) early_stopping_rounds else NULL

    utils::capture.output({
      model <- lightgbm::lgb.train(
        params = params,
        data = dtrain,
        nrounds = nrounds,
        valids = valids,
        eval = if (use_ts_refinement) lgb_eval else NULL,
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
      train_dir = tempdir(),
      random_seed = 42
    )

    extra_params <- list(...)
    y_val            <- extra_params$y_val
    early_stopping_rounds <- extra_params$early_stopping_rounds

    control_params <- c("verbose", "metric", "best_params", "y_val", "mbo_iters",
                        "mbo_init_design", "mbo_folds", "mbo_infill_opt", "early_stopping_rounds")
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
    all_feats <- colnames(x_train)
    nfolds <- min(5, nrow(x_mat))
    
    # Precompute standard deviations to make importances scale-invariant
    sd_x <- apply(x_mat, 2, stats::sd, na.rm = TRUE)
    sd_x[is.na(sd_x) | sd_x == 0] <- 1
    
    if (task == "regression") {
      model <- glmnet::cv.glmnet(x_mat, y_train, family = "gaussian", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- as.numeric(stats::predict(model, newx = as.matrix(x_val), s = "lambda.min"))
      }
      
      coefs <- as.matrix(stats::coef(model, s = "lambda.min"))
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
      importances <- importances * sd_x[names(importances)]
      
    } else if (task == "classification") {
      model <- glmnet::cv.glmnet(x_mat, as.factor(y_train), family = "binomial", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        preds <- as.numeric(stats::predict(model, newx = as.matrix(x_val), s = "lambda.min", type = "response"))
      }
      
      coefs <- as.matrix(stats::coef(model, s = "lambda.min"))
      coefs <- coefs[rownames(coefs) != "(Intercept)", , drop = FALSE]
      importances <- stats::setNames(abs(as.numeric(coefs)), rownames(coefs))
      importances <- importances * sd_x[names(importances)]
      
    } else if (task == "multiclass") {
      model <- glmnet::cv.glmnet(x_mat, as.factor(y_train), family = "multinomial", alpha = 0.5, nfolds = nfolds)
      
      preds <- NULL
      if (!is.null(x_val)) {
        p_array <- stats::predict(model, newx = as.matrix(x_val), s = "lambda.min", type = "response")
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
    
    list(model = model, predictions = preds, importances = importances)
  },
  predict_func = function(model, x_new, task, ...) {
    x_mat <- as.matrix(x_new)
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
