# Bayesian Optimization Hyperparameter Tuner for LightGBM
#
# Evaluator that tunes LightGBM hyperparameters using Bayesian Optimization
# via 'mlr3mbo', 'paradox', and 'bbotk'.
#
# @importFrom stats predict
# @importFrom utils capture.output
# @importFrom stats setNames
# @importFrom stats runif
# @rawNamespace # NULL
register_evaluator(
  "lightgbm_mbo",
  train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                         task = "classification",
                         threads = 2, num_class = NULL, nrounds = 50,
                         metric = "default", mbo_iters = 5, mbo_init_design = 8,
                         mbo_folds = 3, mbo_infill_opt = "focussearch",
                         verbose = FALSE, best_params = NULL, ...) {
    
    # Check for required packages
    if (!requireNamespace("mlr3mbo", quietly = TRUE) ||
        !requireNamespace("paradox", quietly = TRUE) ||
        !requireNamespace("bbotk", quietly = TRUE)) {
      stop("The packages 'mlr3mbo', 'paradox', and 'bbotk' are required to use the 'lightgbm_mbo' evaluator. Please install them.")
    }
    
    # Determine strategy: use provided validation set or internal CV
    use_split <- !is.null(x_val) && !is.null(y_val)
    
    if (verbose) {
      metric_name <- if (is.function(metric)) "custom" else metric
      if (use_split) {
        message(sprintf("\n[MBO] Starting LightGBM Hyperparameter Tuning (Iters: %d, Strategy: split, Metric: %s)...", 
                        mbo_iters, metric_name))
      } else {
        message(sprintf("\n[MBO] Starting LightGBM Hyperparameter Tuning (Iters: %d, Strategy: cv-%d, Metric: %s)...", 
                        mbo_iters, mbo_folds, metric_name))
      }
    }
    
    if (!use_split) {
      # CV mode: create internal folds for MBO objective evaluation
      n_samples <- nrow(x_train)
      folds <- sample(rep(1:mbo_folds, length.out = n_samples))
    }
    
    # Objective function to evaluate hyperparameter sets
    fn <- function(x) {
      params <- list(
        objective = switch(task,
          classification = "binary",
          multiclass     = "multiclass",
          "regression"),
        num_leaves = x$num_leaves,
        learning_rate = x$learning_rate,
        max_depth = x$max_depth,
        feature_fraction = x$feature_fraction,
        num_threads = threads,
        verbose = -1
      )
      if (task == "multiclass") params$num_class <- num_class
      
      if (use_split) {
        # Split mode: train on x_train, evaluate on provided x_val/y_val
        dtrain <- lightgbm::lgb.Dataset(data = x_train, label = y_train)
        utils::capture.output({
          model <- lightgbm::lgb.train(
            params = params, data = dtrain, nrounds = nrounds, verbose = -1
          )
        })
        preds <- stats::predict(model, x_val)
        score <- compute_metric(y_val, preds, task, metric = metric, num_class = num_class)
        rm(dtrain)
        
        if (verbose) {
          message(sprintf("  [MBO Eval] lr=%.4f, leaves=%d, depth=%d, feat_frac=%.2f -> Val Fitness: %.4f", 
                          x$learning_rate, x$num_leaves, x$max_depth, x$feature_fraction, score))
        }
        return(score)
      } else {
        # CV mode: internal cross-validation on x_train
        scores <- numeric(mbo_folds)
        for (i in seq_len(mbo_folds)) {
          idx_train <- which(folds != i)
          idx_val <- which(folds == i)
          
          x_tr <- x_train[idx_train, , drop = FALSE]
          y_tr <- y_train[idx_train]
          x_va <- x_train[idx_val, , drop = FALSE]
          y_va <- y_train[idx_val]
          
          dtrain <- lightgbm::lgb.Dataset(data = x_tr, label = y_tr)
          utils::capture.output({
            model <- lightgbm::lgb.train(
              params = params, data = dtrain, nrounds = nrounds, verbose = -1
            )
          })
          
          preds <- stats::predict(model, x_va)
          scores[i] <- compute_metric(y_va, preds, task, metric = metric, num_class = num_class)
          rm(dtrain)
        }
        
        mean_score <- mean(scores)
        if (verbose) {
          message(sprintf("  [MBO Eval] lr=%.4f, leaves=%d, depth=%d, feat_frac=%.2f -> CV Fitness: %.4f", 
                          x$learning_rate, x$num_leaves, x$max_depth, x$feature_fraction, mean_score))
        }
        return(mean_score)
      }
    }
    
    # Parameter Set to optimize
    ps <- paradox::ps(
      learning_rate = paradox::p_dbl(lower = 0.01, upper = 0.3),
      num_leaves = paradox::p_int(lower = 7, upper = 63),
      max_depth = paradox::p_int(lower = 3, upper = 10),
      feature_fraction = paradox::p_dbl(lower = 0.5, upper = 1.0)
    )
    
    # Define bbotk objective function
    obj_fun_bbotk <- function(xs) {
      params_list <- lapply(xs, function(val) {
        if (is.factor(val)) as.character(val) else val
      })
      score <- fn(params_list)
      list(y = score)
    }
    
    codomain <- paradox::ps(y = paradox::p_dbl(tags = "maximize"))
    
    objective <- bbotk::ObjectiveRFun$new(
      fun = obj_fun_bbotk,
      domain = ps,
      codomain = codomain
    )
    
    if (!mbo_infill_opt %in% c("focussearch", "ea")) {
      stop("mbo_infill_opt must be either 'focussearch' or 'ea'.")
    }
    
    if (mbo_infill_opt == "ea") {
      warning("mbo_infill_opt = 'ea' is deprecated and ignored. mlr3mbo handles infill optimisation internally.", call. = FALSE)
    }
    
    instance <- bbotk::OptimInstanceBatchSingleCrit$new(
      objective = objective,
      terminator = bbotk::trm("evals", n_evals = mbo_init_design + mbo_iters)
    )
    
    # Generate initial design using Maximin Latin Hypercube Design (LHS)
    design <- paradox::generate_design_lhs(ps, n = mbo_init_design)$data
    if (!is.null(best_params)) {
      req_params <- c("learning_rate", "num_leaves", "max_depth", "feature_fraction")
      if (all(req_params %in% names(best_params))) {
        best_df <- data.table::as.data.table(best_params[req_params])
        for (col in names(best_df)) {
          if (is.factor(design[[col]])) {
            best_df[[col]] <- factor(as.character(best_df[[col]]), levels = levels(design[[col]]))
          } else if (is.integer(design[[col]])) {
            best_df[[col]] <- as.integer(best_df[[col]])
          } else if (is.numeric(design[[col]])) {
            best_df[[col]] <- as.numeric(best_df[[col]])
          }
        }
        design <- rbind(best_df, design)
        if (verbose) {
          message("[MBO] Seeding initial design with best parameters from previous tuning.")
        }
      }
    }
    
    # Evaluate initial design points first
    instance$eval_batch(design)
    
    # Run optimization
    optimizer <- bbotk::opt("mbo")
    optimizer$optimize(instance)
    
    # Extract best hyperparams
    best_params <- as.list(instance$result[, ps$ids(), with = FALSE])
    best_params <- lapply(best_params, function(val) {
      if (is.factor(val)) as.character(val) else val
    })
    
    if (verbose) {
      message(sprintf("[MBO] Optimization Complete. Best Parameters: lr=%.4f, leaves=%d, depth=%d, feat_frac=%.2f", 
                      best_params$learning_rate, best_params$num_leaves, best_params$max_depth, best_params$feature_fraction))
    }
    
    # Train final model on full dataset with best parameters
    dtrain_full <- lightgbm::lgb.Dataset(data = x_train, label = y_train)
    final_params <- list(
      objective = switch(task,
        classification = "binary",
        multiclass     = "multiclass",
        "regression"),
      metric = switch(task,
        classification = "binary_logloss",
        multiclass     = "multi_logloss",
        "rmse"),
      num_leaves    = best_params$num_leaves,
      learning_rate = best_params$learning_rate,
      max_depth     = best_params$max_depth,
      feature_fraction = best_params$feature_fraction,
      num_threads   = threads,
      verbose       = -1
    )
    if (task == "multiclass") final_params$num_class <- num_class
    
    utils::capture.output({
      final_model <- lightgbm::lgb.train(
        params = final_params, data = dtrain_full, nrounds = nrounds, verbose = -1
      )
    })
    
    preds <- if (!is.null(x_val)) stats::predict(final_model, x_val) else NULL
    
    imp <- tryCatch({
      lightgbm::lgb.importance(final_model, percentage = TRUE)
    }, error = function(e) {
      data.frame(Feature = character(), Gain = numeric())
    })
    importances <- if (nrow(imp) > 0) stats::setNames(imp$Gain, imp$Feature) else NULL
    
    rm(dtrain_full)
    list(model = final_model, predictions = preds, importances = importances, best_params = best_params)
  },
  predict_func = function(model, x_new, task, ...) {
    stats::predict(model, x_new)
  },
  base_evaluator = "lightgbm"
)
