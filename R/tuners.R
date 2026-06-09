# Bayesian Optimization Hyperparameter Tuner for LightGBM
#
# Evaluator that tunes LightGBM hyperparameters using Bayesian Optimization
# via 'mlrMBO', 'ParamHelpers', and 'smoof'.
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
    if (!requireNamespace("mlrMBO", quietly = TRUE) ||
        !requireNamespace("ParamHelpers", quietly = TRUE) ||
        !requireNamespace("smoof", quietly = TRUE)) {
      stop("The packages 'mlrMBO', 'ParamHelpers', and 'smoof' are required to use the 'lightgbm_mbo' evaluator. Please install them.")
    }
    
    # Determine strategy: use provided validation set or internal CV
    use_split <- !is.null(x_val) && !is.null(y_val)
    
    if (verbose) {
      if (use_split) {
        message(sprintf("\n[MBO] Starting LightGBM Hyperparameter Tuning (Iters: %d, Strategy: split, Metric: %s)...", 
                        mbo_iters, metric))
      } else {
        message(sprintf("\n[MBO] Starting LightGBM Hyperparameter Tuning (Iters: %d, Strategy: cv-%d, Metric: %s)...", 
                        mbo_iters, mbo_folds, metric))
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
        return(-score)
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
        return(-mean_score)
      }
    }
    
    # Parameter Set to optimize
    ps <- ParamHelpers::makeParamSet(
      ParamHelpers::makeNumericParam("learning_rate", lower = 0.01, upper = 0.3),
      ParamHelpers::makeIntegerParam("num_leaves", lower = 7, upper = 63),
      ParamHelpers::makeIntegerParam("max_depth", lower = 3, upper = 10),
      ParamHelpers::makeNumericParam("feature_fraction", lower = 0.5, upper = 1.0)
    )
    
    # Define smoof objective function
    obj_fun <- smoof::makeSingleObjectiveFunction(
      name = "lightgbm_hyperparameter_tuning",
      fn = fn,
      par.set = ps,
      has.simple.signature = FALSE,
      minimize = TRUE
    )
    
    if (!mbo_infill_opt %in% c("focussearch", "ea")) {
      stop("mbo_infill_opt must be either 'focussearch' or 'ea'.")
    }
    if (mbo_infill_opt == "ea" && !requireNamespace("emoa", quietly = TRUE)) {
      stop("The package 'emoa' is required to use the 'ea' infill optimizer. Please install it.")
    }
    control <- mlrMBO::makeMBOControl()
    control <- mlrMBO::setMBOControlTermination(control, iters = mbo_iters)
    control <- mlrMBO::setMBOControlInfill(control, opt = mbo_infill_opt)
    
    # Generate initial design using Maximin Latin Hypercube Design (LHS)
    design <- ParamHelpers::generateDesign(n = mbo_init_design, par.set = ps, fun = lhs::maximinLHS)
    if (!is.null(best_params)) {
      req_params <- c("learning_rate", "num_leaves", "max_depth", "feature_fraction")
      if (all(req_params %in% names(best_params))) {
        best_df <- data.frame(
          learning_rate = as.numeric(best_params$learning_rate),
          num_leaves = as.integer(best_params$num_leaves),
          max_depth = as.integer(best_params$max_depth),
          feature_fraction = as.numeric(best_params$feature_fraction)
        )
        design <- rbind(best_df, design)
        if (verbose) {
          message("[MBO] Seeding initial design with best parameters from previous tuning.")
        }
      }
    }
    
    # Run bayesian optimization
    # Try Kriging (GP) surrogate first; fall back to Random Forest if Kriging
    # encounters numerical singularities (common on small datasets)
    mbo_res <- tryCatch({
      mlrMBO::mbo(obj_fun, design = design, control = control, show.info = verbose)
    }, error = function(e) {
      if (!requireNamespace("mlr", quietly = TRUE) || !requireNamespace("randomForest", quietly = TRUE)) {
        stop(sprintf("[MBO] Kriging surrogate failed, and fallback Random Forest surrogate is unavailable because suggested packages 'mlr' and/or 'randomForest' are not installed. Original Kriging error: %s", conditionMessage(e)))
      }
      if (verbose) {
        message("[MBO] Kriging surrogate failed, falling back to Random Forest surrogate.")
      }
      rf_surrogate <- mlr::makeLearner("regr.randomForest", predict.type = "se")
      suppressWarnings(
        mlrMBO::mbo(obj_fun, design = design, learner = rf_surrogate,
                    control = control, show.info = verbose)
      )
    })
    best_params <- mbo_res$x
    
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
