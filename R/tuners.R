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
  train_func = function(x_train, y_train, x_val = NULL, task = "classification",
                         threads = 2, num_class = NULL, nrounds = 50,
                         metric = "default", mbo_iters = 5, mbo_init_design = 8,
                         mbo_folds = 3, ...) {
    
    # Check for required packages
    if (!requireNamespace("mlrMBO", quietly = TRUE) ||
        !requireNamespace("ParamHelpers", quietly = TRUE) ||
        !requireNamespace("smoof", quietly = TRUE)) {
      stop("The packages 'mlrMBO', 'ParamHelpers', and 'smoof' are required to use the 'lightgbm_mbo' evaluator. Please install them.")
    }
    
    # Define simple folds for internal cross-validation to guide tuning
    set.seed(42)
    n_samples <- nrow(x_train)
    folds <- sample(rep(1:mbo_folds, length.out = n_samples))
    
    # Objective function to evaluate hyperparameter sets
    fn <- function(x) {
      scores <- numeric(mbo_folds)
      for (i in seq_len(mbo_folds)) {
        idx_train <- which(folds != i)
        idx_val <- which(folds == i)
        
        x_tr <- x_train[idx_train, , drop = FALSE]
        y_tr <- y_train[idx_train]
        x_va <- x_train[idx_val, , drop = FALSE]
        y_va <- y_train[idx_val]
        
        dtrain <- lightgbm::lgb.Dataset(data = x_tr, label = y_tr)
        
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
        
        utils::capture.output({
          model <- lightgbm::lgb.train(
            params = params, data = dtrain, nrounds = nrounds, verbose = -1
          )
        })
        
        preds <- stats::predict(model, x_va)
        scores[i] <- compute_metric(y_va, preds, task, metric = metric, num_class = num_class)
        rm(dtrain)
      }
      
      # mlrMBO minimizes by default, and compute_metric is higher-is-better (fitness).
      # Thus, return the negative of the mean score.
      return(-mean(scores))
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
    
    # Configure MBO Control
    control <- mlrMBO::makeMBOControl()
    control <- mlrMBO::setMBOControlTermination(control, iters = mbo_iters)
    
    # Generate initial design
    design <- ParamHelpers::generateDesign(n = mbo_init_design, par.set = ps)
    
    # Configure robust surrogate model (regr.lm with standard error for numerical stability)
    surrogate <- mlr::makeLearner("regr.lm", predict.type = "se")
    
    # Run bayesian optimization
    mbo_res <- mlrMBO::mbo(obj_fun, design = design, learner = surrogate, control = control, show.info = FALSE)
    best_params <- mbo_res$x
    
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
  }
)
