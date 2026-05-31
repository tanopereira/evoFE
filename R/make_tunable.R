#' Create a Tunable Evaluator from a Registered Base Model
#'
#' @param base_model_name Character. Name of the registered base evaluator (e.g., "xgboost").
#' @param param_ranges List. A nested list defining parameter bounds and types.
#'   Example:
#'   \code{
#'   list(
#'     learning_rate = list(type = "numeric", lower = 0.01, upper = 0.3),
#'     max_depth = list(type = "integer", lower = 3, upper = 10)
#'   )
#'   }
#' @param tuner_name Character. The name under which to register the tuned evaluator.
#' @importFrom mlrMBO makeMBOControl setMBOControlTermination mbo
#' @importFrom ParamHelpers makeParamSet makeNumericParam makeIntegerParam makeDiscreteParam generateDesign
#' @importFrom smoof makeSingleObjectiveFunction
#' @importFrom lhs maximinLHS
#' @importFrom utils modifyList
#' @export
make_tunable <- function(base_model_name, param_ranges, tuner_name = paste0(base_model_name, "_mbo")) {
  # 1. Retrieve the base model configuration
  if (!exists(base_model_name, envir = evo_evaluators)) {
    stop(sprintf("Model '%s' is not registered in evo_evaluators. Registered models: %s", 
                 base_model_name, paste(names(evo_evaluators), collapse = ", ")))
  }
  base_evaluator <- evo_evaluators[[base_model_name]]
  
  # 2. Convert user parameter ranges list into a ParamHelpers ParamSet
  make_param <- function(name, def) {
    if (def$type == "numeric") {
      ParamHelpers::makeNumericParam(name, lower = def$lower, upper = def$upper)
    } else if (def$type == "integer") {
      ParamHelpers::makeIntegerParam(name, lower = def$lower, upper = def$upper)
    } else if (def$type == "discrete") {
      ParamHelpers::makeDiscreteParam(name, values = def$values)
    } else {
      stop(sprintf("Unsupported parameter type: %s", def$type))
    }
  }
  
  ps <- do.call(ParamHelpers::makeParamSet, 
                lapply(names(param_ranges), function(n) make_param(n, param_ranges[[n]])))
  
  # 3. Create a dynamic training function wrapper
  tuned_train_func <- function(x_train, y_train, x_val = NULL, y_val = NULL,
                               task = "classification", threads = 2, num_class = NULL,
                               metric = "default", mbo_iters = 5, mbo_init_design = 8,
                               mbo_folds = 3, verbose = FALSE, best_params = NULL, ...) {
    
    # Check for required packages
    if (!requireNamespace("mlrMBO", quietly = TRUE) ||
        !requireNamespace("ParamHelpers", quietly = TRUE) ||
        !requireNamespace("smoof", quietly = TRUE)) {
      stop("The packages 'mlrMBO', 'ParamHelpers', and 'smoof' are required to use the tuned evaluator. Please install them.")
    }

    use_split <- !is.null(x_val) && !is.null(y_val)
    if (!use_split) {
      set.seed(42)
      folds <- sample(rep(1:mbo_folds, length.out = nrow(x_train)))
    }
    
    # Define optimization objective function
    fn <- function(x) {
      # Merge tuned parameters (x) with static parameters (...)
      trial_params <- utils::modifyList(list(...), x)
      
      if (use_split) {
        # Train on split and compute target validation metric
        res <- do.call(base_evaluator$train_func, c(
          list(x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task, 
               threads = threads, num_class = num_class, metric = metric, verbose = FALSE),
          trial_params
        ))
        score <- compute_metric(y_val, res$predictions, task, metric = metric, num_class = num_class)
        return(-score) # Negate because MBO minimizes
      } else {
        # Cross-validation mode
        scores <- numeric(mbo_folds)
        for (i in seq_len(mbo_folds)) {
          idx_train <- which(folds != i)
          idx_val <- which(folds == i)
          
          res <- do.call(base_evaluator$train_func, c(
            list(x_train = x_train[idx_train, , drop = FALSE], 
                 y_train = y_train[idx_train], 
                 x_val = x_train[idx_val, , drop = FALSE], 
                 y_val = y_train[idx_val],
                 task = task, threads = threads, num_class = num_class, 
                 metric = metric, verbose = FALSE),
            trial_params
          ))
          scores[i] <- compute_metric(y_train[idx_val], res$predictions, task, metric = metric, num_class = num_class)
        }
        return(-mean(scores))
      }
    }
    
    # 4. Run mlrMBO optimization
    obj_fun <- smoof::makeSingleObjectiveFunction(
      name = paste0(base_model_name, "_mbo_tuning"),
      fn = fn, par.set = ps, has.simple.signature = FALSE, minimize = TRUE
    )
    
    control <- mlrMBO::makeMBOControl()
    control <- mlrMBO::setMBOControlTermination(control, iters = mbo_iters)
    
    # Generate initial design
    design <- ParamHelpers::generateDesign(n = mbo_init_design, par.set = ps, fun = lhs::maximinLHS)
    
    # Seed with previous best_params if provided
    if (!is.null(best_params)) {
      req_params <- names(param_ranges)
      if (all(req_params %in% names(best_params))) {
        best_df <- as.data.frame(best_params[req_params])
        design <- rbind(best_df, design)
        if (verbose) {
          message("[MBO] Seeding initial design with best parameters from previous tuning.")
        }
      }
    }
    
    # Run optimization (fall back to Random Forest surrogate on Kriging errors)
    mbo_res <- tryCatch({
      mlrMBO::mbo(obj_fun, design = design, control = control, show.info = verbose)
    }, error = function(e) {
      if (verbose) {
        message("[MBO] Kriging surrogate failed, falling back to Random Forest surrogate.")
      }
      rf_surrogate <- mlr::makeLearner("regr.randomForest", predict.type = "se")
      suppressWarnings(
        mlrMBO::mbo(obj_fun, design = design, learner = rf_surrogate,
                    control = control, show.info = verbose)
      )
    })
    
    best_hyperparams <- mbo_res$x
    
    if (verbose) {
      message(sprintf("[MBO] Optimization Complete for %s. Best parameters: %s",
                      base_model_name, paste0(names(best_hyperparams), "=", unlist(best_hyperparams), collapse = ", ")))
    }
    
    # 5. Train final model with the optimal parameters on the full dataset
    final_params <- utils::modifyList(list(...), best_hyperparams)
    final_res <- do.call(base_evaluator$train_func, c(
      list(x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task, 
           threads = threads, num_class = num_class, metric = metric, verbose = verbose),
      final_params
    ))
    
    final_res$best_params <- best_hyperparams
    return(final_res)
  }
  
  # 6. Register under the new tuner name
  register_evaluator(
    name = tuner_name,
    train_func = tuned_train_func,
    predict_func = base_evaluator$predict_func
  )
}
