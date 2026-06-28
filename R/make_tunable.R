#' Create a Tunable Evaluator from a Registered Base Model
#'
#' @description
#' Wraps an existing registered model evaluator in a Bayesian Optimization tuning loop
#' using the \pkg{mlr3mbo} framework. It automatically generates a parameter space, constructs
#' a cross-validation or split-validation objective function, searches for the optimal
#' hyperparameters, and registers the tuned evaluator.
#'
#' @details
#' The tuning loop uses a Latin Hypercube Design (LHS) for the initial parameters layout. It uses
#' the \pkg{mlr3mbo} package to run Bayesian Optimization to optimize hyperparameters.
#'
#' Evaluators registered via \code{make_tunable} accept several control parameters passed via \code{...}:
#' \describe{
#'   \item{\code{mbo_iters}}{Integer: Number of Bayesian Optimization iterations (default 5).}
#'   \item{\code{mbo_init_design}}{Integer: Number of initial layout designs generated (default 8).}
#'   \item{\code{mbo_folds}}{Integer: Number of internal CV folds used for evaluation when no validation split is provided (default 3).}
#'   \item{\code{mbo_infill_opt}}{Character: Strategy for infill optimization to find the next candidate parameter set. Supported values are \code{"focussearch"} (default) and \code{"ea"} (deprecated).}
#'   \item{\code{best_params}}{List: Optional list of initial parameters to seed the MBO search.}
#' }
#'
#' @param base_model_name Character. Name of the registered base evaluator
#'   (e.g., \code{"xgboost"}, \code{"lightgbm"}).
#' @param param_ranges List. A nested list defining the parameter names, types, and bounds/values.
#'   Each parameter definition must be a list containing:
#'   \describe{
#'     \item{\code{type}}{Character: \code{"numeric"}, \code{"integer"}, or \code{"discrete"}.}
#'     \item{\code{lower}}{Numeric/Integer: Lower bound of the search space
#'       (required for \code{"numeric"} and \code{"integer"}).}
#'     \item{\code{upper}}{Numeric/Integer: Upper bound of the search space
#'       (required for \code{"numeric"} and \code{"integer"}).}
#'     \item{\code{values}}{Vector: Set of valid values (required for \code{"discrete"}).}
#'   }
#' @param tuner_name Character. The name under which to register the tuned evaluator. Defaults to
#'   \code{paste0(base_model_name, "_mbo")}.
#'
#' @return Invisibly returns \code{NULL}. Registers the tuned evaluator in the global \code{evo_evaluators} environment.
#'
#' @importFrom paradox ps p_dbl p_int p_fct generate_design_lhs
#' @importFrom bbotk ObjectiveRFun OptimInstanceBatchSingleCrit trm opt
#' @importFrom lhs maximinLHS
#' @importFrom utils modifyList
#'
#' @examples
#' \dontrun{
#' # 1. Register a simple mock evaluator
#' register_evaluator(
#'   "mock_base",
#'   train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
#'                         task = "regression", ...) {
#'     args <- list(...)
#'     val_score <- 100 - abs(args$param_a - 4.5)
#'     list(
#'       model = list(args = args, val_score = val_score),
#'       predictions = if (!is.null(x_val)) {
#'         rep(val_score, nrow(x_val))
#'       } else {
#'         NULL
#'       }
#'     )
#'   },
#'   predict_func = function(model, x_new, task, ...) {
#'     rep(model$val_score, nrow(x_new))
#'   }
#' )
#'
#' # 2. Make it tunable
#' param_ranges <- list(
#'   param_a = list(type = "numeric", lower = 1.0, upper = 8.0)
#' )
#' make_tunable("mock_base", param_ranges, tuner_name = "mock_tuned")
#'
#' # 3. Train the tuned model on mock data
#' x_train <- matrix(rnorm(20), ncol = 2)
#' colnames(x_train) <- c("x1", "x2")
#' y_train <- rnorm(10)
#' x_val <- matrix(rnorm(10), ncol = 2)
#' y_val <- rnorm(5)
#'
#' fit <- train_model(
#'   x_train, y_train, x_val = x_val, y_val = y_val,
#'   task = "regression", evaluator = "mock_tuned",
#'   mbo_iters = 3, mbo_init_design = 5, mbo_folds = 2
#' )
#' print(fit$best_params)
#' }
#' @export
make_tunable <- function(base_model_name, param_ranges, tuner_name = paste0(base_model_name, "_mbo")) {
  # 1. Retrieve the base model configuration
  if (!exists(base_model_name, envir = evo_evaluators)) {
    stop(sprintf("Model '%s' is not registered in evo_evaluators. Registered models: %s", 
                 base_model_name, paste(names(evo_evaluators), collapse = ", ")))
  }
  base_evaluator <- evo_evaluators[[base_model_name]]
  
  # 2. Separate tunable and fixed parameters
  tunable_defs <- list()
  fixed_params <- list()
  
  for (pname in names(param_ranges)) {
    def <- param_ranges[[pname]]
    if (isFALSE(def$tunable)) {
      if (!"value" %in% names(def)) {
        stop(sprintf("Parameter '%s' is marked as not tunable (tunable=FALSE) but has no 'value'.", pname))
      }
      fixed_params[[pname]] <- def$value
    } else {
      tunable_defs[[pname]] <- def
    }
  }

  # 3. Convert tunable parameter ranges list into a paradox ParamSet
  make_param <- function(name, def) {
    if (def$type == "numeric") {
      paradox::p_dbl(lower = def$lower, upper = def$upper)
    } else if (def$type == "integer") {
      paradox::p_int(lower = def$lower, upper = def$upper)
    } else if (def$type == "discrete") {
      paradox::p_fct(levels = as.character(def$values))
    } else {
      stop(sprintf("Unsupported parameter type: %s", def$type))
    }
  }
  
  param_list <- lapply(names(tunable_defs), function(n) make_param(n, tunable_defs[[n]]))
  names(param_list) <- names(tunable_defs)
  ps <- do.call(paradox::ps, param_list)
  
  # 4. Create a dynamic training function wrapper
  tuned_train_func <- function(x_train, y_train, x_val = NULL, y_val = NULL,
                               task = "classification", threads = 2, num_class = NULL,
                               metric = "default", mbo_iters = 5, mbo_init_design = 8,
                               mbo_folds = 3, mbo_infill_opt = "focussearch",
                               verbose = FALSE, best_params = NULL, ...) {
    
    # Check for required packages
    if (!requireNamespace("mlr3mbo", quietly = TRUE) ||
        !requireNamespace("paradox", quietly = TRUE) ||
        !requireNamespace("bbotk", quietly = TRUE)) {
      stop("The packages 'mlr3mbo', 'paradox', and 'bbotk' are required to use the tuned evaluator. Please install them.")
    }

    use_split <- !is.null(x_val) && !is.null(y_val)
    
    if (verbose) {
      metric_name <- if (is.function(metric)) "custom" else metric
      if (use_split) {
        message(sprintf("\n[MBO] Starting %s Hyperparameter Tuning (Iters: %d, Strategy: split, Metric: %s)...", 
                        base_model_name, mbo_iters, metric_name))
      } else {
        message(sprintf("\n[MBO] Starting %s Hyperparameter Tuning (Iters: %d, Strategy: cv-%d, Metric: %s)...", 
                        base_model_name, mbo_iters, mbo_folds, metric_name))
      }
    }

    if (!use_split) {
      folds <- sample(rep(1:mbo_folds, length.out = nrow(x_train)))
    }
    
    # Define optimization objective function
    fn <- function(x) {
      # Extract parameters from the MBO proposal and merge with fixed_params and extra args
      trial_params <- utils::modifyList(list(...), fixed_params)
      trial_params <- utils::modifyList(trial_params, x)
      
      if (use_split) {
        # Train on split and compute target validation metric
        res <- do.call(base_evaluator$train_func, c(
          list(x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task, 
               threads = threads, num_class = num_class, metric = metric, verbose = FALSE),
          trial_params
        ))
        score <- compute_metric(y_val, res$predictions, task, metric = metric, num_class = num_class)
        
        if (verbose) {
          param_str <- paste0(names(x), "=", unlist(x), collapse = ", ")
          message(sprintf("  [MBO Eval] %s -> Val Fitness: %.4f", param_str, score))
        }
        return(score)
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
        
        mean_score <- mean(scores)
        if (verbose) {
          param_str <- paste0(names(x), "=", unlist(x), collapse = ", ")
          message(sprintf("  [MBO Eval] %s -> CV Fitness: %.4f", param_str, mean_score))
        }
        return(mean_score)
      }
    }
    
    # 4. Run bbotk/mlr3mbo optimization
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
    
    # Generate initial design
    design <- paradox::generate_design_lhs(ps, n = mbo_init_design)$data
    
    # Seed with previous best_params if provided
    if (!is.null(best_params)) {
      req_params <- names(tunable_defs)
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
    best_hyperparams <- as.list(instance$result[, ps$ids(), with = FALSE])
    best_hyperparams <- lapply(best_hyperparams, function(val) {
      if (is.factor(val)) as.character(val) else val
    })
    
    if (verbose) {
      message(sprintf("[MBO] Optimization Complete for %s. Best parameters: %s",
                      base_model_name, paste0(names(best_hyperparams), "=", unlist(best_hyperparams), collapse = ", ")))
    }
    
    # 5. Train final model with the optimal parameters on the full dataset
    final_params <- utils::modifyList(list(...), fixed_params)
    final_params <- utils::modifyList(final_params, best_hyperparams)
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
    predict_func = base_evaluator$predict_func,
    base_evaluator = base_model_name
  )
}
