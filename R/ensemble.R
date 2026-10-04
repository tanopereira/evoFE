#' Caruana, Stacked, and Equal-Weight Island Ensembling
#'
#' Performs ensemble selection over the validation/out-of-fold predictions from evolved
#' islands, creating an optimal multi-model ensemble. Three methods are available:
#' Caruana greedy forward selection with replacement (default), non-negative
#' elastic-net stacking with an honest nested cross-validated performance estimate,
#' or equal/uniform weighting ($w_j = 1/K$).
#'
#' @param recipe An \code{evo_recipe} object produced by \code{\link{evolve_features}}.
#' @param data A data.frame or data.table containing the original training data used
#'   during evolution. Required for lazy final model training of surviving islands.
#' @param target_col Character string. Name of the target column. If \code{NULL},
#'   it is inferred from the recipe or data.
#' @param method Character string. Ensembling strategy: \code{"caruana"} (greedy forward
#'   selection with replacement), \code{"stack"} (non-negative elastic-net stacking of
#'   out-of-fold island predictions with an honest nested cross-validated performance
#'   estimate), or \code{"equal"} (uniform weighting across all islands; aliases \code{"uniform"},
#'   \code{"average"}). Default: \code{"caruana"}.
#' @param caruana_rounds Positive integer. Number of greedy selection rounds (default: 50).
#'   Only used when \code{method = "caruana"}.
#' @param bag_samples Logical. If \code{TRUE}, uses multi-bag bootstrap sampling of validation
#'   predictions during selection rounds to prevent validation set overfitting (default: \code{FALSE}).
#' @param sample_ratio Numeric between 0 and 1. Fraction of validation samples used when
#'   \code{bag_samples = TRUE} (default: 0.8). Only used when \code{method = "caruana"}.
#' @param stack_folds Positive integer. Number of folds for the honest nested evaluation
#'   of the stacked ensemble (default: \code{min(5, cv_folds)} in cv mode when fold
#'   assignments are available, otherwise 5). Only used when \code{method = "stack"}.
#' @param stack_alpha Numeric in \code{[0, 1]}. Elastic-net mixing parameter of the
#'   stacking meta-learner (\code{1} = lasso, \code{0} = ridge). Default: \code{0.5}.
#' @param rank_average Logical or NULL. If \code{TRUE}, ranks predictions to uniform
#'   quantiles in [0, 1] before computing ensemble weights and predictions.
#'   Recommended for ranking-based metrics like AUC. If \code{NULL} (default),
#'   automatically defaults to \code{TRUE} when \code{metric = "auc"}.
#' @param seed Optional integer seed for reproducible bagged sampling. Does not mutate
#'   the user's global RNG state.
#' @param threads Integer. Number of threads to use for model training.
#' @param verbose Logical. Whether to print progress messages.
#' @param ... Additional arguments passed to \code{train_model}.
#'
#' @return An \code{evo_ensemble} object containing:
#'   \item{active_recipes}{Named list of feature engineering recipes for surviving islands.}
#'   \item{active_models}{Named list of trained models for surviving islands.}
#'   \item{weights}{Named numeric vector of ensemble weights (summing to 1).}
#'   \item{caruana_history}{Data frame of validation loss trajectory across selection rounds
#'     (only for \code{method = "caruana"}).}
#'   \item{single_best_fitness}{Unpenalized validation fitness of the single best island model.}
#'   \item{ensemble_val_fitness}{Validation fitness achieved by the ensemble.}
#'   \item{method}{The ensembling method used.}
#'   \item{rank_average}{Logical indicating whether rank averaging was used.}
#'   \item{stack_cv_fitness}{Honest nested cross-validated fitness of the stacking procedure
#'     (only for \code{method = "stack"}).}
#'   \item{task}{The learning task ("classification", "regression", or "multiclass").}
#'   \item{evaluator}{The evaluator model engine used.}
#'   \item{classes}{Target class levels (for multiclass classification).}
#'   \item{metric}{Evaluation metric used.}
#'
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' # Evolve features across 3 islands
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   islands = 3,
#'   cv_folds = 2,
#'   verbose = FALSE
#' )
#'
#' # Build Caruana ensemble from island predictions
#' ens <- ensemble_islands(recipe, data = df, caruana_rounds = 20, verbose = FALSE)
#' print(ens)
#' }
#' @export
ensemble_islands <- function(recipe, data, target_col = NULL,
                             method = c("caruana", "stack", "equal"),
                             caruana_rounds = 50,
                             bag_samples = FALSE,
                             sample_ratio = 0.8,
                             stack_folds = NULL,
                             stack_alpha = 0.5,
                             rank_average = NULL,
                             seed = NULL,
                             threads = NULL,
                             verbose = TRUE, ...) {
  # Normalize recipe input: single evo_recipe or list of evo_recipe objects
  if (inherits(recipe, "evo_recipe")) {
    recipe_list <- list(recipe1 = recipe)
  } else if (is.list(recipe)) {
    recipe_list <- recipe
    for (idx in seq_along(recipe_list)) {
      if (!inherits(recipe_list[[idx]], "evo_recipe")) {
        stop(sprintf("Element %d in 'recipe' list is not an object of class 'evo_recipe'.", idx))
      }
    }
  } else {
    stop("Input 'recipe' must be an object of class 'evo_recipe' or a list of 'evo_recipe' objects.")
  }

  if (missing(data) || is.null(data)) {
    stop("Argument 'data' (full training dataset) is required for lazy final model fitting.")
  }

  first_recipe <- recipe_list[[1]]

  # Thread resolution:
  # 1. Thread alias passed via ... (e.g. nthreads, num_threads, threads)
  # 2. Explicitly supplied threads argument (if not NULL)
  # 3. Inherited threads from first_recipe$threads
  # 4. Fallback to physical core count
  user_args_top <- list(...)
  resolved_user_threads <- resolve_param_aliases(user_args_top)$threads
  if (!is.null(resolved_user_threads)) {
    threads <- resolved_user_threads
  } else if (is.null(threads)) {
    if (!is.null(first_recipe$threads)) {
      threads <- as.integer(first_recipe$threads)
    } else {
      threads <- default_threads()
    }
  } else {
    threads <- as.integer(threads)
  }

  # Handle positional method passed in 3rd argument (e.g. ensemble_islands(rec, data, "equal"))
  valid_methods <- c("caruana", "stack", "equal", "uniform", "average")
  if (is.character(target_col) && length(target_col) == 1 &&
      target_col %in% valid_methods && !target_col %in% names(data)) {
    method <- target_col
    target_col <- NULL
  }
  if (is.character(method) && length(method) > 1) {
    method <- method[1]
  }
  method <- match.arg(method, valid_methods)
  if (method %in% c("uniform", "average")) method <- "equal"
  old_threads <- getOption("evoFE.threads")
  on.exit(options(evoFE.threads = old_threads), add = TRUE)
  options(evoFE.threads = threads)

  if (!is.numeric(caruana_rounds) || caruana_rounds < 1) {
    stop("'caruana_rounds' must be a positive integer >= 1.")
  }
  caruana_rounds <- as.integer(caruana_rounds)

  # Infer target_col
  if (is.null(target_col)) {
    if (!is.null(first_recipe$target_col)) {
      target_col <- first_recipe$target_col
    } else {
      ind <- first_recipe$best_individual
      all_known <- unique(c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols))
      cand <- setdiff(names(data), all_known)
      if (length(cand) == 1) {
        target_col <- cand
      } else {
        stop("Could not automatically infer 'target_col'. Please specify target_col explicitly.")
      }
    }
  }

  if (!target_col %in% names(data)) {
    stop(sprintf("Target column '%s' not found in 'data'.", target_col))
  }

  task <- first_recipe$task
  metric <- first_recipe$metric
  classes <- first_recipe$classes
  num_class <- if (!is.null(classes)) length(classes) else NULL

  # Determine rank_average: default to TRUE if metric is 'auc', otherwise FALSE
  if (is.null(rank_average)) {
    rank_average <- !is.null(metric) && is.character(metric) && tolower(metric) == "auc"
  } else if (!is.logical(rank_average) || length(rank_average) != 1L) {
    stop("'rank_average' must be a logical scalar (TRUE or FALSE) or NULL.")
  }

  # Collect validation prediction vectors, targets, and evaluators across all recipes
  val_preds_list <- list()
  cand_metadata <- list()
  y_val <- NULL

  for (r_idx in seq_along(recipe_list)) {
    rec <- recipe_list[[r_idx]]
    rec_prefix <- if (!is.null(names(recipe_list)) && names(recipe_list)[r_idx] != "") {
      names(recipe_list)[r_idx]
    } else if (length(recipe_list) > 1) {
      paste0("recipe_", r_idx)
    } else {
      ""
    }

    if (is.null(rec$island_bests) || length(rec$island_bests) == 0) {
      next
    }

    for (i in seq_along(rec$island_bests)) {
      ind_i <- rec$island_bests[[i]]
      if (!is.null(ind_i) && !is.null(ind_i$val_preds)) {
        cand_name <- if (nchar(rec_prefix) > 0) paste0(rec_prefix, "_island_", i) else paste0("island_", i)
        cand_eval <- if (!is.null(ind_i$evaluator)) ind_i$evaluator else rec$evaluator

        val_preds_list[[cand_name]] <- ind_i$val_preds
        cand_metadata[[cand_name]] <- list(
          recipe = rec,
          ind = ind_i,
          evaluator = cand_eval
        )
      }
    }
  }

  if (length(val_preds_list) == 0) {
    stop("No valid validation prediction vectors found across the provided recipe(s).")
  }

  if (length(val_preds_list) < 2) {
    warning("Only 1 candidate island prediction vector available. Ensembling requires >= 2 candidate islands.")
  }

  # Validate and align candidate validation predictions across recipes
  cand_row_counts <- vapply(val_preds_list, function(p) {
    if (is.matrix(p)) nrow(p) else length(p)
  }, integer(1))

  cand_has_na <- any(vapply(val_preds_list, anyNA, logical(1)))

  # Check if row counts are inhomogeneous, contain NAs, or don't cover the full dataset when mixing recipes or in metacv
  is_metacv_equal <- (method == "equal" &&
                      identical(first_recipe$evaluation_strategy, "metacv") &&
                      !is.null(first_recipe$metacv_island_oof_preds) &&
                      length(recipe_list) == 1L)

  needs_harmonization <- !is_metacv_equal && (
    cand_has_na ||
    length(unique(cand_row_counts)) > 1L ||
    (length(recipe_list) > 1L && any(cand_row_counts != nrow(data))) ||
    (identical(first_recipe$evaluation_strategy, "metacv") && any(cand_row_counts != nrow(data)))
  )

  stored_folds <- first_recipe$fold_ids

  # Persistent in-place alignment caching
  alignment_cache <- first_recipe$alignment_cache
  if (is.null(alignment_cache) || !is.environment(alignment_cache)) {
    alignment_cache <- new.env(hash = TRUE, parent = emptyenv())
    first_recipe$alignment_cache <- alignment_cache
  }

  cache_key <- digest::digest(list(dim(data), target_col, names(val_preds_list)), algo = "xxhash64")

  if (needs_harmonization && exists(cache_key, envir = alignment_cache, inherits = FALSE)) {
    if (verbose) {
      message("  [Cache Hit] Using cached aligned out-of-fold validation predictions (0.000 s)...")
    }
    cached <- get(cache_key, envir = alignment_cache, inherits = FALSE)
    val_preds_list <- cached$val_preds_list
    cand_metadata <- cached$cand_metadata
    stored_folds <- cached$stored_folds
    y_val <- cached$y_val
    needs_harmonization <- FALSE
  }

  if (needs_harmonization) {
    if (verbose) {
      message("  Aligning out-of-fold validation predictions across candidate models on the training dataset...")
    }
    if (!is.null(stored_folds) && length(stored_folds) == nrow(data) && !any(is.na(stored_folds))) {
      common_fold_ids <- stored_folds
      common_folds <- length(unique(stored_folds))
    } else {
      common_folds <- min(5L, max(2L, nrow(data) %/% 4L))
      common_fold_ids <- if (task %in% c("classification", "multiclass")) {
        fid <- integer(nrow(data))
        y_fac <- as.factor(data[[target_col]])
        for (lv in levels(y_fac)) {
          ids <- which(as.character(y_fac) == lv)
          if (length(ids) > 0) {
            fid[ids] <- sample(rep(seq_len(min(common_folds, length(ids))), length.out = length(ids)))
          }
        }
        un <- which(fid == 0)
        if (length(un) > 0) fid[un] <- sample(rep(seq_len(common_folds), length.out = length(un)))
        fid
      } else {
        sample(rep(seq_len(common_folds), length.out = nrow(data)))
      }
    }

    for (nm in names(val_preds_list)) {
      ind_re <- cand_metadata[[nm]]$ind
      rec_re <- cand_metadata[[nm]]$recipe
      ind_re$fitness <- NA_real_
      cand_eval <- cand_metadata[[nm]]$evaluator

      cand_extra <- if (!is.null(ind_re$extra_args) && length(ind_re$extra_args) > 0) {
        ind_re$extra_args
      } else if (!is.null(rec_re$extra_args) && length(rec_re$extra_args) > 0) {
        rec_re$extra_args
      } else if (!is.null(first_recipe$extra_args) && length(first_recipe$extra_args) > 0) {
        first_recipe$extra_args
      } else {
        list()
      }
      eval_extra <- utils::modifyList(cand_extra, list(...))

      cand_threads_eval <- if (!is.null(threads)) {
        threads
      } else if (!is.null(ind_re$threads)) {
        ind_re$threads
      } else if (!is.null(rec_re$threads)) {
        rec_re$threads
      } else {
        first_recipe$threads
      }
      if (is.null(cand_threads_eval)) cand_threads_eval <- default_threads()

      ind_re <- do.call(evaluate_fitness, c(
        list(
          ind_re, data = data, target_col = target_col,
          task = task, cv_folds = common_folds,
          evaluation_strategy = "cv", fold_ids = common_fold_ids,
          evaluator = cand_eval, threads = cand_threads_eval,
          metric = metric, verbose = FALSE, allow_prune = TRUE
        ),
        eval_extra
      ))
      val_preds_list[[nm]] <- ind_re$val_preds
      cand_metadata[[nm]]$ind <- ind_re
    }
    stored_folds <- common_fold_ids
    y_val <- cand_metadata[[1]]$ind$y_val
    if (is.null(y_val) || any(is.na(y_val))) {
      y_val <- if (task == "multiclass") {
        as.integer(factor(data[[target_col]], levels = classes)) - 1
      } else {
        data[[target_col]]
      }
    }

    # Store aligned results in alignment_cache by reference
    assign(cache_key, list(
      val_preds_list = val_preds_list,
      cand_metadata = cand_metadata,
      stored_folds = stored_folds,
      y_val = y_val
    ), envir = alignment_cache)
  } else if (is.null(y_val)) {
    y_val <- if (is_metacv_equal) {
      if (task == "multiclass") {
        as.integer(factor(data[[target_col]], levels = classes)) - 1
      } else {
        data[[target_col]]
      }
    } else {
      y_cand <- cand_metadata[[1]]$ind$y_val
      if (is.null(y_cand) || any(is.na(y_cand))) {
        if (task == "multiclass") {
          as.integer(factor(data[[target_col]], levels = classes)) - 1
        } else {
          data[[target_col]]
        }
      } else {
        y_cand
      }
    }
  }

  # Ensure targets and predictions are normalized
  if (task == "classification") {
    if (is.factor(y_val)) {
      y_val <- as.integer(y_val) - 1L
    } else if (is.character(y_val)) {
      y_val <- as.integer(as.factor(y_val)) - 1L
    } else if (is.logical(y_val)) {
      y_val <- as.integer(y_val)
    } else if (is.numeric(y_val) && !all(stats::na.omit(y_val) %in% c(0, 1))) {
      y_val <- as.integer(as.factor(y_val)) - 1L
    }
  } else if (task == "multiclass") {
    for (nm in names(val_preds_list)) {
      p <- val_preds_list[[nm]]
      if (!is.matrix(p)) {
        val_preds_list[[nm]] <- matrix(p, ncol = num_class, byrow = FALSE)
      }
    }
    if (!is.numeric(y_val) || any(y_val >= num_class) || any(y_val < 0)) {
      y_val <- as.integer(factor(y_val, levels = classes)) - 1
    }
  }

  if (verbose) {
    method_title <- if (method == "caruana") "Caruana" else if (method == "stack") "stacked" else "equal-weight"
    message(sprintf("\nStarting %s ensemble selection across %d candidate island models...",
                    method_title, length(val_preds_list)))
  }

  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])

  selection_res <- NULL
  if (method == "caruana") {
    selection_res <- caruana_select(
      y_true = y_val,
      val_preds_list = val_preds_list,
      task = task,
      metric = metric,
      rounds = caruana_rounds,
      bag_samples = bag_samples,
      sample_ratio = sample_ratio,
      seed = seed,
      num_class = num_class,
      rank_average = rank_average,
      verbose = verbose
    )
  } else if (method == "stack") {
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' is required for method = \"stack\". ",
           "Install it or use method = \"caruana\".")
    }
    fold_partition <- NULL
    stack_k <- stack_folds
    if (!is.null(stored_folds) && is.atomic(stored_folds) &&
        length(stored_folds) == n_obs && !all(is.na(stored_folds))) {
      fold_partition <- as.integer(stored_folds)
      k_avail <- length(unique(fold_partition))
      if (is.null(stack_k)) stack_k <- min(5L, k_avail)
    } else {
      if (verbose) {
        message("  No stored CV fold assignments matching the out-of-fold predictions; using internal folds for the nested estimate.")
      }
      if (is.null(stack_k)) stack_k <- 5L
    }
    stack_k <- max(2L, min(as.integer(stack_k), length(y_val) - 1L))

    selection_res <- .stack_select(
      y_true = y_val,
      val_preds_list = val_preds_list,
      task = task,
      metric = metric,
      num_class = num_class,
      classes = classes,
      stack_folds = stack_k,
      fold_partition = fold_partition,
      alpha = stack_alpha,
      seed = seed,
      rank_average = rank_average,
      verbose = verbose
    )
  } else if (method == "equal") {
    weights <- rep(1 / length(val_preds_list), length(val_preds_list))
    names(weights) <- names(val_preds_list)
    preds_for_blend <- if (isTRUE(rank_average)) {
      lapply(val_preds_list, rank_transform_predictions)
    } else {
      val_preds_list
    }
    ens_preds <- if (is_metacv_equal && !isTRUE(rank_average)) {
      first_recipe$metacv_island_oof_preds
    } else {
      if (task == "multiclass") {
        res_mat <- matrix(0, nrow = nrow(preds_for_blend[[1]]), ncol = ncol(preds_for_blend[[1]]))
        for (nm in names(weights)) {
          res_mat <- res_mat + weights[[nm]] * preds_for_blend[[nm]]
        }
        res_mat
      } else {
        res_vec <- numeric(length(preds_for_blend[[1]]))
        for (nm in names(weights)) {
          res_vec <- res_vec + weights[[nm]] * preds_for_blend[[nm]]
        }
        res_vec
      }
    }
    final_fitness <- if (task == "multiclass") {
      compute_metric(y_val, ens_preds, task, metric, num_class)
    } else {
      compute_metric(y_val, ens_preds, task, metric)
    }
    selection_res <- list(
      weights = weights,
      final_fitness = final_fitness
    )
  }

  weights <- selection_res$weights
  active_names <- names(weights[weights > 0])

  # Apples-to-apples comparison: ensemble fitness is an unpenalized validation
  # metric, so compare against the island's raw (unpenalized) validation score
  # rather than the complexity-penalized selection fitness.
  single_best_fitness <- max(vapply(cand_metadata, function(m) {
    ind_i <- m$ind
    if (is.null(ind_i)) return(-Inf)
    if (!is.null(ind_i$raw_fitness) && is.finite(ind_i$raw_fitness)) {
      ind_i$raw_fitness
    } else if (!is.null(ind_i$fitness) && !is.na(ind_i$fitness)) {
      ind_i$fitness
    } else {
      -Inf
    }
  }, double(1)))

  if (verbose) {
    message(sprintf("\nEnsemble Selection Complete: %d / %d island models active.", length(active_names), length(val_preds_list)))
    if (method == "stack" && !is.null(selection_res$stack_cv_fitness)) {
      message(sprintf("  Single Best Fitness: %.4f  |  Ensemble Fitness: %.4f  |  Honest CV Fitness: %.4f",
                      single_best_fitness, selection_res$final_fitness, selection_res$stack_cv_fitness))
    } else {
      message(sprintf("  Single Best Fitness: %.4f  |  Ensemble Fitness: %.4f", single_best_fitness, selection_res$final_fitness))
    }
  }

  active_recipes <- list()
  active_models <- list()
  active_evaluators <- list()

  # Shared state cache for dataset transformation
  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  dt_full <- data.table::as.data.table(data)

  # Lazy training: Reuse best_model for matching global best island, fit remaining active islands
  for (name in active_names) {
    meta <- cand_metadata[[name]]
    ind_i <- meta$ind
    rec_i <- meta$recipe
    eval_i <- meta$evaluator
    ind_str <- individual_to_recipe_string(ind_i)
    best_ind_str <- individual_to_recipe_string(rec_i$best_individual)

    active_evaluators[[name]] <- eval_i

    user_args_passed <- list(...)
    has_user_override <- length(user_args_passed) > 0 ||
      (!is.null(threads) && !is.null(rec_i$threads) && as.integer(threads) != as.integer(rec_i$threads))

    # Check if this candidate matches rec_i$best_individual & rec_i$best_model is available (and no parameter overrides requested)
    if (!has_user_override && ind_str == best_ind_str && !is.null(rec_i$best_model)) {
      if (verbose) {
        message(sprintf("  [%s] Evaluator: %s | Weight: %5.1f%% | Reusing existing global best model (zero retraining).", name, eval_i, weights[[name]] * 100))
      }
      active_recipes[[name]] <- rec_i$best_individual
      active_models[[name]] <- rec_i$best_model
    } else {
      if (verbose) {
        message(sprintf("  [%s] Evaluator: %s | Weight: %5.1f%% | Lazily training final model on full dataset...", name, eval_i, weights[[name]] * 100))
      }

      dt_i <- data.table::copy(dt_full)
      res_full <- apply_individual(ind_i, dt_i, NULL, target_col, state_cache = state_cache, allow_prune = TRUE)
      applied_ind <- res_full$ind
      active_recipes[[name]] <- applied_ind

      gene_cols <- if (length(applied_ind$genes) > 0) vapply(applied_ind$genes, function(g) g$output_col, character(1)) else character(0)
      features <- c(applied_ind$numeric_cols, applied_ind$categorical_cols, applied_ind$datetime_cols, gene_cols)
      features <- setdiff(features, target_col)

      x_full <- .sanitize_feature_matrix(res_full$train[, features, with = FALSE])
      y_full <- res_full$train[[target_col]]
      if (task == "multiclass") {
        y_full <- as.integer(factor(y_full, levels = classes)) - 1
      }

      # Train model using candidate's specific evaluator and best params
      cand_info <- cand_metadata[[name]]
      rec_i <- cand_info$recipe

      base_extra <- if (!is.null(ind_i$extra_args) && length(ind_i$extra_args) > 0) {
        ind_i$extra_args
      } else if (!is.null(rec_i$extra_args) && length(rec_i$extra_args) > 0) {
        rec_i$extra_args
      } else if (!is.null(first_recipe$extra_args) && length(first_recipe$extra_args) > 0) {
        first_recipe$extra_args
      } else {
        list()
      }
      final_args_i <- utils::modifyList(base_extra, list(...))
      if (!is.null(ind_i$best_iteration) && is.numeric(ind_i$best_iteration) &&
          is.finite(ind_i$best_iteration) && ind_i$best_iteration > 0) {
        total_data_size <- nrow(x_full)
        training_size <- if (!is.null(ind_i$train_size) && is.numeric(ind_i$train_size) && ind_i$train_size > 0) {
          as.numeric(ind_i$train_size)
        } else if (!is.null(recipe$best_individual$train_size) && is.numeric(recipe$best_individual$train_size) && recipe$best_individual$train_size > 0) {
          as.numeric(recipe$best_individual$train_size)
        } else {
          total_data_size
        }
        target_iters <- if (is_tree_evaluator(eval_i)) {
          scale_evaluator_iterations(eval_i, ind_i$best_iteration, training_size, total_data_size)
        } else {
          as.integer(ind_i$best_iteration)
        }
        final_args_i <- apply_iteration_target(final_args_i, target_iters, eval_i)
      }

      final_eval_i <- unwrap_evaluator(eval_i)
      if (!is.null(ind_i$best_params) && length(ind_i$best_params) > 0) {
        final_args_i <- utils::modifyList(final_args_i, as.list(ind_i$best_params))
      }

      cand_threads <- if (!is.null(threads)) {
        threads
      } else if (!is.null(ind_i$threads)) {
        ind_i$threads
      } else if (!is.null(rec_i$threads)) {
        rec_i$threads
      } else {
        first_recipe$threads
      }
      if (is.null(cand_threads)) cand_threads <- default_threads()

      res_m <- do.call(train_model, c(
        list(
          x_train = x_full, y_train = y_full,
          task = task, evaluator = final_eval_i,
          threads = cand_threads, num_class = num_class, metric = metric,
          verbose = verbose, best_params = ind_i$best_params
        ),
        final_args_i
      ))
      active_models[[name]] <- res_m$model
    }
  }

  # Pre-compute presentation metrics for evo_ensemble using calculate_headroom()
  base_fit <- if (!is.null(first_recipe$baseline_fitness) && is.finite(first_recipe$baseline_fitness)) {
    first_recipe$baseline_fitness
  } else {
    NULL
  }

  ens_hr <- if (!is.null(base_fit)) {
    calculate_headroom(selection_res$final_fitness, base_fit, task)
  } else {
    NULL
  }

  # Propagate or pre-compute island breakdown metrics
  isl_baselines <- first_recipe$island_baselines
  isl_improvements <- first_recipe$island_improvements
  isl_headroom_closed <- first_recipe$island_headroom_closed

  if (!is.null(isl_baselines) && (is.null(isl_improvements) || is.null(isl_headroom_closed))) {
    if (!is.null(first_recipe$island_bests) && length(first_recipe$island_bests) == length(isl_baselines)) {
      isl_fits <- vapply(first_recipe$island_bests, function(ind) ind$fitness, numeric(1))
      isl_hr <- calculate_headroom(isl_fits, isl_baselines, task)
      if (is.null(isl_improvements)) isl_improvements <- isl_hr$improvement
      if (is.null(isl_headroom_closed)) isl_headroom_closed <- isl_hr$headroom_closed
    }
  }

  structure(
    list(
      active_recipes = active_recipes,
      active_models = active_models,
      active_evaluators = active_evaluators,
      weights = weights,
      caruana_history = selection_res$history,
      single_best_fitness = single_best_fitness,
      ensemble_val_fitness = selection_res$final_fitness,
      baseline_fitness = base_fit,
      improvement = if (!is.null(ens_hr)) ens_hr$improvement else NULL,
      headroom_closed = if (!is.null(ens_hr)) ens_hr$headroom_closed else NULL,
      ensemble_headroom_closed = if (!is.null(ens_hr)) ens_hr$headroom_closed else NULL,
      island_baselines = isl_baselines,
      island_improvements = isl_improvements,
      island_headroom_closed = isl_headroom_closed,
      method = method,
      rank_average = isTRUE(rank_average),
      stack_cv_fitness = if (!is.null(selection_res$stack_cv_fitness)) selection_res$stack_cv_fitness else NULL,
      task = task,
      evaluator = first_recipe$evaluator,
      target_col = target_col,
      classes = classes,
      metric = metric,
      threads = threads,
      extra_args = list(...)
    ),
    class = "evo_ensemble"
  )
}
# caruana_select moved to R/ensemble_caruana.R
# .stack_select moved to R/ensemble_stack.R

