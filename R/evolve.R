truncate_cols <- function(cols, max_show = 10) {
  if (length(cols) <= max_show) {
    return(paste(cols, collapse = ", "))
  }
  paste0(paste(cols[1:max_show], collapse = ", "), ", ... (+ ", length(cols) - max_show, " more)")
}

# --- Internal helpers ---

#' Check if terminal supports ANSI colors
#' @keywords internal
supports_color <- function() {
  term <- Sys.getenv("TERM")
  if (term %in% c("dumb", "")) {
    return(FALSE)
  }
  if (.Platform$OS.type == "windows") {
    return(interactive() || !is.na(Sys.getenv("RSTUDIO", unset = NA)))
  }
  isatty(stdout()) || !is.na(Sys.getenv("RSTUDIO", unset = NA))
}

#' Stratified or random splitting helper
#' @keywords internal
stratified_split <- function(y, ratio) {
  n <- length(y)
  ratios <- ratio / sum(ratio)

  # For regression or if y has only 1 level, do standard random split
  if ((is.numeric(y) && length(unique(y)) > 10) || length(unique(y)) <= 1) {
    shuffled_idx <- sample(seq_len(n))
    if (length(ratios) == 2) {
      n_train <- round(n * ratios[1])
      if (n_train < 1 && n >= 1) n_train <- 1
      train_idx <- shuffled_idx[seq_len(n_train)]
      res <- rep("val", n)
      res[train_idx] <- "train"
    } else {
      n_train <- round(n * ratios[1])
      if (n_train < 1 && n >= 1) n_train <- 1
      n_val <- round(n * ratios[2])
      if (n_val < 1 && (n - n_train) >= 1) n_val <- 1
      if (n_train + n_val > n) {
        n_val <- max(0, n - n_train)
      }
      train_idx <- shuffled_idx[seq_len(n_train)]
      if (n_val > 0) {
        val_idx <- shuffled_idx[(n_train + 1):(n_train + n_val)]
      } else {
        val_idx <- integer(0)
      }
      res <- rep("holdout", n)
      res[train_idx] <- "train"
      res[val_idx] <- "val"
    }
    return(res)
  }

  # For classification/multiclass, split class-by-class
  y_factor <- as.factor(y)
  levels_y <- levels(y_factor)

  res <- character(n)

  for (lvl in levels_y) {
    lvl_idx <- which(y_factor == lvl)
    n_lvl <- length(lvl_idx)
    shuffled_lvl_idx <- sample(lvl_idx)

    if (length(ratios) == 2) {
      n_train <- round(n_lvl * ratios[1])
      if (n_train < 1 && n_lvl >= 1) n_train <- 1
      if (n_train > n_lvl) n_train <- n_lvl

      train_idx <- shuffled_lvl_idx[seq_len(n_train)]
      val_idx <- setdiff(shuffled_lvl_idx, train_idx)

      res[train_idx] <- "train"
      res[val_idx] <- "val"
    } else {
      n_train <- round(n_lvl * ratios[1])
      if (n_train < 1 && n_lvl >= 1) n_train <- 1

      n_val <- round(n_lvl * ratios[2])
      if (n_val < 1 && (n_lvl - n_train) >= 1) n_val <- 1

      if (n_train + n_val > n_lvl) {
        n_val <- max(0, n_lvl - n_train)
      }

      train_idx <- shuffled_lvl_idx[seq_len(n_train)]
      if (n_val > 0) {
        val_idx <- shuffled_lvl_idx[(n_train + 1):(n_train + n_val)]
      } else {
        val_idx <- integer(0)
      }
      holdout_idx <- setdiff(shuffled_lvl_idx, c(train_idx, val_idx))

      res[train_idx] <- "train"
      res[val_idx] <- "val"
      res[holdout_idx] <- "holdout"
    }
  }

  # Fill any remaining unassigned elements (due to rounding) with "train"
  res[res == ""] <- "train"
  res
}

#' Evaluate all unevaluated individuals in a population
#' @keywords internal
evaluate_pop <- function(pop, data, target_col, task, cv_folds, evaluation_strategy,
                         split_ids, shared_splits, evaluator,
                         fold_ids, shared_folds, shared_full, state_cache,
                         fitness_cache, threads, verbose, running_best_fitness,
                         metric = "default", allow_prune = TRUE,
                         complexity_penalty = 0, island = NULL, ...) {
  for (i in seq_along(pop)) {
    if (!is.na(pop[[i]]$fitness)) next

    recipe_str <- individual_to_recipe_string(pop[[i]])
    cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
    cached <- exists(cache_key, envir = fitness_cache, inherits = FALSE)

    if (cached) {
      pop[[i]] <- get(cache_key, envir = fitness_cache)
    } else {
      pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids, shared_splits = shared_splits,
        evaluator = evaluator, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose,
        allow_prune = allow_prune, complexity_penalty = complexity_penalty, ...
      )
      assign(cache_key, pop[[i]], envir = fitness_cache)
    }

    if (verbose) {
      improved <- !is.na(pop[[i]]$fitness) && (is.na(running_best_fitness) || pop[[i]]$fitness > running_best_fitness)
      green_start <- if (supports_color()) "\033[32m" else ""
      red_start <- if (supports_color()) "\033[31m" else ""
      color_reset <- if (supports_color()) "\033[0m" else ""

      new_best_str <- if (improved) " (New Best!)" else ""
      cache_str <- if (cached) " (cached)" else ""
      msg_color <- if (improved) green_start else red_start

      if (!is.null(island)) {
        msg <- sprintf(
          "  [Island %d] Tested Individual %d%s -> Fitness: %.4f%s",
          island, i, new_best_str, pop[[i]]$fitness, cache_str
        )
      } else {
        msg <- sprintf(
          "  Tested Individual %d%s -> Fitness: %.4f%s",
          i, new_best_str, pop[[i]]$fitness, cache_str
        )
      }
      message(paste0(msg_color, msg, color_reset))

      if (improved) {
        running_best_fitness <- pop[[i]]$fitness
      }
    }
  }
  list(pop = pop, running_best_fitness = running_best_fitness)
}

#' Check whether a candidate individual is a duplicate or known-inferior
#' @keywords internal
is_invalid_individual <- function(c_ind, pop_list, cache, best_fit) {
  # Check 1: Duplicate in current generation
  get_out <- function(ind) {
    if (length(ind$genes) == 0) {
      return(character(0))
    }
    sort(vapply(ind$genes, function(g) g$output_col, character(1)))
  }
  c_out <- get_out(c_ind)
  for (existing in pop_list) {
    e_out <- get_out(existing)
    if (length(c_out) == length(e_out) && all(c_out == e_out)) {
      return(TRUE)
    }
  }

  # Check 2: Taboo search — reject recipes that are clearly inferior to the best.
  # Use a meaningful epsilon so borderline recipes aren't permanently banned.
  recipe_str <- individual_to_recipe_string(c_ind)
  cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
  if (exists(cache_key, envir = cache, inherits = FALSE)) {
    cached_ind <- get(cache_key, envir = cache)
    known_fit <- cached_ind$fitness
    if (abs(best_fit) <= 1) {
      taboo_threshold <- max(0.00002, 0.1 * (1 - abs(best_fit)))
    } else {
      taboo_threshold <- max(0.00002, 0.01 * abs(best_fit))
    }
    if (!is.infinite(best_fit) && isTRUE(known_fit < best_fit - taboo_threshold)) {
      return(TRUE)
    }
  }

  return(FALSE)
}

#' Tournament selection
#' @keywords internal
#' @examples
#' \donttest{
#' pop <- list(
#'   list(fitness = 0.5),
#'   list(fitness = 0.8),
#'   list(fitness = 0.2)
#' )
#' best <- tournament_select(pop, k = 2)
#' }
#' @export
tournament_select <- function(pop, k = 3) {
  k <- min(k, length(pop))
  candidates <- sample(seq_along(pop), k)
  fitnesses <- sapply(candidates, function(i) {
    f <- pop[[i]]$fitness
    if (is.na(f)) -Inf else f
  })
  pop[[candidates[which.max(fitnesses)]]]
}

#' Run evolutionary feature engineering
#'
#' @param data A data.frame or data.table
#' @param target_col Name of the target column
#' @param task "classification" or "regression"
#' @param generations Number of generations (max iterations)
#' @param pop_size Population size
#' @param cv_folds Number of cross-validation folds
#' @param evaluation_strategy "cv" or "split". Strategy to evaluate candidate recipes.
#' @param split_ratio A numeric vector of length 2 or 3 defining
#'   train/validation/holdout proportions (e.g. c(0.6, 0.2, 0.2)).
#' @param split_ids An optional character vector of split assignments (e.g.
#'   \code{c("train", "train", "val", "holdout", "train")}). Must have the same
#'   length as the number of rows in \code{data} and contain only "train", "val",
#'   or "holdout" labels (with at least "train" and "val" present). When
#'   provided, \code{evaluation_strategy} is automatically set to "split" and the
#'   actual split proportions are computed from the vector.
#' @param early_stopping_generations Stop if fitness doesn't improve for this
#'   many generations
#' @param evaluator The ML model to use ("lightgbm", "xgboost", "catboost", or a
#'   custom registered evaluator name).
#' @param dynamic_population Logical. If TRUE, population expands dynamically
#'   during stagnation.
#' @param dynamic_population_growth_rate Growth rate multiplier for population
#'   expansion during stagnation (default 1.5).
#' @param dynamic_population_decay_rate Decay rate multiplier for population
#'   contraction back to baseline (default 0.7).
#' @param crossover_type Crossover type: "both" (default, 50\% random / 50\%
#'   union), "random", or "union"
#' @param threads Number of threads to use for parallel execution (default 2)
#' @param max_clustering_size Maximum unique training rows to cluster (default
#'   5000, 0/NULL for unlimited)
#' @param verbose Logical. If TRUE, prints progress.
#' @param metric The metric to optimize ("default", "auc", "f1", "mae", or a
#'   custom function).
#' @param model_all_final_genes Logical. If TRUE, the final model is trained using
#'   the union of all unique genes evolved in the final population, rather than
#'   only the best individual's genes.
#' @param model_all_historical_genes Logical. If TRUE, the final model is trained
#'   using the union of all unique genes evolved across all generations, rather
#'   than only the best individual's genes.
#' @param allowed_transformers Character vector of allowed transformer names,
#'   or \code{"all"} / \code{"basic"} / \code{"robust"} / \code{"clustering"}.
#' @param complexity_penalty Non-negative numeric penalty subtracted from each
#'   individual's raw fitness as \code{complexity_penalty * n_genes}.  A small
#'   value (e.g. \code{0.001}) encourages parsimonious recipes and reduces
#'   overfitting on small datasets.  Default \code{0} disables the penalty.
#' @param islands Integer. Number of independent subpopulations (islands) to evolve (default 1).
#' @param migration_interval Integer. Number of generations between migrations (default 5).
#' @param migration_rate Integer. Number of top individuals to migrate from each island to its neighbor (default 1).
#' @param gene_migration_prob Numeric. Probability of injecting a migrated gene during mutation (default 0.2).
#' @param ... Additional arguments passed to the underlying evaluator training
#'   functions.
#' @importFrom utils tail
#' @return An \code{evo_recipe} S3 object:
#'   a list with elements
#'   \code{best_individual} (the top-scoring \code{evo_individual}),
#'   \code{history} (list of all evaluated individuals across generations),
#'   \code{task}, \code{best_model} (the trained model object),
#'   \code{evaluator}, and \code{classes} (class levels for multiclass tasks,
#'   otherwise \code{NULL}).
#' @examples
#' \donttest{
#' # Quick classification example using mtcars
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' set.seed(42)
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   verbose = FALSE
#' )
#' print(recipe)
#' }
#' @export
evolve_features <- function(data, target_col, task = "classification",
                            generations = 10, pop_size = 10, cv_folds = 3,
                            evaluation_strategy = "cv", split_ratio = c(0.6, 0.2, 0.2),
                            split_ids = NULL,
                            early_stopping_generations = 3, evaluator = "lightgbm",
                            dynamic_population = TRUE,
                            dynamic_population_growth_rate = 1.5,
                            dynamic_population_decay_rate = 0.7,
                            crossover_type = "both",
                            threads = 2, max_clustering_size = 5000,
                            verbose = TRUE, metric = "default",
                            model_all_final_genes = FALSE,
                            model_all_historical_genes = FALSE,
                            allowed_transformers = "all",
                            complexity_penalty = 0,
                            islands = 1,
                            migration_interval = 5,
                            migration_rate = 1,
                            gene_migration_prob = 0.2, ...) {



  # Parse allowed_transformers
  all_trans <- names(evo_transformers)
  if (is.null(allowed_transformers)) allowed_transformers <- "all"
  if (length(allowed_transformers) == 1) {
    if (allowed_transformers == "all") {
      allowed_transformers <- all_trans
    } else if (allowed_transformers == "basic") {
      allowed_transformers <- intersect(all_trans, c(
        "add", "subtract", "multiply", "divide",
        "log", "sqrt", "reciprocal", "power", "displaced_log",
        "normalized_difference", "frequency_encode",
        "one_hot_encode", "target_encode", "pooled_target_encode", "target_encode_multiclass",
        "rank_transform", "groupby_mean", "groupby_min", "groupby_max", "concat"
      ))
    } else if (allowed_transformers == "clustering") {
      allowed_transformers <- intersect(all_trans, c("genie", "genie_centroid_dist", "lumbermark", "lumbermark_centroid_dist", "mst_score", "deadwood", "umap", "random_projection", "truncated_svd", "pca"))
    } else if (allowed_transformers == "robust") {
      allowed_transformers <- intersect(all_trans, c(
        "log", "sqrt", "reciprocal", "power", "displaced_log", "rank_transform",
        "add", "subtract", "multiply", "divide",
        "normalized_difference", "log_ratio",
        "target_encode", "pooled_target_encode", "woe_encode", "frequency_encode",
        "groupby_mean", "groupby_median", "groupby_sd",
        "groupby_zscore", "groupby_ratio", "groupby_quantile",
        "groupby_min", "groupby_max",
        "quantile_binning", "pca", "concat"
      ))
    }
  }
  allowed_transformers <- intersect(allowed_transformers, all_trans)
  if (length(allowed_transformers) == 0) {
    warning("No valid transformers found in 'allowed_transformers'. Falling back to 'all'.")
    allowed_transformers <- all_trans
  }

  # Temporarily configure max clustering size and threads options
  old_max_size <- getOption("evoFE.max_clustering_size")
  old_threads <- getOption("evoFE.threads")
  options(evoFE.max_clustering_size = max_clustering_size, evoFE.threads = threads)
  on.exit(
    {
      options(evoFE.max_clustering_size = old_max_size)
      options(evoFE.threads = old_threads)
    },
    add = TRUE
  )

  # Prevent macOS OpenMP thread collisions between data.table, lightgbm, and other libraries
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    old_omp <- RhpcBLASctl::omp_get_max_threads()
    old_blas <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::omp_set_num_threads(threads)
    RhpcBLASctl::blas_set_num_threads(threads)
    on.exit(
      {
        RhpcBLASctl::omp_set_num_threads(old_omp)
        RhpcBLASctl::blas_set_num_threads(old_blas)
      },
      add = TRUE
    )
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    old_dt <- data.table::getDTthreads()
    data.table::setDTthreads(threads)
    on.exit(
      {
        data.table::setDTthreads(old_dt)
      },
      add = TRUE
    )
  }
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    tryCatch(
      {
        old_qf <- quitefastmst::omp_get_max_threads()
        quitefastmst::omp_set_num_threads(threads)
        on.exit(
          {
            tryCatch(
              {
                quitefastmst::omp_set_num_threads(old_qf)
              },
              error = function(e) NULL
            )
          },
          add = TRUE
        )
      },
      error = function(e) NULL
    )
  }

  if (!task %in% c("classification", "multiclass", "regression")) {
    stop("task must be one of: 'classification', 'multiclass', 'regression'.")
  }

  if (!is.function(metric)) {
    metric_lower <- tolower(metric)
    valid_metrics <- list(
      classification = c("default", "auc", "f1", "eval-ts-refinement", "ts-refinement", "ts_refinement"),
      multiclass = c("default", "auc", "eval-ts-refinement", "ts-refinement", "ts_refinement"),
      regression = c("default", "mae")
    )
    if (!metric_lower %in% valid_metrics[[task]]) {
      stop(sprintf(
        "Metric '%s' is not supported for task '%s'. Supported metrics are: %s",
        metric, task, paste(valid_metrics[[task]], collapse = ", ")
      ))
    }
  }

  if (!target_col %in% names(data)) {
    stop(sprintf("Target column '%s' not found in the dataset.", target_col))
  }

  if (!is.null(split_ids)) {
    if (length(split_ids) != nrow(data)) {
      stop(sprintf("split_ids must have the same length as the number of rows in data (expected %d, got %d).", nrow(data), length(split_ids)))
    }
    invalid_ids <- setdiff(unique(split_ids), c("train", "val", "holdout"))
    if (length(invalid_ids) > 0) {
      stop(sprintf("split_ids must only contain 'train', 'val', or 'holdout' labels. Found invalid labels: %s", paste(invalid_ids, collapse = ", ")))
    }
    if (!all(c("train", "val") %in% split_ids)) {
      stop("split_ids must contain at least 'train' and 'val' labels.")
    }
    if (evaluation_strategy == "cv") {
      warning("split_ids was provided but evaluation_strategy is 'cv'. Setting evaluation_strategy to 'split'.")
      evaluation_strategy <- "split"
    }
  }

  # Validate island parameters
  if (!is.numeric(islands) || islands < 1) {
    stop("islands must be a positive integer >= 1.")
  }
  islands <- as.integer(islands)

  if (islands > 1) {
    if (!is.numeric(migration_interval) || migration_interval < 1) {
      stop("migration_interval must be a positive integer >= 1.")
    }
    migration_interval <- as.integer(migration_interval)

    if (!is.numeric(migration_rate) || migration_rate < 1) {
      stop("migration_rate must be a positive integer >= 1.")
    }
    migration_rate <- as.integer(migration_rate)

    if (migration_rate >= pop_size) {
      stop("migration_rate must be less than pop_size.")
    }

    if (!is.numeric(gene_migration_prob) || gene_migration_prob < 0 || gene_migration_prob > 1) {
      stop("gene_migration_prob must be a numeric value between 0 and 1.")
    }
  }

  original_cols <- setdiff(names(data), target_col)
  datetime_cols <- names(data)[vapply(data, .is_datetime_col, logical(1))]
  datetime_cols <- setdiff(datetime_cols, target_col)
  numeric_cols <- names(data)[vapply(data, is.numeric, logical(1))]
  numeric_cols <- setdiff(numeric_cols, target_col)
  numeric_cols <- setdiff(numeric_cols, datetime_cols)
  categorical_cols <- setdiff(original_cols, c(numeric_cols, datetime_cols))

  classes <- NULL
  num_class <- NULL
  if (task == "multiclass") {
    target_factor <- as.factor(data[[target_col]])
    classes <- levels(target_factor)
    num_class <- length(classes)
  }

  if (verbose) {
    message("Starting Evolutionary Feature Engineering...")
    message(sprintf("  Task: %s", task))
    message(sprintf("  Evaluator: %s", evaluator))

    if (evaluation_strategy == "cv") {
      message(sprintf("  Generations: %d, Population Size: %d, CV Folds: %d", generations, pop_size, cv_folds))
    } else {
      if (!is.null(split_ids)) {
        counts <- table(split_ids)
        lbls <- intersect(c("train", "val", "holdout"), names(counts))
        ratios <- round(as.numeric(counts[lbls]) / sum(counts), 3)
        ratio_str <- paste(ratios, collapse = "/")
        message(sprintf(
          "  Generations: %d, Population Size: %d, Strategy: Split (%s)",
          generations, pop_size, ratio_str
        ))
      } else {
        message(sprintf(
          "  Generations: %d, Population Size: %d, Strategy: Split (%s)",
          generations, pop_size, paste(split_ratio, collapse = "/")
        ))
      }
    }
    message(sprintf("  Original Numeric columns: %s", truncate_cols(numeric_cols)))
    message(sprintf("  Original Categorical columns: %s", truncate_cols(categorical_cols)))
    if (length(datetime_cols) > 0) {
      message(sprintf("  Original Datetime columns: %s", truncate_cols(datetime_cols)))
    }
  }
  # Pre-calculate fixed CV folds or split IDs
  fold_ids <- NULL
  shared_folds <- NULL
  split_ids_val <- NULL
  shared_splits <- NULL

  if (evaluation_strategy == "cv") {
    fold_ids <- cut(seq(1, nrow(data)), breaks = cv_folds, labels = FALSE)
    fold_ids <- sample(fold_ids)

    # Shared data.table cache for folds and full data to avoid redundant computations
    shared_folds <- list()
    for (f in 1:cv_folds) {
      shared_folds[[f]] <- list(
        train = data.table::as.data.table(data[fold_ids != f, ]),
        val = data.table::as.data.table(data[fold_ids == f, ])
      )
    }
  } else if (evaluation_strategy == "split") {
    if (is.null(split_ids)) {
      split_ids_val <- stratified_split(data[[target_col]], split_ratio)
    } else {
      split_ids_val <- split_ids
    }

    shared_splits <- list(
      train = data.table::as.data.table(data[split_ids_val == "train", ]),
      val = data.table::as.data.table(data[split_ids_val == "val", ])
    )
    if ("holdout" %in% split_ids_val) {
      shared_splits$holdout <- data.table::as.data.table(data[split_ids_val == "holdout", ])
    }

    if (verbose) {
      msg_split <- sprintf("  Split sizes -> Train: %d, Val: %d", nrow(shared_splits$train), nrow(shared_splits$val))
      if (!is.null(shared_splits$holdout)) {
        msg_split <- paste0(msg_split, sprintf(", Holdout: %d", nrow(shared_splits$holdout)))
      }
      message(msg_split)
    }
  } else {
    stop("Unknown evaluation_strategy. Must be 'cv' or 'split'.")
  }

  shared_full <- data.table::as.data.table(data)

  # Fitness cache to avoid re-evaluating identical recipes
  fitness_cache <- new.env(hash = TRUE, parent = emptyenv())
  # State cache for full dataset to avoid re-fitting stateful transformers
  state_cache <- new.env(hash = TRUE, parent = emptyenv())

  # 1. Generation 0: Evaluate baseline individual first (original features only)
  baseline_ind <- create_individual(genes = list(), numeric_cols = numeric_cols, categorical_cols = categorical_cols, datetime_cols = datetime_cols)
  if (verbose) {
    message("\n--- Generation 0 (Baseline) ---")
    message(sprintf("  Individual 1: %s", individual_to_recipe_string(baseline_ind)))
  }
  baseline_ind <- evaluate_fitness(
    baseline_ind, data, target_col,
    task = task, cv_folds = cv_folds,
    evaluation_strategy = evaluation_strategy,
    split_ids = split_ids_val, shared_splits = shared_splits,
    evaluator = evaluator, fold_ids = fold_ids,
    shared_folds = shared_folds,
    shared_full = shared_full, state_cache = state_cache,
    threads = threads, metric = metric, verbose = verbose,
    complexity_penalty = complexity_penalty, ...
  )
  if (verbose) {
    message(sprintf("  Tested Individual 1 -> Fitness: %.4f", baseline_ind$fitness))
  }

  # Cache the baseline individual's fitness
  recipe_str <- individual_to_recipe_string(baseline_ind)
  cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
  assign(cache_key, baseline_ind, envir = fitness_cache)

  if (islands == 1) {
    # 2. Initialize population for Generation 1 using baseline importances
    pop <- initialize_population(pop_size, numeric_cols, categorical_cols, datetime_cols = datetime_cols, initial_genes = 2, task = task, importances = baseline_ind$importances, allowed_transformers = allowed_transformers)
    pop[[1]] <- baseline_ind

    global_best_fitness <- baseline_ind$fitness
    running_best_fitness <- baseline_ind$fitness
    generations_without_improvement <- 0
    fitness_history <- numeric(generations)

    historical_best_genes <- list()
    current_pop_size <- pop_size

    for (g in 1:generations) {
      if (verbose) {
        if (global_best_fitness > -Inf) {
          message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
        } else {
          message(sprintf("\n--- Generation %d / %d ---", g, generations))
        }
        # Print all individuals in the population for this generation
        for (i in seq_along(pop)) {
          fit_str <- if (is.na(pop[[i]]$fitness)) "Unevaluated" else sprintf("%.4f", pop[[i]]$fitness)
          message(sprintf("  Individual %d (%s): %s", i, fit_str, individual_to_recipe_string(pop[[i]])))
        }
      }

      # Evaluate fitness
      eval_res <- evaluate_pop(pop, data, target_col, task, cv_folds, evaluation_strategy,
        split_ids_val, shared_splits, evaluator,
        fold_ids, shared_folds, shared_full, state_cache,
        fitness_cache, threads, verbose, running_best_fitness,
        metric = metric, complexity_penalty = complexity_penalty, ...
      )
      pop <- eval_res$pop
      running_best_fitness <- eval_res$running_best_fitness

      # Sort population by fitness descending
      fitness_vals <- sapply(pop, function(ind) ind$fitness)
      pop <- pop[order(fitness_vals, decreasing = TRUE)]

      # Track historical best genes from this generation
      historical_best_genes <- c(historical_best_genes, pop[[1]]$genes)

      best_fitness <- pop[[1]]$fitness
      fitness_history[g] <- best_fitness
      if (verbose) message(sprintf("  Gen %d Best Fitness: %.4f", g, best_fitness))

      if (verbose) {
        message(sprintf("  Gen %d Best Recipe: %s", g, individual_to_recipe_string(pop[[1]])))
      }

      # Early stopping check
      if (g == 1 || (!is.na(best_fitness) && (is.na(global_best_fitness) || best_fitness > global_best_fitness))) {
        global_best_fitness <- best_fitness
        generations_without_improvement <- 0
      } else {
        generations_without_improvement <- generations_without_improvement + 1
      }

      if (!is.null(early_stopping_generations) && generations_without_improvement >= early_stopping_generations) {
        message(sprintf("  Early stopping triggered after %d generations without improvement.", early_stopping_generations))
        fitness_history <- fitness_history[1:g]
        break
      }

      if (g == generations) break

      # Selection: keep top 50% of current population
      num_survivors <- min(length(pop), max(2, floor(length(pop) / 2)))
      survivors <- pop[1:num_survivors]

      # Collect outputs from evaluated genes — only these are safe for chaining
      tested_gene_outputs <- unique(unlist(lapply(pop, function(ind) {
        if (length(ind$genes) == 0) {
          return(character(0))
        }
        vapply(ind$genes, function(g) g$output_col, character(1))
      })))

      # Aggregate importances from survivors
      global_importances <- list()
      for (s in survivors) {
        if (length(s$importances) > 0) {
          for (feat in names(s$importances)) {
            if (is.null(global_importances[[feat]])) {
              global_importances[[feat]] <- c(s$importances[[feat]])
            } else {
              global_importances[[feat]] <- c(global_importances[[feat]], s$importances[[feat]])
            }
          }
        }
      }

      if (length(global_importances) > 0) {
        global_importances_vec <- sapply(global_importances, mean)
      } else {
        global_importances_vec <- numeric(0)
      }
      # Adaptive mutation rate and temperature: increase exploration during stagnation
      stagnation_ratio <- if (!is.null(early_stopping_generations) && early_stopping_generations > 0) {
        min(1, generations_without_improvement / early_stopping_generations)
      } else {
        0
      }
      adaptive_mutation_rate <- 0.3 + 0.4 * stagnation_ratio
      temperature <- 0.1 + 0.9 * stagnation_ratio

      # Determine target population size (Stagnation Expansion / Gradual Contraction State-Machine)
      target_pop_size <- pop_size
      if (dynamic_population) {
        if (generations_without_improvement > 0) {
          # Expand population during stagnation relative to current size
          current_pop_size <- max(current_pop_size + 1, floor(current_pop_size * dynamic_population_growth_rate))
        } else {
          # Gradual decay back to pop_size when there is improvement
          current_pop_size <- max(pop_size, floor(current_pop_size * dynamic_population_decay_rate))
        }
        target_pop_size <- min(current_pop_size, pop_size * 5L)
      }

      # Next generation
      next_gen <- list()
      # Elitism: keep best
      next_gen[[1]] <- survivors[[1]]

      # Fill the rest
      while (length(next_gen) < target_pop_size) {
        idx <- length(next_gen) + 1
        is_expansion <- idx > pop_size

        if (is_expansion) {
          # Expansion slots: High exploration (no crossover, extremely high temperature)
          p <- tournament_select(pop, k = 3)
          child <- mutate(p, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers)
        } else if (stats::runif(1) < (1 - adaptive_mutation_rate)) {
          # Crossover
          p1 <- tournament_select(pop, k = 3)
          p2 <- tournament_select(pop, k = 3)

          # Determine whether to use union or random crossover
          use_union <- FALSE
          if (crossover_type == "union") {
            use_union <- TRUE
          } else if (crossover_type == "both") {
            use_union <- stats::runif(1) < 0.5
          }

          if (use_union) {
            child <- union_crossover(p1, p2, verbose = FALSE)
          } else {
            child <- crossover(p1, p2, verbose = FALSE)
          }

          if (stats::runif(1) < 0.2) {
            child <- mutate(child, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers)
          }
        } else {
          # Mutate
          p <- tournament_select(pop, k = 3)
          child <- mutate(p, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers)
        }

        # Validation Check: Duplicate in next_gen OR already known to be worse than best
        attempts <- 0
        while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 15) {
          child <- mutate(child, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = if (is_expansion) 100.0 else temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers)
          attempts <- attempts + 1
        }

        next_gen <- c(next_gen, list(child))
      }
      pop <- next_gen
    }

    # Final evaluation of new individuals
    eval_res <- evaluate_pop(pop, data, target_col, task, cv_folds, evaluation_strategy,
      split_ids_val, shared_splits, evaluator,
      fold_ids, shared_folds, shared_full, state_cache,
      fitness_cache, threads, verbose, running_best_fitness,
      metric = metric, complexity_penalty = complexity_penalty, ...
    )
    pop <- eval_res$pop
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]

    best_ind <- pop[[1]]

  } else {
    # 2. Initialize populations for all islands
    pop_list <- list()
    for (j in 1:islands) {
      pop_list[[j]] <- initialize_population(
        pop_size, numeric_cols, categorical_cols, datetime_cols = datetime_cols,
        initial_genes = 2, task = task, importances = baseline_ind$importances,
        allowed_transformers = allowed_transformers
      )
      pop_list[[j]][[1]] <- baseline_ind
    }

    global_best_fitness <- baseline_ind$fitness
    global_best_individual <- baseline_ind

    # Local trackers for each island
    island_best_fitness <- rep(baseline_ind$fitness, islands)
    island_best_individual <- lapply(1:islands, function(x) baseline_ind)
    island_gens_without_improvement <- rep(0, islands)
    island_current_pop_size <- rep(pop_size, islands)

    # Global trackers
    generations_without_improvement <- 0
    fitness_history <- numeric(generations)

    historical_best_genes <- list()

    # Gene-level migration pool: list of lists
    migrated_genes_pool <- lapply(1:islands, function(x) list())

    for (g in 1:generations) {
      if (verbose) {
        if (global_best_fitness > -Inf) {
          message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
        } else {
          message(sprintf("\n--- Generation %d / %d ---", g, generations))
        }
      }

      # Evaluate, sort, and breed for each island sequentially
      for (j in 1:islands) {
        if (verbose) {
          message(sprintf("\n  --- [Island %d] (Current Local Best Fitness: %.4f) ---", j, island_best_fitness[j]))
          # Print all individuals in the population for this island
          for (i in seq_along(pop_list[[j]])) {
            fit_str <- if (is.na(pop_list[[j]][[i]]$fitness)) "Unevaluated" else sprintf("%.4f", pop_list[[j]][[i]]$fitness)
            message(sprintf("    [Island %d] Individual %d (%s): %s", j, i, fit_str, individual_to_recipe_string(pop_list[[j]][[i]])))
          }
        }

        # Evaluate fitness of this island's population
        eval_res <- evaluate_pop(pop_list[[j]], data, target_col, task, cv_folds, evaluation_strategy,
          split_ids_val, shared_splits, evaluator,
          fold_ids, shared_folds, shared_full, state_cache,
          fitness_cache, threads, verbose, island_best_fitness[j],
          metric = metric, complexity_penalty = complexity_penalty, island = j, ...
        )
        pop_list[[j]] <- eval_res$pop

        # Sort population by fitness descending
        fitness_vals <- sapply(pop_list[[j]], function(ind) ind$fitness)
        pop_list[[j]] <- pop_list[[j]][order(fitness_vals, decreasing = TRUE)]

        # Track historical best genes from this generation
        historical_best_genes <- c(historical_best_genes, pop_list[[j]][[1]]$genes)

        best_fitness_island <- pop_list[[j]][[1]]$fitness
        if (verbose) {
          message(sprintf("    [Island %d] Gen %d Best Fitness: %.4f", j, g, best_fitness_island))
          message(sprintf("    [Island %d] Gen %d Best Recipe: %s", j, g, individual_to_recipe_string(pop_list[[j]][[1]])))
        }

        # Local early stopping/progress track for island-specific stagnation
        if (best_fitness_island > island_best_fitness[j]) {
          island_best_fitness[j] <- best_fitness_island
          island_best_individual[[j]] <- pop_list[[j]][[1]]
          island_gens_without_improvement[j] <- 0
        } else {
          island_gens_without_improvement[j] <- island_gens_without_improvement[j] + 1
        }

        # Track global best across all islands
        if (best_fitness_island > global_best_fitness) {
          global_best_fitness <- best_fitness_island
          global_best_individual <- pop_list[[j]][[1]]
        }
      }

      # Track global improvement
      fitness_history[g] <- global_best_fitness
      if (g == 1 || (global_best_fitness > fitness_history[max(1, g - 1)])) {
        generations_without_improvement <- 0
      } else {
        generations_without_improvement <- generations_without_improvement + 1
      }

      # Global early stopping check
      if (!is.null(early_stopping_generations) && generations_without_improvement >= early_stopping_generations) {
        message(sprintf("  Early stopping triggered after %d generations without global improvement.", early_stopping_generations))
        fitness_history <- fitness_history[1:g]
        break
      }

      if (g == generations) break

      # --- MIGRATION PHASE ---
      if (g %% migration_interval == 0) {
        if (verbose) {
          message(sprintf("\n*** [Migration Phase] Triggering migration at Generation %d ***", g))
        }

        # Temp copy of populations to avoid using updated destination populations within the same step
        old_pop_list <- pop_list

        for (j in 1:islands) {
          dest <- (j %% islands) + 1

          # 1. Recipe-level migration (guarded for dynamic population sizes)
          effective_rate <- min(migration_rate, length(old_pop_list[[j]]),
                                length(pop_list[[dest]]) - 1L)
          if (effective_rate > 0) {
            migrant_inds <- old_pop_list[[j]][1:effective_rate]

            # Replace the worst individuals of the target population
            worst_start <- length(pop_list[[dest]]) - effective_rate + 1
            worst_end <- length(pop_list[[dest]])
            pop_list[[dest]][worst_start:worst_end] <- migrant_inds

            if (verbose) {
              message(sprintf("  Migrating top %d recipe(s) from Island %d to Island %d", effective_rate, j, dest))
            }
          }

          # 2. Gene-level migration — ring topology
          # Genes diffuse naturally through the ring over successive migrations,
          # creating a gradient of innovation: each island builds hierarchically
          # on its neighbor's genes rather than all islands converging immediately.
          best_ind <- old_pop_list[[j]][[1]]
          best_genes <- best_ind$genes
          if (length(best_genes) > 0) {
            existing_formulas <- vapply(migrated_genes_pool[[dest]], gene_to_formula, character(1))
            n_injected <- 0L
            for (g_mig in best_genes) {
              formula <- gene_to_formula(g_mig)
              if (!formula %in% existing_formulas) {
                migrated_genes_pool[[dest]] <- c(migrated_genes_pool[[dest]], list(g_mig))
                n_injected <- n_injected + 1L
              }
            }
            if (length(migrated_genes_pool[[dest]]) > 20) {
              migrated_genes_pool[[dest]] <- tail(migrated_genes_pool[[dest]], 20)
            }
            if (verbose && n_injected > 0) {
              message(sprintf("  Injected %d gene(s) into Island %d gene pool from Island %d", n_injected, dest, j))
            }
          }
        }
      }

      # --- BREEDING PHASE FOR NEXT GENERATION ---
      for (j in 1:islands) {
        pop <- pop_list[[j]]

        # Selection: keep top 50% of current population
        num_survivors <- min(length(pop), max(2, floor(length(pop) / 2)))
        survivors <- pop[1:num_survivors]

        # Collect outputs from evaluated genes
        tested_gene_outputs <- unique(unlist(lapply(pop, function(ind) {
          if (length(ind$genes) == 0) {
            return(character(0))
          }
          vapply(ind$genes, function(g) g$output_col, character(1))
        })))

        # Aggregate importances from survivors
        global_importances <- list()
        for (s in survivors) {
          if (length(s$importances) > 0) {
            for (feat in names(s$importances)) {
              if (is.null(global_importances[[feat]])) {
                global_importances[[feat]] <- c(s$importances[[feat]])
              } else {
                global_importances[[feat]] <- c(global_importances[[feat]], s$importances[[feat]])
              }
            }
          }
        }

        if (length(global_importances) > 0) {
          global_importances_vec <- sapply(global_importances, mean)
        } else {
          global_importances_vec <- numeric(0)
        }

        # Adaptive mutation rate and temperature: increase exploration during stagnation
        stagnation_ratio <- if (!is.null(early_stopping_generations) && early_stopping_generations > 0) {
          min(1, island_gens_without_improvement[j] / early_stopping_generations)
        } else {
          0
        }
        adaptive_mutation_rate <- 0.3 + 0.4 * stagnation_ratio
        temperature <- 0.1 + 0.9 * stagnation_ratio

        # Determine target population size (Stagnation Expansion / Gradual Contraction State-Machine)
        target_pop_size <- pop_size
        if (dynamic_population) {
          if (island_gens_without_improvement[j] > 0) {
            island_current_pop_size[j] <- max(island_current_pop_size[j] + 1, floor(island_current_pop_size[j] * dynamic_population_growth_rate))
          } else {
            island_current_pop_size[j] <- max(pop_size, floor(island_current_pop_size[j] * dynamic_population_decay_rate))
          }
          target_pop_size <- min(island_current_pop_size[j], pop_size * 5L)
        }

        next_gen <- list()
        next_gen[[1]] <- survivors[[1]]

        # Fill the rest
        while (length(next_gen) < target_pop_size) {
          idx <- length(next_gen) + 1
          is_expansion <- idx > pop_size

          if (is_expansion) {
            p <- tournament_select(pop, k = 3)
            child <- mutate(p, verbose = FALSE, force_add = TRUE, importances = global_importances_vec,
                            temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs,
                            allowed_transformers = allowed_transformers,
                            migrated_genes = migrated_genes_pool[[j]],
                            gene_migration_prob = gene_migration_prob)
          } else if (stats::runif(1) < (1 - adaptive_mutation_rate)) {
            p1 <- tournament_select(pop, k = 3)
            p2 <- tournament_select(pop, k = 3)

            use_union <- FALSE
            if (crossover_type == "union") {
              use_union <- TRUE
            } else if (crossover_type == "both") {
              use_union <- stats::runif(1) < 0.5
            }

            if (use_union) {
              child <- union_crossover(p1, p2, verbose = FALSE)
            } else {
              child <- crossover(p1, p2, verbose = FALSE)
            }

            if (stats::runif(1) < 0.2) {
              child <- mutate(child, verbose = FALSE, importances = global_importances_vec,
                              temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs,
                              allowed_transformers = allowed_transformers,
                              migrated_genes = migrated_genes_pool[[j]],
                              gene_migration_prob = gene_migration_prob)
            }
          } else {
            p <- tournament_select(pop, k = 3)
            child <- mutate(p, verbose = FALSE, importances = global_importances_vec,
                            temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs,
                            allowed_transformers = allowed_transformers,
                            migrated_genes = migrated_genes_pool[[j]],
                            gene_migration_prob = gene_migration_prob)
          }

          # Validation Check: Duplicate in next_gen OR already known to be worse than best
          attempts <- 0
          while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 15) {
            child <- mutate(child, verbose = FALSE, force_add = TRUE, importances = global_importances_vec,
                            temperature = if (is_expansion) 100.0 else temperature, task = task,
                            tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers,
                            migrated_genes = migrated_genes_pool[[j]],
                            gene_migration_prob = gene_migration_prob)
            attempts <- attempts + 1
          }

          next_gen <- c(next_gen, list(child))
        }
        pop_list[[j]] <- next_gen
      }
    }

    # Final evaluation of individuals on all islands
    for (j in 1:islands) {
      eval_res <- evaluate_pop(pop_list[[j]], data, target_col, task, cv_folds, evaluation_strategy,
        split_ids_val, shared_splits, evaluator,
        fold_ids, shared_folds, shared_full, state_cache,
        fitness_cache, threads, verbose, island_best_fitness[j],
        metric = metric, complexity_penalty = complexity_penalty, island = j, ...
      )
      pop_list[[j]] <- eval_res$pop
      fitness_vals <- sapply(pop_list[[j]], function(ind) ind$fitness)
      pop_list[[j]] <- pop_list[[j]][order(fitness_vals, decreasing = TRUE)]

      if (pop_list[[j]][[1]]$fitness > global_best_fitness) {
        global_best_fitness <- pop_list[[j]][[1]]$fitness
        global_best_individual <- pop_list[[j]][[1]]
      }
    }

    # Combine all island populations for final selection
    pop <- unlist(pop_list, recursive = FALSE)
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]
    best_ind <- global_best_individual
  }

  if (model_all_final_genes) {
    if (verbose) {
      message("\nEvaluating pooled features (all final genes)...")
    }

    # 1. Collect all genes from all individuals in the final population
    all_genes <- unlist(lapply(pop, function(ind) ind$genes), recursive = FALSE)

    # 2. De-duplicate genes by their unique output column name
    unique_cols <- unique(vapply(all_genes, function(g) g$output_col, character(1)))
    deduped_genes <- list()
    for (gene in all_genes) {
      if (gene$output_col %in% unique_cols) {
        deduped_genes[[gene$output_col]] <- gene
        unique_cols <- setdiff(unique_cols, gene$output_col)
      }
    }
    deduped_genes <- unname(deduped_genes)

    # 3. Create the super-individual
    super_ind <- create_individual(
      genes = deduped_genes,
      numeric_cols = numeric_cols,
      categorical_cols = categorical_cols,
      datetime_cols = datetime_cols
    )

    # 4. Evaluate the super-individual's fitness
    super_ind <- evaluate_fitness(
      super_ind, data, target_col,
      task = task, cv_folds = cv_folds,
      evaluation_strategy = evaluation_strategy,
      split_ids = split_ids_val, shared_splits = shared_splits,
      evaluator = evaluator, fold_ids = fold_ids,
      shared_folds = shared_folds,
      shared_full = shared_full, state_cache = state_cache,
      threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
      complexity_penalty = complexity_penalty, ...
    )

    if (is.null(super_ind$best_params) && !is.null(best_ind$best_params)) {
      super_ind$best_params <- best_ind$best_params
    }

    if (!is.na(super_ind$fitness) && (is.na(best_ind$fitness) || super_ind$fitness > best_ind$fitness)) {
      if (verbose) {
        message(sprintf(
          "  Pooled features improved validation fitness from %.4f to %.4f. Using pooled features.",
          best_ind$fitness, super_ind$fitness
        ))
      }
      best_ind <- super_ind
    } else {
      if (verbose) {
        message(sprintf(
          "  Pooled features (fitness: %.4f) did not exceed best individual (fitness: %.4f). Using best individual.",
          super_ind$fitness, best_ind$fitness
        ))
      }
    }
  }

  if (model_all_historical_genes) {
    if (verbose) {
      message("\nEvaluating historical pooled features (best genes from all generations)...")
    }

    # Append the final selected best individual's genes to historical best genes
    historical_best_genes <- c(historical_best_genes, best_ind$genes)

    if (length(historical_best_genes) > 0) {
      # De-duplicate genes by their unique output column name
      unique_cols_hist <- unique(vapply(historical_best_genes, function(g) g$output_col, character(1)))
      deduped_historical_genes <- list()
      for (gene in historical_best_genes) {
        if (gene$output_col %in% unique_cols_hist) {
          deduped_historical_genes[[gene$output_col]] <- gene
          unique_cols_hist <- setdiff(unique_cols_hist, gene$output_col)
        }
      }
      deduped_historical_genes <- unname(deduped_historical_genes)

      # Create the historical super-individual
      super_ind_hist <- create_individual(
        genes = deduped_historical_genes,
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols,
        datetime_cols = datetime_cols
      )

      # Evaluate the historical super-individual's fitness
      super_ind_hist <- evaluate_fitness(
        super_ind_hist, data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids_val, shared_splits = shared_splits,
        evaluator = evaluator, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
        complexity_penalty = complexity_penalty, ...
      )

      if (is.null(super_ind_hist$best_params) && !is.null(best_ind$best_params)) {
        super_ind_hist$best_params <- best_ind$best_params
      }

      if (!is.na(super_ind_hist$fitness) && (is.na(best_ind$fitness) || super_ind_hist$fitness > best_ind$fitness)) {
        if (verbose) {
          message(sprintf(
            "  Historical pooled features improved validation fitness from %.4f to %.4f. Using historical pooled features.",
            best_ind$fitness, super_ind_hist$fitness
          ))
        }
        best_ind <- super_ind_hist
      } else {
        if (verbose) {
          message(sprintf(
            "  Historical pooled features (fitness: %.4f) did not exceed current best fitness (fitness: %.4f). Keeping current best individual.",
            super_ind_hist$fitness, best_ind$fitness
          ))
        }
      }
    } else {
      if (verbose) {
        message("  No historical genes found to evaluate.")
      }
    }
  }

  if (evaluation_strategy == "split" && ("holdout" %in% split_ids_val || !is.null(shared_splits$holdout))) {
    best_ind <- evaluate_holdout_fitness(best_ind, data, split_ids_val, shared_splits,
      target_col, task, evaluator, threads, state_cache,
      classes, num_class,
      metric = metric, verbose = verbose, ...
    )
  }

  if (verbose) {
    message(sprintf("\nEvolution Complete. Best Fitness: %.4f", best_ind$fitness))
    if (!is.null(best_ind$holdout_fitness) && !is.na(best_ind$holdout_fitness)) {
      message(sprintf("Best Holdout Fitness: %.4f", best_ind$holdout_fitness))
    }
    message(sprintf("Best recipe: %s", individual_to_recipe_string(best_ind)))
    if (length(best_ind$genes) > 0) {
      best_cols_str <- paste(sapply(best_ind$genes, function(g) g$output_col), collapse = ", ")
      message(sprintf("Generated columns: %s", best_cols_str))
    }
  }

  # Train best model on the full data using the best evolved features
  if (verbose) {
    message("Training final model on full dataset...")
  }
  best_params <- best_ind$best_params
  res_full <- apply_individual(best_ind, shared_full, NULL, target_col, state_cache = state_cache)
  best_ind <- res_full$ind

  gene_cols <- if (length(best_ind$genes) > 0) vapply(best_ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(best_ind$numeric_cols, best_ind$categorical_cols, best_ind$datetime_cols, gene_cols)

  x_full <- data.matrix(res_full$train[, features, with = FALSE])
  x_full[!is.finite(x_full)] <- NA
  y_full <- res_full$train[[target_col]]
  if (task == "multiclass") {
    y_full <- as.integer(factor(y_full, levels = classes)) - 1
  }

  # Train the final model on the full dataset. Since we are using all data, we use the original
  # tuner evaluator (e.g., lightgbm_mbo) so that it performs hyperparameter tuning on the full
  # dataset, using the best parameters found during evolution as a seed.
  res_model <- train_model(x_full, y_full,
    task = task, evaluator = evaluator,
    threads = threads, num_class = num_class, metric = metric,
    verbose = verbose, best_params = best_params, ...
  )
  best_model <- res_model$model

  structure(
    list(
      best_individual = best_ind,
      history = pop,
      fitness_history = fitness_history,
      task = task,
      best_model = best_model,
      evaluator = evaluator,
      classes = classes,
      metric = metric
    ),
    class = "evo_recipe"
  )
}
