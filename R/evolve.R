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
#'
#' Selects the individual with the highest fitness among a randomly chosen tournament of size \code{k}.
#'
#' @param pop List of candidate individual objects (each with a numeric \code{fitness} element).
#' @param k Integer tournament size (number of candidates drawn at random).
#' @return The winning candidate individual object from \code{pop}.
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
#' @param evaluation_strategy "cv", "split", or "metacv". Strategy to evaluate candidate recipes.
#'   When "metacv", cross-validation folds are mapped across islands: island j trains on K-1 folds
#'   and validates on fold j, offering a Kx speedup while testing out-of-fold generalization on migration.
#' @param split_ratio A numeric vector of length 2 or 3 defining
#'   train/validation/holdout proportions (e.g. c(0.6, 0.2, 0.2)).
#' @param split_ids An optional character vector of split assignments (e.g.
#'   \code{c("train", "train", "val", "holdout", "train")}). Must have the same
#'   length as the number of rows in \code{data} and contain only "train", "val",
#'   or "holdout" labels (with at least "train" and "val" present). When
#'   provided, \code{evaluation_strategy} is automatically set to "split" and the
#'   actual split proportions are computed from the vector.
#' @param holdout_frac Numeric in \code{[0, 0.5)}. When greater than 0 with
#'   \code{evaluation_strategy = "cv"} or \code{"metacv"}, this fraction of rows is stratified and
#'   held out of the entire evolutionary search. After evolution, the winning
#'   recipe (with its frozen transformer states) and the final model are scored
#'   once on these never-seen rows; the result is exposed as
#'   \code{holdout_fitness} / \code{search_gap} on the returned object — an
#'   unbiased estimate of generalization that reveals how much the search
#'   overfit its selection folds.
#' @param cv_strategy Fold construction strategy for CV: \code{"random"}
#'   (default, rows shuffled into folds), \code{"time"} (rows ordered by
#'   \code{time_col} and split into contiguous chronological blocks so validation
#'   always lies in the future of training), or \code{"group"} (all rows sharing
#'   a \code{group_col} value land in the same fold). Use \code{"time"} for
#'   temporal data and \code{"group"} for clustered data to avoid leakage.
#' @param time_col Column name used when \code{cv_strategy = "time"}. Must be
#'   datetime or numeric.
#' @param group_col Column name used when \code{cv_strategy = "group"}.
#' @param multi_fidelity Logical (default FALSE). If TRUE, individuals during
#'   warm-up generations are first screened on row-subsampled folds
#'   (\code{mf_sample_frac}); the most promising half is then re-evaluated at
#'   full fidelity before any selection decision, so all fitness comparisons
#'   remain apples-to-apples. Reduces compute cost with minimal search-quality loss.
#' @param mf_sample_frac Row fraction kept per fold during multi-fidelity
#'   screening, in \code{(0, 1)}.
#' @param mf_warmup_frac Fraction of generations (of \code{generations}) run in
#'   low-fidelity screening mode before full-fidelity-only evaluation begins.
#' @param early_stopping_generations Stop if fitness doesn't improve for this
#'   many generations
#' @param evaluator The ML model to use ("lightgbm", "xgboost", "catboost", or a
#'   custom registered evaluator name).
#' @param seed Optional integer. Seeds the entire stochastic pipeline (fold
#'   construction, holdout split, population initialization, mutation and
#'   crossover) without touching the caller's \code{.Random.seed}: the user's
#'   RNG state is saved on entry and restored on exit (CRAN-safe). Island
#'   \code{j} derives its initial population from \code{seed + 1000*j}, so
#'   island identities are stable regardless of island count. The seed is also
#'   forwarded to the final model fit (and to evaluators that accept one).
#'   Multi-threaded LightGBM/XGBoost remain only statistically reproducible
#'   due to floating-point reduction order; use \code{threads = 1} for bitwise
#'   identical reruns.
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
#' @param metric The metric to optimize ("default", "auc", "f1", "mae", "cal_rmse", "cal_mae", or a
#'   custom function).
#' @param model_all_final_genes Logical. If TRUE, the final model is trained using
#'   the union of all unique genes evolved in the final population, rather than
#'   only the best individual's genes.
#' @param model_all_historical_genes Logical. If TRUE, the final model is trained
#'   using the union of all unique genes evolved across all generations, rather
#'   than only the best individual's genes.
#' @param allowed_transformers Character vector of allowed transformer names,
#'   or \code{"all"} / \code{"basic"} / \code{"robust"} / \code{"clustering"}.
#' @param complexity_penalty Non-negative numeric multiplier for complexity penalty (default 0).
#'   When set to \code{1.0}, applies standard BIC or PAC-Bayes parsimony pressure.
#'   A value of \code{0} disables complexity penalisation.
#' @param complexity_mode Character string specifying the complexity penalty strategy:
#'   \code{"bic_dynamic"} (default, asymptotic BIC scaling \code{ln(N) / (2N)} dynamically relaxed with progress),
#'   \code{"bic"} (constant asymptotic BIC penalty \code{ln(N) / (2N)} throughout evolution),
#'   \code{"pac_bayes_dynamic"} (PAC-Bayes generalization bound scaling \code{1 / (2*sqrt(N))} dynamically relaxed with progress),
#'   \code{"pac_bayes"} (constant PAC-Bayes generalization bound scaling \code{1 / (2*sqrt(N))}),
#'   or \code{"none"} (disabled).
#' @param complexity_floor Numeric in \code{[0, 1]}. Minimum safety floor factor for dynamic penalties (default \code{0.20}, representing a 20\% minimum floor of base penalty).
#' @param complexity_target Character string specifying the complexity count target:
#'   \code{"all_features"} (default, penalizes total number of active features including raw features and genes, rewarding active feature pruning) or \code{"genes"} (penalizes only derived genes).
#' @param migration Optional \code{evo_migration_config} object created by \code{migration_config()}.
#' @param islands Integer. Number of islands for multi-island parallel evolution (default 1).
#' @param migration_interval Integer. Number of generations between migrations (default 5).
#' @param migration_rate Integer. Number of top individuals to migrate from each island to its neighbor (default 1).
#' @param gene_migration_prob Numeric. Probability of injecting a migrated gene during mutation (default 0.2).
#' @param migration_topology Character string specifying the island migration scheme: \code{"ring"} (default unidirectional ring), \code{"gibbs_stagnation"} (probabilistic push targeting stagnated islands), \code{"gibbs_fitness"} (probabilistic push targeting lower-fitness islands), \code{"dual_gibbs_pull"} (demand-driven pull where stagnated islands request migrants from high-fitness donors), or \code{"random"} (uniform random destination).
#' @param migration_temperature Numeric > 0. Temperature parameter for Gibbs softmax migration probability distributions (default 1.0).
#' @param pull_stagnation_threshold Integer >= 1. Stagnation generation threshold used as sigmoid midpoint for pull requests in \code{"dual_gibbs_pull"} (default 3).
#' @param raw_toggle_prob Numeric in \code{[0, 1]}. Probability that a mutation
#'   event toggles one or more raw input features in an individual's active mask
#'   rather than adding/modifying/removing a gene.  A dynamic geometric
#'   distribution determines how many features are toggled per event.  Default
#'   \code{0.15}.
#' @param recalculate_mask_prob Numeric in \code{[0, 1]}. Probability that a
#'   mutation event completely recalculates the individual's active raw feature
#'   mask from scratch using feature importances and a sigmoid inclusion
#'   probability.  Default \code{0.05}.
#' @param mask_temp_factor Numeric > 0. Temperature scaling factor applied to
#'   feature importances during active mask initialization and recalculation.
#'   Higher values flatten the importance distribution (more uniform sampling);
#'   lower values concentrate sampling on the highest-importance features.
#'   Default \code{0.5}.
#' @param row_split_islands Logical. If TRUE, splits data rows across islands (default FALSE).
#' @param per_island_validation Logical. If TRUE, evaluates candidate recipes using each island's specific row split (default FALSE).
#' @param record Logical. If TRUE, records detailed evolutionary logs and launches the interactive evolution live viewer (default FALSE).
#' @param port Optional port number for the live viewer server. If NULL, a random free port is used (or retrieves from the global option 'evoFE.viewer_port').
#' @param ... Additional arguments passed to the underlying evaluator training
#'   functions.
#' @importFrom utils tail head
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
                            holdout_frac = 0,
                            cv_strategy = "random", time_col = NULL, group_col = NULL,
                            multi_fidelity = FALSE, mf_sample_frac = 0.5,
                            mf_warmup_frac = 0.5,
                            early_stopping_generations = 3, evaluator = "lightgbm",
                            seed = NULL,
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
                            complexity_mode = "bic_dynamic",
                            complexity_floor = 0.20,
                            complexity_target = "all_features",
                            migration = NULL,
                            islands = 1,
                            migration_interval = 5,
                            migration_rate = 1,
                            gene_migration_prob = 0.2,
                            migration_topology = "ring",
                            migration_temperature = 1.0,
                            pull_stagnation_threshold = 3,
                            raw_toggle_prob = 0.15,
                            recalculate_mask_prob = 0.05,
                            mask_temp_factor = 0.5,
                            row_split_islands = FALSE,
                            per_island_validation = FALSE,
                            record = FALSE,
                            port = NULL, ...) {
  # Validate complexity arguments
  if (!is.numeric(complexity_penalty) || length(complexity_penalty) != 1 || complexity_penalty < 0) {
    stop("'complexity_penalty' must be a non-negative number.")
  }
  complexity_mode <- match.arg(complexity_mode, c("bic_dynamic", "bic", "pac_bayes_dynamic", "pac_bayes", "none"))
  if (!is.numeric(complexity_floor) || length(complexity_floor) != 1 || complexity_floor < 0 || complexity_floor > 1) {
    stop("'complexity_floor' must be a numeric value between 0 and 1.")
  }
  complexity_target <- match.arg(complexity_target, c("all_features", "genes"))

  # Normalize evaluation_strategy
  if (is.character(evaluation_strategy) && length(evaluation_strategy) == 1) {
    if (evaluation_strategy %in% c("meta_cv", "metacv")) {
      evaluation_strategy <- "metacv"
    }
  }

  # If custom migration config is provided, sync islands count and topology from migration$topology
  if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    islands <- migration$topology$islands
    if (!is.null(migration$topology$type)) {
      migration_topology <- migration$topology$type
    }
  }

  # Validate metacv islands & cv_folds alignment
  if (evaluation_strategy == "metacv") {
    if (missing(islands) && !missing(cv_folds)) {
      islands <- cv_folds
    } else if (!missing(islands) && missing(cv_folds)) {
      cv_folds <- islands
    } else if (missing(islands) && missing(cv_folds)) {
      islands <- 3L
      cv_folds <- 3L
    } else if (islands != cv_folds) {
      stop("'islands' must equal 'cv_folds' when evaluation_strategy is 'metacv'.")
    }
    if (islands < 2) {
      stop("evaluation_strategy = 'metacv' requires at least 2 islands (got 1).")
    }
  }

  # Validate island parameters early to support list allowed_transformers validation
  if (!is.numeric(islands) || islands < 1) {
    stop("islands must be a positive integer >= 1.")
  }
  islands <- as.integer(islands)

  # Validate row_split_islands
  if (!is.logical(row_split_islands) || length(row_split_islands) != 1) {
    stop("row_split_islands must be a logical scalar (TRUE or FALSE).")
  }
  if (row_split_islands && evaluation_strategy == "metacv") {
    stop("row_split_islands is not supported with evaluation_strategy = 'metacv'. metacv automatically partitions folds across islands.")
  }
  if (row_split_islands && islands == 1) {
    warning("row_split_islands is TRUE but islands is 1. Setting row_split_islands to FALSE.")
    row_split_islands <- FALSE
  }

  # Validate per_island_validation
  if (!is.logical(per_island_validation) || length(per_island_validation) != 1) {
    stop("per_island_validation must be a logical scalar (TRUE or FALSE).")
  }
  if (per_island_validation && !row_split_islands) {
    stop("per_island_validation = TRUE requires row_split_islands = TRUE.")
  }
  if (per_island_validation && evaluation_strategy %in% c("cv", "metacv")) {
    stop("per_island_validation = TRUE is only supported with evaluation_strategy = 'split'.")
  }

  # Validate record
  if (!is.logical(record) || length(record) != 1) {
    stop("record must be a logical scalar (TRUE or FALSE).")
  }

  # Parse & validate evaluator (supports single evaluator or vector/list per island)
  if (is.list(evaluator)) {
    evaluator <- unlist(evaluator)
  }
  if (!is.character(evaluator)) {
    stop("evaluator must be a character string or vector of evaluator names.")
  }
  if (length(evaluator) == 1) {
    island_evaluators <- rep(evaluator, islands)
  } else if (length(evaluator) == islands) {
    island_evaluators <- evaluator
  } else {
    stop(sprintf("Length of evaluator (%d) must be 1 or match the number of islands (%d).", length(evaluator), islands))
  }

  for (ev in island_evaluators) {
    if (!exists(ev, envir = evo_evaluators)) {
      stop(sprintf(
        "Evaluator '%s' is not registered in evo_evaluators. Registered evaluators: %s",
        ev, paste(names(evo_evaluators), collapse = ", ")
      ))
    }
  }
  evaluator_main <- island_evaluators[1]

  # Parse allowed_transformers
  all_trans <- names(evo_transformers)

  .resolve_allowed_transformers <- function(at, all_t) {
    if (is.null(at)) at <- "all"
    if (length(at) == 1) {
      if (at == "all") {
        at <- all_t
      } else if (at == "basic") {
        at <- intersect(all_t, c(
          "add", "subtract", "multiply", "divide",
          "log", "sqrt", "reciprocal", "power", "displaced_log",
          "normalized_difference", "frequency_encode",
          "one_hot_encode", "target_encode", "pooled_target_encode", "target_encode_multiclass",
          "feature_hash",
          "rank_transform", "groupby_mean", "groupby_min", "groupby_max", "concat"
        ))
      } else if (at == "clustering") {
        at <- intersect(all_t, c(
          "genie", "genie_centroid_dist", "lumbermark", "lumbermark_centroid_dist",
          "mst_score", "deadwood", "umap", "random_projection", "truncated_svd",
          "pca", "umap_genie", "umap_lumbermark", "mca", "famd", "between_group_pca"
        ))
      } else if (at == "robust") {
        at <- intersect(all_t, c(
          "log", "sqrt", "reciprocal", "power", "displaced_log", "rank_transform",
          "add", "subtract", "multiply", "divide",
          "normalized_difference", "log_ratio",
          "target_encode", "pooled_target_encode", "woe_encode", "frequency_encode",
          "feature_hash",
          "groupby_mean", "groupby_median", "groupby_sd",
          "groupby_zscore", "groupby_ratio", "groupby_quantile",
          "groupby_min", "groupby_max",
          "quantile_binning", "pca", "concat", "mca", "famd", "between_group_pca"
        ))
      }
    }
    at <- intersect(at, all_t)
    if (length(at) == 0) {
      warning("No valid transformers found in 'allowed_transformers'. Falling back to 'all'.")
      at <- all_t
    }
    at
  }

  if (is.list(allowed_transformers)) {
    if (islands == 1) {
      if (length(allowed_transformers) > 1) {
        stop("If allowed_transformers is a list, its length must match the number of islands (1).")
      }
      allowed_transformers <- .resolve_allowed_transformers(allowed_transformers[[1]], all_trans)
    } else {
      if (length(allowed_transformers) != islands) {
        stop(sprintf("If allowed_transformers is a list, its length (%d) must match the number of islands (%d).", length(allowed_transformers), islands))
      }
      allowed_transformers <- lapply(allowed_transformers, .resolve_allowed_transformers, all_t = all_trans)
    }
  } else {
    allowed_transformers <- .resolve_allowed_transformers(allowed_transformers, all_trans)
  }

  get_island_transformers <- function(j) {
    if (is.list(allowed_transformers)) {
      allowed_transformers[[j]]
    } else {
      allowed_transformers
    }
  }

  # Temporarily configure max clustering size and threads options
  old_max_size <- getOption("evoFE.max_clustering_size")
  old_threads <- getOption("evoFE.threads")
  old_opt_dt <- getOption("datatable.threads")
  options(evoFE.max_clustering_size = max_clustering_size, evoFE.threads = threads)

  # Query all initial thread settings BEFORE setting any thread limits
  old_dt <- if (requireNamespace("data.table", quietly = TRUE)) tryCatch(data.table::getDTthreads(), error = function(e) NULL) else NULL
  old_omp <- if (requireNamespace("RhpcBLASctl", quietly = TRUE)) tryCatch(RhpcBLASctl::omp_get_max_threads(), error = function(e) NULL) else NULL
  old_blas <- if (requireNamespace("RhpcBLASctl", quietly = TRUE)) tryCatch(RhpcBLASctl::blas_get_num_procs(), error = function(e) NULL) else NULL
  old_qf <- if (requireNamespace("quitefastmst", quietly = TRUE)) tryCatch(quitefastmst::omp_get_max_threads(), error = function(e) NULL) else NULL

  # Scrupulously register on.exit() restorations in the top-level evolve_features() frame
  on.exit(
    {
      options(evoFE.max_clustering_size = old_max_size)
      options(evoFE.threads = old_threads)
      if (!is.null(old_omp) && !is.na(old_omp) && is.numeric(old_omp) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        tryCatch(RhpcBLASctl::omp_set_num_threads(old_omp), error = function(e) NULL)
      }
      if (!is.null(old_qf) && !is.na(old_qf) && is.numeric(old_qf) && requireNamespace("quitefastmst", quietly = TRUE)) {
        tryCatch(quitefastmst::omp_set_num_threads(old_qf), error = function(e) NULL)
      }
      if (!is.null(old_blas) && !is.na(old_blas) && is.numeric(old_blas) && requireNamespace("RhpcBLASctl", quietly = TRUE)) {
        tryCatch(RhpcBLASctl::blas_set_num_threads(old_blas), error = function(e) NULL)
      }
      if (!is.null(old_dt) && !is.na(old_dt) && is.numeric(old_dt)) {
        tryCatch(data.table::setDTthreads(old_dt), error = function(e) NULL)
      }
      if (!is.null(old_opt_dt) && !is.na(old_opt_dt) && is.numeric(old_opt_dt)) {
        tryCatch(options(datatable.threads = old_opt_dt), error = function(e) NULL)
      }
    },
    add = TRUE
  )

  # ---- Reproducible RNG scope (CRAN-safe save/restore of .Random.seed) ----
  # Registered AFTER the thread-restoration handler above on purpose: multiple
  # on.exit handlers run in registration order, and
  # quitefastmst::omp_set_num_threads() recreates .Random.seed as a side
  # effect, so this restore must be the final one to execute.
  # Validate seed
  if (!is.null(seed) && (!is.numeric(seed) || length(seed) != 1 || is.na(seed) ||
      !is.finite(seed))) {
    stop("'seed' must be NULL or a single finite number.")
  }
  seed <- if (!is.null(seed)) as.integer(seed) else NULL
  old_seed <- if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
    get(".Random.seed", envir = globalenv(), inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = globalenv())
    } else if (exists(".Random.seed", envir = globalenv(), inherits = FALSE)) {
      rm(".Random.seed", envir = globalenv())
    }
  }, add = TRUE)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  # Only AFTER capturing initial values, apply thread modifications if requested
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    if (!is.null(old_omp) && !is.na(old_omp) && is.numeric(old_omp) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(RhpcBLASctl::omp_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
    if (!is.null(old_blas) && !is.na(old_blas) && is.numeric(old_blas) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(RhpcBLASctl::blas_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    if (!is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(data.table::setDTthreads(as.integer(threads)), error = function(e) NULL)
    }
  }
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    if (!is.null(old_qf) && !is.na(old_qf) && is.numeric(old_qf) && !is.null(threads) && !is.na(threads) && is.numeric(threads)) {
      tryCatch(quitefastmst::omp_set_num_threads(as.integer(threads)), error = function(e) NULL)
    }
  }

  if (!task %in% c("classification", "multiclass", "regression")) {
    stop("task must be one of: 'classification', 'multiclass', 'regression'.")
  }

  if (!is.function(metric)) {
    metric_lower <- tolower(metric)
    valid_metrics <- list(
      classification = c("default", "auc", "f1", "eval-ts-refinement", "ts-refinement", "ts_refinement"),
      multiclass = c("default", "auc", "eval-ts-refinement", "ts-refinement", "ts_refinement"),
      regression = c("default", "mae", "cal_rmse", "cal-rmse", "cal_mae", "cal-mae")
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
    if (evaluation_strategy %in% c("cv", "metacv")) {
      warning(sprintf("split_ids was provided but evaluation_strategy is '%s'. Setting evaluation_strategy to 'split'.", evaluation_strategy))
      evaluation_strategy <- "split"
    }
  }

  # Carve a confirmation holdout that is excluded from the entire search and
  # only scored once, after evolution completes (see holdout_frac).
  confirmation_dt <- NULL
  if (holdout_frac > 0) {
    if (!evaluation_strategy %in% c("cv", "metacv")) {
      warning("'holdout_frac' is ignored with evaluation_strategy = 'split'. Use split_ids or a 3-part split_ratio instead.")
    } else {
      conf_ids <- stratified_split(data[[target_col]], c(1 - holdout_frac, holdout_frac))
      conf_rows <- which(conf_ids == "val")
      confirmation_dt <- data.table::as.data.table(data[conf_rows, ])
      data <- data[which(conf_ids != "val"), ]
      if (verbose) {
        message(sprintf(
          "  Confirmation holdout: %d rows (%.0f%%) held out of the search entirely.",
          nrow(confirmation_dt), 100 * holdout_frac
        ))
      }
      if (nrow(confirmation_dt) < 5) {
        warning("Confirmation holdout has fewer than 5 rows; skipping final confirmation scoring.")
        confirmation_dt <- NULL
      }
    }
  }

  # Validate holdout confirmation fraction
  if (!is.numeric(holdout_frac) || length(holdout_frac) != 1 ||
      is.na(holdout_frac) || holdout_frac < 0 || holdout_frac >= 0.5) {
    stop("'holdout_frac' must be a single number in [0, 0.5).")
  }

  # Validate CV fold strategy
  cv_strategy <- match.arg(cv_strategy, c("random", "time", "group"))
  if (!evaluation_strategy %in% c("cv", "metacv")) {
    if (cv_strategy != "random") {
      warning("'cv_strategy' is only used with evaluation_strategy = 'cv' or 'metacv'. Ignoring.")
      cv_strategy <- "random"
    }
  } else if (cv_strategy == "time") {
    if (is.null(time_col) || length(time_col) != 1 || !time_col %in% names(data)) {
      stop("'cv_strategy = \"time\"' requires 'time_col' to name an existing column in data.")
    }
  } else if (cv_strategy == "group") {
    if (is.null(group_col) || length(group_col) != 1 || !group_col %in% names(data)) {
      stop("'cv_strategy = \"group\"' requires 'group_col' to name an existing column in data.")
    }
    if (group_col == target_col) {
      stop("'group_col' must not be the target column.")
    }
  }

  # Validate multi-fidelity evaluation
  if (!is.logical(multi_fidelity) || length(multi_fidelity) != 1 || is.na(multi_fidelity)) {
    stop("'multi_fidelity' must be TRUE or FALSE.")
  }
  if (!is.numeric(mf_sample_frac) || length(mf_sample_frac) != 1 ||
      is.na(mf_sample_frac) || mf_sample_frac <= 0 || mf_sample_frac >= 1) {
    stop("'mf_sample_frac' must be a single number in (0, 1).")
  }
  if (!is.numeric(mf_warmup_frac) || length(mf_warmup_frac) != 1 ||
      is.na(mf_warmup_frac) || mf_warmup_frac < 0 || mf_warmup_frac > 1) {
    stop("'mf_warmup_frac' must be a single number in [0, 1].")
  }

  # Validate island parameters
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

    valid_topologies <- c("ring", "grid", "torus", "hypercube", "tiered", "hfc", "complete", "feature_distance", "gibbs_stagnation", "gibbs_fitness", "dual_gibbs_pull", "random")
    if (!is.character(migration_topology) || length(migration_topology) != 1 ||
      !migration_topology %in% valid_topologies) {
      stop(sprintf("migration_topology must be one of: %s", paste(valid_topologies, collapse = ", ")))
    }

    if (!is.numeric(migration_temperature) || migration_temperature <= 0) {
      stop("migration_temperature must be a positive numeric value > 0.")
    }

    if (!is.numeric(pull_stagnation_threshold) || pull_stagnation_threshold < 1) {
      stop("pull_stagnation_threshold must be a positive integer >= 1.")
    }
    pull_stagnation_threshold <- as.integer(pull_stagnation_threshold)
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
    message(sprintf("  Evaluators (%d islands): %s", islands, paste(island_evaluators, collapse = ", ")))

    if (evaluation_strategy == "cv") {
      message(sprintf("  Generations: %d, Population Size: %d, CV Folds: %d", generations, pop_size, cv_folds))
    } else if (evaluation_strategy == "metacv") {
      message(sprintf("  Generations: %d, Population Size: %d, Strategy: MetaCV (%d islands / folds)", generations, pop_size, islands))
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
  island_shared_splits <- NULL
  island_shared_folds <- NULL

  if (evaluation_strategy == "cv") {
    fold_ids <- .build_cv_folds(data, cv_folds, cv_strategy, time_col, group_col)

    if (row_split_islands) {
      island_shared_folds <- lapply(1:islands, function(j) list())
      for (f in 1:cv_folds) {
        train_indices <- which(fold_ids != f)
        train_indices <- sample(train_indices)
        split_indices <- split(train_indices, cut(seq_along(train_indices), islands, labels = FALSE))
        for (j in 1:islands) {
          island_shared_folds[[j]][[f]] <- list(
            train = data.table::as.data.table(data[split_indices[[j]], ]),
            val = data.table::as.data.table(data[fold_ids == f, ])
          )
        }
      }
    } else {
      # Shared data.table cache for folds and full data to avoid redundant computations
      shared_folds <- list()
      for (f in 1:cv_folds) {
        shared_folds[[f]] <- list(
          train = data.table::as.data.table(data[fold_ids != f, ]),
          val = data.table::as.data.table(data[fold_ids == f, ])
        )
      }
    }
  } else if (evaluation_strategy == "metacv") {
    fold_ids <- .build_cv_folds(data, islands, cv_strategy, time_col, group_col)
    island_shared_splits <- lapply(1:islands, function(j) {
      list(
        train = data.table::as.data.table(data[fold_ids != j, ]),
        val   = data.table::as.data.table(data[fold_ids == j, ])
      )
    })
    shared_splits <- list(
      train = data.table::as.data.table(data),
      val   = data.table::as.data.table(data)
    )
    if (verbose) {
      message(sprintf("  MetaCV partitions -> %d folds mapped across %d islands", islands, islands))
      for (j in 1:islands) {
        message(sprintf("    Island %d -> Train: %d rows (Folds -%d), Val: %d rows (Fold %d)",
                        j, nrow(island_shared_splits[[j]]$train), j,
                        nrow(island_shared_splits[[j]]$val), j))
      }
    }
  } else if (evaluation_strategy == "split") {
    if (is.null(split_ids)) {
      split_ids_val <- stratified_split(data[[target_col]], split_ratio)
    } else {
      split_ids_val <- split_ids
    }

    global_train_dt <- data.table::as.data.table(data[split_ids_val == "train", ])
    global_val_dt <- data.table::as.data.table(data[split_ids_val == "val", ])
    global_holdout_dt <- NULL
    if ("holdout" %in% split_ids_val) {
      global_holdout_dt <- data.table::as.data.table(data[split_ids_val == "holdout", ])
    }

    if (row_split_islands) {
      train_indices <- which(split_ids_val == "train")
      train_indices <- sample(train_indices)
      split_indices <- split(train_indices, cut(seq_along(train_indices), islands, labels = FALSE))
      island_shared_splits <- list()
      for (j in 1:islands) {
        if (per_island_validation) {
          # Carve a local val from within this island's rows.
          # Use only train+val components of split_ratio; holdout is handled globally.
          local_split_frac <- split_ratio[1] / sum(split_ratio[1:2])
          n_j <- length(split_indices[[j]])
          n_local_train <- max(1L, floor(local_split_frac * n_j))
          local_train_idx <- split_indices[[j]][seq_len(n_local_train)]
          local_val_idx <- split_indices[[j]][seq(n_local_train + 1L, n_j)]
          island_shared_splits[[j]] <- list(
            train = data.table::as.data.table(data[local_train_idx, ]),
            val   = data.table::as.data.table(data[local_val_idx, ])
          )
        } else {
          island_shared_splits[[j]] <- list(
            train = data.table::as.data.table(data[split_indices[[j]], ]),
            val   = global_val_dt
          )
        }
        if (!is.null(global_holdout_dt)) {
          island_shared_splits[[j]]$holdout <- global_holdout_dt
        }
      }
      shared_splits <- list(
        train = global_train_dt,
        val = global_val_dt,
        holdout = global_holdout_dt
      )
    } else {
      shared_splits <- list(
        train = global_train_dt,
        val = global_val_dt
      )
      if (!is.null(global_holdout_dt)) {
        shared_splits$holdout <- global_holdout_dt
      }
    }

    if (verbose) {
      if (row_split_islands) {
        if (per_island_validation) {
          local_split_frac <- split_ratio[1] / sum(split_ratio[1:2])
          n_j_approx <- round(nrow(global_train_dt) / islands)
          n_local_train_approx <- round(local_split_frac * n_j_approx)
          n_local_val_approx <- n_j_approx - n_local_train_approx
          train_size_str <- sprintf(
            "%d (split into %d islands of ~%d local train / ~%d local val rows)",
            nrow(global_train_dt), islands, n_local_train_approx, n_local_val_approx
          )
          msg_split <- sprintf("  Split sizes -> Train: %s, Global Val (tournament only): %d", train_size_str, nrow(global_val_dt))
        } else {
          train_size_str <- sprintf(
            "%d (split into %d local sets of ~%d rows)",
            nrow(global_train_dt), islands, round(nrow(global_train_dt) / islands)
          )
          msg_split <- sprintf("  Split sizes -> Train: %s, Val: %d", train_size_str, nrow(global_val_dt))
        }
      } else {
        msg_split <- sprintf("  Split sizes -> Train: %d, Val: %d", nrow(global_train_dt), nrow(global_val_dt))
      }
      if (!is.null(global_holdout_dt)) {
        msg_split <- paste0(msg_split, sprintf(", Holdout: %d", nrow(global_holdout_dt)))
      }
      message(msg_split)
    }
  } else {
    stop("Unknown evaluation_strategy. Must be 'cv', 'split', or 'metacv'.")
  }

  shared_full <- data.table::as.data.table(data)

  # --- Multi-fidelity evaluation setup ---
  # During warm-up generations, individuals are screened on row-subsampled
  # folds; the most promising fraction is then re-evaluated at full fidelity
  # before any selection decision uses their fitness.
  mf_warmup_gens <- 0L
  mf_shared_folds <- NULL
  mf_island_shared_folds <- NULL
  mf_shared_splits <- NULL
  mf_island_shared_splits <- NULL
  mf_shared_full <- NULL
  if (multi_fidelity) {
    mf_warmup_gens <- max(1L, as.integer(floor(generations * mf_warmup_frac)))
    .mf_subsample <- function(part) {
      if (is.null(part) || nrow(part) < 30L) return(part)
      k <- max(20L, ceiling(nrow(part) * mf_sample_frac))
      if (k >= nrow(part)) return(part)
      part[sample.int(nrow(part), k), ]
    }
    .mf_subsample_fold <- function(fl) {
      if (is.null(fl)) return(NULL)
      res <- list(train = .mf_subsample(fl$train), val = fl$val)
      if (!is.null(fl$holdout)) res$holdout <- fl$holdout
      res
    }
    if (!is.null(shared_folds)) {
      mf_shared_folds <- lapply(shared_folds, .mf_subsample_fold)
    }
    if (!is.null(island_shared_folds)) {
      mf_island_shared_folds <- lapply(island_shared_folds, function(per_island) {
        lapply(per_island, .mf_subsample_fold)
      })
    }
    if (!is.null(shared_splits)) {
      mf_shared_splits <- .mf_subsample_fold(shared_splits)
    }
    if (!is.null(island_shared_splits)) {
      mf_island_shared_splits <- lapply(island_shared_splits, .mf_subsample_fold)
    }
    mf_shared_full <- if (!is.null(shared_full) && nrow(shared_full) > 30L) .mf_subsample(shared_full) else NULL
    if (verbose) {
      message(sprintf(
        "  Multi-fidelity: screening on %.0f%% of training rows (min 30 rows floor) for the first %d generation(s), then full-fidelity promotion.",
        100 * mf_sample_frac, mf_warmup_gens
      ))
    }
  }

  # Fitness cache to avoid re-evaluating identical recipes
  fitness_cache <- new.env(hash = TRUE, parent = emptyenv())
  # State cache for full dataset to avoid re-fitting stateful transformers
  state_cache <- new.env(hash = TRUE, parent = emptyenv())

  viewer <- NULL
  evolution_log <- NULL

  tiers_count <- if (!is.null(migration) && inherits(migration, "evo_migration_config") && !is.null(migration$topology$tiers)) {
    migration$topology$tiers
  } else {
    3L
  }

  topo_obj <- if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    migration$topology
  } else {
    switch(migration_topology,
      "grid" = topology_grid(islands),
      "torus" = topology_torus(islands),
      "hypercube" = topology_hypercube(islands),
      "tiered" = topology_tiered(islands, tiers = tiers_count),
      "hfc" = topology_tiered(islands, tiers = tiers_count),
      "complete" = topology_complete(islands),
      "feature_distance" = topology_feature_distance(islands),
      topology_ring(islands)
    )
  }

  adj_list_payload <- if (!is.null(topo_obj) && !is.null(topo_obj$adj_list)) topo_obj$adj_list else NULL

  policy_str <- "push_uniform"
  policy_thresh_val <- "min_peer"
  payload_str <- "full_individual"
  if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    if (!is.null(migration$payload)) {
      payload_str <- migration$payload
    }
    if (!is.null(migration$policy)) {
      pol <- migration$policy
      if (inherits(pol, "evo_policy_push_uniform")) {
        policy_str <- "push_uniform"
      } else if (inherits(pol, "evo_policy_gibbs_push")) {
        policy_str <- paste0("gibbs_push_", pol$weight_by)
      } else if (inherits(pol, "evo_policy_gibbs_pull")) {
        policy_str <- paste0("gibbs_pull_", pol$weight_by)
      } else if (inherits(pol, "evo_policy_tiered_admission")) {
        policy_str <- "tiered_admission"
        if (!is.null(pol$min_fitness_threshold)) {
          policy_thresh_val <- pol$min_fitness_threshold
        }
      }
    }
  }

  if (record) {
    evolution_log <- list(
      config = list(
        islands = islands, pop_size = pop_size, generations = generations,
        tiers = tiers_count,
        adj_list = adj_list_payload,
        task = task, evaluator = evaluator, evaluation_strategy = evaluation_strategy,
        row_split_islands = row_split_islands, per_island_validation = per_island_validation,
        target_col = target_col, migration_interval = migration_interval,
        migration_topology = migration_topology, migration_policy = policy_str,
        min_fitness_threshold = policy_thresh_val,
        migration_payload = payload_str, migration_temperature = migration_temperature,
        pull_stagnation_threshold = pull_stagnation_threshold,
        early_stopping_generations = early_stopping_generations,
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols,
        datetime_cols = datetime_cols
      ),
      baseline = NULL,
      generations = list(),
      tournament = NULL,
      pooled = NULL,
      historical = NULL,
      final = NULL
    )

    viewer <- start_evolution_viewer(port = port)
    on.exit(if (!is.null(viewer)) tryCatch(viewer$stop(), error = function(e) NULL), add = TRUE)
    if (interactive()) {
      utils::browseURL(viewer$url)
      # Poll for websocket connection (up to 10 seconds)
      max_wait <- 10.0
      slept <- 0.0
      while (is.null(viewer$get_connection()) && slept < max_wait) {
        Sys.sleep(0.1)
        slept <- slept + 0.1
        suppressWarnings(httpuv::service(10))
      }
    }
    viewer$send(list(type = "config", data = evolution_log$config))
  }

  island_fitness_caches <- NULL
  island_state_caches <- NULL
  island_baseline_inds <- list()

  # 1. Generation 0: Evaluate baseline individual first (original features only)
  baseline_ind <- create_individual(
    genes = list(),
    numeric_cols = numeric_cols,
    categorical_cols = categorical_cols,
    datetime_cols = datetime_cols,
    all_numeric_cols = numeric_cols,
    all_categorical_cols = categorical_cols,
    all_datetime_cols = datetime_cols
  )
  if (verbose) {
    message("\n--- Generation 0 (Baseline) ---")
    message(sprintf("  Individual 1: %s", individual_to_recipe_string(baseline_ind)))
  }
  baseline_ind <- evaluate_fitness(
    baseline_ind, data, target_col,
    task = task, cv_folds = cv_folds,
    evaluation_strategy = evaluation_strategy,
    split_ids = split_ids_val, shared_splits = shared_splits,
    evaluator = evaluator_main, fold_ids = fold_ids,
    shared_folds = shared_folds,
    shared_full = shared_full, state_cache = state_cache,
    threads = threads, metric = metric, verbose = verbose,
    complexity_penalty = complexity_penalty, complexity_mode = complexity_mode,
    complexity_floor = complexity_floor,
    complexity_target = complexity_target,
    baseline_fitness = NULL, running_best_fitness = NULL,
    n_samples = nrow(data), ...
  )
  if (verbose) {
    message(sprintf("  Tested Individual 1 -> Fitness: %.4f", baseline_ind$fitness))
  }

  # Cache the baseline individual's fitness
  recipe_str <- individual_to_recipe_string(baseline_ind)
  cache_key <- digest::digest(paste0(evaluator_main, "::", recipe_str), algo = "md5", serialize = FALSE)
  assign(cache_key, baseline_ind, envir = fitness_cache)

  if (islands > 1) {
    island_fitness_caches <- lapply(1:islands, function(x) new.env(hash = TRUE, parent = emptyenv()))
    island_state_caches <- lapply(1:islands, function(x) new.env(hash = TRUE, parent = emptyenv()))
    for (j in 1:islands) {
      local_baseline <- create_individual(
        genes = list(),
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols,
        datetime_cols = datetime_cols,
        all_numeric_cols = numeric_cols,
        all_categorical_cols = categorical_cols,
        all_datetime_cols = datetime_cols
      )
      local_baseline <- evaluate_fitness(
        local_baseline, data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids_val,
        shared_splits = if (row_split_islands || evaluation_strategy == "metacv") island_shared_splits[[j]] else shared_splits,
        evaluator = island_evaluators[j],
        fold_ids = fold_ids,
        shared_folds = if (row_split_islands) island_shared_folds[[j]] else shared_folds,
        shared_full = shared_full,
        state_cache = island_state_caches[[j]],
        threads = threads, metric = metric,
        verbose = FALSE, allow_prune = TRUE,
        complexity_penalty = complexity_penalty, complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = NULL, running_best_fitness = NULL,
        n_samples = nrow(data), ...
      )
      local_baseline$evaluator <- island_evaluators[j]
      island_baseline_inds[[j]] <- local_baseline
      local_recipe_str <- individual_to_recipe_string(local_baseline)
      local_cache_key <- digest::digest(paste0(island_evaluators[j], "::", local_recipe_str), algo = "md5", serialize = FALSE)
      assign(local_cache_key, local_baseline, envir = island_fitness_caches[[j]])

      if (verbose) {
        message(sprintf("  [Island %d Baseline] (%s) -> Fitness: %.4f", j, island_evaluators[j], local_baseline$fitness))
      }
    }
  }

  if (record) {
    # Generate 5-row baseline data sample
    res_sample <- tryCatch(
      {
        apply_individual(baseline_ind, head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = head(shared_full, 5))
    )
    baseline_dt <- res_sample$train
    baseline_list <- lapply(names(baseline_dt), function(col) {
      val <- baseline_dt[[col]]
      if (is.numeric(val)) round(val, 4) else as.character(val)
    })
    names(baseline_list) <- names(baseline_dt)

    evolution_log$baseline <- list(
      fitness = baseline_ind$fitness,
      recipe = individual_to_recipe_string(baseline_ind),
      sample = baseline_list,
      importances = if (!is.null(baseline_ind$importances)) as.list(baseline_ind$importances) else list(),
      islands = if (islands > 1) {
        lapply(1:islands, function(j) {
          list(
            island = j,
            evaluator = island_evaluators[j],
            fitness = island_baseline_inds[[j]]$fitness
          )
        })
      } else {
        NULL
      }
    )
    viewer$send(list(type = "baseline", data = evolution_log$baseline))
  }

  best_ind_source <- "Island 1"

  if (islands == 1) {
    # 2. Initialize population for Generation 1 using baseline importances
    pop <- initialize_population(pop_size, numeric_cols, categorical_cols, datetime_cols = datetime_cols, initial_genes = 2, task = task, importances = baseline_ind$importances, allowed_transformers = allowed_transformers, mask_temp_factor = mask_temp_factor)
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

      if (record) {
        viewer$send(list(type = "status", data = list(island = 1, status = "evaluating", generation = g)))
      }
      # Evaluate fitness
      eval_res <- evaluate_pop_mf(pop, data, target_col, task, cv_folds, evaluation_strategy,
        split_ids_val, shared_splits, evaluator,
        fold_ids, shared_folds, shared_full, state_cache,
        fitness_cache, threads, verbose, running_best_fitness,
        metric = metric, complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_ind$fitness,
        n_samples = nrow(data),
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
        mf_on = multi_fidelity && g <= mf_warmup_gens,
        lf_shared_splits = mf_shared_splits,
        lf_shared_folds = mf_shared_folds, lf_shared_full = mf_shared_full, ...
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
      if (verbose) message(sprintf("  Gen %d Best Recipe: %s", g, individual_to_recipe_string(pop[[1]])))

      # Early stopping check
      if (g == 1 || (!is.na(best_fitness) && (is.na(global_best_fitness) || best_fitness > global_best_fitness))) {
        global_best_fitness <- best_fitness
        generations_without_improvement <- 0
      } else {
        generations_without_improvement <- generations_without_improvement + 1
      }

      if (record) {
        viewer$send(list(type = "island_evaluated", data = list(
          island = 1,
          generation = g,
          best_fitness = pop[[1]]$fitness,
          stagnation = generations_without_improvement,
          all_fitness = fitness_vals
        )))
      }

      if (record) {
        # Generate 5-row transformed data sample
        res_sample <- tryCatch(
          {
            apply_individual(pop[[1]], head(shared_full, 5), NULL, NULL, state_cache = state_cache)
          },
          error = function(e) list(train = head(shared_full, 5))
        )
        best_dt <- res_sample$train
        best_list <- lapply(names(best_dt), function(col) {
          val <- best_dt[[col]]
          if (is.numeric(val)) round(val, 4) else as.character(val)
        })
        names(best_list) <- names(best_dt)

        # Prepare serialized genes
        serialized_genes <- lapply(pop[[1]]$genes, function(gene) {
          col <- gene$output_col
          imp_val <- if (!is.null(pop[[1]]$importances) && col %in% names(pop[[1]]$importances)) {
            as.numeric(pop[[1]]$importances[[col]])
          } else {
            0.0
          }
          list(
            formula = gene_to_formula(gene),
            output_col = col,
            importance = imp_val
          )
        })
        if (length(serialized_genes) > 0) {
          gene_imps <- sapply(serialized_genes, function(x) x$importance)
          serialized_genes <- serialized_genes[order(gene_imps, decreasing = TRUE)]
        }

        gen_snapshot <- list(
          generation = g,
          islands = list(
            list(
              island = 1,
              best_fitness = best_fitness,
              stagnation = generations_without_improvement,
              pop_size = length(pop),
              population = lapply(head(pop, 5), function(ind) {
                list(fitness = ind$fitness, n_genes = length(ind$genes))
              }),
              all_fitness = sapply(pop, function(ind) ind$fitness)
            )
          ),
          global_best_fitness = global_best_fitness,
          global_best_recipe = individual_to_recipe_string(pop[[1]]),
          global_best_n_genes = length(pop[[1]]$genes),
          global_best_importances = if (!is.null(pop[[1]]$importances)) as.list(pop[[1]]$importances) else list(),
          global_best_genes = serialized_genes,
          sample = best_list
        )
        evolution_log$generations[[g]] <- gen_snapshot
        viewer$send(list(type = "generation", data = gen_snapshot))
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

      # Collect outputs from evaluated genes <U+2014> only these are safe for chaining
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
          child <- mutate(p, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
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
            child <- mutate(child, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
          }
        } else {
          # Mutate
          p <- tournament_select(pop, k = 3)
          child <- mutate(p, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
        }

        # Validation Check: Duplicate in next_gen OR already known to be worse than best
        attempts <- 0
        while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 15) {
          child <- mutate(child, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = if (is_expansion) 100.0 else temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
          attempts <- attempts + 1
        }

        next_gen <- c(next_gen, list(child))
      }
      pop <- next_gen
    }

    # Final evaluation of new individuals
    eval_res <- evaluate_pop_mf(pop, data, target_col, task, cv_folds, evaluation_strategy,
      split_ids_val, shared_splits, evaluator,
      fold_ids, shared_folds, shared_full, state_cache,
      fitness_cache, threads, verbose, running_best_fitness,
      metric = metric, complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_ind$fitness,
      n_samples = nrow(data),
      cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
      mf_on = multi_fidelity && g <= mf_warmup_gens,
      lf_shared_splits = mf_shared_splits,
      lf_shared_folds = mf_shared_folds, lf_shared_full = mf_shared_full, ...
    )
    pop <- eval_res$pop
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]

    best_ind <- pop[[1]]
  } else {
    # 2. Initialize populations for all islands
    pop_list <- list()
    for (j in 1:islands) {
      # Per-island sub-stream: island j's initial population depends only on
      # (seed, j), not on how many islands are being run.
      if (!is.null(seed)) set.seed(seed + 1000L * j)
      pop_list[[j]] <- initialize_population(
        pop_size, numeric_cols, categorical_cols,
        datetime_cols = datetime_cols,
        initial_genes = 2, task = task, importances = baseline_ind$importances,
        allowed_transformers = get_island_transformers(j),
        mask_temp_factor = mask_temp_factor
      )
      pop_list[[j]][[1]] <- island_baseline_inds[[j]]
    }

    # Local trackers for each island
    island_best_fitness <- vapply(island_baseline_inds, function(ind) ind$fitness, numeric(1))
    island_best_individual <- lapply(1:islands, function(j) island_baseline_inds[[j]])
    island_gens_without_improvement <- rep(0, islands)
    island_improved_by_migration <- rep(FALSE, islands)
    island_current_pop_size <- rep(pop_size, islands)

    if (row_split_islands || evaluation_strategy == "metacv") {
      best_idx <- which.max(island_best_fitness)
      global_best_fitness <- island_best_fitness[best_idx]
      global_best_individual <- island_baseline_inds[[best_idx]]
      best_ind_source <- paste0("Island ", best_idx)
    } else {
      global_best_fitness <- baseline_ind$fitness
      global_best_individual <- baseline_ind
      best_ind_source <- "Island 1"
    }

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

        if (record) {
          viewer$send(list(type = "status", data = list(island = j, status = "evaluating", generation = g)))
        }
        # Evaluate fitness of this island's population
        eval_res <- evaluate_pop_mf(pop_list[[j]], data, target_col, task, cv_folds, evaluation_strategy,
          split_ids_val,
          if (row_split_islands || evaluation_strategy == "metacv") island_shared_splits[[j]] else shared_splits,
          island_evaluators[j],
          fold_ids,
          if (row_split_islands) island_shared_folds[[j]] else shared_folds,
          shared_full,
          if (row_split_islands || evaluation_strategy == "metacv") island_state_caches[[j]] else state_cache,
          if (row_split_islands || evaluation_strategy == "metacv") island_fitness_caches[[j]] else fitness_cache,
          threads, verbose, island_best_fitness[j],
          metric = metric, complexity_penalty = complexity_penalty,
          complexity_mode = complexity_mode,
          complexity_floor = complexity_floor,
          complexity_target = complexity_target,
          baseline_fitness = if (!is.null(island_baseline_inds[[j]])) island_baseline_inds[[j]]$fitness else baseline_ind$fitness,
          n_samples = nrow(data), island = j,
          cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
          mf_on = multi_fidelity && g <= mf_warmup_gens,
          lf_shared_splits = if (row_split_islands || evaluation_strategy == "metacv") mf_island_shared_splits[[j]] else mf_shared_splits,
          lf_shared_folds = if (row_split_islands) mf_island_shared_folds[[j]] else mf_shared_folds,
          lf_shared_full = mf_shared_full, ...
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
          pop_list[[j]][[1]]$evaluator <- island_evaluators[j]
          island_best_individual[[j]] <- pop_list[[j]][[1]]
          island_gens_without_improvement[j] <- 0
          island_improved_by_migration[j] <- FALSE
        } else if (island_improved_by_migration[j] && !is.na(best_fitness_island) && best_fitness_island == island_best_fitness[j]) {
          island_gens_without_improvement[j] <- 0
          island_improved_by_migration[j] <- FALSE
        } else {
          island_gens_without_improvement[j] <- island_gens_without_improvement[j] + 1
        }

        if (record) {
          viewer$send(list(type = "island_evaluated", data = list(
            island = j,
            generation = g,
            best_fitness = pop_list[[j]][[1]]$fitness,
            stagnation = island_gens_without_improvement[j],
            all_fitness = fitness_vals
          )))
        }

        # Track global best across all islands
        if (best_fitness_island > global_best_fitness) {
          global_best_fitness <- best_fitness_island
          global_best_individual <- pop_list[[j]][[1]]
          best_ind_source <- paste0("Island ", j)
        }
      }

      if (record) {
        # Generate 5-row transformed data sample
        res_sample <- tryCatch(
          {
            apply_individual(global_best_individual, head(shared_full, 5), NULL, NULL, state_cache = state_cache)
          },
          error = function(e) list(train = head(shared_full, 5))
        )
        best_dt <- res_sample$train
        best_list <- lapply(names(best_dt), function(col) {
          val <- best_dt[[col]]
          if (is.numeric(val)) round(val, 4) else as.character(val)
        })
        names(best_list) <- names(best_dt)

        # Prepare serialized genes
        serialized_genes <- lapply(global_best_individual$genes, function(gene) {
          col <- gene$output_col
          imp_val <- if (!is.null(global_best_individual$importances) && col %in% names(global_best_individual$importances)) {
            as.numeric(global_best_individual$importances[[col]])
          } else {
            0.0
          }
          list(
            formula = gene_to_formula(gene),
            output_col = col,
            importance = imp_val
          )
        })
        if (length(serialized_genes) > 0) {
          gene_imps <- sapply(serialized_genes, function(x) x$importance)
          serialized_genes <- serialized_genes[order(gene_imps, decreasing = TRUE)]
        }

        gen_snapshot <- list(
          generation = g,
          islands = lapply(seq_len(islands), function(j) {
            pop_j <- pop_list[[j]]
            list(
              island = j,
              best_fitness = island_best_fitness[j],
              stagnation = island_gens_without_improvement[j],
              pop_size = length(pop_j),
              population = lapply(head(pop_j, 5), function(ind) {
                list(fitness = ind$fitness, n_genes = length(ind$genes))
              }),
              all_fitness = sapply(pop_j, function(ind) ind$fitness)
            )
          }),
          global_best_fitness = global_best_fitness,
          global_best_recipe = individual_to_recipe_string(global_best_individual),
          global_best_n_genes = length(global_best_individual$genes),
          global_best_importances = if (!is.null(global_best_individual$importances)) as.list(global_best_individual$importances) else list(),
          global_best_genes = serialized_genes,
          sample = best_list
        )
        # Find which island contains the global best individual
        for (j in seq_len(islands)) {
          if (identical(pop_list[[j]][[1]], global_best_individual)) {
            gen_snapshot$global_best_island <- j
            break
          }
        }
        evolution_log$generations[[g]] <- gen_snapshot
        viewer$send(list(type = "generation", data = gen_snapshot))
      }

      # Track global improvement
      fitness_history[g] <- global_best_fitness
      if (g == 1 || (global_best_fitness > fitness_history[max(1, g - 1)])) {
        generations_without_improvement <- 0
      } else {
        generations_without_improvement <- generations_without_improvement + 1
      }

      # Early stopping check
      if (!is.null(early_stopping_generations)) {
        if (per_island_validation) {
          # Fitness scores are not comparable across islands <U+2014> stop only when all islands stagnate
          if (all(island_gens_without_improvement >= early_stopping_generations)) {
            message(sprintf(
              "  Early stopping triggered: all %d islands stagnated for %d generations.",
              islands, early_stopping_generations
            ))
            fitness_history <- fitness_history[1:g]
            break
          }
        } else {
          if (generations_without_improvement >= early_stopping_generations) {
            message(sprintf("  Early stopping triggered after %d generations without global improvement.", early_stopping_generations))
            fitness_history <- fitness_history[1:g]
            break
          }
        }
      }

      if (g == generations) break

      # --- MIGRATION PHASE ---
      if (g %% migration_interval == 0) {
        if (verbose) {
          message(sprintf("\n*** [Migration Phase] Triggering migration at Generation %d (Topology: %s) ***", g, migration_topology))
        }

        # Temp copy of populations to avoid using updated destination populations within the same step
        old_pop_list <- pop_list

        # Build migration transactions list: list of list(from, to, is_pull)
        migration_txs <- list()

        state <- list(
          pop_list = pop_list,
          island_best_fitness = island_best_fitness,
          island_gens_without_improvement = island_gens_without_improvement
        )

        if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
          migration_txs <- resolve_migration_transactions(migration$policy, migration$topology, state)
        } else if (migration_topology == "dual_gibbs_pull") {
          for (j in 1:islands) {
            s_j <- island_gens_without_improvement[j]
            p_pull <- 1 / (1 + exp(-(s_j - pull_stagnation_threshold) / migration_temperature))
            if (stats::runif(1) < p_pull) {
              candidates <- setdiff(1:islands, j)
              if (length(candidates) > 0) {
                donor_fits <- island_best_fitness[candidates]
                donor_fits[is.na(donor_fits)] <- -Inf
                max_f <- max(donor_fits)
                if (is.finite(max_f)) {
                  logits <- (donor_fits - max_f) / migration_temperature
                  probs <- exp(logits) / sum(exp(logits))
                } else {
                  probs <- rep(1 / length(candidates), length(candidates))
                }
                donor <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1, prob = probs)
                migration_txs[[length(migration_txs) + 1]] <- list(from = donor, to = j, is_pull = TRUE)
              }
            }
          }
        } else {
          for (j in 1:islands) {
            dest <- j
            if (migration_topology == "ring") {
              dest <- (j %% islands) + 1
            } else if (migration_topology == "random") {
              candidates <- setdiff(1:islands, j)
              dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1)
            } else if (migration_topology == "gibbs_stagnation") {
              candidates <- setdiff(1:islands, j)
              stags <- island_gens_without_improvement[candidates]
              max_s <- max(stags)
              logits <- (stags - max_s) / migration_temperature
              probs <- exp(logits) / sum(exp(logits))
              dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1, prob = probs)
            } else if (migration_topology == "gibbs_fitness") {
              candidates <- setdiff(1:islands, j)
              fits <- island_best_fitness[candidates]
              valid_fits <- fits[!is.na(fits)]
              if (length(valid_fits) > 0) {
                fits[is.na(fits)] <- min(valid_fits)
              } else {
                fits[] <- 0
              }
              max_f <- max(fits)
              diffs <- max_f - fits
              max_d <- max(diffs)
              logits <- (diffs - max_d) / migration_temperature
              probs <- exp(logits) / sum(exp(logits))
              dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1, prob = probs)
            } else if (migration_topology %in% c("tiered", "hfc")) {
              topo_obj <- topology_tiered(islands, tiers = tiers_count)
              policy_obj <- policy_tiered_admission(min_fitness_threshold = policy_thresh_val)
              txs <- resolve_migration_transactions(policy_obj, topo_obj, state)
              migration_txs <- c(migration_txs, txs)
              break
            } else {
              topo_obj <- switch(migration_topology,
                "grid" = topology_grid(islands),
                "torus" = topology_torus(islands),
                "hypercube" = topology_hypercube(islands),
                "complete" = topology_complete(islands),
                "feature_distance" = topology_feature_distance(islands),
                topology_ring(islands)
              )
              candidates <- get_neighbors(topo_obj, j)
              if (length(candidates) == 0) candidates <- setdiff(1:islands, j)
              dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1)
            }

            migration_txs[[length(migration_txs) + 1]] <- list(from = j, to = dest, is_pull = FALSE)
          }
        }

        # Process each migration transaction
        for (tx in migration_txs) {
          src <- tx$from
          dest <- tx$to
          n_injected <- 0L
          effective_rate <- 0L
          new_genes <- list()

          payload_strategy <- if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
            migration$payload
          } else {
            "full_individual"
          }

          # 1. Recipe-level migration (only if payload_strategy == "full_individual")
          if (identical(payload_strategy, "full_individual")) {
            if (length(old_pop_list[[src]]) > 0) {
              effective_rate <- min(
                migration_rate, length(old_pop_list[[src]]),
                length(pop_list[[dest]]) - 1L
              )
            }
            if (effective_rate > 0) {
              migrant_inds <- old_pop_list[[src]][1:effective_rate]

              if (row_split_islands || per_island_validation || evaluation_strategy == "metacv" || island_evaluators[src] != island_evaluators[dest]) {
                # Fitness was evaluated on src local split or evaluator — must re-evaluate on dest local split/evaluator
                migrant_inds <- lapply(migrant_inds, function(ind) {
                  ind$fitness <- NA_real_
                  ind
                })
                eval_migrant <- evaluate_pop(migrant_inds, data, target_col, task, cv_folds, evaluation_strategy,
                  split_ids_val,
                  if (row_split_islands || evaluation_strategy == "metacv") island_shared_splits[[dest]] else shared_splits,
                  island_evaluators[dest],
                  fold_ids,
                  if (row_split_islands) island_shared_folds[[dest]] else shared_folds,
                  shared_full,
                  if (row_split_islands || evaluation_strategy == "metacv") island_state_caches[[dest]] else state_cache,
                  if (row_split_islands || evaluation_strategy == "metacv") island_fitness_caches[[dest]] else fitness_cache,
                  threads, verbose, island_best_fitness[dest],
                  metric = metric, complexity_penalty = complexity_penalty,
                  complexity_mode = complexity_mode,
                  complexity_floor = complexity_floor,
                  complexity_target = complexity_target,
                  baseline_fitness = if (!is.null(island_baseline_inds[[dest]])) island_baseline_inds[[dest]]$fitness else baseline_ind$fitness,
                  n_samples = nrow(data), island = dest, ...
                )
                migrant_inds <- eval_migrant$pop
              }

              # Replace the worst individuals of the target population
              worst_start <- length(pop_list[[dest]]) - effective_rate + 1
              worst_end <- length(pop_list[[dest]])
              pop_list[[dest]][worst_start:worst_end] <- migrant_inds

              # Re-sort destination population immediately by fitness (highest first)
              dest_fits <- vapply(pop_list[[dest]], function(ind) {
                if (!is.null(ind$fitness) && !is.na(ind$fitness)) ind$fitness else -Inf
              }, numeric(1))
              pop_list[[dest]] <- pop_list[[dest]][order(dest_fits, decreasing = TRUE)]

              # Update best tracking immediately for destination island and global best
              new_dest_best <- pop_list[[dest]][[1]]
              is_new_dest_best <- FALSE
              is_new_global_best <- FALSE

              if (!is.null(new_dest_best$fitness) && !is.na(new_dest_best$fitness)) {
                if (is.na(island_best_fitness[dest]) || new_dest_best$fitness > island_best_fitness[dest]) {
                  island_best_fitness[dest] <- new_dest_best$fitness
                  island_best_individual[[dest]] <- new_dest_best
                  island_gens_without_improvement[dest] <- 0L
                  island_improved_by_migration[dest] <- TRUE
                  is_new_dest_best <- TRUE
                }
                if (is.na(global_best_fitness) || new_dest_best$fitness > global_best_fitness) {
                  global_best_fitness <- new_dest_best$fitness
                  global_best_individual <- new_dest_best
                  best_ind_source <- paste0("Island ", dest)
                  is_new_global_best <- TRUE
                }
              }

              if (verbose) {
                msg_prefix <- if (tx$is_pull) "Pulling" else "Migrating"
                message(sprintf(
                  "  %s top %d recipe(s) from Island %d (%s) to Island %d (%s)",
                  msg_prefix, effective_rate, src, island_evaluators[src], dest, island_evaluators[dest]
                ))
                migrant_fit <- if (length(migrant_inds) > 0 && !is.null(migrant_inds[[1]]$fitness)) migrant_inds[[1]]$fitness else NA
                if (!is.na(migrant_fit)) {
                  message(sprintf(
                    "    Evaluated Migrant Fitness on Island %d: %.4f (Current Destination Best: %.4f)",
                    dest, migrant_fit, island_best_fitness[dest]
                  ))
                }
                if (is_new_global_best) {
                  message(sprintf("    [Island %d] New Global Best Fitness: %.4f", dest, global_best_fitness))
                } else if (is_new_dest_best) {
                  message(sprintf("    [Island %d] New Best Fitness: %.4f", dest, island_best_fitness[dest]))
                }
              }
            }
          }

          # 2. Gene-level migration
          if (length(old_pop_list[[src]]) > 0) {
            best_ind <- old_pop_list[[src]][[1]]
            best_genes <- best_ind$genes
            if (length(best_genes) > 0) {
              existing_formulas <- vapply(migrated_genes_pool[[dest]], gene_to_formula, character(1))

              # Find new genes
              new_genes <- list()
              for (g_mig in best_genes) {
                formula <- gene_to_formula(g_mig)
                if (!formula %in% existing_formulas) {
                  new_genes <- c(new_genes, list(g_mig))
                }
              }

              if (length(new_genes) > 0) {
                # Sort new genes by feature importance (highest first)
                gene_imps <- vapply(new_genes, function(g) {
                  col <- g$output_col
                  if (!is.null(best_ind$importances) && col %in% names(best_ind$importances)) {
                    as.numeric(best_ind$importances[[col]])
                  } else {
                    0.0
                  }
                }, double(1))

                new_genes <- new_genes[order(gene_imps, decreasing = TRUE)]

                # Limit to top 20 most important new genes
                if (length(new_genes) > 20) {
                  new_genes <- new_genes[1:20]
                }

                migrated_genes_pool[[dest]] <- c(migrated_genes_pool[[dest]], new_genes)
                n_injected <- length(new_genes)
              } else {
                n_injected <- 0L
              }

              if (length(migrated_genes_pool[[dest]]) > 20) {
                migrated_genes_pool[[dest]] <- tail(migrated_genes_pool[[dest]], 20)
              }
              if (verbose && n_injected > 0) {
                actual_injected <- min(n_injected, 20L)
                message(sprintf("  Injected %d gene(s) into Island %d gene pool from Island %d", actual_injected, dest, src))
              }
            }
          }


          migrated_gene_details <- list()
          if (length(new_genes) > 0) {
            migrated_gene_details <- lapply(new_genes, function(g_mig) {
              col <- g_mig$output_col
              imp_val <- if (!is.null(best_ind$importances) && col %in% names(best_ind$importances)) {
                as.numeric(best_ind$importances[[col]])
              } else {
                0.0
              }
              list(
                formula = gene_to_formula(g_mig),
                output_col = col,
                importance = imp_val
              )
            })
          }

          ind_src <- if (length(pop_list[[src]]) > 0) pop_list[[src]][[1]] else NULL
          ind_dest <- if (length(pop_list[[dest]]) > 0) pop_list[[dest]][[1]] else NULL
          feat_dist <- .calc_feature_distance(ind_src, ind_dest)

          migration_event <- list(
            from = src,
            to = dest,
            topology = migration_topology,
            is_pull = tx$is_pull,
            n_recipes = effective_rate,
            n_genes = n_injected,
            migrated_genes = migrated_gene_details,
            feature_distance = feat_dist
          )

          if (record) {
            if (is.null(evolution_log$generations[[g]]$migrations)) {
              evolution_log$generations[[g]]$migrations <- list()
            }
            evolution_log$generations[[g]]$migrations <- c(
              evolution_log$generations[[g]]$migrations,
              list(migration_event)
            )
          }

          if (!is.null(viewer)) {
            viewer$send(list(type = "migration", data = migration_event))
          }
        }

        # Re-sort all island populations descending by fitness so migrated elites participate in survivor selection & elitism
        for (k in 1:islands) {
          fitness_vals <- sapply(pop_list[[k]], function(ind) ind$fitness)
          pop_list[[k]] <- pop_list[[k]][order(fitness_vals, decreasing = TRUE)]
          top_fit <- pop_list[[k]][[1]]$fitness
          if (!is.na(top_fit)) {
            if (is.na(island_best_fitness[k]) || top_fit > island_best_fitness[k]) {
              island_best_fitness[k] <- top_fit
              island_best_individual[[k]] <- pop_list[[k]][[1]]
              island_gens_without_improvement[k] <- 0
              island_improved_by_migration[k] <- TRUE
            }
            if (top_fit > global_best_fitness) {
              global_best_fitness <- top_fit
              global_best_individual <- pop_list[[k]][[1]]
              best_ind_source <- paste0("Island ", k)
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
            child <- mutate(p,
              verbose = FALSE, force_add = TRUE, importances = global_importances_vec,
              temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs,
              allowed_transformers = get_island_transformers(j),
              migrated_genes = migrated_genes_pool[[j]],
              gene_migration_prob = gene_migration_prob,
              raw_toggle_prob = raw_toggle_prob,
              recalculate_mask_prob = recalculate_mask_prob
            )
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
              child <- mutate(child,
                verbose = FALSE, importances = global_importances_vec,
                temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs,
                allowed_transformers = get_island_transformers(j),
                migrated_genes = migrated_genes_pool[[j]],
                gene_migration_prob = gene_migration_prob,
                raw_toggle_prob = raw_toggle_prob,
                recalculate_mask_prob = recalculate_mask_prob
              )
            }
          } else {
            p <- tournament_select(pop, k = 3)
            child <- mutate(p,
              verbose = FALSE, importances = global_importances_vec,
              temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs,
              allowed_transformers = get_island_transformers(j),
              migrated_genes = migrated_genes_pool[[j]],
              gene_migration_prob = gene_migration_prob,
              raw_toggle_prob = raw_toggle_prob,
              recalculate_mask_prob = recalculate_mask_prob
            )
          }

          # Validation Check: Duplicate in next_gen OR already known to be worse than best
          attempts <- 0
          while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 15) {
            child <- mutate(child,
              verbose = FALSE, force_add = TRUE, importances = global_importances_vec,
              temperature = if (is_expansion) 100.0 else temperature, task = task,
              tested_gene_outputs = tested_gene_outputs, allowed_transformers = get_island_transformers(j),
              migrated_genes = migrated_genes_pool[[j]],
              gene_migration_prob = gene_migration_prob,
              raw_toggle_prob = raw_toggle_prob,
              recalculate_mask_prob = recalculate_mask_prob
            )
            attempts <- attempts + 1
          }

          next_gen <- c(next_gen, list(child))
        }
        pop_list[[j]] <- next_gen
      }
    }

    # Final evaluation of individuals on all islands
    for (j in 1:islands) {
      eval_res <- evaluate_pop_mf(pop_list[[j]], data, target_col, task, cv_folds, evaluation_strategy,
        split_ids_val,
        if (row_split_islands || evaluation_strategy == "metacv") island_shared_splits[[j]] else shared_splits,
        island_evaluators[j],
        fold_ids,
        if (row_split_islands) island_shared_folds[[j]] else shared_folds,
        shared_full,
        if (row_split_islands || evaluation_strategy == "metacv") island_state_caches[[j]] else state_cache,
        if (row_split_islands || evaluation_strategy == "metacv") island_fitness_caches[[j]] else fitness_cache,
        threads, verbose, island_best_fitness[j],
        metric = metric, complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = if (!is.null(island_baseline_inds[[j]])) island_baseline_inds[[j]]$fitness else baseline_ind$fitness,
        n_samples = nrow(data), island = j,
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
        mf_on = multi_fidelity && g <= mf_warmup_gens,
        lf_shared_splits = if (row_split_islands || evaluation_strategy == "metacv") mf_island_shared_splits[[j]] else mf_shared_splits,
        lf_shared_folds = if (row_split_islands) mf_island_shared_folds[[j]] else mf_shared_folds,
        lf_shared_full = mf_shared_full, ...
      )
      pop_list[[j]] <- eval_res$pop
      fitness_vals <- sapply(pop_list[[j]], function(ind) ind$fitness)
      pop_list[[j]] <- pop_list[[j]][order(fitness_vals, decreasing = TRUE)]

      if (pop_list[[j]][[1]]$fitness > global_best_fitness) {
        global_best_fitness <- pop_list[[j]][[1]]$fitness
        global_best_individual <- pop_list[[j]][[1]]
        best_ind_source <- paste0("Island ", j)
      }
    }

    # Combine all island populations for final selection
    pop <- unlist(pop_list, recursive = FALSE)
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]
    best_ind <- global_best_individual

    # If multi-fidelity was enabled, ensure all island bests are evaluated at full fidelity
    if (multi_fidelity) {
      for (j in seq_len(islands)) {
        ind_j <- island_best_individual[[j]]
        p_j <- ind_j$val_preds
        n_rows_j <- if (is.matrix(p_j)) nrow(p_j) else length(p_j)
        if (evaluation_strategy != "metacv" && n_rows_j < nrow(data)) {
          ind_j$fitness <- NA_real_
          cand_eval <- if (!is.null(ind_j$evaluator)) ind_j$evaluator else island_evaluators[j]
          ind_j <- evaluate_fitness(
            ind_j, data, target_col,
            task = task, cv_folds = cv_folds,
            evaluation_strategy = evaluation_strategy,
            split_ids = split_ids_val,
            shared_splits = if (row_split_islands) island_shared_splits[[j]] else shared_splits,
            evaluator = cand_eval, fold_ids = fold_ids,
            shared_folds = if (row_split_islands) island_shared_folds[[j]] else shared_folds,
            shared_full = shared_full, state_cache = if (row_split_islands) island_state_caches[[j]] else state_cache,
            threads = threads, metric = metric, verbose = FALSE,
            allow_prune = FALSE,
            complexity_penalty = complexity_penalty,
            complexity_mode = complexity_mode,
            complexity_floor = complexity_floor,
            complexity_target = complexity_target,
            baseline_fitness = if (!is.null(island_baseline_inds[[j]])) island_baseline_inds[[j]]$fitness else baseline_ind$fitness,
            running_best_fitness = global_best_fitness,
            n_samples = nrow(data), ...
          )
          island_best_individual[[j]] <- ind_j
        }
      }
      # Update global best if any re-evaluated island best improved
      best_j_idx <- which.max(sapply(island_best_individual, function(ind) ind$fitness))
      if (island_best_individual[[best_j_idx]]$fitness > best_ind$fitness) {
        best_ind <- island_best_individual[[best_j_idx]]
        global_best_fitness <- best_ind$fitness
      }
    }
  }

  if (row_split_islands) {
    if (per_island_validation) {
      # Tournament: re-evaluate each island's best individual on the full global dataset
      if (verbose) {
        message(sprintf("\nRunning final tournament: re-evaluating best individual from each of %d islands on full training dataset...", islands))
      }
      candidates <- lapply(seq_len(islands), function(j) {
        ind <- island_best_individual[[j]]
        ind$fitness <- NA_real_
        cand_eval <- if (!is.null(ind$evaluator)) ind$evaluator else island_evaluators[j]
        ind <- evaluate_fitness(
          ind, data, target_col,
          task = task, cv_folds = cv_folds,
          evaluation_strategy = evaluation_strategy,
          split_ids = split_ids_val, shared_splits = shared_splits,
          evaluator = cand_eval, fold_ids = fold_ids,
          shared_folds = shared_folds,
          shared_full = shared_full, state_cache = state_cache,
          threads = threads, metric = metric, verbose = FALSE,
          allow_prune = FALSE,
          complexity_penalty = complexity_penalty,
          complexity_mode = complexity_mode,
          complexity_floor = complexity_floor,
          complexity_target = complexity_target,
          baseline_fitness = baseline_ind$fitness,
          running_best_fitness = global_best_fitness,
          n_samples = nrow(data), ...
        )
        if (verbose) {
          message(sprintf("  [Island %d] Global fitness: %.4f  Recipe: %s", j, ind$fitness, individual_to_recipe_string(ind)))
        }
        ind
      })
      tournament_fitness <- sapply(candidates, function(ind) ind$fitness)
      winner_idx <- which.max(tournament_fitness)
      best_ind <- candidates[[winner_idx]]
      island_best_individual <- candidates
      best_ind_source <- paste0("Island ", winner_idx)
      if (verbose) {
        message(sprintf("  Tournament winner: Island %d (fitness %.4f)", winner_idx, best_ind$fitness))
      }
    } else {
      # Re-evaluate all island bests on the full training dataset
      if (verbose) {
        message("\nRe-evaluating island bests on full training dataset...")
      }
      island_best_individual <- lapply(seq_len(islands), function(j) {
        ind_j <- island_best_individual[[j]]
        ind_j$fitness <- NA_real_
        cand_eval <- if (!is.null(ind_j$evaluator)) ind_j$evaluator else island_evaluators[j]
        evaluate_fitness(
          ind_j, data, target_col,
          task = task, cv_folds = cv_folds,
          evaluation_strategy = evaluation_strategy,
          split_ids = split_ids_val, shared_splits = shared_splits,
          evaluator = cand_eval, fold_ids = fold_ids,
          shared_folds = shared_folds,
          shared_full = shared_full, state_cache = state_cache,
          threads = threads, metric = metric, verbose = FALSE,
          allow_prune = FALSE,
          complexity_penalty = complexity_penalty,
          complexity_mode = complexity_mode,
          complexity_floor = complexity_floor,
          complexity_target = complexity_target,
          baseline_fitness = baseline_ind$fitness,
          running_best_fitness = global_best_fitness,
          n_samples = nrow(data), ...
        )
      })
      winner_idx <- which.max(sapply(island_best_individual, function(ind) ind$fitness))
      best_ind <- island_best_individual[[winner_idx]]
      best_ind_source <- paste0("Island ", winner_idx)
      if (verbose) {
        message(sprintf("  Global fitness of best individual: %.4f", best_ind$fitness))
      }
    }
  }

  oof_preds <- NULL
  if (evaluation_strategy == "metacv") {
    # Stitch Out-Of-Fold predictions from each island best
    if (task == "multiclass") {
      oof_preds <- matrix(NA_real_, nrow = nrow(data), ncol = num_class)
      for (j in seq_len(islands)) {
        ind_j <- island_best_individual[[j]]
        val_idx <- which(fold_ids == j)
        if (!is.null(ind_j$val_preds)) {
          vp <- ind_j$val_preds
          if (!is.matrix(vp)) vp <- matrix(vp, ncol = num_class, byrow = FALSE)
          oof_preds[val_idx, ] <- vp
        }
      }
    } else {
      oof_preds <- rep(NA_real_, nrow(data))
      for (j in seq_len(islands)) {
        ind_j <- island_best_individual[[j]]
        val_idx <- which(fold_ids == j)
        if (!is.null(ind_j$val_preds)) {
          oof_preds[val_idx] <- ind_j$val_preds
        }
      }
    }

    # Tournament: evaluate each island's best candidate with full CV to select the global champion
    if (verbose) {
      message(sprintf("\nRunning MetaCV tournament: evaluating full CV fitness for best individual from each of %d islands...", islands))
    }
    candidates <- lapply(seq_len(islands), function(j) {
      ind <- island_best_individual[[j]]
      ind$fitness <- NA_real_
      cand_eval <- if (!is.null(ind$evaluator)) ind$evaluator else island_evaluators[j]
      ind <- evaluate_fitness(
        ind, data, target_col,
        task = task, cv_folds = islands,
        evaluation_strategy = "cv",
        split_ids = NULL, shared_splits = NULL,
        evaluator = cand_eval, fold_ids = fold_ids,
        shared_folds = NULL,
        shared_full = NULL, state_cache = state_cache,
        threads = threads, metric = metric, verbose = FALSE,
        allow_prune = FALSE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_ind$fitness,
        running_best_fitness = global_best_fitness,
        n_samples = nrow(data), ...
      )
      if (verbose) {
        message(sprintf("  [Island %d] Full CV fitness: %.4f  Recipe: %s", j, ind$fitness, individual_to_recipe_string(ind)))
      }
      ind
    })
    island_best_individual <- candidates
    island_best_fitness <- sapply(candidates, function(ind) ind$fitness)
    tournament_fitness <- island_best_fitness
    winner_idx <- which.max(tournament_fitness)
    best_ind <- candidates[[winner_idx]]
    best_ind_source <- paste0("Island ", winner_idx)
    if (verbose) {
      message(sprintf("  MetaCV Tournament winner: Island %d (fitness %.4f)", winner_idx, best_ind$fitness))
    }
  }

  if (record && (row_split_islands || evaluation_strategy == "metacv")) {
    if (per_island_validation) {
      tournament_data <- list(
        candidates = lapply(seq_len(islands), function(j) {
          list(
            island = j,
            local_fitness = island_best_fitness[j],
            global_fitness = tournament_fitness[j],
            recipe = individual_to_recipe_string(candidates[[j]]),
            n_genes = length(candidates[[j]]$genes)
          )
        }),
        winner_island = winner_idx,
        winner_fitness = best_ind$fitness
      )
      evolution_log$tournament <- tournament_data
      viewer$send(list(type = "tournament", data = tournament_data))
    } else {
      tournament_data <- list(
        candidates = list(
          list(
            island = 1,
            local_fitness = global_best_fitness,
            global_fitness = best_ind$fitness,
            recipe = individual_to_recipe_string(best_ind),
            n_genes = length(best_ind$genes)
          )
        ),
        winner_island = 1,
        winner_fitness = best_ind$fitness
      )
      evolution_log$tournament <- tournament_data
      viewer$send(list(type = "tournament", data = tournament_data))
    }
  }

  if (model_all_final_genes) {
    if (verbose) {
      message("\nEvaluating pooled features (all final genes)...")
    }

    # 1. Collect all genes from all individuals in the final population
    all_genes <- unlist(lapply(pop, function(ind) ind$genes), recursive = FALSE)

    # 2. De-duplicate genes by their unique output column name
    unique_cols <- if (length(all_genes) > 0) unique(vapply(all_genes, function(g) g$output_col, character(1))) else character(0)
    deduped_genes <- list()
    for (gene in all_genes) {
      if (gene$output_col %in% unique_cols) {
        deduped_genes[[gene$output_col]] <- gene
        unique_cols <- setdiff(unique_cols, gene$output_col)
      }
    }
    deduped_genes <- unname(deduped_genes)

    super_ind <- NULL
    adopted_pooled <- FALSE

    if (length(deduped_genes) > 0) {
      # 3. Create the super-individual
      super_ind <- create_individual(
        genes = deduped_genes,
        numeric_cols = best_ind$numeric_cols,
        categorical_cols = best_ind$categorical_cols,
        datetime_cols = best_ind$datetime_cols,
        all_numeric_cols = best_ind$all_numeric_cols,
        all_categorical_cols = best_ind$all_categorical_cols,
        all_datetime_cols = best_ind$all_datetime_cols
      )

      best_eval_curr <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main

      # 4. Evaluate the super-individual's fitness
      super_ind <- evaluate_fitness(
        super_ind, data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids_val, shared_splits = shared_splits,
        evaluator = best_eval_curr, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_ind$fitness,
        running_best_fitness = best_ind$fitness,
        n_samples = nrow(data), ...
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
        best_ind$evaluator <- best_eval_curr
        best_ind_source <- "Adopted (Pooled)"
        adopted_pooled <- TRUE
      } else {
        if (verbose) {
          message(sprintf(
            "  Pooled features (fitness: %.4f) did not exceed best individual (fitness: %.4f). Using best individual.",
            super_ind$fitness, best_ind$fitness
          ))
        }
      }
    } else {
      if (verbose) {
        message("  No final genes found to evaluate.")
      }
    }

    if (record) {
      evolution_log$pooled <- list(
        n_genes = length(deduped_genes),
        fitness = if (!is.null(super_ind)) super_ind$fitness else best_ind$fitness,
        adopted = adopted_pooled
      )
      if (!is.null(viewer)) {
        viewer$send(list(type = "pooled", data = evolution_log$pooled))
      }
    }
  }

  if (model_all_historical_genes) {
    if (verbose) {
      message("\nEvaluating historical pooled features (best genes from all generations)...")
    }

    # Append the final selected best individual's genes to historical best genes
    if (!is.null(best_ind$genes) && length(best_ind$genes) > 0) {
      historical_best_genes <- c(historical_best_genes, best_ind$genes)
    }

    deduped_historical_genes <- list()
    super_ind_hist <- NULL
    adopted_hist <- FALSE

    if (length(historical_best_genes) > 0) {
      # De-duplicate genes by their unique output column name
      unique_cols_hist <- unique(vapply(historical_best_genes, function(g) g$output_col, character(1)))
      for (gene in historical_best_genes) {
        if (gene$output_col %in% unique_cols_hist) {
          deduped_historical_genes[[gene$output_col]] <- gene
          unique_cols_hist <- setdiff(unique_cols_hist, gene$output_col)
        }
      }
      deduped_historical_genes <- unname(deduped_historical_genes)
    }

    if (length(deduped_historical_genes) > 0) {
      # Create the historical super-individual
      super_ind_hist <- create_individual(
        genes = deduped_historical_genes,
        numeric_cols = best_ind$numeric_cols,
        categorical_cols = best_ind$categorical_cols,
        datetime_cols = best_ind$datetime_cols,
        all_numeric_cols = best_ind$all_numeric_cols,
        all_categorical_cols = best_ind$all_categorical_cols,
        all_datetime_cols = best_ind$all_datetime_cols
      )

      best_eval_curr <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main
      # Evaluate the historical super-individual's fitness
      super_ind_hist <- evaluate_fitness(
        super_ind_hist, data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids_val, shared_splits = shared_splits,
        evaluator = best_eval_curr, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_ind$fitness,
        running_best_fitness = best_ind$fitness,
        n_samples = nrow(data), ...
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
        best_ind$evaluator <- best_eval_curr
        best_ind_source <- "Adopted (Historical)"
        adopted_hist <- TRUE
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

    if (record) {
      evolution_log$historical <- list(
        n_genes = length(deduped_historical_genes),
        fitness = if (!is.null(super_ind_hist)) super_ind_hist$fitness else best_ind$fitness,
        adopted = adopted_hist
      )
      if (!is.null(viewer)) {
        viewer$send(list(type = "historical", data = evolution_log$historical))
      }
    }
  }

  if (evaluation_strategy == "split" && ("holdout" %in% split_ids_val || !is.null(shared_splits$holdout))) {
    best_eval_curr <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main
    best_ind <- evaluate_holdout_fitness(best_ind, data, split_ids_val, shared_splits,
      target_col, task, best_eval_curr, threads, state_cache,
      classes, num_class,
      metric = metric, verbose = verbose, seed = seed, ...
    )
  }

  if (verbose) {
    if (!is.null(best_ind$raw_fitness) && !is.na(best_ind$raw_fitness)) {
      if (!is.null(best_ind$penalty) && is.finite(best_ind$penalty) && best_ind$penalty > 0) {
        message(sprintf("\nEvolution Complete. Best Validation Score: %.4f (Penalized Selection Fitness: %.4f)", best_ind$raw_fitness, best_ind$fitness))
      } else {
        message(sprintf("\nEvolution Complete. Best Validation Score: %.4f", best_ind$raw_fitness))
      }
    } else {
      message(sprintf("\nEvolution Complete. Best Fitness: %.4f", best_ind$fitness))
    }
    if (!is.null(best_ind$holdout_fitness) && !is.na(best_ind$holdout_fitness)) {
      message(sprintf("Best Holdout Score: %.4f", best_ind$holdout_fitness))
    }
    message(sprintf("Best recipe: %s", individual_to_recipe_string(best_ind)))
    if (length(best_ind$genes) > 0) {
      best_cols_str <- paste(sapply(best_ind$genes, function(g) g$output_col), collapse = ", ")
      message(sprintf("Generated columns: %s", best_cols_str))
    }
  }

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

  best_evaluator <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main
  res_model <- train_model(x_full, y_full,
    task = task, evaluator = best_evaluator,
    threads = threads, num_class = num_class, metric = metric,
    verbose = verbose, best_params = best_params, seed = seed, ...
  )
  best_model <- res_model$model

  if (!is.null(confirmation_dt) && nrow(confirmation_dt) > 0) {
    if (verbose) message("Scoring final recipe on the untouched confirmation holdout...")
    conf_fitness <- NA_real_
    res_conf <- tryCatch(
      apply_individual(best_ind, data.table::copy(confirmation_dt), NULL, NULL, state_cache = state_cache),
      error = function(e) NULL
    )
    if (!is.null(res_conf)) {
      conf_features <- c(res_conf$ind$numeric_cols, res_conf$ind$categorical_cols,
        res_conf$ind$datetime_cols,
        if (length(res_conf$ind$genes) > 0) vapply(res_conf$ind$genes, function(g) g$output_col, character(1)) else character(0)
      )
      x_conf <- data.matrix(res_conf$train[, conf_features, with = FALSE])
      x_conf[!is.finite(x_conf)] <- NA
      preds_conf <- tryCatch(
        evo_evaluators[[best_evaluator]]$predict_func(best_model, x_conf, task = task),
        error = function(e) NULL
      )
      if (!is.null(preds_conf)) {
        y_conf <- confirmation_dt[[target_col]]
        if (task == "multiclass") {
          y_conf_enc <- as.integer(factor(y_conf, levels = classes)) - 1
          if (!is.matrix(preds_conf)) {
            preds_conf <- matrix(preds_conf, ncol = num_class, byrow = TRUE)
          }
          conf_fitness <- compute_metric(y_conf_enc, preds_conf, task, metric, num_class)
        } else {
          conf_fitness <- compute_metric(y_conf, preds_conf, task, metric)
        }
        best_ind$holdout_fitness <- conf_fitness
      }
    }
    if (!verbose && is.na(conf_fitness)) message("Warning: confirmation scoring failed on the holdout.")
  }

  search_gap <- NULL
  if (!is.null(best_ind$holdout_fitness) && !is.na(best_ind$holdout_fitness) &&
      !is.null(best_ind$raw_fitness) && is.finite(best_ind$raw_fitness)) {
    search_gap <- best_ind$holdout_fitness - best_ind$raw_fitness
    if (verbose) {
      message(sprintf(
        "Search gap (holdout - validation): %+.4f. Small gaps indicate the search did not overfit the selection folds.",
        search_gap
      ))
    }
  }

  if (record) {
    res_sample <- tryCatch(
      {
        apply_individual(best_ind, head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = head(shared_full, 5))
    )
    best_dt <- res_sample$train
    best_list <- lapply(names(best_dt), function(col) {
      val <- best_dt[[col]]
      if (is.numeric(val)) round(val, 4) else as.character(val)
    })
    names(best_list) <- names(best_dt)

    serialized_genes <- lapply(best_ind$genes, function(gene) {
      col <- gene$output_col
      imp_val <- if (!is.null(best_ind$importances) && col %in% names(best_ind$importances)) {
        as.numeric(best_ind$importances[[col]])
      } else {
        0.0
      }
      list(
        formula = gene_to_formula(gene),
        output_col = col,
        importance = imp_val
      )
    })
    if (length(serialized_genes) > 0) {
      gene_imps <- sapply(serialized_genes, function(x) x$importance)
      serialized_genes <- serialized_genes[order(gene_imps, decreasing = TRUE)]
    }

    final_data <- list(
      raw_fitness = if (!is.null(best_ind$raw_fitness)) best_ind$raw_fitness else best_ind$fitness,
      penalty = if (!is.null(best_ind$penalty)) best_ind$penalty else 0.0,
      best_fitness = best_ind$fitness,
      best_recipe = individual_to_recipe_string(best_ind),
      holdout_fitness = if (exists("best_ind") && !is.null(best_ind$holdout_fitness)) best_ind$holdout_fitness else NA_real_,
      n_genes = length(best_ind$genes),
      global_best_importances = if (!is.null(best_ind$importances)) as.list(best_ind$importances) else list(),
      global_best_genes = serialized_genes,
      sample = best_list,
      source_island = best_ind_source
    )
    final_data$search_gap <- search_gap
    evolution_log$final <- final_data
    viewer$send(list(type = "complete", data = final_data))
  }

  structure(
    list(
      best_individual = best_ind,
      history = pop,
      fitness_history = fitness_history,
      task = task,
      best_model = best_model,
      evaluator = best_evaluator,
      classes = classes,
      metric = metric,
      holdout_fitness = if (!is.null(best_ind$holdout_fitness)) best_ind$holdout_fitness else NULL,
      search_gap = search_gap,
      cv_strategy = cv_strategy,
      evaluation_strategy = evaluation_strategy,
      fold_ids = if (evaluation_strategy %in% c("cv", "metacv")) fold_ids else NULL,
      split_ids = if (evaluation_strategy == "split" && !is.null(split_ids_val)) split_ids_val else NULL,
      oof_preds = oof_preds,
      island_bests = if (exists("island_best_individual") && !is.null(island_best_individual)) island_best_individual else list(best_ind),
      evolution_log = if (record) evolution_log else NULL
    ),
    class = "evo_recipe"
  )
}
