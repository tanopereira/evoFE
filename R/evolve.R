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
#' @param verbose Integer or logical. If \code{0} or \code{FALSE}, runs silently.
#'   If \code{1} or \code{TRUE}, prints generation progress.
#'   If \code{2}, prints detailed transformer-level logging.
#'   If \code{3}, enables live learner iteration logs for underlying models (LightGBM, XGBoost, CatBoost, RealMLP).
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
#' @param metacv_selection Strategy to select the winning recipe after MetaCV evolution: \code{"fitness"}
#'   (default, selects the island recipe with the highest absolute validation fitness; zero extra CV fits),
#'   \code{"tournament"} (runs full K-fold CV on each island's best candidate, $K^2$ model fits, to select
#'   1 single global champion recipe across identical folds), or \code{"headroom"} (selects the recipe with
#'   the highest normalized headroom closed).
#' @param metacv_mode Deprecated alias for \code{metacv_selection}.
#' @param record Logical. If TRUE, records detailed evolutionary logs and launches the interactive evolution live viewer (default FALSE).
#' @param port Optional port number for the live viewer server. If NULL, a random free port is used (or retrieves from the global option 'evoFE.viewer_port').
#' @param global_unsupervised Logical. If TRUE (default), unsupervised stateful transformers (e.g. UMAP, PCA, Lumbermark, Genie) are fit globally on the full input feature matrix X to ensure invariant cluster IDs and manifold coordinates across CV folds with zero target leakage, while supervised transformers remain strictly per-fold. Set to FALSE for strict per-fold unsupervised fitting.
#' @param ... Additional arguments passed to the underlying evaluator training
#'   functions.
#' @importFrom utils tail head
#' @return An \code{evo_recipe} S3 object:
#'   a list with elements
#'   \code{best_individual} (the top-scoring \code{evo_individual}),
#'   \code{history} (list of all evaluated individuals across generations),
#'   \code{task}, \code{best_model} (the trained model object),
#'   \code{best_iteration} (the optimal number of iterations/epochs determined from validation/early stopping),
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
#'   generations = 2,
#'   pop_size = 5,
#'   cv_folds = 2,
#'   verbose = FALSE
#' )
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
                            threads = default_threads(),
                            max_clustering_size = 5000,
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
                            metacv_selection = c("fitness", "tournament", "headroom"),
                            metacv_mode = NULL,
                            record = FALSE,
                            port = NULL,
                            global_unsupervised = getOption("evoFE.global_unsupervised", TRUE), ...) {
  # Normalize thread aliases passed via ... (e.g. nthreads, nthread, num_threads, n_jobs)
  extra_args_top <- list(...)
  threads <- resolve_thread_count(threads, extra_args_top)

  old_glob_unsup <- getOption("evoFE.global_unsupervised", NULL)
  options(evoFE.global_unsupervised = isTRUE(global_unsupervised))
  on.exit({
    options(evoFE.global_unsupervised = old_glob_unsup)
  }, add = TRUE)

  # Validate complexity arguments
  if (!is.numeric(complexity_penalty) || length(complexity_penalty) != 1 || complexity_penalty < 0) {
    stop("'complexity_penalty' must be a non-negative number.")
  }
  complexity_mode <- match.arg(complexity_mode, c("bic_dynamic", "bic", "pac_bayes_dynamic", "pac_bayes", "none"))
  if (!is.numeric(complexity_floor) || length(complexity_floor) != 1 || complexity_floor < 0 || complexity_floor > 1) {
    stop("'complexity_floor' must be a numeric value between 0 and 1.")
  }
  complexity_target <- match.arg(complexity_target, c("all_features", "genes"))

  # Normalize and validate MetaCV arguments
  meta_conf <- validate_metacv_config(
    evaluation_strategy = evaluation_strategy,
    islands = islands,
    cv_folds = cv_folds,
    migration = migration,
    row_split_islands = row_split_islands,
    per_island_validation = per_island_validation,
    metacv_selection = metacv_selection,
    metacv_mode = metacv_mode,
    missing_islands = missing(islands),
    missing_cv_folds = missing(cv_folds)
  )
  evaluation_strategy <- meta_conf$evaluation_strategy
  islands <- meta_conf$islands
  cv_folds <- meta_conf$cv_folds
  metacv_selection <- meta_conf$metacv_selection

  # If custom migration config is provided, sync islands count and topology from migration$topology
  if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    islands <- migration$topology$islands
    if (!is.null(migration$topology$type)) {
      migration_topology <- migration$topology$type
    }
  }

  # Validate island parameters early
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
    get_evaluator(ev)
  }
  evaluator_main <- island_evaluators[1]

  # Parse allowed_transformers
  all_trans <- names(evo_transformers)
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

  # Setup core environment & register CRAN-safe restoration
  env_state <- setup_core_env(threads = threads, max_clustering_size = max_clustering_size, seed = seed)
  on.exit(restore_core_env(env_state), add = TRUE)

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

  shared_full <- data.table::as.data.table(data)

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
      island_shared_folds <- lapply(seq_len(islands), function(j) list())
      for (f in seq_len(cv_folds)) {
        train_indices <- which(fold_ids != f)
        train_indices <- sample(train_indices)
        split_indices <- split(train_indices, cut(seq_along(train_indices), islands, labels = FALSE))
        for (j in seq_len(islands)) {
          island_shared_folds[[j]][[f]] <- list(
            train = data.table::as.data.table(data[split_indices[[j]], ]),
            val = data.table::as.data.table(data[fold_ids == f, ])
          )
        }
      }
    } else {
      shared_folds <- list()
      for (f in seq_len(cv_folds)) {
        shared_folds[[f]] <- list(
          train = data.table::as.data.table(data[fold_ids != f, ]),
          val = data.table::as.data.table(data[fold_ids == f, ])
        )
      }
    }
  } else if (evaluation_strategy == "metacv") {
    metacv_parts <- build_metacv_partitions(data, islands, cv_strategy, time_col, group_col, verbose)
    fold_ids <- metacv_parts$fold_ids
    island_shared_splits <- metacv_parts$island_shared_splits
    shared_splits <- NULL
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
      for (j in seq_len(islands)) {
        if (per_island_validation) {
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
        val = global_val_dt,
        holdout = global_holdout_dt
      )
    }
  }

  # Multi-fidelity screening setup
  mf_warmup_gens <- ceiling(generations * mf_warmup_frac)
  mf_shared_splits <- NULL
  mf_island_shared_splits <- NULL
  mf_shared_folds <- NULL
  mf_island_shared_folds <- NULL
  mf_shared_full <- NULL

  if (multi_fidelity) {
    if (!is.numeric(mf_sample_frac) || length(mf_sample_frac) != 1 ||
        mf_sample_frac <= 0 || mf_sample_frac >= 1) {
      stop("'mf_sample_frac' must be a numeric value strictly between 0 and 1.")
    }
    if (!is.numeric(mf_warmup_frac) || length(mf_warmup_frac) != 1 ||
        mf_warmup_frac <= 0 || mf_warmup_frac > 1) {
      stop("'mf_warmup_frac' must be a numeric value in (0, 1].")
    }

    .mf_subsample <- function(part) {
      if (is.null(part) || nrow(part) == 0L) return(part)
      n_take <- max(30L, ceiling(nrow(part) * mf_sample_frac))
      n_take <- min(n_take, nrow(part))
      part[seq_len(n_take), ]
    }
    .mf_subsample_fold <- function(fl) {
      if (is.null(fl)) return(fl)
      list(train = .mf_subsample(fl$train), val = fl$val)
    }

    if (!is.null(shared_splits)) {
      mf_shared_splits <- list(
        train = .mf_subsample(shared_splits$train),
        val = shared_splits$val,
        holdout = shared_splits$holdout
      )
    }
    if (!is.null(island_shared_splits)) {
      mf_island_shared_splits <- lapply(island_shared_splits, .mf_subsample_fold)
    }
    if (!is.null(shared_folds)) {
      mf_shared_folds <- lapply(shared_folds, .mf_subsample_fold)
    }
    if (!is.null(island_shared_folds)) {
      mf_island_shared_folds <- lapply(island_shared_folds, function(isl) lapply(isl, .mf_subsample_fold))
    }
    mf_shared_full <- if (!is.null(shared_full) && nrow(shared_full) > 30L) .mf_subsample(shared_full) else NULL
    if (verbose) {
      message(sprintf(
        "  Multi-fidelity: screening on %.0f%% of training rows (min 30 rows floor) for the first %d generation(s), then full-fidelity promotion.",
        100 * mf_sample_frac, mf_warmup_gens
      ))
    }
  }

  # Fitness and state caches
  fitness_cache <- new.env(hash = TRUE, parent = emptyenv())
  state_cache <- new.env(hash = TRUE, parent = emptyenv())

  viewer <- NULL
  evolution_log <- NULL

  topo_obj <- if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    migration$topology
  } else {
    switch(migration_topology,
      "grid" = topology_grid(islands),
      "torus" = topology_torus(islands),
      "hypercube" = topology_hypercube(islands),
      "tiered" = topology_tiered(islands),
      "hfc" = topology_tiered(islands),
      "complete" = topology_complete(islands),
      "feature_distance" = topology_feature_distance(islands),
      topology_ring(islands)
    )
  }

  tiers_count <- if (!is.null(topo_obj$tiers)) topo_obj$tiers else 3L
  adj_list_payload <- if (!is.null(topo_obj) && !is.null(topo_obj$adj_list)) topo_obj$adj_list else NULL

  policy_str <- "push_uniform"
  policy_thresh_val <- "min_peer"
  payload_str <- "full_individual"
  if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    if (!is.null(migration$payload)) payload_str <- migration$payload
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
        tiers = tiers_count, adj_list = adj_list_payload,
        task = task, evaluator = evaluator, evaluation_strategy = evaluation_strategy,
        row_split_islands = row_split_islands, per_island_validation = per_island_validation,
        target_col = target_col, migration_interval = migration_interval,
        migration_topology = migration_topology, migration_policy = policy_str,
        min_fitness_threshold = policy_thresh_val,
        migration_payload = payload_str, migration_temperature = migration_temperature,
        pull_stagnation_threshold = pull_stagnation_threshold,
        early_stopping_generations = early_stopping_generations,
        numeric_cols = numeric_cols, categorical_cols = categorical_cols,
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

  # Generation 0: Baseline individual
  baseline_ind <- create_individual(
    genes = list(),
    numeric_cols = numeric_cols,
    categorical_cols = categorical_cols,
    datetime_cols = datetime_cols,
    all_numeric_cols = numeric_cols,
    all_categorical_cols = categorical_cols,
    all_datetime_cols = datetime_cols
  )
  if (evaluation_strategy != "metacv") {
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

    recipe_str <- individual_to_recipe_string(baseline_ind)
    cache_key <- digest::digest(paste0(evaluator_main, "::", recipe_str), algo = "md5", serialize = FALSE)
    assign(cache_key, baseline_ind, envir = fitness_cache)
  } else {
    if (verbose) {
      message("\n--- Generation 0 (Baseline) ---")
      message(sprintf("  Individual 1: %s (evaluating across %d MetaCV folds...)",
                      individual_to_recipe_string(baseline_ind), islands))
    }
  }

  if (islands > 1) {
    island_fitness_caches <- lapply(seq_len(islands), function(x) new.env(hash = TRUE, parent = emptyenv()))
    island_state_caches <- lapply(seq_len(islands), function(x) new.env(hash = TRUE, parent = emptyenv()))
    for (j in seq_len(islands)) {
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

    if (evaluation_strategy == "metacv") {
      baseline_ind <- stitch_metacv_baseline_oof(
        island_baseline_inds = island_baseline_inds,
        fold_ids = fold_ids,
        data = data,
        task = task,
        num_class = num_class,
        evaluator_main = evaluator_main,
        fitness_cache = fitness_cache,
        baseline_ind = baseline_ind,
        islands = islands,
        verbose = verbose
      )
    }
  }

  if (record) {
    res_sample <- tryCatch(
      {
        apply_individual(baseline_ind, utils::head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = utils::head(shared_full, 5))
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
        lapply(seq_len(islands), function(j) {
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

  # Initialize initial population(s)
  if (islands == 1) {
    pop <- initialize_population(
      pop_size, numeric_cols, categorical_cols,
      datetime_cols = datetime_cols, initial_genes = 2,
      task = task, importances = baseline_ind$importances,
      allowed_transformers = allowed_transformers,
      mask_temp_factor = mask_temp_factor
    )
    pop[[1]] <- baseline_ind
    pop_init <- pop
    pop_list_init <- NULL
    global_best_fitness_init <- baseline_ind$fitness
    global_best_individual_init <- baseline_ind
    best_ind_source_init <- "Island 1"
  } else {
    pop_list <- initialize_island_populations(
      islands = islands, pop_size = pop_size,
      numeric_cols = numeric_cols, categorical_cols = categorical_cols,
      datetime_cols = datetime_cols, task = task,
      baseline_ind = baseline_ind, island_baseline_inds = island_baseline_inds,
      allowed_transformers = allowed_transformers,
      mask_temp_factor = mask_temp_factor, seed = seed
    )
    pop_init <- NULL
    pop_list_init <- pop_list
    if (row_split_islands || evaluation_strategy == "metacv") {
      best_idx <- which.max(vapply(island_baseline_inds, function(ind) ind$fitness, numeric(1)))
      global_best_fitness_init <- island_baseline_inds[[best_idx]]$fitness
      global_best_individual_init <- island_baseline_inds[[best_idx]]
      best_ind_source_init <- paste0("Island ", best_idx)
    } else {
      global_best_fitness_init <- baseline_ind$fitness
      global_best_individual_init <- baseline_ind
      best_ind_source_init <- "Island 1"
    }
  }

  # Run evolutionary search loop
  evo_res <- run_evolution_loop(
    islands = islands, pop_init = pop_init, pop_list_init = pop_list_init,
    generations = generations, pop_size = pop_size, data = data,
    target_col = target_col, task = task, cv_folds = cv_folds,
    evaluation_strategy = evaluation_strategy, split_ids_val = split_ids_val,
    shared_splits = shared_splits, island_shared_splits = island_shared_splits,
    evaluator = evaluator_main, island_evaluators = island_evaluators,
    fold_ids = fold_ids, shared_folds = shared_folds,
    island_shared_folds = island_shared_folds, shared_full = shared_full,
    state_cache = state_cache, island_state_caches = island_state_caches,
    fitness_cache = fitness_cache, island_fitness_caches = island_fitness_caches,
    threads = threads, verbose = verbose, metric = metric,
    complexity_penalty = complexity_penalty, complexity_mode = complexity_mode,
    complexity_floor = complexity_floor, complexity_target = complexity_target,
    island_baseline_inds = island_baseline_inds, baseline_ind = baseline_ind,
    cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
    multi_fidelity = multi_fidelity, mf_warmup_frac = mf_warmup_frac,
    mf_shared_splits = mf_shared_splits,
    mf_island_shared_splits = mf_island_shared_splits,
    mf_shared_folds = mf_shared_folds,
    mf_island_shared_folds = mf_island_shared_folds,
    mf_shared_full = mf_shared_full, record = record, viewer = viewer,
    evolution_log = evolution_log,
    early_stopping_generations = early_stopping_generations,
    dynamic_population = dynamic_population,
    dynamic_population_growth_rate = dynamic_population_growth_rate,
    dynamic_population_decay_rate = dynamic_population_decay_rate,
    crossover_type = crossover_type, allowed_transformers = allowed_transformers,
    raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob,
    migration = migration, migration_interval = migration_interval,
    migration_rate = migration_rate, gene_migration_prob = gene_migration_prob, migration_topology = migration_topology,
    migration_temperature = migration_temperature,
    pull_stagnation_threshold = pull_stagnation_threshold,
    row_split_islands = row_split_islands,
    per_island_validation = per_island_validation,
    get_island_transformers = get_island_transformers,
    global_best_fitness_init = global_best_fitness_init,
    global_best_individual_init = global_best_individual_init,
    best_ind_source_init = best_ind_source_init, ...
  )

  pop <- evo_res$pop
  pop_list <- evo_res$pop_list
  best_ind <- evo_res$best_ind
  global_best_fitness <- evo_res$global_best_fitness
  fitness_history <- evo_res$fitness_history
  island_best_fitness <- evo_res$island_best_fitness
  island_best_individual <- evo_res$island_best_individual
  historical_best_genes <- evo_res$historical_best_genes
  best_ind_source <- evo_res$best_ind_source
  evolution_log <- evo_res$evolution_log

  # Row-split tournament
  if (row_split_islands) {
    if (per_island_validation) {
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
          allow_prune = TRUE,
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
  metacv_island_oof_preds <- NULL
  ensemble_oof_fitness <- NULL

  if (evaluation_strategy == "metacv") {
    oof_res <- stitch_metacv_oof_predictions(
      island_best_individual = island_best_individual,
      fold_ids = fold_ids, data = data, target_col = target_col,
      task = task, metric = metric, num_class = num_class, classes = classes
    )
    metacv_island_oof_preds <- oof_res$metacv_island_oof_preds
    ensemble_oof_fitness <- oof_res$ensemble_oof_fitness

    champ_res <- select_metacv_champion(
      island_best_individual = island_best_individual,
      metacv_selection = metacv_selection,
      island_baseline_inds = island_baseline_inds,
      baseline_ind = baseline_ind, data = data, target_col = target_col,
      task = task, islands = islands, fold_ids = fold_ids,
      island_shared_splits = island_shared_splits, shared_full = shared_full,
      state_cache = state_cache, threads = threads, metric = metric,
      verbose = verbose, island_evaluators = island_evaluators,
      global_best_fitness = global_best_fitness,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      metacv_island_oof_preds = metacv_island_oof_preds,
      ensemble_oof_fitness = ensemble_oof_fitness, ...
    )

    best_ind <- champ_res$best_ind
    winner_idx <- champ_res$winner_idx
    best_ind_source <- champ_res$best_ind_source
    oof_preds <- champ_res$oof_preds
    tournament_fitness <- champ_res$tournament_fitness
    candidates <- champ_res$candidates
    island_best_individual <- champ_res$island_best_individual
    island_best_fitness <- champ_res$island_best_fitness
  } else if (evaluation_strategy == "cv" && !is.null(best_ind$val_preds)) {
    oof_preds <- best_ind$val_preds
  }

  if (record && (row_split_islands || evaluation_strategy == "metacv")) {
    cands_list <- if (exists("candidates") && !is.null(candidates)) candidates else island_best_individual
    tourn_fit <- if (exists("tournament_fitness") && !is.null(tournament_fitness)) tournament_fitness else vapply(cands_list, function(ind) ind$fitness, numeric(1))
    win_idx <- if (exists("winner_idx") && !is.null(winner_idx)) winner_idx else which.max(tourn_fit)

    tournament_data <- list(
      candidates = lapply(seq_len(islands), function(j) {
        cand <- cands_list[[j]]
        list(
          island = j,
          local_fitness = if (exists("island_best_fitness") && length(island_best_fitness) >= j) island_best_fitness[j] else cand$fitness,
          global_fitness = tourn_fit[j],
          recipe = individual_to_recipe_string(cand),
          n_genes = length(cand$genes)
        )
      }),
      winner_island = win_idx,
      winner_fitness = best_ind$fitness
    )
    evolution_log$tournament <- tournament_data
    viewer$send(list(type = "tournament", data = tournament_data))
  }

  # Final feature pooling
  if (model_all_final_genes) {
    pool_res <- pool_final_genes(
      pop = pop, best_ind = best_ind, data = data, target_col = target_col,
      task = task, cv_folds = cv_folds,
      evaluation_strategy = evaluation_strategy,
      split_ids_val = split_ids_val, shared_splits = shared_splits,
      evaluator_main = evaluator_main, fold_ids = fold_ids,
      island_shared_splits = island_shared_splits, shared_folds = shared_folds,
      shared_full = shared_full, state_cache = state_cache,
      threads = threads, metric = metric, verbose = verbose,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_ind$fitness, islands = islands,
      record = record, evolution_log = evolution_log, viewer = viewer,
      best_ind_source = best_ind_source, oof_preds = oof_preds, ...
    )
    best_ind <- pool_res$best_ind
    best_ind_source <- pool_res$best_ind_source
    oof_preds <- pool_res$oof_preds
    evolution_log <- pool_res$evolution_log
  }

  # Historical feature pooling
  if (model_all_historical_genes) {
    hist_res <- pool_historical_genes(
      historical_best_genes = historical_best_genes, best_ind = best_ind,
      data = data, target_col = target_col, task = task, cv_folds = cv_folds,
      evaluation_strategy = evaluation_strategy, split_ids_val = split_ids_val,
      shared_splits = shared_splits, evaluator_main = evaluator_main,
      fold_ids = fold_ids, island_shared_splits = island_shared_splits,
      shared_folds = shared_folds, shared_full = shared_full,
      state_cache = state_cache, threads = threads, metric = metric,
      verbose = verbose, complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_ind$fitness, islands = islands,
      record = record, evolution_log = evolution_log, viewer = viewer,
      best_ind_source = best_ind_source, oof_preds = oof_preds, ...
    )
    best_ind <- hist_res$best_ind
    best_ind_source <- hist_res$best_ind_source
    oof_preds <- hist_res$oof_preds
    evolution_log <- hist_res$evolution_log
  }

  # Holdout evaluation for split strategy
  if (evaluation_strategy == "split" && ("holdout" %in% split_ids_val || !is.null(shared_splits$holdout))) {
    best_eval_curr <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main
    best_ind <- evaluate_holdout_fitness(
      best_ind, data, split_ids_val, shared_splits,
      target_col, task, best_eval_curr, threads, state_cache,
      classes, num_class,
      metric = metric, verbose = verbose, seed = seed, ...
    )
  }

  if (verbose) {
    if (evaluation_strategy == "metacv") {
      message(sprintf("\nEvolution Complete. Winning Island %d Validation Score: %.4f (MetaCV Stitched OOF Score: %.4f)",
                      winner_idx, best_ind$fitness, ensemble_oof_fitness))
    } else if (!is.null(best_ind$raw_fitness) && !is.na(best_ind$raw_fitness)) {
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
  best_iteration <- best_ind$best_iteration
  train_size <- best_ind$train_size
  res_full <- apply_individual(best_ind, data.table::copy(shared_full), NULL, target_col, state_cache = state_cache)
  best_ind <- res_full$ind
  if (!is.null(best_iteration)) {
    best_ind$best_iteration <- best_iteration
  }
  if (!is.null(train_size)) {
    best_ind$train_size <- train_size
  }

  gene_cols <- if (length(best_ind$genes) > 0) vapply(best_ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(best_ind$numeric_cols, best_ind$categorical_cols, best_ind$datetime_cols, gene_cols)

  x_full <- .sanitize_feature_matrix(res_full$train[, features, with = FALSE])
  y_full <- res_full$train[[target_col]]
  if (task == "multiclass") {
    y_full <- as.integer(factor(y_full, levels = classes)) - 1
  }

  best_evaluator <- if (!is.null(best_ind$evaluator)) best_ind$evaluator else evaluator_main
  final_model_args <- list(...)
  target_iters <- NULL
  if (!is.null(best_ind$best_iteration) && is.numeric(best_ind$best_iteration) &&
      is.finite(best_ind$best_iteration) && best_ind$best_iteration > 0) {
    total_data_size <- nrow(x_full)
    training_size <- if (!is.null(best_ind$train_size) && is.numeric(best_ind$train_size) && best_ind$train_size > 0) {
      as.numeric(best_ind$train_size)
    } else if (evaluation_strategy == "cv" && !is.null(cv_folds) && cv_folds > 1) {
      round(total_data_size * (cv_folds - 1) / cv_folds)
    } else if (!is.null(split_ratio) && length(split_ratio) >= 2) {
      round(total_data_size * split_ratio[1] / sum(split_ratio[1:2]))
    } else {
      total_data_size
    }

    target_iters <- if (is_tree_evaluator(best_evaluator)) {
      scale_evaluator_iterations(best_evaluator, best_ind$best_iteration, training_size, total_data_size)
    } else {
      as.integer(best_ind$best_iteration)
    }
    if (verbose) {
      scale_factor <- if (training_size > 0 && total_data_size > training_size) total_data_size / training_size else 1.0
      if (is_tree_evaluator(best_evaluator) && scale_factor > 1.0) {
        message(sprintf("  Using best validation iterations scaled by data size (%.2fx: %d -> %d) for final model",
                        scale_factor, best_ind$best_iteration, target_iters))
      } else {
        iter_label <- if (unwrap_evaluator(best_evaluator) == "realmlp") "epochs" else "iterations/rounds"
        message(sprintf("  Using best validation %s for final model: %d", iter_label, target_iters))
      }
    }
    final_model_args <- apply_iteration_target(final_model_args, target_iters, best_evaluator)
  }

  final_evaluator <- unwrap_evaluator(best_evaluator)
  if (!is.null(best_params) && length(best_params) > 0) {
    final_model_args <- utils::modifyList(final_model_args, as.list(best_params))
  }

  res_model <- do.call(train_model, c(
    list(
      x_train = x_full, y_train = y_full,
      task = task, evaluator = final_evaluator,
      threads = threads, num_class = num_class, metric = metric,
      verbose = verbose, best_params = best_params, seed = seed
    ),
    final_model_args
  ))
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
      x_conf <- .sanitize_feature_matrix(res_conf$train[, conf_features, with = FALSE])
      preds_conf <- tryCatch(
        get_evaluator(best_evaluator)$predict_func(best_model, x_conf, task = task),
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

  effective_fitness <- best_ind$fitness

  search_gap <- NULL
  if (!is.null(best_ind$holdout_fitness) && !is.na(best_ind$holdout_fitness)) {
    val_score <- if (!is.null(best_ind$raw_fitness) && is.finite(best_ind$raw_fitness)) {
      best_ind$raw_fitness
    } else {
      effective_fitness
    }
    if (!is.null(val_score) && is.finite(val_score)) {
      search_gap <- best_ind$holdout_fitness - val_score
      if (verbose) {
        message(sprintf(
          "Search gap (holdout - validation): %+.4f. Small gaps indicate the search did not overfit the selection folds.",
          search_gap
        ))
      }
    }
  }

  if (record) {
    res_sample <- tryCatch(
      {
        apply_individual(best_ind, utils::head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = utils::head(shared_full, 5))
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

    final_hr <- calculate_headroom(effective_fitness, baseline_ind$fitness, task)
    final_data <- list(
      raw_fitness = if (!is.null(best_ind$raw_fitness)) best_ind$raw_fitness else effective_fitness,
      penalty = if (!is.null(best_ind$penalty)) best_ind$penalty else 0.0,
      best_fitness = effective_fitness,
      baseline_fitness = baseline_ind$fitness,
      improvement = final_hr$improvement,
      headroom_closed = if (!is.null(final_hr$headroom_closed)) final_hr$headroom_closed else 0.0,
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

  recipe_hr <- calculate_headroom(best_ind$fitness, baseline_ind$fitness, task)

  metacv_hr <- if (evaluation_strategy == "metacv" && !is.null(ensemble_oof_fitness)) {
    calculate_headroom(ensemble_oof_fitness, baseline_ind$fitness, task)
  } else NULL

  has_islands <- (islands > 1 && length(island_baseline_inds) == islands)
  island_baselines_vec <- if (has_islands) {
    vapply(island_baseline_inds, function(x) x$fitness, numeric(1))
  } else NULL

  island_best_fit <- if (has_islands && exists("island_best_individual")) {
    vapply(island_best_individual, function(ind) ind$fitness, numeric(1))
  } else NULL

  island_hr <- if (has_islands && !is.null(island_best_fit)) {
    calculate_headroom(island_best_fit, island_baselines_vec, task)
  } else NULL

  island_bests_list <- if (exists("island_best_individual") && !is.null(island_best_individual)) island_best_individual else list(best_ind)
  for (idx in seq_along(island_bests_list)) {
    if (is.null(island_bests_list[[idx]]$extra_args)) {
      island_bests_list[[idx]]$extra_args <- extra_args_top
    }
    if (is.null(island_bests_list[[idx]]$threads)) {
      island_bests_list[[idx]]$threads <- threads
    }
  }
  if (is.null(best_ind$extra_args)) best_ind$extra_args <- extra_args_top
  if (is.null(best_ind$threads)) best_ind$threads <- threads

  res_obj <- list(
    best_individual = best_ind,
    history = pop,
    fitness_history = fitness_history,
    task = task,
    best_model = best_model,
    best_iteration = if (!is.null(target_iters)) target_iters else if (!is.null(best_ind$best_iteration)) best_ind$best_iteration else NULL,
    evaluator = best_evaluator,
    target_col = target_col,
    classes = classes,
    metric = metric,
    baseline_fitness = baseline_ind$fitness,
    improvement = recipe_hr$improvement,
    headroom_closed = recipe_hr$headroom_closed,
    single_best_improvement = recipe_hr$improvement,
    single_best_headroom_closed = recipe_hr$headroom_closed,
    ensemble_improvement = if (!is.null(metacv_hr)) metacv_hr$improvement else NULL,
    ensemble_headroom_closed = if (!is.null(metacv_hr)) metacv_hr$headroom_closed else NULL,
    island_baselines = island_baselines_vec,
    island_best_fitness = island_best_fit,
    island_improvements = if (!is.null(island_hr)) island_hr$improvement else NULL,
    island_headroom_closed = if (!is.null(island_hr)) island_hr$headroom_closed else NULL,
    holdout_fitness = if (!is.null(best_ind$holdout_fitness)) best_ind$holdout_fitness else NULL,
    search_gap = search_gap,
    cv_strategy = cv_strategy,
    evaluation_strategy = evaluation_strategy,
    metacv_selection = if (evaluation_strategy == "metacv") metacv_selection else NULL,
    fold_ids = if (evaluation_strategy %in% c("cv", "metacv")) fold_ids else NULL,
    split_ids = if (evaluation_strategy == "split" && !is.null(split_ids_val)) split_ids_val else NULL,
    oof_preds = oof_preds,
    metacv_island_oof_preds = if (evaluation_strategy == "metacv") metacv_island_oof_preds else NULL,
    metacv_oof_fitness = if (evaluation_strategy == "metacv") ensemble_oof_fitness else NULL,
    island_bests = island_bests_list,
    threads = threads,
    extra_args = extra_args_top,
    evolution_log = if (record) evolution_log else NULL,
    alignment_cache = new.env(hash = TRUE, parent = emptyenv())
  )

  class(res_obj) <- "evo_recipe"
  res_obj
}
