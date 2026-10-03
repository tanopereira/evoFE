# Feature Pooling Operators

#' Pool and evaluate all unique genes from the final population
#'
#' @param pop Final evaluated population list
#' @param best_ind Current best individual
#' @param data Full training dataset
#' @param target_col Target column name
#' @param task Task type ("classification", "multiclass", "regression")
#' @param cv_folds Number of CV folds
#' @param evaluation_strategy Evaluation strategy
#' @param split_ids_val Split IDs vector
#' @param shared_splits Shared splits list
#' @param evaluator_main Primary evaluator name
#' @param fold_ids Fold IDs vector
#' @param island_shared_splits Island splits list
#' @param shared_folds Shared folds list
#' @param shared_full Shared full data.table
#' @param state_cache State cache environment
#' @param threads Number of threads
#' @param metric Metric name or function
#' @param verbose Logical for verbose messages
#' @param complexity_penalty Complexity penalty multiplier
#' @param complexity_mode Complexity mode
#' @param complexity_floor Complexity floor
#' @param complexity_target Complexity target
#' @param baseline_fitness Baseline fitness value
#' @param islands Number of islands
#' @param record Logical indicating whether to log
#' @param evolution_log Optional evolution log list
#' @param viewer Optional viewer server object
#' @param best_ind_source Source string of current best
#' @param oof_preds Current OOF predictions
#' @param ... Additional arguments passed to evaluate_fitness
#' @return List with updated best_ind, best_ind_source, adopted_pooled, oof_preds, evolution_log
#' @noRd
pool_final_genes <- function(pop, best_ind, data, target_col, task = "classification",
                             cv_folds = 3L, evaluation_strategy = "cv",
                             split_ids_val = NULL, shared_splits = NULL,
                             evaluator_main = "lightgbm", fold_ids = NULL,
                             island_shared_splits = NULL, shared_folds = NULL,
                             shared_full = NULL, state_cache = NULL,
                             threads = 2L, metric = "default", verbose = FALSE,
                             complexity_penalty = 0, complexity_mode = "bic_dynamic",
                             complexity_floor = 0.20, complexity_target = "all_features",
                             baseline_fitness = NULL, islands = 1L,
                             record = FALSE, evolution_log = NULL, viewer = NULL,
                             best_ind_source = "Island 1", oof_preds = NULL, ...) {
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
      gene$state <- NULL
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
      task = task,
      cv_folds = if (evaluation_strategy == "metacv") islands else cv_folds,
      evaluation_strategy = if (evaluation_strategy == "metacv") "cv" else evaluation_strategy,
      split_ids = split_ids_val,
      shared_splits = if (evaluation_strategy == "metacv") NULL else shared_splits,
      evaluator = best_eval_curr, fold_ids = fold_ids,
      shared_folds = if (evaluation_strategy == "metacv") island_shared_splits else shared_folds,
      shared_full = shared_full, state_cache = state_cache,
      threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_fitness,
      running_best_fitness = best_ind$fitness,
      n_samples = nrow(data), ...
    )

    if (is.null(super_ind$best_params) && !is.null(best_ind$best_params)) {
      super_ind$best_params <- best_ind$best_params
    }
    if (is.null(super_ind$best_iteration) && !is.null(best_ind$best_iteration)) {
      super_ind$best_iteration <- best_ind$best_iteration
    }
    if (is.null(super_ind$train_size) && !is.null(best_ind$train_size)) {
      super_ind$train_size <- best_ind$train_size
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
      oof_preds <- best_ind$val_preds
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
    if (is.null(evolution_log)) evolution_log <- list()
    evolution_log$pooled <- list(
      n_genes = length(deduped_genes),
      fitness = if (!is.null(super_ind)) super_ind$fitness else best_ind$fitness,
      adopted = adopted_pooled
    )
    if (!is.null(viewer)) {
      viewer$send(list(type = "pooled", data = evolution_log$pooled))
    }
  }

  list(
    best_ind = best_ind,
    best_ind_source = best_ind_source,
    adopted_pooled = adopted_pooled,
    oof_preds = oof_preds,
    evolution_log = evolution_log
  )
}

#' Pool and evaluate all unique genes evolved across all generations
#'
#' @param historical_best_genes List of historical best genes across generations
#' @param best_ind Current best individual
#' @param data Full training dataset
#' @param target_col Target column name
#' @param task Task type ("classification", "multiclass", "regression")
#' @param cv_folds Number of CV folds
#' @param evaluation_strategy Evaluation strategy
#' @param split_ids_val Split IDs vector
#' @param shared_splits Shared splits list
#' @param evaluator_main Primary evaluator name
#' @param fold_ids Fold IDs vector
#' @param island_shared_splits Island splits list
#' @param shared_folds Shared folds list
#' @param shared_full Shared full data.table
#' @param state_cache State cache environment
#' @param threads Number of threads
#' @param metric Metric name or function
#' @param verbose Logical for verbose messages
#' @param complexity_penalty Complexity penalty multiplier
#' @param complexity_mode Complexity mode
#' @param complexity_floor Complexity floor
#' @param complexity_target Complexity target
#' @param baseline_fitness Baseline fitness value
#' @param islands Number of islands
#' @param record Logical indicating whether to log
#' @param evolution_log Optional evolution log list
#' @param viewer Optional viewer server object
#' @param best_ind_source Source string of current best
#' @param oof_preds Current OOF predictions
#' @param ... Additional arguments passed to evaluate_fitness
#' @return List with updated best_ind, best_ind_source, adopted_hist, oof_preds, evolution_log
#' @noRd
pool_historical_genes <- function(historical_best_genes, best_ind, data, target_col, task = "classification",
                                  cv_folds = 3L, evaluation_strategy = "cv",
                                  split_ids_val = NULL, shared_splits = NULL,
                                  evaluator_main = "lightgbm", fold_ids = NULL,
                                  island_shared_splits = NULL, shared_folds = NULL,
                                  shared_full = NULL, state_cache = NULL,
                                  threads = 2L, metric = "default", verbose = FALSE,
                                  complexity_penalty = 0, complexity_mode = "bic_dynamic",
                                  complexity_floor = 0.20, complexity_target = "all_features",
                                  baseline_fitness = NULL, islands = 1L,
                                  record = FALSE, evolution_log = NULL, viewer = NULL,
                                  best_ind_source = "Island 1", oof_preds = NULL, ...) {
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
        gene$state <- NULL
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
      task = task,
      cv_folds = if (evaluation_strategy == "metacv") islands else cv_folds,
      evaluation_strategy = if (evaluation_strategy == "metacv") "cv" else evaluation_strategy,
      split_ids = split_ids_val,
      shared_splits = if (evaluation_strategy == "metacv") NULL else shared_splits,
      evaluator = best_eval_curr, fold_ids = fold_ids,
      shared_folds = if (evaluation_strategy == "metacv") island_shared_splits else shared_folds,
      shared_full = shared_full, state_cache = state_cache,
      threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_fitness,
      running_best_fitness = best_ind$fitness,
      n_samples = nrow(data), ...
    )

    if (is.null(super_ind_hist$best_params) && !is.null(best_ind$best_params)) {
      super_ind_hist$best_params <- best_ind$best_params
    }
    if (is.null(super_ind_hist$best_iteration) && !is.null(best_ind$best_iteration)) {
      super_ind_hist$best_iteration <- best_ind$best_iteration
    }
    if (is.null(super_ind_hist$train_size) && !is.null(best_ind$train_size)) {
      super_ind_hist$train_size <- best_ind$train_size
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
      oof_preds <- best_ind$val_preds
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
    if (is.null(evolution_log)) evolution_log <- list()
    evolution_log$historical <- list(
      n_genes = length(deduped_historical_genes),
      fitness = if (!is.null(super_ind_hist)) super_ind_hist$fitness else best_ind$fitness,
      adopted = adopted_hist
    )
    if (!is.null(viewer)) {
      viewer$send(list(type = "historical", data = evolution_log$historical))
    }
  }

  list(
    best_ind = best_ind,
    best_ind_source = best_ind_source,
    adopted_hist = adopted_hist,
    oof_preds = oof_preds,
    evolution_log = evolution_log
  )
}
