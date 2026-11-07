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
                             threads = default_threads(), metric = "default", verbose = FALSE,
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
    if (!is.null(best_ind$best_params)) {
      super_ind$best_params <- best_ind$best_params
    }
    resolved_pool <- resolve_tuned_evaluator(best_eval_curr, super_ind$best_params, list(...))

    # 4. Evaluate the super-individual's fitness
    super_ind <- do.call(evaluate_fitness, c(
      list(
        super_ind, data = data, target_col = target_col,
        task = task,
        cv_folds = if (evaluation_strategy == "metacv") islands else cv_folds,
        evaluation_strategy = if (evaluation_strategy == "metacv") "cv" else evaluation_strategy,
        split_ids = split_ids_val,
        shared_splits = if (evaluation_strategy == "metacv") NULL else shared_splits,
        evaluator = resolved_pool$evaluator, fold_ids = fold_ids,
        shared_folds = if (evaluation_strategy == "metacv") island_shared_splits else shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_fitness,
        running_best_fitness = best_ind$fitness,
        n_samples = nrow(data)
      ),
      resolved_pool$extra_args
    ))

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
                                  threads = default_threads(), metric = "default", verbose = FALSE,
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
    if (!is.null(best_ind$best_params)) {
      super_ind_hist$best_params <- best_ind$best_params
    }
    resolved_hist <- resolve_tuned_evaluator(best_eval_curr, super_ind_hist$best_params, list(...))
    # Evaluate the historical super-individual's fitness
    super_ind_hist <- do.call(evaluate_fitness, c(
      list(
        super_ind_hist, data = data, target_col = target_col,
        task = task,
        cv_folds = if (evaluation_strategy == "metacv") islands else cv_folds,
        evaluation_strategy = if (evaluation_strategy == "metacv") "cv" else evaluation_strategy,
        split_ids = split_ids_val,
        shared_splits = if (evaluation_strategy == "metacv") NULL else shared_splits,
        evaluator = resolved_hist$evaluator, fold_ids = fold_ids,
        shared_folds = if (evaluation_strategy == "metacv") island_shared_splits else shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_fitness,
        running_best_fitness = best_ind$fitness,
        n_samples = nrow(data)
      ),
      resolved_hist$extra_args
    ))

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

#' Average two row-aligned OOF prediction vectors or matrices
#' @keywords internal
#' @noRd
.blend_oof_predictions <- function(p1, p2) {
  if (is.null(p1)) return(p2)
  if (is.null(p2)) return(p1)
  if (is.matrix(p1) && is.matrix(p2) && identical(dim(p1), dim(p2))) {
    both_fin <- is.finite(p1) & is.finite(p2)
    out <- p1
    out[both_fin] <- 0.5 * (p1[both_fin] + p2[both_fin])
    only2 <- !is.finite(p1) & is.finite(p2)
    out[only2] <- p2[only2]
    dimnames(out) <- dimnames(p1)
    return(out)
  }
  if (!is.matrix(p1) && !is.matrix(p2) && length(p1) == length(p2)) {
    both_fin <- is.finite(p1) & is.finite(p2)
    out <- p1
    out[both_fin] <- 0.5 * (p1[both_fin] + p2[both_fin])
    only2 <- !is.finite(p1) & is.finite(p2)
    out[only2] <- p2[only2]
    return(out)
  }
  p1
}

#' Multi-seed Top-K CV confirmation gate against selection-variance overfitting
#'
#' Re-evaluates up to \code{confirm_top_k} distinct top candidate recipes on an
#' independent K-fold partition and selects the recipe maximizing mean fitness
#' across both CV repeats while blending row-aligned 2x-OOF predictions.
#'
#' @keywords internal
#' @noRd
confirm_top_k_candidates <- function(confirm_top_k, best_ind, pop = NULL,
                                     island_best_individual = NULL,
                                     extra_candidates = NULL,
                                     data, target_col, task = "classification",
                                     cv_folds = 3L, cv_strategy = "random",
                                     time_col = NULL, group_col = NULL,
                                     evaluator_main = "lightgbm",
                                     shared_full = NULL, state_cache = NULL,
                                     threads = default_threads(),
                                     metric = "default", verbose = FALSE,
                                     complexity_penalty = 0,
                                     complexity_mode = "bic_dynamic",
                                     complexity_floor = 0.20,
                                     complexity_target = "all_features",
                                     baseline_fitness = NULL,
                                     sd_penalty = 0, seed = NULL,
                                     record = FALSE, evolution_log = NULL,
                                     viewer = NULL, best_ind_source = "Island 1",
                                     oof_preds = NULL, ...) {
  raw_pool <- c(
    list(best_ind),
    if (!is.null(extra_candidates)) extra_candidates else list(),
    if (!is.null(island_best_individual)) island_best_individual else list(),
    if (!is.null(pop)) pop else list()
  )
  valid_pool <- Filter(function(ind) {
    is.list(ind) && !is.null(ind$fitness) && length(ind$fitness) == 1L && is.finite(ind$fitness)
  }, raw_pool)

  if (length(valid_pool) == 0L) {
    return(list(
      best_ind = best_ind, best_ind_source = best_ind_source,
      oof_preds = oof_preds, evolution_log = evolution_log,
      confirm_fitness = NULL, mean_cv_fitness = NULL
    ))
  }

  ord <- order(vapply(valid_pool, function(ind) ind$fitness, numeric(1)), decreasing = TRUE)
  valid_pool <- valid_pool[ord]

  seen_keys <- character(0)
  finalists <- list()
  for (cand in valid_pool) {
    k_str <- individual_to_cache_key(cand)
    if (!(k_str %in% seen_keys)) {
      seen_keys <- c(seen_keys, k_str)
      finalists[[length(finalists) + 1L]] <- cand
      if (length(finalists) >= confirm_top_k) break
    }
  }

  k_actual <- length(finalists)
  if (verbose) {
    message(sprintf("\nRunning top-%d CV confirmation gate on fresh %d-fold split...", k_actual, cv_folds))
  }

  seed_confirm <- if (!is.null(seed) && is.numeric(seed) && is.finite(seed)) {
    as.integer(seed) + 10007L
  } else {
    NULL
  }

  build_split2 <- function() {
    .build_cv_folds(
      data, cv_folds, cv_strategy, time_col, group_col,
      target_col = target_col, task = task, shuffle_groups = TRUE
    )
  }
  fold_ids_confirm <- if (!is.null(seed_confirm)) {
    with_seed(seed_confirm, build_split2())
  } else {
    build_split2()
  }
  shared_folds_confirm <- .build_shared_folds(data, fold_ids_confirm)
  state_cache_confirm <- if (!is.null(state_cache)) new.env(hash = TRUE, parent = state_cache) else NULL
  fitness_cache_confirm <- new.env(hash = TRUE, parent = emptyenv())

  old_bfm <- getOption("evoFE.baseline_fold_metrics", NULL)
  options(evoFE.baseline_fold_metrics = NULL)
  on.exit(options(evoFE.baseline_fold_metrics = old_bfm), add = TRUE)

  evald_finalists <- vector("list", k_actual)
  fit_split1 <- numeric(k_actual)
  fit_split2 <- numeric(k_actual)
  mean_fits <- numeric(k_actual)
  dots_list <- list(...)

  for (j in seq_len(k_actual)) {
    cand <- finalists[[j]]
    fit_split1[j] <- cand$fitness
    cand_clean <- strip_individual_state(cand, keep_unsupervised = getOption("evoFE.global_unsupervised", TRUE))
    cand_eval <- if (!is.null(cand$evaluator)) cand$evaluator else evaluator_main
    resolved_c <- resolve_tuned_evaluator(cand_eval, cand$best_params, dots_list)
    extra_j <- resolved_c$extra_args
    if (!is.null(seed_confirm)) {
      extra_j$seed <- seed_confirm
    }

    cand_2 <- do.call(evaluate_fitness, c(
      list(
        cand_clean, data = data, target_col = target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = "cv",
        evaluator = resolved_c$evaluator,
        fold_ids = fold_ids_confirm,
        shared_folds = shared_folds_confirm,
        shared_full = shared_full,
        state_cache = state_cache_confirm,
        fitness_cache = fitness_cache_confirm,
        threads = threads, metric = metric, verbose = FALSE,
        allow_prune = FALSE,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        baseline_fitness = baseline_fitness,
        running_best_fitness = fit_split1[1L],
        n_samples = nrow(data),
        sd_penalty = sd_penalty
      ),
      extra_j
    ))
    evald_finalists[[j]] <- cand_2
    fit_split2[j] <- cand_2$fitness
    mean_fits[j] <- if (is.finite(fit_split1[j]) && is.finite(fit_split2[j])) {
      0.5 * (fit_split1[j] + fit_split2[j])
    } else {
      -Inf
    }
    if (verbose) {
      message(sprintf(
        "  [Candidate %d] Split-1: %.4f | Split-2: %.4f | Mean: %.4f  Recipe: %s",
        j, fit_split1[j], fit_split2[j], mean_fits[j], individual_to_recipe_string(cand)
      ))
    }
  }

  win_idx <- if (any(is.finite(mean_fits))) which.max(mean_fits) else 1L
  winner <- finalists[[win_idx]]
  conf_win <- evald_finalists[[win_idx]]

  winner$confirm_fitness <- fit_split2[win_idx]
  winner$mean_cv_fitness <- if (is.finite(mean_fits[win_idx])) mean_fits[win_idx] else fit_split1[win_idx]
  winner$val_preds <- .blend_oof_predictions(winner$val_preds, conf_win$val_preds)
  if (!is.null(winner$val_preds_trans) || !is.null(conf_win$val_preds_trans)) {
    winner$val_preds_trans <- .blend_oof_predictions(winner$val_preds_trans, conf_win$val_preds_trans)
  }
  if (!is.null(winner$best_iteration) && !is.null(conf_win$best_iteration) &&
      is.finite(winner$best_iteration) && is.finite(conf_win$best_iteration)) {
    winner$best_iteration <- as.integer(round(0.5 * (winner$best_iteration + conf_win$best_iteration)))
  }

  if (win_idx > 1L) {
    best_ind_source <- sprintf("Confirmed Rank %d", win_idx)
    if (verbose) {
      message(sprintf(
        "  Confirmation gate promoted Candidate %d (2-split mean %.4f vs Split-1 leader %.4f).",
        win_idx, winner$mean_cv_fitness, mean_fits[1L]
      ))
    }
  }
  oof_preds <- winner$val_preds

  if (record) {
    if (is.null(evolution_log)) evolution_log <- list()
    evolution_log$confirmation <- list(
      top_k = k_actual,
      winner_rank = win_idx,
      split1_fitness = fit_split1,
      split2_fitness = fit_split2,
      mean_fitness = mean_fits
    )
  }

  list(
    best_ind = winner,
    best_ind_source = best_ind_source,
    oof_preds = oof_preds,
    evolution_log = evolution_log,
    confirm_fitness = winner$confirm_fitness,
    mean_cv_fitness = winner$mean_cv_fitness
  )
}
