# MetaCV Partitioning, OOF Prediction Stitching, and Champion Selection

#' Validate and normalize MetaCV configuration
#'
#' @param evaluation_strategy Evaluation strategy string
#' @param islands Number of islands
#' @param cv_folds Number of cross-validation folds
#' @param migration Optional migration config object
#' @param row_split_islands Logical indicating row-split islands
#' @param per_island_validation Logical indicating per-island validation
#' @param metacv_selection Selection mode: "fitness", "tournament", or "headroom"
#' @param metacv_mode Deprecated alias for metacv_selection
#' @return Normalized configuration list
#' @noRd
validate_metacv_config <- function(evaluation_strategy, islands, cv_folds, migration = NULL,
                                   row_split_islands = FALSE, per_island_validation = FALSE,
                                   metacv_selection = c("fitness", "tournament", "headroom"),
                                   metacv_mode = NULL,
                                   missing_islands = missing(islands),
                                   missing_cv_folds = missing(cv_folds)) {
  if (!is.null(metacv_mode)) {
    warning("Argument 'metacv_mode' is deprecated; please use 'metacv_selection' instead.")
    if (metacv_mode == "tournament") {
      metacv_selection <- "tournament"
    } else if (metacv_mode == "headroom") {
      metacv_selection <- "headroom"
    } else {
      metacv_selection <- "fitness"
    }
  }
  metacv_selection <- match.arg(metacv_selection, c("fitness", "tournament", "headroom"))

  if (is.character(evaluation_strategy) && length(evaluation_strategy) == 1) {
    if (evaluation_strategy %in% c("meta_cv", "metacv")) {
      evaluation_strategy <- "metacv"
    }
  }

  if (evaluation_strategy == "metacv") {
    if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
      islands <- migration$topology$islands
      cv_folds <- islands
    } else if (missing_islands && !missing_cv_folds) {
      islands <- cv_folds
    } else if (!missing_islands && missing_cv_folds) {
      cv_folds <- islands
    } else if (missing_islands && missing_cv_folds) {
      islands <- 3L
      cv_folds <- 3L
    } else if (islands != cv_folds) {
      stop("'islands' must equal 'cv_folds' when evaluation_strategy is 'metacv'.")
    }
    if (islands < 2) {
      stop("evaluation_strategy = 'metacv' requires at least 2 islands (got 1).")
    }
    if (row_split_islands) {
      stop("row_split_islands is not supported with evaluation_strategy = 'metacv'. metacv automatically partitions folds across islands.")
    }
    if (per_island_validation) {
      stop("per_island_validation = TRUE is only supported with evaluation_strategy = 'split'.")
    }
  }

  list(
    evaluation_strategy = evaluation_strategy,
    islands = as.integer(islands),
    cv_folds = as.integer(cv_folds),
    metacv_selection = metacv_selection
  )
}

#' Build MetaCV fold partitions across islands
#'
#' @param data Full dataset data.frame/data.table
#' @param islands Number of islands (equal to cv_folds)
#' @param cv_strategy Strategy for fold splitting ("random", "time", "group")
#' @param time_col Time column name
#' @param group_col Group column name
#' @param verbose Logical indicating whether to print partition details
#' @return List with fold_ids and island_shared_splits
#' @noRd
build_metacv_partitions <- function(data, islands, cv_strategy = "random", time_col = NULL,
                                    group_col = NULL, verbose = FALSE) {
  fold_ids <- .build_cv_folds(data, islands, cv_strategy, time_col, group_col)
  island_shared_splits <- lapply(seq_len(islands), function(j) {
    tr_idx <- which(fold_ids != j)
    va_idx <- which(fold_ids == j)
    tr_dt <- data.table::as.data.table(data[tr_idx, ])
    va_dt <- data.table::as.data.table(data[va_idx, ])
    data.table::setattr(tr_dt, ".row_id", tr_idx)
    data.table::setattr(va_dt, ".row_id", va_idx)
    list(train = tr_dt, val = va_dt)
  })
  if (verbose) {
    message(sprintf("  MetaCV partitions -> %d folds mapped across %d islands", islands, islands))
    for (j in seq_len(islands)) {
      message(sprintf("    Island %d -> Train: %d rows (Folds -%d), Val: %d rows (Fold %d)",
                      j, nrow(island_shared_splits[[j]]$train), j,
                      nrow(island_shared_splits[[j]]$val), j))
    }
  }
  list(fold_ids = fold_ids, island_shared_splits = island_shared_splits)
}

#' Stitch out-of-fold predictions for MetaCV baseline
#'
#' @param island_baseline_inds List of baseline individuals per island
#' @param fold_ids Vector of fold IDs
#' @param data Full dataset
#' @param task "classification", "multiclass", or "regression"
#' @param num_class Number of classes for multiclass
#' @param evaluator_main Primary evaluator name
#' @param fitness_cache Cache environment for fitness values
#' @param baseline_ind Initial baseline individual object
#' @param islands Number of islands
#' @param verbose Logical for verbose messages
#' @return Updated baseline individual object
#' @noRd
stitch_metacv_baseline_oof <- function(island_baseline_inds, fold_ids, data, task, num_class = NULL,
                                       evaluator_main = "lightgbm", fitness_cache = NULL,
                                       baseline_ind = NULL, islands = length(island_baseline_inds),
                                       verbose = FALSE) {
  oof_base_preds <- if (task == "multiclass") {
    matrix(NA_real_, nrow = nrow(data), ncol = num_class)
  } else {
    rep(NA_real_, nrow(data))
  }
  for (j in seq_len(islands)) {
    v_idx <- which(fold_ids == j)
    vp <- island_baseline_inds[[j]]$val_preds
    if (!is.null(vp)) {
      if (task == "multiclass") {
        if (!is.matrix(vp)) vp <- matrix(vp, ncol = num_class, byrow = FALSE)
        oof_base_preds[v_idx, ] <- vp
      } else {
        oof_base_preds[v_idx] <- vp
      }
    }
  }
  island_base_fits <- vapply(island_baseline_inds, function(ind) ind$fitness, numeric(1))
  finite_base_fits <- island_base_fits[is.finite(island_base_fits)]
  mean_base_fit <- if (length(finite_base_fits) > 0) mean(finite_base_fits) else -Inf
  baseline_ind$fitness <- mean_base_fit
  baseline_ind$raw_fitness <- mean_base_fit
  baseline_ind$val_preds <- oof_base_preds
  baseline_ind$evaluator <- evaluator_main

  if (!is.null(fitness_cache)) {
    recipe_str <- individual_to_recipe_string(baseline_ind)
    cache_key <- digest::digest(paste0(evaluator_main, "::", recipe_str), algo = "md5", serialize = FALSE)
    assign(cache_key, baseline_ind, envir = fitness_cache)
  }

  if (verbose) {
    message(sprintf("  Tested Individual 1 (MetaCV Baseline) -> Fitness: %.4f (mean over %d folds)", baseline_ind$fitness, islands))
  }

  baseline_ind
}

#' Stitch Out-Of-Fold predictions from island bests
#'
#' @param island_best_individual List of best individuals per island
#' @param fold_ids Vector of fold IDs
#' @param data Full dataset
#' @param target_col Target column name
#' @param task "classification", "multiclass", or "regression"
#' @param metric Metric name or function
#' @param num_class Number of classes for multiclass
#' @param classes Class levels for multiclass
#' @return List with metacv_island_oof_preds and ensemble_oof_fitness
#' @noRd
stitch_metacv_oof_predictions <- function(island_best_individual, fold_ids, data, target_col,
                                          task, metric, num_class = NULL, classes = NULL,
                                          threads = NULL) {
  islands <- length(island_best_individual)
  if (task == "multiclass") {
    stitched_preds <- matrix(NA_real_, nrow = nrow(data), ncol = num_class)
    for (j in seq_len(islands)) {
      ind_j <- island_best_individual[[j]]
      val_idx <- which(fold_ids == j)
      if (!is.null(ind_j$val_preds)) {
        vp <- ind_j$val_preds
        if (!is.matrix(vp)) vp <- matrix(vp, ncol = num_class, byrow = FALSE)
        stitched_preds[val_idx, ] <- vp
      }
    }
  } else {
    stitched_preds <- rep(NA_real_, nrow(data))
    for (j in seq_len(islands)) {
      ind_j <- island_best_individual[[j]]
      val_idx <- which(fold_ids == j)
      if (!is.null(ind_j$val_preds)) {
        stitched_preds[val_idx] <- ind_j$val_preds
      }
    }
  }

  y_eval_oof <- if (task == "multiclass") {
    as.integer(factor(data[[target_col]], levels = classes)) - 1
  } else {
    data[[target_col]]
  }
  ensemble_oof_fitness <- if (task == "multiclass") {
    compute_metric(y_eval_oof, stitched_preds, task, metric, num_class, threads = threads)
  } else {
    compute_metric(y_eval_oof, stitched_preds, task, metric, threads = threads)
  }

  list(
    metacv_island_oof_preds = stitched_preds,
    ensemble_oof_fitness = ensemble_oof_fitness
  )
}

#' Select winning recipe after MetaCV evolution
#'
#' @param island_best_individual List of best individuals per island
#' @param metacv_selection Selection mode: "fitness", "tournament", or "headroom"
#' @param island_baseline_inds List of baseline individuals per island
#' @param baseline_ind Global baseline individual
#' @param data Full dataset
#' @param target_col Target column name
#' @param task "classification", "multiclass", or "regression"
#' @param islands Number of islands
#' @param fold_ids Vector of fold IDs
#' @param island_shared_splits List of train/val splits per island
#' @param shared_full Full dataset data.table
#' @param state_cache State cache environment
#' @param threads Thread count
#' @param metric Metric name or function
#' @param verbose Logical indicating whether to print progress
#' @param island_evaluators Vector of evaluator names per island
#' @param global_best_fitness Current best global fitness
#' @param complexity_penalty Complexity penalty factor
#' @param complexity_mode Complexity mode
#' @param complexity_floor Complexity floor
#' @param complexity_target Complexity target
#' @param metacv_island_oof_preds Pre-computed stitched OOF predictions
#' @param ensemble_oof_fitness Pre-computed ensemble OOF fitness
#' @param ... Additional arguments forwarded to evaluate_fitness
#' @return List containing selection results
#' @noRd
select_metacv_champion <- function(island_best_individual, metacv_selection,
                                   island_baseline_inds, baseline_ind, data, target_col, task,
                                   islands, fold_ids, island_shared_splits, shared_full,
                                   state_cache, threads, metric, verbose = FALSE,
                                   island_evaluators = NULL, global_best_fitness = -Inf,
                                   complexity_penalty = 0, complexity_mode = "bic_dynamic",
                                   complexity_floor = 0.20, complexity_target = "all_features",
                                   metacv_island_oof_preds = NULL, ensemble_oof_fitness = NULL,
                                   ...) {
  if (metacv_selection %in% c("fitness", "headroom")) {
    # FAST SELECTION: Skip the K^2 CV tournament.
    oof_preds <- metacv_island_oof_preds
    island_best_fitness <- vapply(island_best_individual, function(ind) ind$fitness, numeric(1))

    if (metacv_selection == "fitness") {
      winner_idx <- which.max(island_best_fitness)
      if (length(winner_idx) == 0 || is.na(winner_idx)) winner_idx <- 1L
    } else {
      ideal_metric <- if (task %in% c("classification", "multiclass")) 1.0 else 0.0
      island_headrooms <- vapply(seq_len(islands), function(j) {
        b_fit <- if (!is.null(island_baseline_inds[[j]])) island_baseline_inds[[j]]$fitness else baseline_ind$fitness
        denom <- ideal_metric - b_fit
        if (abs(denom) < 1e-6) 0.0 else (island_best_fitness[j] - b_fit) / denom
      }, numeric(1))
      winner_idx <- which.max(island_headrooms)
      if (length(winner_idx) == 0 || is.na(winner_idx)) winner_idx <- which.max(island_best_fitness)
    }
    best_ind <- island_best_individual[[winner_idx]]
    best_ind_source <- paste0("Island ", winner_idx)

    if (verbose) {
      sel_desc <- if (metacv_selection == "fitness") "validation fitness" else "headroom closed"
      message(sprintf("\nFinalizing MetaCV: selected champion recipe from Island %d by %s (zero CV tournament overhead)...",
                      winner_idx, sel_desc))
      message(sprintf("  Winning Recipe Fitness: %.4f (Baseline: %.4f | Stitched OOF Fitness: %.4f)",
                      best_ind$fitness, baseline_ind$fitness, ensemble_oof_fitness))
    }
    tournament_fitness <- island_best_fitness
    candidates <- island_best_individual
  } else {
    # TOURNAMENT MODE: evaluate each island's best candidate with full CV to select the global champion
    if (verbose) {
      message(sprintf("\nRunning MetaCV tournament: evaluating full CV fitness for best individual from each of %d islands...", islands))
    }

    # Deduplicate candidate recipes to avoid re-running identical full CV evaluations
    cand_recipes <- vapply(island_best_individual, individual_to_recipe_string, character(1))
    cand_evals <- vapply(seq_len(islands), function(j) {
      ind <- island_best_individual[[j]]
      if (!is.null(ind$evaluator)) ind$evaluator else island_evaluators[j]
    }, character(1))
    cand_keys <- paste0(cand_evals, "::", cand_recipes)
    cv_eval_cache <- list()
    candidates <- vector("list", islands)

    for (j in seq_len(islands)) {
      cache_key <- cand_keys[j]
      if (!is.null(cv_eval_cache[[cache_key]])) {
        candidates[[j]] <- cv_eval_cache[[cache_key]]
        if (verbose) {
          message(sprintf("  [Island %d] Full CV fitness: %.4f (cached)  Recipe: %s",
                          j, candidates[[j]]$fitness, individual_to_recipe_string(candidates[[j]])))
        }
      } else {
        ind <- island_best_individual[[j]]
        ind <- strip_individual_state(ind)
        cand_eval <- cand_evals[j]
        ind <- evaluate_fitness(
          ind, data, target_col,
          task = task, cv_folds = islands,
          evaluation_strategy = "cv",
          split_ids = NULL, shared_splits = NULL,
          evaluator = cand_eval, fold_ids = fold_ids,
          shared_folds = island_shared_splits,
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
        cv_eval_cache[[cache_key]] <- ind
        candidates[[j]] <- ind
        if (verbose) {
          message(sprintf("  [Island %d] Full CV fitness: %.4f  Recipe: %s",
                          j, ind$fitness, individual_to_recipe_string(ind)))
        }
      }
    }
    island_best_individual <- candidates
    island_best_fitness <- vapply(candidates, function(ind) ind$fitness, numeric(1))
    tournament_fitness <- island_best_fitness
    winner_idx <- which.max(tournament_fitness)
    best_ind <- candidates[[winner_idx]]
    best_ind_source <- paste0("Island ", winner_idx)
    oof_preds <- best_ind$val_preds
    if (verbose) {
      message(sprintf("  MetaCV Tournament winner: Island %d (fitness %.4f)", winner_idx, best_ind$fitness))
    }
  }

  list(
    best_ind = best_ind,
    winner_idx = winner_idx,
    best_ind_source = best_ind_source,
    oof_preds = oof_preds,
    tournament_fitness = tournament_fitness,
    candidates = candidates,
    island_best_individual = island_best_individual,
    island_best_fitness = island_best_fitness
  )
}
