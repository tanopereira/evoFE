# Evolutionary Search Engine & Generational Stepping

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

#' Single generation step for a single island
#'
#' @noRd
step_generation_single <- function(g, pop, data, target_col, task, cv_folds, evaluation_strategy,
                                   split_ids_val, shared_splits, evaluator,
                                   fold_ids, shared_folds, shared_full, state_cache,
                                   fitness_cache, threads, verbose, running_best_fitness,
                                   metric, complexity_penalty, complexity_mode,
                                   complexity_floor, complexity_target, baseline_ind,
                                   cv_strategy, time_col, group_col, multi_fidelity,
                                   mf_warmup_gens, mf_shared_splits, mf_shared_folds, mf_shared_full,
                                   record, viewer, evolution_log, global_best_fitness,
                                   generations_without_improvement, early_stopping_generations,
                                   generations, pop_size, dynamic_population,
                                   dynamic_population_growth_rate, dynamic_population_decay_rate,
                                   current_pop_size, crossover_type, allowed_transformers,
                                   raw_toggle_prob, recalculate_mask_prob, historical_best_genes, ...) {
  if (verbose) {
    if (global_best_fitness > -Inf) {
      message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
    } else {
      message(sprintf("\n--- Generation %d / %d ---", g, generations))
    }
    for (i in seq_along(pop)) {
      fit_str <- if (is.na(pop[[i]]$fitness)) "Unevaluated" else sprintf("%.4f", pop[[i]]$fitness)
      message(sprintf("  Individual %d (%s): %s", i, fit_str, individual_to_recipe_string(pop[[i]])))
    }
  }

  if (record) {
    viewer$send(list(type = "status", data = list(island = 1, status = "evaluating", generation = g)))
  }

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

  fitness_vals <- sapply(pop, function(ind) ind$fitness)
  pop <- pop[order(fitness_vals, decreasing = TRUE)]

  historical_best_genes <- c(historical_best_genes, pop[[1]]$genes)

  best_fitness <- pop[[1]]$fitness
  if (verbose) message(sprintf("  Gen %d Best Fitness: %.4f", g, best_fitness))
  if (verbose) message(sprintf("  Gen %d Best Recipe: %s", g, individual_to_recipe_string(pop[[1]])))

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

    res_sample <- tryCatch(
      {
        apply_individual(pop[[1]], utils::head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = utils::head(shared_full, 5))
    )
    best_dt <- res_sample$train
    best_list <- lapply(names(best_dt), function(col) {
      val <- best_dt[[col]]
      if (is.numeric(val)) round(val, 4) else as.character(val)
    })
    names(best_list) <- names(best_dt)

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
          population = lapply(utils::head(pop, 5), function(ind) {
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
    if (is.null(evolution_log)) evolution_log <- list(generations = list())
    evolution_log$generations[[g]] <- gen_snapshot
    viewer$send(list(type = "generation", data = gen_snapshot))
  }

  stop_early <- FALSE
  if (!is.null(early_stopping_generations) && generations_without_improvement >= early_stopping_generations) {
    message(sprintf("  Early stopping triggered after %d generations without improvement.", early_stopping_generations))
    stop_early <- TRUE
  }

  if (g == generations || stop_early) {
    return(list(
      pop = pop,
      running_best_fitness = running_best_fitness,
      global_best_fitness = global_best_fitness,
      generations_without_improvement = generations_without_improvement,
      best_fitness = best_fitness,
      current_pop_size = current_pop_size,
      historical_best_genes = historical_best_genes,
      evolution_log = evolution_log,
      stop_early = stop_early
    ))
  }

  num_survivors <- min(length(pop), max(2, floor(length(pop) / 2)))
  survivors <- pop[1:num_survivors]

  tested_gene_outputs <- unique(unlist(lapply(pop, function(ind) {
    if (length(ind$genes) == 0) return(character(0))
    vapply(ind$genes, function(g) g$output_col, character(1))
  })))

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

  global_importances_vec <- if (length(global_importances) > 0) sapply(global_importances, mean) else numeric(0)

  stagnation_ratio <- if (!is.null(early_stopping_generations) && early_stopping_generations > 0) {
    min(1, generations_without_improvement / early_stopping_generations)
  } else {
    0
  }
  adaptive_mutation_rate <- 0.3 + 0.4 * stagnation_ratio
  temperature <- 0.1 + 0.9 * stagnation_ratio

  target_pop_size <- pop_size
  if (dynamic_population) {
    if (generations_without_improvement > 0) {
      current_pop_size <- max(current_pop_size + 1, floor(current_pop_size * dynamic_population_growth_rate))
    } else {
      current_pop_size <- max(pop_size, floor(current_pop_size * dynamic_population_decay_rate))
    }
    target_pop_size <- min(current_pop_size, pop_size * 5L)
  }

  next_gen <- list()
  next_gen[[1]] <- survivors[[1]]

  while (length(next_gen) < target_pop_size) {
    idx <- length(next_gen) + 1
    is_expansion <- idx > pop_size

    if (is_expansion) {
      p <- tournament_select(pop, k = 3)
      child <- mutate(p, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
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
        child <- mutate(child, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
      }
    } else {
      p <- tournament_select(pop, k = 3)
      child <- mutate(p, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
    }

    attempts <- 0
    while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 15) {
      child <- mutate(child, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = if (is_expansion) 100.0 else temperature, task = task, tested_gene_outputs = tested_gene_outputs, allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob)
      attempts <- attempts + 1
    }

    next_gen <- c(next_gen, list(child))
  }

  list(
    pop = next_gen,
    running_best_fitness = running_best_fitness,
    global_best_fitness = global_best_fitness,
    generations_without_improvement = generations_without_improvement,
    best_fitness = best_fitness,
    current_pop_size = current_pop_size,
    historical_best_genes = historical_best_genes,
    evolution_log = evolution_log,
    stop_early = FALSE
  )
}

#' Single generation step across multiple islands
#'
#' @noRd
step_generation_multi <- function(g, pop_list, islands, data, target_col, task, cv_folds, evaluation_strategy,
                                  split_ids_val, shared_splits, island_shared_splits,
                                  island_evaluators, fold_ids, shared_folds, island_shared_folds,
                                  shared_full, state_cache, island_state_caches,
                                  fitness_cache, island_fitness_caches, threads, verbose,
                                  island_best_fitness, island_best_individual,
                                  island_gens_without_improvement, island_improved_by_migration,
                                  island_current_pop_size, migrated_genes_pool,
                                  metric, complexity_penalty, complexity_mode,
                                  complexity_floor, complexity_target, island_baseline_inds, baseline_ind,
                                  cv_strategy, time_col, group_col, multi_fidelity,
                                  mf_warmup_gens, mf_shared_splits, mf_island_shared_splits,
                                  mf_shared_folds, mf_island_shared_folds, mf_shared_full,
                                  record, viewer, evolution_log, global_best_fitness,
                                  global_best_individual, best_ind_source,
                                  generations_without_improvement, early_stopping_generations,
                                  generations, pop_size, dynamic_population,
                                  dynamic_population_growth_rate, dynamic_population_decay_rate,
                                  crossover_type, allowed_transformers, raw_toggle_prob,
                                  recalculate_mask_prob, historical_best_genes,
                                  migration, migration_interval, migration_rate,
                                  gene_migration_prob = 0.2,
                                  migration_topology, migration_temperature,
                                  pull_stagnation_threshold, row_split_islands,
                                  per_island_validation, tiers_count = NULL,
                                  policy_thresh_val = NULL, get_island_transformers, ...) {
  if (verbose) {
    if (global_best_fitness > -Inf) {
      message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
    } else {
      message(sprintf("\n--- Generation %d / %d ---", g, generations))
    }
  }

  for (j in seq_len(islands)) {
    if (verbose) {
      message(sprintf("\n  --- [Island %d] (Current Local Best Fitness: %.4f) ---", j, island_best_fitness[j]))
      for (i in seq_along(pop_list[[j]])) {
        fit_str <- if (is.na(pop_list[[j]][[i]]$fitness)) "Unevaluated" else sprintf("%.4f", pop_list[[j]][[i]]$fitness)
        message(sprintf("    [Island %d] Individual %d (%s): %s", j, i, fit_str, individual_to_recipe_string(pop_list[[j]][[i]])))
      }
    }

    if (record) {
      viewer$send(list(type = "status", data = list(island = j, status = "evaluating", generation = g)))
    }

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

    historical_best_genes <- c(historical_best_genes, pop_list[[j]][[1]]$genes)

    best_fitness_island <- pop_list[[j]][[1]]$fitness
    if (verbose) {
      message(sprintf("    [Island %d] Gen %d Best Fitness: %.4f", j, g, best_fitness_island))
      message(sprintf("    [Island %d] Gen %d Best Recipe: %s", j, g, individual_to_recipe_string(pop_list[[j]][[1]])))
    }

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

    if (best_fitness_island > global_best_fitness) {
      global_best_fitness <- best_fitness_island
      global_best_individual <- pop_list[[j]][[1]]
      best_ind_source <- paste0("Island ", j)
    }
  }

  if (record) {
    res_sample <- tryCatch(
      {
        apply_individual(global_best_individual, utils::head(shared_full, 5), NULL, NULL, state_cache = state_cache)
      },
      error = function(e) list(train = utils::head(shared_full, 5))
    )
    best_dt <- res_sample$train
    best_list <- lapply(names(best_dt), function(col) {
      val <- best_dt[[col]]
      if (is.numeric(val)) round(val, 4) else as.character(val)
    })
    names(best_list) <- names(best_dt)

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

    ideal_val <- if (task %in% c("classification", "multiclass")) 1.0 else 0.0

    best_island_idx <- 1L
    for (j in seq_len(islands)) {
      if (identical(pop_list[[j]][[1]], global_best_individual)) {
        best_island_idx <- j
        break
      }
    }
    best_island_baseline <- if (length(island_baseline_inds) >= best_island_idx && !is.null(island_baseline_inds[[best_island_idx]]$fitness)) {
      island_baseline_inds[[best_island_idx]]$fitness
    } else {
      baseline_ind$fitness
    }
    best_island_h_denom <- ideal_val - best_island_baseline
    best_island_headroom_closed <- if (abs(best_island_h_denom) < 1e-6 || is.na(global_best_fitness)) 0.0 else (global_best_fitness - best_island_baseline) / best_island_h_denom

    gen_snapshot <- list(
      generation = g,
      islands = lapply(seq_len(islands), function(j) {
        pop_j <- pop_list[[j]]
        base_j <- if (length(island_baseline_inds) >= j && !is.null(island_baseline_inds[[j]]$fitness)) island_baseline_inds[[j]]$fitness else baseline_ind$fitness
        h_denom <- ideal_val - base_j
        h_closed <- if (abs(h_denom) < 1e-6 || is.na(island_best_fitness[j])) 0.0 else (island_best_fitness[j] - base_j) / h_denom
        list(
          island = j,
          best_fitness = island_best_fitness[j],
          baseline_fitness = base_j,
          headroom_closed = h_closed,
          stagnation = island_gens_without_improvement[j],
          pop_size = length(pop_j),
          population = lapply(utils::head(pop_j, 5), function(ind) {
            list(fitness = ind$fitness, n_genes = length(ind$genes))
          }),
          all_fitness = sapply(pop_j, function(ind) ind$fitness)
        )
      }),
      global_best_fitness = global_best_fitness,
      global_best_island = best_island_idx,
      global_best_island_baseline = best_island_baseline,
      global_headroom_closed = best_island_headroom_closed,
      global_best_recipe = individual_to_recipe_string(global_best_individual),
      global_best_n_genes = length(global_best_individual$genes),
      global_best_importances = if (!is.null(global_best_individual$importances)) as.list(global_best_individual$importances) else list(),
      global_best_genes = serialized_genes,
      sample = best_list
    )
    if (is.null(evolution_log)) evolution_log <- list(generations = list())
    evolution_log$generations[[g]] <- gen_snapshot
    viewer$send(list(type = "generation", data = gen_snapshot))
  }

  stop_early <- FALSE
  if (!is.null(early_stopping_generations)) {
    if (per_island_validation || evaluation_strategy == "metacv") {
      if (all(island_gens_without_improvement >= early_stopping_generations)) {
        message(sprintf(
          "  Early stopping triggered: all %d islands stagnated for %d generations.",
          islands, early_stopping_generations
        ))
        stop_early <- TRUE
      }
    } else {
      if (generations_without_improvement >= early_stopping_generations) {
        message(sprintf("  Early stopping triggered after %d generations without global improvement.", early_stopping_generations))
        stop_early <- TRUE
      }
    }
  }

  if (g == generations || stop_early) {
    return(list(
      pop_list = pop_list,
      island_best_fitness = island_best_fitness,
      island_best_individual = island_best_individual,
      island_gens_without_improvement = island_gens_without_improvement,
      island_improved_by_migration = island_improved_by_migration,
      island_current_pop_size = island_current_pop_size,
      migrated_genes_pool = migrated_genes_pool,
      global_best_fitness = global_best_fitness,
      global_best_individual = global_best_individual,
      best_ind_source = best_ind_source,
      generations_without_improvement = generations_without_improvement,
      historical_best_genes = historical_best_genes,
      evolution_log = evolution_log,
      stop_early = stop_early
    ))
  }

  # --- MIGRATION PHASE ---
  if (g %% migration_interval == 0) {
    if (verbose) {
      message(sprintf("\n*** [Migration Phase] Triggering migration at Generation %d (Topology: %s) ***", g, migration_topology))
    }

    old_pop_list <- pop_list
    state <- list(
      pop_list = pop_list,
      island_best_fitness = island_best_fitness,
      island_gens_without_improvement = island_gens_without_improvement
    )

    use_rel_fits <- per_island_validation || evaluation_strategy == "metacv"
    effective_island_fits <- compute_effective_island_fitness(island_best_fitness, island_baseline_inds, task, use_rel_fits = use_rel_fits)

    if (verbose && use_rel_fits) {
      headroom_pcts <- paste(sprintf("Island %d: %+.1f%%", seq_len(islands), effective_island_fits * 100), collapse = ", ")
      message(sprintf("  [Migration Headroom Closed] %s", headroom_pcts))
    }

    migration_txs <- resolve_island_migration_transactions(
      state, migration = migration, migration_topology = migration_topology,
      migration_temperature = migration_temperature,
      pull_stagnation_threshold = pull_stagnation_threshold,
      islands = islands, tiers_count = tiers_count,
      policy_thresh_val = policy_thresh_val,
      effective_island_fits = effective_island_fits
    )

    mig_res <- execute_island_migration(
      migration_txs = migration_txs, pop_list = pop_list, old_pop_list = old_pop_list,
      island_best_fitness = island_best_fitness, island_best_individual = island_best_individual,
      island_gens_without_improvement = island_gens_without_improvement,
      island_improved_by_migration = island_improved_by_migration,
      migrated_genes_pool = migrated_genes_pool, migration = migration,
      migration_rate = migration_rate, gene_migration_prob = gene_migration_prob, migration_topology = migration_topology,
      island_evaluators = island_evaluators, row_split_islands = row_split_islands,
      per_island_validation = per_island_validation, evaluation_strategy = evaluation_strategy,
      data = data, target_col = target_col, task = task, cv_folds = cv_folds,
      split_ids_val = split_ids_val, island_shared_splits = island_shared_splits,
      shared_splits = shared_splits, fold_ids = fold_ids,
      island_shared_folds = island_shared_folds, shared_folds = shared_folds,
      shared_full = shared_full, island_state_caches = island_state_caches,
      state_cache = state_cache, island_fitness_caches = island_fitness_caches,
      fitness_cache = fitness_cache, threads = threads, verbose = verbose,
      metric = metric, complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode, complexity_floor = complexity_floor,
      complexity_target = complexity_target, island_baseline_inds = island_baseline_inds,
      baseline_ind = baseline_ind, g = g, evolution_log = evolution_log,
      viewer = viewer, record = record, effective_island_fits = effective_island_fits,
      use_rel_fits = use_rel_fits, global_best_fitness = global_best_fitness,
      global_best_individual = global_best_individual, best_ind_source = best_ind_source, ...
    )

    pop_list <- mig_res$pop_list
    island_best_fitness <- mig_res$island_best_fitness
    island_best_individual <- mig_res$island_best_individual
    island_gens_without_improvement <- mig_res$island_gens_without_improvement
    island_improved_by_migration <- mig_res$island_improved_by_migration
    migrated_genes_pool <- mig_res$migrated_genes_pool
    global_best_fitness <- mig_res$global_best_fitness
    global_best_individual <- mig_res$global_best_individual
    best_ind_source <- mig_res$best_ind_source
    evolution_log <- mig_res$evolution_log
  }

  # --- BREEDING PHASE ---
  for (j in seq_len(islands)) {
    pop <- pop_list[[j]]
    num_survivors <- min(length(pop), max(2, floor(length(pop) / 2)))
    survivors <- pop[1:num_survivors]

    tested_gene_outputs <- unique(unlist(lapply(pop, function(ind) {
      if (length(ind$genes) == 0) return(character(0))
      vapply(ind$genes, function(g) g$output_col, character(1))
    })))

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

    global_importances_vec <- if (length(global_importances) > 0) sapply(global_importances, mean) else numeric(0)

    stagnation_ratio <- if (!is.null(early_stopping_generations) && early_stopping_generations > 0) {
      min(1, island_gens_without_improvement[j] / early_stopping_generations)
    } else {
      0
    }
    adaptive_mutation_rate <- 0.3 + 0.4 * stagnation_ratio
    temperature <- 0.1 + 0.9 * stagnation_ratio

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

    cur_island_cache <- if (row_split_islands || evaluation_strategy == "metacv") island_fitness_caches[[j]] else fitness_cache
    cur_island_best_fit <- if (row_split_islands || evaluation_strategy == "metacv") island_best_fitness[j] else global_best_fitness

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

      attempts <- 0
      while (is_invalid_individual(child, next_gen, cur_island_cache, cur_island_best_fit, evaluator = island_evaluators[j]) && attempts < 15) {
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

  list(
    pop_list = pop_list,
    island_best_fitness = island_best_fitness,
    island_best_individual = island_best_individual,
    island_gens_without_improvement = island_gens_without_improvement,
    island_improved_by_migration = island_improved_by_migration,
    island_current_pop_size = island_current_pop_size,
    migrated_genes_pool = migrated_genes_pool,
    global_best_fitness = global_best_fitness,
    global_best_individual = global_best_individual,
    best_ind_source = best_ind_source,
    generations_without_improvement = generations_without_improvement,
    historical_best_genes = historical_best_genes,
    evolution_log = evolution_log,
    stop_early = FALSE
  )
}

#' Run full evolutionary generational loop
#'
#' @noRd
run_evolution_loop <- function(islands, pop_init, pop_list_init, generations,
                               pop_size, data, target_col, task, cv_folds, evaluation_strategy,
                               split_ids_val, shared_splits, island_shared_splits,
                               evaluator, island_evaluators, fold_ids, shared_folds, island_shared_folds,
                               shared_full, state_cache, island_state_caches,
                               fitness_cache, island_fitness_caches, threads, verbose,
                               metric, complexity_penalty, complexity_mode,
                               complexity_floor, complexity_target, island_baseline_inds, baseline_ind,
                               cv_strategy, time_col, group_col, multi_fidelity,
                               mf_warmup_frac, mf_shared_splits, mf_island_shared_splits,
                               mf_shared_folds, mf_island_shared_folds, mf_shared_full,
                               record, viewer, evolution_log,
                               early_stopping_generations, dynamic_population,
                               dynamic_population_growth_rate, dynamic_population_decay_rate,
                               crossover_type, allowed_transformers, raw_toggle_prob,
                               recalculate_mask_prob, migration, migration_interval,
                               migration_rate, gene_migration_prob = 0.2, migration_topology, migration_temperature,
                               pull_stagnation_threshold, row_split_islands,
                               per_island_validation, get_island_transformers,
                               global_best_fitness_init, global_best_individual_init,
                               best_ind_source_init, ...) {
  fitness_history <- numeric(generations)
  historical_best_genes <- list()
  mf_warmup_gens <- ceiling(generations * mf_warmup_frac)

  if (islands == 1) {
    pop <- pop_init
    running_best_fitness <- baseline_ind$fitness
    global_best_fitness <- global_best_fitness_init
    generations_without_improvement <- 0
    current_pop_size <- pop_size

    for (g in seq_len(generations)) {
      step_res <- step_generation_single(
        g = g, pop = pop, data = data, target_col = target_col, task = task,
        cv_folds = cv_folds, evaluation_strategy = evaluation_strategy,
        split_ids_val = split_ids_val, shared_splits = shared_splits, evaluator = evaluator,
        fold_ids = fold_ids, shared_folds = shared_folds, shared_full = shared_full,
        state_cache = state_cache, fitness_cache = fitness_cache, threads = threads,
        verbose = verbose, running_best_fitness = running_best_fitness,
        metric = metric, complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode, complexity_floor = complexity_floor,
        complexity_target = complexity_target, baseline_ind = baseline_ind,
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
        multi_fidelity = multi_fidelity, mf_warmup_gens = mf_warmup_gens,
        mf_shared_splits = mf_shared_splits, mf_shared_folds = mf_shared_folds,
        mf_shared_full = mf_shared_full, record = record, viewer = viewer,
        evolution_log = evolution_log, global_best_fitness = global_best_fitness,
        generations_without_improvement = generations_without_improvement,
        early_stopping_generations = early_stopping_generations,
        generations = generations, pop_size = pop_size,
        dynamic_population = dynamic_population,
        dynamic_population_growth_rate = dynamic_population_growth_rate,
        dynamic_population_decay_rate = dynamic_population_decay_rate,
        current_pop_size = current_pop_size, crossover_type = crossover_type,
        allowed_transformers = allowed_transformers, raw_toggle_prob = raw_toggle_prob,
        recalculate_mask_prob = recalculate_mask_prob,
        historical_best_genes = historical_best_genes, ...
      )

      pop <- step_res$pop
      running_best_fitness <- step_res$running_best_fitness
      global_best_fitness <- step_res$global_best_fitness
      generations_without_improvement <- step_res$generations_without_improvement
      fitness_history[g] <- step_res$best_fitness
      current_pop_size <- step_res$current_pop_size
      historical_best_genes <- step_res$historical_best_genes
      evolution_log <- step_res$evolution_log

      if (step_res$stop_early) {
        fitness_history <- fitness_history[1:g]
        break
      }
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
      mf_on = multi_fidelity && (length(fitness_history) <= mf_warmup_gens),
      lf_shared_splits = mf_shared_splits,
      lf_shared_folds = mf_shared_folds, lf_shared_full = mf_shared_full, ...
    )
    pop <- eval_res$pop
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]
    best_ind <- pop[[1]]

    list(
      pop = pop,
      pop_list = list(pop),
      best_ind = best_ind,
      global_best_fitness = global_best_fitness,
      fitness_history = fitness_history,
      island_best_fitness = best_ind$fitness,
      island_best_individual = list(best_ind),
      island_baseline_inds = island_baseline_inds,
      historical_best_genes = historical_best_genes,
      best_ind_source = "Island 1",
      evolution_log = evolution_log
    )
  } else {
    pop_list <- pop_list_init
    island_best_fitness <- vapply(island_baseline_inds, function(ind) ind$fitness, numeric(1))
    island_best_individual <- lapply(seq_len(islands), function(j) island_baseline_inds[[j]])
    island_gens_without_improvement <- rep(0, islands)
    island_improved_by_migration <- rep(FALSE, islands)
    island_current_pop_size <- rep(pop_size, islands)
    migrated_genes_pool <- lapply(seq_len(islands), function(x) list())

    global_best_fitness <- global_best_fitness_init
    global_best_individual <- global_best_individual_init
    best_ind_source <- best_ind_source_init
    generations_without_improvement <- 0

    for (g in seq_len(generations)) {
      step_res <- step_generation_multi(
        g = g, pop_list = pop_list, islands = islands, data = data,
        target_col = target_col, task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy, split_ids_val = split_ids_val,
        shared_splits = shared_splits, island_shared_splits = island_shared_splits,
        island_evaluators = island_evaluators, fold_ids = fold_ids,
        shared_folds = shared_folds, island_shared_folds = island_shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        island_state_caches = island_state_caches, fitness_cache = fitness_cache,
        island_fitness_caches = island_fitness_caches, threads = threads,
        verbose = verbose, island_best_fitness = island_best_fitness,
        island_best_individual = island_best_individual,
        island_gens_without_improvement = island_gens_without_improvement,
        island_improved_by_migration = island_improved_by_migration,
        island_current_pop_size = island_current_pop_size,
        migrated_genes_pool = migrated_genes_pool, metric = metric,
        complexity_penalty = complexity_penalty, complexity_mode = complexity_mode,
        complexity_floor = complexity_floor, complexity_target = complexity_target,
        island_baseline_inds = island_baseline_inds, baseline_ind = baseline_ind,
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
        multi_fidelity = multi_fidelity, mf_warmup_gens = mf_warmup_gens,
        mf_shared_splits = mf_shared_splits,
        mf_island_shared_splits = mf_island_shared_splits,
        mf_shared_folds = mf_shared_folds,
        mf_island_shared_folds = mf_island_shared_folds,
        mf_shared_full = mf_shared_full, record = record, viewer = viewer,
        evolution_log = evolution_log, global_best_fitness = global_best_fitness,
        global_best_individual = global_best_individual,
        best_ind_source = best_ind_source,
        generations_without_improvement = generations_without_improvement,
        early_stopping_generations = early_stopping_generations,
        generations = generations, pop_size = pop_size,
        dynamic_population = dynamic_population,
        dynamic_population_growth_rate = dynamic_population_growth_rate,
        dynamic_population_decay_rate = dynamic_population_decay_rate,
        crossover_type = crossover_type, allowed_transformers = allowed_transformers,
        raw_toggle_prob = raw_toggle_prob, recalculate_mask_prob = recalculate_mask_prob,
        historical_best_genes = historical_best_genes, migration = migration,
        migration_interval = migration_interval, migration_rate = migration_rate,
        migration_topology = migration_topology,
        migration_temperature = migration_temperature,
        pull_stagnation_threshold = pull_stagnation_threshold,
        row_split_islands = row_split_islands,
        per_island_validation = per_island_validation,
        get_island_transformers = get_island_transformers, ...
      )

      pop_list <- step_res$pop_list
      island_best_fitness <- step_res$island_best_fitness
      island_best_individual <- step_res$island_best_individual
      island_gens_without_improvement <- step_res$island_gens_without_improvement
      island_improved_by_migration <- step_res$island_improved_by_migration
      island_current_pop_size <- step_res$island_current_pop_size
      migrated_genes_pool <- step_res$migrated_genes_pool
      global_best_fitness <- step_res$global_best_fitness
      global_best_individual <- step_res$global_best_individual
      best_ind_source <- step_res$best_ind_source
      historical_best_genes <- step_res$historical_best_genes
      evolution_log <- step_res$evolution_log

      fitness_history[g] <- global_best_fitness
      if (g == 1 || (global_best_fitness > fitness_history[max(1, g - 1)])) {
        generations_without_improvement <- 0
      } else {
        generations_without_improvement <- generations_without_improvement + 1
      }

      if (step_res$stop_early) {
        fitness_history <- fitness_history[1:g]
        break
      }
    }

    # Final evaluation of individuals on all islands
    for (j in seq_len(islands)) {
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
        mf_on = multi_fidelity && (length(fitness_history) <= mf_warmup_gens),
        lf_shared_splits = if (row_split_islands || evaluation_strategy == "metacv") mf_island_shared_splits[[j]] else mf_shared_splits,
        lf_shared_folds = if (row_split_islands) mf_island_shared_folds[[j]] else mf_shared_folds,
        lf_shared_full = mf_shared_full, ...
      )
      pop_list[[j]] <- eval_res$pop
      fitness_vals <- sapply(pop_list[[j]], function(ind) ind$fitness)
      pop_list[[j]] <- pop_list[[j]][order(fitness_vals, decreasing = TRUE)]

      if (!is.null(pop_list[[j]][[1]]$fitness) && !is.na(pop_list[[j]][[1]]$fitness) &&
          (is.na(island_best_fitness[j]) || pop_list[[j]][[1]]$fitness > island_best_fitness[j])) {
        island_best_fitness[j] <- pop_list[[j]][[1]]$fitness
        pop_list[[j]][[1]]$evaluator <- island_evaluators[j]
        island_best_individual[[j]] <- pop_list[[j]][[1]]
      }

      if (!is.null(pop_list[[j]][[1]]$fitness) && !is.na(pop_list[[j]][[1]]$fitness) &&
          (is.na(global_best_fitness) || pop_list[[j]][[1]]$fitness > global_best_fitness)) {
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
            allow_prune = TRUE,
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
      best_j_idx <- which.max(sapply(island_best_individual, function(ind) ind$fitness))
      if (island_best_individual[[best_j_idx]]$fitness > best_ind$fitness) {
        best_ind <- island_best_individual[[best_j_idx]]
        global_best_fitness <- best_ind$fitness
      }
    }

    list(
      pop = pop,
      pop_list = pop_list,
      best_ind = best_ind,
      global_best_fitness = global_best_fitness,
      fitness_history = fitness_history,
      island_best_fitness = island_best_fitness,
      island_best_individual = island_best_individual,
      island_baseline_inds = island_baseline_inds,
      historical_best_genes = historical_best_genes,
      best_ind_source = best_ind_source,
      evolution_log = evolution_log
    )
  }
}
