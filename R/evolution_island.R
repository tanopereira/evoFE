# Island Population Initialization and Migration

#' Initialize island populations
#'
#' @param islands Number of islands
#' @param pop_size Population size per island
#' @param numeric_cols Character vector of numeric column names
#' @param categorical_cols Character vector of categorical column names
#' @param datetime_cols Character vector of datetime column names
#' @param task Task type ("classification", "multiclass", or "regression")
#' @param baseline_ind Baseline individual object
#' @param island_baseline_inds List of baseline individuals per island
#' @param allowed_transformers Allowed transformer list or character vector
#' @param mask_temp_factor Feature mask sampling temperature
#' @param seed Optional random seed
#' @return List of island populations
#' @noRd
initialize_island_populations <- function(islands, pop_size, numeric_cols, categorical_cols,
                                          datetime_cols = NULL, task = "classification",
                                          baseline_ind = NULL, island_baseline_inds = list(),
                                          allowed_transformers = "all", mask_temp_factor = 0.5,
                                          seed = NULL) {
  pop_list <- list()
  for (j in seq_len(islands)) {
    # Per-island sub-stream: island j's initial population depends only on
    # (seed, j), not on how many islands are being run.
    # INVARIANT: set.seed is called ONLY here in the island evolution lifecycle.
    if (!is.null(seed)) set.seed(seed + 1000L * j)
    trans_j <- if (is.list(allowed_transformers)) allowed_transformers[[j]] else allowed_transformers
    pop_list[[j]] <- initialize_population(
      pop_size, numeric_cols, categorical_cols,
      datetime_cols = datetime_cols,
      initial_genes = 2, task = task, importances = baseline_ind$importances,
      allowed_transformers = trans_j,
      mask_temp_factor = mask_temp_factor
    )
    if (length(island_baseline_inds) >= j && !is.null(island_baseline_inds[[j]])) {
      pop_list[[j]][[1]] <- island_baseline_inds[[j]]
    }
  }
  pop_list
}

#' Compute effective island fitness (normalized headroom closed)
#'
#' @param island_best_fitness Numeric vector of best fitnesses per island
#' @param island_baseline_inds List of baseline individuals per island
#' @param task Task type
#' @param use_rel_fits Logical indicating whether relative fitness is used
#' @return Numeric vector of effective island fitness values
#' @noRd
compute_effective_island_fitness <- function(island_best_fitness, island_baseline_inds, task, use_rel_fits = FALSE) {
  if (use_rel_fits) {
    ideal_mig <- if (task %in% c("classification", "multiclass")) 1.0 else 0.0
    island_baselines <- vapply(island_baseline_inds, function(x) {
      if (!is.null(x$fitness) && is.finite(x$fitness)) x$fitness else 0
    }, numeric(1))
    headroom <- ideal_mig - island_baselines
    headroom[abs(headroom) < 1e-6] <- 1e-6
    (island_best_fitness - island_baselines) / headroom
  } else {
    island_best_fitness
  }
}

#' Resolve island migration transactions
#'
#' @param state Current island search state
#' @param migration Optional evo_migration_config object
#' @param migration_topology Topology string
#' @param migration_temperature Temperature parameter
#' @param pull_stagnation_threshold Stagnation threshold for dual Gibbs pull
#' @param islands Number of islands
#' @param tiers_count Number of tiers for tiered topology
#' @param policy_thresh_val Threshold for tiered policy
#' @param effective_island_fits Effective fitness vector
#' @return List of migration transaction descriptors
#' @noRd
resolve_island_migration_transactions <- function(state, migration = NULL,
                                                  migration_topology = "ring",
                                                  migration_temperature = 1.0,
                                                  pull_stagnation_threshold = 3,
                                                  islands = 1L,
                                                  tiers_count = NULL,
                                                  policy_thresh_val = NULL,
                                                  effective_island_fits = NULL) {
  migration_txs <- list()
  island_gens_without_improvement <- state$island_gens_without_improvement
  if (is.null(effective_island_fits)) {
    effective_island_fits <- state$island_best_fitness
  }

  if (!is.null(migration) && inherits(migration, "evo_migration_config")) {
    migration_txs <- resolve_migration_transactions(migration$policy, migration$topology, state)
  } else if (migration_topology == "dual_gibbs_pull") {
    for (j in seq_len(islands)) {
      s_j <- island_gens_without_improvement[j]
      p_pull <- 1 / (1 + exp(-(s_j - pull_stagnation_threshold) / migration_temperature))
      if (stats::runif(1) < p_pull) {
        candidates <- setdiff(seq_len(islands), j)
        if (length(candidates) > 0) {
          donor_fits <- effective_island_fits[candidates]
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
    for (j in seq_len(islands)) {
      dest <- j
      if (migration_topology == "ring") {
        dest <- (j %% islands) + 1
      } else if (migration_topology == "random") {
        candidates <- setdiff(seq_len(islands), j)
        dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1)
      } else if (migration_topology == "gibbs_stagnation") {
        candidates <- setdiff(seq_len(islands), j)
        stags <- island_gens_without_improvement[candidates]
        max_s <- max(stags)
        logits <- (stags - max_s) / migration_temperature
        probs <- exp(logits) / sum(exp(logits))
        dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1, prob = probs)
      } else if (migration_topology == "gibbs_fitness") {
        candidates <- setdiff(seq_len(islands), j)
        fits <- effective_island_fits[candidates]
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
        if (length(candidates) == 0) candidates <- setdiff(seq_len(islands), j)
        dest <- if (length(candidates) == 1) candidates[1] else sample(candidates, 1)
      }

      migration_txs[[length(migration_txs) + 1]] <- list(from = j, to = dest, is_pull = FALSE)
    }
  }
  migration_txs
}

#' Execute island migration transactions
#'
#' @param migration_txs Migration transaction list
#' @param pop_list Current population list per island
#' @param old_pop_list Snapshot of population list before migration
#' @param island_best_fitness Best fitness per island
#' @param island_best_individual Best individual per island
#' @param island_gens_without_improvement Stagnation count per island
#' @param island_improved_by_migration Logical improvement tracker
#' @param migrated_genes_pool List of gene pools per island
#' @param migration Optional evo_migration_config object
#' @param migration_rate Migration rate
#' @param migration_topology Topology string
#' @param island_evaluators Vector of evaluator names per island
#' @param row_split_islands Logical indicating row-split islands
#' @param per_island_validation Logical indicating per-island validation
#' @param evaluation_strategy Evaluation strategy string
#' @param data Full dataset
#' @param target_col Target column name
#' @param task Task type
#' @param cv_folds CV folds count
#' @param split_ids_val Split IDs vector
#' @param island_shared_splits Island splits list
#' @param shared_splits Global shared splits list
#' @param fold_ids Fold IDs vector
#' @param island_shared_folds Island folds list
#' @param shared_folds Global shared folds list
#' @param shared_full Shared full data.table
#' @param island_state_caches Island state caches
#' @param state_cache Global state cache
#' @param island_fitness_caches Island fitness caches
#' @param fitness_cache Global fitness cache
#' @param threads Thread count
#' @param verbose Logical for verbose output
#' @param metric Metric
#' @param complexity_penalty Complexity penalty multiplier
#' @param complexity_mode Complexity mode string
#' @param complexity_floor Complexity floor
#' @param complexity_target Complexity target
#' @param island_baseline_inds List of island baselines
#' @param baseline_ind Global baseline individual
#' @param g Current generation number
#' @param evolution_log Optional evolution log list
#' @param viewer Optional viewer server object
#' @param record Logical indicating whether to log
#' @param effective_island_fits Effective fitness vector
#' @param use_rel_fits Logical indicating whether relative fitness is used
#' @param global_best_fitness Current global best fitness
#' @param global_best_individual Current global best individual
#' @param best_ind_source String indicating source island
#' @param ... Additional arguments passed to evaluate_pop
#' @return Updated island state list
#' @noRd
execute_island_migration <- function(migration_txs, pop_list, old_pop_list,
                                     island_best_fitness, island_best_individual,
                                     island_gens_without_improvement,
                                     island_improved_by_migration,
                                     migrated_genes_pool, migration = NULL,
                                     migration_rate = 1, migration_topology = "ring",
                                     island_evaluators = NULL,
                                     row_split_islands = FALSE,
                                     per_island_validation = FALSE,
                                     evaluation_strategy = "cv",
                                     data = NULL, target_col = NULL, task = "classification",
                                     cv_folds = 3L, split_ids_val = NULL,
                                     island_shared_splits = NULL, shared_splits = NULL,
                                     fold_ids = NULL, island_shared_folds = NULL,
                                     shared_folds = NULL, shared_full = NULL,
                                     island_state_caches = NULL, state_cache = NULL,
                                     island_fitness_caches = NULL, fitness_cache = NULL,
                                     threads = 2L, verbose = FALSE,
                                     metric = "default", complexity_penalty = 0,
                                     complexity_mode = "bic_dynamic", complexity_floor = 0.20,
                                     complexity_target = "all_features",
                                     island_baseline_inds = list(), baseline_ind = NULL,
                                     g = 1L, evolution_log = NULL, viewer = NULL,
                                     record = FALSE, effective_island_fits = NULL,
                                     use_rel_fits = FALSE,
                                     global_best_fitness = -Inf,
                                     global_best_individual = NULL,
                                     best_ind_source = "Island 1", ...) {
  islands <- length(pop_list)

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
        worst_start <- length(pop_list[[dest]]) - effective_rate + 1
        worst_end <- length(pop_list[[dest]])
        migrant_inds <- old_pop_list[[src]][1:effective_rate]

        data_differs <- row_split_islands || per_island_validation || evaluation_strategy == "metacv"
        eval_differs <- island_evaluators[src] != island_evaluators[dest]

        if (data_differs || eval_differs) {
          if (data_differs) {
            # Data partition differs: strip both fitness and transformer states to prevent cross-split leakage
            migrant_inds <- lapply(migrant_inds, strip_individual_state)
          } else {
            # Data partition is identical, only evaluator differs:
            # Feature transformations are data-dependent and model-agnostic; preserve states and only reset fitness
            migrant_inds <- lapply(migrant_inds, function(ind) {
              ind$fitness <- NA_real_
              ind$raw_fitness <- NA_real_
              ind$val_preds <- NULL
              ind$y_val <- NULL
              ind
            })
          }
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
            n_samples = nrow(data), island = dest,
            ind_indices = worst_start:worst_end, ...
          )
          migrant_inds <- eval_migrant$pop
        }

        # Replace the worst individuals of the target population
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
      best_ind_src <- old_pop_list[[src]][[1]]
      best_genes <- best_ind_src$genes
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
            if (!is.null(best_ind_src$importances) && col %in% names(best_ind_src$importances)) {
              as.numeric(best_ind_src$importances[[col]])
            } else {
              0.0
            }
          }, double(1))

          new_genes <- new_genes[order(gene_imps, decreasing = TRUE)]

          # Limit to top 20 most important new genes
          if (length(new_genes) > 20) {
            new_genes <- new_genes[1:20]
          }

          # Strip fitted states before adding to destination pool to prevent stale/leaked states
          for (k in seq_along(new_genes)) {
            new_genes[[k]]$state <- NULL
          }

          migrated_genes_pool[[dest]] <- c(migrated_genes_pool[[dest]], new_genes)
          n_injected <- length(new_genes)
        } else {
          n_injected <- 0L
        }

        if (length(migrated_genes_pool[[dest]]) > 20) {
          migrated_genes_pool[[dest]] <- utils::tail(migrated_genes_pool[[dest]], 20)
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
        imp_val <- if (!is.null(best_ind_src$importances) && col %in% names(best_ind_src$importances)) {
          as.numeric(best_ind_src$importances[[col]])
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
      feature_distance = feat_dist,
      donor_headroom = if (exists("effective_island_fits") && use_rel_fits) effective_island_fits[src] else NULL,
      dest_headroom = if (exists("effective_island_fits") && use_rel_fits) effective_island_fits[dest] else NULL
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
  for (k in seq_len(islands)) {
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

  list(
    pop_list = pop_list,
    island_best_fitness = island_best_fitness,
    island_best_individual = island_best_individual,
    island_gens_without_improvement = island_gens_without_improvement,
    island_improved_by_migration = island_improved_by_migration,
    migrated_genes_pool = migrated_genes_pool,
    global_best_fitness = global_best_fitness,
    global_best_individual = global_best_individual,
    best_ind_source = best_ind_source,
    evolution_log = evolution_log
  )
}
