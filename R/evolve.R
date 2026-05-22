truncate_cols <- function(cols, max_show = 10) {
  if (length(cols) <= max_show) {
    return(paste(cols, collapse = ", "))
  }
  paste0(paste(cols[1:max_show], collapse = ", "), ", ... (+ ", length(cols) - max_show, " more)")
}

#' Run evolutionary feature engineering
#'
#' @param data A data.frame or data.table
#' @param target_col Name of the target column
#' @param task "classification" or "regression"
#' @param generations Number of generations (max iterations)
#' @param pop_size Population size
#' @param cv_folds Number of cross-validation folds
#' @param early_stopping_rounds Stop if fitness doesn't improve for this many generations
#' @param evaluator The ML model to use ("lightgbm" or "xgboost")
#' @param crossover_type Crossover type: "both" (default, 50% random / 50% union), "random", or "union"
#' @param verbose Logical. If TRUE, prints progress.
#' @export
evolve_features <- function(data, target_col, task = "classification", 
                            generations = 10, pop_size = 10, cv_folds = 3, 
                            early_stopping_rounds = 3, evaluator = "lightgbm",
                            dynamic_population = TRUE, crossover_type = "both", verbose = TRUE) {
  # Prevent macOS OpenMP thread collisions between data.table, lightgbm, and other libraries
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::omp_set_num_threads(1)
    RhpcBLASctl::blas_set_num_threads(1)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::setDTthreads(1)
  }

  if (!target_col %in% names(data)) {
    stop(sprintf("Target column '%s' not found in the dataset.", target_col))
  }
  
  original_cols <- setdiff(names(data), target_col)
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  numeric_cols <- setdiff(numeric_cols, target_col)
  categorical_cols <- setdiff(original_cols, numeric_cols)
  
  if (verbose) {
    message("Starting Evolutionary Feature Engineering...")
    message(sprintf("  Task: %s", task))
    message(sprintf("  Evaluator: %s", evaluator))
    message(sprintf("  Generations: %d, Population Size: %d, CV Folds: %d", generations, pop_size, cv_folds))
    message(sprintf("  Original Numeric columns: %s", truncate_cols(numeric_cols)))
    message(sprintf("  Original Categorical columns: %s", truncate_cols(categorical_cols)))
  }
  
  pop <- initialize_population(pop_size, numeric_cols, categorical_cols, initial_genes = 2)
  
  if (verbose) {
    message("\n[Gen 0] Initialized Population:")
    for (i in seq_along(pop)) {
      message(sprintf("  Individual %d: %s", i, individual_to_recipe_string(pop[[i]])))
    }
  }
  
  global_best_fitness <- -Inf
  running_best_fitness <- -Inf
  generations_without_improvement <- 0
  
  # Fitness cache to avoid re-evaluating identical recipes
  fitness_cache <- new.env(hash = TRUE, parent = emptyenv())
  # Pre-calculate fixed CV folds so fitness is comparable across individuals and generations
  fold_ids <- cut(seq(1, nrow(data)), breaks = cv_folds, labels = FALSE)
  fold_ids <- sample(fold_ids)
  
  for (g in 1:generations) {
    if (verbose) {
      if (global_best_fitness > -Inf) {
        message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
      } else {
        message(sprintf("\n--- Generation %d / %d ---", g, generations))
      }
    }
    
    # Evaluate fitness
    for (i in seq_along(pop)) {
      needs_eval <- is.na(pop[[i]]$fitness)
      if (needs_eval) {
        recipe_str <- individual_to_recipe_string(pop[[i]])
        if (exists(recipe_str, envir = fitness_cache)) {
          pop[[i]]$fitness <- get(recipe_str, envir = fitness_cache)
          if (verbose) {
            if (pop[[i]]$fitness > running_best_fitness) {
              message(sprintf("  *** NEW BEST! Tested Individual %d: %s -> Fitness: %.4f (cached) ***", i, recipe_str, pop[[i]]$fitness))
              running_best_fitness <- pop[[i]]$fitness
            } else {
              message(sprintf("  Tested Individual %d: %s -> Fitness: %.4f (cached)", i, recipe_str, pop[[i]]$fitness))
            }
          }
        } else {
          pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col, task, cv_folds, evaluator, fold_ids)
          assign(recipe_str, pop[[i]]$fitness, envir = fitness_cache)
          if (verbose) {
            if (pop[[i]]$fitness > running_best_fitness) {
              message(sprintf("  *** NEW BEST! Tested Individual %d: %s -> Fitness: %.4f ***", i, recipe_str, pop[[i]]$fitness))
              running_best_fitness <- pop[[i]]$fitness
            } else {
              message(sprintf("  Tested Individual %d: %s -> Fitness: %.4f", i, recipe_str, pop[[i]]$fitness))
            }
          }
        }
      }
    }
    
    # Sort population by fitness descending
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]
    
    best_fitness <- pop[[1]]$fitness
    if (verbose) message(sprintf("  Gen %d Best Fitness: %.4f", g, best_fitness))
    
    if (verbose) {
      message(sprintf("  Gen %d Best Recipe: %s", g, individual_to_recipe_string(pop[[1]])))
    }
    
    # Active gene pool tracking
    if (verbose) {
      all_genes <- list()
      for (ind in pop) {
        all_genes <- c(all_genes, ind$genes)
      }
      if (length(all_genes) > 0) {
        gene_formulas <- sapply(all_genes, gene_to_formula)
        gene_counts <- table(gene_formulas)
        gene_counts <- sort(gene_counts, decreasing = TRUE)
        message("  Active Gene Pool (frequency in population):")
        show_count <- min(15, length(gene_counts))
        for (i in 1:show_count) {
          name <- names(gene_counts)[i]
          message(sprintf("    - %s: %d", name, gene_counts[[name]]))
        }
        if (length(gene_counts) > show_count) {
          message(sprintf("    ... and %d other genes", length(gene_counts) - show_count))
        }
      } else {
        message("  Active Gene Pool: [Empty]")
      }
    }
    
    # Early stopping check
    if (g == 1 || best_fitness > global_best_fitness) {
      global_best_fitness <- best_fitness
      generations_without_improvement <- 0
    } else {
      generations_without_improvement <- generations_without_improvement + 1
    }
    
    if (!is.null(early_stopping_rounds) && generations_without_improvement >= early_stopping_rounds) {
      message(sprintf("  Early stopping triggered after %d generations without improvement.", early_stopping_rounds))
      break
    }
    
    if (g == generations) break
    
    # Selection: keep top 50% of current population
    num_survivors <- max(2, floor(length(pop) / 2))
    survivors <- pop[1:num_survivors]
    
    if (verbose) {
      message(sprintf("\n[Breeding] Creating next generation (keeping top %d survivors)...", num_survivors))
    }
    
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
    temperature <- 0.1
    
    # Determine target population size (Stagnation Expansion)
    target_pop_size <- pop_size
    if (dynamic_population && generations_without_improvement > 0) {
      target_pop_size <- floor(pop_size * (1.5 ^ generations_without_improvement))
      if (verbose) {
        message(sprintf("\n[Stagnation Sizing] Population expanding from %d to %d slots to boost exploration!", pop_size, target_pop_size))
      }
    }
    
    # Next generation
    next_gen <- list()
    
    # Elitism: keep best
    next_gen[[1]] <- survivors[[1]]
    if (verbose) {
      message(sprintf("  Individual 1 [Elitism] Kept best individual: %s (fitness: %.4f)", 
                      individual_to_recipe_string(survivors[[1]]), survivors[[1]]$fitness))
    }
    
    # Fill the rest
    while (length(next_gen) < target_pop_size) {
      idx <- length(next_gen) + 1
      is_expansion <- idx > pop_size
      
      if (is_expansion) {
        # Expansion slots: High exploration (no crossover, extremely high temperature)
        p_idx <- sample(seq_along(survivors), 1)
        p <- survivors[[p_idx]]
        if (verbose) {
          message(sprintf("  Individual %d [Dynamic Expansion] Aggressive mutation of Parent %d:", idx, p_idx))
        }
        child <- mutate(p, verbose = verbose, force_add = TRUE, importances = global_importances_vec, temperature = 100.0)
      } else if (stats::runif(1) < 0.7) {
        # Crossover
        p1_idx <- sample(seq_along(survivors), 1)
        p2_idx <- sample(seq_along(survivors), 1)
        p1 <- survivors[[p1_idx]]
        p2 <- survivors[[p2_idx]]
        
        # Determine whether to use union or random crossover
        use_union <- FALSE
        if (crossover_type == "union") {
          use_union <- TRUE
        } else if (crossover_type == "both") {
          use_union <- stats::runif(1) < 0.5
        }
        
        if (use_union) {
          if (verbose) {
            message(sprintf("  Individual %d via Union Crossover (Parent %d x Parent %d):", idx, p1_idx, p2_idx))
          }
          child <- union_crossover(p1, p2, verbose = verbose)
        } else {
          if (verbose) {
            message(sprintf("  Individual %d via Crossover (Parent %d x Parent %d):", idx, p1_idx, p2_idx))
          }
          child <- crossover(p1, p2, verbose = verbose)
        }
        
        if (stats::runif(1) < 0.2) {
          child <- mutate(child, verbose = verbose, importances = global_importances_vec, temperature = temperature)
        }
      } else {
        # Mutate
        p_idx <- sample(seq_along(survivors), 1)
        p <- survivors[[p_idx]]
        if (verbose) {
          message(sprintf("  Individual %d via Mutation of Parent %d:", idx, p_idx))
        }
        child <- mutate(p, verbose = verbose, importances = global_importances_vec, temperature = temperature)
      }
      
      # Validation Check: Duplicate in next_gen OR already known to be worse than best
      is_invalid <- function(c_ind, pop_list, cache, best_fit) {
        # Check 1: Duplicate in current generation
        get_out <- function(ind) {
          if (length(ind$genes) == 0) return(character(0))
          sort(vapply(ind$genes, function(g) g$output_col, character(1)))
        }
        c_out <- get_out(c_ind)
        for (existing in pop_list) {
          e_out <- get_out(existing)
          if (length(c_out) == length(e_out) && all(c_out == e_out)) return(TRUE)
        }
        
        # Check 2: Taboo search for sub-optimal known recipes
        recipe_str <- individual_to_recipe_string(c_ind)
        if (exists(recipe_str, envir = cache)) {
          known_fit <- get(recipe_str, envir = cache)
          # Use an epsilon to avoid rejecting identical but equivalent recipes if they match best
          if (!is.infinite(best_fit) && known_fit < best_fit - 1e-9) return(TRUE)
        }
        
        return(FALSE)
      }
      
      attempts <- 0
      while (is_invalid(child, next_gen, fitness_cache, global_best_fitness) && attempts < 5) {
        if (verbose) message(sprintf("    [Validation] Recipe is duplicate or known sub-optimal. Forcing mutation (Attempt %d/5)...", attempts + 1))
        child <- mutate(child, verbose = verbose, force_add = TRUE, importances = global_importances_vec, temperature = if (is_expansion) 100.0 else temperature)
        attempts <- attempts + 1
      }
      
      next_gen <- c(next_gen, list(child))
    }
    pop <- next_gen
  }
  
  # Final evaluation of new individuals
  for (i in seq_along(pop)) {
    needs_eval <- is.na(pop[[i]]$fitness)
    if (needs_eval) {
      recipe_str <- individual_to_recipe_string(pop[[i]])
      if (exists(recipe_str, envir = fitness_cache)) {
        pop[[i]]$fitness <- get(recipe_str, envir = fitness_cache)
        if (verbose) {
          if (pop[[i]]$fitness > running_best_fitness) {
            message(sprintf("  *** NEW BEST! Final tested Individual %d: %s -> Fitness: %.4f (cached) ***", i, recipe_str, pop[[i]]$fitness))
            running_best_fitness <- pop[[i]]$fitness
          } else {
            message(sprintf("  Final tested Individual %d: %s -> Fitness: %.4f (cached)", i, recipe_str, pop[[i]]$fitness))
          }
        }
      } else {
        pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col, task, cv_folds, evaluator, fold_ids)
        assign(recipe_str, pop[[i]]$fitness, envir = fitness_cache)
        if (verbose) {
          if (pop[[i]]$fitness > running_best_fitness) {
            message(sprintf("  *** NEW BEST! Final tested Individual %d: %s -> Fitness: %.4f ***", i, recipe_str, pop[[i]]$fitness))
            running_best_fitness <- pop[[i]]$fitness
          } else {
            message(sprintf("  Final tested Individual %d: %s -> Fitness: %.4f", i, recipe_str, pop[[i]]$fitness))
          }
        }
      }
    }
  }
  fitness_vals <- sapply(pop, function(ind) ind$fitness)
  pop <- pop[order(fitness_vals, decreasing = TRUE)]
  
  best_ind <- pop[[1]]
  
  if (verbose) {
    message(sprintf("\nEvolution Complete. Best Fitness: %.4f", best_ind$fitness))
    message(sprintf("Best recipe: %s", individual_to_recipe_string(best_ind)))
    if (length(best_ind$genes) > 0) {
      best_cols_str <- paste(sapply(best_ind$genes, function(g) g$output_col), collapse = ", ")
      message(sprintf("Generated columns: %s", best_cols_str))
    }
  }
  
  structure(
    list(
      best_individual = best_ind,
      history = pop,
      task = task
    ),
    class = "evo_recipe"
  )
}
