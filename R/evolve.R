truncate_cols <- function(cols, max_show = 10) {
  if (length(cols) <= max_show) {
    return(paste(cols, collapse = ", "))
  }
  paste0(paste(cols[1:max_show], collapse = ", "), ", ... (+ ", length(cols) - max_show, " more)")
}

# --- Internal helpers ---

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
                         fitness_cache, threads, verbose, running_best_fitness) {
  for (i in seq_along(pop)) {
    if (!is.na(pop[[i]]$fitness)) next

    recipe_str <- individual_to_recipe_string(pop[[i]])
    cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
    cached <- exists(cache_key, envir = fitness_cache)

    if (cached) {
      pop[[i]] <- get(cache_key, envir = fitness_cache)
    } else {
      pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col, task = task, cv_folds = cv_folds,
                                    evaluation_strategy = evaluation_strategy,
                                    split_ids = split_ids, shared_splits = shared_splits,
                                    evaluator = evaluator, fold_ids = fold_ids, 
                                    shared_folds = shared_folds,
                                    shared_full = shared_full, state_cache = state_cache,
                                    threads = threads)
      assign(cache_key, pop[[i]], envir = fitness_cache)
    }

    if (verbose) {
      new_best_str <- if (pop[[i]]$fitness > running_best_fitness) " (New Best!)" else ""
      cache_str <- if (cached) " (cached)" else ""
      message(sprintf("  Tested Individual %d%s -> Fitness: %.4f%s",
                      i, new_best_str, pop[[i]]$fitness, cache_str))
      if (pop[[i]]$fitness > running_best_fitness) {
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
  cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
  if (exists(cache_key, envir = cache)) {
    cached_ind <- get(cache_key, envir = cache)
    known_fit <- cached_ind$fitness
    # Use an epsilon to avoid rejecting recipes that match the best
    if (!is.infinite(best_fit) && known_fit < best_fit - 1e-9) return(TRUE)
  }

  return(FALSE)
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
#' @param split_ratio A numeric vector of length 2 or 3 defining train/validation/holdout proportions (e.g. c(0.6, 0.2, 0.2)).
#' @param split_ids An optional character vector of split assignments (e.g. "train", "val", "holdout").
#' @param early_stopping_rounds Stop if fitness doesn't improve for this many generations
#' @param evaluator The ML model to use ("lightgbm" or "xgboost")
#' @param dynamic_population Logical. If TRUE, population expands dynamically during stagnation.
#' @param crossover_type Crossover type: "both" (default, 50\% random / 50\% union), "random", or "union"
#' @param threads Number of threads to use for parallel execution (default 8)
#' @param max_clustering_size Maximum unique training rows to cluster (default 5000, 0/NULL for unlimited)
#' @param seed Optional integer seed for reproducibility.
#' @param verbose Logical. If TRUE, prints progress.
#' @export
evolve_features <- function(data, target_col, task = "classification", 
                            generations = 10, pop_size = 10, cv_folds = 3, 
                            evaluation_strategy = "cv", split_ratio = c(0.6, 0.2, 0.2),
                            split_ids = NULL,
                            early_stopping_rounds = 3, evaluator = "lightgbm",
                            dynamic_population = TRUE, crossover_type = "both", 
                            threads = 8, max_clustering_size = 5000, 
                            seed = NULL, verbose = TRUE) {
  if (!is.null(seed)) set.seed(seed)
  
  # Temporarily configure max clustering size and threads options
  old_max_size <- getOption("evoFE.max_clustering_size")
  old_threads <- getOption("evoFE.threads")
  options(evoFE.max_clustering_size = max_clustering_size, evoFE.threads = threads)
  on.exit({
    options(evoFE.max_clustering_size = old_max_size)
    options(evoFE.threads = old_threads)
  }, add = TRUE)
  
  # Prevent macOS OpenMP thread collisions between data.table, lightgbm, and other libraries
  if (requireNamespace("RhpcBLASctl", quietly = TRUE)) {
    RhpcBLASctl::omp_set_num_threads(threads)
    RhpcBLASctl::blas_set_num_threads(threads)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    data.table::setDTthreads(threads)
  }
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    tryCatch({
      quitefastmst::omp_set_num_threads(threads)
    }, error = function(e) NULL)
  }

  if (!target_col %in% names(data)) {
    stop(sprintf("Target column '%s' not found in the dataset.", target_col))
  }
  
  original_cols <- setdiff(names(data), target_col)
  numeric_cols <- names(data)[sapply(data, is.numeric)]
  numeric_cols <- setdiff(numeric_cols, target_col)
  categorical_cols <- setdiff(original_cols, numeric_cols)
  
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
      message(sprintf("  Generations: %d, Population Size: %d, Strategy: Split (%s)", 
                      generations, pop_size, paste(split_ratio, collapse = "/")))
    }
    message(sprintf("  Original Numeric columns: %s", truncate_cols(numeric_cols)))
    message(sprintf("  Original Categorical columns: %s", truncate_cols(categorical_cols)))
  }
  
  pop <- initialize_population(pop_size, numeric_cols, categorical_cols, initial_genes = 2, task = task)
  
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
  # State cache for full dataset to avoid re-fitting stateful transformers
  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  
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
  
  for (g in 1:generations) {
    if (verbose) {
      if (global_best_fitness > -Inf) {
        message(sprintf("\n--- Generation %d / %d (Current Best Fitness: %.4f) ---", g, generations, global_best_fitness))
      } else {
        message(sprintf("\n--- Generation %d / %d ---", g, generations))
      }
    }
    
    # Evaluate fitness
    eval_res <- evaluate_pop(pop, data, target_col, task, cv_folds, evaluation_strategy,
                              split_ids_val, shared_splits, evaluator,
                              fold_ids, shared_folds, shared_full, state_cache,
                              fitness_cache, threads, verbose, running_best_fitness)
    pop <- eval_res$pop
    running_best_fitness <- eval_res$running_best_fitness
    
    # Sort population by fitness descending
    fitness_vals <- sapply(pop, function(ind) ind$fitness)
    pop <- pop[order(fitness_vals, decreasing = TRUE)]
    
    best_fitness <- pop[[1]]$fitness
    if (verbose) message(sprintf("  Gen %d Best Fitness: %.4f", g, best_fitness))
    
    if (verbose) {
      message(sprintf("  Gen %d Best Recipe: %s", g, individual_to_recipe_string(pop[[1]])))
    }
    
    # (Active gene pool tracking removed to reduce verbosity)
    
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
    
    # (Breeding starts silently)
    
    # Collect outputs from evaluated genes — only these are safe for chaining
    tested_gene_outputs <- unique(unlist(lapply(pop, function(ind) {
      if (length(ind$genes) == 0) return(character(0))
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
    temperature <- 0.1
    
    # Determine target population size (Stagnation Expansion)
    target_pop_size <- pop_size
    if (dynamic_population && generations_without_improvement > 0) {
      target_pop_size <- floor(pop_size * (1.5 ^ generations_without_improvement))
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
        p_idx <- sample(seq_along(survivors), 1)
        p <- survivors[[p_idx]]
        child <- mutate(p, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = 100.0, task = task, tested_gene_outputs = tested_gene_outputs)
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
          child <- union_crossover(p1, p2, verbose = FALSE)
        } else {
          child <- crossover(p1, p2, verbose = FALSE)
        }
        
        if (stats::runif(1) < 0.2) {
          child <- mutate(child, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs)
        }
      } else {
        # Mutate
        p_idx <- sample(seq_along(survivors), 1)
        p <- survivors[[p_idx]]
        child <- mutate(p, verbose = FALSE, importances = global_importances_vec, temperature = temperature, task = task, tested_gene_outputs = tested_gene_outputs)
      }
      
      # Validation Check: Duplicate in next_gen OR already known to be worse than best
      attempts <- 0
      while (is_invalid_individual(child, next_gen, fitness_cache, global_best_fitness) && attempts < 5) {
        child <- mutate(child, verbose = FALSE, force_add = TRUE, importances = global_importances_vec, temperature = if (is_expansion) 100.0 else temperature, task = task, tested_gene_outputs = tested_gene_outputs)
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
                            fitness_cache, threads, verbose, running_best_fitness)
  pop <- eval_res$pop
  fitness_vals <- sapply(pop, function(ind) ind$fitness)
  pop <- pop[order(fitness_vals, decreasing = TRUE)]
  
  best_ind <- pop[[1]]
  
  if (evaluation_strategy == "split" && ("holdout" %in% split_ids_val || !is.null(shared_splits$holdout))) {
    best_ind <- evaluate_holdout_fitness(best_ind, data, split_ids_val, shared_splits,
                                         target_col, task, evaluator, threads, state_cache,
                                         classes, num_class)
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
  res_full <- apply_individual(best_ind, shared_full, NULL, target_col, state_cache = state_cache)
  best_ind <- res_full$ind
  
  gene_cols <- if (length(best_ind$genes) > 0) vapply(best_ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(best_ind$numeric_cols, best_ind$categorical_cols, gene_cols)
  
  x_full <- data.matrix(res_full$train[, features, with = FALSE])
  x_full[!is.finite(x_full)] <- NA
  y_full <- res_full$train[[target_col]]
  if (task == "multiclass") {
    y_full <- as.integer(factor(y_full, levels = classes)) - 1
  }
  
  res_model <- train_model(x_full, y_full, task = task, evaluator = evaluator,
                            threads = threads, num_class = num_class)
  best_model <- res_model$model
  
  structure(
    list(
      best_individual = best_ind,
      history = pop,
      task = task,
      best_model = best_model,
      evaluator = evaluator,
      classes = classes
    ),
    class = "evo_recipe"
  )
}
