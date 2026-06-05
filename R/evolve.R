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
  if (term %in% c("dumb", "")) return(FALSE)
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
                         metric = "default", ...) {
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
                                    threads = threads, metric = metric, verbose = verbose, ...)
      assign(cache_key, pop[[i]], envir = fitness_cache)
    }

    if (verbose) {
      improved <- pop[[i]]$fitness > running_best_fitness
      green_start <- if (supports_color()) "\033[32m" else ""
      red_start   <- if (supports_color()) "\033[31m" else ""
      color_reset <- if (supports_color()) "\033[0m" else ""
      
      new_best_str <- if (improved) " (New Best!)" else ""
      cache_str <- if (cached) " (cached)" else ""
      msg_color <- if (improved) green_start else red_start
      
      msg <- sprintf("  Tested Individual %d%s -> Fitness: %.4f%s",
                     i, new_best_str, pop[[i]]$fitness, cache_str)
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
#' @param evaluator The ML model to use ("lightgbm", "xgboost", "catboost", or a custom registered evaluator name).
#' @param dynamic_population Logical. If TRUE, population expands dynamically during stagnation.
#' @param dynamic_population_growth_rate Growth rate multiplier for population expansion during stagnation (default 1.5).
#' @param dynamic_population_decay_rate Decay rate multiplier for population contraction back to baseline (default 0.7).
#' @param crossover_type Crossover type: "both" (default, 50\% random / 50\% union), "random", or "union"
#' @param threads Number of threads to use for parallel execution (default 2)
#' @param max_clustering_size Maximum unique training rows to cluster (default 5000, 0/NULL for unlimited)
#' @param seed Optional integer seed for reproducibility.
#' @param verbose Logical. If TRUE, prints progress.
#' @param metric The metric to optimize ("default", "auc", "f1", "mae", or a custom function).
#' @param model_all_final_genes Logical. If TRUE, the final model is trained using the union of all unique genes evolved in the final population, rather than only the best individual's genes.
#' @param model_all_historical_genes Logical. If TRUE, the final model is trained using the union of all unique genes evolved across all generations, rather than only the best individual's genes.
#' @param ... Additional arguments passed to the underlying evaluator training functions.
#' @examples
#' \donttest{
#' # Quick classification example using mtcars
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   seed = 42,
#'   verbose = FALSE
#' )
#' print(recipe)
#' }
#' @export
evolve_features <- function(data, target_col, task = "classification", 
                            generations = 10, pop_size = 10, cv_folds = 3, 
                            evaluation_strategy = "cv", split_ratio = c(0.6, 0.2, 0.2),
                            split_ids = NULL,
                            early_stopping_rounds = 3, evaluator = "lightgbm",
                            dynamic_population = TRUE,
                            dynamic_population_growth_rate = 1.5,
                            dynamic_population_decay_rate = 0.7,
                            crossover_type = "both", 
                            threads = 2, max_clustering_size = 5000, 
                            seed = NULL, verbose = TRUE, metric = "default", 
                            model_all_final_genes = FALSE,
                            model_all_historical_genes = FALSE, ...) {
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
    old_omp <- RhpcBLASctl::omp_get_max_threads()
    old_blas <- RhpcBLASctl::blas_get_num_procs()
    RhpcBLASctl::omp_set_num_threads(threads)
    RhpcBLASctl::blas_set_num_threads(threads)
    on.exit({
      RhpcBLASctl::omp_set_num_threads(old_omp)
      RhpcBLASctl::blas_set_num_threads(old_blas)
    }, add = TRUE)
  }
  if (requireNamespace("data.table", quietly = TRUE)) {
    old_dt <- data.table::getDTthreads()
    data.table::setDTthreads(threads)
    on.exit({
      data.table::setDTthreads(old_dt)
    }, add = TRUE)
  }
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    tryCatch({
      old_qf <- quitefastmst::omp_get_max_threads()
      quitefastmst::omp_set_num_threads(threads)
      on.exit({
        tryCatch({
          quitefastmst::omp_set_num_threads(old_qf)
        }, error = function(e) NULL)
      }, add = TRUE)
    }, error = function(e) NULL)
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
      stop(sprintf("Metric '%s' is not supported for task '%s'. Supported metrics are: %s",
                   metric, task, paste(valid_metrics[[task]], collapse = ", ")))
    }
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
  baseline_ind <- create_individual(genes = list(), numeric_cols = numeric_cols, categorical_cols = categorical_cols)
  if (verbose) {
    message("\n--- Generation 0 (Baseline) ---")
    message(sprintf("  Individual 1: %s", individual_to_recipe_string(baseline_ind)))
  }
  baseline_ind <- evaluate_fitness(
    baseline_ind, data, target_col, task = task, cv_folds = cv_folds,
    evaluation_strategy = evaluation_strategy,
    split_ids = split_ids_val, shared_splits = shared_splits,
    evaluator = evaluator, fold_ids = fold_ids, 
    shared_folds = shared_folds,
    shared_full = shared_full, state_cache = state_cache,
    threads = threads, metric = metric, verbose = verbose, ...
  )
  if (verbose) {
    message(sprintf("  Tested Individual 1 -> Fitness: %.4f", baseline_ind$fitness))
  }

  # Cache the baseline individual's fitness
  recipe_str <- individual_to_recipe_string(baseline_ind)
  cache_key <- digest::digest(recipe_str, algo = "md5", serialize = FALSE)
  assign(cache_key, baseline_ind, envir = fitness_cache)
  
  # 2. Initialize population for Generation 1 using baseline importances
  pop <- initialize_population(pop_size, numeric_cols, categorical_cols, initial_genes = 2, task = task, importances = baseline_ind$importances)
  pop[[1]] <- baseline_ind
  
  if (verbose) {
    message("\n[Gen 1] Initialized Population:")
    for (i in seq_along(pop)) {
      message(sprintf("  Individual %d: %s", i, individual_to_recipe_string(pop[[i]])))
    }
  }
  
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
    }
    
    # Evaluate fitness
    eval_res <- evaluate_pop(pop, data, target_col, task, cv_folds, evaluation_strategy,
                              split_ids_val, shared_splits, evaluator,
                              fold_ids, shared_folds, shared_full, state_cache,
                              fitness_cache, threads, verbose, running_best_fitness,
                              metric = metric, ...)
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
      fitness_history <- fitness_history[1:g]
      break
    }
    
    if (g == generations) break
    
    # Selection: keep top 50% of current population
    num_survivors <- min(length(pop), max(2, floor(length(pop) / 2)))
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
      target_pop_size <- current_pop_size
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
                            fitness_cache, threads, verbose, running_best_fitness,
                            metric = metric, ...)
  pop <- eval_res$pop
  fitness_vals <- sapply(pop, function(ind) ind$fitness)
  pop <- pop[order(fitness_vals, decreasing = TRUE)]
  
  best_ind <- pop[[1]]
  
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
      categorical_cols = categorical_cols
    )
    
    # 4. Evaluate the super-individual's fitness
    super_ind <- evaluate_fitness(
      super_ind, data, target_col, task = task, cv_folds = cv_folds,
      evaluation_strategy = evaluation_strategy,
      split_ids = split_ids_val, shared_splits = shared_splits,
      evaluator = evaluator, fold_ids = fold_ids, 
      shared_folds = shared_folds,
      shared_full = shared_full, state_cache = state_cache,
      threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE, ...
    )
    
    if (is.null(super_ind$best_params) && !is.null(best_ind$best_params)) {
      super_ind$best_params <- best_ind$best_params
    }
    
    if (super_ind$fitness > best_ind$fitness) {
      if (verbose) {
        message(sprintf("  Pooled features improved validation fitness from %.4f to %.4f. Using pooled features.", 
                        best_ind$fitness, super_ind$fitness))
      }
      best_ind <- super_ind
    } else {
      if (verbose) {
        message(sprintf("  Pooled features (fitness: %.4f) did not exceed best individual (fitness: %.4f). Using best individual.", 
                        super_ind$fitness, best_ind$fitness))
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
        categorical_cols = categorical_cols
      )
      
      # Evaluate the historical super-individual's fitness
      super_ind_hist <- evaluate_fitness(
        super_ind_hist, data, target_col, task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids_val, shared_splits = shared_splits,
        evaluator = evaluator, fold_ids = fold_ids, 
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose, allow_prune = TRUE, ...
      )
      
      if (is.null(super_ind_hist$best_params) && !is.null(best_ind$best_params)) {
        super_ind_hist$best_params <- best_ind$best_params
      }
      
      if (super_ind_hist$fitness > best_ind$fitness) {
        if (verbose) {
          message(sprintf("  Historical pooled features improved validation fitness from %.4f to %.4f. Using historical pooled features.", 
                          best_ind$fitness, super_ind_hist$fitness))
        }
        best_ind <- super_ind_hist
      } else {
        if (verbose) {
          message(sprintf("  Historical pooled features (fitness: %.4f) did not exceed current best fitness (fitness: %.4f). Keeping current best individual.", 
                          super_ind_hist$fitness, best_ind$fitness))
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
                                         classes, num_class, metric = metric, verbose = verbose, ...)
  }
  
  if (verbose) {
    message("\nCache contents:")
    for (k in ls(envir = fitness_cache)) {
      ind_temp <- get(k, envir = fitness_cache)
      message(sprintf("  Key: %s -> Recipe: %s -> Fitness: %.4f", k, individual_to_recipe_string(ind_temp), ind_temp$fitness))
    }
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
  features <- c(best_ind$numeric_cols, best_ind$categorical_cols, gene_cols)
  
  x_full <- data.matrix(res_full$train[, features, with = FALSE])
  x_full[!is.finite(x_full)] <- NA
  y_full <- res_full$train[[target_col]]
  if (task == "multiclass") {
    y_full <- as.integer(factor(y_full, levels = classes)) - 1
  }
  
  # Train the final model on the full dataset. Since we are using all data, we use the original
  # tuner evaluator (e.g., lightgbm_mbo) so that it performs hyperparameter tuning on the full
  # dataset, using the best parameters found during evolution as a seed.
  res_model <- train_model(x_full, y_full, task = task, evaluator = evaluator,
                            threads = threads, num_class = num_class, metric = metric,
                            verbose = verbose, best_params = best_params, ...)
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
