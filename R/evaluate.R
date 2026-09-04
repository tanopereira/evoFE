#' Apply a single gene to a dataset
#'
#' @param gene A gene list representing a feature transformation.
#' @param train_data A data.frame or data.table representing the training data.
#' @param val_data Optional validation data.frame or data.table.
#' @param target_col Name of the target column.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param data_hash Optional pre-computed xxhash64 digest of the target column, to avoid redundant hashing when applying multiple genes.
#' @return A list with three elements: \code{train} (the modified training
#'   \code{data.table} with the new gene column appended), \code{val} (the
#'   modified validation \code{data.table} or \code{NULL}), and \code{gene}
#'   (the gene list, with its \code{state} element populated if the transformer
#'   is stateful).
#' @export
apply_gene <- function(gene, train_data, val_data = NULL, target_col = NULL, state_cache = NULL, data_hash = NULL) {
  t_def <- evo_transformers[[gene$transformer_name]]

  col_exists_train <- gene$output_col %in% names(train_data)
  col_exists_val <- if (!is.null(val_data)) gene$output_col %in% names(val_data) else TRUE

  if (col_exists_train && col_exists_val && !is.null(gene$state)) {
    return(list(train = train_data, val = val_data, gene = gene))
  }

  state <- NULL
  has_cached_state <- FALSE
  cache_key <- NULL
  if (!is.null(state_cache) && !is.null(target_col)) {
    if (is.null(data_hash)) {
      data_hash <- digest::digest(train_data[[target_col]], algo = "xxhash64")
    }
    cache_key <- digest::digest(paste0(gene_to_state_formula(gene), "_", data_hash), algo = "md5", serialize = FALSE)
    if (exists(cache_key, envir = state_cache, inherits = FALSE)) {
      state <- get(cache_key, envir = state_cache)
      gene$state <- state
      has_cached_state <- TRUE
    }
  }

  # If we are fitting (target_col provided) and it's stateful
  if (!has_cached_state && !is.null(t_def$fit_func) && !is.null(target_col)) {
    if (!is.null(gene$state)) {
      state <- gene$state
    } else {
      # Skip fitting in CV folds if columns already exist
      if (is.null(state_cache) && col_exists_train && col_exists_val) {
        # Skip fitting, state remains NULL
      } else {
        state <- t_def$fit_func(train_data, gene, target_col)
        gene$state <- state
        if (!is.null(cache_key)) {
          assign(cache_key, state, envir = state_cache)
        }
      }
    }
  } else if (!is.null(gene$state)) {
    # If we are predicting
    state <- gene$state
  }

  out_type <- if (!is.null(t_def$output_type)) t_def$output_type else "numeric"

  # Apply to train
  if (!col_exists_train) {
    new_col_train <- t_def$apply_func(train_data, gene, state)
    if (is.double(new_col_train)) {
      new_col_train[!is.finite(new_col_train) | abs(new_col_train) > 3.402823e38] <- NA_real_
    }

    # Reject constant columns (0 variance)
    if (!is.null(target_col) && length(unique(new_col_train[!is.na(new_col_train)])) <= 1) {
      stop("Constant column generated")
    }

    # Reject columns that are near-perfect duplicates of existing features.
    # Guard with !is.null(target_col): during inference (holdout/predict) we
    # must apply every gene that was accepted at training time <U+2014> correlation on
    # a different data split must never prune a gene the model depends on.
    cor_threshold <- getOption("evoFE.redundancy_cor_threshold", 0.95)
    if (!is.null(target_col) && is.numeric(new_col_train) && cor_threshold < 1) {
      num_mask <- vapply(train_data, is.numeric, logical(1))
      existing_num_cols <- names(train_data)[num_mask]
      if (gene$output_col %in% existing_num_cols) {
        existing_num_cols <- existing_num_cols[existing_num_cols != gene$output_col]
      }
      if (length(existing_num_cols) > 0) {
        new_is_finite <- is.finite(new_col_train)
        if (sum(new_is_finite) > 2 && suppressWarnings(stats::sd(new_col_train[new_is_finite])) > 0) {
          for (ecol in existing_num_cols) {
            ev <- train_data[[ecol]]
            # Find indices where both vectors are finite to preserve row alignment
            valid_idx <- new_is_finite & is.finite(ev)
            if (sum(valid_idx) > 2) {
              r <- tryCatch(
                suppressWarnings(abs(stats::cor(new_col_train[valid_idx], ev[valid_idx],
                  use = "complete.obs"
                ))),
                error = function(e) 0
              )
              if (!is.na(r) && r >= cor_threshold) stop("Redundant column")
            }
          }
        }
      }
    }

    if (out_type == "categorical") {
      new_col_train <- as.factor(new_col_train)
    } else if (is.double(new_col_train)) {
      new_col_train[!is.finite(new_col_train) | abs(new_col_train) > 3.402823e38] <- NA_real_
    }

    if (data.table::is.data.table(train_data)) {
      train_data[, (gene$output_col) := new_col_train]
    } else {
      train_data[[gene$output_col]] <- new_col_train
    }
  }

  # Apply to val
  if (!is.null(val_data) && !col_exists_val) {
    new_col_val <- t_def$apply_func(val_data, gene, state)
    if (out_type == "categorical") {
      # Fallback level alignment in case train_data has already been converted to factor
      train_factor <- train_data[[gene$output_col]]
      train_levels <- if (is.factor(train_factor)) levels(train_factor) else unique(as.character(train_factor))
      new_col_val <- factor(new_col_val, levels = train_levels)
    } else if (is.double(new_col_val)) {
      new_col_val[!is.finite(new_col_val) | abs(new_col_val) > 3.402823e38] <- NA_real_
    }
    if (data.table::is.data.table(val_data)) {
      val_data[, (gene$output_col) := new_col_val]
    } else {
      val_data[[gene$output_col]] <- new_col_val
    }
  }

  list(train = train_data, val = val_data, gene = gene)
}

#' Apply an entire individual's recipe to data
#'
#' @param ind An evo_individual object.
#' @param train_data A data.frame or data.table representing the training data.
#' @param val_data Optional validation data.frame or data.table.
#' @param target_col Name of the target column.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param allow_prune Logical. If TRUE, genes that fail application are skipped instead of failing the entire individual.
#' @return A list with three elements: \code{train} (the transformed training
#'   \code{data.table} with all gene columns applied), \code{val} (the
#'   transformed validation \code{data.table} or \code{NULL}), and \code{ind}
#'   (the updated \code{evo_individual} whose genes now carry fitted states).
#' @export
apply_individual <- function(ind, train_data, val_data = NULL, target_col = NULL, state_cache = NULL, allow_prune = TRUE) {
  dt_train <- if (data.table::is.data.table(train_data)) train_data else data.table::as.data.table(train_data)
  dt_val <- if (!is.null(val_data)) {
    if (data.table::is.data.table(val_data)) val_data else data.table::as.data.table(val_data)
  } else {
    NULL
  }

  # Pre-compute target column hash once for all genes (avoids redundant hashing)
  pre_hash <- if (!is.null(state_cache) && !is.null(target_col)) {
    digest::digest(dt_train[[target_col]], algo = "xxhash64")
  } else {
    NULL
  }

  new_genes <- list()
  for (gene in ind$genes) {
    res <- tryCatch(
      {
        if (!all(gene$input_cols %in% names(dt_train))) {
          stop("Input column missing")
        }
        apply_gene(gene, dt_train, dt_val, target_col, state_cache = state_cache, data_hash = pre_hash)
      },
      error = function(e) {
        NULL
      }
    )

    if (is.null(res) || !is.null(res$skip)) {
      if (allow_prune) {
        next
      } else {
        return(NULL)
      }
    }

    dt_train <- res$train
    dt_val <- res$val
    new_genes[[length(new_genes) + 1L]] <- res$gene
  }

  ind$genes <- new_genes

  # Safety check: if genes were pruned and total active columns fell below min_active,
  # restore raw columns to meet the safety floor
  all_num <- ind$all_numeric_cols
  all_cat <- ind$all_categorical_cols
  all_date <- ind$all_datetime_cols
  total_avail <- length(all_num) + length(all_cat) + length(all_date)
  if (total_avail > 0) {
    total_active <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols) + length(ind$genes)
    min_active <- if (total_avail >= 2) 2 else 1
    if (total_active < min_active) {
      all_cols <- c(all_num, all_cat, all_date)
      active_cols <- c(ind$numeric_cols, ind$categorical_cols, ind$datetime_cols)
      inactive_cols <- setdiff(all_cols, active_cols)
      needed <- min_active - total_active
      if (length(inactive_cols) > 0) {
        to_activate <- sample(inactive_cols, min(length(inactive_cols), needed))
        for (col in to_activate) {
          if (col %in% all_num) {
            ind$numeric_cols <- unique(c(ind$numeric_cols, col))
          } else if (col %in% all_cat) {
            ind$categorical_cols <- unique(c(ind$categorical_cols, col))
          } else if (col %in% all_date) ind$datetime_cols <- unique(c(ind$datetime_cols, col))
        }
      }
    }
  }

  list(train = dt_train, val = dt_val, ind = ind)
}

compute_exp_neg_logloss <- function(y_true, y_pred) {
  if (is.factor(y_true)) {
    y_true <- as.integer(y_true) - 1L
  } else if (is.character(y_true)) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  } else if (is.logical(y_true)) {
    y_true <- as.integer(y_true)
  } else if (is.numeric(y_true) && !all(stats::na.omit(y_true) %in% c(0, 1))) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  }
  p <- pmax(pmin(as.numeric(y_pred), 1 - 1e-15), 1e-15)
  ll <- -mean(y_true * log(p) + (1 - y_true) * log(1 - p), na.rm = TRUE)
  exp(-ll)
}

compute_exp_neg_multiclass_logloss <- function(y_true, y_pred, num_class) {
  n <- length(y_true)
  if (!is.matrix(y_pred)) {
    y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
  }
  y_idx <- as.integer(y_true)
  if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
    idx_col <- y_idx
  } else {
    idx_col <- y_idx + 1L
  }
  idx_col <- pmax(1L, pmin(as.integer(idx_col), as.integer(num_class)))
  probs <- y_pred[cbind(seq_len(n), idx_col)]
  probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
  ll <- -mean(log(probs), na.rm = TRUE)
  exp(-ll)
}


#' Compute Complexity Penalty for an Individual
#'
#' Computes the parsimony complexity penalty using Bayesian Information Criterion (BIC)
#' scaling (\code{ln(N) / (2N)}) and optional dynamic convergence scaling based on the
#' relative gap to the metric ceiling.
#'
#' @param n_genes Integer. Number of evolved genes in the individual.
#' @param n_genes Integer. Number of evolved genes in the recipe.
#' @param n_features Optional integer. Total number of active features (active raw features plus genes). If NULL, defaults to n_genes.
#' @param n_samples Integer. Number of dataset samples (rows).
#' @param running_best_fitness Numeric. Current running best fitness in the population/island.
#' @param baseline_fitness Numeric. Generation 0 baseline fitness (raw features only).
#' @param metric Character or function. Metric being optimized.
#' @param task Character. "classification", "multiclass", or "regression".
#' @param complexity_penalty Numeric. Dimensionless penalty multiplier (default 0).
#' @param complexity_mode Character. "bic_dynamic" (default), "bic", "pac_bayes_dynamic", "pac_bayes", or "none".
#' @param complexity_floor Numeric. Minimum safety floor factor for dynamic penalties (default 0.20, representing 20\% of base penalty).
#' @param complexity_target Character. "all_features" (default, penalizes total active features) or "genes" (penalizes only derived genes).
#' @param n_features Optional integer. Total number of active features (active raw features plus genes). If NULL, defaults to n_genes.
#' @param epsilon_floor Deprecated alias for \code{complexity_floor}.
#' @return Non-negative numeric penalty to subtract from raw fitness.
#' @export
compute_complexity_penalty <- function(n_genes,
                                       n_samples,
                                       running_best_fitness = NULL,
                                       baseline_fitness = NULL,
                                       metric = "default",
                                       task = "classification",
                                       complexity_penalty = 0,
                                       complexity_mode = "bic_dynamic",
                                       complexity_floor = 0.20,
                                       complexity_target = "all_features",
                                       n_features = NULL,
                                       epsilon_floor = 0.20) {
  floor_val <- if (!missing(complexity_floor)) complexity_floor else epsilon_floor
  target_mode <- if (identical(complexity_target, "genes") || identical(complexity_target, "gene_only")) "genes" else "all_features"
  k <- if (target_mode == "genes") {
    n_genes
  } else {
    if (!is.null(n_features)) n_features else n_genes
  }

  if (complexity_penalty <= 0 || k <= 0 || complexity_mode == "none") {
    return(0)
  }

  n_samples <- as.numeric(n_samples)
  if (is.na(n_samples) || n_samples < 2) {
    n_samples <- 2
  }

  # Base penalty factor per feature/gene
  is_dynamic <- grepl("_dynamic$", complexity_mode)
  base_mode <- sub("_dynamic$", "", complexity_mode)

  base_factor <- switch(base_mode,
    "pac_bayes" = complexity_penalty / (2 * sqrt(n_samples)),
    "bic"       = complexity_penalty * (log(n_samples) / (2 * n_samples)),
    0
  )

  if (base_factor <= 0) {
    return(0)
  }

  p <- if (is_dynamic) {
    # Dynamic penalty mode: scale by normalized gap to ideal
    ideal <- if (task %in% c("classification", "multiclass")) 1.0 else 0.0

    if (is.null(baseline_fitness) || is.null(running_best_fitness) ||
      is.na(baseline_fitness) || is.na(running_best_fitness) ||
      !is.finite(baseline_fitness) || !is.finite(running_best_fitness)) {
      base_factor
    } else {
      denom <- ideal - baseline_fitness
      numer <- ideal - running_best_fitness
      gap_ratio <- if (abs(denom) <= 1e-12) 1.0 else numer / denom
      dynamic_factor <- max(floor_val, min(1.0, gap_ratio))
      base_factor * dynamic_factor
    }
  } else {
    base_factor
  }

  # Compound penalty across k features: (1 + p)^k - 1
  # Ensures adding k features simultaneously (e.g. multi-dimensional UMAP / PCA embeddings)
  # has the exact same compound error hurdle as sequential single-feature additions.
  expm1(k * log1p(p))
}

#' Evaluate the fitness of an individual
#'
#' Trains a model using the features specified by the individual's recipe and evaluates
#' performance using cross-validation or train/val split.
#'
#' @param ind An evo_individual object.
#' @param data A data.frame or data.table containing the dataset.
#' @param target_col Name of the target column.
#' @param task "classification" or "regression".
#' @param cv_folds Number of cross-validation folds.
#' @param evaluation_strategy Character string, either "cv" (cross-validation) or "split" (train/validation split).
#' @param split_ids Optional vector of pre-defined split assignments (e.g. \code{c("train", "train", "val", "holdout", "train")}). Must have the same length as the number of rows in \code{data} and contain only "train", "val", or "holdout" labels.
#' @param shared_splits Optional list of shared data.table splits for in-place caching.
#' @param evaluator Character string specifying the model backend: "lightgbm", "xgboost", "catboost", "rf", or "lm".
#' @param fold_ids Optional integer vector of pre-assigned fold indices.
#' @param shared_folds Optional list of shared data.table fold splits for in-place caching.
#' @param shared_full Optional shared full data.table.
#' @param state_cache Optional environment used to cache transformer training states across evaluations.
#' @param threads Number of threads for model training.
#' @param metric Character string or evaluation metric.
#' @param verbose Logical.
#' @param allow_prune Logical.
#' @param complexity_penalty Numeric. Dimensionless penalty multiplier (default 0).
#' @param complexity_mode Character. Complexity penalty strategy: "bic_dynamic" (default),
#'   "bic", "pac_bayes_dynamic", "pac_bayes", or "none".
#' @param complexity_floor Numeric in \code{[0, 1]}. Minimum safety floor factor for dynamic penalty (default \code{0.20}).
#' @param complexity_target Character. "all_features" (default, penalizes total active features) or "genes" (penalizes only derived genes).
#' @param running_best_fitness Optional numeric. Current running best fitness for dynamic BIC.
#' @param baseline_fitness Optional numeric. Generation 0 baseline fitness for dynamic BIC.
#' @param n_samples Optional integer. Dataset sample size N for BIC calculations. Defaults to nrow(data).
#' @param cv_strategy Fold construction strategy for CV: \code{"random"} (default), \code{"time"}, or \code{"group"}.
#' @param time_col Column name used when \code{cv_strategy = "time"}.
#' @param group_col Column name used when \code{cv_strategy = "group"}.
#' @param ... Additional arguments passed to the underlying evaluator training functions.
#' @return The input \code{evo_individual} with its \code{fitness} field set to
#'   the computed score (higher is better), \code{importances} set to a named
#'   numeric vector of feature importances, \code{holdout_fitness} set to
#'   \code{NULL}, and \code{genes} updated with fitted transformer states.
#' @export
evaluate_fitness <- function(ind, data, target_col, task = "classification",
                             cv_folds = 3, evaluation_strategy = "cv",
                             split_ids = NULL, shared_splits = NULL,
                             evaluator = "lightgbm", fold_ids = NULL,
                             shared_folds = NULL, shared_full = NULL,
                             state_cache = NULL, threads = 2,
                             metric = "default", verbose = FALSE, allow_prune = TRUE,
                             complexity_penalty = 0, complexity_mode = "bic_dynamic",
                             complexity_floor = 0.20, complexity_target = "all_features",
                             running_best_fitness = NULL, baseline_fitness = NULL,
                             n_samples = NULL, cv_strategy = "random",
                             time_col = NULL, group_col = NULL, ...) {
  if (!is.na(ind$fitness)) {
    return(ind)
  }

  num_class <- NULL
  classes <- NULL
  if (task == "multiclass") {
    target_factor <- as.factor(data[[target_col]])
    classes <- levels(target_factor)
    num_class <- length(classes)
  }

  if (evaluation_strategy %in% c("split", "metacv", "meta_cv")) {
    # Train / Validation split strategy
    if (!is.null(shared_splits)) {
      train_fold <- shared_splits$train
      val_fold <- shared_splits$val
    } else {
      train_fold <- data.table::as.data.table(data[split_ids == "train", ])
      val_fold <- data.table::as.data.table(data[split_ids == "val", ])
    }

    # Copy train/val folds so we can modify them when applying recipe
    train_fold <- data.table::copy(train_fold)
    val_fold <- data.table::copy(val_fold)

    # Apply genes
    res <- tryCatch(
      {
        apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache, allow_prune = allow_prune)
      },
      error = function(e) {
        NULL
      }
    )

    if (is.null(res)) {
      # Lethal mutation: invalid gene dependency graph
      ind$raw_fitness <- -Inf
      ind$penalty <- 0.0
      ind$fitness <- -Inf
      ind$holdout_fitness <- NULL
      return(ind)
    }

    train_fold_feat <- res$train
    val_fold_feat <- res$val

    # Features = original + new
    gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
    features <- c(res$ind$numeric_cols, res$ind$categorical_cols, res$ind$datetime_cols, gene_cols)

    # Convert features to numeric matrix
    has_cat <- length(res$ind$categorical_cols) > 0 || length(res$ind$datetime_cols) > 0 ||
      any(vapply(res$ind$genes, function(g) {
        t_def <- evo_transformers[[g$transformer_name]]
        !is.null(t_def$output_type) && t_def$output_type == "categorical"
      }, logical(1)))
    dt_sub_tr <- train_fold_feat[, features, with = FALSE]
    dt_sub_va <- val_fold_feat[, features, with = FALSE]
    x_train <- .sanitize_feature_matrix(dt_sub_tr)
    x_val <- .sanitize_feature_matrix(dt_sub_va)
    y_train <- train_fold_feat[[target_col]]
    y_val <- val_fold_feat[[target_col]]
    if (task == "multiclass") {
      y_train <- as.integer(factor(y_train, levels = classes)) - 1
      y_val <- as.integer(factor(y_val, levels = classes)) - 1
    } else if (task == "classification") {
      if (is.factor(y_train)) {
        y_train <- as.integer(y_train) - 1L
        y_val <- as.integer(y_val) - 1L
      } else if (is.character(y_train)) {
        y_fac <- as.factor(y_train)
        y_train <- as.integer(y_fac) - 1L
        y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
      } else if (is.logical(y_train)) {
        y_train <- as.integer(y_train)
        y_val <- as.integer(y_val)
      } else if (is.numeric(y_train) && !all(stats::na.omit(y_train) %in% c(0, 1))) {
        y_fac <- as.factor(y_train)
        y_train <- as.integer(y_fac) - 1L
        y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
      }
    }

    res_model <- train_model(x_train, y_train, x_val,
      y_val = y_val, task = task,
      evaluator = evaluator, threads = threads,
      num_class = num_class, metric = metric,
      verbose = verbose, ...
    )
    preds <- res_model$predictions

    # Store importances (single fold)
    if (!is.null(res_model$importances) && length(res_model$importances) > 0) {
      imp <- res_model$importances
      imp_sum <- sum(imp, na.rm = TRUE)
      if (imp_sum > 0) {
        imp <- imp / imp_sum
        threshold <- getOption("evoFE.importance_threshold", 0.001)
        imp[imp < threshold] <- 0
      }
      ind$importances <- imp
    } else {
      ind$importances <- numeric(0)
    }

    if (!is.null(res_model$best_params)) {
      ind$best_params <- res_model$best_params
    }

    # Validation score -> raw_fitness
    if (task == "multiclass") {
      y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
      raw_score <- compute_metric(y_val_encoded, preds, task, metric, num_class)
      ind$val_preds <- preds
      ind$y_val <- y_val_encoded
    } else if (task == "classification") {
      y_val_encoded <- y_val
      raw_score <- compute_metric(y_val_encoded, preds, task, metric)
      ind$val_preds <- preds
      ind$y_val <- y_val_encoded
    } else {
      raw_score <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
      ind$val_preds <- preds
      ind$y_val <- val_fold_feat[[target_col]]
    }

    ind$raw_fitness <- raw_score
    ind$penalty <- 0.0

    # Complexity penalty: discourage long recipes (parsimony pressure)
    if (complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
      n_samp <- if (!is.null(n_samples)) n_samples else if (!is.null(data)) nrow(data) else if (!is.null(shared_full)) nrow(shared_full) else 100
      n_active_raw <- length(res$ind$numeric_cols) + length(res$ind$categorical_cols) + length(res$ind$datetime_cols)
      n_genes <- length(res$ind$genes)
      pen <- compute_complexity_penalty(
        n_genes = n_genes,
        n_samples = n_samp,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        metric = metric,
        task = task,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        n_features = n_active_raw + n_genes
      )
      ind$penalty <- pen
      if (task == "regression") {
        ind$fitness <- if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen)
      } else {
        error_gap <- max(0, min(1.0, 1.0 - raw_score))
        ind$fitness <- raw_score - error_gap * pen
      }
    } else {
      ind$fitness <- raw_score
    }

    ind$holdout_fitness <- NULL

    # Propagate the updated genes (with state)
    ind$genes <- res$ind$genes
  } else {
    # --- Existing CV strategy ---
    use_shared <- !is.null(shared_folds) && !is.null(shared_full)

    if (use_shared) {
      dt <- shared_full
      folds <- if (!is.null(fold_ids)) fold_ids else rep(seq_along(shared_folds), length.out = nrow(dt))
    } else {
      dt <- data.table::as.data.table(data)
      if (is.null(fold_ids)) {
        folds <- .build_cv_folds(dt, cv_folds, cv_strategy, time_col, group_col)
      } else {
        folds <- fold_ids
      }
    }

    unique_folds <- sort(unique(folds))
    k_folds <- length(unique_folds)
    metrics <- rep(NA_real_, k_folds)
    fold_importances <- list()
    last_fit_genes <- NULL

    n_total <- nrow(dt)
    if (task == "multiclass") {
      oof_preds <- matrix(NA_real_, nrow = n_total, ncol = num_class)
      oof_y <- integer(n_total)
    } else {
      oof_preds <- rep(NA_real_, n_total)
      oof_y <- vector(mode = typeof(dt[[target_col]]), length = n_total)
    }

    for (fi in seq_along(unique_folds)) {
      f <- unique_folds[fi]
      if (use_shared) {
        train_fold <- data.table::copy(shared_folds[[f]]$train)
        val_fold <- data.table::copy(shared_folds[[f]]$val)
        val_idx <- if (!is.null(fold_ids)) which(fold_ids == f) else seq_len(nrow(val_fold))
      } else {
        train_idx <- which(folds != f)
        val_idx <- which(folds == f)
        train_fold <- dt[train_idx, ]
        val_fold <- dt[val_idx, ]
      }

      res <- tryCatch(
        {
          apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache, allow_prune = allow_prune)
        },
        error = function(e) {
          NULL
        }
      )

      if (is.null(res)) {
        # This fold failed (e.g. constant column on this split) — skip it.
        # The individual is only killed if every fold fails (handled below).
        next
      }

      train_fold_feat <- res$train
      val_fold_feat <- res$val
      if (!is.null(res$ind)) last_fit_genes <- res$ind$genes

      # Features = original + new
      gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
      features <- c(res$ind$numeric_cols, res$ind$categorical_cols, res$ind$datetime_cols, gene_cols)

      # Convert features to numeric matrix
      has_cat <- length(res$ind$categorical_cols) > 0 || length(res$ind$datetime_cols) > 0 ||
        any(vapply(res$ind$genes, function(g) {
          t_def <- evo_transformers[[g$transformer_name]]
          !is.null(t_def$output_type) && t_def$output_type == "categorical"
        }, logical(1)))
      dt_sub_tr <- train_fold_feat[, features, with = FALSE]
      dt_sub_va <- val_fold_feat[, features, with = FALSE]
      x_train <- .sanitize_feature_matrix(dt_sub_tr)
      x_val <- .sanitize_feature_matrix(dt_sub_va)
      y_train <- train_fold_feat[[target_col]]
      y_val <- val_fold_feat[[target_col]]
      if (task == "multiclass") {
        y_train <- as.integer(factor(y_train, levels = classes)) - 1
        y_val <- as.integer(factor(y_val, levels = classes)) - 1
      } else if (task == "classification") {
        if (is.factor(y_train)) {
          y_train <- as.integer(y_train) - 1L
          y_val <- as.integer(y_val) - 1L
        } else if (is.character(y_train)) {
          y_fac <- as.factor(y_train)
          y_train <- as.integer(y_fac) - 1L
          y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
        } else if (is.logical(y_train)) {
          y_train <- as.integer(y_train)
          y_val <- as.integer(y_val)
        } else if (is.numeric(y_train) && !all(stats::na.omit(y_train) %in% c(0, 1))) {
          y_fac <- as.factor(y_train)
          y_train <- as.integer(y_fac) - 1L
          y_val <- as.integer(factor(y_val, levels = levels(y_fac))) - 1L
        }
      }

      res_model <- train_model(x_train, y_train, x_val,
        y_val = y_val, task = task,
        evaluator = evaluator, threads = threads,
        num_class = num_class, metric = metric,
        verbose = verbose, ...
      )
      preds <- res_model$predictions
      if (!is.null(res_model$importances)) {
        fold_importances[[fi]] <- res_model$importances
      }
      if (!is.null(res_model$best_params)) {
        ind$best_params <- res_model$best_params
      }

      if (task == "multiclass") {
        y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
        metrics[fi] <- compute_metric(y_val_encoded, preds, task, metric, num_class)
        if (length(val_idx) == length(y_val_encoded)) {
          if (is.matrix(preds)) {
            oof_preds[val_idx, ] <- preds
          } else {
            oof_preds[val_idx, ] <- matrix(preds, ncol = num_class, byrow = FALSE)
          }
          oof_y[val_idx] <- y_val_encoded
        }
      } else if (task == "classification") {
        y_val_encoded <- y_val
        metrics[fi] <- compute_metric(y_val_encoded, preds, task, metric)
        if (length(val_idx) == length(preds)) {
          oof_preds[val_idx] <- preds
          oof_y[val_idx] <- y_val_encoded
        }
      } else {
        metrics[fi] <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
        if (length(val_idx) == length(preds)) {
          oof_preds[val_idx] <- preds
          oof_y[val_idx] <- val_fold_feat[[target_col]]
        }
      }

      # Clean up the model if the evaluator provides a cleanup function (e.g. to prevent TF memory leaks)
      if (!is.null(evo_evaluators[[evaluator]]$cleanup_func)) {
        evo_evaluators[[evaluator]]$cleanup_func(res_model$model)
      }
    }

    # Fitness: average over successful folds only; -Inf only when every fold failed.
    finite_metrics <- metrics[!is.na(metrics)]
    raw_score <- if (length(finite_metrics) == 0) -Inf else mean(finite_metrics)

    ind$raw_fitness <- raw_score
    ind$penalty <- 0.0
    ind$val_preds <- oof_preds
    ind$y_val <- oof_y

    # Complexity penalty: discourage long recipes (parsimony pressure)
    if (complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
      n_samp <- if (!is.null(n_samples)) n_samples else if (!is.null(data)) nrow(data) else if (!is.null(shared_full)) nrow(shared_full) else 100
      n_active_raw <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols)
      n_genes <- length(ind$genes)
      pen <- compute_complexity_penalty(
        n_genes = n_genes,
        n_samples = n_samp,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        metric = metric,
        task = task,
        complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        n_features = n_active_raw + n_genes
      )
      ind$penalty <- pen
      if (task == "regression") {
        ind$fitness <- if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen)
      } else {
        error_gap <- max(0, min(1.0, 1.0 - raw_score))
        ind$fitness <- raw_score - error_gap * pen
      }
    } else {
      ind$fitness <- raw_score
    }

    # Aggregate importances across folds
    all_feats <- unique(unlist(lapply(fold_importances, names)))
    if (length(all_feats) > 0) {
      avg_imp <- sapply(all_feats, function(feat) {
        vals <- sapply(fold_importances, function(fold) {
          if (feat %in% names(fold)) fold[[feat]] else 0
        })
        mean(vals)
      })

      imp_sum <- sum(avg_imp, na.rm = TRUE)
      if (imp_sum > 0) {
        avg_imp <- avg_imp / imp_sum
        threshold <- getOption("evoFE.importance_threshold", 0.001)
        avg_imp[avg_imp < threshold] <- 0
      }
      ind$importances <- avg_imp
    } else {
      ind$importances <- numeric(0)
    }

    # Propagate fitted gene states from the last successful fold
    if (!is.null(last_fit_genes)) ind$genes <- last_fit_genes

    ind$holdout_fitness <- NULL
  }

  ind
}

#' Evaluate holdout fitness for an individual
#'
#' @param ind An \code{evo_individual} object.
#' @param data A data.frame or data.table.
#' @param split_ids Character vector of split identifiers.
#' @param shared_splits Optional pre-split data tables.
#' @param target_col Target column name.
#' @param task Task type.
#' @param evaluator Evaluator name.
#' @param threads Thread count.
#' @param state_cache State cache environment.
#' @param classes Class levels.
#' @param num_class Number of classes.
#' @param metric Metric name.
#' @param verbose Verbosity.
#' @param ... Additional arguments.
#' @return An updated \code{evo_individual} with \code{holdout_fitness} evaluated on the holdout partition.
#' @keywords internal
evaluate_holdout_fitness <- function(ind, data, split_ids, shared_splits,
                                     target_col, task, evaluator, threads,
                                     state_cache, classes, num_class, metric = "default",
                                     verbose = FALSE, ...) {
  if (!is.null(shared_splits)) {
    train_fold <- shared_splits$train
    val_fold <- shared_splits$val
    holdout_fold <- shared_splits$holdout
  } else {
    train_fold <- data.table::as.data.table(data[split_ids == "train", ])
    val_fold <- data.table::as.data.table(data[split_ids == "val", ])
    holdout_fold <- if ("holdout" %in% split_ids) data.table::as.data.table(data[split_ids == "holdout", ]) else NULL
  }

  if (is.null(holdout_fold)) {
    ind$holdout_fitness <- NULL
    return(ind)
  }

  train_fold <- data.table::copy(train_fold)
  val_fold <- data.table::copy(val_fold)
  holdout_fold <- data.table::copy(holdout_fold)

  res <- tryCatch(
    {
      apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache)
    },
    error = function(e) NULL
  )

  if (is.null(res)) {
    ind$holdout_fitness <- -Inf
    return(ind)
  }

  train_fold_feat <- res$train
  val_fold_feat <- res$val

  gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(res$ind$numeric_cols, res$ind$categorical_cols, res$ind$datetime_cols, gene_cols)

  x_train <- .sanitize_feature_matrix(train_fold_feat[, features, with = FALSE])
  x_val   <- .sanitize_feature_matrix(val_fold_feat[, features, with = FALSE])
  y_train <- train_fold_feat[[target_col]]
  y_val <- val_fold_feat[[target_col]]
  if (task == "multiclass") {
    y_train <- as.integer(factor(y_train, levels = classes)) - 1
    y_val <- as.integer(factor(y_val, levels = classes)) - 1
  }

  # Crucial distinction: The holdout fold is strictly for testing generalization performance on unseen
  # data, and must NEVER be used to drive parameter tuning. Therefore, we bypass the tuner (e.g.
  # lightgbm_mbo) and train the base evaluator (e.g. lightgbm) directly using the best parameters
  # found during evolution.
  final_evaluator <- evaluator
  eval_entry <- evo_evaluators[[evaluator]]
  if (!is.null(eval_entry) && !is.null(eval_entry$base_evaluator)) {
    final_evaluator <- eval_entry$base_evaluator
  }

  # Merge best_params into ...
  final_args <- utils::modifyList(list(...), as.list(ind$best_params))

  res_model <- do.call(train_model, c(
    list(
      x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task,
      evaluator = final_evaluator, threads = threads,
      num_class = num_class, metric = metric,
      verbose = verbose
    ),
    final_args
  ))

  if (!is.null(res_model$best_params)) {
    ind$best_params <- res_model$best_params
  }

  res_holdout <- tryCatch(
    {
      apply_individual(res$ind, holdout_fold, NULL, NULL, state_cache = state_cache)
    },
    error = function(e) NULL
  )

  if (!is.null(res_holdout)) {
    x_holdout <- .sanitize_feature_matrix(res_holdout$train[, features, with = FALSE])

    evaluator_entry <- evo_evaluators[[evaluator]]
    if (is.null(evaluator_entry)) {
      stop(sprintf(
        "Unknown evaluator '%s'. Registered evaluators are: %s",
        evaluator, paste(names(evo_evaluators), collapse = ", ")
      ))
    }
    preds_holdout <- evaluator_entry$predict_func(res_model$model, x_holdout, task = task)

    if (task == "multiclass") {
      y_holdout_encoded <- as.integer(factor(holdout_fold[[target_col]], levels = classes)) - 1
      if (!is.matrix(preds_holdout)) {
        preds_holdout <- matrix(preds_holdout, ncol = num_class, byrow = TRUE)
      }
      ind$holdout_fitness <- compute_metric(y_holdout_encoded, preds_holdout, task, metric, num_class)
    } else {
      ind$holdout_fitness <- compute_metric(holdout_fold[[target_col]], preds_holdout, task, metric)
    }
  } else {
    ind$holdout_fitness <- -Inf
  }

  # Propagate updated genes
  ind$genes <- res$ind$genes
  ind
}

# --- METRIC COMPUTATION HELPERS ---

compute_auc <- function(y_true, y_pred) {
  if (is.factor(y_true)) {
    y_true <- as.integer(y_true) - 1L
  } else if (is.character(y_true)) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  } else if (is.logical(y_true)) {
    y_true <- as.integer(y_true)
  } else if (is.numeric(y_true) && !all(stats::na.omit(y_true) %in% c(0, 1))) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  }
  n_pos <- sum(y_true == 1, na.rm = TRUE)
  n_neg <- sum(y_true == 0, na.rm = TRUE)
  if (n_pos == 0 || n_neg == 0) {
    return(0.5)
  }
  r <- rank(y_pred)
  u <- sum(r[y_true == 1]) - (as.numeric(n_pos) * (n_pos + 1)) / 2
  u / (as.numeric(n_pos) * n_neg)
}

compute_multiclass_auc <- function(y_true, y_pred_matrix, num_class) {
  if (!is.matrix(y_pred_matrix)) {
    y_pred_matrix <- matrix(y_pred_matrix, ncol = num_class, byrow = FALSE)
  }
  y_idx <- as.integer(y_true)
  if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
    y_true_0 <- y_idx - 1L
  } else {
    y_true_0 <- y_idx
  }
  aucs <- numeric(num_class)
  for (k in 1:num_class) {
    y_true_bin <- as.integer(y_true_0 == (k - 1))
    aucs[k] <- compute_auc(y_true_bin, y_pred_matrix[, k])
  }
  mean(aucs, na.rm = TRUE)
}

compute_f1 <- function(y_true, y_pred) {
  if (is.factor(y_true)) {
    y_true <- as.integer(y_true) - 1L
  } else if (is.character(y_true)) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  } else if (is.logical(y_true)) {
    y_true <- as.integer(y_true)
  } else if (is.numeric(y_true) && !all(stats::na.omit(y_true) %in% c(0, 1))) {
    y_true <- as.integer(as.factor(y_true)) - 1L
  }
  preds <- as.integer(as.numeric(y_pred) >= 0.5)
  tp <- sum(y_true == 1 & preds == 1, na.rm = TRUE)
  fp <- sum(y_true == 0 & preds == 1, na.rm = TRUE)
  fn <- sum(y_true == 1 & preds == 0, na.rm = TRUE)
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
  if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall)
}

compute_mae <- function(y_true, y_pred) {
  mean(abs(as.numeric(y_true) - as.numeric(y_pred)), na.rm = TRUE)
}

compute_metric <- function(y_true, y_pred, task, metric, num_class = NULL) {
  if (is.function(metric)) {
    return(metric(y_true, y_pred))
  }

  metric <- tolower(metric)

  if (task == "classification") {
    if (metric %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      min_loss <- compute_ts_refinement(y_true, y_pred, task = task, is_logits = FALSE)
      return(exp(-min_loss))
    }
    switch(metric,
      auc = compute_auc(y_true, y_pred),
      f1 = compute_f1(y_true, y_pred),
      compute_exp_neg_logloss(y_true, y_pred)
    )
  } else if (task == "multiclass") {
    if (metric %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
      min_loss <- compute_ts_refinement(y_true, y_pred, task = task, num_class = num_class, is_logits = FALSE)
      return(exp(-min_loss))
    }
    switch(metric,
      auc = {
        if (!is.matrix(y_pred)) {
          y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
        }
        compute_multiclass_auc(y_true, y_pred, num_class)
      },
      compute_exp_neg_multiclass_logloss(y_true, y_pred, num_class)
    )
  } else {
    val_score <- switch(metric,
      mae = compute_mae(y_true, y_pred),
      cal_rmse = compute_calibrated_rmse(y_true, y_pred),
      `cal-rmse` = compute_calibrated_rmse(y_true, y_pred),
      cal_mae = compute_calibrated_mae(y_true, y_pred),
      `cal-mae` = compute_calibrated_mae(y_true, y_pred),
      sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2, na.rm = TRUE))
    )
    -val_score
  }
}


#' Temperature Scaled Refinement Metric
#'
#' Computes the Temperature Scaled Refinement (TS-Refinement) metric for binary or multiclass classification.
#' The metric finds the temperature $T$ that minimizes the Laplace-smoothed log-loss of the temperature-scaled prediction margins (logits).
#'
#' @param y_true Numeric vector of true labels (0/1 for classification, or 0 to C-1 for multiclass classification).
#' @param y_pred Numeric vector or matrix of predicted probabilities (or logits, if \code{is_logits = TRUE}).
#' @param task Character. Either \code{"classification"} (binary) or \code{"multiclass"}.
#' @param num_class Integer. Number of classes (required for multiclass).
#' @param alpha Numeric. Laplace smoothing parameter (default is 1).
#' @param is_logits Logical. If \code{TRUE}, the input predictions \code{y_pred} are treated directly as prediction margins (logits). If \code{FALSE}, they are treated as probabilities and converted to logits.
#' @return Numeric. The minimized smoothed log-loss.
#' @export
compute_ts_refinement <- function(y_true, y_pred, task = "classification", num_class = NULL, alpha = 1, is_logits = FALSE) {
  if (!task %in% c("classification", "multiclass")) {
    stop("TS-Refinement metric is only supported for 'classification' and 'multiclass' tasks.")
  }

  if (task == "classification") {
    if (is.factor(y_true)) {
      y_true <- as.integer(y_true) - 1L
    } else if (is.character(y_true)) {
      y_true <- as.integer(as.factor(y_true)) - 1L
    } else if (is.logical(y_true)) {
      y_true <- as.integer(y_true)
    } else if (is.numeric(y_true) && !all(stats::na.omit(y_true) %in% c(0, 1))) {
      y_true <- as.integer(as.factor(y_true)) - 1L
    }
    y_pred <- as.numeric(y_pred)
    if (is_logits) {
      z <- y_pred
    } else {
      # Reconstruct logits (margins) from probabilities
      p <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
      z <- log(p / (1 - p))
    }
    # Clamp infinite values and impute NA/NaN
    z[is.na(z) | is.nan(z)] <- 0
    z <- pmax(pmin(z, 35), -35)

    # Laplace smooth the labels based on true class count
    N1 <- sum(y_true == 1, na.rm = TRUE)
    N0 <- sum(y_true == 0, na.rm = TRUE)
    N_true <- ifelse(y_true == 1, N1, N0)

    y_smooth <- ifelse(y_true == 1,
      (N_true + alpha) / (N_true + 2 * alpha),
      alpha / (N_true + 2 * alpha)
    )

    obj_fn <- function(temp) {
      probs_T <- 1 / (1 + exp(-z / temp))
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
      ll <- -mean(y_smooth * log(probs_T) + (1 - y_smooth) * log(1 - probs_T), na.rm = TRUE)
      ll
    }

    opt <- stats::optimize(f = obj_fn, interval = c(0.001, 10))
    best_temp <- opt$minimum

    # Return the un-smoothed log-loss (Option B) at the optimal temperature
    probs_T <- 1 / (1 + exp(-z / best_temp))
    probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
    ll_unsmoothed <- -mean(y_true * log(probs_T) + (1 - y_true) * log(1 - probs_T), na.rm = TRUE)
    return(ll_unsmoothed)
  } else if (task == "multiclass") {
    if (is.null(num_class)) {
      stop("num_class must be specified for multiclass TS-Refinement.")
    }

    if (!is.matrix(y_pred)) {
      y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
    }

    y_idx <- as.integer(y_true)
    if (min(y_idx, na.rm = TRUE) >= 1 && max(y_idx, na.rm = TRUE) <= num_class) {
      y_true_0 <- y_idx - 1L
    } else {
      y_true_0 <- y_idx
    }
    y_true_0 <- pmax(0L, pmin(as.integer(y_true_0), as.integer(num_class - 1L)))

    if (is_logits) {
      z <- y_pred
    } else {
      p <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
      z <- log(p)
    }
    # Clamp infinite values and impute NA/NaN
    z[is.na(z) | is.nan(z)] <- 0
    z <- pmax(pmin(z, 35), -35)

    # Laplace smooth labels based on class-count sweep formulation
    n <- length(y_true_0)
    N_vec <- tabulate(y_true_0 + 1, nbins = num_class)
    N_k_row <- N_vec[y_true_0 + 1]

    true_target <- (N_k_row + alpha) / (N_k_row + 2 * alpha)
    leftover_mass <- alpha / (N_k_row + 2 * alpha)

    denom <- n - N_k_row
    mass_per_item <- ifelse(denom == 0, 0, leftover_mass / pmax(1, denom))

    N_mat <- matrix(N_vec, nrow = n, ncol = num_class, byrow = TRUE)
    y_smooth <- sweep(N_mat, 1, mass_per_item, "*")
    y_smooth[cbind(seq_len(n), y_true_0 + 1)] <- true_target

    obj_fn <- function(temp) {
      z_scaled <- z / temp
      z_max <- apply(z_scaled, 1, max)
      z_stable <- z_scaled - z_max
      exp_z <- exp(z_stable)
      sum_exp_z <- rowSums(exp_z)
      probs_T <- exp_z / sum_exp_z
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)

      ll <- -mean(rowSums(y_smooth * log(probs_T)), na.rm = TRUE)
      ll
    }

    opt <- stats::optimize(f = obj_fn, interval = c(0.001, 10))
    best_temp <- opt$minimum

    # Return the un-smoothed log-loss (Option B) at the optimal temperature
    z_scaled <- z / best_temp
    z_max <- apply(z_scaled, 1, max)
    z_stable <- z_scaled - z_max
    exp_z <- exp(z_stable)
    sum_exp_z <- rowSums(exp_z)
    probs_T <- exp_z / sum_exp_z
    probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)

    y_hard <- matrix(0, nrow = n, ncol = num_class)
    y_hard[cbind(seq_len(n), y_true_0 + 1)] <- 1
    ll_unsmoothed <- -mean(rowSums(y_hard * log(probs_T)), na.rm = TRUE)
    return(ll_unsmoothed)
  }
}

#' Compute Calibrated RMSE
#'
#' Computes the Root Mean Squared Error (RMSE) of y_pred after optimal linear post-calibration.
#' Mathematically equivalent to SD(y_true) * sqrt(1 - R^2) where R is the Pearson correlation.
#'
#' @param y_true Numeric vector of true target values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric calibrated RMSE score.
#' @export
compute_calibrated_rmse <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)

  if (length(y_true) <= 1L) {
    return(0.0)
  }

  sd_true <- stats::sd(y_true)
  sd_pred <- stats::sd(y_pred)

  if (is.na(sd_true) || sd_true < 1e-9) {
    return(0.0)
  }

  if (is.na(sd_pred) || sd_pred < 1e-9) {
    return(sqrt(mean((y_true - mean(y_pred, na.rm = TRUE))^2, na.rm = TRUE)))
  }

  r <- stats::cor(y_pred, y_true, use = "complete.obs")
  if (is.na(r)) {
    return(sqrt(mean((y_true - y_pred)^2, na.rm = TRUE)))
  }

  sd_true * sqrt(pmax(0.0, 1.0 - r^2))
}

#' Compute Calibrated MAE
#'
#' Computes the Mean Absolute Error (MAE) of y_pred after optimal L1 linear post-calibration.
#' Finds intercept 'a' and slope 'b' minimizing Mean(|y_true - (a + b * y_pred)|) via 2D optimization.
#'
#' @param y_true Numeric vector of true target values.
#' @param y_pred Numeric vector of predicted values.
#' @return Numeric calibrated MAE score.
#' @export
compute_calibrated_mae <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)

  if (length(y_true) == 0L) {
    return(0.0)
  }

  valid <- !is.na(y_true) & !is.na(y_pred)
  y_true <- y_true[valid]
  y_pred <- y_pred[valid]

  if (length(y_true) <= 1L) {
    return(0.0)
  }

  obj_fn <- function(par) {
    a <- par[1]
    b <- par[2]
    mean(abs(y_true - (a + b * y_pred)))
  }

  cov_y <- stats::cov(y_true, y_pred)
  var_pred <- stats::var(y_pred)

  start_b <- if (!is.na(var_pred) && var_pred > 1e-9) cov_y / var_pred else 1.0
  start_a <- mean(y_true) - start_b * mean(y_pred)

  opt <- tryCatch(
    {
      stats::optim(
        par = c(start_a, start_b),
        fn = obj_fn,
        method = "Nelder-Mead"
      )
    },
    error = function(e) {
      list(value = mean(abs(y_true - y_pred)))
    }
  )

  opt$value
}
