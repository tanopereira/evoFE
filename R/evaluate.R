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
        if (!is.null(state_cache) && !is.null(target_col)) {
          if (is.null(data_hash)) {
            data_hash <- digest::digest(train_data[[target_col]], algo = "xxhash64")
          }
          cache_key <- digest::digest(paste0(gene_to_state_formula(gene), "_", data_hash), algo = "md5", serialize = FALSE)
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
    
    # Reject constant columns (0 variance)
    if (!is.null(target_col) && length(unique(new_col_train[!is.na(new_col_train)])) <= 1) {
      stop("Constant column generated")
    }
    
    if (out_type == "categorical") {
      new_col_train <- as.factor(new_col_train)
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
    res <- tryCatch({
      if (!all(gene$input_cols %in% names(dt_train))) {
        stop("Input column missing")
      }
      apply_gene(gene, dt_train, dt_val, target_col, state_cache = state_cache, data_hash = pre_hash)
    }, error = function(e) {
      NULL
    })
    
    if (is.null(res) || !is.null(res$skip)) {
      if (allow_prune) {
        next
      } else {
        return(NULL)
      }
    }
    
    dt_train <- res$train
    dt_val <- res$val
    new_genes <- c(new_genes, list(res$gene))
  }
  
  ind$genes <- new_genes
  list(train = dt_train, val = dt_val, ind = ind)
}

compute_exp_neg_logloss <- function(y_true, y_pred) {
  p <- pmax(pmin(y_pred, 1 - 1e-15), 1e-15)
  ll <- -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
  exp(-ll)
}

compute_exp_neg_multiclass_logloss <- function(y_true, y_pred, num_class) {
  n <- length(y_true)
  if (!is.matrix(y_pred)) {
    y_pred <- matrix(y_pred, ncol = num_class, byrow = TRUE)
  }
  probs <- y_pred[cbind(1:n, y_true + 1)]
  probs <- pmax(pmin(probs, 1 - 1e-15), 1e-15)
  ll <- -mean(log(probs))
  exp(-ll)
}


#' Evaluate the fitness of an individual
#'
#' @param ind An evo_individual object.
#' @param data A data.frame or data.table containing the dataset.
#' @param target_col Name of the target column.
#' @param task "classification" or "regression".
#' @param cv_folds Number of cross-validation folds.
#' @param evaluation_strategy Character string, either "cv" (cross-validation) or "split" (train/validation split).
#' @param split_ids Optional vector of pre-defined split assignments (e.g. \code{c("train", "train", "val", "holdout", "train")}). Must have the same length as the number of rows in \code{data} and contain only "train", "val", or "holdout" labels.
#' @param shared_splits Optional list of shared data.table splits for in-place caching.
#' @param evaluator The ML model to use ("lightgbm", "xgboost", "catboost", or a custom registered evaluator name).
#' @param fold_ids Optional vector of pre-defined fold assignments.
#' @param shared_folds Optional list of shared data.table CV folds for in-place caching.
#' @param shared_full Optional data.table of the full dataset for in-place caching.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param threads Number of threads to use for parallel execution (default 2)
#' @param metric The metric to optimize ("default", "auc", "f1", "mae", or a custom function).
#' @param verbose Logical indicating if progress should be printed.
#' @param allow_prune Logical. If TRUE, genes that fail application are skipped instead of failing the entire individual.
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
                             metric = "default", verbose = FALSE, allow_prune = TRUE, ...) {
  if (!is.na(ind$fitness)) return(ind)
  
  num_class <- NULL
  classes <- NULL
  if (task == "multiclass") {
    target_factor <- as.factor(data[[target_col]])
    classes <- levels(target_factor)
    num_class <- length(classes)
  }
  
  if (evaluation_strategy == "split") {
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
    res <- tryCatch({
      apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache, allow_prune = allow_prune)
    }, error = function(e) {
      NULL
    })
    
    if (is.null(res)) {
      # Lethal mutation: invalid gene dependency graph
      ind$fitness <- -Inf
      ind$holdout_fitness <- NULL
      return(ind)
    }
    
    train_fold_feat <- res$train
    val_fold_feat <- res$val
    
    # Features = original + new
    gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
    features <- c(res$ind$numeric_cols, res$ind$categorical_cols, gene_cols)
    
    # Convert features to numeric matrix
    x_train <- data.matrix(train_fold_feat[, features, with = FALSE])
    x_val <- data.matrix(val_fold_feat[, features, with = FALSE])
    x_train[!is.finite(x_train)] <- NA
    x_val[!is.finite(x_val)] <- NA
    y_train <- train_fold_feat[[target_col]]
    y_val <- val_fold_feat[[target_col]]
    if (task == "multiclass") {
      y_train <- as.integer(factor(y_train, levels = classes)) - 1
      y_val <- as.integer(factor(y_val, levels = classes)) - 1
    }
    
    res_model <- train_model(x_train, y_train, x_val, y_val = y_val, task = task,
                              evaluator = evaluator, threads = threads,
                              num_class = num_class, metric = metric,
                              verbose = verbose, ...)
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
    
    # Validation score -> fitness
    if (task == "multiclass") {
      y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
      ind$fitness <- compute_metric(y_val_encoded, preds, task, metric, num_class)
    } else {
      ind$fitness <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
    }
    
    ind$holdout_fitness <- NULL
    
    # Propagate the updated genes (with state)
    ind$genes <- res$ind$genes
    
  } else {
    # --- Existing CV strategy ---
    use_shared <- !is.null(shared_folds) && !is.null(shared_full)
    
    if (use_shared) {
      dt <- shared_full
    } else {
      dt <- data.table::as.data.table(data)
      if (is.null(fold_ids)) {
        folds <- cut(seq(1, nrow(dt)), breaks = cv_folds, labels = FALSE)
        folds <- sample(folds)
      } else {
        folds <- fold_ids
      }
    }
    
    metrics <- rep(NA_real_, cv_folds)
    fold_importances <- list()
    
    for (f in 1:cv_folds) {
      if (use_shared) {
        train_fold <- data.table::copy(shared_folds[[f]]$train)
        val_fold <- data.table::copy(shared_folds[[f]]$val)
      } else {
        train_idx <- which(folds != f)
        val_idx <- which(folds == f)
        train_fold <- dt[train_idx, ]
        val_fold <- dt[val_idx, ]
      }
      
      res <- tryCatch({
        apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache, allow_prune = allow_prune)
      }, error = function(e) {
        NULL
      })
      
      if (is.null(res)) {
        # This fold failed (e.g. constant column on this split) — skip it.
        # The individual is only killed if every fold fails (handled below).
        next
      }
      
      train_fold_feat <- res$train
      val_fold_feat <- res$val
      
      # Features = original + new
      gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
      features <- c(res$ind$numeric_cols, res$ind$categorical_cols, gene_cols)
      
      # Convert features to numeric matrix
      x_train <- data.matrix(train_fold_feat[, features, with = FALSE])
      x_val <- data.matrix(val_fold_feat[, features, with = FALSE])
      x_train[!is.finite(x_train)] <- NA
      x_val[!is.finite(x_val)] <- NA
      y_train <- train_fold_feat[[target_col]]
      y_val <- val_fold_feat[[target_col]]
      if (task == "multiclass") {
        y_train <- as.integer(factor(y_train, levels = classes)) - 1
        y_val <- as.integer(factor(y_val, levels = classes)) - 1
      }
      
      res_model <- train_model(x_train, y_train, x_val, y_val = y_val, task = task,
                                evaluator = evaluator, threads = threads,
                                num_class = num_class, metric = metric,
                                verbose = verbose, ...)
      preds <- res_model$predictions
      if (!is.null(res_model$importances)) {
        fold_importances[[f]] <- res_model$importances
      }
      if (!is.null(res_model$best_params)) {
        ind$best_params <- res_model$best_params
      }
      
      if (task == "multiclass") {
        y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
        metrics[f] <- compute_metric(y_val_encoded, preds, task, metric, num_class)
      } else {
        metrics[f] <- compute_metric(val_fold_feat[[target_col]], preds, task, metric)
      }
    }
    
    # Fitness: average over successful folds only; -Inf only when every fold failed.
    finite_metrics <- metrics[!is.na(metrics)]
    ind$fitness <- if (length(finite_metrics) == 0) -Inf else mean(finite_metrics)
    
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
    
    ind$holdout_fitness <- NULL
  }
  
  ind
}

#' Evaluate holdout fitness for an individual
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
  
  res <- tryCatch({
    apply_individual(ind, train_fold, val_fold, target_col, state_cache = state_cache)
  }, error = function(e) NULL)
  
  if (is.null(res)) {
    ind$holdout_fitness <- -Inf
    return(ind)
  }
  
  train_fold_feat <- res$train
  val_fold_feat <- res$val
  
  gene_cols <- if (length(res$ind$genes) > 0) vapply(res$ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(res$ind$numeric_cols, res$ind$categorical_cols, gene_cols)
  
  x_train <- data.matrix(train_fold_feat[, features, with = FALSE])
  x_val <- data.matrix(val_fold_feat[, features, with = FALSE])
  x_train[!is.finite(x_train)] <- NA
  x_val[!is.finite(x_val)] <- NA
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
    list(x_train = x_train, y_train = y_train, x_val = x_val, y_val = y_val, task = task,
         evaluator = final_evaluator, threads = threads,
         num_class = num_class, metric = metric,
         verbose = verbose),
    final_args
  ))
  
  if (!is.null(res_model$best_params)) {
    ind$best_params <- res_model$best_params
  }
  
  res_holdout <- tryCatch({
    apply_individual(res$ind, holdout_fold, NULL, NULL, state_cache = state_cache)
  }, error = function(e) NULL)
  
  if (!is.null(res_holdout)) {
    x_holdout <- data.matrix(res_holdout$train[, features, with = FALSE])
    x_holdout[!is.finite(x_holdout)] <- NA
    
    evaluator_entry <- evo_evaluators[[evaluator]]
    if (is.null(evaluator_entry)) {
      stop(sprintf("Unknown evaluator '%s'. Registered evaluators are: %s", 
                   evaluator, paste(names(evo_evaluators), collapse = ", ")))
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
  n_pos <- sum(y_true == 1)
  n_neg <- sum(y_true == 0)
  if (n_pos == 0 || n_neg == 0) return(0.5)
  r <- rank(y_pred)
  u <- sum(r[y_true == 1]) - (as.numeric(n_pos) * (n_pos + 1)) / 2
  u / (as.numeric(n_pos) * n_neg)
}

compute_multiclass_auc <- function(y_true, y_pred_matrix, num_class) {
  aucs <- numeric(num_class)
  for (k in 1:num_class) {
    y_true_bin <- as.integer(y_true == (k - 1))
    aucs[k] <- compute_auc(y_true_bin, y_pred_matrix[, k])
  }
  mean(aucs)
}

compute_f1 <- function(y_true, y_pred) {
  preds <- as.integer(y_pred >= 0.5)
  tp <- sum(y_true == 1 & preds == 1)
  fp <- sum(y_true == 0 & preds == 1)
  fn <- sum(y_true == 1 & preds == 0)
  precision <- if (tp + fp == 0) 0 else tp / (tp + fp)
  recall <- if (tp + fn == 0) 0 else tp / (tp + fn)
  if (precision + recall == 0) 0 else 2 * (precision * recall) / (precision + recall)
}

compute_mae <- function(y_true, y_pred) {
  mean(abs(y_true - y_pred))
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
          y_pred <- matrix(y_pred, ncol = num_class, byrow = TRUE)
        }
        compute_multiclass_auc(y_true, y_pred, num_class)
      },
      compute_exp_neg_multiclass_logloss(y_true, y_pred, num_class)
    )
  } else {
    val_score <- switch(metric,
      mae = compute_mae(y_true, y_pred),
      sqrt(mean((y_true - y_pred)^2))
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
    N1 <- sum(y_true == 1)
    N0 <- sum(y_true == 0)
    N_true <- ifelse(y_true == 1, N1, N0)
    
    y_smooth <- ifelse(y_true == 1,
                       (N_true + alpha) / (N_true + 2 * alpha),
                       alpha            / (N_true + 2 * alpha))
    
    obj_fn <- function(temp) {
      probs_T <- 1 / (1 + exp(-z / temp))
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
      ll <- -mean(y_smooth * log(probs_T) + (1 - y_smooth) * log(1 - probs_T))
      ll
    }
    
    opt <- stats::optimize(f = obj_fn, interval = c(0.001, 10))
    best_temp <- opt$minimum
    
    # Return the un-smoothed log-loss (Option B) at the optimal temperature
    probs_T <- 1 / (1 + exp(-z / best_temp))
    probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
    ll_unsmoothed <- -mean(y_true * log(probs_T) + (1 - y_true) * log(1 - probs_T))
    return(ll_unsmoothed)
    
  } else if (task == "multiclass") {
    if (is.null(num_class)) {
      stop("num_class must be specified for multiclass TS-Refinement.")
    }
    
    if (!is.matrix(y_pred)) {
      y_pred <- matrix(y_pred, ncol = num_class, byrow = FALSE)
    }
    
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
    n <- length(y_true)
    N_vec <- tabulate(y_true + 1, nbins = num_class)
    N_k_row <- N_vec[y_true + 1]
    
    true_target <- (N_k_row + alpha) / (N_k_row + 2 * alpha)
    leftover_mass <- alpha / (N_k_row + 2 * alpha)
    
    denom <- n - N_k_row
    mass_per_item <- ifelse(denom == 0, 0, leftover_mass / denom)
    
    N_mat <- matrix(N_vec, nrow = n, ncol = num_class, byrow = TRUE)
    y_smooth <- sweep(N_mat, 1, mass_per_item, "*")
    y_smooth[cbind(1:n, y_true + 1)] <- true_target
    
    obj_fn <- function(temp) {
      z_scaled <- z / temp
      z_max <- apply(z_scaled, 1, max)
      z_stable <- z_scaled - z_max
      exp_z <- exp(z_stable)
      sum_exp_z <- rowSums(exp_z)
      probs_T <- exp_z / sum_exp_z
      probs_T <- pmax(pmin(probs_T, 1 - 1e-15), 1e-15)
      
      ll <- -mean(rowSums(y_smooth * log(probs_T)))
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
    y_hard[cbind(1:n, y_true + 1)] <- 1
    ll_unsmoothed <- -mean(rowSums(y_hard * log(probs_T)))
    return(ll_unsmoothed)
  }
}



