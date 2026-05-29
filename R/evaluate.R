#' Apply a single gene to a dataset
#'
#' @param gene A gene list representing a feature transformation.
#' @param train_data A data.frame or data.table representing the training data.
#' @param val_data Optional validation data.frame or data.table.
#' @param target_col Name of the target column.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @export
apply_gene <- function(gene, train_data, val_data = NULL, target_col = NULL, state_cache = NULL) {
  t_def <- evo_transformers[[gene$transformer_name]]
  
  col_exists_train <- gene$output_col %in% names(train_data)
  col_exists_val <- if (!is.null(val_data)) gene$output_col %in% names(val_data) else TRUE
  
  if (col_exists_train && col_exists_val && !is.null(gene$state)) {
    return(list(train = train_data, val = val_data, gene = gene))
  }
  
  state <- NULL
  has_cached_state <- FALSE
  if (!is.null(state_cache)) {
    cache_key <- digest::digest(gene_to_state_formula(gene), algo = "md5", serialize = FALSE)
    if (exists(cache_key, envir = state_cache)) {
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
        if (!is.null(state_cache)) {
          cache_key <- digest::digest(gene_to_state_formula(gene), algo = "md5", serialize = FALSE)
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
#' @export
apply_individual <- function(ind, train_data, val_data = NULL, target_col = NULL, state_cache = NULL) {
  dt_train <- if (data.table::is.data.table(train_data)) train_data else data.table::as.data.table(train_data)
  dt_val <- if (!is.null(val_data)) {
    if (data.table::is.data.table(val_data)) val_data else data.table::as.data.table(val_data)
  } else {
    NULL
  }
  
  new_genes <- list()
  for (gene in ind$genes) {
    res <- tryCatch({
      if (!all(gene$input_cols %in% names(dt_train))) {
        stop("Input column missing")
      }
      apply_gene(gene, dt_train, dt_val, target_col, state_cache = state_cache)
    }, error = function(e) {
      list(skip = TRUE)
    })
    
    if (is.null(res$skip)) {
      dt_train <- res$train
      dt_val <- res$val
      new_genes <- c(new_genes, list(res$gene))
    }
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
#' @param evaluator The ML model to use ("lightgbm" or "xgboost").
#' @param fold_ids Optional vector of pre-defined fold assignments.
#' @param shared_folds Optional list of shared data.table CV folds for in-place caching.
#' @param shared_full Optional data.table of the full dataset for in-place caching.
#' @param state_cache Optional environment to cache full-dataset fitted states of stateful transformers.
#' @param threads Number of threads to use for parallel execution (default 8)
#' @export
evaluate_fitness <- function(ind, data, target_col, task = "classification", 
                             cv_folds = 3, evaluation_strategy = "cv",
                             split_ids = NULL, shared_splits = NULL,
                             evaluator = "lightgbm", fold_ids = NULL, 
                             shared_folds = NULL, shared_full = NULL, 
                             state_cache = NULL, threads = 8) {
  if (!is.na(ind$fitness)) return(ind)
  
  num_class <- NULL
  classes <- NULL
  if (task == "multiclass") {
    target_factor <- as.factor(data[[target_col]])
    classes <- levels(target_factor)
    num_class <- length(classes)
  }
  
  if (evaluation_strategy == "split") {
    # Train / Validation / Holdout split strategy
    if (!is.null(shared_splits)) {
      train_fold <- shared_splits$train
      val_fold <- shared_splits$val
      holdout_fold <- shared_splits$holdout # might be NULL
    } else {
      train_fold <- data.table::as.data.table(data[split_ids == "train", ])
      val_fold <- data.table::as.data.table(data[split_ids == "val", ])
      holdout_fold <- if ("holdout" %in% split_ids) data.table::as.data.table(data[split_ids == "holdout", ]) else NULL
    }
    
    # Copy train/val/holdout folds so we can modify them when applying recipe
    train_fold <- data.table::copy(train_fold)
    val_fold <- data.table::copy(val_fold)
    if (!is.null(holdout_fold)) holdout_fold <- data.table::copy(holdout_fold)
    
    # Apply genes
    res <- tryCatch({
      apply_individual(ind, train_fold, val_fold, target_col)
    }, error = function(e) {
      NULL
    })
    
    if (is.null(res)) {
      # Lethal mutation: invalid gene dependency graph
      ind$fitness <- if (task == "classification" || task == "multiclass") -Inf else -Inf
      ind$holdout_fitness <- -Inf
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
    if (task == "multiclass") {
      y_train <- as.integer(factor(y_train, levels = classes)) - 1
    }
    
    res_model <- train_model(x_train, y_train, x_val, task = task,
                              evaluator = evaluator, threads = threads,
                              num_class = num_class)
    preds <- res_model$predictions
    
    # Store importances (single fold)
    if (!is.null(res_model$importances)) {
      ind$importances <- res_model$importances
    } else {
      ind$importances <- numeric(0)
    }
    
    # Validation score -> fitness
    if (task == "classification") {
      val_score <- compute_exp_neg_logloss(val_fold_feat[[target_col]], preds)
      ind$fitness <- val_score
    } else if (task == "multiclass") {
      y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
      val_score <- compute_exp_neg_multiclass_logloss(y_val_encoded, preds, num_class)
      ind$fitness <- val_score
    } else {
      val_score <- sqrt(mean((val_fold_feat[[target_col]] - preds)^2))
      ind$fitness <- -val_score
    }
    
    # Optional holdout evaluation
    if (!is.null(holdout_fold)) {
      res_holdout <- tryCatch({
        apply_individual(res$ind, holdout_fold, NULL, NULL)
      }, error = function(e) {
        NULL
      })
      
      if (!is.null(res_holdout)) {
        x_holdout <- data.matrix(res_holdout$train[, features, with = FALSE])
        x_holdout[!is.finite(x_holdout)] <- NA
        
        # Predict on holdout using the trained model
        if (evaluator == "lightgbm") {
          preds_holdout <- stats::predict(res_model$model, x_holdout)
        } else if (evaluator == "xgboost") {
          dmatrix_holdout <- xgboost::xgb.DMatrix(data = x_holdout)
          preds_holdout <- stats::predict(res_model$model, dmatrix_holdout)
        }
        
        if (task == "classification") {
          ind$holdout_fitness <- compute_exp_neg_logloss(holdout_fold[[target_col]], preds_holdout)
        } else if (task == "multiclass") {
          y_holdout_encoded <- as.integer(factor(holdout_fold[[target_col]], levels = classes)) - 1
          if (!is.matrix(preds_holdout)) {
            preds_holdout <- matrix(preds_holdout, ncol = num_class, byrow = TRUE)
          }
          ind$holdout_fitness <- compute_exp_neg_multiclass_logloss(y_holdout_encoded, preds_holdout, num_class)
        } else {
          holdout_rmse <- sqrt(mean((holdout_fold[[target_col]] - preds_holdout)^2))
          ind$holdout_fitness <- -holdout_rmse
        }
      } else {
        ind$holdout_fitness <- -Inf
      }
    } else {
      ind$holdout_fitness <- NULL
    }
    
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
    
    metrics <- numeric(cv_folds)
    fold_importances <- list()
    
    for (f in 1:cv_folds) {
      if (use_shared) {
        train_fold <- shared_folds[[f]]$train
        val_fold <- shared_folds[[f]]$val
      } else {
        train_idx <- which(folds != f)
        val_idx <- which(folds == f)
        train_fold <- data.table::copy(dt[train_idx, ])
        val_fold <- data.table::copy(dt[val_idx, ])
      }
      
      # Apply genes
      res <- tryCatch({
        apply_individual(ind, train_fold, val_fold, target_col)
      }, error = function(e) {
        NULL
      })
      
      if (is.null(res)) {
        # Lethal mutation: invalid gene dependency graph
        metrics[f] <- if (task == "classification" || task == "multiclass") -Inf else Inf
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
      if (task == "multiclass") {
        y_train <- as.integer(factor(y_train, levels = classes)) - 1
      }
      
      res_model <- train_model(x_train, y_train, x_val, task = task,
                                evaluator = evaluator, threads = threads,
                                num_class = num_class)
      preds <- res_model$predictions
      if (!is.null(res_model$importances)) {
        fold_importances[[f]] <- res_model$importances
      }
      
      if (task == "classification") {
        metrics[f] <- compute_exp_neg_logloss(val_fold_feat[[target_col]], preds)
      } else if (task == "multiclass") {
        y_val_encoded <- as.integer(factor(val_fold_feat[[target_col]], levels = classes)) - 1
        metrics[f] <- compute_exp_neg_multiclass_logloss(y_val_encoded, preds, num_class)
      } else {
        metrics[f] <- sqrt(mean((val_fold_feat[[target_col]] - preds)^2))
      }
    }
    
    # Fitness: higher is better
    if (task == "classification" || task == "multiclass") {
      ind$fitness <- mean(metrics)
    } else {
      ind$fitness <- -mean(metrics)
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
      ind$importances <- avg_imp
    } else {
      ind$importances <- numeric(0)
    }
    
    ind$holdout_fitness <- NULL
  }
  
  ind
}

