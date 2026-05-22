#' Apply a single gene to a dataset
#' @export
apply_gene <- function(gene, train_data, val_data = NULL, target_col = NULL) {
  t_def <- evo_transformers[[gene$transformer_name]]
  
  state <- NULL
  # If we are fitting (target_col provided) and it's stateful
  if (!is.null(t_def$fit_func) && !is.null(target_col)) {
    state <- t_def$fit_func(train_data, gene, target_col)
    gene$state <- state
  } else if (!is.null(gene$state)) {
    # If we are predicting
    state <- gene$state
  }
  
  # Apply to train
  new_col_train <- t_def$apply_func(train_data, gene, state)
  
  # Reject constant columns (0 variance)
  if (length(unique(new_col_train[!is.na(new_col_train)])) <= 1) {
    stop("Constant column generated")
  }
  
  train_data[[gene$output_col]] <- new_col_train
  
  # Apply to val
  if (!is.null(val_data)) {
    new_col_val <- t_def$apply_func(val_data, gene, state)
    val_data[[gene$output_col]] <- new_col_val
  }
  
  list(train = train_data, val = val_data, gene = gene)
}

#' Apply an entire individual's recipe to data
#' @export
apply_individual <- function(ind, train_data, val_data = NULL, target_col = NULL) {
  dt_train <- data.table::as.data.table(train_data)
  dt_val <- if (!is.null(val_data)) data.table::as.data.table(val_data) else NULL
  
  new_genes <- list()
  for (gene in ind$genes) {
    res <- apply_gene(gene, dt_train, dt_val, target_col)
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

#' Evaluate the fitness of an individual
#' @export
evaluate_fitness <- function(ind, data, target_col, task = "classification", cv_folds = 3, evaluator = "lightgbm", fold_ids = NULL) {
  if (!is.na(ind$fitness)) return(ind)
  
  dt <- data.table::as.data.table(data)
  
  # K-Fold CV
  if (is.null(fold_ids)) {
    folds <- cut(seq(1, nrow(dt)), breaks = cv_folds, labels = FALSE)
    folds <- sample(folds)
  } else {
    folds <- fold_ids
  }
  
  metrics <- numeric(cv_folds)
  fold_importances <- list()
  
  for (f in 1:cv_folds) {
    train_idx <- which(folds != f)
    val_idx <- which(folds == f)
    
    train_fold <- dt[train_idx, ]
    val_fold <- dt[val_idx, ]
    
    # Apply genes
    res <- tryCatch({
      apply_individual(ind, train_fold, val_fold, target_col)
    }, error = function(e) {
      NULL
    })
    
    if (is.null(res)) {
      # Lethal mutation: invalid gene dependency graph
      metrics[f] <- if (task == "classification") -Inf else Inf
      next
    }
    
    train_fold_feat <- res$train
    val_fold_feat <- res$val
    
    # Features = original + new
    gene_cols <- if (length(ind$genes) > 0) vapply(ind$genes, function(g) g$output_col, character(1)) else character(0)
    features <- c(ind$numeric_cols, ind$categorical_cols, gene_cols)
    
    # Ensure all features are numeric for LightGBM
    # In a full package, we'd handle categorical features properly.
    # Here we convert everything to numeric as a simplification.
    x_train <- as.matrix(sapply(train_fold_feat[, features, with = FALSE], as.numeric))
    x_val <- as.matrix(sapply(val_fold_feat[, features, with = FALSE], as.numeric))
    y_train <- train_fold_feat[[target_col]]
    
    if (evaluator == "lightgbm") {
      dtrain <- lightgbm::lgb.Dataset(
        data = x_train,
        label = y_train
      )
      
      params <- list(
        objective = if(task == "classification") "binary" else "regression",
        metric = if(task == "classification") "binary_logloss" else "rmse",
        num_leaves = 15,
        learning_rate = 0.1,
        verbose = -1,
        num_threads = 1
      )
      
      capture.output({
        model <- lightgbm::lgb.train(
          params = params,
          data = dtrain,
          nrounds = 50,
          verbose = -1
        )
      })
      preds <- stats::predict(model, x_val)
      imp <- tryCatch({
        lightgbm::lgb.importance(model, percentage = TRUE)
      }, error = function(e) {
        data.frame(Feature = character(), Gain = numeric())
      })
      if (nrow(imp) > 0) {
        fold_importances[[f]] <- stats::setNames(imp$Gain, imp$Feature)
      }
      
      # Let R handle garbage collection naturally to prevent macOS pointer race conditions
      rm(model, dtrain)
      
    } else if (evaluator == "xgboost") {
      dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train)
      dval <- xgboost::xgb.DMatrix(data = x_val)
      
      params <- list(
        objective = if(task == "classification") "binary:logistic" else "reg:squarederror",
        eval_metric = if(task == "classification") "logloss" else "rmse",
        nthread = 1,
        max_depth = 4,
        eta = 0.1
      )
      
      capture.output({
        model <- xgboost::xgb.train(
          params = params,
          data = dtrain,
          nrounds = 50,
          verbose = 0
        )
      })
      preds <- stats::predict(model, dval)
      imp <- xgboost::xgb.importance(model = model)
      if (nrow(imp) > 0) {
        fold_importances[[f]] <- stats::setNames(imp$Gain, imp$Feature)
      }
      
      # Let R handle garbage collection naturally to prevent macOS pointer race conditions
      rm(model, dtrain, dval)
      
    } else {
      stop("Unknown evaluator specified.")
    }
    
    if (task == "classification") {
      metrics[f] <- compute_exp_neg_logloss(val_fold_feat[[target_col]], preds)
    } else {
      metrics[f] <- sqrt(mean((val_fold_feat[[target_col]] - preds)^2))
    }
  }
  
  # Fitness: higher is better
  if (task == "classification") {
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
  
  # Finally, fit transformers on full data to capture state
  res_full <- tryCatch({
    apply_individual(ind, dt, NULL, target_col)
  }, error = function(e) {
    NULL
  })
  
  if (!is.null(res_full)) {
    res_full$ind$fitness <- ind$fitness
    return(res_full$ind)
  } else {
    # If it failed on full data, return ind with terrible fitness
    ind$fitness <- if (task == "classification") -Inf else -Inf # Fitness is always maximized (-RMSE)
    return(ind)
  }
}
