# Dimension-reduction transformers: PCA, SVD, UMAP, MCA, FAMD, BG-PCA (+ supervised variants).
# Split out of transformers.R; loaded after it (alphabetical file order).

evo_transformers$pca <- create_transformer(
  name = "pca",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    tryCatch({
      pca_model <- stats::prcomp(x, center = TRUE, scale. = TRUE)
      list(model = pca_model, valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      list(model = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    if (is.null(state$preds_cache)) {
      # Fallback in case state didn't initialize it
      preds <- stats::predict(state$model, x)
    } else {
      x_key <- digest::digest(x, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- stats::predict(state$model, x)
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    preds[, comp_idx]
  },
  name_generator = function(gene) .gene_col_name(gene, "pca")
)

# Truncated SVD
evo_transformers$truncated_svd <- create_transformer(
  name = "truncated_svd",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(x)))))
      res <- svd(x, nu = 0, nv = min(C, ncol(x)))
      list(v = res$v, valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      list(v = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    # Calculate full projection (all nv columns)
    if (is.null(state$preds_cache)) {
      preds <- x %*% state$v
    } else {
      x_key <- digest::digest(x, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- x %*% state$v
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "svd")
)





# --- STATEFUL CATEGORICAL TRANSFORMERS ---

evo_transformers$umap <- create_transformer(
  name = "umap",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    verbose <- is_verbose()
    x_s <- .cluster_prep_x(x, min_rows = 15, verbose = verbose, tag = "UMAP Fit")
    if (is.null(x_s)) {
      x_s <- x
    }
    
    n_neighbors <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    if (nrow(x_s) < n_neighbors) {
      n_neighbors <- max(2, nrow(x_s) - 1)
    }
    
    dens_scale <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    
    t0 <- Sys.time()
    tryCatch({
      threads <- getOption("evoFE.threads", 1)
      C <- max(2L, as.integer(round(log2(ncol(x_s)))))
      model <- uwot::umap(x_s, n_neighbors = n_neighbors, n_components = C, 
                          dens_scale = dens_scale, ret_model = TRUE, 
                          n_threads = threads, verbose = FALSE, init = "random")
      if (verbose) {
        message(sprintf("  [UMAP Fit] umap on %d rows. %d neighbors (dens_scale = %.2f). %.3f s",
                        nrow(x_s), n_neighbors, dens_scale, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      preds_cache <- new.env(hash = TRUE, parent = emptyenv())
      list(
        model = model,
        valid = TRUE,
        x_train = x_s,
        training_embedding = model$embedding,
        preds_cache = preds_cache
      )
    }, error = function(e) {
      if (verbose) {
        message(sprintf("  [UMAP Fit] Failed: %s", conditionMessage(e)))
      }
      list(model = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    verbose <- is_verbose()
    if (is.null(state) || !state$valid || is.null(state$x_train)) {
      if (verbose) {
        message("  [UMAP Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(0, nrow(data)))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    compute <- function() {
      if (nrow(x) == nrow(state$x_train) && all(x == state$x_train)) {
        if (verbose) {
          message("  [UMAP Apply] (Fast Path): using cached training predictions. 0.000 s")
        }
        state$training_embedding
      } else {
        t0 <- Sys.time()
        threads <- getOption("evoFE.threads", 1)
        preds <- uwot::umap_transform(x, model = state$model, n_threads = threads, verbose = FALSE)
        if (verbose) {
          message(sprintf("  [UMAP Apply] Transform: %d test rows. %.3f s",
                          nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
        }
        preds
      }
    }
    
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(x, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    preds[, comp_idx]
  },
  name_generator = function(gene) .gene_col_name(gene, "ump")
)

# MST-based Anomaly Score
evo_transformers$random_projection <- create_transformer(
  name = "random_projection",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    P <- length(input_cols)
    w <- stats::rnorm(P)
    w_norm <- sqrt(sum(w^2))
    w <- if (w_norm == 0) rep(1 / sqrt(P), P) else w / w_norm
    list(w = w)
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    if (is.null(state) || is.null(state$w)) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    as.vector(x %*% state$w)
  },
  name_generator = function(gene) .gene_col_name(gene, "rp")
)

# Lumbermark Clustering
evo_transformers$mca <- create_transformer(
  name = "mca",
  type = "multivariate",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    D <- length(input_cols)
    N <- nrow(data)
    
    training_levels <- list()
    for (col in input_cols) {
      training_levels[[col]] <- sort(unique(as.character(data[[col]])))
    }
    
    cols_list <- list()
    for (col in input_cols) {
      x <- as.character(data[[col]])
      levels <- training_levels[[col]]
      mat <- matrix(0, nrow = N, ncol = length(levels))
      colnames(mat) <- paste0(col, "_", levels)
      valid_idx <- which(!is.na(x) & x %in% levels)
      if (length(valid_idx) > 0) {
        x_fact <- factor(x, levels = levels)
        mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
      }
      cols_list[[col]] <- mat
    }
    Z <- do.call(cbind, cols_list)
    
    C_col <- colSums(Z)
    valid_cols <- C_col > 0
    if (sum(valid_cols) == 0) {
      return(list(valid = FALSE))
    }
    
    Z <- Z[, valid_cols, drop = FALSE]
    C_col <- C_col[valid_cols]
    
    training_levels_filtered <- list()
    for (col in input_cols) {
      levels <- training_levels[[col]]
      sub_cols <- paste0(col, "_", levels)
      keep_levels <- levels[sub_cols %in% colnames(Z)]
      training_levels_filtered[[col]] <- keep_levels
    }
    
    Z_centered <- sweep(Z, 2, C_col / N, "-")
    A <- sweep(Z_centered, 2, sqrt(C_col), "/")
    
    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(Z)))))
      nv <- min(C, ncol(Z) - D)
      if (nv < 2) nv <- 2
      nv <- min(nv, ncol(Z))
      
      res <- svd(A, nu = 0, nv = nv)
      list(
        v = res$v,
        training_levels = training_levels_filtered,
        C_col = C_col,
        N = N,
        D = D,
        valid = TRUE,
        preds_cache = new.env(hash = TRUE, parent = emptyenv())
      )
    }, error = function(e) {
      list(valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    
    N_new <- nrow(data)
    cols_list <- list()
    for (col in input_cols) {
      x <- as.character(data[[col]])
      levels <- state$training_levels[[col]]
      mat <- matrix(0, nrow = N_new, ncol = length(levels))
      colnames(mat) <- paste0(col, "_", levels)
      valid_idx <- which(!is.na(x) & x %in% levels)
      if (length(valid_idx) > 0) {
        x_fact <- factor(x, levels = levels)
        mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
      }
      cols_list[[col]] <- mat
    }
    Z <- do.call(cbind, cols_list)
    
    compute <- function() {
      Z_centered <- sweep(Z, 2, state$C_col / state$N, "-")
      A <- sweep(Z_centered, 2, sqrt(state$C_col), "/")
      (1 / sqrt(state$D)) * (A %*% state$v)
    }
    
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "mca")
)

# Factor Analysis of Mixed Data (FAMD)
evo_transformers$famd <- create_transformer(
  name = "famd",
  type = "multivariate",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    N <- nrow(data)
    
    is_num <- sapply(input_cols, function(col) is.numeric(data[[col]]))
    num_cols <- input_cols[is_num]
    cat_cols <- input_cols[!is_num]
    
    if (length(num_cols) == 0 && length(cat_cols) == 0) {
      return(list(valid = FALSE))
    }
    
    mu <- numeric(0)
    sigma <- numeric(0)
    Z_num <- matrix(0, nrow = N, ncol = 0)
    if (length(num_cols) > 0) {
      X_num <- as.matrix(data[, num_cols, with = FALSE])
      mu <- colMeans(X_num, na.rm = TRUE)
      sigma <- apply(X_num, 2, stats::sd, na.rm = TRUE)
      sigma[sigma == 0 | is.na(sigma)] <- 1
      X_num_imp <- X_num
      for (j in seq_len(ncol(X_num_imp))) {
        X_num_imp[is.na(X_num_imp[, j]), j] <- mu[j]
      }
      Z_num <- scale(X_num_imp, center = mu, scale = sigma)
    }
    
    training_levels <- list()
    Z_cat <- matrix(0, nrow = N, ncol = 0)
    C_col <- numeric(0)
    p_col <- numeric(0)
    
    if (length(cat_cols) > 0) {
      for (col in cat_cols) {
        training_levels[[col]] <- sort(unique(as.character(data[[col]])))
      }
      
      cols_list <- list()
      for (col in cat_cols) {
        x <- as.character(data[[col]])
        levels <- training_levels[[col]]
        mat <- matrix(0, nrow = N, ncol = length(levels))
        colnames(mat) <- paste0(col, "_", levels)
        valid_idx <- which(!is.na(x) & x %in% levels)
        if (length(valid_idx) > 0) {
          x_fact <- factor(x, levels = levels)
          mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
        }
        cols_list[[col]] <- mat
      }
      Z_cat_raw <- do.call(cbind, cols_list)
      C_col <- colSums(Z_cat_raw)
      valid_cat_cols <- C_col > 0
      
      if (sum(valid_cat_cols) > 0) {
        Z_cat_raw <- Z_cat_raw[, valid_cat_cols, drop = FALSE]
        C_col <- C_col[valid_cat_cols]
        p_col <- C_col / N
        
        Z_cat_centered <- sweep(Z_cat_raw, 2, p_col, "-")
        Z_cat <- sweep(Z_cat_centered, 2, sqrt(p_col), "/")
        
        for (col in cat_cols) {
          levels <- training_levels[[col]]
          sub_cols <- paste0(col, "_", levels)
          keep_levels <- levels[sub_cols %in% colnames(Z_cat_raw)]
          training_levels[[col]] <- keep_levels
        }
      }
    }
    
    Z <- cbind(Z_num, Z_cat)
    if (ncol(Z) == 0) return(list(valid = FALSE))
    
    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(Z)))))
      nv <- min(C, ncol(Z))
      res <- svd(Z, nu = 0, nv = nv)
      list(
        v = res$v,
        mu = mu,
        sigma = sigma,
        training_levels = training_levels,
        p_col = p_col,
        num_cols = num_cols,
        cat_cols = cat_cols,
        valid = TRUE,
        preds_cache = new.env(hash = TRUE, parent = emptyenv())
      )
    }, error = function(e) {
      list(valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    
    N_new <- nrow(data)
    
    Z_num <- matrix(0, nrow = N_new, ncol = 0)
    if (length(state$num_cols) > 0) {
      X_num <- as.matrix(data[, state$num_cols, with = FALSE])
      for (j in seq_len(ncol(X_num))) {
        X_num[is.na(X_num[, j]), j] <- state$mu[j]
      }
      Z_num <- scale(X_num, center = state$mu, scale = state$sigma)
    }
    
    Z_cat <- matrix(0, nrow = N_new, ncol = 0)
    if (length(state$cat_cols) > 0 && length(state$p_col) > 0) {
      cols_list <- list()
      for (col in state$cat_cols) {
        x <- as.character(data[[col]])
        levels <- state$training_levels[[col]]
        mat <- matrix(0, nrow = N_new, ncol = length(levels))
        valid_idx <- which(!is.na(x) & x %in% levels)
        if (length(valid_idx) > 0) {
          x_fact <- factor(x, levels = levels)
          mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
        }
        cols_list[[col]] <- mat
      }
      Z_cat_raw <- do.call(cbind, cols_list)
      Z_cat_centered <- sweep(Z_cat_raw, 2, state$p_col, "-")
      Z_cat <- sweep(Z_cat_centered, 2, sqrt(state$p_col), "/")
    }
    
    Z <- cbind(Z_num, Z_cat)
    
    compute <- function() {
      Z %*% state$v
    }
    
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "famd")
)

# Between-Class PCA (BCA)
evo_transformers$between_group_pca <- create_transformer(
  name = "between_group_pca",
  type = "multivariate",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_cols <- input_cols[-1]
    N <- nrow(data)
    
    if (length(num_cols) == 0) return(list(valid = FALSE))
    
    X_num <- as.matrix(data[, num_cols, with = FALSE])
    mu <- colMeans(X_num, na.rm = TRUE)
    sigma <- apply(X_num, 2, stats::sd, na.rm = TRUE)
    sigma[sigma == 0 | is.na(sigma)] <- 1
    
    X_num_imp <- X_num
    for (j in seq_len(ncol(X_num_imp))) {
      X_num_imp[is.na(X_num_imp[, j]), j] <- mu[j]
    }
    Z_num <- scale(X_num_imp, center = mu, scale = sigma)
    
    cat_vals <- as.character(data[[cat_col]])
    cat_vals[is.na(cat_vals)] <- "NA"
    
    dt <- data.table::data.table(cat = cat_vals, Z_num)
    group_means <- dt[, lapply(.SD, mean), by = cat]
    
    dt_mean <- group_means[dt[, .(cat)], on = "cat"]
    Z_mean <- as.matrix(dt_mean[, -1, with = FALSE])
    
    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(Z_mean)))))
      nv <- min(C, ncol(Z_mean))
      res <- svd(Z_mean, nu = 0, nv = nv)
      list(
        v = res$v,
        mu = mu,
        sigma = sigma,
        num_cols = num_cols,
        valid = TRUE,
        preds_cache = new.env(hash = TRUE, parent = emptyenv())
      )
    }, error = function(e) {
      list(valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    num_cols <- input_cols[-1]
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    
    X_num <- as.matrix(data[, state$num_cols, with = FALSE])
    for (j in seq_len(ncol(X_num))) {
      X_num[is.na(X_num[, j]), j] <- state$mu[j]
    }
    Z_num <- scale(X_num, center = state$mu, scale = state$sigma)
    
    compute <- function() {
      Z_num %*% state$v
    }
    
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z_num, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "bgpca")
)

# Helper: normalize target y into a mathematically valid numeric vector for single-column target encoding
evo_transformers$supervised_bgpca <- create_transformer(
  name = "supervised_bgpca",
  type = "multivariate",
  input_type = "numeric",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (is.null(target_col)) return(list(valid = FALSE))
    input_cols <- gene$input_cols
    N <- nrow(data)
    if (length(input_cols) == 0) return(list(valid = FALSE))

    X <- as.matrix(data[, input_cols, with = FALSE])
    mu <- colMeans(X, na.rm = TRUE)
    sigma <- apply(X, 2, stats::sd, na.rm = TRUE)
    sigma[sigma == 0 | is.na(sigma)] <- 1
    for (j in seq_len(ncol(X))) X[is.na(X[, j]), j] <- mu[j]
    Z <- scale(X, center = mu, scale = sigma)

    grps <- .target_to_groups(data[[target_col]])
    dt <- data.table::data.table(grp = grps, Z)
    group_means <- dt[, lapply(.SD, mean), by = grp]
    dt_mean <- group_means[dt[, .(grp)], on = "grp"]
    Z_mean <- as.matrix(dt_mean[, -1, with = FALSE])

    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(Z_mean)))))
      nv <- min(C, ncol(Z_mean))
      res <- svd(Z_mean, nu = 0, nv = nv)
      list(v = res$v, mu = mu, sigma = sigma, valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    input_cols <- gene$input_cols
    X <- as.matrix(data[, input_cols, with = FALSE])
    for (j in seq_len(ncol(X))) X[is.na(X[, j]), j] <- state$mu[j]
    Z <- scale(X, center = state$mu, scale = state$sigma)

    compute <- function() Z %*% state$v
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "sbgpca")
)

# Supervised MCA (groups by target, discriminant on categorical features)
evo_transformers$supervised_mca <- create_transformer(
  name = "supervised_mca",
  type = "multivariate",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (is.null(target_col)) return(list(valid = FALSE))
    input_cols <- gene$input_cols
    N <- nrow(data)
    D <- length(input_cols)

    training_levels <- list()
    for (col in input_cols) {
      training_levels[[col]] <- sort(unique(as.character(data[[col]])))
    }

    cols_list <- list()
    for (col in input_cols) {
      x <- as.character(data[[col]])
      levels <- training_levels[[col]]
      mat <- matrix(0, nrow = N, ncol = length(levels))
      colnames(mat) <- paste0(col, "_", levels)
      valid_idx <- which(!is.na(x) & x %in% levels)
      if (length(valid_idx) > 0) {
        x_fact <- factor(x, levels = levels)
        mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
      }
      cols_list[[col]] <- mat
    }
    Z <- do.call(cbind, cols_list)

    C_col <- colSums(Z)
    valid_cols <- C_col > 0
    if (sum(valid_cols) == 0) return(list(valid = FALSE))
    Z <- Z[, valid_cols, drop = FALSE]
    C_col <- C_col[valid_cols]

    training_levels_filtered <- list()
    for (col in input_cols) {
      levels <- training_levels[[col]]
      sub_cols <- paste0(col, "_", levels)
      training_levels_filtered[[col]] <- levels[sub_cols %in% colnames(Z)]
    }

    Z_centered <- sweep(Z, 2, C_col / N, "-")
    A <- sweep(Z_centered, 2, sqrt(C_col), "/")

    grps <- .target_to_groups(data[[target_col]])
    dt <- data.table::data.table(grp = grps, A)
    group_means <- dt[, lapply(.SD, mean), by = grp]
    dt_mean <- group_means[dt[, .(grp)], on = "grp"]
    A_mean <- as.matrix(dt_mean[, -1, with = FALSE])

    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(A_mean)))))
      nv <- min(C, ncol(A_mean))
      res <- svd(A_mean, nu = 0, nv = nv)
      list(v = res$v, training_levels = training_levels_filtered,
           C_col = C_col, N = N, D = D, valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))

    N_new <- nrow(data)
    cols_list <- list()
    for (col in input_cols) {
      x <- as.character(data[[col]])
      levels <- state$training_levels[[col]]
      mat <- matrix(0, nrow = N_new, ncol = length(levels))
      valid_idx <- which(!is.na(x) & x %in% levels)
      if (length(valid_idx) > 0) {
        x_fact <- factor(x, levels = levels)
        mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
      }
      cols_list[[col]] <- mat
    }
    Z <- do.call(cbind, cols_list)
    Z_centered <- sweep(Z, 2, state$C_col / state$N, "-")
    A <- sweep(Z_centered, 2, sqrt(state$C_col), "/")

    compute <- function() (1 / sqrt(state$D)) * (A %*% state$v)
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "smca")
)

# Supervised FAMD (groups by target, discriminant on mixed features)
evo_transformers$supervised_famd <- create_transformer(
  name = "supervised_famd",
  type = "multivariate",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (is.null(target_col)) return(list(valid = FALSE))
    input_cols <- gene$input_cols
    N <- nrow(data)

    is_num <- sapply(input_cols, function(col) is.numeric(data[[col]]))
    num_cols <- input_cols[is_num]
    cat_cols <- input_cols[!is_num]
    if (length(num_cols) == 0 && length(cat_cols) == 0) return(list(valid = FALSE))

    mu <- numeric(0); sigma <- numeric(0)
    Z_num <- matrix(0, nrow = N, ncol = 0)
    if (length(num_cols) > 0) {
      X_num <- as.matrix(data[, num_cols, with = FALSE])
      mu <- colMeans(X_num, na.rm = TRUE)
      sigma <- apply(X_num, 2, stats::sd, na.rm = TRUE)
      sigma[sigma == 0 | is.na(sigma)] <- 1
      for (j in seq_len(ncol(X_num))) X_num[is.na(X_num[, j]), j] <- mu[j]
      Z_num <- scale(X_num, center = mu, scale = sigma)
    }

    training_levels <- list()
    Z_cat <- matrix(0, nrow = N, ncol = 0)
    p_col <- numeric(0)
    if (length(cat_cols) > 0) {
      for (col in cat_cols) training_levels[[col]] <- sort(unique(as.character(data[[col]])))
      cols_list <- list()
      for (col in cat_cols) {
        x <- as.character(data[[col]])
        levels <- training_levels[[col]]
        mat <- matrix(0, nrow = N, ncol = length(levels))
        colnames(mat) <- paste0(col, "_", levels)
        valid_idx <- which(!is.na(x) & x %in% levels)
        if (length(valid_idx) > 0) {
          x_fact <- factor(x, levels = levels)
          mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
        }
        cols_list[[col]] <- mat
      }
      Z_cat_raw <- do.call(cbind, cols_list)
      C_col <- colSums(Z_cat_raw)
      valid_cat_cols <- C_col > 0
      if (sum(valid_cat_cols) > 0) {
        Z_cat_raw <- Z_cat_raw[, valid_cat_cols, drop = FALSE]
        C_col <- C_col[valid_cat_cols]
        p_col <- C_col / N
        Z_cat_centered <- sweep(Z_cat_raw, 2, p_col, "-")
        Z_cat <- sweep(Z_cat_centered, 2, sqrt(p_col), "/")
        for (col in cat_cols) {
          levels <- training_levels[[col]]
          sub_cols <- paste0(col, "_", levels)
          training_levels[[col]] <- levels[sub_cols %in% colnames(Z_cat_raw)]
        }
      }
    }

    Z <- cbind(Z_num, Z_cat)
    if (ncol(Z) == 0) return(list(valid = FALSE))

    grps <- .target_to_groups(data[[target_col]])
    dt <- data.table::data.table(grp = grps, Z)
    group_means <- dt[, lapply(.SD, mean), by = grp]
    dt_mean <- group_means[dt[, .(grp)], on = "grp"]
    Z_mean <- as.matrix(dt_mean[, -1, with = FALSE])

    tryCatch({
      C <- max(2L, as.integer(round(log2(ncol(Z_mean)))))
      nv <- min(C, ncol(Z_mean))
      res <- svd(Z_mean, nu = 0, nv = nv)
      list(v = res$v, mu = mu, sigma = sigma,
           training_levels = training_levels, p_col = p_col,
           num_cols = num_cols, cat_cols = cat_cols, valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))

    N_new <- nrow(data)
    Z_num <- matrix(0, nrow = N_new, ncol = 0)
    if (length(state$num_cols) > 0) {
      X_num <- as.matrix(data[, state$num_cols, with = FALSE])
      for (j in seq_len(ncol(X_num))) X_num[is.na(X_num[, j]), j] <- state$mu[j]
      Z_num <- scale(X_num, center = state$mu, scale = state$sigma)
    }
    Z_cat <- matrix(0, nrow = N_new, ncol = 0)
    if (length(state$cat_cols) > 0 && length(state$p_col) > 0) {
      cols_list <- list()
      for (col in state$cat_cols) {
        x <- as.character(data[[col]])
        levels <- state$training_levels[[col]]
        mat <- matrix(0, nrow = N_new, ncol = length(levels))
        valid_idx <- which(!is.na(x) & x %in% levels)
        if (length(valid_idx) > 0) {
          x_fact <- factor(x, levels = levels)
          mat[cbind(valid_idx, as.integer(x_fact[valid_idx]))] <- 1
        }
        cols_list[[col]] <- mat
      }
      Z_cat_raw <- do.call(cbind, cols_list)
      Z_cat_centered <- sweep(Z_cat_raw, 2, state$p_col, "-")
      Z_cat <- sweep(Z_cat_centered, 2, sqrt(state$p_col), "/")
    }
    Z <- cbind(Z_num, Z_cat)

    compute <- function() Z %*% state$v
    if (is.null(state$preds_cache)) {
      preds <- compute()
    } else {
      x_key <- digest::digest(Z, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
      } else {
        preds <- compute()
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "sfamd")
)

# --- SKRUB-INSPIRED ENCODERS ---

# String Similarity Encoder (Jaccard similarity of 3-grams to prototype categories)
