# Graph & clustering transformers: MST scores, Genie/Lumbermark/deadwood.
# Split out of transformers.R; loaded after it (alphabetical file order).

evo_transformers$mst_score <- create_transformer(
  name = "mst_score",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, scores = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("quitefastmst", quietly = TRUE)) stop("quitefastmst package is not available")
      x_s <- .cluster_prep_x(x, min_rows = 6, verbose = verbose, tag = "MST Fit")
      if (is.null(x_s)) return(list(x_train = NULL, scores = NULL, valid = FALSE))
      t0 <- Sys.time()
      res_mst <- quitefastmst::mst_euclid(x_s)
      if (verbose) message(sprintf("  [MST Fit] mst_euclid on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      dt_edges <- data.table::data.table(
        node = c(res_mst$mst.index[, 1], res_mst$mst.index[, 2]),
        dist = c(res_mst$mst.dist, res_mst$mst.dist)
      )
      scores <- dt_edges[, .(score = max(dist)), by = node][order(node)]$score
      list(x_train = x_s, scores = as.numeric(scores), valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(x_train = NULL, scores = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    .cluster_knn_apply(x_test, state, function(idx, s) s$scores[idx],
                       verbose = is_verbose(), tag = "MST Apply")
  },
  name_generator = function(gene) .gene_col_name(gene, "mst")
)

# Genie Clustering
evo_transformers$genie <- create_transformer(
  name = "genie",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    gini_threshold <- if (!is.null(gene$params$gini_threshold)) gene$params$gini_threshold else 0.3
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("genieclust", quietly = TRUE)) stop("genieclust package is not available")
      x_s <- .cluster_prep_x(x, min_rows = max(6L, k), verbose = verbose, tag = "Genie Fit")
      if (is.null(x_s)) return(list(x_train = NULL, labels = NULL, valid = FALSE))
      t0 <- Sys.time()
      labels <- genieclust::genie(x_s, k = k, gini_threshold = gini_threshold)
      if (verbose) message(sprintf("  [Genie Fit] genie on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      list(x_train = x_s, labels = as.integer(labels), valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(x_train = NULL, labels = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid)) return(rep(1L, nrow(data)))
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    .cluster_knn_apply(x_test, state, function(idx, s) s$labels[idx],
                       verbose = is_verbose(), tag = "Genie Apply")
  },
  name_generator = function(gene) .gene_col_name(gene, "gnie")
)

# UMAP-Genie Combo
evo_transformers$umap_genie <- create_transformer(
  name = "umap_genie",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    gini_threshold <- if (!is.null(gene$params$gini_threshold)) gene$params$gini_threshold else 0.3
    n_neighbors <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    dens_scale <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    
    verbose <- is_verbose()
    x_s <- .cluster_prep_x(x, min_rows = max(15L, k), verbose = verbose, tag = "UMAP-Genie Fit")
    if (is.null(x_s)) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    
    if (nrow(x_s) < n_neighbors) {
      n_neighbors <- max(2, nrow(x_s) - 1)
    }
    
    t0 <- Sys.time()
    tryCatch({
      if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot package is not available")
      if (!requireNamespace("genieclust", quietly = TRUE)) stop("genieclust package is not available")
      
      threads <- getOption("evoFE.threads", 1)
      C <- max(2L, as.integer(round(log2(ncol(x_s)))))
      
      # 1. Fit UMAP
      umap_model <- uwot::umap(x_s, n_neighbors = n_neighbors, n_components = C, 
                              dens_scale = dens_scale, ret_model = TRUE, 
                              n_threads = threads, n_sgd_threads = "auto",
                              verbose = FALSE, init = "random")
      x_s_umap <- umap_model$embedding
      
      # 2. Run Genie clustering on UMAP embedding
      labels <- genieclust::genie(x_s_umap, k = k, gini_threshold = gini_threshold)
      
      if (verbose) {
        message(sprintf("  [UMAP-Genie Fit] umap + genie on %d rows. %d components, %d clusters. %.3f s",
                        nrow(x_s), C, k, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      preds_cache <- new.env(hash = TRUE, parent = emptyenv())
      list(
        umap_model = umap_model,
        x_train = x_s,
        x_train_umap = x_s_umap,
        labels = as.integer(labels),
        valid = TRUE,
        preds_cache = preds_cache
      )
    }, error = function(e) {
      if (verbose) {
        message(sprintf("  [UMAP-Genie Fit] Failed: %s", conditionMessage(e)))
      }
      list(umap_model = NULL, x_train = NULL, x_train_umap = NULL, labels = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid)) return(rep(1L, nrow(data)))
    
    input_cols <- gene$input_cols
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    
    verbose <- is_verbose()
    
    # Fast path check on original x_test
    if (nrow(x_test) == nrow(state$x_train) && all(x_test == state$x_train)) {
      if (verbose) {
        message("  [UMAP-Genie Apply] (Fast Path): using cached training predictions. 0.000 s")
      }
      return(state$labels)
    }
    
    compute <- function() {
      t0 <- Sys.time()
      threads <- getOption("evoFE.threads", 1)
      
      # 1. Transform test data to UMAP space
      x_test_umap <- uwot::umap_transform(x_test, model = state$umap_model, n_threads = threads,
                                          n_sgd_threads = "auto", verbose = FALSE)
      
      # 2. KNN lookup in UMAP space
      dummy_state <- list(
        x_train = state$x_train_umap,
        labels = state$labels,
        valid = TRUE,
        preds_cache = NULL
      )
      
      labels_test <- .cluster_knn_apply(x_test_umap, dummy_state, function(idx, s) s$labels[idx],
                                        verbose = FALSE, tag = "UMAP-Genie Apply Internal")
      
      if (verbose) {
        message(sprintf("  [UMAP-Genie Apply] Transform + KNN: %d test rows. %.3f s",
                        nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      as.integer(labels_test)
    }
    
    if (is.null(state$preds_cache)) {
      return(compute())
    }
    
    x_key <- digest::digest(x_test, algo = "xxhash64")
    if (exists(x_key, envir = state$preds_cache)) {
      return(get(x_key, envir = state$preds_cache))
    }
    
    preds <- compute()
    assign(x_key, preds, envir = state$preds_cache)
    preds
  },
  name_generator = function(gene) .gene_col_name(gene, "ug")
)

# UMAP-Lumbermark Combo
evo_transformers$umap_lumbermark <- create_transformer(
  name = "umap_lumbermark",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    n_neighbors <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    dens_scale <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    
    verbose <- is_verbose()
    x_s <- .cluster_prep_x(x, min_rows = max(15L, 2L * k), verbose = verbose, tag = "UMAP-Lumbermark Fit")
    if (is.null(x_s)) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    
    if (nrow(x_s) < n_neighbors) {
      n_neighbors <- max(2, nrow(x_s) - 1)
    }
    
    t0 <- Sys.time()
    tryCatch({
      if (!requireNamespace("uwot", quietly = TRUE)) stop("uwot package is not available")
      if (!requireNamespace("lumbermark", quietly = TRUE)) stop("lumbermark package is not available")
      
      threads <- getOption("evoFE.threads", 1)
      C <- max(2L, as.integer(round(log2(ncol(x_s)))))
      
      # 1. Fit UMAP
      umap_model <- uwot::umap(x_s, n_neighbors = n_neighbors, n_components = C, 
                              dens_scale = dens_scale, ret_model = TRUE, 
                              n_threads = threads, n_sgd_threads = "auto",
                              verbose = FALSE, init = "random")
      x_s_umap <- umap_model$embedding
      
      # 2. Run Lumbermark clustering on UMAP embedding
      labels <- lumbermark::lumbermark(x_s_umap, k = k, min_cluster_size = 2)
      
      if (verbose) {
        message(sprintf("  [UMAP-Lumb Fit] umap + lumbermark on %d rows. %d components, %d clusters. %.3f s",
                        nrow(x_s), C, k, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      preds_cache <- new.env(hash = TRUE, parent = emptyenv())
      list(
        umap_model = umap_model,
        x_train = x_s,
        x_train_umap = x_s_umap,
        labels = as.integer(labels),
        valid = TRUE,
        preds_cache = preds_cache
      )
    }, error = function(e) {
      if (verbose) {
        message(sprintf("  [UMAP-Lumb Fit] Failed: %s", conditionMessage(e)))
      }
      list(umap_model = NULL, x_train = NULL, x_train_umap = NULL, labels = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid)) return(rep(1L, nrow(data)))
    
    input_cols <- gene$input_cols
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    
    verbose <- is_verbose()
    
    # Fast path check on original x_test
    if (nrow(x_test) == nrow(state$x_train) && all(x_test == state$x_train)) {
      if (verbose) {
        message("  [UMAP-Lumb Apply] (Fast Path): using cached training predictions. 0.000 s")
      }
      return(state$labels)
    }
    
    compute <- function() {
      t0 <- Sys.time()
      threads <- getOption("evoFE.threads", 1)
      
      # 1. Transform test data to UMAP space
      x_test_umap <- uwot::umap_transform(x_test, model = state$umap_model, n_threads = threads,
                                          n_sgd_threads = "auto", verbose = FALSE)
      
      # 2. KNN lookup in UMAP space
      dummy_state <- list(
        x_train = state$x_train_umap,
        labels = state$labels,
        valid = TRUE,
        preds_cache = NULL
      )
      
      labels_test <- .cluster_knn_apply(x_test_umap, dummy_state, function(idx, s) s$labels[idx],
                                        verbose = FALSE, tag = "UMAP-Lumb Apply Internal")
      
      if (verbose) {
        message(sprintf("  [UMAP-Lumb Apply] Transform + KNN: %d test rows. %.3f s",
                        nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      as.integer(labels_test)
    }
    
    if (is.null(state$preds_cache)) {
      return(compute())
    }
    
    x_key <- digest::digest(x_test, algo = "xxhash64")
    if (exists(x_key, envir = state$preds_cache)) {
      return(get(x_key, envir = state$preds_cache))
    }
    
    preds <- compute()
    assign(x_key, preds, envir = state$preds_cache)
    preds
  },
  name_generator = function(gene) .gene_col_name(gene, "ulm")
)

# Group-by Ratio
evo_transformers$lumbermark <- create_transformer(
  name = "lumbermark",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("lumbermark", quietly = TRUE)) stop("lumbermark package is not available")
      x_s <- .cluster_prep_x(x, min_rows = max(6L, 2L * k), verbose = verbose, tag = "Lumbermark Fit")
      if (is.null(x_s)) return(list(x_train = NULL, labels = NULL, valid = FALSE))
      t0 <- Sys.time()
      labels <- lumbermark::lumbermark(x_s, k = k, min_cluster_size = 2)
      if (verbose) message(sprintf("  [Lumbermark Fit] lumbermark on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      list(x_train = x_s, labels = as.integer(labels), valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(x_train = NULL, labels = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid)) return(rep(1L, nrow(data)))
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    .cluster_knn_apply(x_test, state, function(idx, s) s$labels[idx],
                       verbose = is_verbose(), tag = "Lumbermark Apply")
  },
  name_generator = function(gene) .gene_col_name(gene, "lmb")
)

# Deadwood Anomaly Detection
evo_transformers$deadwood <- create_transformer(
  name = "deadwood",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(x_train = NULL, labels = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("deadwood", quietly = TRUE)) stop("deadwood package is not available")
      x_s <- .cluster_prep_x(x, min_rows = 6, verbose = verbose, tag = "Deadwood Fit")
      if (is.null(x_s)) return(list(x_train = NULL, labels = NULL, valid = FALSE))
      t0 <- Sys.time()
      outliers <- deadwood::deadwood(x_s)
      if (verbose) message(sprintf("  [Deadwood Fit] deadwood on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      list(x_train = x_s, labels = as.numeric(outliers), valid = TRUE,
           preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(x_train = NULL, labels = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    .cluster_knn_apply(x_test, state, function(idx, s) s$labels[idx],
                       verbose = is_verbose(), tag = "Deadwood Apply")
  },
  name_generator = function(gene) .gene_col_name(gene, "dwd")
)

# One-Hot Encoding
evo_transformers$genie_centroid_dist <- create_transformer(
  name = "genie_centroid_dist",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(centroids = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    gini_threshold <- if (!is.null(gene$params$gini_threshold)) gene$params$gini_threshold else 0.3
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("genieclust", quietly = TRUE)) stop("genieclust package is not available")
      x_s <- .cluster_prep_x(x, min_rows = max(6L, k), verbose = verbose, tag = "GenieCDist Fit")
      if (is.null(x_s)) return(list(centroids = NULL, valid = FALSE))
      t0 <- Sys.time()
      labels <- genieclust::genie(x_s, k = k, gini_threshold = gini_threshold)
      if (verbose) message(sprintf("  [GenieCDist Fit] genie on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      centroids <- lapply(seq_len(k), function(j) {
        pts <- x_s[labels == j, , drop = FALSE]
        colMeans(if (nrow(pts) == 0) x_s else pts)
      })
      list(centroids = centroids, valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(centroids = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid) || is.null(state$centroids))
      return(rep(0, nrow(data)))
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (comp_idx > length(state$centroids)) comp_idx <- length(state$centroids)
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    verbose <- is_verbose()
    
    compute_dists <- function() {
      t0 <- Sys.time()
      res <- lapply(state$centroids, function(c) sqrt(rowSums(sweep(x_test, 2, c, "-")^2)))
      if (verbose) {
        message(sprintf("  [GenieCDist Apply] computed distances on %d rows. %.3f s",
                        nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      res
    }
    
    if (is.null(state$preds_cache)) return(compute_dists()[[comp_idx]])
    key <- digest::digest(x_test, algo = "xxhash64")
    if (!exists(key, envir = state$preds_cache)) {
      assign(key, compute_dists(), envir = state$preds_cache)
    } else {
      printed_key <- paste0(key, "_printed")
      if (verbose && !exists(printed_key, envir = state$preds_cache)) {
        message("  [GenieCDist Apply] (Cache Hit): returning cached distances. 0.000 s")
        assign(printed_key, TRUE, envir = state$preds_cache)
      }
    }
    get(key, envir = state$preds_cache)[[comp_idx]]
  },
  name_generator = function(gene) .gene_col_name(gene, "gncd")
)

# Lumbermark Centroid Distance
 evo_transformers$lumbermark_centroid_dist <- create_transformer(
  name = "lumbermark_centroid_dist",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    if (length(gene$input_cols) < 2) return(list(centroids = NULL, valid = FALSE))
    x <- as.matrix(data[, gene$input_cols, with = FALSE])
    x[is.na(x)] <- 0; storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    verbose <- is_verbose()
    tryCatch({
      if (!requireNamespace("lumbermark", quietly = TRUE)) stop("lumbermark package is not available")
      x_s <- .cluster_prep_x(x, min_rows = max(6L, 2L * k), verbose = verbose, tag = "LumbCDist Fit")
      if (is.null(x_s)) return(list(centroids = NULL, valid = FALSE))
      t0 <- Sys.time()
      labels <- lumbermark::lumbermark(x_s, k = k, min_cluster_size = 2)
      if (verbose) message(sprintf("  [LumbCDist Fit] lumbermark on %d rows. %.3f s", nrow(x_s), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      centroids <- lapply(seq_len(k), function(j) {
        pts <- x_s[labels == j, , drop = FALSE]
        colMeans(if (nrow(pts) == 0) x_s else pts)
      })
      list(centroids = centroids, valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) list(centroids = NULL, valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    if (is.null(state) || !isTRUE(state$valid) || is.null(state$centroids))
      return(rep(0, nrow(data)))
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (comp_idx > length(state$centroids)) comp_idx <- length(state$centroids)
    x_test <- as.matrix(data[, gene$input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0; storage.mode(x_test) <- "double"
    verbose <- is_verbose()
    
    compute_dists <- function() {
      t0 <- Sys.time()
      res <- lapply(state$centroids, function(c) sqrt(rowSums(sweep(x_test, 2, c, "-")^2)))
      if (verbose) {
        message(sprintf("  [LumbCDist Apply] computed distances on %d rows. %.3f s",
                        nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      res
    }
    
    if (is.null(state$preds_cache)) return(compute_dists()[[comp_idx]])
    key <- digest::digest(x_test, algo = "xxhash64")
    if (!exists(key, envir = state$preds_cache)) {
      assign(key, compute_dists(), envir = state$preds_cache)
    } else {
      printed_key <- paste0(key, "_printed")
      if (verbose && !exists(printed_key, envir = state$preds_cache)) {
        message("  [LumbCDist Apply] (Cache Hit): returning cached distances. 0.000 s")
        assign(printed_key, TRUE, envir = state$preds_cache)
      }
    }
    get(key, envir = state$preds_cache)[[comp_idx]]
  },
  name_generator = function(gene) .gene_col_name(gene, "lmcd")
)

# --- NEW TRANSFORMERS ---

# Power Transform
