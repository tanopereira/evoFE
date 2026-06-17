#' Create a transformer definition
#'
#' @param name Transformer name
#' @param type Type: "unary", "binary", "supervised_unary"
#' @param input_type Type of input: "numeric" or "categorical"
#' @param output_type Type of output: "numeric" or "categorical"
#' @param fit_func function(data, input_cols, target_col = NULL) returning state
#' @param apply_func function(data, input_cols, state = NULL) returning new column vector
#' @param name_generator function(input_cols) returning output column name
#' @param allow_replace Logical. Whether column sampling allows replacement.
#' @return An \code{evo_transformer} S3 object:
#'   a list with elements
#'   \code{name}, \code{type}, \code{input_type}, \code{output_type},
#'   \code{fit_func}, \code{apply_func}, \code{name_generator}, and
#'   \code{allow_replace}.
#' @import data.table
#' @importFrom uwot umap
#' @importFrom genieclust genie
#' @examples
#' # Define a transformer that adds a constant value of 10 to a variable
#' add_ten_trans <- create_transformer(
#'   name = "add_ten",
#'   type = "unary",
#'   input_type = "numeric",
#'   apply_func = function(data, gene, state = NULL) {
#'     data[[gene$input_cols[1]]] + 10
#'   },
#'   name_generator = function(gene) paste0("add10_", gene$input_cols[1])
#' )
#' print(add_ten_trans)
#' @export
create_transformer <- function(name, type, input_type = "numeric", output_type = "numeric", fit_func = NULL, apply_func, name_generator, allow_replace = FALSE) {
  structure(
    list(
      name = name,
      type = type,
      input_type = input_type,
      output_type = output_type,
      fit_func = fit_func,
      apply_func = apply_func,
      name_generator = name_generator,
      allow_replace = allow_replace
    ),
    class = "evo_transformer"
  )
}
#' Built-in feature transformers
#'
#' An environment containing default transformer definitions available for feature engineering.
#'
#' @return A named list of \code{evo_transformer} objects, each defining a
#'   feature transformation (e.g. \code{log}, \code{pca}, \code{target_encode}).
#' @export
evo_transformers <- new.env(parent = emptyenv())

#' Register a custom feature transformer
#'
#' Adds a user-defined feature transformer to the available pool for feature evolution.
#'
#' @param name Unique character string naming the transformer.
#' @param transformer An object of class \code{evo_transformer} created
#'   via \code{create_transformer}.
#' @examples
#' # Create a custom transformer
#' add_ten_trans <- create_transformer(
#'   name = "add_ten",
#'   type = "unary",
#'   input_type = "numeric",
#'   apply_func = function(data, gene, state = NULL) {
#'     data[[gene$input_cols[1]]] + 10
#'   },
#'   name_generator = function(gene) paste0("add10_", gene$input_cols[1])
#' )
#'
#' # Register it
#' register_transformer("add_ten", add_ten_trans)
#'
#' # Verify it is registered
#' exists("add_ten", envir = evo_transformers)
#' @export
register_transformer <- function(name, transformer) {
  if (!inherits(transformer, "evo_transformer")) {
    stop("transformer must be an object of class 'evo_transformer' (created via create_transformer).")
  }
  evo_transformers[[name]] <- transformer
  invisible(transformer)
}

is_verbose <- function() {
  val <- getOption("evoFE.verbose", 0)
  isTRUE(val) || val >= 2
}

# Produce a short, stable column name for a gene: "{prefix}_{6-char hash}".
# The hash covers transformer name + input columns + params, so identical
# genes always get identical names (deduplication) and different genes
# almost certainly get different names (collision probability ~1/16M).
# Full human-readable details are available via gene_to_formula().
.gene_col_name <- function(gene, prefix) {
  h <- substr(
    digest::digest(list(gene$transformer_name, gene$input_cols, gene$params),
                   algo = "xxhash32"),
    1, 6
  )
  paste0(prefix, "_", h)
}

# --- CLUSTERING SCAFFOLD HELPERS ---

# Deduplicate rows and optionally downsample to max_clustering_size.
# Returns the processed matrix, or NULL if there are too few rows.
# `min_rows` is checked *after* deduplication AND after downsampling.
.cluster_prep_x <- function(x, min_rows = 6, verbose = FALSE, tag = "Fit") {
  t0 <- Sys.time()
  dt_x <- data.table::as.data.table(x)
  unique_idx <- which(!duplicated(dt_x))
  x_unique <- x[unique_idx, , drop = FALSE]
  if (verbose)
    message(sprintf("  [%s] Dedup: %d unique / %d rows. %.3f s",
                    tag, nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  if (nrow(x_unique) < min_rows) return(NULL)

  max_size <- getOption("evoFE.max_clustering_size", 5000)
  if (length(max_size) > 1) max_size <- max_size[1]
  if (!is.numeric(max_size) || is.null(max_size)) max_size <- 0
  if (max_size > 0 && nrow(x_unique) > max_size) {
    # Preserve the caller's RNG state
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv))
      get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
      if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
      else if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
    }, add = TRUE)
    x_unique <- x_unique[sample(seq_len(nrow(x_unique)), max_size), , drop = FALSE]
    if (verbose)
      message(sprintf("  [%s] Downsampled to %d rows. %.3f s",
                      tag, nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }

  if (nrow(x_unique) < min_rows) return(NULL)
  x_unique
}

# Cache-aware KNN apply for clustering transformers.
# `get_preds(nearest_indices, state)` maps neighbour indices → prediction vector.
# When x_test is identical to the training set, indices are the identity permutation.
.cluster_knn_apply <- function(x_test, state, get_preds, verbose = FALSE, tag = "Apply") {
  if (is.null(state) || !isTRUE(state$valid))
    return(rep(0, nrow(x_test)))

  x_train <- state$x_train

  compute <- function() {
    if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
      if (verbose)
        message(sprintf("  [%s] KNN (Fast Path): using cached training predictions. 0.000 s", tag))
      get_preds(seq_len(nrow(x_train)), state)
    } else {
      t0 <- Sys.time()
      idx <- tryCatch(
        quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1],
        error = function(e) {
          x_train_mat <- as.matrix(x_train)
          x_test_mat <- as.matrix(x_test)
          if (!is.numeric(x_train_mat)) storage.mode(x_train_mat) <- "numeric"
          if (!is.numeric(x_test_mat)) storage.mode(x_test_mat) <- "numeric"

          if (nrow(x_test_mat) == 0L) {
            return(integer(0))
          }

          train_norms_sq <- rowSums(x_train_mat^2)
          D_approx <- sweep(-2 * tcrossprod(x_test_mat, x_train_mat), 2, train_norms_sq, "+")

          if (nrow(x_test_mat) == 1L) {
            res <- which.min(as.vector(D_approx))
            if (length(res) == 0L) NA_integer_ else res
          } else {
            apply(D_approx, 1, function(x) {
              res <- which.min(x)
              if (length(res) == 0L) NA_integer_ else res
            })
          }
        }
      )
      if (verbose)
        message(sprintf("  [%s] KNN: %d test -> %d train rows. %.3f s",
                        tag, nrow(x_test), nrow(x_train),
                        as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      get_preds(idx, state)
    }
  }

  if (is.null(state$preds_cache)) return(compute())

  key <- digest::digest(x_test, algo = "xxhash64")
  if (exists(key, envir = state$preds_cache))
    return(get(key, envir = state$preds_cache))

  preds <- compute()
  assign(key, preds, envir = state$preds_cache)
  preds
}

# --- STATELESS UNARY TRANSFORMERS ---

evo_transformers$log <- create_transformer(
  name = "log",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    # Use log1p of absolute value to handle zero and negative numbers
    log1p(abs(x))
  },
  name_generator = function(gene) .gene_col_name(gene, "log")
)

evo_transformers$sqrt <- create_transformer(
  name = "sqrt",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    sqrt(abs(x))
  },
  name_generator = function(gene) .gene_col_name(gene, "sqrt")
)

evo_transformers$reciprocal <- create_transformer(
  name = "reciprocal",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    ifelse(x == 0, 0, 1 / x)
  },
  name_generator = function(gene) .gene_col_name(gene, "rec")
)

# --- STATELESS BINARY TRANSFORMERS ---

evo_transformers$add <- create_transformer(
  name = "add",
  type = "multivariate",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    Reduce(`+`, lapply(input_cols, function(c) as.numeric(data[[c]])))
  },
  name_generator = function(gene) .gene_col_name(gene, "add"),
  allow_replace = TRUE
)

evo_transformers$subtract <- create_transformer(
  name = "subtract",
  type = "binary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    as.numeric(data[[input_cols[1]]]) - as.numeric(data[[input_cols[2]]])
  },
  name_generator = function(gene) .gene_col_name(gene, "sub")
)

evo_transformers$multiply <- create_transformer(
  name = "multiply",
  type = "multivariate",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    Reduce(`*`, lapply(input_cols, function(c) as.numeric(data[[c]])))
  },
  name_generator = function(gene) .gene_col_name(gene, "mul"),
  allow_replace = TRUE
)

evo_transformers$divide <- create_transformer(
  name = "divide",
  type = "binary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- data[[input_cols[2]]]
    ifelse(y == 0, 0, x / y)
  },
  name_generator = function(gene) .gene_col_name(gene, "div")
)

# --- STATEFUL SUPERVISED TRANSFORMERS ---

# Target Encoding
evo_transformers$target_encode <- create_transformer(
  name = "target_encode",
  type = "supervised_unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- data[[target_col]]
    
    # Calculate global mean
    global_mean <- mean(y, na.rm = TRUE)
    
    # Calculate category means and counts
    dt <- data.table::data.table(x = x, y = y)
    stats <- dt[, .(mean = mean(y, na.rm = TRUE), n = .N), by = x]
    
    # Smoothing parameters (can be adjusted)
    min_samples_leaf <- 10
    smoothing <- 10
    
    # Calculate smoothed target encoding
    # formula: (n * mean + smoothing * global_mean) / (n + smoothing)
    stats[, smoothed := (n * mean + smoothing * global_mean) / (n + smoothing)]
    
    mapping <- stats[, .(x, smoothed)]
    data.table::setkey(mapping, x)
    
    list(mapping = mapping, global_mean = global_mean)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    
    # Join with mapping
    dt <- data.table::data.table(x = x)
    mapping <- state$mapping
    
    # Map, filling missing categories with global mean
    res <- mapping[dt, on = "x"]$smoothed
    res[is.na(res)] <- state$global_mean
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "te")
)

# --- STATEFUL MULTIVARIATE TRANSFORMERS ---

# PCA
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

evo_transformers$frequency_encode <- create_transformer(
  name = "frequency_encode",
  type = "unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(x = x)
    mapping <- dt[, .N, by = x]
    data.table::setkey(mapping, x)
    list(mapping = mapping, default_val = median(mapping$N))
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(x = x)
    res <- state$mapping[dt, on = "x"]$N
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "freq")
)

# --- STATEFUL MIXED TRANSFORMERS ---

evo_transformers$groupby_mean <- create_transformer(
  name = "groupby_mean",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = mean(dt$n, na.rm = TRUE))
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbm")
)

evo_transformers$groupby_sd <- create_transformer(
  name = "groupby_sd",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(val = stats::sd(n, na.rm = TRUE)), by = c]
    mapping[is.na(val), val := 0]
    data.table::setkey(mapping, c)
    global_sd <- stats::sd(dt$n, na.rm = TRUE)
    if (is.na(global_sd)) global_sd <- 0
    list(mapping = mapping, default_val = global_sd)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbsd")
)

evo_transformers$groupby_max <- create_transformer(
  name = "groupby_max",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(val = max(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    global_max <- max(dt$n, na.rm = TRUE)
    list(mapping = mapping, default_val = global_max)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbmx")
)

evo_transformers$groupby_min <- create_transformer(
  name = "groupby_min",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(val = min(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    global_min <- min(dt$n, na.rm = TRUE)
    list(mapping = mapping, default_val = global_min)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbmn")
)

# --- STATEFUL MULTIVARIATE TRANSFORMERS (ADDITIONAL) ---

# UMAP Dimension Reduction
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
    
    n_neighbors <- 15
    if (nrow(x_s) < 15) {
      n_neighbors <- max(2, nrow(x_s) - 1)
    }
    
    t0 <- Sys.time()
    tryCatch({
      threads <- getOption("evoFE.threads", 1)
      C <- max(2L, as.integer(round(log2(ncol(x_s)))))
      model <- uwot::umap(x_s, n_neighbors = n_neighbors, n_components = C, 
                          ret_model = TRUE, n_threads = threads, verbose = FALSE, init = "random")
      if (verbose) {
        message(sprintf("  [UMAP Fit] umap on %d rows. %d neighbors. %.3f s",
                        nrow(x_s), n_neighbors, as.numeric(difftime(Sys.time(), t0, units = "secs"))))
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

# Group-by Ratio
evo_transformers$groupby_ratio <- create_transformer(
  name = "groupby_ratio",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = mean(dt$n, na.rm = TRUE))
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_cat <- data[[input_cols[1]]]
    x_num <- as.numeric(data[[input_cols[2]]])
    dt <- data.table::data.table(c = x_cat)
    means <- state$mapping[dt, on = "c"]$val
    means[is.na(means)] <- state$default_val
    res <- ifelse(means == 0, 0, x_num / means)
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbr")
)

# Group-by Z-Score
evo_transformers$groupby_zscore <- create_transformer(
  name = "groupby_zscore",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = as.numeric(data[[num_col]]))
    mapping <- dt[, .(mean_val = mean(n, na.rm = TRUE), sd_val = stats::sd(n, na.rm = TRUE)), by = c]
    mapping[is.na(sd_val), sd_val := 0]
    data.table::setkey(mapping, c)
    global_mean <- mean(dt$n, na.rm = TRUE)
    global_sd <- stats::sd(dt$n, na.rm = TRUE)
    if (is.na(global_sd)) global_sd <- 0
    list(mapping = mapping, default_mean = global_mean, default_sd = global_sd)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_cat <- data[[input_cols[1]]]
    x_num <- as.numeric(data[[input_cols[2]]])
    dt <- data.table::data.table(c = x_cat)
    mapped <- state$mapping[dt, on = "c"]
    means <- mapped$mean_val
    sds <- mapped$sd_val
    means[is.na(means)] <- state$default_mean
    sds[is.na(sds)] <- state$default_sd
    
    res <- ifelse(sds == 0, 0, (x_num - means) / sds)
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbz")
)

# Quantile Binning
evo_transformers$quantile_binning <- create_transformer(
  name = "quantile_binning",
  type = "unary",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    Q <- if (!is.null(gene$params$Q)) gene$params$Q else 5
    boundaries <- tryCatch({
      stats::quantile(x, probs = seq(0, 1, length.out = Q + 1), na.rm = TRUE, names = FALSE)
    }, error = function(e) {
      c(-Inf, rep(0, Q - 1), Inf)
    })
    boundaries <- unique(boundaries)
    list(boundaries = boundaries)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    boundaries <- state$boundaries
    if (length(boundaries) <= 1) return(rep(1L, length(x)))
    res <- findInterval(x, boundaries, all.inside = TRUE)
    res[is.na(res)] <- 0
    as.integer(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "qb")
)

# Log Binning
evo_transformers$log_binning <- create_transformer(
  name = "log_binning",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    base <- if (!is.null(gene$params$base)) gene$params$base else 2
    res <- floor(log(abs(x) + 1, base = base))
    res[!is.finite(res)] <- 0
    as.integer(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "lb")
)

# Quantile Binning (Categorical)
evo_transformers$quantile_binning_cat <- create_transformer(
  name = "quantile_binning_cat",
  type = "unary",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    Q <- if (!is.null(gene$params$Q)) gene$params$Q else 5
    boundaries <- tryCatch({
      stats::quantile(x, probs = seq(0, 1, length.out = Q + 1), na.rm = TRUE, names = FALSE)
    }, error = function(e) {
      c(-Inf, rep(0, Q - 1), Inf)
    })
    boundaries <- unique(boundaries)
    list(boundaries = boundaries)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    boundaries <- state$boundaries
    if (length(boundaries) <= 1) return(rep(1L, length(x)))
    res <- findInterval(x, boundaries, all.inside = TRUE)
    res[is.na(res)] <- 0
    as.integer(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "qbc")
)

# Log Binning (Categorical)
evo_transformers$log_binning_cat <- create_transformer(
  name = "log_binning_cat",
  type = "unary",
  input_type = "numeric",
  output_type = "categorical",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    base <- if (!is.null(gene$params$base)) gene$params$base else 2
    res <- floor(log(abs(x) + 1, base = base))
    res[!is.finite(res)] <- 0
    as.integer(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "lbc")
)

# Normalized Difference
evo_transformers$normalized_difference <- create_transformer(
  name = "normalized_difference",
  type = "binary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    a <- data[[input_cols[1]]]
    b <- data[[input_cols[2]]]
    res <- (a - b) / (abs(a) + abs(b) + 1e-8)
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "nd")
)

# Log Ratio
evo_transformers$log_ratio <- create_transformer(
  name = "log_ratio",
  type = "binary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    a <- data[[input_cols[1]]]
    b <- data[[input_cols[2]]]
    log1p(abs(a)) - log1p(abs(b))
  },
  name_generator = function(gene) .gene_col_name(gene, "lr")
)

# Random Projection
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
evo_transformers$one_hot_encode <- create_transformer(
  name = "one_hot_encode",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    
    # Calculate category frequencies
    freq <- table(x, useNA = "no")
    df_freq <- as.data.frame(freq, stringsAsFactors = FALSE)
    if (nrow(df_freq) == 0) {
      return(list(top_categories = character(0)))
    }
    names(df_freq) <- c("category", "count")
    total_n <- length(x[!is.na(x)])
    df_freq$pct <- df_freq$count / max(1, total_n)
    
    # Keep categories with frequency >= 5%, up to a maximum of 5 categories
    # Sorted by frequency descending
    df_freq <- df_freq[order(df_freq$count, decreasing = TRUE), ]
    top_cats <- df_freq$category[df_freq$pct >= 0.05]
    if (length(top_cats) > 5) {
      top_cats <- top_cats[1:5]
    }
    
    list(top_categories = top_cats)
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    
    if (is.null(state) || is.null(state$top_categories)) {
      return(rep(0, length(x)))
    }
    
    top_cats <- state$top_categories
    
    if (comp_idx == 6) {
      # "other" category: not in top categories, or NA
      as.numeric(!(x %in% top_cats) | is.na(x))
    } else {
      # 1 to 5: check if category exists at this index
      if (comp_idx <= length(top_cats)) {
        target_cat <- top_cats[comp_idx]
        as.numeric(!is.na(x) & x == target_cat)
      } else {
        # Index is out of bounds (fewer than comp_idx categories kept)
        rep(0, length(x))
      }
    }
  },
  name_generator = function(gene) .gene_col_name(gene, "ohe")
)

# --- ADDITIONAL TRANSFORMERS ---

# Datetime Feature Extractor
evo_transformers$datetime_extract <- create_transformer(
  name = "datetime_extract",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt_parsed <- tryCatch({
      if (inherits(x, c("POSIXct", "POSIXlt", "Date"))) {
        as.POSIXct(x)
      } else {
        as.POSIXct(as.character(x), tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
      }
    }, error = function(e) {
      as.POSIXct(rep(NA, length(x)))
    })
    comp <- if (!is.null(gene$params$component)) gene$params$component else "month"
    res <- switch(comp,
      year = as.integer(format(dt_parsed, "%Y")),
      month = as.integer(format(dt_parsed, "%m")),
      day = as.integer(format(dt_parsed, "%d")),
      hour = as.integer(format(dt_parsed, "%H")),
      day_of_week = as.integer(format(dt_parsed, "%u")),
      weekend = as.integer(format(dt_parsed, "%u") %in% c("6", "7")),
      rep(0L, length(x))
    )
    res[is.na(res)] <- 0L
    as.numeric(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "dt")
)

# Multiclass Target Encoding
evo_transformers$target_encode_multiclass <- create_transformer(
  name = "target_encode_multiclass",
  type = "supervised_unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- data[[target_col]]
    
    classes <- sort(unique(y))
    mappings <- list()
    global_means <- list()
    smoothing <- 10
    
    for (k in seq_along(classes)) {
      y_bin <- as.numeric(y == classes[k])
      global_mean <- mean(y_bin, na.rm = TRUE)
      global_means[[k]] <- global_mean
      
      dt <- data.table::data.table(x = x, y = y_bin)
      stats <- dt[, .(mean = mean(y, na.rm = TRUE), n = .N), by = x]
      stats[, smoothed := (n * mean + smoothing * global_mean) / (n + smoothing)]
      
      mapping <- stats[, .(x, smoothed)]
      data.table::setkey(mapping, x)
      mappings[[k]] <- mapping
    }
    
    list(mappings = mappings, global_means = global_means, classes = classes)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    
    if (is.null(state) || is.null(state$mappings) || comp_idx > length(state$mappings)) {
      return(rep(0, length(x)))
    }
    
    dt <- data.table::data.table(x = x)
    mapping <- state$mappings[[comp_idx]]
    global_mean <- state$global_means[[comp_idx]]
    
    res <- mapping[dt, on = "x"]$smoothed
    res[is.na(res)] <- global_mean
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "temc")
)


# Genie Centroid Distance
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
    compute_dists <- function() lapply(state$centroids, function(c) sqrt(rowSums(sweep(x_test, 2, c, "-")^2)))
    if (is.null(state$preds_cache)) return(compute_dists()[[comp_idx]])
    key <- digest::digest(x_test, algo = "xxhash64")
    if (!exists(key, envir = state$preds_cache)) assign(key, compute_dists(), envir = state$preds_cache)
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
    compute_dists <- function() lapply(state$centroids, function(c) sqrt(rowSums(sweep(x_test, 2, c, "-")^2)))
    if (is.null(state$preds_cache)) return(compute_dists()[[comp_idx]])
    key <- digest::digest(x_test, algo = "xxhash64")
    if (!exists(key, envir = state$preds_cache)) assign(key, compute_dists(), envir = state$preds_cache)
    get(key, envir = state$preds_cache)[[comp_idx]]
  },
  name_generator = function(gene) .gene_col_name(gene, "lmcd")
)
