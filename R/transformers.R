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
#' @return An \code{evo_transformer} S3 object: a list with elements
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
#' @param transformer An object of class \code{evo_transformer} created via \code{create_transformer}.
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
  name_generator = function(gene) paste0("log(", gene$input_cols[1], ")")
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
  name_generator = function(gene) paste0("sqrt(", gene$input_cols[1], ")")
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
  name_generator = function(gene) paste0("rec(", gene$input_cols[1], ")")
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
  name_generator = function(gene) paste0("((", paste(gene$input_cols, collapse = "+"), "))"),
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
  name_generator = function(gene) paste0("((", gene$input_cols[1], "-", gene$input_cols[2], "))")
)

evo_transformers$multiply <- create_transformer(
  name = "multiply",
  type = "multivariate",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    Reduce(`*`, lapply(input_cols, function(c) as.numeric(data[[c]])))
  },
  name_generator = function(gene) paste0("((", paste(gene$input_cols, collapse = "*"), "))"),
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
  name_generator = function(gene) paste0("((", gene$input_cols[1], "/", gene$input_cols[2], "))")
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
  name_generator = function(gene) paste0("te_", gene$input_cols[1])
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
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("PCA", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
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
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("SVD", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
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
  name_generator = function(gene) paste0("freq_", gene$input_cols[1])
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
  name_generator = function(gene) paste0("mean_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
  name_generator = function(gene) paste0("sd_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
  name_generator = function(gene) paste0("max_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
  name_generator = function(gene) paste0("min_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
    
    n_neighbors <- 15
    if (nrow(x) < 15) {
      n_neighbors <- max(2, nrow(x) - 1)
    }
    
    verbose <- is_verbose()
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[UMAP Fit] Start on %d rows, %d cols. n_neighbors = %d", nrow(x), ncol(x), n_neighbors))
    }
    
    tryCatch({
      threads <- getOption("evoFE.threads", 1)
      C <- max(2L, as.integer(round(log2(ncol(x)))))
      model <- uwot::umap(x, n_neighbors = n_neighbors, n_components = C, 
                          ret_model = TRUE, n_threads = threads, verbose = FALSE, init = "random")
      if (verbose) {
        elapsed <- difftime(Sys.time(), start_time, units = "secs")
        message(sprintf("[UMAP Fit] Completed in %.3f seconds", as.numeric(elapsed)))
      }
      
      preds_cache <- new.env(hash = TRUE, parent = emptyenv())
      if (!is.null(model$embedding)) {
        x_hash <- digest::digest(x, algo = "xxhash64")
        assign(x_hash, model$embedding, envir = preds_cache)
      }
      
      list(model = model, valid = TRUE, preds_cache = preds_cache)
    }, error = function(e) {
      if (verbose) {
        message(sprintf("[UMAP Fit] Failed: %s", conditionMessage(e)))
      }
      list(model = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    verbose <- is_verbose()
    if (is.null(state) || !state$valid) {
      if (verbose) {
        message("[UMAP Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(0, nrow(data)))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[UMAP Apply] Start on %d rows, %d cols. Component = %d", nrow(x), ncol(x), comp_idx))
    }
    
    if (is.null(state$preds_cache)) {
      threads <- getOption("evoFE.threads", 1)
      preds <- uwot::umap_transform(x, model = state$model, n_threads = threads, verbose = FALSE)
    } else {
      x_key <- digest::digest(x, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
        if (verbose) {
          message("  Retrieve UMAP projection from cache.")
        }
      } else {
        threads <- getOption("evoFE.threads", 1)
        preds <- uwot::umap_transform(x, model = state$model, n_threads = threads, verbose = FALSE)
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    
    if (verbose) {
      elapsed <- difftime(Sys.time(), start_time, units = "secs")
      message(sprintf("[UMAP Apply] Completed in %.3f seconds", as.numeric(elapsed)))
    }
    preds[, comp_idx]
  },
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("UMAP", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# MST-based Anomaly Score
evo_transformers$mst_score <- create_transformer(
  name = "mst_score",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    if (length(input_cols) < 2) {
      return(list(x_train = NULL, scores = NULL, valid = FALSE))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    verbose <- is_verbose()
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[MST Fit] Start on %d rows, %d cols.", nrow(x), ncol(x)))
    }
    
    tryCatch({
      t0 <- Sys.time()
      dt_x <- data.table::as.data.table(x)
      cols <- names(dt_x)
      dt_x[, first_id := .I[1], by = cols]
      first_ids <- dt_x$first_id
      unique_ids <- unique(first_ids)
      x_unique <- x[unique_ids, , drop = FALSE]
      if (verbose) {
        message(sprintf("  Deduplication: %d unique rows out of %d. Elapsed: %.3f s", nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (ncol(x) < 2 || nrow(x_unique) <= 5) {
        return(list(x_train = NULL, scores = NULL, valid = FALSE))
      }
      
      if (!requireNamespace("quitefastmst", quietly = TRUE)) {
        stop("quitefastmst package is not available")
      }
      
      t0 <- Sys.time()
      # Handle max_clustering_size
      max_size <- getOption("evoFE.max_clustering_size", 5000)
      if (is.null(max_size) || !is.numeric(max_size)) max_size <- 0
      if (max_size > 0 && nrow(x_unique) > max_size) {
        old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
        on.exit({
          if (!is.null(old_seed)) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
          } else {
            if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
          }
        }, add = TRUE)
        sampled_rows <- sample(seq_len(nrow(x_unique)), max_size)
        x_sampled <- x_unique[sampled_rows, , drop = FALSE]
        if (verbose) {
          message(sprintf("  Downsampling: sampled %d rows from %d. Elapsed: %.3f s", nrow(x_sampled), nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
        }
      } else {
        x_sampled <- x_unique
      }
      
      if (nrow(x_sampled) <= 5) {
        return(list(x_train = NULL, scores = NULL, valid = FALSE))
      }
      
      t0 <- Sys.time()
      res_mst <- quitefastmst::mst_euclid(x_sampled)
      if (verbose) {
        message(sprintf("  quitefastmst::mst_euclid: computed MST on %d rows. Elapsed: %.3f s", nrow(x_sampled), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      t0 <- Sys.time()
      dt_edges <- data.table::data.table(
        node = c(res_mst$mst.index[, 1], res_mst$mst.index[, 2]),
        dist = c(res_mst$mst.dist, res_mst$mst.dist)
      )
      scores_sampled <- dt_edges[, .(score = max(dist)), by = node][order(node)]$score
      if (verbose) {
        message(sprintf("  Score aggregation: computed max distance for each node. Elapsed: %.3f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (verbose) {
        elapsed <- difftime(Sys.time(), start_time, units = "secs")
        message(sprintf("[MST Fit] Completed in %.3f seconds", as.numeric(elapsed)))
      }
      list(x_train = x_sampled, scores = as.numeric(scores_sampled), valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      if (verbose) {
        message(sprintf("[MST Fit] Failed: %s", conditionMessage(e)))
      }
      list(x_train = NULL, scores = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    verbose <- is_verbose()
    if (is.null(state) || !state$valid) {
      if (verbose) {
        message("[MST Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(0, nrow(data)))
    }
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0
    storage.mode(x_test) <- "double"
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[MST Apply] Start on %d test rows against %d train rows.", nrow(x_test), nrow(state$x_train)))
    }
    
    if (is.null(state$preds_cache)) {
      # Fallback
      x_train <- state$x_train
      train_scores <- state$scores
      if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
        preds <- train_scores
      } else {
        nearest_indices <- tryCatch({
          quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
        }, error = function(e) {
          apply(x_test, 1, function(row) {
            dists <- colSums((t(x_train) - row)^2)
            which.min(dists)
          })
        })
        preds <- train_scores[nearest_indices]
      }
    } else {
      x_key <- digest::digest(x_test, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
        if (verbose) {
          message("  Retrieve MST projection from cache.")
        }
      } else {
        x_train <- state$x_train
        train_scores <- state$scores
        if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
          preds <- train_scores
        } else {
          nearest_indices <- tryCatch({
            t0 <- Sys.time()
            res <- quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
            if (verbose) {
              message(sprintf("  quitefastmst::knn_euclid: projected %d test rows on %d train rows. Elapsed: %.3f s", nrow(x_test), nrow(x_train), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          }, error = function(e) {
            t0 <- Sys.time()
            res <- apply(x_test, 1, function(row) {
              dists <- colSums((t(x_train) - row)^2)
              which.min(dists)
            })
            if (verbose) {
              message(sprintf("  R fallback KNN (Euclidean distance): projected %d test rows. Elapsed: %.3f s", nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          })
          preds <- train_scores[nearest_indices]
        }
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (verbose) {
      elapsed <- difftime(Sys.time(), start_time, units = "secs")
      message(sprintf("[MST Apply] Completed in %.3f seconds", as.numeric(elapsed)))
    }
    preds
  },
  name_generator = function(gene) {
    paste0("MSTScore(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# Genie Clustering
evo_transformers$genie <- create_transformer(
  name = "genie",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    if (length(input_cols) < 2) {
      return(list(x_train = NULL, labels = NULL, valid = FALSE))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    
    verbose <- is_verbose()
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Genie Fit] Start on %d rows, %d cols. k = %d", nrow(x), ncol(x), k))
    }
    
    tryCatch({
      t0 <- Sys.time()
      dt_x <- data.table::as.data.table(x)
      cols <- names(dt_x)
      dt_x[, first_id := .I[1], by = cols]
      first_ids <- dt_x$first_id
      unique_ids <- unique(first_ids)
      x_unique <- x[unique_ids, , drop = FALSE]
      if (verbose) {
        message(sprintf("  Deduplication: %d unique rows out of %d. Elapsed: %.3f s", nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (ncol(x) < 2 || nrow(x_unique) <= 5 || nrow(x_unique) < k) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      if (!requireNamespace("genieclust", quietly = TRUE)) {
        stop("genieclust package is not available")
      }
      
      t0 <- Sys.time()
      # Handle max_clustering_size
      max_size <- getOption("evoFE.max_clustering_size", 5000)
      if (is.null(max_size) || !is.numeric(max_size)) max_size <- 0
      if (max_size > 0 && nrow(x_unique) > max_size) {
        old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
        on.exit({
          if (!is.null(old_seed)) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
          } else {
            if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
          }
        }, add = TRUE)
        sampled_rows <- sample(seq_len(nrow(x_unique)), max_size)
        x_sampled <- x_unique[sampled_rows, , drop = FALSE]
        if (verbose) {
          message(sprintf("  Downsampling: sampled %d rows from %d. Elapsed: %.3f s", nrow(x_sampled), nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
        }
      } else {
        x_sampled <- x_unique
      }
      
      if (nrow(x_sampled) <= 5 || nrow(x_sampled) < k) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      t0 <- Sys.time()
      clust_sampled <- genieclust::genie(x_sampled, k = k)
      if (verbose) {
        message(sprintf("  genieclust::genie: computed clusters. Elapsed: %.3f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (verbose) {
        elapsed <- difftime(Sys.time(), start_time, units = "secs")
        message(sprintf("[Genie Fit] Completed in %.3f seconds", as.numeric(elapsed)))
      }
      list(x_train = x_sampled, labels = as.integer(clust_sampled), valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      if (verbose) {
        message(sprintf("[Genie Fit] Failed: %s", conditionMessage(e)))
      }
      list(x_train = NULL, labels = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    verbose <- is_verbose()
    if (is.null(state) || !state$valid) {
      if (verbose) {
        message("[Genie Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(1, nrow(data)))
    }
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0
    storage.mode(x_test) <- "double"
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Genie Apply] Start on %d test rows against %d train rows.", nrow(x_test), nrow(state$x_train)))
    }
    
    if (is.null(state$preds_cache)) {
      # Fallback
      x_train <- state$x_train
      train_labels <- state$labels
      if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
        preds <- train_labels
      } else {
        nearest_indices <- tryCatch({
          quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
        }, error = function(e) {
          apply(x_test, 1, function(row) {
            dists <- colSums((t(x_train) - row)^2)
            which.min(dists)
          })
        })
        preds <- train_labels[nearest_indices]
      }
    } else {
      x_key <- digest::digest(x_test, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
        if (verbose) {
          message("  Retrieve Genie projection from cache.")
        }
      } else {
        x_train <- state$x_train
        train_labels <- state$labels
        if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
          preds <- train_labels
        } else {
          nearest_indices <- tryCatch({
            t0 <- Sys.time()
            res <- quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
            if (verbose) {
              message(sprintf("  quitefastmst::knn_euclid: projected %d test rows on %d train rows. Elapsed: %.3f s", nrow(x_test), nrow(x_train), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          }, error = function(e) {
            t0 <- Sys.time()
            res <- apply(x_test, 1, function(row) {
              dists <- colSums((t(x_train) - row)^2)
              which.min(dists)
            })
            if (verbose) {
              message(sprintf("  R fallback KNN (Euclidean distance): projected %d test rows. Elapsed: %.3f s", nrow(x_test), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          })
          preds <- train_labels[nearest_indices]
        }
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (verbose) {
      elapsed <- difftime(Sys.time(), start_time, units = "secs")
      message(sprintf("[Genie Apply] Completed in %.3f seconds", as.numeric(elapsed)))
    }
    preds
  },
  name_generator = function(gene) {
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    paste0("Genie", k, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
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
  name_generator = function(gene) paste0("ratio_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
  name_generator = function(gene) paste0("zscore_", gene$input_cols[2], "_by_", gene$input_cols[1])
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
  name_generator = function(gene) {
    Q <- if (!is.null(gene$params$Q)) gene$params$Q else 5
    paste0("qbin", Q, "(", gene$input_cols[1], ")")
  }
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
  name_generator = function(gene) {
    base <- if (!is.null(gene$params$base)) gene$params$base else 2
    paste0("logbin", base, "(", gene$input_cols[1], ")")
  }
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
  name_generator = function(gene) {
    Q <- if (!is.null(gene$params$Q)) gene$params$Q else 5
    paste0("qbin_cat", Q, "(", gene$input_cols[1], ")")
  }
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
  name_generator = function(gene) {
    base <- if (!is.null(gene$params$base)) gene$params$base else 2
    paste0("logbin_cat", base, "(", gene$input_cols[1], ")")
  }
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
  name_generator = function(gene) paste0("normdiff(", gene$input_cols[1], "_", gene$input_cols[2], ")")
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
  name_generator = function(gene) paste0("logratio(", gene$input_cols[1], "_", gene$input_cols[2], ")")
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
  name_generator = function(gene) {
    paste0("RP(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# Lumbermark Clustering
evo_transformers$lumbermark <- create_transformer(
  name = "lumbermark",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    if (length(input_cols) < 2) {
      return(list(x_train = NULL, labels = NULL, valid = FALSE))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    
    verbose <- is_verbose()
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Lumbermark Fit] Start on %d rows, %d cols. k = %d", nrow(x), ncol(x), k))
    }
    
    tryCatch({
      t0 <- Sys.time()
      dt_x <- data.table::as.data.table(x)
      cols <- names(dt_x)
      dt_x[, first_id := .I[1], by = cols]
      first_ids <- dt_x$first_id
      unique_ids <- unique(first_ids)
      x_unique <- x[unique_ids, , drop = FALSE]
      if (verbose) {
        message(sprintf("  Deduplication: %d unique rows out of %d. Elapsed: %.3f s", nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (ncol(x) < 2 || nrow(x_unique) <= 5 || nrow(x_unique) < 2 * k) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      t0 <- Sys.time()
      # Handle max_clustering_size
      max_size <- getOption("evoFE.max_clustering_size", 5000)
      if (is.null(max_size) || !is.numeric(max_size)) max_size <- 0
      if (max_size > 0 && nrow(x_unique) > max_size) {
        old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
        on.exit({
          if (!is.null(old_seed)) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
          } else {
            if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
          }
        }, add = TRUE)
        sampled_rows <- sample(seq_len(nrow(x_unique)), max_size)
        x_sampled <- x_unique[sampled_rows, , drop = FALSE]
        if (verbose) {
          message(sprintf("  Downsampling: sampled %d rows from %d. Elapsed: %.3f s", nrow(x_sampled), nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
        }
      } else {
        x_sampled <- x_unique
      }
      
      if (nrow(x_sampled) <= 5 || nrow(x_sampled) < 2 * k) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      if (!requireNamespace("lumbermark", quietly = TRUE)) {
        stop("lumbermark package is not available")
      }
      
      t0 <- Sys.time()
      clust_sampled <- lumbermark::lumbermark(x_sampled, k = k, min_cluster_size = 2)
      if (verbose) {
        message(sprintf("  lumbermark::lumbermark: computed clusters. Elapsed: %.3f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (verbose) {
        elapsed <- difftime(Sys.time(), start_time, units = "secs")
        message(sprintf("[Lumbermark Fit] Completed in %.3f seconds", as.numeric(elapsed)))
      }
      list(x_train = x_sampled, labels = as.integer(clust_sampled), valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      if (verbose) {
        message(sprintf("[Lumbermark Fit] Failed: %s", conditionMessage(e)))
      }
      list(x_train = NULL, labels = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    verbose <- is_verbose()
    if (is.null(state) || !state$valid) {
      if (verbose) {
        message("[Lumbermark Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(1, nrow(data)))
    }
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0
    storage.mode(x_test) <- "double"
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Lumbermark Apply] Start on %d test rows against %d train rows.", nrow(x_test), nrow(state$x_train)))
    }
    
    if (is.null(state$preds_cache)) {
      x_train <- state$x_train
      train_labels <- state$labels
      if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
        preds <- train_labels
      } else {
        nearest_indices <- tryCatch({
          quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
        }, error = function(e) {
          apply(x_test, 1, function(row) {
            dists <- colSums((t(x_train) - row)^2)
            which.min(dists)
          })
        })
        preds <- train_labels[nearest_indices]
      }
    } else {
      x_key <- digest::digest(x_test, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
        if (verbose) {
          message("  Retrieve Lumbermark projection from cache.")
        }
      } else {
        x_train <- state$x_train
        train_labels <- state$labels
        if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
          preds <- train_labels
        } else {
          nearest_indices <- tryCatch({
            t0 <- Sys.time()
            res <- quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
            if (verbose) {
              message(sprintf("  quitefastmst::knn_euclid: projected %d test rows on %d train rows. Elapsed: %.3f s", nrow(x_test), nrow(x_train), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          }, error = function(e) {
            apply(x_test, 1, function(row) {
              dists <- colSums((t(x_train) - row)^2)
              which.min(dists)
            })
          })
          preds <- train_labels[nearest_indices]
        }
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (verbose) {
      elapsed <- difftime(Sys.time(), start_time, units = "secs")
      message(sprintf("[Lumbermark Apply] Completed in %.3f seconds", as.numeric(elapsed)))
    }
    preds
  },
  name_generator = function(gene) {
    k <- if (!is.null(gene$params$k)) gene$params$k else 2
    paste0("Lumb", k, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# Deadwood Anomaly Detection
evo_transformers$deadwood <- create_transformer(
  name = "deadwood",
  type = "multivariate",
  input_type = "numeric",
  output_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    if (length(input_cols) < 2) {
      return(list(x_train = NULL, labels = NULL, valid = FALSE))
    }
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    storage.mode(x) <- "double"
    
    verbose <- is_verbose()
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Deadwood Fit] Start on %d rows, %d cols.", nrow(x), ncol(x)))
    }
    
    tryCatch({
      t0 <- Sys.time()
      dt_x <- data.table::as.data.table(x)
      cols <- names(dt_x)
      dt_x[, first_id := .I[1], by = cols]
      first_ids <- dt_x$first_id
      unique_ids <- unique(first_ids)
      x_unique <- x[unique_ids, , drop = FALSE]
      if (verbose) {
        message(sprintf("  Deduplication: %d unique rows out of %d. Elapsed: %.3f s", nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (ncol(x) < 2 || nrow(x_unique) <= 5) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      t0 <- Sys.time()
      # Handle max_clustering_size
      max_size <- getOption("evoFE.max_clustering_size", 5000)
      if (is.null(max_size) || !is.numeric(max_size)) max_size <- 0
      if (max_size > 0 && nrow(x_unique) > max_size) {
        old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
        on.exit({
          if (!is.null(old_seed)) {
            assign(".Random.seed", old_seed, envir = .GlobalEnv)
          } else {
            if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
          }
        }, add = TRUE)
        sampled_rows <- sample(seq_len(nrow(x_unique)), max_size)
        x_sampled <- x_unique[sampled_rows, , drop = FALSE]
        if (verbose) {
          message(sprintf("  Downsampling: sampled %d rows from %d. Elapsed: %.3f s", nrow(x_sampled), nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
        }
      } else {
        x_sampled <- x_unique
      }
      
      if (nrow(x_sampled) <= 5) {
        return(list(x_train = NULL, labels = NULL, valid = FALSE))
      }
      
      if (!requireNamespace("deadwood", quietly = TRUE)) {
        stop("deadwood package is not available")
      }
      
      t0 <- Sys.time()
      outliers_sampled <- deadwood::deadwood(x_sampled)
      if (verbose) {
        message(sprintf("  deadwood::deadwood: computed outliers. Elapsed: %.3f s", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      }
      
      if (verbose) {
        elapsed <- difftime(Sys.time(), start_time, units = "secs")
        message(sprintf("[Deadwood Fit] Completed in %.3f seconds", as.numeric(elapsed)))
      }
      list(x_train = x_sampled, labels = as.numeric(outliers_sampled), valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
    }, error = function(e) {
      if (verbose) {
        message(sprintf("[Deadwood Fit] Failed: %s", conditionMessage(e)))
      }
      list(x_train = NULL, labels = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    verbose <- is_verbose()
    if (is.null(state) || !state$valid) {
      if (verbose) {
        message("[Deadwood Apply] Skipped because fitted state is invalid or NULL.")
      }
      return(rep(0, nrow(data)))
    }
    x_test <- as.matrix(data[, input_cols, with = FALSE])
    x_test[is.na(x_test)] <- 0
    storage.mode(x_test) <- "double"
    if (verbose) {
      start_time <- Sys.time()
      message(sprintf("[Deadwood Apply] Start on %d test rows against %d train rows.", nrow(x_test), nrow(state$x_train)))
    }
    
    if (is.null(state$preds_cache)) {
      x_train <- state$x_train
      train_labels <- state$labels
      if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
        preds <- train_labels
      } else {
        nearest_indices <- tryCatch({
          quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
        }, error = function(e) {
          apply(x_test, 1, function(row) {
            dists <- colSums((t(x_train) - row)^2)
            which.min(dists)
          })
        })
        preds <- train_labels[nearest_indices]
      }
    } else {
      x_key <- digest::digest(x_test, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        preds <- get(x_key, envir = state$preds_cache)
        if (verbose) {
          message("  Retrieve Deadwood projection from cache.")
        }
      } else {
        x_train <- state$x_train
        train_labels <- state$labels
        if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
          preds <- train_labels
        } else {
          nearest_indices <- tryCatch({
            t0 <- Sys.time()
            res <- quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1]
            if (verbose) {
              message(sprintf("  quitefastmst::knn_euclid: projected %d test rows on %d train rows. Elapsed: %.3f s", nrow(x_test), nrow(x_train), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
            }
            res
          }, error = function(e) {
            apply(x_test, 1, function(row) {
              dists <- colSums((t(x_train) - row)^2)
              which.min(dists)
            })
          })
          preds <- train_labels[nearest_indices]
        }
        assign(x_key, preds, envir = state$preds_cache)
      }
    }
    
    if (verbose) {
      elapsed <- difftime(Sys.time(), start_time, units = "secs")
      message(sprintf("[Deadwood Apply] Completed in %.3f seconds", as.numeric(elapsed)))
    }
    preds
  },
  name_generator = function(gene) {
    paste0("Deadwood(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
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
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    idx_str <- if (comp_idx == 6) "other" else as.character(comp_idx)
    paste0("ohe_", idx_str, "_", gene$input_cols[1])
  }
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
  name_generator = function(gene) {
    comp <- if (!is.null(gene$params$component)) gene$params$component else "month"
    paste0(comp, "_", gene$input_cols[1])
  }
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
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("te_mc_", comp_idx, "_", gene$input_cols[1])
  }
)


