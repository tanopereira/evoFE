#' Create a transformer definition
#'
#' @param name Transformer name
#' @param type Type: "unary", "binary", "supervised_unary"
#' @param input_type Type of input: "numeric" or "categorical"
#' @param fit_func function(data, input_cols, target_col = NULL) returning state
#' @param apply_func function(data, input_cols, state = NULL) returning new column vector
#' @param name_generator function(input_cols) returning output column name
#' @param allow_replace Logical. Whether column sampling allows replacement.
#' @export
create_transformer <- function(name, type, input_type = "numeric", fit_func = NULL, apply_func, name_generator, allow_replace = FALSE) {
  structure(
    list(
      name = name,
      type = type,
      input_type = input_type,
      fit_func = fit_func,
      apply_func = apply_func,
      name_generator = name_generator,
      allow_replace = allow_replace
    ),
    class = "evo_transformer"
  )
}

#' @export
evo_transformers <- list()

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
    Reduce(`+`, lapply(input_cols, function(c) data[[c]]))
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
    data[[input_cols[1]]] - data[[input_cols[2]]]
  },
  name_generator = function(gene) paste0("((", gene$input_cols[1], "-", gene$input_cols[2], "))")
)

evo_transformers$multiply <- create_transformer(
  name = "multiply",
  type = "multivariate",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    Reduce(`*`, lapply(input_cols, function(c) data[[c]]))
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
    tryCatch({
      pca_model <- stats::prcomp(x, center = TRUE, scale. = TRUE)
      list(model = pca_model, valid = TRUE)
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
    preds <- stats::predict(state$model, x)
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
    tryCatch({
      res <- svd(x, nu = 0, nv = comp_idx)
      list(v = res$v, valid = TRUE)
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
    if (comp_idx > ncol(state$v)) comp_idx <- ncol(state$v)
    as.vector(x %*% state$v[, comp_idx])
  },
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("SVD", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# K-Means Distance
evo_transformers$kmeans_dist <- create_transformer(
  name = "kmeans_dist",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    tryCatch({
      km <- stats::kmeans(x, centers = k, nstart = 1)
      list(centers = km$centers, valid = TRUE)
    }, error = function(e) {
      list(centers = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    if (comp_idx > nrow(state$centers)) comp_idx <- nrow(state$centers)
    sqrt(rowSums((sweep(x, 2, state$centers[comp_idx, ]))^2))
  },
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("KMD", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# K-Means Cluster ID
evo_transformers$kmeans_cluster <- create_transformer(
  name = "kmeans_cluster",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    tryCatch({
      km <- stats::kmeans(x, centers = k, nstart = 1)
      list(centers = km$centers, valid = TRUE)
    }, error = function(e) {
      list(centers = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    apply(x, 1, function(row) {
      which.min(colSums((t(state$centers) - row)^2))
    })
  },
  name_generator = function(gene) {
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    paste0("KMC_", k, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# GenieClust Distance
evo_transformers$genie_dist <- create_transformer(
  name = "genie_dist",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    tryCatch({
      # genieclust requires >= 2 columns. If 1, duplicate it to trick the algorithm.
      if (ncol(x) == 1) {
        x_genie <- cbind(x, x)
      } else {
        x_genie <- x
      }
      
      # Prevent genieclust C++ segfault on zero-variance matrices
      if (max(x_genie) == min(x_genie)) {
        return(list(centers = NULL, valid = FALSE))
      }
      
      h <- genieclust::genie(x_genie, k = k)
      labels <- as.integer(h)
      
      # Calculate geometric centroids of the hierarchical clusters
      centers <- matrix(0, nrow = k, ncol = ncol(x))
      for (i in 1:k) {
        if (sum(labels == i) > 1) {
          centers[i, ] <- colMeans(x[labels == i, , drop = FALSE])
        } else if (sum(labels == i) == 1) {
          centers[i, ] <- x[labels == i, ]
        }
      }
      list(centers = centers, valid = TRUE)
    }, error = function(e) {
      list(centers = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    if (comp_idx > nrow(state$centers)) comp_idx <- nrow(state$centers)
    sqrt(rowSums((sweep(x, 2, state$centers[comp_idx, ]))^2))
  },
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("Genie", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)

# GenieClust Cluster ID
evo_transformers$genie_cluster <- create_transformer(
  name = "genie_cluster",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    tryCatch({
      if (ncol(x) == 1) {
        x_genie <- cbind(x, x)
      } else {
        x_genie <- x
      }
      
      # Prevent genieclust C++ segfault on zero-variance matrices
      if (max(x_genie) == min(x_genie)) {
        return(list(centers = NULL, valid = FALSE))
      }
      
      h <- genieclust::genie(x_genie, k = k)
      labels <- as.integer(h)
      centers <- matrix(0, nrow = k, ncol = ncol(x))
      for (i in 1:k) {
        if (sum(labels == i) > 1) {
          centers[i, ] <- colMeans(x[labels == i, , drop = FALSE])
        } else if (sum(labels == i) == 1) {
          centers[i, ] <- x[labels == i, ]
        }
      }
      list(centers = centers, valid = TRUE)
    }, error = function(e) {
      list(centers = NULL, valid = FALSE)
    })
  },
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    if (is.null(state) || !state$valid) return(rep(0, nrow(data)))
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    apply(x, 1, function(row) {
      which.min(colSums((t(state$centers) - row)^2))
    })
  },
  name_generator = function(gene) {
    k <- if (!is.null(gene$params$k)) gene$params$k else 3
    paste0("GenieC_", k, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
  }
)


# UMAP Coordinate
evo_transformers$umap <- create_transformer(
  name = "umap",
  type = "multivariate",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.matrix(data[, input_cols, with = FALSE])
    x[is.na(x)] <- 0
    tryCatch({
      # Train UMAP to extract 3 dimensions
      n_comp <- min(3, ncol(x))
      u <- uwot::umap(x, n_components = n_comp, n_neighbors = 15, ret_model = TRUE, n_threads = 1, n_sgd_threads = 1)
      list(model = u, valid = TRUE)
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
    
    # uwot prediction
    tryCatch({
      preds <- uwot::umap_transform(x, state$model, n_threads = 1, n_sgd_threads = 1)
      if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
      preds[, comp_idx]
    }, error = function(e) {
      rep(0, nrow(x))
    })
  },
  name_generator = function(gene) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    paste0("UMAP", comp_idx, "(", paste(substr(gene$input_cols, 1, 3), collapse = "_"), ")")
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
