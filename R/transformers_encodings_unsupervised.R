# Unsupervised categorical encoders: frequency, one-hot, hashing, n-gram & similarity encodings.
# Split out of transformers.R; loaded after it (alphabetical file order).

.fit_frequency_mapping <- function(x) {
  dt <- data.table::data.table(x = x)
  mapping <- dt[, .N, by = x]
  data.table::setkey(mapping, x)
  def_val <- if (nrow(mapping) > 0) stats::median(mapping$N) else 0
  list(mapping = mapping, default_val = def_val)
}

.apply_frequency_mapping <- function(x, state) {
  if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
    def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
    return(rep(as.numeric(def_val), if (!is.null(x)) length(x) else 0L))
  }
  dt <- data.table::data.table(x = x)
  res <- state$mapping[dt, on = "x"]$N
  if (is.null(res)) {
    def_val <- if (!is.null(state$default_val)) state$default_val else 0
    return(rep(as.numeric(def_val), length(x)))
  }
  res[is.na(res)] <- state$default_val
  as.numeric(res)
}

evo_transformers$frequency_encode <- create_transformer(
  name = "frequency_encode",
  type = "unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col = NULL) {
    .fit_frequency_mapping(data[[gene$input_cols[1]]])
  },
  apply_func = function(data, gene, state) {
    .apply_frequency_mapping(data[[gene$input_cols[1]]], state)
  },
  name_generator = function(gene) .gene_col_name(gene, "freq")
)

evo_transformers$numeric_freq <- create_transformer(
  name = "numeric_freq",
  type = "unary",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    .fit_frequency_mapping(data[[gene$input_cols[1]]])
  },
  apply_func = function(data, gene, state) {
    .apply_frequency_mapping(data[[gene$input_cols[1]]], state)
  },
  name_generator = function(gene) .gene_col_name(gene, "nfreq")
)


# --- STATEFUL MIXED TRANSFORMERS ---

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
evo_transformers$concat <- create_transformer(
  name = "concat",
  type = "multivariate",
  input_type = "categorical",
  output_type = "categorical",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    col_list <- lapply(input_cols, function(c) as.character(data[[c]]))
    do.call(paste, c(col_list, list(sep = "_")))
  },
  name_generator = function(gene) .gene_col_name(gene, "concat"),
  allow_replace = FALSE
)

# Feature Hashing
evo_transformers$feature_hash <- create_transformer(
  name = "feature_hash",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    num_bins <- if (!is.null(gene$params$num_bins)) gene$params$num_bins else 8
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    
    # Pre-allocate output
    res <- rep(NA_real_, length(x))
    
    non_na_mask <- !is.na(x)
    if (any(non_na_mask)) {
      valid_x <- x[non_na_mask]
      u_x <- unique(valid_x)
      v_hash <- digest::getVDigest(algo = "xxhash32")
      hex_vals <- v_hash(u_x)
      int_vals <- strtoi(substr(hex_vals, 1, 7), 16L)
      bin_indices <- (int_vals %% num_bins) + 1
      u_res <- as.numeric(bin_indices == comp_idx)
      res[non_na_mask] <- u_res[match(valid_x, u_x)]
    }
    
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "fh")
)

# Multiple Correspondence Analysis (MCA)
evo_transformers$similarity_encode <- create_transformer(
  name = "similarity_encode",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    valid_x <- x[!is.na(x) & x != ""]
    if (length(valid_x) == 0) return(list(valid = FALSE))

    counts <- sort(table(valid_x), decreasing = TRUE)
    prototypes <- names(counts)[1:min(5L, length(counts))]

    get_3grams <- function(s) {
      if (is.na(s) || nchar(s) == 0) return(character(0))
      s_clean <- paste0("^", tolower(s), "$")
      n <- nchar(s_clean)
      if (n < 3) return(s_clean)
      substring(s_clean, 1:(n - 2), 3:n)
    }

    proto_3grams <- lapply(prototypes, get_3grams)
    list(
      prototypes = prototypes,
      proto_3grams = proto_3grams,
      valid = TRUE,
      preds_cache = new.env(hash = TRUE, parent = emptyenv())
    )
  },
  apply_func = function(data, gene, state = NULL) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid) || comp_idx > length(state$prototypes)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    proto_grams <- state$proto_3grams[[comp_idx]]
    proto_len <- length(proto_grams)

    compute_sim <- function(str_vec) {
      get_3grams <- function(s) {
        if (is.na(s) || nchar(s) == 0) return(character(0))
        s_clean <- paste0("^", tolower(s), "$")
        n <- nchar(s_clean)
        if (n < 3) return(s_clean)
        substring(s_clean, 1:(n - 2), 3:n)
      }
      compute_single <- function(s) {
        if (is.na(s) || nchar(s) == 0) return(0)
        g <- get_3grams(s)
        if (length(g) == 0 && proto_len == 0) return(1)
        if (length(g) == 0 || proto_len == 0) return(0)
        intersection <- length(intersect(g, proto_grams))
        union_len <- length(union(g, proto_grams))
        if (union_len == 0) 0 else intersection / union_len
      }
      u_s <- unique(str_vec)
      u_res <- vapply(u_s, compute_single, numeric(1), USE.NAMES = FALSE)
      u_res[match(str_vec, u_s)]
    }

    if (is.null(state$preds_cache)) {
      compute_sim(x)
    } else {
      x_key <- paste0("comp_", comp_idx, "_", digest::digest(x, algo = "xxhash64"))
      if (exists(x_key, envir = state$preds_cache)) {
        get(x_key, envir = state$preds_cache)
      } else {
        res <- compute_sim(x)
        assign(x_key, res, envir = state$preds_cache)
        res
      }
    }
  },
  name_generator = function(gene) .gene_col_name(gene, "sim")
)

# MinHash Sub-string Encoder
evo_transformers$minhash_encode <- create_transformer(
  name = "minhash_encode",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    seeds <- c(1013L, 2017L, 3089L, 4099L, 5023L, 6037L, 7053L, 8081L)
    list(seeds = seeds, valid = TRUE, preds_cache = new.env(hash = TRUE, parent = emptyenv()))
  },
  apply_func = function(data, gene, state = NULL) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    seeds <- state$seeds
    if (comp_idx > length(seeds)) comp_idx <- ((comp_idx - 1) %% length(seeds)) + 1
    seed <- seeds[comp_idx]

    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])

    compute_minhash <- function(str_vec) {
      compute_single <- function(s) {
        if (is.na(s) || nchar(s) == 0) return(0)
        s_clean <- paste0("^", tolower(s), "$")
        n <- nchar(s_clean)
        grams <- if (n < 3) s_clean else substring(s_clean, 1:(n - 2), 3:n)
        hashes <- vapply(grams, function(g) {
          h_hex <- digest::digest(paste0(g, "_", seed), algo = "xxhash32")
          strtoi(substr(h_hex, 1, 7), base = 16L)
        }, integer(1), USE.NAMES = FALSE)
        min(hashes) / 268435455
      }
      u_s <- unique(str_vec)
      u_res <- vapply(u_s, compute_single, numeric(1), USE.NAMES = FALSE)
      u_res[match(str_vec, u_s)]
    }

    if (is.null(state$preds_cache)) {
      compute_minhash(x)
    } else {
      x_key <- paste0("comp_", comp_idx, "_", digest::digest(x, algo = "xxhash64"))
      if (exists(x_key, envir = state$preds_cache)) {
        get(x_key, envir = state$preds_cache)
      } else {
        res <- compute_minhash(x)
        assign(x_key, res, envir = state$preds_cache)
        res
      }
    }
  },
  name_generator = function(gene) .gene_col_name(gene, "minhash")
)

# Sub-string N-gram Topic Encoder (inspired by skrub GapEncoder)
evo_transformers$gap_encode <- create_transformer(
  name = "gap_encode",
  type = "unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    valid_x <- x[!is.na(x) & x != ""]
    if (length(valid_x) == 0) return(list(valid = FALSE))

    get_3grams <- function(s) {
      if (is.na(s) || nchar(s) == 0) return(character(0))
      s_clean <- paste0("^", tolower(s), "$")
      n <- nchar(s_clean)
      if (n < 3) return(s_clean)
      substring(s_clean, 1:(n - 2), 3:n)
    }

    # Deduplicate unique categories to extract grams and count frequencies
    u_valid_x <- unique(valid_x)
    cat_counts <- table(valid_x)
    u_counts <- as.numeric(cat_counts[u_valid_x])
    u_grams <- lapply(u_valid_x, get_3grams)

    gram_counts_map <- new.env(hash = TRUE, parent = emptyenv())
    for (i in seq_along(u_valid_x)) {
      g <- u_grams[[i]]
      if (length(g) > 0) {
        tab <- table(g)
        cnt <- u_counts[i]
        for (gn in names(tab)) {
          prev <- if (exists(gn, envir = gram_counts_map, inherits = FALSE)) get(gn, envir = gram_counts_map) else 0
          assign(gn, prev + as.numeric(tab[[gn]]) * cnt, envir = gram_counts_map)
        }
      }
    }
    all_grams <- ls(gram_counts_map)
    if (length(all_grams) == 0) return(list(valid = FALSE))
    gram_freqs <- vapply(all_grams, function(gn) get(gn, envir = gram_counts_map), numeric(1))
    top_grams <- names(sort(gram_freqs, decreasing = TRUE))[1:min(30L, length(all_grams))]

    # Build unique n-gram counts matrix
    mat_u <- matrix(0, nrow = length(u_valid_x), ncol = length(top_grams))
    colnames(mat_u) <- top_grams
    for (i in seq_along(u_valid_x)) {
      g <- u_grams[[i]]
      if (length(g) > 0) {
        tab <- table(g)
        match_idx <- match(names(tab), top_grams)
        valid_m <- !is.na(match_idx)
        if (any(valid_m)) {
          mat_u[i, match_idx[valid_m]] <- as.numeric(tab[valid_m])
        }
      }
    }

    total_n <- length(x)
    col_means <- colSums(mat_u * u_counts) / max(1, total_n)
    mat_c_u <- sweep(mat_u, 2, col_means, "-")
    n_zero <- total_n - sum(u_counts)
    cov_mat <- crossprod(mat_c_u * sqrt(u_counts)) + n_zero * tcrossprod(col_means)

    tryCatch({
      nv <- min(4L, ncol(cov_mat))
      eig <- eigen(cov_mat, symmetric = TRUE)
      v <- eig$vectors[, seq_len(nv), drop = FALSE]
      list(
        top_grams = top_grams,
        col_means = col_means,
        v = v,
        valid = TRUE,
        preds_cache = new.env(hash = TRUE, parent = emptyenv())
      )
    }, error = function(e) list(valid = FALSE))
  },
  apply_func = function(data, gene, state = NULL) {
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    if (is.null(state) || !isTRUE(state$valid)) return(rep(0, nrow(data)))
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    top_grams <- state$top_grams

    compute_proj <- function() {
      get_3grams <- function(s) {
        if (is.na(s) || nchar(s) == 0) return(character(0))
        s_clean <- paste0("^", tolower(s), "$")
        n <- nchar(s_clean)
        if (n < 3) return(s_clean)
        substring(s_clean, 1:(n - 2), 3:n)
      }
      u_x <- unique(x)
      mat_u <- matrix(0, nrow = length(u_x), ncol = length(top_grams))
      for (i in seq_along(u_x)) {
        s <- u_x[i]
        if (!is.na(s) && nchar(s) > 0) {
          g <- get_3grams(s)
          if (length(g) > 0) {
            tab <- table(g)
            match_idx <- match(names(tab), top_grams)
            valid_m <- !is.na(match_idx)
            if (any(valid_m)) {
              mat_u[i, match_idx[valid_m]] <- as.numeric(tab[valid_m])
            }
          }
        }
      }
      mat_centered_u <- sweep(mat_u, 2, state$col_means, "-")
      proj_u <- mat_centered_u %*% state$v
      proj_u[match(x, u_x), , drop = FALSE]
    }

    preds <- if (is.null(state$preds_cache)) {
      compute_proj()
    } else {
      x_key <- digest::digest(x, algo = "xxhash64")
      if (exists(x_key, envir = state$preds_cache)) {
        get(x_key, envir = state$preds_cache)
      } else {
        res <- compute_proj()
        assign(x_key, res, envir = state$preds_cache)
        res
      }
    }
    if (comp_idx > ncol(preds)) comp_idx <- ncol(preds)
    as.vector(preds[, comp_idx])
  },
  name_generator = function(gene) .gene_col_name(gene, "gap")
)

# Datetime Cyclic Features (sine/cosine periodicity)
