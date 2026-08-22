# Unsupervised categorical encoders: frequency, one-hot, hashing, n-gram & similarity encodings.
# Split out of transformers.R; loaded after it (alphabetical file order).

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
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt <- data.table::data.table(x = x)
    res <- state$mapping[dt, on = "x"]$N
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "freq")
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
      v_hash <- digest::getVDigest(algo = "xxhash32")
      hex_vals <- v_hash(x[non_na_mask])
      int_vals <- strtoi(substr(hex_vals, 1, 7), 16L)
      bin_indices <- (int_vals %% num_bins) + 1
      res[non_na_mask] <- as.numeric(bin_indices == comp_idx)
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
      sapply(str_vec, function(s) {
        if (is.na(s) || nchar(s) == 0) return(0)
        g <- get_3grams(s)
        if (length(g) == 0 && proto_len == 0) return(1)
        if (length(g) == 0 || proto_len == 0) return(0)
        intersection <- length(intersect(g, proto_grams))
        union_len <- length(union(g, proto_grams))
        if (union_len == 0) 0 else intersection / union_len
      }, USE.NAMES = FALSE)
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
      sapply(str_vec, function(s) {
        if (is.na(s) || nchar(s) == 0) return(0)
        s_clean <- paste0("^", tolower(s), "$")
        n <- nchar(s_clean)
        grams <- if (n < 3) s_clean else substring(s_clean, 1:(n - 2), 3:n)
        hashes <- sapply(grams, function(g) {
          h_hex <- digest::digest(paste0(g, "_", seed), algo = "xxhash32")
          strtoi(substr(h_hex, 1, 7), base = 16L)
        }, USE.NAMES = FALSE)
        min(hashes) / 268435455
      }, USE.NAMES = FALSE)
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

    all_grams <- unlist(lapply(valid_x, get_3grams))
    if (length(all_grams) == 0) return(list(valid = FALSE))

    gram_counts <- sort(table(all_grams), decreasing = TRUE)
    top_grams <- names(gram_counts)[1:min(30L, length(gram_counts))]

    mat <- matrix(0, nrow = length(x), ncol = length(top_grams))
    colnames(mat) <- top_grams
    for (i in seq_along(x)) {
      g <- get_3grams(x[i])
      if (length(g) > 0) {
        tab <- table(g)
        match_idx <- match(names(tab), top_grams)
        valid_m <- !is.na(match_idx)
        if (any(valid_m)) {
          mat[i, match_idx[valid_m]] <- as.numeric(tab[valid_m])
        }
      }
    }

    col_means <- colMeans(mat)
    mat_centered <- sweep(mat, 2, col_means, "-")

    tryCatch({
      nv <- min(4L, ncol(mat))
      res <- svd(mat_centered, nu = 0, nv = nv)
      list(
        top_grams = top_grams,
        col_means = col_means,
        v = res$v,
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
      mat <- matrix(0, nrow = length(x), ncol = length(top_grams))
      for (i in seq_along(x)) {
        g <- get_3grams(x[i])
        if (length(g) > 0) {
          tab <- table(g)
          match_idx <- match(names(tab), top_grams)
          valid_m <- !is.na(match_idx)
          if (any(valid_m)) {
            mat[i, match_idx[valid_m]] <- as.numeric(tab[valid_m])
          }
        }
      }
      mat_centered <- sweep(mat, 2, state$col_means, "-")
      mat_centered %*% state$v
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
