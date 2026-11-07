# Supervised (target-based) encoders: mean/WoE/quantile target encodings.
# Split out of transformers.R; loaded after it (alphabetical file order).

evo_transformers$target_encode <- create_transformer(
  name = "target_encode",
  type = "supervised_unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(mapping = NULL, global_mean = 0, valid = FALSE))
    }
    
    # Calculate global mean
    global_mean <- mean(y, na.rm = TRUE)
    
    # Calculate category means and counts
    dt <- data.table::data.table(x = x, y = y)
    stats <- dt[, .(mean = mean(y, na.rm = TRUE), sum_y = sum(y, na.rm = TRUE), n = sum(!is.na(y))), by = x]
    
    # Smoothing parameters (can be adjusted)
    min_samples_leaf <- 10
    smoothing <- 10
    
    # Calculate smoothed target encoding
    # formula: (n * mean + smoothing * global_mean) / (n + smoothing)
    stats[, smoothed := (n * mean + smoothing * global_mean) / (n + smoothing)]
    
    mapping <- stats[, .(x, smoothed)]
    data.table::setkey(mapping, x)

    # Leave-one-out (LOO) smoothed values on the training split to prevent in-sample target leakage
    mapped <- stats[dt, on = "x"]
    n_loo <- pmax(0, mapped$n - as.integer(!is.na(y)))
    sum_loo <- mapped$sum_y - ifelse(is.na(y), 0, y)
    loo_vals <- (sum_loo + smoothing * global_mean) / (n_loo + smoothing)
    loo_vals[is.na(loo_vals)] <- global_mean
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL
    
    list(mapping = mapping, global_mean = global_mean, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
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

# Pooled Target Encoding (Empirical Bayes Shrinkage)
evo_transformers$pooled_target_encode <- create_transformer(
  name = "pooled_target_encode",
  type = "supervised_unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(mapping = NULL, global_mean = 0, valid = FALSE))
    }
    
    # Calculate global mean and total variance of the target
    global_mean <- mean(y, na.rm = TRUE)
    var_y <- stats::var(y, na.rm = TRUE)
    if (is.na(var_y)) var_y <- 0
    
    # Calculate category means, sums, within-category variances, and counts
    dt <- data.table::data.table(x = x, y = y)
    stats <- dt[, .(
      mean = mean(y, na.rm = TRUE),
      sum_y = sum(y, na.rm = TRUE),
      var = stats::var(y, na.rm = TRUE),
      n = sum(!is.na(y))
    ), by = x]
    
    # Estimate pooled within-category variance (sigma^2)
    df_sum <- sum(stats$n - 1, na.rm = TRUE)
    var_within <- if (df_sum > 0) {
      sum((stats$n - 1) * stats$var, na.rm = TRUE) / df_sum
    } else {
      mean_var <- mean(stats$var, na.rm = TRUE)
      if (is.na(mean_var)) var_y else mean_var
    }
    if (is.na(var_within)) var_within <- var_y
    
    # Estimate variance of category means (between-group variance, tau^2)
    var_between <- stats::var(stats$mean, na.rm = TRUE)
    if (is.na(var_between)) var_between <- 0
    
    # Compute dynamic smoothing factor k = var_within / var_between
    k <- if (var_between > 0) var_within / var_between else Inf
    
    # Calculate Empirical Bayes smoothed target encoding
    # weight: lambda = n / (n + k)
    stats[, smoothed := if (is.infinite(k)) global_mean else (n * mean + k * global_mean) / (n + k)]
    
    mapping <- stats[, .(x, smoothed)]
    data.table::setkey(mapping, x)

    loo_train <- NULL
    if (is.finite(k)) {
      mapped <- stats[dt, on = "x"]
      n_loo <- pmax(0, mapped$n - as.integer(!is.na(y)))
      sum_loo <- mapped$sum_y - ifelse(is.na(y), 0, y)
      loo_vals <- ifelse(n_loo > 0, (sum_loo + k * global_mean) / (n_loo + k), global_mean)
      loo_vals[!is.finite(loo_vals)] <- global_mean
      if (length(unique(loo_vals)) > 1L) loo_train <- loo_vals
    }
    
    list(mapping = mapping, global_mean = global_mean, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
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
  name_generator = function(gene) .gene_col_name(gene, "pte")
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
    loo_train_list <- list()
    smoothing <- 10
    
    for (k in seq_along(classes)) {
      y_bin <- as.numeric(y == classes[k])
      global_mean <- mean(y_bin, na.rm = TRUE)
      global_means[[k]] <- global_mean
      
      dt <- data.table::data.table(x = x, y = y_bin)
      stats <- dt[, .(mean = mean(y, na.rm = TRUE), sum_y = sum(y, na.rm = TRUE), n = sum(!is.na(y))), by = x]
      stats[, smoothed := (n * mean + smoothing * global_mean) / (n + smoothing)]
      
      mapping <- stats[, .(x, smoothed)]
      data.table::setkey(mapping, x)
      mappings[[k]] <- mapping

      mapped <- stats[dt, on = "x"]
      n_loo <- pmax(0, mapped$n - as.integer(!is.na(y_bin)))
      sum_loo <- mapped$sum_y - ifelse(is.na(y_bin), 0, y_bin)
      loo_vals <- (sum_loo + smoothing * global_mean) / (n_loo + smoothing)
      loo_vals[is.na(loo_vals)] <- global_mean
      loo_train_list[[k]] <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL
    }
    
    list(mappings = mappings, global_means = global_means, classes = classes, loo_train_list = loo_train_list)
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


# Weight of Evidence Encoding (binary classification only)
evo_transformers$woe_encode <- create_transformer(
  name = "woe_encode",
  type = "supervised_unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- data[[target_col]]

    # WOE is only defined for binary targets
    y_vals <- sort(unique(y[!is.na(y)]))
    if (length(y_vals) != 2) {
      return(list(mapping = NULL, global_woe = 0))
    }

    # Event = second (higher) level
    y_bin <- as.numeric(y == y_vals[2])
    total_events    <- sum(y_bin, na.rm = TRUE)
    total_nonevents <- sum(1 - y_bin, na.rm = TRUE)

    if (total_events == 0 || total_nonevents == 0) {
      return(list(mapping = NULL, global_woe = 0))
    }

    dt    <- data.table::data.table(x = x, y = y_bin)
    stats <- dt[, .(n_events = sum(y, na.rm = TRUE), n = .N), by = x]
    stats[, n_nonevents := n - n_events]

    # Laplace smoothing to avoid log(0)
    eps <- 0.5
    stats[, dist_events    := (n_events    + eps) / (total_events    + 2 * eps)]
    stats[, dist_nonevents := (n_nonevents + eps) / (total_nonevents + 2 * eps)]
    stats[, woe := log(dist_events / dist_nonevents)]
    stats[!is.finite(woe), woe := 0]

    mapping <- stats[, .(x, woe)]
    data.table::setkey(mapping, x)

    mapped <- stats[dt, on = "x"]
    is_ev <- !is.na(y_bin) & y_bin == 1
    is_nev <- !is.na(y_bin) & y_bin == 0
    ev_loo <- pmax(0, mapped$n_events - ifelse(is_ev, 1, 0))
    nev_loo <- pmax(0, mapped$n_nonevents - ifelse(is_nev, 1, 0))
    tot_ev_loo <- pmax(1, total_events - ifelse(is_ev, 1, 0))
    tot_nev_loo <- pmax(1, total_nonevents - ifelse(is_nev, 1, 0))
    loo_woe <- log(((ev_loo + eps) / (tot_ev_loo + 2 * eps)) / ((nev_loo + eps) / (tot_nev_loo + 2 * eps)))
    loo_woe[!is.finite(loo_woe)] <- 0
    loo_train <- if (length(unique(loo_woe)) > 1L) loo_woe else NULL

    list(mapping = mapping, global_woe = 0, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(state) || is.null(state$mapping)) {
      return(rep(0, length(x)))
    }
    dt  <- data.table::data.table(x = x)
    res <- state$mapping[dt, on = "x"]$woe
    res[is.na(res)] <- state$global_woe
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "woe")
)

# Helper: normalize target y into a mathematically valid numeric vector for single-column target encoding
.normalize_target_for_encoding <- function(y) {
  if (is.null(y)) return(NULL)
  if (is.numeric(y)) return(y)
  y_clean <- y[!is.na(y)]
  u_vals <- sort(unique(y_clean))
  
  if (length(u_vals) == 2) {
    return(as.numeric(y == u_vals[2]))
  } else if (length(u_vals) == 1) {
    return(as.numeric(y == u_vals[1]))
  } else {
    return(NULL)
  }
}

# Helper: discretize a target vector into group labels
.target_to_groups <- function(y) {
  if (is.numeric(y)) {
    breaks <- stats::quantile(y, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
    # Handle ties by making breaks unique
    breaks <- unique(breaks)
    if (length(breaks) < 2) return(rep("q1", length(y)))
    grps <- as.character(cut(y, breaks = breaks, include.lowest = TRUE, labels = FALSE))
    grps[is.na(grps)] <- "NA"
    grps
  } else {
    grps <- as.character(y)
    grps[is.na(grps)] <- "NA"
    grps
  }
}

# Target Quantile Encoding
evo_transformers$target_quantile_encode <- create_transformer(
  name = "target_quantile_encode",
  type = "supervised_unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(mapping = NULL, global_q = 0, valid = FALSE))
    }
    q <- if (!is.null(gene$params$q)) gene$params$q else 0.5

    global_q <- as.numeric(stats::quantile(y, probs = q, na.rm = TRUE))

    dt <- data.table::data.table(x = x, y = y)
    stats_dt <- dt[, .(cat_q = as.numeric(stats::quantile(y, probs = q, na.rm = TRUE)), n = sum(!is.na(y))), by = x]

    smoothing <- 10
    stats_dt[, smoothed := (n * cat_q + smoothing * global_q) / (n + smoothing)]

    mapping <- stats_dt[, .(x, smoothed)]
    data.table::setkey(mapping, x)

    # Exact O(N log N) Type-7 leave-one-out quantiles per category
    N_rows <- length(y)
    loo_vals <- rep(global_q, N_rows)
    grp_idx <- split(seq_len(N_rows), addNA(factor(x)))
    for (idx in grp_idx) {
      y_g <- y[idx]
      fin <- !is.na(y_g)
      nc <- sum(fin)
      if (nc == 0L) next
      if (nc == 1L) {
        if (any(!fin)) {
          cat_q1 <- y_g[fin][1L]
          loo_vals[idx[!fin]] <- (1.0 * cat_q1 + smoothing * global_q) / (1.0 + smoothing)
        }
        next
      }
      fin_idx <- idx[fin]
      y_sub <- y_g[fin]
      ord <- order(y_sub)
      v <- y_sub[ord]
      m <- nc - 1L
      h <- 1.0 + (m - 1.0) * q
      j0 <- as.integer(floor(h))
      j1 <- min(m, j0 + 1L)
      gam <- h - j0
      r_seq <- seq_len(nc)
      q_sorted <- (1.0 - gam) * v[j0 + (j0 >= r_seq)] + gam * v[j1 + (j1 >= r_seq)]
      q_loo <- numeric(nc)
      q_loo[ord] <- q_sorted
      loo_vals[fin_idx] <- (m * q_loo + smoothing * global_q) / (m + smoothing)
      if (any(!fin)) {
        cat_qf <- as.numeric(stats::quantile(y_sub, probs = q, names = FALSE))
        loo_vals[idx[!fin]] <- (nc * cat_qf + smoothing * global_q) / (nc + smoothing)
      }
    }
    loo_vals[!is.finite(loo_vals)] <- global_q
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(mapping = mapping, global_q = global_q, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    dt <- data.table::data.table(x = x)
    mapping <- state$mapping

    res <- mapping[dt, on = "x"]$smoothed
    res[is.na(res)] <- state$global_q
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "tqe")
)

# Categorical Interaction Target Encoding
evo_transformers$cat_interaction_target_encode <- create_transformer(
  name = "cat_interaction_target_encode",
  type = "supervised_binary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x1 <- as.character(data[[input_cols[1]]])
    x2 <- as.character(data[[input_cols[2]]])
    x_joint <- paste0(x1, "___", x2)
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(mapping = NULL, global_mean = 0, valid = FALSE))
    }

    global_mean <- mean(y, na.rm = TRUE)
    dt <- data.table::data.table(x_joint = x_joint, y = y)
    stats_dt <- dt[, .(mean_y = mean(y, na.rm = TRUE), sum_y = sum(y, na.rm = TRUE), n = sum(!is.na(y))), by = x_joint]

    smoothing <- 10
    stats_dt[, smoothed := (n * mean_y + smoothing * global_mean) / (n + smoothing)]

    mapping <- stats_dt[, .(x_joint, smoothed)]
    data.table::setkey(mapping, x_joint)

    mapped <- stats_dt[dt, on = "x_joint"]
    n_loo <- pmax(0, mapped$n - as.integer(!is.na(y)))
    sum_loo <- mapped$sum_y - ifelse(is.na(y), 0, y)
    loo_vals <- (sum_loo + smoothing * global_mean) / (n_loo + smoothing)
    loo_vals[is.na(loo_vals)] <- global_mean
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(mapping = mapping, global_mean = global_mean, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x1 <- as.character(data[[input_cols[1]]])
    x2 <- as.character(data[[input_cols[2]]])
    x_joint <- paste0(x1, "___", x2)

    dt <- data.table::data.table(x_joint = x_joint)
    mapping <- state$mapping

    res <- mapping[dt, on = "x_joint"]$smoothed
    res[is.na(res)] <- state$global_mean
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "cite")
)

# Target Standard Deviation Encoding (with Leave-One-Out jackknife SD on training data)
evo_transformers$target_sd_encode <- create_transformer(
  name = "target_sd_encode",
  type = "supervised_unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(mapping = NULL, global_sd = 0, valid = FALSE))
    }

    global_mean <- mean(y, na.rm = TRUE)
    if (is.na(global_mean)) global_mean <- 0
    global_sd <- .safe_sd(y)
    y_c <- y - global_mean

    dt <- data.table::data.table(x = x, yc = y_c)
    stats_dt <- dt[, .(
      sum_yc = sum(yc, na.rm = TRUE),
      sum_yc2 = sum(yc * yc, na.rm = TRUE),
      n = sum(!is.na(yc))
    ), by = x]

    stats_dt[, raw_sd := ifelse(
      n >= 2L,
      sqrt(pmax(0.0, (sum_yc2 - (sum_yc * sum_yc) / n) / (n - 1.0))),
      global_sd
    )]
    smoothing <- 10
    stats_dt[, smoothed := (n * raw_sd + smoothing * global_sd) / (n + smoothing)]

    mapping <- stats_dt[, .(x, smoothed)]
    data.table::setkey(mapping, x)

    mapped <- stats_dt[dt, on = "x"]
    has_y <- !is.na(y_c)
    n_loo <- pmax(0L, mapped$n - as.integer(has_y))
    yc_clean <- ifelse(has_y, y_c, 0.0)
    s1_loo <- mapped$sum_yc - yc_clean
    s2_loo <- mapped$sum_yc2 - yc_clean * yc_clean
    sd_loo <- ifelse(
      n_loo >= 2L,
      sqrt(pmax(0.0, (s2_loo - (s1_loo * s1_loo) / pmax(1L, n_loo)) / pmax(1.0, n_loo - 1.0))),
      global_sd
    )
    loo_vals <- (n_loo * sd_loo + smoothing * global_sd) / (n_loo + smoothing)
    loo_vals[!is.finite(loo_vals)] <- global_sd
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(mapping = mapping, global_sd = global_sd, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    dt <- data.table::data.table(x = x)
    res <- state$mapping[dt, on = "x"]$smoothed
    res[is.na(res)] <- state$global_sd
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "tsd")
)

# k-Nearest Neighbors Target Encoding
evo_transformers$knn_target_encode <- create_transformer(
  name = "knn_target_encode",
  type = "supervised_multivariate",
  input_type = "numeric",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(valid = FALSE, global_mean = 0.0, loo_train = NULL))
    }
    global_mean <- mean(y, na.rm = TRUE)
    if (!is.finite(global_mean)) global_mean <- 0.0

    X <- as.matrix(data[, input_cols, with = FALSE])
    if (!is.numeric(X)) storage.mode(X) <- "double"
    mu <- colMeans(X, na.rm = TRUE)
    sigma <- apply(X, 2, stats::sd, na.rm = TRUE)
    valid_cols <- !is.na(sigma) & is.finite(sigma) & sigma >= 1e-8
    if (!any(valid_cols)) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }
    sigma[!valid_cols] <- 1.0
    mu[is.na(mu) | !is.finite(mu)] <- 0.0

    for (j in seq_len(ncol(X))) {
      nas <- is.na(X[, j]) | !is.finite(X[, j])
      if (any(nas)) X[nas, j] <- mu[j]
    }

    X_scaled <- sweep(sweep(X, 2, mu, "-"), 2, sigma, "/")
    X_query <- if (ncol(X_scaled) == 1L) cbind(X_scaled, 0.0) else X_scaled

    N <- nrow(X_query)
    if (N < 2L) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }

    y_clean <- y
    y_clean[is.na(y_clean) | !is.finite(y_clean)] <- global_mean

    # Collapse exact duplicate coordinates so ties are 100% invariant to row order
    dt_u <- data.table::as.data.table(X_query)
    coord_cols <- names(dt_u)
    dt_u[, `:=`(.y = y_clean, .orig_row = seq_len(N))]
    u_stats <- dt_u[, .(.cnt = .N, .sum_y = sum(.y)), by = coord_cols]
    M <- nrow(u_stats)
    if (M < 2L) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }
    u_stats[, .gid := seq_len(M)]
    grp_id <- u_stats[dt_u, on = coord_cols]$.gid

    U_all <- as.matrix(u_stats[, coord_cols, with = FALSE])
    storage.mode(U_all) <- "double"
    u_cnt <- u_stats$.cnt
    u_sum <- u_stats$.sum_y

    k_req <- if (!is.null(gene$params$k)) as.integer(gene$params$k) else 5L
    k_eff <- min(k_req, N - 1L)
    if (k_eff < 1L) k_eff <- 1L

    max_ref <- max(2L, as.integer(getOption("evoFE.max_clustering_size", 20000L)))
    ref_idx <- NULL
    if (M > max_ref) {
      ref_idx <- unique(as.integer(round(seq(1L, M, length.out = max_ref))))
      u_ref <- U_all[ref_idx, , drop = FALSE]
      ref_cnt <- u_cnt[ref_idx]
      ref_mean <- u_sum[ref_idx] / ref_cnt
      k_u <- min(k_eff + 1L, nrow(u_ref))
      nn_res <- tryCatch(quitefastmst::knn_euclid(u_ref, k = k_u, Y = U_all), error = function(e) NULL)
    } else {
      u_ref <- U_all
      ref_cnt <- u_cnt
      ref_mean <- u_sum / ref_cnt
      k_u <- min(k_eff, M - 1L)
      nn_res <- tryCatch(quitefastmst::knn_euclid(u_ref, k = k_u, Y = NULL), error = function(e) NULL)
    }

    if (is.null(nn_res) || is.null(nn_res$nn.index)) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }

    nn_idx <- matrix(nn_res$nn.index, nrow = M, ncol = k_u)
    rem_g <- pmax(0.0, as.numeric(k_eff - (u_cnt - 1L)))
    other_sum <- numeric(M)
    other_cnt <- numeric(M)
    if (any(rem_g > 0)) {
      m_seq <- seq_len(M)
      for (j in seq_len(k_u)) {
        rj <- nn_idx[, j]
        is_self <- if (!is.null(ref_idx)) (ref_idx[rj] == m_seq) else rep(FALSE, M)
        avail <- ifelse(is_self, 0.0, as.numeric(ref_cnt[rj]))
        w <- pmin(rem_g, avail)
        other_sum <- other_sum + w * ref_mean[rj]
        other_cnt <- other_cnt + w
        rem_g <- rem_g - w
        if (all(rem_g <= 0)) break
      }
    }

    c_self <- u_cnt[grp_id] - 1L
    s_self <- u_sum[grp_id] - y_clean
    tot_cnt <- c_self + other_cnt[grp_id]
    mean_nn <- ifelse(
      c_self >= k_eff,
      s_self / pmax(1L, c_self),
      ifelse(tot_cnt > 0, (s_self + other_sum[grp_id]) / tot_cnt, global_mean)
    )

    s <- 2.0
    loo_vals <- (k_eff * mean_nn + s * global_mean) / (k_eff + s)
    loo_vals[!is.finite(loo_vals)] <- global_mean
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(
      valid = TRUE,
      u_ref = u_ref,
      ref_cnt = ref_cnt,
      ref_mean = ref_mean,
      mu = mu,
      sigma = sigma,
      k = k_eff,
      global_mean = global_mean,
      loo_train = loo_train
    )
  },
  apply_func = function(data, gene, state) {
    n_out <- nrow(data)
    if (n_out == 0L) return(numeric(0))
    g_mean <- if (!is.null(state) && !is.null(state$global_mean) && is.finite(state$global_mean)) state$global_mean else 0.0
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$u_ref)) {
      return(rep(g_mean, n_out))
    }
    input_cols <- gene$input_cols
    X <- as.matrix(data[, input_cols, with = FALSE])
    if (!is.numeric(X)) storage.mode(X) <- "double"

    for (j in seq_len(ncol(X))) {
      nas <- is.na(X[, j]) | !is.finite(X[, j])
      if (any(nas)) X[nas, j] <- state$mu[j]
    }

    X_scaled <- sweep(sweep(X, 2, state$mu, "-"), 2, state$sigma, "/")
    X_query <- if (ncol(X_scaled) == 1L) cbind(X_scaled, 0.0) else X_scaled

    k_q <- min(state$k, nrow(state$u_ref))
    nn_res <- tryCatch({
      quitefastmst::knn_euclid(state$u_ref, k = k_q, Y = X_query)
    }, error = function(e) NULL)

    if (is.null(nn_res) || is.null(nn_res$nn.index)) {
      return(rep(g_mean, n_out))
    }

    nn_idx <- matrix(nn_res$nn.index, nrow = n_out, ncol = k_q)
    if (all(state$ref_cnt == 1L)) {
      y_neighbors <- matrix(state$ref_mean[nn_idx], nrow = n_out, ncol = k_q)
      mean_nn <- rowMeans(y_neighbors)
    } else {
      rem_vec <- rep(as.numeric(state$k), n_out)
      acc_sum <- numeric(n_out)
      acc_cnt <- numeric(n_out)
      for (j in seq_len(k_q)) {
        idx_j <- nn_idx[, j]
        w <- pmin(rem_vec, state$ref_cnt[idx_j])
        acc_sum <- acc_sum + w * state$ref_mean[idx_j]
        acc_cnt <- acc_cnt + w
        rem_vec <- rem_vec - w
        if (all(rem_vec <= 0)) break
      }
      mean_nn <- ifelse(acc_cnt > 0, acc_sum / acc_cnt, g_mean)
    }

    s <- 2.0
    res <- (state$k * mean_nn + s * g_mean) / (state$k + s)
    res[!is.finite(res)] <- g_mean
    as.numeric(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "knnte")
)
