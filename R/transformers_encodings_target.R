# Supervised (target-based) encoders: mean/WoE/quantile target encodings.
# Split out of transformers.R; loaded after it (alphabetical file order).

.make_oof_folds <- function(y, data = NULL, n_folds = 5L) {
  n <- length(y)
  if (n <= 1L) return(rep(1L, n))
  if (!is.null(data)) {
    preset <- attr(data, ".oof_folds", exact = TRUE)
    if (!is.null(preset) && length(preset) == n && !anyNA(preset) && length(unique(preset)) >= 2L) {
      return(as.integer(preset))
    }
  }
  k <- min(as.integer(n_folds), n)
  if (k < 2L) return(rep(1L, n))
  idx <- seq_len(n)
  tie <- (as.numeric(idx) * 2654435761.0) %% 4294967296.0
  ord <- if (is.numeric(y) || is.logical(y)) {
    order(as.numeric(y), tie, method = "radix", na.last = TRUE)
  } else {
    order(as.character(y), tie, method = "radix", na.last = TRUE)
  }
  folds <- integer(n)
  folds[ord] <- ((seq_len(n) - 1L) %% k) + 1L
  folds
}

.compute_oof_mean_from_dt <- function(dt_xy, stats, smoothing, global_mean, folds) {
  if (!is.finite(smoothing)) return(NULL)
  y <- dt_xy$y
  has_y <- !is.na(y)
  dt_f <- data.table::data.table(
    x = dt_xy$x,
    f = folds,
    y_clean = ifelse(has_y, y, 0.0),
    y_obs = as.integer(has_y)
  )
  f_stats <- dt_f[, .(f_sum = sum(y_clean), f_n = sum(y_obs)), by = .(x, f)]
  f_stats[stats, `:=`(
    oof_sum = i.sum_y - f_sum,
    oof_n   = pmax(0, i.n - f_n)
  ), on = "x"]
  f_stats[, oof_val := ifelse(
    oof_n > 0,
    (oof_sum + smoothing * global_mean) / (oof_n + smoothing),
    global_mean
  )]
  oof_vals <- f_stats[dt_f[, .(x, f)], on = .(x, f)]$oof_val
  oof_vals[!is.finite(oof_vals)] <- global_mean
  if (length(unique(oof_vals)) > 1L) oof_vals else NULL
}

.fit_smoothed_mean_te <- function(x_key, y_raw, smoothing = 10, key_col = "x", folds = NULL, data = NULL) {
  y <- .normalize_target_for_encoding(y_raw)
  if (is.null(y) || sum(!is.na(y)) == 0L) {
    return(list(mapping = NULL, global_mean = 0, valid = FALSE))
  }

  global_mean <- mean(y, na.rm = TRUE)
  dt <- data.table::data.table(x = x_key, y = y)
  stats <- dt[, .(mean_y = mean(y, na.rm = TRUE), sum_y = sum(y, na.rm = TRUE), n = sum(!is.na(y))), by = x]
  stats[, smoothed := (n * mean_y + smoothing * global_mean) / (n + smoothing)]

  mapping <- stats[, .(x, smoothed)]
  if (!identical(key_col, "x")) {
    data.table::setnames(mapping, "x", key_col)
  }
  data.table::setkeyv(mapping, key_col)

  if (is.null(folds)) folds <- .make_oof_folds(y, data = data)
  loo_train <- .compute_oof_mean_from_dt(dt, stats, smoothing, global_mean, folds)

  list(mapping = mapping, global_mean = global_mean, loo_train = loo_train)
}

.apply_smoothed_mean_te <- function(x_key, state, key_col = "x") {
  if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
    return(rep(0, length(x_key)))
  }
  dt <- data.table::data.table(x = x_key)
  if (!identical(key_col, "x")) {
    data.table::setnames(dt, "x", key_col)
  }
  res <- tryCatch(state$mapping[dt, on = key_col]$smoothed, error = function(e) NULL)
  if (is.null(res)) return(rep(state$global_mean, length(x_key)))
  res[is.na(res)] <- state$global_mean
  res
}

.fit_col_discretizer <- function(x, n_bins = 20L) {
  if (!is.numeric(x)) {
    return(list(mode = "cat"))
  }
  x_num <- as.numeric(x)
  fin <- is.finite(x_num)
  u_fin <- sort(unique(x_num[fin]))
  if (length(u_fin) == 0L) {
    return(list(mode = "empty"))
  }
  if (length(u_fin) <= n_bins) {
    return(list(mode = "num_exact", levels = u_fin, keys = paste0("v_", seq_along(u_fin))))
  }
  breaks <- unique(stats::quantile(x_num[fin], probs = seq(0, 1, length.out = n_bins + 1L), na.rm = TRUE, names = FALSE))
  if (length(breaks) < 2L) {
    return(list(mode = "num_exact", levels = u_fin, keys = paste0("v_", seq_along(u_fin))))
  }
  list(mode = "num_bins", breaks = breaks)
}

.apply_col_discretizer <- function(x, state) {
  if (is.null(x)) return(character(0))
  n_rows <- length(x)
  if (n_rows == 0L) return(character(0))
  if (is.null(state) || identical(state$mode, "cat")) {
    out <- as.character(x)
    out[is.na(out)] <- ".__NA__"
    return(out)
  }
  if (identical(state$mode, "empty")) {
    return(rep(".__NA__", n_rows))
  }
  x_num <- if (is.numeric(x)) as.numeric(x) else suppressWarnings(as.numeric(as.character(x)))
  out <- rep(".__NA__", n_rows)
  fin <- is.finite(x_num)
  if (!any(fin)) return(out)
  if (identical(state$mode, "num_exact")) {
    lvls <- state$levels
    if (length(lvls) <= 1L) {
      out[fin] <- state$keys[1L]
    } else {
      mids <- 0.5 * (lvls[-length(lvls)] + lvls[-1L])
      idx <- findInterval(x_num[fin], mids) + 1L
      out[fin] <- state$keys[idx]
    }
    return(out)
  }
  if (length(state$breaks) < 2L) {
    out[fin] <- "b_1"
    return(out)
  }
  idx <- findInterval(x_num[fin], state$breaks, all.inside = TRUE)
  out[fin] <- paste0("b_", idx)
  out
}

evo_transformers$target_encode <- create_transformer(
  name = "target_encode",
  type = "supervised_unary",
  input_type = "categorical",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    .fit_smoothed_mean_te(x, data[[target_col]], smoothing = 10, key_col = "x", data = data)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    .apply_smoothed_mean_te(x, state, key_col = "x")
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
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
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

    loo_train <- if (is.finite(k)) {
      folds <- .make_oof_folds(y, data = data)
      .compute_oof_mean_from_dt(dt, stats, k, global_mean, folds)
    } else {
      NULL
    }
    
    list(mapping = mapping, global_mean = global_mean, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    
    # Join with mapping
    dt <- data.table::data.table(x = x)
    mapping <- state$mapping
    
    # Map, filling missing categories with global mean
    res <- tryCatch(mapping[dt, on = "x"]$smoothed, error = function(e) NULL)
    if (is.null(res)) return(rep(state$global_mean, nrow(data)))
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
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    y <- data[[target_col]]
    
    classes <- sort(unique(y))
    mappings <- list()
    global_means <- list()
    loo_train_list <- list()
    smoothing <- 10
    folds <- .make_oof_folds(y, data = data)

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

      loo_train_list[[k]] <- .compute_oof_mean_from_dt(dt, stats, smoothing, global_mean, folds)
    }
    
    list(mappings = mappings, global_means = global_means, classes = classes, loo_train_list = loo_train_list)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    comp_idx <- if (!is.null(gene$params$comp_idx)) gene$params$comp_idx else 1
    
    if (is.null(state) || is.null(state$mappings) || comp_idx > length(state$mappings)) {
      return(rep(0, length(x)))
    }
    
    dt <- data.table::data.table(x = x)
    mapping <- state$mappings[[comp_idx]]
    global_mean <- state$global_means[[comp_idx]]
    
    res <- tryCatch(mapping[dt, on = "x"]$smoothed, error = function(e) NULL)
    if (is.null(res)) return(rep(global_mean, length(x)))
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
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
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

    folds <- .make_oof_folds(y_bin, data = data)
    has_y <- !is.na(y_bin)
    dt_f <- data.table::data.table(
      x = x,
      f = folds,
      ev = as.numeric(has_y & y_bin == 1),
      nev = as.numeric(has_y & y_bin == 0)
    )
    f_tot <- dt_f[, .(f_tot_ev = sum(ev), f_tot_nev = sum(nev)), by = f]
    f_stats <- dt_f[, .(f_ev = sum(ev), f_nev = sum(nev)), by = .(x, f)]
    f_stats[stats, `:=`(
      oof_ev = pmax(0, i.n_events - f_ev),
      oof_nev = pmax(0, i.n_nonevents - f_nev)
    ), on = "x"]
    f_stats[f_tot, `:=`(
      oof_tot_ev = pmax(1, total_events - i.f_tot_ev),
      oof_tot_nev = pmax(1, total_nonevents - i.f_tot_nev)
    ), on = "f"]
    f_stats[, oof_woe := log(((oof_ev + eps) / (oof_tot_ev + 2 * eps)) / ((oof_nev + eps) / (oof_tot_nev + 2 * eps)))]
    f_stats[!is.finite(oof_woe), oof_woe := 0]
    loo_woe <- f_stats[dt_f[, .(x, f)], on = .(x, f)]$oof_woe
    loo_woe[!is.finite(loo_woe)] <- 0
    loo_train <- if (length(unique(loo_woe)) > 1L) loo_woe else NULL

    list(mapping = mapping, global_woe = 0, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    if (is.null(state) || is.null(state$mapping)) {
      return(rep(0, length(x)))
    }
    dt  <- data.table::data.table(x = x)
    res <- tryCatch(state$mapping[dt, on = "x"]$woe, error = function(e) NULL)
    if (is.null(res)) return(rep(state$global_woe, length(x)))
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

    # K-fold out-of-fold quantile target encoding
    N_rows <- length(y)
    folds <- .make_oof_folds(y, data = data)
    loo_vals <- rep(global_q, N_rows)
    dt_f <- data.table::data.table(x = x, y = y, f = folds)
    for (fk in sort(unique(folds))) {
      in_fk <- which(folds == fk)
      oof_stats <- dt_f[f != fk, .(
        cat_q = as.numeric(stats::quantile(y, probs = q, na.rm = TRUE)),
        n = sum(!is.na(y))
      ), by = x]
      oof_stats[n > 0, oof_val := (n * cat_q + smoothing * global_q) / (n + smoothing)]
      oof_stats[n == 0, oof_val := global_q]
      dt_q <- data.table::data.table(x = x[in_fk])
      res_fk <- oof_stats[dt_q, on = "x"]$oof_val
      res_fk[is.na(res_fk) | !is.finite(res_fk)] <- global_q
      loo_vals[in_fk] <- res_fk
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
    .fit_smoothed_mean_te(x_joint, data[[target_col]], smoothing = 10, key_col = "x_joint", data = data)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x1 <- as.character(data[[input_cols[1]]])
    x2 <- as.character(data[[input_cols[2]]])
    x_joint <- paste0(x1, "___", x2)
    .apply_smoothed_mean_te(x_joint, state, key_col = "x_joint")
  },
  name_generator = function(gene) .gene_col_name(gene, "cite")
)

# Hierarchical Exact-Value + Quantile-Bin Empirical-Bayes Target Encoding for Numerics
evo_transformers$numeric_target_encode <- create_transformer(
  name = "numeric_target_encode",
  type = "supervised_unary",
  input_type = "numeric",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    raw_x <- data[[gene$input_cols[1L]]]
    x <- if (is.numeric(raw_x)) as.numeric(raw_x) else suppressWarnings(as.numeric(as.character(raw_x)))
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y) || sum(!is.na(y)) == 0L) {
      return(list(mapping = NULL, breaks = NULL, bin_smoothed = NULL, global_mean = 0, valid = FALSE, loo_train = NULL))
    }

    global_mean <- mean(y, na.rm = TRUE)
    smoothing <- 10.0
    fin_x <- is.finite(x)
    if (!any(fin_x)) {
      return(list(mapping = NULL, breaks = NULL, bin_smoothed = global_mean, global_mean = global_mean, valid = FALSE, loo_train = NULL))
    }

    u_fin_x <- unique(x[fin_x])
    breaks <- NULL
    b_idx <- integer(length(x))
    if (length(u_fin_x) >= 2L) {
      breaks <- unique(stats::quantile(x[fin_x], probs = seq(0, 1, length.out = 33L), na.rm = TRUE, names = FALSE))
      if (length(breaks) >= 2L) {
        b_idx[fin_x] <- findInterval(x[fin_x], breaks, all.inside = TRUE)
      } else {
        b_idx[fin_x] <- 1L
      }
    } else {
      b_idx[fin_x] <- 1L
    }

    has_y <- !is.na(y)
    y_clean <- ifelse(has_y, y, 0.0)
    y_obs <- as.integer(has_y)

    max_b0 <- max(b_idx + 1L)
    bin_sum <- numeric(max_b0)
    bin_n <- numeric(max_b0)
    dt_bin <- data.table::data.table(b_idx = b_idx, y = y)
    bin_stats <- dt_bin[, .(sum_y = sum(y, na.rm = TRUE), n = sum(!is.na(y))), by = b_idx]
    b_slots <- bin_stats$b_idx + 1L
    bin_sum[b_slots] <- bin_stats$sum_y
    bin_n[b_slots] <- as.numeric(bin_stats$n)

    bin_smoothed <- ifelse(
      bin_n > 0,
      (bin_sum + smoothing * global_mean) / (bin_n + smoothing),
      global_mean
    )

    x_key <- rep(".__NA__", length(x))
    x_key[fin_x] <- as.character(signif(x[fin_x], 12))

    dt_ex <- data.table::data.table(x_key = x_key, y = y, b_idx = b_idx)
    ex_stats <- dt_ex[, .(
      sum_y = sum(y, na.rm = TRUE),
      n = sum(!is.na(y)),
      b_idx = b_idx[1L]
    ), by = x_key]
    ex_prior <- bin_smoothed[ex_stats$b_idx + 1L]
    ex_stats[, smoothed := (sum_y + smoothing * ex_prior) / (n + smoothing)]

    mapping <- ex_stats[, .(x_key, smoothed)]
    data.table::setkey(mapping, x_key)

    folds <- .make_oof_folds(y, data = data)
    dt_oof <- data.table::data.table(x_key = x_key, b_idx = b_idx, f = folds, y_clean = y_clean, y_obs = y_obs)
    bf_stats <- dt_oof[, .(fb_sum = sum(y_clean), fb_n = sum(y_obs)), by = .(b_idx, f)]
    bf_stats[, `:=`(
      oof_b_sum = bin_sum[b_idx + 1L] - fb_sum,
      oof_b_n   = pmax(0, bin_n[b_idx + 1L] - fb_n)
    )]
    bf_stats[, bin_oof := ifelse(
      oof_b_n > 0,
      (oof_b_sum + smoothing * global_mean) / (oof_b_n + smoothing),
      global_mean
    )]

    ef_stats <- dt_oof[, .(fe_sum = sum(y_clean), fe_n = sum(y_obs), b_idx = b_idx[1L]), by = .(x_key, f)]
    ef_stats[ex_stats, `:=`(
      oof_sum = i.sum_y - fe_sum,
      oof_n   = pmax(0, i.n - fe_n)
    ), on = "x_key"]
    ef_stats[bf_stats, bin_oof := i.bin_oof, on = .(b_idx, f)]
    ef_stats[!is.finite(bin_oof), bin_oof := global_mean]
    ef_stats[, oof_val := (oof_sum + smoothing * bin_oof) / (oof_n + smoothing)]

    loo_vals <- ef_stats[dt_oof[, .(x_key, f)], on = .(x_key, f)]$oof_val
    loo_vals[!is.finite(loo_vals)] <- global_mean
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(
      mapping = mapping,
      breaks = breaks,
      bin_smoothed = bin_smoothed,
      global_mean = global_mean,
      valid = TRUE,
      loo_train = loo_train
    )
  },
  apply_func = function(data, gene, state) {
    n_out <- nrow(data)
    if (n_out == 0L) return(numeric(0))
    g_mean <- if (!is.null(state) && !is.null(state$global_mean) && is.finite(state$global_mean)) state$global_mean else 0.0
    if (is.null(state) || !isTRUE(state$valid) || is.null(state$bin_smoothed)) {
      return(rep(g_mean, n_out))
    }

    raw_x <- data[[gene$input_cols[1L]]]
    x <- if (is.numeric(raw_x)) as.numeric(raw_x) else suppressWarnings(as.numeric(as.character(raw_x)))
    fin_x <- is.finite(x)

    b_idx <- integer(n_out)
    if (any(fin_x)) {
      if (!is.null(state$breaks) && length(state$breaks) >= 2L) {
        b_idx[fin_x] <- findInterval(x[fin_x], state$breaks, all.inside = TRUE)
      } else {
        b_idx[fin_x] <- 1L
      }
    }
    slots <- pmin(pmax(b_idx + 1L, 1L), length(state$bin_smoothed))
    fallback <- as.numeric(state$bin_smoothed[slots])
    fallback[!is.finite(fallback)] <- g_mean

    if (is.null(state$mapping)) {
      return(fallback)
    }

    x_key <- rep(".__NA__", n_out)
    if (any(fin_x)) {
      x_key[fin_x] <- as.character(signif(x[fin_x], 12))
    }
    dt_q <- data.table::data.table(x_key = x_key)
    res <- tryCatch(state$mapping[dt_q, on = "x_key"]$smoothed, error = function(e) NULL)
    if (is.null(res)) return(fallback)
    miss <- is.na(res) | !is.finite(res)
    if (any(miss)) res[miss] <- fallback[miss]
    as.numeric(res)
  },
  name_generator = function(gene) .gene_col_name(gene, "nte")
)

# Mixed-Type Pairwise Joint Target Encoding (cat x cat, cat x num, num x num)
evo_transformers$pair_target_encode <- create_transformer(
  name = "pair_target_encode",
  type = "supervised_binary",
  input_type = "any",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    cols <- gene$input_cols
    if (length(cols) < 2L) {
      return(list(valid = FALSE, global_mean = 0, loo_train = NULL))
    }
    d1 <- .fit_col_discretizer(data[[cols[1L]]])
    d2 <- .fit_col_discretizer(data[[cols[2L]]])
    k1 <- .apply_col_discretizer(data[[cols[1L]]], d1)
    k2 <- .apply_col_discretizer(data[[cols[2L]]], d2)
    x_joint <- paste0(k1, "___", k2)
    te_state <- .fit_smoothed_mean_te(x_joint, data[[target_col]], smoothing = 10, key_col = "x_joint", data = data)
    loo_train <- te_state$loo_train
    te_state$loo_train <- NULL
    list(
      d1 = d1,
      d2 = d2,
      te = te_state,
      valid = !isTRUE(te_state$valid == FALSE),
      global_mean = te_state$global_mean,
      loo_train = loo_train
    )
  },
  apply_func = function(data, gene, state) {
    n_out <- nrow(data)
    if (n_out == 0L) return(numeric(0))
    if (is.null(state) || !isTRUE(state$valid) || is.null(state$te) || length(gene$input_cols) < 2L) {
      g_mean <- if (!is.null(state) && !is.null(state$global_mean) && is.finite(state$global_mean)) state$global_mean else 0.0
      return(rep(g_mean, n_out))
    }
    cols <- gene$input_cols
    k1 <- .apply_col_discretizer(data[[cols[1L]]], state$d1)
    k2 <- .apply_col_discretizer(data[[cols[2L]]], state$d2)
    x_joint <- paste0(k1, "___", k2)
    .apply_smoothed_mean_te(x_joint, state$te, key_col = "x_joint")
  },
  name_generator = function(gene) .gene_col_name(gene, "pte2")
)

# Target Standard Deviation Encoding (with Leave-One-Out jackknife SD on training data)
evo_transformers$target_sd_encode <- create_transformer(
  name = "target_sd_encode",
  type = "supervised_unary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
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

    folds <- .make_oof_folds(y, data = data)
    has_y <- !is.na(y_c)
    dt_f <- data.table::data.table(
      x = x,
      f = folds,
      yc_clean = ifelse(has_y, y_c, 0.0),
      y_obs = as.integer(has_y)
    )
    f_stats <- dt_f[, .(
      f_s1 = sum(yc_clean),
      f_s2 = sum(yc_clean * yc_clean),
      f_n  = sum(y_obs)
    ), by = .(x, f)]
    f_stats[stats_dt, `:=`(
      oof_n  = pmax(0L, i.n - f_n),
      oof_s1 = i.sum_yc - f_s1,
      oof_s2 = i.sum_yc2 - f_s2
    ), on = "x"]
    f_stats[, sd_oof := ifelse(
      oof_n >= 2L,
      sqrt(pmax(0.0, (oof_s2 - (oof_s1 * oof_s1) / pmax(1L, oof_n)) / pmax(1.0, oof_n - 1.0))),
      global_sd
    )]
    f_stats[, oof_val := (oof_n * sd_oof + smoothing * global_sd) / (oof_n + smoothing)]
    loo_vals <- f_stats[dt_f[, .(x, f)], on = .(x, f)]$oof_val
    loo_vals[!is.finite(loo_vals)] <- global_sd
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(mapping = mapping, global_sd = global_sd, loo_train = loo_train)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$mapping)) {
      return(rep(0, nrow(data)))
    }
    input_cols <- gene$input_cols
    x <- as.character(data[[input_cols[1]]])
    x[is.na(x)] <- ".__NA__"
    dt <- data.table::data.table(x = x)
    res <- tryCatch(state$mapping[dt, on = "x"]$smoothed, error = function(e) NULL)
    if (is.null(res)) return(rep(state$global_sd, nrow(data)))
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
    coord_cols <- paste0(".__c", seq_len(ncol(X_query)))
    dt_u <- data.table::as.data.table(X_query)
    data.table::setnames(dt_u, coord_cols)
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

    folds <- .make_oof_folds(y_clean, data = data)
    dt_gf <- data.table::data.table(gid = grp_id, f = folds, y = y_clean)
    gf_stats <- dt_gf[, .(f_cnt = .N, f_sum = sum(y)), by = .(gid, f)]
    row_gf <- gf_stats[dt_gf[, .(gid, f)], on = .(gid, f)]
    c_self <- pmax(0, u_cnt[grp_id] - row_gf$f_cnt)
    s_self <- u_sum[grp_id] - row_gf$f_sum

    min_c_self_m <- integer(M)
    if (any(u_cnt > 1L)) {
      max_f_per_g <- gf_stats[, .(max_fc = max(f_cnt)), by = gid]
      min_c_self_m[max_f_per_g$gid] <- pmax(0L, u_cnt[max_f_per_g$gid] - max_f_per_g$max_fc)
    }

    nn_idx <- matrix(nn_res$nn.index, nrow = M, ncol = k_u)
    rem_g <- pmax(0.0, as.numeric(k_eff - min_c_self_m))
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

    o_cnt_row <- other_cnt[grp_id]
    o_sum_row <- other_sum[grp_id]
    need_other <- pmax(0.0, as.numeric(k_eff) - c_self)
    take_other <- pmin(need_other, o_cnt_row)
    other_part_sum <- ifelse(o_cnt_row > 0, take_other * (o_sum_row / o_cnt_row), 0.0)
    tot_cnt <- c_self + take_other
    mean_nn <- ifelse(
      c_self >= k_eff,
      s_self / pmax(1, c_self),
      ifelse(tot_cnt > 0, (s_self + other_part_sum) / tot_cnt, global_mean)
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

.build_ridge_basis <- function(X_sub, mu_k, sigma_k, degree) {
  Z <- sweep(sweep(X_sub, 2L, mu_k, "-"), 2L, sigma_k, "/")
  Z[!is.finite(Z)] <- 0.0
  Z <- pmin(pmax(Z, -10.0), 10.0)
  if (identical(as.integer(degree), 2L)) {
    cbind(Z, (Z * Z - 1.0) * sqrt(0.5))
  } else {
    Z
  }
}

# Closed-Form Hat-Matrix Leave-One-Out Ridge Target Encoding (linear / quadratic)
evo_transformers$ridge_oof_encode <- create_transformer(
  name = "ridge_oof_encode",
  type = "supervised_multivariate",
  input_type = "numeric",
  output_type = "numeric",
  fit_func = function(data, gene, target_col) {
    input_cols <- gene$input_cols
    y <- .normalize_target_for_encoding(data[[target_col]])
    if (is.null(y)) {
      return(list(valid = FALSE, global_mean = 0.0, loo_train = NULL))
    }
    valid_y <- is.finite(y)
    if (sum(valid_y) < 4L) {
      return(list(valid = FALSE, global_mean = 0.0, loo_train = NULL))
    }
    y_fit <- y[valid_y]
    global_mean <- mean(y_fit)
    if (!is.finite(global_mean)) global_mean <- 0.0
    if (.safe_sd(y_fit) < 1e-12) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }

    X <- as.matrix(data[, input_cols, with = FALSE])
    if (!is.numeric(X)) storage.mode(X) <- "double"
    X_fit <- X[valid_y, , drop = FALSE]
    mu <- colMeans(X_fit, na.rm = TRUE)
    sigma <- apply(X_fit, 2L, stats::sd, na.rm = TRUE)
    keep_cols <- which(!is.na(sigma) & is.finite(sigma) & sigma >= 1e-8)
    if (length(keep_cols) == 0L) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }
    mu_k <- as.numeric(mu[keep_cols])
    sigma_k <- as.numeric(sigma[keep_cols])

    lambda <- if (!is.null(gene$params$lambda) && is.numeric(gene$params$lambda) &&
                  is.finite(gene$params$lambda) && gene$params$lambda > 0) {
      as.numeric(gene$params$lambda)
    } else {
      1.0
    }
    degree <- if (!is.null(gene$params$degree)) as.integer(gene$params$degree) else 1L

    B_all <- .build_ridge_basis(X[, keep_cols, drop = FALSE], mu_k, sigma_k, degree)
    B_fit_raw <- B_all[valid_y, , drop = FALSE]
    phi_mean <- as.numeric(colMeans(B_fit_raw))
    Phi_all <- sweep(B_all, 2L, phi_mean, "-")
    Phi_fit <- Phi_all[valid_y, , drop = FALSE]
    y_centered <- y_fit - global_mean

    d <- ncol(Phi_fit)
    P_diag <- diag(rep(lambda, d), nrow = d, ncol = d)
    XtX_fit <- crossprod(Phi_fit)
    Xty_fit <- crossprod(Phi_fit, y_centered)
    A <- XtX_fit + P_diag
    R_chol <- tryCatch(chol(A), error = function(e) NULL)
    if (is.null(R_chol)) {
      return(list(valid = FALSE, global_mean = global_mean, loo_train = NULL))
    }
    A_inv <- chol2inv(R_chol)
    beta <- as.numeric(A_inv %*% Xty_fit)

    # K-fold OOF predictions (follows existing CV/MetaCV folds when attached)
    all_folds <- .make_oof_folds(y, data = data)
    folds_fit <- all_folds[valid_y]
    u_f <- sort(unique(folds_fit))
    loo_vals <- global_mean + as.numeric(Phi_all %*% beta)
    if (length(u_f) >= 2L) {
      valid_idx <- which(valid_y)
      sum_y_fit <- sum(y_fit)
      n_fit <- length(y_fit)
      for (fk in u_f) {
        in_k <- which(folds_fit == fk)
        n_oof <- n_fit - length(in_k)
        if (n_oof < 1L) next
        Phi_k <- Phi_fit[in_k, , drop = FALSE]
        A_k <- (XtX_fit - crossprod(Phi_k)) + P_diag
        Xty_k <- Xty_fit - crossprod(Phi_k, y_centered[in_k])
        beta_k <- tryCatch({
          Rk <- chol(A_k)
          as.numeric(chol2inv(Rk) %*% Xty_k)
        }, error = function(e) {
          Phi_oof <- Phi_fit[-in_k, , drop = FALSE]
          Ak2 <- crossprod(Phi_oof) + P_diag
          Rk2 <- tryCatch(chol(Ak2), error = function(e2) NULL)
          if (is.null(Rk2)) beta else as.numeric(chol2inv(Rk2) %*% crossprod(Phi_oof, y_centered[-in_k]))
        })
        g_oof <- (sum_y_fit - sum(y_fit[in_k])) / n_oof
        loo_vals[valid_idx[in_k]] <- g_oof + as.numeric(Phi_k %*% beta_k)
      }
    }

    is_bin <- all(y_fit == 0.0 | y_fit == 1.0)
    clip_bounds <- if (is_bin) {
      c(0.0, 1.0)
    } else {
      s_y <- .safe_sd(y_fit)
      c(min(y_fit) - s_y, max(y_fit) + s_y)
    }
    loo_vals <- pmin(pmax(loo_vals, clip_bounds[1L]), clip_bounds[2L])
    loo_vals[!is.finite(loo_vals)] <- global_mean
    loo_train <- if (length(unique(loo_vals)) > 1L) loo_vals else NULL

    list(
      valid = TRUE,
      keep_cols = keep_cols,
      mu = mu_k,
      sigma = sigma_k,
      phi_mean = phi_mean,
      degree = degree,
      lambda = lambda,
      beta = beta,
      clip_bounds = clip_bounds,
      global_mean = global_mean,
      loo_train = loo_train
    )
  },
  apply_func = function(data, gene, state) {
    n_out <- nrow(data)
    if (n_out == 0L) return(numeric(0))
    g_mean <- if (!is.null(state) && !is.null(state$global_mean) && is.finite(state$global_mean)) state$global_mean else 0.0
    if (is.null(state) || isTRUE(state$valid == FALSE) || is.null(state$beta) || is.null(state$keep_cols)) {
      return(rep(g_mean, n_out))
    }
    input_cols <- gene$input_cols
    X <- as.matrix(data[, input_cols, with = FALSE])
    if (!is.numeric(X)) storage.mode(X) <- "double"
    B_new <- .build_ridge_basis(X[, state$keep_cols, drop = FALSE], state$mu, state$sigma, state$degree)
    Phi_new <- sweep(B_new, 2L, state$phi_mean, "-")
    preds <- g_mean + as.numeric(Phi_new %*% state$beta)
    if (!is.null(state$clip_bounds) && length(state$clip_bounds) == 2L) {
      preds <- pmin(pmax(preds, state$clip_bounds[1L]), state$clip_bounds[2L])
    }
    preds[!is.finite(preds)] <- g_mean
    preds
  },
  name_generator = function(gene) .gene_col_name(gene, "rdgte")
)
