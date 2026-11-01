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
    
    # Calculate category means, within-category variances, and counts
    dt <- data.table::data.table(x = x, y = y)
    stats <- dt[, .(mean = mean(y, na.rm = TRUE), var = stats::var(y, na.rm = TRUE), n = sum(!is.na(y))), by = x]
    
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
    
    list(mapping = mapping, global_mean = global_mean)
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
    stats_dt <- dt[, .(cat_q = as.numeric(stats::quantile(y, probs = q, na.rm = TRUE)), n = .N), by = x]

    smoothing <- 10
    stats_dt[, smoothed := (n * cat_q + smoothing * global_q) / (n + smoothing)]

    mapping <- stats_dt[, .(x, smoothed)]
    data.table::setkey(mapping, x)

    list(mapping = mapping, global_q = global_q)
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
