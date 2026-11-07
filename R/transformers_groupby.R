# Group-by aggregation transformers (categorical x numeric).
# Split out of transformers.R; loaded after it (alphabetical file order).

.fit_groupby_mean <- function(data, gene, target_col = NULL) {
  input_cols <- gene$input_cols
  cat_col <- input_cols[1]
  num_col <- input_cols[2]
  c_val <- as.character(data[[cat_col]])
  c_val[is.na(c_val)] <- ".__NA__"
  dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
  mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
  data.table::setkey(mapping, c)
  global_mean <- mean(dt$n, na.rm = TRUE)
  if (is.na(global_mean)) global_mean <- 0
  list(mapping = mapping, default_val = global_mean)
}

.fit_groupby_median <- function(data, gene, target_col = NULL) {
  input_cols <- gene$input_cols
  cat_col <- input_cols[1]
  num_col <- input_cols[2]
  c_val <- as.character(data[[cat_col]])
  c_val[is.na(c_val)] <- ".__NA__"
  dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
  mapping <- dt[, .(val = stats::median(n, na.rm = TRUE)), by = c]
  data.table::setkey(mapping, c)
  global_med <- stats::median(dt$n, na.rm = TRUE)
  if (is.na(global_med)) global_med <- 0
  list(mapping = mapping, default_val = global_med)
}

.apply_groupby_lookup <- function(data, gene, state) {
  input_cols <- gene$input_cols
  x <- data[[input_cols[1]]]
  if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
    def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
    return(rep(def_val, nrow(data)))
  }
  x_char <- as.character(x)
  x_char[is.na(x_char)] <- ".__NA__"
  dt <- data.table::data.table(c = x_char)
  res <- tryCatch(state$mapping[dt, on = "c"]$val, error = function(e) NULL)
  def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
  if (is.null(res)) {
    return(rep(def_val, nrow(data)))
  }
  res[is.na(res)] <- def_val
  res
}

evo_transformers$groupby_mean <- create_transformer(
  name = "groupby_mean",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = .fit_groupby_mean,
  apply_func = .apply_groupby_lookup,
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
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = .safe_sd(n)), by = c]
    mapping[is.na(val), val := 0]
    data.table::setkey(mapping, c)
    global_sd <- .safe_sd(dt$n)
    if (is.na(global_sd)) global_sd <- 0
    list(mapping = mapping, default_val = global_sd)
  },
  apply_func = .apply_groupby_lookup,
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
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    valid_n <- dt$n[!is.na(dt$n)]
    global_max <- if (length(valid_n) > 0) max(valid_n) else 0
    mapping <- dt[, .(val = if (all(is.na(n))) NA_real_ else max(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = global_max)
  },
  apply_func = .apply_groupby_lookup,
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
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    valid_n <- dt$n[!is.na(dt$n)]
    global_min <- if (length(valid_n) > 0) min(valid_n) else 0
    mapping <- dt[, .(val = if (all(is.na(n))) NA_real_ else min(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = global_min)
  },
  apply_func = .apply_groupby_lookup,
  name_generator = function(gene) .gene_col_name(gene, "gbmn")
)

# --- STATEFUL MULTIVARIATE TRANSFORMERS (ADDITIONAL) ---

# Group-by Ratio
evo_transformers$groupby_ratio <- create_transformer(
  name = "groupby_ratio",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = .fit_groupby_mean,
  apply_func = function(data, gene, state) {
    if (is.null(state) || is.null(state$mapping)) return(rep(0, nrow(data)))
    input_cols <- gene$input_cols
    x_num <- .to_numeric(data[[input_cols[2]]])
    means <- .apply_groupby_lookup(data, gene, state)
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
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(mean_val = mean(n, na.rm = TRUE), sd_val = .safe_sd(n)), by = c]
    mapping[is.na(sd_val), sd_val := 0]
    data.table::setkey(mapping, c)
    global_mean <- mean(dt$n, na.rm = TRUE)
    global_sd <- .safe_sd(dt$n)
    list(mapping = mapping, default_mean = global_mean, default_sd = global_sd)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || is.null(state$mapping)) return(rep(0, nrow(data)))
    input_cols <- gene$input_cols
    x_cat <- as.character(data[[input_cols[1]]])
    x_cat[is.na(x_cat)] <- ".__NA__"
    x_num <- .to_numeric(data[[input_cols[2]]])
    dt <- data.table::data.table(c = x_cat)
    mapped <- tryCatch(state$mapping[dt, on = "c"], error = function(e) NULL)
    if (is.null(mapped)) return(rep(0, nrow(data)))
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

# Group-by Median
evo_transformers$groupby_median <- create_transformer(
  name = "groupby_median",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = .fit_groupby_median,
  apply_func = .apply_groupby_lookup,
  name_generator = function(gene) .gene_col_name(gene, "gbmed")
)

# Group-by Quantile (Q1 or Q3)
evo_transformers$groupby_quantile <- create_transformer(
  name = "groupby_quantile",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    q <- if (!is.null(gene$params$q)) gene$params$q else 0.25
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = stats::quantile(n, probs = q, na.rm = TRUE, names = FALSE)), by = c]
    data.table::setkey(mapping, c)
    global_q <- stats::quantile(dt$n, probs = q, na.rm = TRUE, names = FALSE)
    if (is.na(global_q)) global_q <- 0
    list(mapping = mapping, default_val = global_q)
  },
  apply_func = .apply_groupby_lookup,
  name_generator = function(gene) .gene_col_name(gene, "gbq")
)

# Group-by Signed Log (composite: deviation from group mean with signed log damping)
evo_transformers$groupby_signed_log <- create_transformer(
  name = "groupby_signed_log",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = .fit_groupby_mean,
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_num <- .to_numeric(data[[input_cols[2]]])
    means <- .apply_groupby_lookup(data, gene, state)
    diff <- x_num - means
    res <- sign(diff) * log1p(abs(diff))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbslog")
)

# Group-by Difference from Mean (additive centered deviation: x_num - group_mean)
evo_transformers$groupby_diff <- create_transformer(
  name = "groupby_diff",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = .fit_groupby_mean,
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_num <- .to_numeric(data[[input_cols[2]]])
    means <- .apply_groupby_lookup(data, gene, state)
    res <- x_num - means
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbd")
)

# Group-by Difference from Median (robust centered deviation: x_num - group_median)
evo_transformers$groupby_median_diff <- create_transformer(
  name = "groupby_median_diff",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = .fit_groupby_median,
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_num <- .to_numeric(data[[input_cols[2]]])
    medians <- .apply_groupby_lookup(data, gene, state)
    res <- x_num - medians
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbmd")
)

# Group-by Range Position: normalized position (x_num - group_min) / (group_max - group_min) in [0, 1]
evo_transformers$groupby_range_pos <- create_transformer(
  name = "groupby_range_pos",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    c_val <- as.character(data[[cat_col]])
    c_val[is.na(c_val)] <- ".__NA__"
    dt <- data.table::data.table(c = c_val, n = .to_numeric(data[[num_col]]))
    valid_n <- dt$n[!is.na(dt$n) & is.finite(dt$n)]
    global_min <- if (length(valid_n) > 0L) min(valid_n) else 0.0
    global_max <- if (length(valid_n) > 0L) max(valid_n) else 0.0
    mapping <- dt[, .(
      min_val = if (all(is.na(n))) NA_real_ else min(n, na.rm = TRUE),
      max_val = if (all(is.na(n))) NA_real_ else max(n, na.rm = TRUE)
    ), by = c]
    mapping[!is.finite(min_val), min_val := NA_real_]
    mapping[!is.finite(max_val), max_val := NA_real_]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_min = global_min, default_max = global_max)
  },
  apply_func = function(data, gene, state) {
    if (is.null(state) || is.null(state$mapping)) return(rep(0.5, nrow(data)))
    input_cols <- gene$input_cols
    x_cat <- as.character(data[[input_cols[1]]])
    x_cat[is.na(x_cat)] <- ".__NA__"
    x_num <- .to_numeric(data[[input_cols[2]]])
    dt <- data.table::data.table(c = x_cat)
    mapped <- tryCatch(state$mapping[dt, on = "c"], error = function(e) NULL)
    if (is.null(mapped)) return(rep(0.5, nrow(data)))
    mins <- mapped$min_val
    maxs <- mapped$max_val
    mins[is.na(mins)] <- state$default_min
    maxs[is.na(maxs)] <- state$default_max
    rng <- maxs - mins
    res <- ifelse(rng <= 1e-12 | !is.finite(rng), 0.5, (x_num - mins) / rng)
    res <- pmin(1.0, pmax(0.0, res))
    res[!is.finite(res)] <- 0.5
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbrp")
)

.extract_sorted_knots <- function(x, max_knots = 256L) {
  x_clean <- x[!is.na(x) & is.finite(x)]
  if (length(x_clean) == 0L) return(numeric(0))
  if (length(x_clean) <= max_knots) {
    sort(x_clean)
  } else {
    as.numeric(stats::quantile(x_clean, probs = seq(0, 1, length.out = max_knots), names = FALSE, na.rm = TRUE))
  }
}

.eval_midrank_knots <- function(v, knots) {
  L <- length(knots)
  if (L == 0L) return(rep(0.5, length(v)))
  res <- rep(0.5, length(v))
  fin <- !is.na(v) & is.finite(v)
  if (!any(fin)) return(res)
  vf <- v[fin]
  r_high <- findInterval(vf, knots)
  r_low <- findInterval(vf, knots, left.open = TRUE)
  res[fin] <- (r_high + r_low) / (2.0 * L)
  res
}

# Group-by Within-Group Percentile Rank in [0, 1]
evo_transformers$groupby_rank <- create_transformer(
  name = "groupby_rank",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    x_cat <- as.character(data[[input_cols[1]]])
    x_cat[is.na(x_cat)] <- ".__NA__"
    x_num <- .to_numeric(data[[input_cols[2]]])
    global_knots <- .extract_sorted_knots(x_num)
    grp_list <- split(x_num, x_cat)
    group_knots <- lapply(grp_list, .extract_sorted_knots)
    list(group_knots = group_knots, global_knots = global_knots)
  },
  apply_func = function(data, gene, state) {
    n_rows <- nrow(data)
    if (n_rows == 0L) return(numeric(0))
    if (is.null(state) || is.null(state$group_knots)) return(rep(0.5, n_rows))
    input_cols <- gene$input_cols
    x_cat <- as.character(data[[input_cols[1]]])
    x_cat[is.na(x_cat)] <- ".__NA__"
    x_num <- .to_numeric(data[[input_cols[2]]])
    res <- rep(0.5, n_rows)
    idx_by_grp <- split(seq_len(n_rows), x_cat)
    for (g_name in names(idx_by_grp)) {
      idx <- idx_by_grp[[g_name]]
      knots <- state$group_knots[[g_name]]
      if (is.null(knots) || length(knots) < 2L) knots <- state$global_knots
      res[idx] <- .eval_midrank_knots(x_num[idx], knots)
    }
    res[!is.finite(res)] <- 0.5
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbrnk")
)

# Group-by Number of Unique Categories (cat1 -> unique count of cat2)
evo_transformers$groupby_nunique <- create_transformer(
  name = "groupby_nunique",
  type = "binary",
  input_type = "categorical",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    c1 <- as.character(data[[input_cols[1]]])
    c1[is.na(c1)] <- ".__NA__"
    c2 <- data[[input_cols[2]]]
    dt <- data.table::data.table(c = c1, c2 = c2)
    mapping <- dt[, .(val = as.numeric(data.table::uniqueN(c2, na.rm = TRUE))), by = c]
    data.table::setkey(mapping, c)
    def_val <- if (nrow(mapping) > 0L) stats::median(mapping$val, na.rm = TRUE) else 0.0
    if (is.na(def_val) || !is.finite(def_val)) def_val <- 0.0
    list(mapping = mapping, default_val = def_val)
  },
  apply_func = .apply_groupby_lookup,
  name_generator = function(gene) .gene_col_name(gene, "gbnu")
)
