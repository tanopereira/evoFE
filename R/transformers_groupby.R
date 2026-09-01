# Group-by aggregation transformers (categorical x numeric).
# Split out of transformers.R; loaded after it (alphabetical file order).

evo_transformers$groupby_mean <- create_transformer(
  name = "groupby_mean",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = mean(dt$n, na.rm = TRUE))
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
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
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = .safe_sd(n)), by = c]
    mapping[is.na(val), val := 0]
    data.table::setkey(mapping, c)
    global_sd <- .safe_sd(dt$n)
    if (is.na(global_sd)) global_sd <- 0
    list(mapping = mapping, default_val = global_sd)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
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
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    valid_n <- dt$n[!is.na(dt$n)]
    global_max <- if (length(valid_n) > 0) max(valid_n) else 0
    mapping <- dt[, .(val = if (all(is.na(n))) NA_real_ else max(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = global_max)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
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
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    valid_n <- dt$n[!is.na(dt$n)]
    global_min <- if (length(valid_n) > 0) min(valid_n) else 0
    mapping <- dt[, .(val = if (all(is.na(n))) NA_real_ else min(n, na.rm = TRUE)), by = c]
    mapping[is.infinite(val), val := NA]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = global_min)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbmn")
)

# --- STATEFUL MULTIVARIATE TRANSFORMERS (ADDITIONAL) ---

# UMAP Dimension Reduction
evo_transformers$groupby_ratio <- create_transformer(
  name = "groupby_ratio",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    list(mapping = mapping, default_val = mean(dt$n, na.rm = TRUE))
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_cat <- data[[input_cols[1]]]
    x_num <- .to_numeric(data[[input_cols[2]]])
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
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(mean_val = mean(n, na.rm = TRUE), sd_val = .safe_sd(n)), by = c]
    mapping[is.na(sd_val), sd_val := 0]
    data.table::setkey(mapping, c)
    global_mean <- mean(dt$n, na.rm = TRUE)
    global_sd <- .safe_sd(dt$n)
    list(mapping = mapping, default_mean = global_mean, default_sd = global_sd)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_cat <- data[[input_cols[1]]]
    x_num <- .to_numeric(data[[input_cols[2]]])
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
evo_transformers$groupby_median <- create_transformer(
  name = "groupby_median",
  type = "mixed_binary",
  input_type = "mixed",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = stats::median(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    global_med <- stats::median(dt$n, na.rm = TRUE)
    if (is.na(global_med)) global_med <- 0
    list(mapping = mapping, default_val = global_med)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt  <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    res[is.na(res)] <- state$default_val
    res
  },
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
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = stats::quantile(n, probs = q, na.rm = TRUE, names = FALSE)), by = c]
    data.table::setkey(mapping, c)
    global_q <- stats::quantile(dt$n, probs = q, na.rm = TRUE, names = FALSE)
    if (is.na(global_q)) global_q <- 0
    list(mapping = mapping, default_val = global_q)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x <- data[[input_cols[1]]]
    if (is.null(x) || length(x) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    dt  <- data.table::data.table(c = x)
    res <- state$mapping[dt, on = "c"]$val
    if (is.null(res)) {
      def_val <- if (!is.null(state$default_val)) state$default_val else 0
      return(rep(def_val, nrow(data)))
    }
    res[is.na(res)] <- state$default_val
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbq")
)

# Group-by Signed Log (composite: deviation from group mean with signed log damping)
evo_transformers$groupby_signed_log <- create_transformer(
  name = "groupby_signed_log",
  type = "mixed_binary",
  input_type = "mixed",
  output_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    input_cols <- gene$input_cols
    cat_col <- input_cols[1]
    num_col <- input_cols[2]
    dt <- data.table::data.table(c = data[[cat_col]], n = .to_numeric(data[[num_col]]))
    mapping <- dt[, .(val = mean(n, na.rm = TRUE)), by = c]
    data.table::setkey(mapping, c)
    global_mean <- mean(dt$n, na.rm = TRUE)
    if (is.na(global_mean)) global_mean <- 0
    list(mapping = mapping, default_val = global_mean)
  },
  apply_func = function(data, gene, state) {
    input_cols <- gene$input_cols
    x_cat <- data[[input_cols[1]]]
    x_num <- .to_numeric(data[[input_cols[2]]])

    if (is.null(x_cat) || length(x_cat) == 0 || is.null(state) || is.null(state$mapping)) {
      def_val <- if (!is.null(state) && !is.null(state$default_val)) state$default_val else 0
      diff <- x_num - def_val
      res <- sign(diff) * log1p(abs(diff))
      res[!is.finite(res)] <- 0
      return(res)
    }

    dt <- data.table::data.table(c = x_cat)
    means <- state$mapping[dt, on = "c"]$val
    means[is.na(means)] <- state$default_val

    diff <- x_num - means
    res <- sign(diff) * log1p(abs(diff))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "gbslog")
)

# Concatenation of Categorical Columns
