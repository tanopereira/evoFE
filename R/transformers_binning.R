# Binning & rank transformers: quantile/log binning, ECDF ranks.
# Split out of transformers.R; loaded after it (alphabetical file order).

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
evo_transformers$rank_transform <- create_transformer(
  name = "rank_transform",
  type = "unary",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    x <- data[[gene$input_cols[1]]]
    x_clean <- x[!is.na(x) & is.finite(x)]
    if (length(x_clean) == 0) return(list(ecdf_fn = NULL))
    list(ecdf_fn = stats::ecdf(x_clean))
  },
  apply_func = function(data, gene, state = NULL) {
    x <- data[[gene$input_cols[1]]]
    if (is.null(state) || is.null(state$ecdf_fn)) {
      # Fallback: within-batch rank scaled to [0, 1]
      r <- rank(x, ties.method = "average", na.last = "keep")
      r[is.na(r)] <- 0.5 * length(x)
      return(r / length(x))
    }
    res <- state$ecdf_fn(x)
    res[is.na(res)] <- 0.5
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "rnk")
)

# Weight of Evidence Encoding (binary classification only)
