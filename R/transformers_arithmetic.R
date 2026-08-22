# Arithmetic transformers: safe logs, powers, ratios of numeric columns.
# Split out of transformers.R; loaded after it (alphabetical file order).

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
  name_generator = function(gene) .gene_col_name(gene, "log")
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
  name_generator = function(gene) .gene_col_name(gene, "sqrt")
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
  name_generator = function(gene) .gene_col_name(gene, "rec")
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
  name_generator = function(gene) .gene_col_name(gene, "add"),
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
  name_generator = function(gene) .gene_col_name(gene, "sub")
)

evo_transformers$multiply <- create_transformer(
  name = "multiply",
  type = "multivariate",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    input_cols <- gene$input_cols
    Reduce(`*`, lapply(input_cols, function(c) as.numeric(data[[c]])))
  },
  name_generator = function(gene) .gene_col_name(gene, "mul"),
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
  name_generator = function(gene) .gene_col_name(gene, "div")
)

# --- STATEFUL SUPERVISED TRANSFORMERS ---

# Target Encoding
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
  name_generator = function(gene) .gene_col_name(gene, "nd")
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
  name_generator = function(gene) .gene_col_name(gene, "lr")
)

# Random Projection
evo_transformers$power <- create_transformer(
  name = "power",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    x <- data[[gene$input_cols[1]]]
    p <- if (!is.null(gene$params$p)) gene$params$p else 2
    # Use signed power to handle negatives: sign(x) * |x|^p
    res <- sign(x) * abs(x)^p
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "pow")
)

# Displaced Log Transform
evo_transformers$displaced_log <- create_transformer(
  name = "displaced_log",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    x <- data[[gene$input_cols[1]]]
    displacement <- if (!is.null(gene$params$displacement)) gene$params$displacement else 100
    res <- log1p(abs(x + displacement))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "dlog")
)

# Rank Transform (ECDF-based percentile rank)
