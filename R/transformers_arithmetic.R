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

# Fourier Basis (Periodic mapping with scale and phase)
evo_transformers$fourier_basis <- create_transformer(
  name = "fourier_basis",
  type = "unary",
  input_type = "numeric",
  apply_func = function(data, gene, state = NULL) {
    x <- as.numeric(data[[gene$input_cols[1]]])
    scale <- if (!is.null(gene$params$scale)) gene$params$scale else 1.0
    phase <- if (!is.null(gene$params$phase)) gene$params$phase else 0.0
    res <- suppressWarnings(sin(scale * x + phase))
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "fbr")
)

# Robust Scale (Median/IQR standardization)
evo_transformers$robust_scale <- create_transformer(
  name = "robust_scale",
  type = "unary",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    x <- as.numeric(data[[gene$input_cols[1]]])
    x_clean <- x[!is.na(x) & is.finite(x)]
    if (length(x_clean) == 0) return(list(center = 0.0, scale = 1.0))
    med <- stats::median(x_clean)
    qs <- stats::quantile(x_clean, probs = c(0.25, 0.75), names = FALSE, na.rm = TRUE)
    iqr_val <- qs[2] - qs[1]
    scale_val <- if (is.finite(iqr_val) && iqr_val > 1e-8) {
      iqr_val
    } else {
      sd_val <- stats::sd(x_clean)
      if (is.finite(sd_val) && sd_val > 1e-8) sd_val else 1.0
    }
    list(center = med, scale = scale_val)
  },
  apply_func = function(data, gene, state = NULL) {
    x <- as.numeric(data[[gene$input_cols[1]]])
    if (is.null(state) || is.null(state$center) || is.null(state$scale)) {
      x_clean <- x[!is.na(x) & is.finite(x)]
      if (length(x_clean) == 0) return(rep(0.0, length(x)))
      med <- stats::median(x_clean)
      qs <- stats::quantile(x_clean, probs = c(0.25, 0.75), names = FALSE, na.rm = TRUE)
      iqr_val <- qs[2] - qs[1]
      scale_val <- if (is.finite(iqr_val) && iqr_val > 1e-8) {
        iqr_val
      } else {
        sd_val <- stats::sd(x_clean)
        if (is.finite(sd_val) && sd_val > 1e-8) sd_val else 1.0
      }
      res <- (x - med) / scale_val
    } else {
      res <- (x - state$center) / state$scale
    }
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "rsc")
)

# Smooth Clip (Soft outlier dampening via hyperbolic tangent)
evo_transformers$smooth_clip <- create_transformer(
  name = "smooth_clip",
  type = "unary",
  input_type = "numeric",
  fit_func = function(data, gene, target_col = NULL) {
    x <- as.numeric(data[[gene$input_cols[1]]])
    x_clean <- x[!is.na(x) & is.finite(x)]
    low_p <- if (!is.null(gene$params$low_pct)) gene$params$low_pct else 0.01
    high_p <- if (!is.null(gene$params$high_pct)) gene$params$high_pct else 0.99
    if (length(x_clean) == 0) {
      return(list(q_low = 0.0, q_high = 0.0, margin = 1.0))
    }
    qs <- stats::quantile(x_clean, probs = c(low_p, high_p), names = FALSE, na.rm = TRUE)
    q_low <- qs[1]
    q_high <- qs[2]
    margin <- if (q_high > q_low) (q_high - q_low) * 0.1 else 1.0
    if (!is.finite(margin) || margin < 1e-8) margin <- 1.0
    list(q_low = q_low, q_high = q_high, margin = margin)
  },
  apply_func = function(data, gene, state = NULL) {
    x <- as.numeric(data[[gene$input_cols[1]]])
    if (is.null(state) || is.null(state$q_low) || is.null(state$q_high) || is.null(state$margin)) {
      x_clean <- x[!is.na(x) & is.finite(x)]
      if (length(x_clean) == 0) return(rep(0.0, length(x)))
      low_p <- if (!is.null(gene$params$low_pct)) gene$params$low_pct else 0.01
      high_p <- if (!is.null(gene$params$high_pct)) gene$params$high_pct else 0.99
      qs <- stats::quantile(x_clean, probs = c(low_p, high_p), names = FALSE, na.rm = TRUE)
      q_low <- qs[1]
      q_high <- qs[2]
      margin <- if (q_high > q_low) (q_high - q_low) * 0.1 else 1.0
      if (!is.finite(margin) || margin < 1e-8) margin <- 1.0
    } else {
      q_low <- state$q_low
      q_high <- state$q_high
      margin <- state$margin
    }
    res <- x
    above <- !is.na(x) & x > q_high
    below <- !is.na(x) & x < q_low
    if (any(above)) {
      res[above] <- q_high + margin * tanh((x[above] - q_high) / margin)
    }
    if (any(below)) {
      res[below] <- q_low + margin * tanh((x[below] - q_low) / margin)
    }
    res[!is.finite(res)] <- 0
    res
  },
  name_generator = function(gene) .gene_col_name(gene, "scl")
)
