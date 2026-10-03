#' Resolve Parameter Aliases for ML Models and Iterative Routines
#'
#' Extracts and normalizes thread counts, iteration counts, and epoch counts
#' from varying user-supplied argument aliases (`extra_args` or `...`).
#'
#' @param extra_args Named list of arguments passed by caller or user.
#' @param defaults Named list of default values (e.g. `list(threads = 2L, nrounds = 100L)`).
#' @return Named list with resolved and normalized parameters (`threads`, `nrounds`, `epochs`, `iterations`).
#' @keywords internal
#' @noRd
resolve_param_aliases <- function(extra_args, defaults = list()) {
  if (is.null(extra_args) || !is.list(extra_args)) {
    extra_args <- list()
  }
  res <- if (is.list(defaults)) defaults else list()

  # 1. Thread aliases
  thread_aliases <- c("threads", "nthreads", "nthread", "num_threads", "n_jobs")
  for (alias in thread_aliases) {
    if (!is.null(extra_args[[alias]])) {
      res$threads <- as.integer(extra_args[[alias]])
    }
  }

  # 2. Iteration / Round / Epoch aliases (checked in precedence order)
  iter_aliases <- c("num_round", "num_rounds", "n_round", "n_rounds", "nround",
                    "iterations", "n_iterations", "epochs", "n_epochs",
                    "realmlp_epochs", "nrounds")
  for (alias in iter_aliases) {
    if (!is.null(extra_args[[alias]])) {
      val <- as.integer(extra_args[[alias]])
      res$nrounds <- val
      res$epochs <- val
      res$iterations <- val
    }
  }

  if (!is.null(res$threads)) res$threads <- as.integer(res$threads)
  if (!is.null(res$nrounds)) res$nrounds <- as.integer(res$nrounds)
  if (!is.null(res$epochs)) res$epochs <- as.integer(res$epochs)
  if (!is.null(res$iterations)) res$iterations <- as.integer(res$iterations)

  res
}

#' Sanitize Input Feature Matrix for C++ and Tree-Based ML Backends
#'
#' Converts input data (matrix or data.frame) into a clean double-precision numeric
#' matrix safe for C++ ML engines (XGBoost, LightGBM, CatBoost, RealMLP).
#' Converts non-finite numbers (Inf, -Inf, NaN) and values exceeding IEEE 754 32-bit
#' single-precision float limits (~3.402823e38) to NA_real_.
#'
#' @param x A matrix, data.frame, or NULL.
#' @return A numeric matrix of type double with attribute `sanitized = TRUE`, or NULL if `x` is NULL.
#' @keywords internal
#' @noRd
sanitize_feature_matrix <- function(x) {
  if (is.null(x)) return(NULL)
  if (isTRUE(attr(x, "sanitized", exact = TRUE))) return(x)
  if (!is.matrix(x)) {
    x <- if (is.data.frame(x)) data.matrix(x) else as.matrix(x)
  }
  if (!is.numeric(x)) {
    storage.mode(x) <- "double"
  }
  max_float <- 3.402823e38
  x[!is.finite(x) | abs(x) > max_float] <- NA_real_
  attr(x, "sanitized") <- TRUE
  x
}

#' @keywords internal
#' @noRd
.sanitize_feature_matrix <- sanitize_feature_matrix

#' Encode Multiclass Target Vector to Zero-Based Integers
#'
#' Encodes a multiclass target vector into a zero-based integer index vector
#' aligned with the specified class levels.
#'
#' @param y Target vector (factor, character, integer, or numeric).
#' @param classes Optional vector of unique class levels defining the ordering.
#'   If NULL, inferred from `levels(y)` if factor, or sorted unique non-NA values.
#' @return An integer vector with values in `0, ..., length(classes) - 1`, with NAs preserved.
#' @keywords internal
#' @noRd
encode_multiclass_target <- function(y, classes = NULL) {
  if (is.null(y)) return(NULL)
  if (is.null(classes)) {
    classes <- if (is.factor(y)) levels(y) else sort(unique(y[!is.na(y)]))
  }
  as.integer(factor(y, levels = classes)) - 1L
}

#' Format Multiclass Predictions into an N x K Probability Matrix
#'
#' Ensures multiclass predictions are shaped as an N-row by K-column probability
#' matrix, assigning column names when class names are provided.
#'
#' @param preds Vector, matrix, or array of predictions.
#' @param classes Character vector of class labels or a single integer indicating
#'   the number of classes.
#' @return A numeric matrix of dimensions N x K.
#' @keywords internal
#' @noRd
format_multiclass_predictions <- function(preds, classes) {
  if (is.null(preds)) return(NULL)
  if (is.numeric(classes) && length(classes) == 1L) {
    num_class <- as.integer(classes)
    col_names <- NULL
  } else {
    num_class <- length(classes)
    col_names <- as.character(classes)
  }
  if (!is.matrix(preds)) {
    preds <- matrix(preds, ncol = num_class, byrow = TRUE)
  }
  if (!is.null(col_names) && ncol(preds) == length(col_names)) {
    colnames(preds) <- col_names
  }
  preds
}

#' Calculate Metric Improvement and Headroom Closed
#'
#' Computes the gain over baseline fitness and the proportion of available
#' headroom closed towards the ideal target (1.0 for classification/multiclass,
#' 0.0 for regression).
#'
#' @param fitness Numeric value or vector of achieved fitness scores.
#' @param baseline Numeric value or vector of baseline fitness scores.
#' @param task Character string indicating task type: "classification", "multiclass",
#'   or "regression". Default is "classification".
#' @return A list containing:
#'   \item{improvement}{Gain relative to baseline (`fitness - baseline`).}
#'   \item{headroom_closed}{Normalized proportion of available headroom closed, guarded by a 1e-6 threshold.}
#'   \item{gain}{Alias to `improvement` for caller convenience.}
#' @keywords internal
#' @noRd
calculate_headroom <- function(fitness, baseline, task = "classification") {
  if (is.null(fitness) || is.null(baseline)) {
    return(list(improvement = NULL, headroom_closed = NULL, gain = NULL))
  }
  if (length(fitness) == 0L || length(baseline) == 0L) {
    return(list(improvement = numeric(0), headroom_closed = numeric(0), gain = numeric(0)))
  }
  if (length(baseline) == 1L && (!is.finite(baseline) || is.na(baseline))) {
    return(list(improvement = NULL, headroom_closed = NULL, gain = NULL))
  }

  ideal <- if (identical(task, "regression")) 0.0 else 1.0
  gain <- fitness - baseline
  denom <- ideal - baseline

  if (length(denom) == 1L) {
    if (!is.finite(denom) || abs(denom) < 1e-6) {
      headroom_closed <- if (length(gain) == 1L) 0.0 else rep(0.0, length(gain))
    } else {
      headroom_closed <- as.numeric(gain / denom)
      headroom_closed[is.na(fitness) | !is.finite(fitness)] <- 0.0
    }
  } else {
    headroom_closed <- gain / denom
    headroom_closed[!is.finite(denom) | abs(denom) < 1e-6 | is.na(fitness) | !is.finite(fitness)] <- 0.0
  }

  list(
    improvement = gain,
    headroom_closed = headroom_closed,
    gain = gain
  )
}

#' Extract Active Features from an Individual Recipe
#'
#' Extracts engineered gene output column names and concatenates them with
#' active original predictor columns (numeric, categorical, datetime).
#'
#' @param ind An individual object containing `genes`, `numeric_cols`,
#'   `categorical_cols`, and `datetime_cols`.
#' @param target_col Optional character string. If provided, target column is
#'   excluded from the feature list.
#' @param unique Logical. If `TRUE`, duplicate column names are removed. Default is `FALSE`.
#' @return Character vector of feature column names.
#' @keywords internal
#' @noRd
extract_individual_features <- function(ind, target_col = NULL, unique = FALSE) {
  if (is.null(ind)) return(character(0))
  gene_cols <- if (!is.null(ind$genes) && length(ind$genes) > 0) {
    vapply(ind$genes, function(g) g$output_col, character(1))
  } else {
    character(0)
  }
  features <- c(ind$numeric_cols, ind$categorical_cols, ind$datetime_cols, gene_cols)
  if (!is.null(target_col)) {
    features <- setdiff(features, target_col)
  }
  if (isTRUE(unique)) {
    features <- unique(features)
  }
  features
}

#' Apply Complexity Penalty to Individual Fitness
#'
#' Calculates the complexity penalty based on recipe length and active features,
#' adjusts raw evaluation score, and attaches `$raw_fitness`, `$penalty`, and `$fitness`.
#'
#' @param raw_score Numeric raw fitness score from model evaluation.
#' @param ind The individual object to penalize.
#' @param n_samples Integer number of training samples.
#' @param task Character string ("classification", "multiclass", "regression").
#' @param complexity_penalty Numeric penalty weight (>= 0).
#' @param complexity_mode Character string ("none", "bic", "bic_dynamic", "pac_bayes", "pac_bayes_dynamic").
#' @param complexity_floor Numeric floor for dynamic penalty scaling.
#' @param complexity_target Character string ("all_features" or "genes").
#' @param running_best_fitness Current best population fitness for dynamic scaling.
#' @param baseline_fitness Baseline model fitness for dynamic scaling.
#' @param metric Metric name or function.
#' @param data Optional training dataset to infer `n_samples` if NULL.
#' @param shared_full Optional shared full dataset to infer `n_samples` if NULL.
#' @return The modified `ind` object with `$raw_fitness`, `$penalty`, and `$fitness` populated.
#' @keywords internal
#' @noRd
apply_complexity_penalty <- function(raw_score,
                                     ind,
                                     n_samples = NULL,
                                     task = "classification",
                                     complexity_penalty = 0,
                                     complexity_mode = "bic_dynamic",
                                     complexity_floor = 0.20,
                                     complexity_target = "all_features",
                                     running_best_fitness = NULL,
                                     baseline_fitness = NULL,
                                     metric = "default",
                                     data = NULL,
                                     shared_full = NULL) {
  ind$raw_fitness <- raw_score
  ind$penalty <- 0.0

  if (complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
    n_samp <- if (!is.null(n_samples)) {
      n_samples
    } else if (!is.null(data)) {
      nrow(data)
    } else if (!is.null(shared_full)) {
      nrow(shared_full)
    } else {
      100
    }
    n_active_raw <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols)
    n_genes <- length(ind$genes)
    pen <- compute_complexity_penalty(
      n_genes = n_genes,
      n_samples = n_samp,
      running_best_fitness = running_best_fitness,
      baseline_fitness = baseline_fitness,
      metric = metric,
      task = task,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      n_features = n_active_raw + n_genes
    )
    ind$penalty <- pen
    if (task == "regression") {
      ind$fitness <- if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen)
    } else {
      error_gap <- max(0, min(1.0, 1.0 - raw_score))
      ind$fitness <- raw_score - error_gap * pen
    }
  } else {
    ind$fitness <- raw_score
  }

  ind
}

#' Execute Code Block with Isolated RNG Seed
#'
#' Temporarily sets the random number generator seed, executes the provided
#' expression or function, and guarantees the previous `.Random.seed` state in
#' `.GlobalEnv` is faithfully restored on exit.
#'
#' @param seed_val Integer seed value. If NULL, code executes without modifying seed.
#' @param expr Expression or zero-argument function to evaluate.
#' @return Result of evaluating `expr`.
#' @keywords internal
#' @noRd
with_seed <- function(seed_val, expr) {
  if (is.null(seed_val)) {
    return(if (is.function(expr)) expr() else force(expr))
  }
  old_seed <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  } else {
    NULL
  }
  on.exit({
    if (!is.null(old_seed)) {
      assign(".Random.seed", old_seed, envir = .GlobalEnv)
    } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
      rm(".Random.seed", envir = .GlobalEnv)
    }
  }, add = TRUE)

  set.seed(seed_val)
  if (is.function(expr)) expr() else force(expr)
}

#' Column Imputation, Variance Flooring, and Standardization for Numeric Matrices
#'
#' Imputes non-finite and missing values using column means, floors zero-variance
#' columns to standard deviation of 1.0, and standardizes via z-score scaling.
#'
#' @param X Matrix or data.frame of numeric features.
#' @param stats Optional precomputed list containing `mu` (or `center`) and `sigma`
#'   (or `scale`). If NULL, computed from `X`.
#' @return A list containing:
#'   \item{X}{Standardized numeric matrix.}
#'   \item{stats}{List containing `mu` and `sigma`.}
#'   \item{mu}{Column means used for centering and imputation.}
#'   \item{sigma}{Column standard deviations used for scaling.}
#' @keywords internal
#' @noRd
matrix_impute_and_scale <- function(X, stats = NULL) {
  if (is.null(X)) return(list(X = NULL, stats = stats, mu = NULL, sigma = NULL))
  if (!is.matrix(X)) {
    X <- if (is.data.frame(X)) data.matrix(X) else as.matrix(X)
  }
  if (ncol(X) == 0L || nrow(X) == 0L) {
    return(list(X = X, stats = stats, mu = numeric(0), sigma = numeric(0)))
  }

  if (is.null(stats)) {
    mu <- colMeans(X, na.rm = TRUE)
    mu[is.na(mu) | !is.finite(mu)] <- 0.0
    sigma <- apply(X, 2, stats::sd, na.rm = TRUE)
    sigma[is.na(sigma) | !is.finite(sigma) | sigma == 0] <- 1.0
    computed_stats <- list(mu = mu, sigma = sigma)
  } else {
    mu <- if (!is.null(stats$mu)) stats$mu else stats$center
    sigma <- if (!is.null(stats$sigma)) stats$sigma else stats$scale
    if (!is.null(colnames(X)) && !is.null(names(mu)) && all(colnames(X) %in% names(mu))) {
      mu <- mu[colnames(X)]
      sigma <- sigma[colnames(X)]
    }
    computed_stats <- stats
  }

  for (j in seq_len(ncol(X))) {
    bad_idx <- is.na(X[, j]) | !is.finite(X[, j])
    if (any(bad_idx)) {
      X[bad_idx, j] <- mu[j]
    }
  }

  Z <- scale(X, center = mu, scale = sigma)

  list(
    X = Z,
    stats = computed_stats,
    mu = mu,
    sigma = sigma
  )
}

#' Resolve Canonical Metric Name from Aliases
#'
#' Normalizes metric strings and aliases (e.g. hyphens vs underscores, calibration
#' and refinement prefixes) to their canonical internal identifiers.
#'
#' @param metric Character string or custom metric function.
#' @return Canonical metric name string, or the original metric function if a function was provided.
#' @keywords internal
#' @noRd
canonical_metric_name <- function(metric) {
  if (is.null(metric) || is.function(metric)) {
    return(metric)
  }
  m <- tolower(trimws(as.character(metric)))
  if (m %in% c("eval-ts-refinement", "ts-refinement", "ts_refinement", "eval_ts_refinement")) {
    return("ts_refinement")
  }
  if (m %in% c("cal_rmse", "cal-rmse")) {
    return("cal_rmse")
  }
  if (m %in% c("cal_mae", "cal-mae")) {
    return("cal_mae")
  }
  if (m %in% c("binary_logloss", "multi_logloss", "mlogloss")) {
    return("logloss")
  }
  m
}

#' Default Physical Core Thread Count
#'
#' @return Integer scalar representing default threads on host machine.
#' @keywords internal
#' @noRd
default_threads <- function() {
  max(1L, parallel::detectCores(logical = FALSE), na.rm = TRUE)
}

#' Apply Iteration Target to Model Arguments Across Evaluator Aliases
#'
#' Sets the target iteration or epoch count in `args` according to the evaluator type,
#' updating any active iteration aliases and ensuring default fields (epochs for realmlp,
#' nrounds otherwise) are populated.
#'
#' @param args List of model arguments.
#' @param target_iters Integer target iterations or epochs.
#' @param evaluator Evaluator name or object.
#' @return Updated arguments list.
#' @keywords internal
#' @noRd
apply_iteration_target <- function(args, target_iters, evaluator) {
  if (is.null(target_iters) || !is.numeric(target_iters) || is.na(target_iters) || target_iters <= 0) {
    return(args)
  }
  target_iters <- as.integer(target_iters)
  iter_aliases <- c("nrounds", "num_rounds", "n_rounds", "num_round", "nround",
                    "epochs", "n_epochs", "iterations", "n_iterations", "realmlp_epochs")
  for (alias in iter_aliases) {
    if (alias %in% names(args)) {
      args[[alias]] <- target_iters
    }
  }
  unwrapped <- unwrap_evaluator(evaluator)
  if (!any(c("epochs", "n_epochs") %in% names(args)) && identical(unwrapped, "realmlp")) {
    args$epochs <- target_iters
  }
  if (!any(c("nrounds", "iterations") %in% names(args))) {
    args$nrounds <- target_iters
  }
  args$early_stopping_rounds <- 0L
  args$early_stopping_round <- 0L
  args
}
