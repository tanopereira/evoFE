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

#' Align Categorical Columns to a Frozen Training Schema
#'
#' Returns a shallow-copied \code{data.table}/\code{data.frame} whose factor or
#' character columns listed in \code{cat_schema$names} are aligned to
#' \code{factor(col, levels = cat_schema$levels[[nm]])} so unseen levels and NAs
#' map to \code{NA} and integer codes \code{1..K} never drift between train,
#' validation, and prediction splits. Logical columns are never converted.
#'
#' @param dt A \code{data.table}, \code{data.frame}, or \code{NULL}.
#' @param cat_schema Categorical schema list with \code{names} and \code{levels}.
#' @return Aligned \code{data.table}/\code{data.frame}, or \code{dt} unchanged when
#'   no alignment is needed.
#' @keywords internal
#' @noRd
.align_cat_schema <- function(dt, cat_schema) {
  if (is.null(dt) || is.null(cat_schema) || is.null(cat_schema$names) || length(cat_schema$names) == 0L) {
    return(dt)
  }
  if (!is.data.frame(dt)) return(dt)
  dt_names <- names(dt)
  schema_levels <- cat_schema$levels
  if (is.null(dt_names) || is.null(schema_levels)) return(dt)

  target_cols <- intersect(cat_schema$names, dt_names)
  if (length(target_cols) == 0L) return(dt)

  is_dt <- data.table::is.data.table(dt)
  out <- NULL
  for (nm in target_cols) {
    lvls <- schema_levels[[nm]]
    if (is.null(lvls)) next
    col <- dt[[nm]]
    if (is.logical(col)) next
    if (is.factor(col)) {
      if (identical(levels(col), lvls)) next
      if (is.null(out)) out <- if (is_dt) shallow_copy_dt(dt) else dt
      aligned <- factor(col, levels = lvls)
      if (is_dt) data.table::set(out, j = nm, value = aligned) else out[[nm]] <- aligned
    } else if (is.character(col)) {
      if (is.null(out)) out <- if (is_dt) shallow_copy_dt(dt) else dt
      aligned <- factor(col, levels = lvls)
      if (is_dt) data.table::set(out, j = nm, value = aligned) else out[[nm]] <- aligned
    }
  }
  if (is.null(out)) dt else out
}

#' Sanitize Input Feature Matrix for C++ and Tree-Based ML Backends
#'
#' Converts input data (matrix or data.frame) into a clean double-precision numeric
#' matrix safe for C++ ML engines (XGBoost, LightGBM, CatBoost, RealMLP).
#' Converts non-finite numbers (Inf, -Inf, NaN) and values exceeding IEEE 754 32-bit
#' single-precision float limits (~3.402823e38) to NA_real_, while preserving
#' \code{1..K} + \code{NA_real_} encoding and \code{attr(mat, "cat_schema")} for
#' factor and character columns (never converting logical columns to categorical).
#'
#' @param x A matrix, data.frame, data.table, or NULL.
#' @param threads Integer thread count for C++ sanitization.
#' @param cat_schema Optional categorical schema from training split for level alignment.
#' @return A numeric matrix of type double with attribute \code{sanitized = TRUE}
#'   (and \code{cat_schema} when categorical columns are present), or NULL if \code{x} is NULL.
#' @keywords internal
#' @noRd
sanitize_feature_matrix <- function(x, threads = default_threads(), cat_schema = NULL) {
  if (is.null(x)) return(NULL)
  if (inherits(x, "sparseMatrix")) return(x)
  if (isTRUE(attr(x, "sanitized", exact = TRUE))) {
    if (!is.null(cat_schema) && is.null(attr(x, "cat_schema", exact = TRUE))) {
      attr(x, "cat_schema") <- cat_schema
    }
    return(x)
  }
  if (!is.numeric(threads) || is.na(threads) || threads < 1L) threads <- 1L
  threads <- as.integer(threads)

  out_schema <- NULL
  if (is.data.frame(x)) {
    if (!is.null(cat_schema)) {
      x <- .align_cat_schema(x, cat_schema)
    }
    char_cols <- names(x)[vapply(x, is.character, logical(1))]
    if (length(char_cols) > 0L) {
      is_dt <- data.table::is.data.table(x)
      if (is_dt) x <- shallow_copy_dt(x)
      for (cn in char_cols) {
        col_v <- x[[cn]]
        lvls <- sort(unique(col_v[!is.na(col_v)]), method = "radix")
        fac_v <- factor(col_v, levels = lvls)
        if (is_dt) data.table::set(x, j = cn, value = fac_v) else x[[cn]] <- fac_v
      }
    }
    is_fac <- vapply(x, is.factor, logical(1))
    cat_idx <- which(is_fac)
    if (length(cat_idx) > 0L) {
      cat_names <- names(x)[cat_idx]
      if (is.null(cat_names)) cat_names <- paste0("V", cat_idx)
      cat_lvls <- lapply(cat_idx, function(idx) levels(x[[idx]]))
      names(cat_lvls) <- cat_names
      cat_cards <- vapply(cat_lvls, length, integer(1))
      out_schema <- list(
        cols = as.integer(cat_idx),
        names = as.character(cat_names),
        levels = cat_lvls,
        cardinalities = cat_cards
      )
    }
  }

  res <- rcpp_sanitize_feature_matrix(x, threads)
  if (is.null(res)) {
    if (!is.matrix(x)) {
      x <- if (is.data.frame(x)) data.matrix(x) else as.matrix(x)
    }
    if (!is.numeric(x)) {
      storage.mode(x) <- "double"
    }
    res <- rcpp_sanitize_feature_matrix(x, threads)
  }

  if (is.null(out_schema) && !is.null(cat_schema) && !is.null(cat_schema$cols) && length(cat_schema$cols) > 0L) {
    out_schema <- cat_schema
  }
  if (!is.null(res) && !is.null(out_schema)) {
    attr(res, "cat_schema") <- out_schema
  }
  res
}

#' @keywords internal
#' @noRd
.sanitize_feature_matrix <- sanitize_feature_matrix

#' Shallow Copy a data.table Container Preserving evoFE Metadata Attributes
#'
#' Creates a new data.table column-pointer container sharing existing column vectors
#' without deep-copying data, while preserving `.row_id` and `.global_state_cache`
#' attributes so appending new gene columns via `:=` does not mutate shared splits.
#'
#' @param dt A data.table, data.frame, or NULL.
#' @return A shallow-copied data.table with preserved attributes.
#' @keywords internal
#' @noRd
shallow_copy_dt <- function(dt) {
  if (is.null(dt)) return(NULL)
  if (!data.table::is.data.table(dt)) return(dt)
  row_id <- attr(dt, ".row_id", exact = TRUE)
  gcache <- attr(dt, ".global_state_cache", exact = TRUE)
  out <- data.table::setDT(as.list(dt))
  if (!is.null(row_id)) data.table::setattr(out, ".row_id", row_id)
  if (!is.null(gcache)) data.table::setattr(out, ".global_state_cache", gcache)
  out
}

#' Extract Best Iteration / Epoch from Evaluator Result
#'
#' @param res_model A list returned by an evaluator training function.
#' @return A positive integer iteration count, or NULL if unavailable.
#' @keywords internal
#' @noRd
extract_best_iteration <- function(res_model) {
  if (is.null(res_model)) return(NULL)
  iter_val <- if (!is.null(res_model$best_iteration)) {
    res_model$best_iteration
  } else if (!is.null(res_model$best_epoch)) {
    res_model$best_epoch
  } else if (!is.null(res_model$model$best_iteration)) {
    res_model$model$best_iteration
  } else if (!is.null(res_model$model$best_epoch)) {
    res_model$model$best_epoch
  } else if (!is.null(res_model$model$best_iter)) {
    res_model$model$best_iter
  } else {
    NULL
  }
  if (!is.null(iter_val) && is.numeric(iter_val) && is.finite(iter_val) && iter_val > 0) {
    as.integer(round(iter_val))
  } else {
    NULL
  }
}

#' Encode Binary Target Vector to 0L/1L Integers
#'
#' Maps factor, character, logical, or non-0/1 numeric binary targets to 0L/1L.
#'
#' @param y Target vector (factor, character, logical, integer, or numeric).
#' @return An integer or numeric vector encoded as 0 and 1, preserving NAs.
#' @keywords internal
#' @noRd
encode_binary_target <- function(y) {
  if (is.null(y)) return(NULL)
  if (is.factor(y)) {
    as.integer(y) - 1L
  } else if (is.character(y)) {
    as.integer(as.factor(y)) - 1L
  } else if (is.logical(y)) {
    as.integer(y)
  } else if (is.numeric(y) && !all(stats::na.omit(y) %in% c(0, 1))) {
    as.integer(as.factor(y)) - 1L
  } else {
    y
  }
}

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

#' Infer or Validate Modeling Task Type from Target Vector
#'
#' @param y Target vector.
#' @param task Requested task string ("auto", "classification", "multiclass", or "regression").
#' @return Validated task string ("classification", "multiclass", or "regression").
#' @keywords internal
#' @noRd
.infer_task_type <- function(y, task = "auto") {
  task <- match.arg(task, c("auto", "classification", "multiclass", "regression"))
  if (task != "auto") return(task)
  u <- unique(y[!is.na(y)])
  if (length(u) < 2L) {
    stop("Target column must contain at least 2 distinct non-missing values.")
  }
  if (is.factor(y) || is.character(y) || is.logical(y)) {
    return(if (length(u) == 2L) "classification" else "multiclass")
  }
  if (is.numeric(y) && length(u) == 2L && all(sort(as.numeric(u)) == c(0, 1))) {
    return("classification")
  }
  "regression"
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
calculate_headroom <- function(fitness, baseline, task = "classification", metric = "default") {
  if (is.null(fitness) || is.null(baseline)) {
    return(list(improvement = NULL, headroom_closed = NULL, gain = NULL))
  }
  if (length(fitness) == 0L || length(baseline) == 0L) {
    return(list(improvement = numeric(0), headroom_closed = numeric(0), gain = numeric(0)))
  }
  if (length(baseline) == 1L && (!is.finite(baseline) || is.na(baseline))) {
    return(list(improvement = NULL, headroom_closed = NULL, gain = NULL))
  }

  ideal <- metric_ideal_ceiling(metric, task)
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
  if (is.numeric(ind)) {
    pen <- ind
    t_mode <- if (is.character(n_samples)) n_samples else task
    if (identical(t_mode, "regression")) {
      return(if (raw_score <= 0) raw_score * (1 + pen) else raw_score * (1 - pen))
    } else {
      error_gap <- max(0, min(1.0, 1.0 - raw_score))
      return(raw_score - error_gap * pen)
    }
  }

  ind$raw_fitness <- raw_score
  ind$penalty <- 0.0

  if (is.numeric(complexity_penalty) && complexity_penalty > 0 && complexity_mode != "none" && is.finite(raw_score)) {
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

#' Preserve RNG Seed Across Side-Effecting Operations
#'
#' Evaluates expression while strictly guaranteeing that `.Random.seed` is restored
#' to its exact state prior to evaluation.
#'
#' @param expr Expression or zero-argument function to evaluate.
#' @return Result of evaluating `expr`.
#' @keywords internal
#' @noRd
with_preserved_seed <- function(expr) {
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

#' Default Physical Core Thread Count
#'
#' @return Integer scalar representing default threads on host machine.
#' @keywords internal
#' @noRd
default_threads <- function() {
  opt <- getOption("evoFE.threads", NULL)
  if (!is.null(opt) && is.numeric(opt) && opt >= 1) {
    return(as.integer(opt))
  }
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
  unwrapped <- unwrap_evaluator(evaluator)
  if (identical(unwrapped, "realmlp")) {
    args$early_stopping_rounds <- 0L
    args$early_stopping_round <- NULL
    if (!is.null(target_iters) && is.numeric(target_iters) && !is.na(target_iters) && target_iters > 0) {
      args$stop_epoch <- as.integer(target_iters)
    }
    return(args)
  }
  if (identical(unwrapped, "tabm")) {
    args$early_stopping_rounds <- 0L
    args$early_stopping_round <- NULL
    if (!is.null(target_iters) && is.numeric(target_iters) && !is.na(target_iters) && target_iters > 0) {
      t_int <- as.integer(target_iters)
      for (alias in c("epochs", "n_epochs")) {
        if (alias %in% names(args)) args[[alias]] <- t_int
      }
      if (!any(c("epochs", "n_epochs") %in% names(args))) {
        args$epochs <- t_int
      }
    }
    return(args)
  }
  if (unwrapped %in% c("lm", "spline_lm", "rff")) {
    return(args)
  }
  if (is.null(target_iters) || !is.numeric(target_iters) || is.na(target_iters) || target_iters <= 0) {
    return(args)
  }
  target_iters <- as.integer(target_iters)
  if (identical(unwrapped, "ranger")) {
    args$num.trees <- target_iters
    return(args)
  }
  iter_aliases <- c("nrounds", "num_rounds", "n_rounds", "num_round", "nround",
                    "epochs", "n_epochs", "iterations", "n_iterations", "realmlp_epochs",
                    "num.trees")
  for (alias in iter_aliases) {
    if (alias %in% names(args)) {
      args[[alias]] <- target_iters
    }
  }
  if (!any(c("nrounds", "iterations") %in% names(args))) {
    args$nrounds <- target_iters
  }
  args$early_stopping_rounds <- 0L
  args$early_stopping_round <- NULL
  if (identical(unwrapped, "catboost")) {
    args$od_wait <- NULL
    args$od_type <- NULL
    args$early_stop_round <- NULL
    args$early_stopping <- NULL
  }
  args
}

#' Check Whether an Evaluator Should Scale Iterations With Dataset Size
#'
#' @param evaluator Evaluator name or object.
#' @return Logical scalar TRUE when validation iterations should scale with data size.
#' @keywords internal
#' @noRd
.evaluator_scales_iterations <- function(evaluator) {
  if (is.list(evaluator)) {
    if (!is.null(evaluator$traits) && !is.null(evaluator$traits$scale_iterations_with_data)) {
      return(isTRUE(evaluator$traits$scale_iterations_with_data))
    }
    if (!is.null(evaluator$base_evaluator)) {
      return(.evaluator_scales_iterations(evaluator$base_evaluator))
    }
    return(FALSE)
  }
  if (!is.character(evaluator) || length(evaluator) != 1L || is.na(evaluator)) {
    return(FALSE)
  }
  base_name <- unwrap_evaluator(evaluator)
  if (base_name %in% c("lightgbm_rf", "ranger", "tabm", "rff", "spline_lm", "realmlp", "lm")) {
    return(FALSE)
  }
  if (exists("has_evaluator", mode = "function") && has_evaluator(base_name)) {
    ev <- get_evaluator(base_name)
    if (!is.null(ev$traits) && !is.null(ev$traits$scale_iterations_with_data)) {
      return(isTRUE(ev$traits$scale_iterations_with_data))
    }
  }
  is_tree_evaluator(evaluator)
}

#' Check if Learner-Level Verbosity is Requested
#'
#' @keywords internal
#' @noRd
is_learner_verbose <- function(verbose = NULL) {
  opt_val <- getOption("evoFE.verbose", 0)
  (is.numeric(verbose) && verbose >= 3) ||
    (is.numeric(opt_val) && opt_val >= 3)
}

#' Rank Transform Predictions to Uniform [0, 1] Quantiles
#'
#' Maps predictions to [0, 1] percentile ranks. For matrices (multiclass),
#' ranks are computed column-wise. Ties are handled by average rank.
#'
#' @param p Numeric vector or matrix of predictions.
#' @return Prediction object with the same shape mapped to [0, 1] ranks.
#' @keywords internal
#' @noRd
rank_transform_predictions <- function(p) {
  if (is.null(p)) return(NULL)
  if (is.matrix(p)) {
    apply(p, 2, function(col) {
      n <- length(col)
      if (n <= 1L) return(rep(0.5, n))
      r <- rank(col, ties.method = "average", na.last = "keep")
      (r - 0.5) / n
    })
  } else {
    n <- length(p)
    if (n <= 1L) return(rep(0.5, n))
    r <- rank(p, ties.method = "average", na.last = "keep")
    (r - 0.5) / n
  }
}

#' Sample a Single Element Without R's Length-1 Numeric Trap
#'
#' @param x Atomic vector or list of candidate elements.
#' @return A single sampled element with identical RNG consumption to `sample(x, 1)` when `length(x) > 1`.
#' @keywords internal
#' @noRd
.sample_one <- function(x) {
  n <- length(x)
  if (n == 0L) return(x)
  idx <- sample.int(n, 1L)
  if (is.list(x)) x[[idx]] else x[idx]
}

#' Row-Subset a Sanitized Feature Matrix While Preserving Metadata
#'
#' Standard R matrix indexing strips non-dim attributes such as \code{"sanitized"}
#' and \code{"cat_schema"}. This helper preserves them across row subsets.
#'
#' @param mat A feature matrix (or NULL).
#' @param idx Integer or logical row index vector.
#' @return The row-subsetted matrix with \code{"sanitized"} and \code{"cat_schema"} preserved.
#' @keywords internal
#' @noRd
.slice_feature_matrix <- function(mat, idx) {
  if (is.null(mat)) return(NULL)
  san <- attr(mat, "sanitized", exact = TRUE)
  cs <- attr(mat, "cat_schema", exact = TRUE)
  sub <- mat[idx, , drop = FALSE]
  if (!is.null(san)) attr(sub, "sanitized") <- san
  if (!is.null(cs)) attr(sub, "cat_schema") <- cs
  sub
}

#' Slice a Vector or Matrix Offset Across Row Indices
#'
#' @param offset Numeric vector, matrix, or \code{NULL}.
#' @param idx Integer or logical row index vector.
#' @return Row-subsetted offset preserving matrix shape and column names, or \code{NULL}.
#' @keywords internal
#' @noRd
.slice_offset_arg <- function(offset, idx) {
  if (is.null(offset)) return(NULL)
  if (is.matrix(offset) || is.data.frame(offset)) {
    return(offset[idx, , drop = FALSE])
  }
  offset[idx]
}

.OFFSET_TRAIN_KEYS <- c("init_score", "base_margin", "baseline", "offset", "train_offset")
.OFFSET_VAL_KEYS   <- c("val_init_score", "val_base_margin", "val_baseline", "val_offset", "offset_val")

#' Extract and Slice Train/Val Offset Pair from Caller Extra Arguments
#'
#' @keywords internal
#' @noRd
.extract_sliced_offset_pair <- function(extra_args, tr_idx, va_idx, n_total) {
  if (is.null(extra_args) || length(extra_args) == 0L || is.null(n_total)) return(NULL)
  matched_key <- NULL
  full_off <- NULL
  for (k in .OFFSET_TRAIN_KEYS) {
    val <- extra_args[[k]]
    if (!is.null(val)) {
      matched_key <- k
      full_off <- val
      break
    }
  }
  if (is.null(full_off) || NROW(full_off) != as.integer(n_total)) return(NULL)
  val_key <- switch(matched_key,
    init_score = "val_init_score",
    base_margin = "val_base_margin",
    baseline = "val_baseline",
    "val_offset"
  )
  list(
    train_key = matched_key,
    val_key = val_key,
    train = if (!is.null(tr_idx)) .slice_offset_arg(full_off, tr_idx) else NULL,
    val = if (!is.null(va_idx)) .slice_offset_arg(full_off, va_idx) else NULL
  )
}

#' Apply a Pre-Sliced Train/Val Offset Pair to Argument List
#'
#' @keywords internal
#' @noRd
.apply_offset_pair <- function(args, offset_pair) {
  if (is.null(offset_pair)) return(args)
  for (k in c(.OFFSET_TRAIN_KEYS, .OFFSET_VAL_KEYS)) {
    args[[k]] <- NULL
  }
  if (!is.null(offset_pair$train)) args[[offset_pair$train_key]] <- offset_pair$train
  if (!is.null(offset_pair$val)) args[[offset_pair$val_key]] <- offset_pair$val
  args
}

#' Slice Any Full-Dataset Offset in Extra Arguments Across Train/Val Row Indices
#'
#' @keywords internal
#' @noRd
.slice_offset_extra_args <- function(extra_args, tr_idx, va_idx, n_total) {
  pair <- .extract_sliced_offset_pair(extra_args, tr_idx, va_idx, n_total)
  if (is.null(pair)) return(extra_args)
  .apply_offset_pair(extra_args, pair)
}

#' Resolve and Normalize Offset / Margin / Baseline Arguments
#'
#' Consolidates \code{init_score}, \code{base_margin}, \code{baseline}, and
#' \code{offset} aliases for training, validation, and prediction inputs.
#'
#' @param extra_args Named list of caller arguments.
#' @param n_train Number of training rows (or \code{NULL} during predict-only resolution).
#' @param n_val Optional number of validation rows.
#' @param n_new Optional number of prediction rows.
#' @param task Task type: \code{"regression"}, \code{"classification"}, or \code{"multiclass"}.
#' @param num_class Optional number of classes for multiclass.
#' @param target_levels Optional character vector of class names (used to set multiclass column names).
#' @return A list with elements \code{train}, \code{val}, and \code{new} (each \code{NULL} if absent).
#' @keywords internal
#' @noRd
.resolve_offset_args <- function(extra_args, n_train = NULL, n_val = NULL,
                                 n_new = NULL, task = "regression",
                                 num_class = NULL, target_levels = NULL) {
  pick_first <- function(keys) {
    for (k in keys) {
      if (!is.null(extra_args[[k]])) return(extra_args[[k]])
    }
    NULL
  }

  normalize_one <- function(off, n_rows, label) {
    if (is.null(off) || is.null(n_rows)) return(NULL)
    n_rows <- as.integer(n_rows)
    if (identical(task, "multiclass")) {
      k <- if (!is.null(num_class)) {
        as.integer(num_class)
      } else if (!is.null(target_levels)) {
        length(target_levels)
      } else if (is.matrix(off)) {
        ncol(off)
      } else {
        stop(sprintf("Multiclass '%s' offset requires 'num_class' or a matrix.", label))
      }
      if (is.matrix(off)) {
        if (nrow(off) != n_rows || ncol(off) != k) {
          stop(sprintf("Multiclass '%s' offset matrix must be %d x %d (got %d x %d).",
                       label, n_rows, k, nrow(off), ncol(off)))
        }
        storage.mode(off) <- "double"
      } else {
        if (length(off) != n_rows * k) {
          stop(sprintf("Multiclass '%s' offset vector must have length %d (got %d).",
                       label, n_rows * k, length(off)))
        }
        off <- matrix(as.numeric(off), nrow = n_rows, ncol = k)
      }
      if (!is.null(target_levels) && length(target_levels) == k && is.null(colnames(off))) {
        colnames(off) <- as.character(target_levels)
      }
      return(off)
    }

    if (is.matrix(off)) {
      if (nrow(off) != n_rows || ncol(off) != 1L) {
        stop(sprintf("'%s' offset matrix must be %d x 1 (got %d x %d).",
                     label, n_rows, nrow(off), ncol(off)))
      }
      return(as.numeric(off[, 1L]))
    }
    if (length(off) != n_rows) {
      stop(sprintf("'%s' offset length (%d) must match row count (%d).",
                   label, length(off), n_rows))
    }
    as.numeric(off)
  }

  raw_tr <- pick_first(.OFFSET_TRAIN_KEYS)
  raw_va <- pick_first(.OFFSET_VAL_KEYS)
  raw_nw <- pick_first(c("newoffset", "new_base_margin", "init_score", "base_margin", "baseline", "offset"))

  list(
    train = normalize_one(raw_tr, n_train, "train"),
    val   = normalize_one(raw_va, n_val, "val"),
    new   = if (!is.null(n_new)) normalize_one(raw_nw, n_new, "new") else raw_nw
  )
}

#' @keywords internal
#' @noRd
.parse_bool_flag <- function(x, default = FALSE) {
  if (is.null(x) || length(x) == 0L) return(default)
  v <- x[[1L]]
  if (is.logical(v)) return(!is.na(v) && v)
  if (is.numeric(v)) return(is.finite(v) && v != 0)
  if (is.character(v)) {
    s <- tolower(trimws(v))
    if (s %in% c("true", "t", "1", "yes")) return(TRUE)
    if (s %in% c("false", "f", "0", "no")) return(FALSE)
  }
  default
}

#' Resolve Valid 1-Based Categorical Column Indices and Cardinalities
#'
#' @keywords internal
#' @noRd
.resolve_cat_cols <- function(x, extra_args = list(),
                              cat_schema = attr(x, "cat_schema", exact = TRUE)) {
  p <- ncol(x)
  cs <- if (!is.null(cat_schema)) cat_schema else extra_args$cat_schema
  raw_cols <- if (!is.null(cs$cols)) cs$cols else extra_args$cat_cols
  if (is.null(raw_cols) || length(raw_cols) == 0L || p == 0L) {
    return(list(cols = integer(0L), cardinalities = integer(0L), schema = cs))
  }
  raw_cols <- as.integer(raw_cols)
  keep <- is.finite(raw_cols) & raw_cols >= 1L & raw_cols <= p
  cols <- raw_cols[keep]
  if (length(cols) == 0L) {
    return(list(cols = integer(0L), cardinalities = integer(0L), schema = cs))
  }
  ord <- !duplicated(cols)
  cols <- cols[ord]

  cards <- integer(length(cols))
  if (!is.null(cs$cardinalities) && length(cs$cardinalities) == length(raw_cols)) {
    cards <- as.integer(cs$cardinalities[keep][ord])
  } else if (!is.null(cs$levels) && length(cs$levels) >= length(cols)) {
    cards <- as.integer(lengths(cs$levels)[seq_along(cols)])
  } else if (!is.null(extra_args$cat_cardinalities) &&
             length(extra_args$cat_cardinalities) == length(raw_cols)) {
    cards <- as.integer(extra_args$cat_cardinalities[keep][ord])
  } else {
    for (i in seq_along(cols)) {
      mx <- suppressWarnings(max(x[, cols[i]], na.rm = TRUE))
      cards[i] <- if (is.finite(mx) && mx >= 1) as.integer(ceiling(mx)) else 1L
    }
  }
  cards <- pmax(1L, cards)
  list(cols = cols, cardinalities = cards, schema = cs)
}

#' Single-Pass Column Mean Imputation and Standard Deviations
#'
#' @keywords internal
#' @noRd
.impute_and_col_stats <- function(mat, means = NULL, compute_sd = TRUE, threads = 1L) {
  fn <- get0("rcpp_impute_and_col_stats", mode = "function")
  if (!is.null(fn)) {
    return(fn(mat, means = means, compute_sd = isTRUE(compute_sd), threads = as.integer(threads)))
  }

  if (!is.matrix(mat) || storage.mode(mat) != "double") {
    mat <- as.matrix(mat)
    storage.mode(mat) <- "double"
  }
  n <- nrow(mat)
  p <- ncol(mat)
  col_means <- if (!is.null(means)) as.numeric(means) else numeric(p)
  col_sds <- if (isTRUE(compute_sd)) rep(1.0, p) else numeric(0L)
  if (n == 0L || p == 0L) {
    return(list(mat = mat, col_means = col_means, col_sds = col_sds))
  }

  rng <- suppressWarnings(range(mat))
  has_nonfinite <- !all(is.finite(rng))
  out_mat <- mat

  if (!has_nonfinite) {
    if (is.null(means)) col_means <- base::colMeans(out_mat)
    if (isTRUE(compute_sd) && n > 1L) {
      for (j in seq_len(p)) {
        s <- stats::sd(out_mat[, j])
        col_sds[j] <- if (is.finite(s) && s > 0) s else 1.0
      }
    }
    return(list(mat = out_mat, col_means = col_means, col_sds = col_sds))
  }

  for (j in seq_len(p)) {
    col_j <- out_mat[, j]
    bad <- !is.finite(col_j)
    m_j <- if (!is.null(means)) {
      col_means[j]
    } else {
      good <- col_j[!bad]
      if (length(good) > 0L) mean(good) else 0.0
    }
    if (!is.finite(m_j)) m_j <- 0.0
    if (is.null(means)) col_means[j] <- m_j
    if (any(bad)) {
      out_mat[bad, j] <- m_j
    }
    if (isTRUE(compute_sd) && n > 1L) {
      s <- stats::sd(out_mat[, j])
      col_sds[j] <- if (is.finite(s) && s > 0) s else 1.0
    }
  }
  list(mat = out_mat, col_means = col_means, col_sds = col_sds)
}

#' Convert a Dense Matrix to Sparse dgCMatrix
#'
#' @keywords internal
#' @noRd
.to_dgc <- function(mat) {
  d <- dim(mat)
  nz <- which(mat != 0, arr.ind = TRUE)
  Matrix::sparseMatrix(
    i = nz[, 1L],
    j = nz[, 2L],
    x = as.numeric(mat[nz]),
    dims = d
  )
}

#' Build Sparse B-Spline + Categorical OHE Design Matrix
#'
#' @keywords internal
#' @noRd
.build_sparse_spline_design <- function(x, cat_schema = NULL, state = NULL,
                                        n_knots = 5L, degree = 2L,
                                        include_linear = TRUE, max_cat_levels = 64L) {
  if (!requireNamespace("Matrix", quietly = TRUE) ||
      !requireNamespace("splines", quietly = TRUE)) {
    stop("Packages 'Matrix' and 'splines' are required for sparse spline designs.")
  }
  n <- nrow(x)
  p <- ncol(x)

  if (is.null(state)) {
    n_knots <- max(0L, as.integer(n_knots))
    degree <- max(1L, min(3L, as.integer(degree)))
    include_linear <- .parse_bool_flag(include_linear, default = TRUE)
    max_cat_levels <- max(1L, as.integer(max_cat_levels))

    cat_info <- .resolve_cat_cols(x, cat_schema = cat_schema)
    cat_set <- cat_info$cols
    num_cols <- setdiff(seq_len(p), cat_set)
    in_names <- colnames(x)
    if (is.null(in_names)) in_names <- paste0("V", seq_len(p))

    sub_idx <- if (n > 20000L) {
      unique(as.integer(round(seq.int(1, n, length.out = 20000L))))
    } else {
      seq_len(n)
    }

    blocks <- list()
    col_group <- integer(0L)
    out_names <- character(0L)
    num_specs <- list()

    for (j in num_cols) {
      v <- as.numeric(x[, j])
      ok <- is.finite(v)
      if (!any(ok)) next
      m <- mean(v[ok])
      if (!all(ok)) v[!ok] <- m
      s <- if (n > 1L) stats::sd(v) else 0.0
      if (!is.finite(s) || s <= 1e-12) next
      vs <- (v - m) / s
      bmin <- min(vs)
      bmax <- max(vs)
      if (!is.finite(bmin) || !is.finite(bmax) || bmax <= bmin) next

      vs_sub <- vs[sub_idx]
      u_n <- length(unique(vs_sub))
      emit_linear <- isTRUE(include_linear) || u_n < 3L
      knots_vec <- NULL
      deg_eff <- 0L
      has_spline <- FALSE

      if (u_n >= 3L && (n_knots > 0L || !isTRUE(include_linear))) {
        ik <- if (n_knots > 0L) {
          probs <- seq(0, 1, length.out = n_knots + 2L)[-c(1L, n_knots + 2L)]
          sort(unique(stats::quantile(vs_sub, probs = probs, names = FALSE, type = 7L)))
        } else {
          numeric(0L)
        }
        ik <- ik[ik > bmin + 1e-12 & ik < bmax - 1e-12]
        if (length(ik) > n_knots && n_knots > 0L) ik <- ik[seq_len(n_knots)]
        deg_eff <- max(1L, min(degree, u_n - 1L))
        B <- tryCatch(
          splines::bs(vs, knots = ik, degree = deg_eff,
                      Boundary.knots = c(bmin, bmax), intercept = FALSE),
          error = function(e) NULL
        )
        if (!is.null(B) && ncol(B) > 0L) {
          has_spline <- TRUE
          knots_vec <- ik
          B_mat <- unclass(B)
          dimnames(B_mat) <- NULL
          sp_B <- .to_dgc(B_mat)
        }
      }

      if (emit_linear) {
        nz_v <- which(vs != 0)
        sp_lin <- Matrix::sparseMatrix(i = nz_v, j = rep.int(1L, length(nz_v)), x = vs[nz_v], dims = c(n, 1L))
        blocks[[length(blocks) + 1L]] <- sp_lin
        col_group <- c(col_group, j)
        out_names <- c(out_names, paste0(in_names[j], "__lin"))
      }
      if (has_spline) {
        blocks[[length(blocks) + 1L]] <- sp_B
        col_group <- c(col_group, rep(j, ncol(sp_B)))
        out_names <- c(out_names, paste0(in_names[j], "__sp", seq_len(ncol(sp_B))))
      }

      num_specs[[length(num_specs) + 1L]] <- list(
        orig_col = j,
        mean = m,
        sd = s,
        bmin = bmin,
        bmax = bmax,
        emit_linear = emit_linear,
        has_spline = has_spline,
        knots = knots_vec,
        degree = deg_eff
      )
    }

    cat_specs <- list()
    for (idx_c in seq_along(cat_set)) {
      j <- cat_set[idx_c]
      K_keep <- min(cat_info$cardinalities[idx_c], max_cat_levels)
      code <- suppressWarnings(as.integer(floor(x[, j])))
      bad <- is.na(code) | code < 1L | code > K_keep
      code[bad] <- K_keep + 1L
      sp_cat <- Matrix::sparseMatrix(i = seq_len(n), j = code, x = 1.0,
                                     dims = c(n, K_keep + 1L))
      blocks[[length(blocks) + 1L]] <- sp_cat
      col_group <- c(col_group, rep(j, K_keep + 1L))
      out_names <- c(out_names, paste0(in_names[j], "__cat", seq_len(K_keep + 1L)))
      cat_specs[[length(cat_specs) + 1L]] <- list(orig_col = j, K_keep = K_keep)
    }

    design <- if (length(blocks) > 0L) {
      do.call(cbind, blocks)
    } else {
      Matrix::sparseMatrix(i = integer(0L), j = integer(0L), x = numeric(0L), dims = c(n, 0L))
    }
    if (ncol(design) > 0L) colnames(design) <- out_names

    state_out <- list(
      p_in = p,
      in_names = in_names,
      num_specs = num_specs,
      cat_specs = cat_specs,
      col_group = col_group,
      out_names = out_names
    )
    return(list(matrix = design, state = state_out))
  }

  # Predict-time alignment reusing exact saved knots, boundary clamps, and OHE widths
  blocks <- list()
  for (sp in state$num_specs) {
    v <- as.numeric(x[, sp$orig_col])
    bad <- !is.finite(v)
    if (any(bad)) v[bad] <- sp$mean
    vs <- (v - sp$mean) / sp$sd

    if (isTRUE(sp$emit_linear)) {
      nz_v <- which(vs != 0)
      blocks[[length(blocks) + 1L]] <- Matrix::sparseMatrix(
        i = nz_v, j = rep.int(1L, length(nz_v)), x = vs[nz_v], dims = c(n, 1L)
      )
    }
    if (isTRUE(sp$has_spline)) {
      vs_clamp <- pmin(pmax(vs, sp$bmin), sp$bmax)
      B <- splines::bs(vs_clamp, knots = sp$knots, degree = sp$degree,
                       Boundary.knots = c(sp$bmin, sp$bmax), intercept = FALSE)
      B_mat <- unclass(B)
      dimnames(B_mat) <- NULL
      blocks[[length(blocks) + 1L]] <- .to_dgc(B_mat)
    }
  }
  for (cs in state$cat_specs) {
    K_keep <- cs$K_keep
    code <- suppressWarnings(as.integer(floor(x[, cs$orig_col])))
    bad <- is.na(code) | code < 1L | code > K_keep
    code[bad] <- K_keep + 1L
    blocks[[length(blocks) + 1L]] <- Matrix::sparseMatrix(
      i = seq_len(n), j = code, x = 1.0, dims = c(n, K_keep + 1L)
    )
  }

  design <- if (length(blocks) > 0L) {
    do.call(cbind, blocks)
  } else {
    Matrix::sparseMatrix(i = integer(0L), j = integer(0L), x = numeric(0L), dims = c(n, 0L))
  }
  if (ncol(design) > 0L && length(state$out_names) == ncol(design)) {
    colnames(design) <- state$out_names
  }
  design
}
