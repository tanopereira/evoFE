#' Create a transformer definition
#'
#' @param name Transformer name
#' @param type Type: "unary", "binary", "supervised_unary"
#' @param input_type Type of input: "numeric" or "categorical"
#' @param output_type Type of output: "numeric" or "categorical"
#' @param fit_func function(data, gene, target_col = NULL) returning state
#' @param apply_func function(data, gene, state = NULL) returning new column vector
#' @param name_generator function(gene) returning output column name
#' @param allow_replace Logical. Whether column sampling allows replacement.
#' @return An \code{evo_transformer} S3 object:
#'   a list with elements
#'   \code{name}, \code{type}, \code{input_type}, \code{output_type},
#'   \code{fit_func}, \code{apply_func}, \code{name_generator}, and
#'   \code{allow_replace}.
#' @import data.table
#' @examples
#' # Define a transformer that adds a constant value of 10 to a variable
#' add_ten_trans <- create_transformer(
#'   name = "add_ten",
#'   type = "unary",
#'   input_type = "numeric",
#'   apply_func = function(data, gene, state = NULL) {
#'     data[[gene$input_cols[1]]] + 10
#'   },
#'   name_generator = function(gene) paste0("add10_", gene$input_cols[1])
#' )
#' print(add_ten_trans)
#' @export
create_transformer <- function(name, type, input_type = "numeric", output_type = "numeric", fit_func = NULL, apply_func, name_generator, allow_replace = FALSE) {
  structure(
    list(
      name = name,
      type = type,
      input_type = input_type,
      output_type = output_type,
      fit_func = fit_func,
      apply_func = apply_func,
      name_generator = name_generator,
      allow_replace = allow_replace
    ),
    class = "evo_transformer"
  )
}

#' Check whether a transformer is supervised
#'
#' @param gene A gene list, transformer name character string, or \code{evo_transformer}.
#' @param t_def Optional \code{evo_transformer} definition.
#' @return Logical indicating whether the transformer requires supervision (uses target).
#' @export
is_supervised_transformer <- function(gene, t_def = NULL) {
  if (is.null(t_def)) {
    t_name <- if (inherits(gene, "evo_transformer")) {
      gene$name
    } else if (is.character(gene)) {
      gene
    } else if (is.list(gene) && !is.null(gene$transformer_name)) {
      gene$transformer_name
    } else {
      NULL
    }
    if (is.null(t_name)) return(FALSE)
    t_def <- evo_transformers[[t_name]]
  }
  if (is.null(t_def)) return(FALSE)
  if (!is.null(t_def$type) && grepl("^supervised", t_def$type)) return(TRUE)
  t_name <- if (is.character(gene)) gene else if (is.list(gene) && !is.null(gene$transformer_name)) gene$transformer_name else t_def$name
  t_name %in% c(
    "target_encode", "pooled_target_encode", "target_encode_multiclass",
    "woe_encode", "target_quantile_encode", "cat_interaction_target_encode",
    "target_sd_encode", "knn_target_encode", "ridge_oof_encode",
    "supervised_bgpca", "supervised_mca", "supervised_famd"
  )
}
#' Built-in feature transformers
#'
#' An environment containing all built-in transformer definitions available
#' for the evolutionary feature-engineering search.  Every entry is an
#' \code{evo_transformer} object produced by \code{\link{create_transformer}}.
#'
#' \strong{Arithmetic (numeric -> numeric)}
#' \describe{
#'   \item{\code{log}}{Safe natural logarithm: \code{log1p(|x|)}.}
#'   \item{\code{sqrt}}{Safe square root: \code{sqrt(|x|)}.}
#'   \item{\code{reciprocal}}{Reciprocal: \code{1/x} (0 where \code{x == 0}).}
#'   \item{\code{power}}{Signed exponentiation: \code{sign(x) * |x|^p} where
#'     \code{p} is sampled from \{0.5, 1/3, 2, 3\}.}
#'   \item{\code{displaced_log}}{Displaced log: \code{log1p(|x + displacement|)} where
#'     \code{displacement} is sampled from \code{[10, 1000]}.}
#'   \item{\code{add}}{Element-wise sum of 2+ numeric columns.}
#'   \item{\code{subtract}}{Element-wise difference of two numeric columns.}
#'   \item{\code{multiply}}{Element-wise product of 2+ numeric columns.}
#'   \item{\code{divide}}{Element-wise ratio (0 where denominator is 0).}
#'   \item{\code{normalized_difference}}{\code{(a - b) / (|a| + |b| + 1e-6)}.}
#'   \item{\code{log_ratio}}{\code{log1p(|a|) - log1p(|b|)}.}
#'   \item{\code{fourier_basis}}{Multi-component Fourier harmonic basis with data-adaptive robust normalization: paired sine and cosine dyadic harmonics (\code{comp_idx} 1--4, 1--6, or 1--8; orders 1, 2, 4, 8) across octave frequency scales and non-redundant phases.}
#'   \item{\code{row_min}}{Row-wise minimum across 2+ numeric columns (dealbreaker metric).}
#'   \item{\code{row_max}}{Row-wise maximum across 2+ numeric columns.}
#'   \item{\code{row_sd}}{Row-wise standard deviation across 2+ numeric columns.}
#'   \item{\code{row_na_count}}{Row-wise count of missing or non-finite values across 2+ numeric columns.}
#'   \item{\code{relative_rating}}{Difference between primary numeric column and mean of other columns (rater debiasing).}
#'   \item{\code{geometric_mean}}{Geometric mean across 2+ numeric columns with positive clamping.}
#'   \item{\code{harmonic_mean}}{Harmonic mean across 2+ numeric columns with positive clamping (soft bottleneck / rate mean).}
#'   \item{\code{pythagorean_imbalance}}{Difference between arithmetic mean and harmonic mean (\code{AM - HM}) measuring rating imbalance.}
#' }

#'
#' \strong{Rank / distribution (numeric -> numeric, stateful)}
#' \describe{
#'   \item{\code{rank_transform}}{ECDF-based percentile rank mapped to
#'     \eqn{[0, 1]}.  Fit on training data; robust to outliers.}
#'   \item{\code{robust_scale}}{Robust median/IQR scaling: \code{(x - median) / IQR}. Fit on training data.}
#'   \item{\code{smooth_clip}}{Smooth outlier dampening via soft hyperbolic tangent roll-off beyond percentiles. Fit on training data.}
#'   \item{\code{yeo_johnson}}{Yeo-Johnson power transformation (\eqn{\lambda \in [-2, 2]} fit via maximum likelihood on training data) with zero-mean unit-variance standardization; supports positive, zero, and negative inputs.}
#'   \item{\code{box_cox}}{Shifted Box-Cox power transformation (\eqn{\lambda \in [-2, 2]} and positivity shift fit via maximum likelihood on training data) with zero-mean unit-variance standardization.}
#' }
#'
#' \strong{Group-by aggregations (mixed cat x num -> numeric, stateful)}
#' \describe{
#'   \item{\code{groupby_mean}}{Per-group mean.}
#'   \item{\code{groupby_sd}}{Per-group standard deviation.}
#'   \item{\code{groupby_max}}{Per-group maximum.}
#'   \item{\code{groupby_min}}{Per-group minimum.}
#'   \item{\code{groupby_ratio}}{\code{value / group_mean}.}
#'   \item{\code{groupby_zscore}}{\code{(value - group_mean) / group_sd}.}
#'   \item{\code{groupby_median}}{Per-group median (robust to outliers).}
#'   \item{\code{groupby_quantile}}{Per-group Q1 or Q3 (\code{q} sampled from
#'     \{0.25, 0.75\}).}
#'   \item{\code{groupby_signed_log}}{\code{sign(value - group_mean) * log1p(|value - group_mean|)}.}
#'   \item{\code{groupby_diff}}{Centered group residual: \code{value - group_mean}.}
#'   \item{\code{groupby_median_diff}}{Robust centered group residual: \code{value - group_median}.}
#'   \item{\code{groupby_range_pos}}{Within-group min-max position: \code{(value - group_min) / (group_max - group_min)} in \eqn{[0, 1]}.}
#'   \item{\code{groupby_rank}}{Within-group empirical percentile rank in \eqn{[0, 1]}.}
#'   \item{\code{groupby_nunique}}{Per-group count of distinct levels of a second categorical feature (\code{cat x cat -> numeric}).}
#' }
#'
#' \strong{Supervised categorical encodings (categorical -> numeric, stateful)}
#' \describe{
#'   \item{\code{target_encode}}{Smoothed mean-target encoding for binary /
#'     regression tasks.}
#'   \item{\code{pooled_target_encode}}{Empirical Bayes pooled target encoding
#'     for binary / regression tasks using dynamic shrinkage based on target variance.}
#'   \item{\code{target_encode_multiclass}}{Class-wise smoothed target encoding
#'     for multiclass tasks.}
#'   \item{\code{target_quantile_encode}}{Category target encoding using smoothed target
#'     quantiles (\code{q} sampled from \{0.25, 0.50, 0.75\}).}
#'   \item{\code{target_sd_encode}}{Smoothed within-category target standard deviation encoding with Leave-One-Out training jackknife.}
#'   \item{\code{knn_target_encode}}{Out-of-fold k-nearest-neighbors target encoding across 1--3 standardized numeric features (\code{k} sampled from \{3, 5, 10, 25\}).}
#'   \item{\code{ridge_oof_encode}}{Exact closed-form hat-matrix leave-one-out Ridge target encoding across 2--5 standardized numeric features (\code{lambda} sampled from \{0.1, 1, 10, 100\}, polynomial \code{degree} sampled from \{1, 2\}).}
#'   \item{\code{cat_interaction_target_encode}}{Smoothed mean-target encoding for joint
#'     Cartesian interaction of two categorical columns.}
#'   \item{\code{woe_encode}}{Weight of Evidence encoding for binary
#'     classification: \code{ln(P(event|cat) / P(non-event|cat))} with Laplace
#'     smoothing.  Falls back to 0 for non-binary targets.}
#' }
#'
#' \strong{Unsupervised categorical / text / datetime}
#' \describe{
#'   \item{\code{concat}}{Concatenates 2 or 3 categorical columns row-wise using an underscore separator.}
#'   \item{\code{frequency_encode}}{Count of each category level in training data.}
#'   \item{\code{numeric_freq}}{Count of each unique numerical value in training data (frequency proxy for spiky/clustered numerics).}
#'   \item{\code{one_hot_encode}}{Binary indicator for up to 5 top categories plus
#'     an "other" bucket (\code{comp_idx} 1-6).}
#'   \item{\code{similarity_encode}}{Character 3-gram Jaccard similarity between string
#'     levels and top-K prototype categories (inspired by \pkg{skrub}).}
#'   \item{\code{minhash_encode}}{Fast sub-string MinHash hashing for high-cardinality strings
#'     (inspired by \pkg{skrub}).}
#'   \item{\code{gap_encode}}{Character 3-gram TF-IDF projection via SVD to extract latent
#'     sub-string topics (inspired by \pkg{skrub}).}
#'   \item{\code{quantile_binning}}{Assigns quantile-based bin index (numeric output).}
#'   \item{\code{quantile_binning_cat}}{Same, with categorical output.}
#'   \item{\code{log_binning}}{Log-scale bin index (numeric output).}
#'   \item{\code{log_binning_cat}}{Same, with categorical output.}
#'   \item{\code{round_transform}}{Rounds continuous numeric feature to \code{digits} decimal places (\code{digits} sampled from \{0, 1, 2, 3\}).}
#'   \item{\code{decimal_digit}}{Extracts the \code{digits}-th decimal place digit (0--9) of a continuous numeric feature (\code{digits} sampled from \{1, 2, 3, 4\}).}
#'   \item{\code{frac_part}}{Extracts continuous fractional remainder \code{|x| - floor(|x|)} in \eqn{[0, 1)}, preserving \code{NA}.}
#'   \item{\code{datetime_extract}}{Extracts year, month, day, hour, day-of-week,
#'     or weekend indicator from date/datetime columns.}
#'   \item{\code{datetime_cyclic}}{Sine and cosine cyclic encoding for periodic date/time
#'     components (hour, day of week, month, day of year).}
#'   \item{\code{date_diff}}{Signed difference in days between two datetime columns.}
#' }
#'
#' \strong{Dimensionality reduction (numeric -> numeric, stateful)}
#' \describe{
#'   \item{\code{pca}}{Selected principal component from \code{prcomp}.}
#'   \item{\code{truncated_svd}}{Selected component from truncated SVD.}
#'   \item{\code{random_projection}}{Random unit-vector linear combination.}
#'   \item{\code{umap}}{UMAP projection component (requires \pkg{uwot}).}
#' }
#'
#' \strong{Manifold / graph learning (numeric -> numeric or categorical, stateful)}
#' \describe{
#'   \item{\code{genie}}{Genie hierarchical cluster label (requires
#'     \pkg{genieclust}).}
#'   \item{\code{genie_centroid_dist}}{Distance to each Genie cluster centroid.}
#'   \item{\code{umap_genie}}{Genie cluster label computed on low-dimensional UMAP embedding (requires \pkg{uwot} and \pkg{genieclust}).}
#'   \item{\code{umap_lumbermark}}{Lumbermark cluster label computed on a
#'     low-dimensional UMAP embedding.  Combines the non-linear structure
#'     discovery of UMAP with the minimum-spanning-tree-based hierarchical
#'     clustering of Lumbermark (requires \pkg{uwot} and \pkg{lumbermark}).}
#'   \item{\code{lumbermark}}{Lumbermark hierarchical cluster label (requires
#'     \pkg{lumbermark}).}
#'   \item{\code{lumbermark_centroid_dist}}{Distance to each Lumbermark cluster
#'     centroid.}
#'   \item{\code{mst_score}}{MST-based anomaly score (requires
#'     \pkg{quitefastmst}).}
#'   \item{\code{deadwood}}{Deadwood outlier indicator (requires \pkg{deadwood}).}
#' }
#'
#' @return An \code{environment} containing registered feature transformers.
#' @seealso \code{\link{create_transformer}}, \code{\link{register_transformer}}
#' @export
evo_transformers <- new.env(parent = emptyenv())

#' Register a custom feature transformer
#'
#' Adds a user-defined feature transformer to the available pool for feature evolution.
#'
#' @param name Unique character string naming the transformer.
#' @param transformer An object of class \code{evo_transformer} created
#'   via \code{create_transformer}.
#' @return Invisible \code{transformer}, the registered transformer object.
#' @examples
#' # Create a custom transformer
#' add_ten_trans <- create_transformer(
#'   name = "add_ten",
#'   type = "unary",
#'   input_type = "numeric",
#'   apply_func = function(data, gene, state = NULL) {
#'     data[[gene$input_cols[1]]] + 10
#'   },
#'   name_generator = function(gene) paste0("add10_", gene$input_cols[1])
#' )
#'
#' # Register it
#' register_transformer("add_ten", add_ten_trans)
#'
#' # Verify it is registered
#' exists("add_ten", envir = evo_transformers)
#' @export
register_transformer <- function(name, transformer) {
  if (!inherits(transformer, "evo_transformer")) {
    stop("transformer must be an object of class 'evo_transformer' (created via create_transformer).")
  }
  if (!is.character(name) || length(name) != 1 || is.na(name) || !nzchar(name)) {
    stop("name must be a single non-empty character string.")
  }
  if (name %in% names(evo_transformers)) {
    warning(sprintf("Overwriting existing transformer '%s'.", name))
  }
  evo_transformers[[name]] <- transformer
  invisible(transformer)
}

is_verbose <- function() {
  val <- getOption("evoFE.verbose", 0)
  isTRUE(val) || val >= 2
}

.safe_sd <- function(x, na.rm = TRUE) {
  if (is.null(x)) return(0)
  x_clean <- if (na.rm) x[!is.na(x) & is.finite(x)] else x
  if (length(x_clean) <= 1) return(0)
  s <- suppressWarnings(stats::sd(x_clean, na.rm = na.rm))
  if (is.na(s) || !is.finite(s)) 0 else s
}

# Produce a short, stable column name for a gene: "{prefix}_{6-char hash}".
# The hash covers transformer name + input columns + params, so identical
# genes always get identical names (deduplication) and different genes
# almost certainly get different names (collision probability ~1/16M).
# Full human-readable details are available via gene_to_formula().
.gene_col_name <- function(gene, prefix) {
  h <- substr(
    digest::digest(list(gene$transformer_name, gene$input_cols, gene$params),
                   algo = "xxhash32"),
    1, 6
  )
  paste0(prefix, "_", h)
}

.to_numeric <- function(x) {
  if (is.null(x)) return(numeric(0))
  if (is.numeric(x)) return(as.numeric(x))
  suppressWarnings(as.numeric(as.character(x)))
}

.is_datetime_col <- function(x) {
  if (inherits(x, c("POSIXt", "Date"))) {
    return(TRUE)
  }
  if (is.character(x) || is.factor(x)) {
    vals <- as.character(x)
    vals <- vals[!is.na(vals) & vals != ""]
    if (length(vals) == 0) return(FALSE)
    sample_vals <- if (length(vals) > 100) with_seed(42L, sample(vals, 100)) else vals
    parsed <- tryCatch({
      as.POSIXct(sample_vals, tryFormats = c("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d"))
    }, error = function(e) NULL)
    if (!is.null(parsed) && !all(is.na(parsed))) {
      return(mean(!is.na(parsed)) >= 0.7)
    }
  }
  FALSE
}

# --- CLUSTERING SCAFFOLD HELPERS ---

# Deduplicate rows and optionally downsample to max_clustering_size.
# Returns the processed matrix, or NULL if there are too few rows.
# `min_rows` is checked *after* deduplication AND after downsampling.
.cluster_prep_x <- function(x, min_rows = 6, verbose = FALSE, tag = "Fit") {
  t0 <- Sys.time()
  dt_x <- data.table::as.data.table(x)
  unique_idx <- which(!duplicated(dt_x))
  x_unique <- x[unique_idx, , drop = FALSE]
  if (verbose)
    message(sprintf("  [%s] Dedup: %d unique / %d rows. %.3f s",
                    tag, nrow(x_unique), nrow(x), as.numeric(difftime(Sys.time(), t0, units = "secs"))))

  if (nrow(x_unique) < min_rows) return(NULL)

  max_size <- getOption("evoFE.max_clustering_size", 5000)
  if (length(max_size) > 1) max_size <- max_size[1]
  if (!is.numeric(max_size) || is.null(max_size)) max_size <- 0
  if (max_size > 0 && nrow(x_unique) > max_size) {
    # Preserve the caller's RNG state
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv))
      get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
      if (!is.null(old_seed)) assign(".Random.seed", old_seed, envir = .GlobalEnv)
      else if (exists(".Random.seed", envir = .GlobalEnv)) rm(".Random.seed", envir = .GlobalEnv)
    }, add = TRUE)
    x_unique <- x_unique[sample(seq_len(nrow(x_unique)), max_size), , drop = FALSE]
    if (verbose)
      message(sprintf("  [%s] Downsampled to %d rows. %.3f s",
                      tag, nrow(x_unique), as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }

  if (nrow(x_unique) < min_rows) return(NULL)
  x_unique
}

# Cache-aware KNN apply for clustering transformers.
# `get_preds(nearest_indices, state)` maps neighbour indices -> prediction vector.
# When x_test is identical to the training set, indices are the identity permutation.
.cluster_knn_apply <- function(x_test, state, get_preds, verbose = FALSE, tag = "Apply") {
  if (is.null(state) || !isTRUE(state$valid))
    return(rep(0, nrow(x_test)))

  x_train <- state$x_train

  compute <- function() {
    if (nrow(x_test) == nrow(x_train) && all(x_test == x_train)) {
      if (verbose)
        message(sprintf("  [%s] KNN (Fast Path): using cached training predictions. 0.000 s", tag))
      get_preds(seq_len(nrow(x_train)), state)
    } else {
      t0 <- Sys.time()
      idx <- tryCatch(
        quitefastmst::knn_euclid(x_train, k = 1L, Y = x_test)$nn.index[, 1],
        error = function(e) {
          x_train_mat <- as.matrix(x_train)
          x_test_mat <- as.matrix(x_test)
          if (!is.numeric(x_train_mat)) storage.mode(x_train_mat) <- "numeric"
          if (!is.numeric(x_test_mat)) storage.mode(x_test_mat) <- "numeric"

          if (nrow(x_test_mat) == 0L) {
            return(integer(0))
          }

          train_norms_sq <- rowSums(x_train_mat^2)
          D_approx <- sweep(-2 * tcrossprod(x_test_mat, x_train_mat), 2, train_norms_sq, "+")

          if (nrow(x_test_mat) == 1L) {
            res <- which.min(as.vector(D_approx))
            if (length(res) == 0L) NA_integer_ else res
          } else {
            apply(D_approx, 1, function(x) {
              res <- which.min(x)
              if (length(res) == 0L) NA_integer_ else res
            })
          }
        }
      )
      if (verbose)
        message(sprintf("  [%s] KNN: %d test -> %d train rows. %.3f s",
                        tag, nrow(x_test), nrow(x_train),
                        as.numeric(difftime(Sys.time(), t0, units = "secs"))))
      get_preds(idx, state)
    }
  }

  if (is.null(state$preds_cache)) return(compute())

  key <- digest::digest(x_test, algo = "xxhash64")
  if (exists(key, envir = state$preds_cache))
    return(get(key, envir = state$preds_cache))

  preds <- compute()
  assign(key, preds, envir = state$preds_cache)
  preds
}

#' Return discrete candidate grid for a transformer parameter
#' @keywords internal
#' @noRd
.transformer_discrete_param_space <- function(transformer_name, param_name) {
  switch(param_name,
    k = if (identical(transformer_name, "knn_target_encode")) c(3L, 5L, 10L, 25L) else 2:5,
    digits = if (identical(transformer_name, "round_transform")) c(0L, 1L, 2L, 3L) else c(1L, 2L, 3L, 4L),
    Q = 3:10,
    base = 2:10,
    num_bins = c(4L, 8L, 16L, 32L),
    component = if (identical(transformer_name, "datetime_cyclic")) {
      c("hour_sin", "hour_cos", "wday_sin", "wday_cos", "month_sin", "month_cos", "yday_sin", "yday_cos")
    } else {
      c("year", "month", "day", "hour", "day_of_week", "weekend")
    },
    p = c(0.5, 1 / 3, 2, 3),
    q = if (identical(transformer_name, "target_quantile_encode")) c(0.25, 0.5, 0.75) else c(0.25, 0.75),
    scale = c(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
    phase = c(0, round(pi / 6, 4), round(pi / 4, 4), round(pi / 3, 4)),
    low_pct = c(0.01, 0.02, 0.05),
    high_pct = c(0.95, 0.98, 0.99),
    lambda = c(0.1, 1.0, 10.0, 100.0),
    degree = c(1L, 2L),
    NULL
  )
}

#' Sample initial parameters for a transformer
#' @keywords internal
#' @noRd
sample_transformer_params <- function(transformer_name, input_cols = character(0)) {
  params <- list()
  if (transformer_name %in% c("pca", "truncated_svd", "umap")) {
    C <- max(2L, as.integer(round(log2(length(input_cols)))))
    params$comp_idx <- .sample_one(seq_len(C))
    if (transformer_name == "umap") {
      params$n_neighbors <- max(2L, stats::rpois(1, 15))
      params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    }
  } else if (transformer_name %in% c("similarity_encode", "mca", "famd", "between_group_pca", "supervised_bgpca", "supervised_mca", "supervised_famd")) {
    params$comp_idx <- .sample_one(1:5)
  } else if (transformer_name == "minhash_encode") {
    params$comp_idx <- .sample_one(1:8)
  } else if (transformer_name == "gap_encode") {
    params$comp_idx <- .sample_one(1:4)
  } else if (transformer_name %in% c("datetime_cyclic", "datetime_extract")) {
    params$component <- .sample_one(.transformer_discrete_param_space(transformer_name, "component"))
  } else if (transformer_name %in% c("target_quantile_encode", "groupby_quantile")) {
    params$q <- .sample_one(.transformer_discrete_param_space(transformer_name, "q"))
  } else if (transformer_name == "feature_hash") {
    params$num_bins <- .sample_one(.transformer_discrete_param_space(transformer_name, "num_bins"))
    params$comp_idx <- .sample_one(seq_len(params$num_bins))
  } else if (transformer_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
    params$k <- .sample_one(.transformer_discrete_param_space(transformer_name, "k"))
    params$comp_idx <- .sample_one(seq_len(params$k))
    if (transformer_name == "genie_centroid_dist") {
      params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
    }
  } else if (transformer_name == "one_hot_encode") {
    params$comp_idx <- .sample_one(1:6)
  } else if (transformer_name == "genie") {
    params$k <- .sample_one(.transformer_discrete_param_space(transformer_name, "k"))
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "umap_genie") {
    params$n_neighbors <- max(2L, stats::rpois(1, 15))
    params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    params$k <- .sample_one(.transformer_discrete_param_space(transformer_name, "k"))
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "umap_lumbermark") {
    params$n_neighbors <- max(2L, stats::rpois(1, 15))
    params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    params$k <- .sample_one(.transformer_discrete_param_space(transformer_name, "k"))
  } else if (transformer_name %in% c("lumbermark", "knn_target_encode")) {
    params$k <- .sample_one(.transformer_discrete_param_space(transformer_name, "k"))
  } else if (transformer_name %in% c("quantile_binning", "quantile_binning_cat")) {
    params$Q <- .sample_one(.transformer_discrete_param_space(transformer_name, "Q"))
  } else if (transformer_name %in% c("log_binning", "log_binning_cat")) {
    params$base <- .sample_one(.transformer_discrete_param_space(transformer_name, "base"))
  } else if (transformer_name %in% c("round_transform", "decimal_digit")) {
    params$digits <- .sample_one(.transformer_discrete_param_space(transformer_name, "digits"))
  } else if (transformer_name == "target_encode_multiclass") {
    params$comp_idx <- .sample_one(1:5)
  } else if (transformer_name == "power") {
    params$p <- .sample_one(.transformer_discrete_param_space(transformer_name, "p"))
  } else if (transformer_name == "displaced_log") {
    params$displacement <- round(stats::runif(1, 10, 1000), 2)
  } else if (transformer_name == "fourier_basis") {
    params$comp_idx <- .sample_one(1:8)
    params$scale <- .sample_one(.transformer_discrete_param_space(transformer_name, "scale"))
    params$phase <- .sample_one(.transformer_discrete_param_space(transformer_name, "phase"))
  } else if (transformer_name == "smooth_clip") {
    params$low_pct <- .sample_one(.transformer_discrete_param_space(transformer_name, "low_pct"))
    params$high_pct <- .sample_one(.transformer_discrete_param_space(transformer_name, "high_pct"))
  } else if (transformer_name == "ridge_oof_encode") {
    params$lambda <- .sample_one(.transformer_discrete_param_space(transformer_name, "lambda"))
    params$degree <- .sample_one(.transformer_discrete_param_space(transformer_name, "degree"))
  }
  params
}

#' Mutate a single parameter value for a gene
#' @keywords internal
#' @noRd
mutate_transformer_param <- function(gene, param_name) {
  old_val <- gene$params[[param_name]]
  t_name <- if (!is.null(gene$transformer_name)) gene$transformer_name else gene$op
  if (param_name == "comp_idx") {
    return(old_val)
  }
  if (param_name == "gini_threshold") {
    return(round(stats::runif(1, 0.1, 0.9), 2))
  }
  if (param_name == "displacement") {
    return(round(stats::runif(1, 10, 1000), 2))
  }
  if (param_name == "dens_scale") {
    return(round(stats::runif(1, 0, 1), 2))
  }
  if (param_name == "n_neighbors") {
    new_val <- max(2L, stats::rpois(1, 15))
    attempts <- 0L
    while (identical(new_val, old_val) && attempts < 5L) {
      new_val <- max(2L, stats::rpois(1, 15))
      attempts <- attempts + 1L
    }
    return(new_val)
  }
  grid <- .transformer_discrete_param_space(t_name, param_name)
  if (!is.null(grid)) {
    candidates <- setdiff(grid, old_val)
    return(if (length(candidates) > 0L) .sample_one(candidates) else old_val)
  }
  old_val
}

#' Compute maximum input column count for a transformer
#' @keywords internal
#' @noRd
transformer_max_cols <- function(transformer_name, n_avail, is_multi = TRUE, current_len = 1L, mode = c("add", "modify")) {
  mode <- match.arg(mode)
  if (!is_multi) return(current_len)
  if (transformer_name %in% c("add", "multiply", "row_min", "row_max", "row_sd", "row_na_count", "relative_rating", "geometric_mean", "harmonic_mean", "pythagorean_imbalance", "ridge_oof_encode")) {
    min(5L, n_avail)
  } else if (transformer_name %in% c("concat", "knn_target_encode")) {
    min(3L, n_avail)
  } else if (transformer_name %in% c("famd", "supervised_famd")) {
    min(8L, n_avail)
  } else if (transformer_name == "between_group_pca") {
    min(6L, n_avail)
  } else {
    max(2L, as.integer(floor((1 - exp(-1)) * n_avail)))
  }
}

#' Filter candidate transformer names for task compatibility
#' @keywords internal
#' @noRd
filter_transformers_for_task <- function(allowed_t, task = "classification") {
  if (task == "multiclass") {
    allowed_t <- setdiff(
      allowed_t,
      c(
        "target_encode", "pooled_target_encode", "target_quantile_encode",
        "cat_interaction_target_encode", "target_sd_encode", "knn_target_encode",
        "ridge_oof_encode", "woe_encode"
      )
    )
  } else if (task == "regression") {
    allowed_t <- setdiff(allowed_t, c("target_encode_multiclass", "woe_encode"))
  } else {
    allowed_t <- setdiff(allowed_t, "target_encode_multiclass")
  }
  allowed_t
}
