#' Create a transformer definition
#'
#' @param name Transformer name
#' @param type Type: "unary", "binary", "supervised_unary"
#' @param input_type Type of input: "numeric" or "categorical"
#' @param output_type Type of output: "numeric" or "categorical"
#' @param fit_func function(data, input_cols, target_col = NULL) returning state
#' @param apply_func function(data, input_cols, state = NULL) returning new column vector
#' @param name_generator function(input_cols) returning output column name
#' @param allow_replace Logical. Whether column sampling allows replacement.
#' @return An \code{evo_transformer} S3 object:
#'   a list with elements
#'   \code{name}, \code{type}, \code{input_type}, \code{output_type},
#'   \code{fit_func}, \code{apply_func}, \code{name_generator}, and
#'   \code{allow_replace}.
#' @import data.table
#' @importFrom uwot umap
#' @importFrom genieclust genie
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
#' }
#'
#' \strong{Rank / distribution (numeric -> numeric, stateful)}
#' \describe{
#'   \item{\code{rank_transform}}{ECDF-based percentile rank mapped to
#'     \eqn{[0, 1]}.  Fit on training data; robust to outliers.}
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
    sample_vals <- if (length(vals) > 100) sample(vals, 100) else vals
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

# --- STATELESS UNARY TRANSFORMERS ---

