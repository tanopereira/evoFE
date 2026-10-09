#' Resolve allowed transformers against registry
#'
#' @param at Vector of transformer names or preset keyword ("all", "basic", "clustering", "robust")
#' @param all_t Vector of all registered transformer names, defaults to names(evo_transformers)
#' @return Character vector of validated transformer names
#' @noRd
resolve_allowed_transformers <- function(at, all_t = names(evo_transformers)) {
  if (is.null(at)) at <- "all"
  if (length(at) == 1) {
    if (at == "all") {
      at <- all_t
    } else if (at == "basic") {
      at <- intersect(all_t, c(
        "add", "subtract", "multiply", "divide",
        "log", "sqrt", "reciprocal", "power", "displaced_log", "fourier_basis",
        "normalized_difference", "frequency_encode", "numeric_freq",
        "row_min", "row_max", "relative_rating", "geometric_mean", "harmonic_mean", "pythagorean_imbalance",
        "one_hot_encode", "target_encode", "pooled_target_encode", "target_encode_multiclass",
        "feature_hash",
        "rank_transform", "robust_scale", "smooth_clip", "groupby_mean", "groupby_min", "groupby_max", "concat"
      ))
    } else if (at == "clustering") {
      at <- intersect(all_t, c(
        "genie", "genie_centroid_dist", "lumbermark", "lumbermark_centroid_dist",
        "mst_score", "deadwood", "umap", "random_projection", "truncated_svd",
        "pca", "umap_genie", "umap_lumbermark", "mca", "famd", "between_group_pca"
      ))
    } else if (at == "robust") {
      at <- intersect(all_t, c(
        "log", "sqrt", "reciprocal", "power", "displaced_log", "rank_transform",
        "robust_scale", "smooth_clip", "fourier_basis",
        "add", "subtract", "multiply", "divide",
        "normalized_difference", "log_ratio",
        "row_min", "row_max", "relative_rating", "geometric_mean", "harmonic_mean", "pythagorean_imbalance",
        "target_encode", "pooled_target_encode", "woe_encode", "frequency_encode", "numeric_freq",
        "feature_hash",
        "groupby_mean", "groupby_median", "groupby_sd",
        "groupby_zscore", "groupby_ratio", "groupby_quantile",
        "groupby_min", "groupby_max", "groupby_signed_log",
        "quantile_binning", "pca", "concat", "mca", "famd", "between_group_pca"
      ))
    }
  }
  at <- intersect(at, all_t)
  if (length(at) == 0) {
    warning("No valid transformers found in 'allowed_transformers'. Falling back to 'all'.")
    at <- all_t
  }
  at
}

.resolve_allowed_transformers <- resolve_allowed_transformers
