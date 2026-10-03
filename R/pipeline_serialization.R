#' Convert a gene to a formula string
#'
#' @param gene A gene list
#' @param truncate Logical. If TRUE (default), long list of input columns is
#'   truncated for display.
#' @return A character string representing the gene as a human-readable
#'   formula, e.g. \code{"log(col1)"} or \code{"pca2(col1, col2)"}.
#' @export
gene_to_formula <- function(gene, truncate = TRUE) {
  cols <- gene$input_cols
  if (truncate && length(cols) > 3) {
    cols_str <- paste0(paste(cols[1:3], collapse = ", "), ", ... + ", length(cols) - 3, " more")
  } else {
    cols_str <- paste(cols, collapse = ", ")
  }
  
  if (gene$transformer_name == "one_hot_encode") {
    comp_str <- if (gene$params$comp_idx == 6) "other" else as.character(gene$params$comp_idx)
    sprintf("ohe_%s(%s)", comp_str, cols_str)
  } else if (!is.null(gene$params$component)) {
    sprintf("%s_%s(%s)", gene$transformer_name, gene$params$component, cols_str)
  } else if (!is.null(gene$params$comp_idx)) {
    if (gene$transformer_name == "genie_centroid_dist") {
      sprintf("genie_cdist%d_k%d_t%.2f(%s)", gene$params$comp_idx, gene$params$k, gene$params$gini_threshold, cols_str)
    } else if (gene$transformer_name == "lumbermark_centroid_dist") {
      sprintf("lumb_cdist%d_k%d(%s)", gene$params$comp_idx, gene$params$k, cols_str)
    } else if (gene$transformer_name == "umap") {
      nn_str <- if (!is.null(gene$params$n_neighbors)) paste0("_nn", gene$params$n_neighbors) else ""
      dens_str <- if (!is.null(gene$params$dens_scale)) paste0("_d", gene$params$dens_scale) else ""
      sprintf("umap%d%s%s(%s)", gene$params$comp_idx, nn_str, dens_str, cols_str)
    } else if (gene$transformer_name == "feature_hash") {
      sprintf("fh%d_bin%d(%s)", gene$params$comp_idx, gene$params$num_bins, cols_str)
    } else if (gene$transformer_name == "between_group_pca") {
      num_cols_str <- if (truncate && length(cols) - 1 > 3) {
        paste0(paste(cols[2:4], collapse = ", "), ", ... + ", length(cols) - 4, " more")
      } else {
        paste(cols[-1], collapse = ", ")
      }
      sprintf("bgpca%d(%s | %s)", gene$params$comp_idx, num_cols_str, cols[1])
    } else {
      sprintf("%s%d(%s)", gene$transformer_name, gene$params$comp_idx, cols_str)
    }
  } else if (!is.null(gene$params$Q)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$Q, cols_str)
  } else if (!is.null(gene$params$base)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$base, cols_str)
  } else if (!is.null(gene$params$p)) {
    sprintf("pow%.4g(%s)", gene$params$p, cols_str)
  } else if (!is.null(gene$params$displacement)) {
    sprintf("dlog%.2f(%s)", gene$params$displacement, cols_str)
  } else if (!is.null(gene$params$q)) {
    sprintf("%s_q%.2f(%s)", gene$transformer_name, gene$params$q, cols_str)
  } else if (gene$transformer_name == "umap_genie") {
    nn_str <- if (!is.null(gene$params$n_neighbors)) paste0("_nn", gene$params$n_neighbors) else ""
    dens_str <- if (!is.null(gene$params$dens_scale)) paste0("_d", gene$params$dens_scale) else ""
    sprintf("umap_genie_k%d_t%.2f%s%s(%s)", gene$params$k, gene$params$gini_threshold, nn_str, dens_str, cols_str)
  } else if (gene$transformer_name == "umap_lumbermark") {
    nn_str <- if (!is.null(gene$params$n_neighbors)) paste0("_nn", gene$params$n_neighbors) else ""
    dens_str <- if (!is.null(gene$params$dens_scale)) paste0("_d", gene$params$dens_scale) else ""
    sprintf("umap_lumbermark_k%d%s%s(%s)", gene$params$k, nn_str, dens_str, cols_str)
  } else if (!is.null(gene$params$k)) {
    if (gene$transformer_name == "genie" && !is.null(gene$params$gini_threshold)) {
      sprintf("genie_k%d_t%.2f(%s)", gene$params$k, gene$params$gini_threshold, cols_str)
    } else {
      sprintf("%s_k%d(%s)", gene$transformer_name, gene$params$k, cols_str)
    }
  } else {
    sprintf("%s(%s)", gene$transformer_name, cols_str)
  }
}

#' Convert a gene to a formula string for state caching (ignoring component index)
#'
#' @param gene A gene list
#' @return A character string representing the gene formula suitable for
#'   state caching.  For multi-component transformers (PCA, SVD, UMAP) the
#'   component index is omitted so that all components share one cache key.
#' @export
gene_to_state_formula <- function(gene) {
  if (gene$transformer_name %in% c("pca", "truncated_svd", "mca", "famd", "between_group_pca")) {
    sprintf("%s(%s)", gene$transformer_name, paste(gene$input_cols, collapse = ", "))
  } else if (gene$transformer_name == "umap") {
    nn <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    ds <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    sprintf("umap_nn%d_d%.2f(%s)", nn, ds, paste(gene$input_cols, collapse = ", "))
  } else if (gene$transformer_name == "genie_centroid_dist") {
    k_val <- if (!is.null(gene$params$k)) gene$params$k else 2
    gini_val <- if (!is.null(gene$params$gini_threshold)) gene$params$gini_threshold else 0.5
    sprintf("genie_cdist_k%d_t%.2f(%s)", k_val, gini_val, paste(gene$input_cols, collapse = ", "))
  } else if (gene$transformer_name == "lumbermark_centroid_dist") {
    k_val <- if (!is.null(gene$params$k)) gene$params$k else 2
    sprintf("lumb_cdist_k%d(%s)", k_val, paste(gene$input_cols, collapse = ", "))
  } else if (gene$transformer_name == "umap_genie") {
    nn <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    ds <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    k_val <- if (!is.null(gene$params$k)) gene$params$k else 2
    gini_val <- if (!is.null(gene$params$gini_threshold)) gene$params$gini_threshold else 0.5
    sprintf("umap_genie_k%d_t%.2f_nn%d_d%.2f(%s)", k_val, gini_val, nn, ds, paste(gene$input_cols, collapse = ", "))
  } else if (gene$transformer_name == "umap_lumbermark") {
    nn <- if (!is.null(gene$params$n_neighbors)) gene$params$n_neighbors else 15
    ds <- if (!is.null(gene$params$dens_scale)) gene$params$dens_scale else 0
    k_val <- if (!is.null(gene$params$k)) gene$params$k else 2
    sprintf("umap_lumb_k%d_nn%d_d%.2f(%s)", k_val, nn, ds, paste(gene$input_cols, collapse = ", "))
  } else {
    gene_to_formula(gene, truncate = FALSE)
  }
}

#' Convert an individual to a recipe string of formulas
#'
#' @param ind An evo_individual
#' @return A character string listing all gene formulas in bracket notation,
#'   e.g. \code{"[log(x), sqrt(y)]"}, or \code{"[Original features only]"}
#'   when the individual has no genes.
#' @export
individual_to_recipe_string <- function(ind) {
  n_active_raw <- length(ind$numeric_cols) + length(ind$categorical_cols) + length(ind$datetime_cols)
  
  all_num <- if (!is.null(ind$all_numeric_cols)) ind$all_numeric_cols else ind$numeric_cols
  all_cat <- if (!is.null(ind$all_categorical_cols)) ind$all_categorical_cols else ind$categorical_cols
  all_date <- if (!is.null(ind$all_datetime_cols)) ind$all_datetime_cols else ind$datetime_cols
  n_all_raw <- length(all_num) + length(all_cat) + length(all_date)
  
  n_genes <- length(ind$genes)
  
  features_str <- if (n_genes == 0) {
    "[Original features only]"
  } else {
    formulas <- vapply(ind$genes, gene_to_formula, character(1))
    paste0("[", paste(formulas, collapse = ", "), "]")
  }
  
  stats_str <- sprintf(" (active: %d/%d raw, %d gene%s)", 
                       n_active_raw, n_all_raw, n_genes, if (n_genes == 1) "" else "s")
  
  paste0(features_str, stats_str)
}
