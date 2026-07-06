#' Create a single gene
#'
#' @param transformer_name Name of the transformer
#' @param input_cols Vector of input column names
#' @return A gene list with elements \code{transformer_name}, \code{input_cols},
#'   \code{params} (transformer-specific parameters), \code{state} (\code{NULL}
#'   until fitted), and \code{output_col} (auto-generated column name).
#' @export
create_gene <- function(transformer_name, input_cols) {
  transformer <- evo_transformers[[transformer_name]]
  params <- list()
  if (transformer_name %in% c("pca", "truncated_svd", "umap")) {
    C <- max(2L, as.integer(round(log2(length(input_cols)))))
    params$comp_idx <- sample(1:C, 1)
    if (transformer_name == "umap") {
      params$n_neighbors <- max(2L, stats::rpois(1, 15))
      params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    }
  } else if (transformer_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
    params$k <- sample(2:5, 1)
    params$comp_idx <- sample(1:params$k, 1)
    if (transformer_name == "genie_centroid_dist") {
      params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
    }
  } else if (transformer_name == "one_hot_encode") {
    params$comp_idx <- sample(1:6, 1)
  } else if (transformer_name == "genie") {
    params$k <- sample(2:5, 1)
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "umap_genie") {
    params$n_neighbors <- max(2L, stats::rpois(1, 15))
    params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    params$k <- sample(2:5, 1)
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "lumbermark") {
    params$k <- sample(2:5, 1)
  } else if (transformer_name %in% c("quantile_binning", "quantile_binning_cat")) {
    params$Q <- sample(3:10, 1)
  } else if (transformer_name %in% c("log_binning", "log_binning_cat")) {
    params$base <- sample(2:10, 1)
  } else if (transformer_name == "target_encode_multiclass") {
    params$comp_idx <- sample(1:5, 1)
  } else if (transformer_name == "datetime_extract") {
    params$component <- sample(c("year", "month", "day", "hour", "day_of_week", "weekend"), 1)
  } else if (transformer_name == "power") {
    params$p <- sample(c(0.5, 1/3, 2, 3), 1)
  } else if (transformer_name == "displaced_log") {
    params$displacement <- round(stats::runif(1, 10, 1000), 2)
  } else if (transformer_name == "groupby_quantile") {
    params$q <- sample(c(0.25, 0.75), 1)
  }
  gene <- list(
    transformer_name = transformer_name,
    input_cols = input_cols,
    params = params,
    state = NULL # Populated during fitness evaluation if stateful
  )
  gene$output_col <- transformer$name_generator(gene)
  gene
}

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
  if (gene$transformer_name %in% c("pca", "truncated_svd")) {
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
    formulas <- sapply(ind$genes, gene_to_formula)
    paste0("[", paste(formulas, collapse = ", "), "]")
  }
  
  stats_str <- sprintf(" (active: %d/%d raw, %d gene%s)", 
                       n_active_raw, n_all_raw, n_genes, if (n_genes == 1) "" else "s")
  
  paste0(features_str, stats_str)
}

topological_sort_genes <- function(genes, original_cols) {
  if (length(genes) == 0) return(list())
  
  available <- original_cols
  sorted_genes <- list()
  remaining_genes <- genes
  
  made_progress <- TRUE
  while (length(remaining_genes) > 0 && made_progress) {
    made_progress <- FALSE
    keep_indices <- c()
    for (i in seq_along(remaining_genes)) {
      gene <- remaining_genes[[i]]
      if (all(gene$input_cols %in% available)) {
        sorted_genes <- c(sorted_genes, list(gene))
        available <- c(available, gene$output_col)
        made_progress <- TRUE
      } else {
        keep_indices <- c(keep_indices, i)
      }
    }
    remaining_genes <- remaining_genes[keep_indices]
  }
  
  sorted_genes
}

#' Create an individual
#'
#' @param genes List of genes
#' @param numeric_cols Vector of numeric column names
#' @param categorical_cols Vector of categorical column names
#' @param datetime_cols Vector of datetime column names
#' @return An \code{evo_individual} S3 object:
#'   a list with elements \code{genes} (topologically sorted),
#'   \code{numeric_cols}, \code{categorical_cols}, and \code{fitness}
#'   (initialised to \code{NA_real_}).
#' @examples
#' \donttest{
#' ind <- create_individual(
#'   genes = list(),
#'   numeric_cols = c("a", "b"),
#'   categorical_cols = c("c")
#' )
#' print(ind)
#' }
#' @export
create_individual <- function(genes = list(), numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                              all_numeric_cols = NULL, all_categorical_cols = NULL, all_datetime_cols = NULL) {
  if (is.null(all_numeric_cols)) all_numeric_cols <- numeric_cols
  if (is.null(all_categorical_cols)) all_categorical_cols <- categorical_cols
  if (is.null(all_datetime_cols)) all_datetime_cols <- datetime_cols
  
  original_cols <- c(all_numeric_cols, all_categorical_cols, all_datetime_cols)
  sorted_genes <- topological_sort_genes(genes, original_cols)
  structure(
    list(
      genes = sorted_genes,
      numeric_cols = numeric_cols,
      categorical_cols = categorical_cols,
      datetime_cols = datetime_cols,
      all_numeric_cols = all_numeric_cols,
      all_categorical_cols = all_categorical_cols,
      all_datetime_cols = all_datetime_cols,
      fitness = NA_real_
    ),
    class = "evo_individual"
  )
}

toggle_raw_feature <- function(ind, importances = numeric(0), temperature = 1.0, verbose = FALSE) {
  all_num <- ind$all_numeric_cols
  all_cat <- ind$all_categorical_cols
  all_date <- ind$all_datetime_cols
  
  active_num <- ind$numeric_cols
  active_cat <- ind$categorical_cols
  active_date <- ind$datetime_cols
  
  inactive_num <- setdiff(all_num, active_num)
  inactive_cat <- setdiff(all_cat, active_cat)
  inactive_date <- setdiff(all_date, active_date)
  
  active_cols <- c(active_num, active_cat, active_date)
  inactive_cols <- c(inactive_num, inactive_cat, inactive_date)
  
  total_active <- length(active_cols) + length(ind$genes)
  total_avail <- length(all_num) + length(all_cat) + length(all_date)
  min_active <- if (total_avail >= 2) 2 else 1
  
  can_deactivate <- (total_active > min_active) && (length(active_cols) > 0)
  can_activate <- length(inactive_cols) > 0
  
  if (!can_deactivate && !can_activate) {
    return(ind)
  }
  
  op <- if (can_deactivate && can_activate) {
    sample(c("deactivate", "activate"), 1)
  } else if (can_deactivate) {
    "deactivate"
  } else {
    "activate"
  }
  
  if (op == "deactivate") {
    weights <- sapply(active_cols, function(c) {
      val <- if (c %in% names(importances)) importances[[c]] else 0.0
      if (is.na(val) || !is.finite(val)) val <- 0.0
      exp(-val / temperature)
    })
    
    weights[is.na(weights) | !is.finite(weights)] <- 0
    if (sum(weights) == 0) weights <- rep(1, length(weights))
    
    col_to_deactivate <- sample(active_cols, 1, prob = weights)
    
    if (col_to_deactivate %in% active_num) {
      ind$numeric_cols <- setdiff(ind$numeric_cols, col_to_deactivate)
    } else if (col_to_deactivate %in% active_cat) {
      ind$categorical_cols <- setdiff(ind$categorical_cols, col_to_deactivate)
    } else if (col_to_deactivate %in% active_date) {
      ind$datetime_cols <- setdiff(ind$datetime_cols, col_to_deactivate)
    }
    
    ind$fitness <- NA_real_
    if (verbose) {
      message(sprintf("    [Mutation] Toggled Off (Deactivated) raw feature: '%s'", col_to_deactivate))
    }
    
  } else {
    col_to_activate <- sample(inactive_cols, 1)
    
    if (col_to_activate %in% inactive_num) {
      ind$numeric_cols <- c(ind$numeric_cols, col_to_activate)
    } else if (col_to_activate %in% inactive_cat) {
      ind$categorical_cols <- c(ind$categorical_cols, col_to_activate)
    } else if (col_to_activate %in% inactive_date) {
      ind$datetime_cols <- c(ind$datetime_cols, col_to_activate)
    }
    
    ind$fitness <- NA_real_
    if (verbose) {
      message(sprintf("    [Mutation] Toggled On (Activated) raw feature: '%s'", col_to_activate))
    }
  }
  
  ind
}

#' Mutate an individual
#'
#' @param ind An evo_individual.
#' @param verbose Logical. Whether to print mutation details.
#' @param force_add Logical. If TRUE, forces adding a new gene.
#' @param importances A numeric vector of feature importances.
#' @param temperature A numeric temperature value controlling selection weights.
#' @param task The task type ("classification", "regression", or "multiclass")
#' @param tested_gene_outputs Character vector of gene output names that have
#'   been evaluated in a previous generation and are safe for chaining. When
#'   NULL (default), all existing gene outputs are available. Pass character(0)
#'   to block all chaining (e.g. during initialization).
#' @param allowed_transformers A character vector of allowed transformer names,
#'   or NULL/"all" to allow all.
#' @param migrated_genes A list of genes migrated from other islands.
#' @param gene_migration_prob Probability of selecting a migrated gene during mutation.
#' @return An \code{evo_individual} with the mutation applied
#'   (gene added, removed, or modified) and \code{fitness} reset
#'   to \code{NA_real_}.
#' @examples
#' \donttest{
#' ind <- create_individual(
#'   numeric_cols = c("a", "b"),
#'   categorical_cols = c("c")
#' )
#' mutated_ind <- mutate(ind)
#' }
#' @export
mutate <- function(ind, verbose = FALSE, force_add = FALSE, importances = numeric(0), temperature = 1.0, task = "classification", tested_gene_outputs = NULL, allowed_transformers = NULL, migrated_genes = list(), gene_migration_prob = 0.2, raw_toggle_prob = 0.2) {
  if (length(ind$all_numeric_cols) == 0 && length(ind$all_categorical_cols) == 0 && length(ind$all_datetime_cols) == 0) return(ind)
  
  if (!force_add && stats::runif(1) < raw_toggle_prob) {
    return(toggle_raw_feature(ind, importances = importances, temperature = temperature, verbose = verbose))
  }
  
  # Categorize existing gene outputs by type, restricted to tested genes
  gene_num <- character(0)
  gene_cat <- character(0)
  gene_date <- character(0)
  for (gene in ind$genes) {
    # Skip if this gene's output hasn't been evaluated yet
    if (!is.null(tested_gene_outputs) && !(gene$output_col %in% tested_gene_outputs)) {
      next
    }
    t_def <- evo_transformers[[gene$transformer_name]]
    out_type <- if (!is.null(t_def$output_type)) t_def$output_type else "numeric"
    if (out_type == "categorical") {
      gene_cat <- c(gene_cat, gene$output_col)
    } else if (out_type == "datetime") {
      gene_date <- c(gene_date, gene$output_col)
    } else {
      gene_num <- c(gene_num, gene$output_col)
    }
  }
  
  weighted_sample <- function(cols, size, replace = FALSE) {
    if (length(cols) == 0) return(character(0))
    if (length(cols) == 1) return(rep(cols, size))
    if (length(importances) == 0) return(sample(cols, size, replace = replace))
    
    # Normalize importances so they sum to 1 to ensure scale invariance across models
    if (length(importances) > 0) {
      importances[!is.finite(importances) | importances < 0] <- 0
      imp_sum <- sum(importances, na.rm = TRUE)
      if (imp_sum > 0) {
        importances <- importances / imp_sum
      }
    }
    
    # Baseline for missing features: minimum of known, or 0.01 (handling NAs/non-finites safely)
    clean_importances <- importances[!is.na(importances) & is.finite(importances) & importances > 0]
    baseline <- if (length(clean_importances) > 0) min(clean_importances) else 0.01
    
    weights <- sapply(cols, function(c) {
      val <- if (c %in% names(importances)) importances[[c]] else baseline
      if (is.na(val) || !is.finite(val)) val <- baseline
      exp(val / temperature)
    })
    
    # Ensure weights are completely clean, positive, finite, and non-NA
    weights[is.na(weights) | !is.finite(weights)] <- 0
    weights[weights < 0] <- 0
    
    # If sampling without replacement, we must have at least 'size' positive weights
    if (!replace && sum(weights > 0) < size) {
      weights <- weights + 1e-10
    }
    
    if (sum(weights) == 0) {
      weights <- rep(1, length(weights))
    }
    
    sample(cols, size, replace = replace, prob = weights)
  }
  
  mut_type <- 3 # Default to Add
  if (!force_add && length(ind$genes) > 0) {
    r <- stats::runif(1)
    if (r < 0.33) {
      mut_type <- 1 # Remove
    } else if (r < 0.66) {
      mut_type <- 2 # Modify
    }
  }
  
  if (mut_type == 1) {
    # Importance-guided gene removal: preferentially remove low-importance genes
    if (length(importances) > 0 && length(ind$genes) > 1) {
      gene_imps <- sapply(ind$genes, function(g) {
        if (g$output_col %in% names(importances)) importances[[g$output_col]] else 0
      })
      # Invert: low importance → high removal probability
      removal_weights <- 1 / (gene_imps + 1e-8)
      removal_weights[!is.finite(removal_weights)] <- 1
      idx <- sample(seq_along(ind$genes), 1, prob = removal_weights)
    } else {
      idx <- sample(seq_along(ind$genes), 1)
    }
    removed <- ind$genes[[idx]]
    ind$genes <- ind$genes[-idx]
    ind$fitness <- NA_real_
    if (verbose) {
      message(sprintf("    [Mutation] Removed gene: %s (%s)", removed$output_col, gene_to_formula(removed)))
    }
  } else if (mut_type == 2) {
    # Enriched Modify mutation: pick a random gene, then apply one of 4 sub-operations
    idx <- sample(seq_along(ind$genes), 1)
    gene_to_mod <- ind$genes[[idx]]
    t_name <- gene_to_mod$transformer_name
    t_def <- evo_transformers[[t_name]]
    is_multi <- t_def$type == "multivariate"
    old_formula <- gene_to_formula(gene_to_mod)
    
    gene_num_ex <- setdiff(gene_num, gene_to_mod$output_col)
    avail_num <- c(ind$all_numeric_cols, gene_num_ex)
    avail_cat <- c(ind$all_categorical_cols, gene_cat)
    avail_date <- c(ind$all_datetime_cols, gene_date)
    avail_for_type <- if (t_def$input_type == "numeric") {
      avail_num
    } else if (t_def$input_type == "categorical") {
      avail_cat
    } else if (t_def$input_type == "datetime") {
      avail_date
    } else {
      c(avail_cat, avail_num)
    }
    unused <- setdiff(avail_for_type, gene_to_mod$input_cols)
    
    max_cols <- if (t_name %in% c("add", "multiply")) {
      min(5, length(avail_for_type))
    } else if (is_multi) {
      max(2, floor((1 - exp(-1)) * length(avail_for_type)))
    } else {
      length(gene_to_mod$input_cols) # non-multivariate genes have fixed arity
    }
    
    # Determine which sub-operations are applicable
    can_add_col <- is_multi && length(unused) > 0 && length(gene_to_mod$input_cols) < max_cols
    can_remove_col <- is_multi && length(gene_to_mod$input_cols) > 2
    can_swap_col <- length(gene_to_mod$input_cols) > 0 && length(unused) > 0
    can_mutate_params <- length(gene_to_mod$params) > 0
    
    applicable_ops <- c()
    if (can_add_col) applicable_ops <- c(applicable_ops, "add_col")
    if (can_remove_col) applicable_ops <- c(applicable_ops, "remove_col")
    if (can_swap_col) applicable_ops <- c(applicable_ops, "swap_col")
    if (can_mutate_params) applicable_ops <- c(applicable_ops, "mutate_params")
    
    mod_success <- FALSE
    if (length(applicable_ops) > 0) {
      sub_op <- sample(applicable_ops, 1)
      
      if (sub_op == "add_col") {
        # Add an unused column to a multivariate gene
        new_col <- weighted_sample(unused, 1)
        gene_to_mod$input_cols <- c(gene_to_mod$input_cols, new_col)
        gene_to_mod$state <- NULL
        gene_to_mod$output_col <- t_def$name_generator(gene_to_mod)
        mod_success <- TRUE
        if (verbose) {
          message(sprintf("    [Mutation] Modify (add_col): added '%s' to %s -> %s", 
                          new_col, old_formula, gene_to_formula(gene_to_mod)))
        }
        
      } else if (sub_op == "remove_col") {
        # Remove one column from a multivariate gene (importance-guided)
        col_imps <- sapply(gene_to_mod$input_cols, function(c) {
          if (c %in% names(importances)) importances[[c]] else 0
        })
        removal_weights <- 1 / (col_imps + 1e-8)
        removal_weights[!is.finite(removal_weights)] <- 1
        drop_idx <- sample(seq_along(gene_to_mod$input_cols), 1, prob = removal_weights)
        dropped_col <- gene_to_mod$input_cols[drop_idx]
        gene_to_mod$input_cols <- gene_to_mod$input_cols[-drop_idx]
        gene_to_mod$state <- NULL
        gene_to_mod$output_col <- t_def$name_generator(gene_to_mod)
        mod_success <- TRUE
        if (verbose) {
          message(sprintf("    [Mutation] Modify (remove_col): removed '%s' from %s -> %s", 
                          dropped_col, old_formula, gene_to_formula(gene_to_mod)))
        }
        
      } else if (sub_op == "swap_col") {
        # Swap one input column for a different unused column
        swap_idx <- sample(seq_along(gene_to_mod$input_cols), 1)
        old_col <- gene_to_mod$input_cols[swap_idx]
        new_col <- weighted_sample(unused, 1)
        gene_to_mod$input_cols[swap_idx] <- new_col
        gene_to_mod$state <- NULL
        gene_to_mod$output_col <- t_def$name_generator(gene_to_mod)
        mod_success <- TRUE
        if (verbose) {
          message(sprintf("    [Mutation] Modify (swap_col): swapped '%s' for '%s' in %s -> %s", 
                          old_col, new_col, old_formula, gene_to_formula(gene_to_mod)))
        }
        
      } else if (sub_op == "mutate_params") {
        # Mutate one parameter of the gene
        param_name <- sample(names(gene_to_mod$params), 1)
        old_val <- gene_to_mod$params[[param_name]]
        
        new_val <- if (param_name == "k") {
          # k for clustering: sample from 2:5, excluding current value if possible
          candidates <- setdiff(2:5, old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "gini_threshold") {
          round(stats::runif(1, 0.1, 0.9), 2)
        } else if (param_name == "Q") {
          candidates <- setdiff(3:10, old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "base") {
          candidates <- setdiff(2:10, old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "displacement") {
          round(stats::runif(1, 10, 1000), 2)
        } else if (param_name == "n_neighbors") {
          new_val <- max(2L, stats::rpois(1, 15))
          attempts <- 0
          while (new_val == old_val && attempts < 5) {
            new_val <- max(2L, stats::rpois(1, 15))
            attempts <- attempts + 1
          }
          new_val
        } else if (param_name == "dens_scale") {
          round(stats::runif(1, 0, 1), 2)
        } else if (param_name == "comp_idx") {
          # For multi-component transformers, don't mutate comp_idx in isolation
          # as components are managed together during Add mutation
          old_val # no-op
        } else if (param_name == "component") {
          components <- c("year", "month", "day", "hour", "day_of_week", "weekend")
          candidates <- setdiff(components, old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "p") {
          candidates <- setdiff(c(0.5, 1/3, 2, 3), old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "q") {
          candidates <- setdiff(c(0.25, 0.75), old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else {
          old_val
        }
        
        if (!identical(new_val, old_val)) {
          gene_to_mod$params[[param_name]] <- new_val
          gene_to_mod$state <- NULL
          gene_to_mod$output_col <- t_def$name_generator(gene_to_mod)
          mod_success <- TRUE
          if (verbose) {
            message(sprintf("    [Mutation] Modify (mutate_params): changed %s from %s to %s in %s -> %s", 
                            param_name, as.character(old_val), as.character(new_val),
                            old_formula, gene_to_formula(gene_to_mod)))
          }
        }
      }
    }
    
    if (mod_success) {
      # Check if modified gene conflicts with existing output names
      existing_out <- if (length(ind$genes) > 1) sapply(ind$genes[-idx], function(g) g$output_col) else character(0)
      if (!(gene_to_mod$output_col %in% existing_out)) {
        ind$genes[[idx]] <- gene_to_mod
        ind$fitness <- NA_real_
      } else {
        mut_type <- 3 # Conflict, fallback to add
      }
    } else {
      mut_type <- 3 # No applicable sub-operation, fallback to add
    }
  }
  
  if (mut_type == 3) {
    # Attempt to inject a migrated gene with probability gene_migration_prob
    injected <- FALSE
    if (length(migrated_genes) > 0 && stats::runif(1) < gene_migration_prob) {
      selected_mig_gene <- sample(migrated_genes, 1)[[1]]
      
      existing_out_cols <- sapply(ind$genes, function(g) g$output_col)
      valid_inputs <- c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols, existing_out_cols)
      
      if (all(selected_mig_gene$input_cols %in% valid_inputs)) {
        existing_formulas <- vapply(ind$genes, gene_to_formula, character(1))
        mig_formula <- gene_to_formula(selected_mig_gene)
        
        if (!(mig_formula %in% existing_formulas)) {
          # Strip the state of the gene so it is re-fitted on the receiving island
          selected_mig_gene$state <- NULL
          ind$genes <- c(ind$genes, list(selected_mig_gene))
          ind$fitness <- NA_real_
          injected <- TRUE
          if (verbose) {
            message(sprintf("    [Mutation] Injected migrated gene: %s (%s)", selected_mig_gene$output_col, mig_formula))
          }
        }
      }
    }
    
    if (!injected) {
      # Add a random gene
      if (is.null(allowed_transformers)) {
      allowed_t <- names(evo_transformers)
    } else {
      allowed_t <- allowed_transformers
    }
    if (task == "multiclass") {
      allowed_t <- setdiff(allowed_t, c("target_encode", "pooled_target_encode"))
    } else {
      allowed_t <- setdiff(allowed_t, c("target_encode_multiclass"))
    }
    if (length(allowed_t) == 0) allowed_t <- names(evo_transformers)
    t_name <- sample(allowed_t, 1)
    t_def <- evo_transformers[[t_name]]
    
    avail_num <- c(ind$all_numeric_cols, gene_num)
    avail_cat <- c(ind$all_categorical_cols, gene_cat)
    avail_date <- c(ind$all_datetime_cols, gene_date)
    
    # Select available columns based on input_type
    available_cols <- if (t_def$input_type == "numeric") {
      avail_num
    } else if (t_def$input_type == "categorical") {
      avail_cat
    } else if (t_def$input_type == "datetime") {
      avail_date
    } else if (t_def$input_type == "mixed") {
      c(avail_cat, avail_num) # Placeholder to pass length check
    } else {
      character(0)
    }
    
    # If no columns available for this type, return early
    if (t_def$input_type == "mixed") {
      if (length(avail_num) == 0 || length(avail_cat) == 0) return(ind)
    } else {
      if (length(available_cols) == 0) return(ind)
    }
    
    # Select random columns
    if (t_def$type %in% c("unary", "supervised_unary") && t_def$input_type != "mixed") {
      cols <- weighted_sample(available_cols, 1)
    } else if (t_def$type == "binary") {
      allow_rep <- if (t_name %in% c("subtract", "divide")) FALSE else TRUE
      if (!allow_rep && length(available_cols) < 2) return(ind)
      cols <- weighted_sample(available_cols, 2, replace = allow_rep)
    } else if (t_def$type == "multivariate") {
      if (length(available_cols) < 2) return(ind)
      max_cols <- if (t_name %in% c("add", "multiply")) {
        min(5, length(available_cols))
      } else if (t_name == "concat") {
        min(3, length(available_cols))
      } else {
        max(2, floor((1 - exp(-1)) * length(available_cols)))
      }
      num_cols <- if (max_cols == 2) 2 else sample(2:max_cols, 1)
      allow_rep <- if (!is.null(t_def$allow_replace)) t_def$allow_replace else FALSE
      cols <- weighted_sample(available_cols, num_cols, replace = allow_rep)
    } else if (t_def$input_type == "mixed") {
      # Mixed takes 1 categorical and 1 numeric
      col_cat <- weighted_sample(avail_cat, 1)
      col_num <- weighted_sample(avail_num, 1)
      cols <- c(col_cat, col_num)
    } else {
      cols <- weighted_sample(available_cols, 1)
    }
    
    # If the transformer is multi-component, add all components
    new_genes_to_add <- list()
    if (t_name %in% c("pca", "truncated_svd", "umap", "genie_centroid_dist", "lumbermark_centroid_dist")) {
      C <- if (t_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
        sample(2:5, 1)
      } else {
        max(2L, as.integer(round(log2(length(cols)))))
      }
      
      gini_threshold <- if (t_name == "genie_centroid_dist") round(stats::runif(1, 0.1, 0.9), 2) else NULL
      n_neighbors <- if (t_name == "umap") max(2L, stats::rpois(1, 15)) else NULL
      dens_scale <- if (t_name == "umap") round(stats::runif(1, 0, 1), 2) else NULL
      
      for (comp in 1:C) {
        g <- create_gene(t_name, cols)
        g$params$comp_idx <- comp
        if (t_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
          g$params$k <- C
          if (!is.null(gini_threshold)) g$params$gini_threshold <- gini_threshold
        } else if (t_name == "umap") {
          g$params$n_neighbors <- n_neighbors
          g$params$dens_scale <- dens_scale
        }
        g$output_col <- t_def$name_generator(g)
        new_genes_to_add <- c(new_genes_to_add, list(g))
      }
    } else {
      new_gene <- create_gene(t_name, cols)
      new_genes_to_add <- list(new_gene)
    }
    
    # Avoid exact duplicates and add genes
    existing_out <- sapply(ind$genes, function(g) g$output_col)
    added_any <- FALSE
    for (g in new_genes_to_add) {
      if (!(g$output_col %in% existing_out)) {
        ind$genes <- c(ind$genes, list(g))
        added_any <- TRUE
        if (verbose) {
          message(sprintf("    [Mutation] Added gene: %s (%s)", 
                          g$output_col, gene_to_formula(g)))
        }
      } else {
        if (verbose) {
          message(sprintf("    [Mutation] Attempted to add duplicate gene: %s (%s) (skipped)", g$output_col, gene_to_formula(g)))
        }
      }
    }
    if (added_any) {
      ind$fitness <- NA_real_
    }
  }
}
  ind$genes <- topological_sort_genes(ind$genes, c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols))
  ind
}

#' Crossover two individuals
#'
#' @param ind1 Parent 1
#' @param ind2 Parent 2
#' @param verbose Logical. Whether to print crossover details.
#' @return An \code{evo_individual} child created by randomly sampling genes
#'   from both parents with duplicate gene outputs removed.
#' @examples
#' \donttest{
#' ind1 <- create_individual(numeric_cols = c("a", "b"))
#' ind1 <- mutate(ind1, force_add = TRUE)
#' ind2 <- create_individual(numeric_cols = c("a", "b"))
#' ind2 <- mutate(ind2, force_add = TRUE)
#' child <- crossover(ind1, ind2)
#' }
#' @export
crossover <- function(ind1, ind2, verbose = FALSE) {
  genes1 <- ind1$genes
  genes2 <- ind2$genes
  
  len1_before <- length(genes1)
  len2_before <- length(genes2)
  
  if (length(genes1) > 0) {
    keep1 <- sample(c(TRUE, FALSE), length(genes1), replace = TRUE)
    genes1 <- genes1[keep1]
  }
  
  if (length(genes2) > 0) {
    keep2 <- sample(c(TRUE, FALSE), length(genes2), replace = TRUE)
    genes2 <- genes2[keep2]
  }
  
  child_genes <- c(genes1, genes2)
  
  # Basic deduplication based on output_col
  if (length(child_genes) > 0) {
    out_cols <- sapply(child_genes, function(g) g$output_col)
    child_genes <- child_genes[!duplicated(out_cols)]
  }
  
  if (verbose) {
    child_genes_str <- if (length(child_genes) > 0) {
      paste(sapply(child_genes, gene_to_formula), collapse = ", ")
    } else {
      "None"
    }
    message(sprintf("    [Crossover] Parent 1 (%d genes) x Parent 2 (%d genes) -> Child genes: [%s]",
                    len1_before, len2_before, child_genes_str))
  }
  
  crossover_mask <- function(active1, active2, all_cols) {
    if (length(all_cols) == 0) return(character(0))
    choose_from_1 <- stats::runif(length(all_cols)) < 0.5
    active_child <- character(0)
    for (i in seq_along(all_cols)) {
      col <- all_cols[i]
      is_active <- if (choose_from_1[i]) (col %in% active1) else (col %in% active2)
      if (is_active) {
        active_child <- c(active_child, col)
      }
    }
    active_child
  }
  
  child_num <- crossover_mask(ind1$numeric_cols, ind2$numeric_cols, ind1$all_numeric_cols)
  child_cat <- crossover_mask(ind1$categorical_cols, ind2$categorical_cols, ind1$all_categorical_cols)
  child_date <- crossover_mask(ind1$datetime_cols, ind2$datetime_cols, ind1$all_datetime_cols)
  
  total_active <- length(child_num) + length(child_cat) + length(child_date) + length(child_genes)
  total_avail <- length(ind1$all_numeric_cols) + length(ind1$all_categorical_cols) + length(ind1$all_datetime_cols)
  min_active <- if (total_avail >= 2) 2 else 1
  
  if (total_active < min_active) {
    active_p1 <- c(ind1$numeric_cols, ind1$categorical_cols, ind1$datetime_cols)
    active_p2 <- c(ind2$numeric_cols, ind2$categorical_cols, ind2$datetime_cols)
    pool <- union(active_p1, active_p2)
    if (length(pool) < min_active) pool <- c(ind1$all_numeric_cols, ind1$all_categorical_cols, ind1$all_datetime_cols)
    
    needed <- min_active - total_active
    if (length(pool) > 0) {
      inactive_pool <- setdiff(pool, c(child_num, child_cat, child_date))
      if (length(inactive_pool) < needed) inactive_pool <- pool
      
      force_active <- sample(inactive_pool, min(length(inactive_pool), needed))
      for (col in force_active) {
        if (col %in% ind1$all_numeric_cols) child_num <- unique(c(child_num, col))
        else if (col %in% ind1$all_categorical_cols) child_cat <- unique(c(child_cat, col))
        else if (col %in% ind1$all_datetime_cols) child_date <- unique(c(child_date, col))
      }
    }
  }
  
  create_individual(
    genes = child_genes,
    numeric_cols = child_num,
    categorical_cols = child_cat,
    datetime_cols = child_date,
    all_numeric_cols = ind1$all_numeric_cols,
    all_categorical_cols = ind1$all_categorical_cols,
    all_datetime_cols = ind1$all_datetime_cols
  )
}

#' Union Crossover of two individuals
#'
#' @param ind1 Parent 1
#' @param ind2 Parent 2
#' @param verbose Logical. Whether to print crossover details.
#' @return An \code{evo_individual} child created by taking the union of all
#'   genes from both parents with duplicate gene outputs removed.
#' @export
union_crossover <- function(ind1, ind2, verbose = FALSE) {
  genes1 <- ind1$genes
  genes2 <- ind2$genes
  
  len1_before <- length(genes1)
  len2_before <- length(genes2)
  
  child_genes <- c(genes1, genes2)
  
  # Basic deduplication based on output_col
  if (length(child_genes) > 0) {
    out_cols <- sapply(child_genes, function(g) g$output_col)
    child_genes <- child_genes[!duplicated(out_cols)]
  }
  
  if (verbose) {
    child_genes_str <- if (length(child_genes) > 0) {
      paste(sapply(child_genes, gene_to_formula), collapse = ", ")
    } else {
      "None"
    }
    message(sprintf("    [Union Crossover] Parent 1 (%d genes) x Parent 2 (%d genes) -> Child genes: [%s]",
                    len1_before, len2_before, child_genes_str))
  }
  
  child_num <- union(ind1$numeric_cols, ind2$numeric_cols)
  child_cat <- union(ind1$categorical_cols, ind2$categorical_cols)
  child_date <- union(ind1$datetime_cols, ind2$datetime_cols)
  
  create_individual(
    genes = child_genes,
    numeric_cols = child_num,
    categorical_cols = child_cat,
    datetime_cols = child_date,
    all_numeric_cols = ind1$all_numeric_cols,
    all_categorical_cols = ind1$all_categorical_cols,
    all_datetime_cols = ind1$all_datetime_cols
  )
}
