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
    } else {
      sprintf("%s%d(%s)", gene$transformer_name, gene$params$comp_idx, cols_str)
    }
  } else if (!is.null(gene$params$Q)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$Q, cols_str)
  } else if (!is.null(gene$params$base)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$base, cols_str)
  } else if (!is.null(gene$params$p)) {
    sprintf("pow%.4g(%s)", gene$params$p, cols_str)
  } else if (!is.null(gene$params$q)) {
    sprintf("%s_q%.2f(%s)", gene$transformer_name, gene$params$q, cols_str)
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
  if (gene$transformer_name %in% c("pca", "truncated_svd", "umap", "genie_centroid_dist", "lumbermark_centroid_dist")) {
    sprintf("%s(%s)", gene$transformer_name, paste(gene$input_cols, collapse = ", "))
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
  if (length(ind$genes) == 0) return("[Original features only]")
  formulas <- sapply(ind$genes, gene_to_formula)
  paste0("[", paste(formulas, collapse = ", "), "]")
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
#' @return An \code{evo_individual} S3 object:
#'   a list with elements \code{genes} (topologically sorted),
#'   \code{numeric_cols}, \code{categorical_cols}, and \code{fitness}
#'   (initialised to \code{NA_real_}).
#' @export
create_individual <- function(genes = list(), numeric_cols = character(0), categorical_cols = character(0)) {
  original_cols <- c(numeric_cols, categorical_cols)
  sorted_genes <- topological_sort_genes(genes, original_cols)
  structure(
    list(
      genes = sorted_genes,
      numeric_cols = numeric_cols,
      categorical_cols = categorical_cols,
      fitness = NA_real_
    ),
    class = "evo_individual"
  )
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
#' @return An \code{evo_individual} with the mutation applied
#'   (gene added, removed, or modified) and \code{fitness} reset
#'   to \code{NA_real_}.
#' @export
mutate <- function(ind, verbose = FALSE, force_add = FALSE, importances = numeric(0), temperature = 1.0, task = "classification", tested_gene_outputs = NULL, allowed_transformers = NULL) {
  if (length(ind$numeric_cols) == 0 && length(ind$categorical_cols) == 0) return(ind)
  
  # Categorize existing gene outputs by type, restricted to tested genes
  gene_num <- character(0)
  gene_cat <- character(0)
  for (gene in ind$genes) {
    # Skip if this gene's output hasn't been evaluated yet
    if (!is.null(tested_gene_outputs) && !(gene$output_col %in% tested_gene_outputs)) {
      next
    }
    t_def <- evo_transformers[[gene$transformer_name]]
    out_type <- if (!is.null(t_def$output_type)) t_def$output_type else "numeric"
    if (out_type == "categorical") {
      gene_cat <- c(gene_cat, gene$output_col)
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
    avail_num <- c(ind$numeric_cols, gene_num_ex)
    avail_cat <- c(ind$categorical_cols, gene_cat)
    avail_for_type <- if (t_def$input_type == "numeric") avail_num
                      else if (t_def$input_type == "categorical") avail_cat
                      else c(avail_cat, avail_num)
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
        } else if (param_name == "comp_idx") {
          # For multi-component transformers, don't mutate comp_idx in isolation
          # as components are managed together during Add mutation
          old_val # no-op
        } else if (param_name == "component") {
          components <- c("year", "month", "day", "hour", "day_of_week", "weekend")
          candidates <- setdiff(components, old_val)
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
    # Add a random gene
    if (is.null(allowed_transformers)) {
      allowed_t <- names(evo_transformers)
    } else {
      allowed_t <- allowed_transformers
    }
    if (task == "multiclass") {
      allowed_t <- setdiff(allowed_t, c("target_encode"))
    } else {
      allowed_t <- setdiff(allowed_t, c("target_encode_multiclass"))
    }
    if (length(allowed_t) == 0) allowed_t <- names(evo_transformers)
    t_name <- sample(allowed_t, 1)
    t_def <- evo_transformers[[t_name]]
    
    avail_num <- c(ind$numeric_cols, gene_num)
    avail_cat <- c(ind$categorical_cols, gene_cat)
    
    # Select available columns based on input_type
    available_cols <- if (t_def$input_type == "numeric") {
      avail_num
    } else if (t_def$input_type == "categorical") {
      avail_cat
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
      
      for (comp in 1:C) {
        g <- create_gene(t_name, cols)
        g$params$comp_idx <- comp
        if (t_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
          g$params$k <- C
          if (!is.null(gini_threshold)) g$params$gini_threshold <- gini_threshold
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
  ind$genes <- topological_sort_genes(ind$genes, c(ind$numeric_cols, ind$categorical_cols))
  ind
}

#' Crossover two individuals
#'
#' @param ind1 Parent 1
#' @param ind2 Parent 2
#' @param verbose Logical. Whether to print crossover details.
#' @return An \code{evo_individual} child created by randomly sampling genes
#'   from both parents with duplicate gene outputs removed.
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
  
  create_individual(genes = child_genes, numeric_cols = ind1$numeric_cols, categorical_cols = ind1$categorical_cols)
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
  
  create_individual(genes = child_genes, numeric_cols = ind1$numeric_cols, categorical_cols = ind1$categorical_cols)
}
