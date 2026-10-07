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
#' @param raw_toggle_prob Numeric in \code{[0, 1]}. Probability that a mutation
#'   event toggles one or more raw input features in the individual's active
#'   mask rather than adding/modifying/removing a gene.  Default \code{0.15}.
#' @param recalculate_mask_prob Numeric in \code{[0, 1]}. Probability that a
#'   mutation event completely recalculates the active raw feature mask from
#'   scratch using feature importances.  Default \code{0.05}.
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
mutate <- function(ind, verbose = FALSE, force_add = FALSE, importances = numeric(0), temperature = 1.0, task = "classification", tested_gene_outputs = NULL, allowed_transformers = NULL, migrated_genes = list(), gene_migration_prob = 0.2, raw_toggle_prob = 0.15, recalculate_mask_prob = 0.05) {
  if (length(ind$all_numeric_cols) == 0 && length(ind$all_categorical_cols) == 0 && length(ind$all_datetime_cols) == 0) return(ind)
  
  if (!force_add) {
    r <- stats::runif(1)
    if (r < recalculate_mask_prob) {
      return(recalculate_mask(ind, importances = importances, temperature = temperature, verbose = verbose))
    } else if (r < recalculate_mask_prob + raw_toggle_prob) {
      return(toggle_raw_feature(ind, importances = importances, temperature = temperature, verbose = verbose))
    }
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
    
    active_raw_cols <- c(ind$numeric_cols, ind$categorical_cols, ind$datetime_cols)
    all_raw_cols <- c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols)
    
    weights <- sapply(cols, function(c) {
      is_active <- TRUE
      if (c %in% all_raw_cols) {
        is_active <- c %in% active_raw_cols
      }
      val <- if (is_active) {
        if (c %in% names(importances)) importances[[c]] else baseline
      } else {
        0.0
      }
      if (is.na(val) || !is.finite(val)) val <- if (is_active) baseline else 0.0
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
      # Invert: low importance -> high removal probability
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
        } else if (param_name == "scale") {
          candidates <- setdiff(c(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "phase") {
          candidates <- setdiff(c(0, round(pi / 6, 4), round(pi / 4, 4), round(pi / 3, 4)), old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "low_pct") {
          candidates <- setdiff(c(0.01, 0.02, 0.05), old_val)
          if (length(candidates) > 0) sample(candidates, 1) else old_val
        } else if (param_name == "high_pct") {
          candidates <- setdiff(c(0.95, 0.98, 0.99), old_val)
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
      avail_num <- c(ind$all_numeric_cols, gene_num)
      avail_cat <- c(ind$all_categorical_cols, gene_cat)
      avail_date <- c(ind$all_datetime_cols, gene_date)
      
      max_add_attempts <- if (force_add) 25L else 5L
      for (add_attempt in seq_len(max_add_attempts)) {
        t_name <- sample(allowed_t, 1)
        t_def <- evo_transformers[[t_name]]
        
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
        
        # If no columns available for this type, continue to next attempt
        if (t_def$input_type == "mixed") {
          if (length(avail_num) == 0 || length(avail_cat) == 0) next
        } else {
          if (length(available_cols) == 0) next
        }
        
        # Select random columns
        if (t_def$input_type == "mixed") {
          if (t_name %in% c("famd", "supervised_famd")) {
            num_cat <- min(sample(1:3, 1), length(avail_cat))
            num_num <- min(sample(1:5, 1), length(avail_num))
            cols <- c(weighted_sample(avail_cat, num_cat), weighted_sample(avail_num, num_num))
          } else if (t_name == "between_group_pca") {
            if (length(avail_num) < 2) next
            num_num <- min(sample(2:5, 1), length(avail_num))
            cols <- c(weighted_sample(avail_cat, 1), weighted_sample(avail_num, num_num))
          } else {
            col_cat <- weighted_sample(avail_cat, 1)
            col_num <- weighted_sample(avail_num, 1)
            cols <- c(col_cat, col_num)
          }
        } else if (t_def$type %in% c("unary", "supervised_unary")) {
          cols <- weighted_sample(available_cols, 1)
        } else if (t_def$type == "binary") {
          allow_rep <- if (t_name %in% c("subtract", "divide", "date_diff")) FALSE else TRUE
          if (!allow_rep && length(available_cols) < 2) next
          cols <- weighted_sample(available_cols, 2, replace = allow_rep)
        } else if (t_def$type == "multivariate") {
          if (length(available_cols) < 2) next
          max_cols <- if (t_name %in% c("add", "multiply")) {
            min(5, length(available_cols))
          } else if (t_name == "concat") {
            min(3, length(available_cols))
          } else {
            max(2, floor((1 - exp(-1)) * length(available_cols)))
          }
          num_cols <- if (max_cols == 2) 2 else sample(2:max_cols, 1)
          allow_rep <- if (!is.null(t_def$allow_replace)) t_def$allow_replace else FALSE
          if (!allow_rep) {
            sampled_cols <- weighted_sample(available_cols, num_cols, replace = TRUE)
            cols <- unique(sampled_cols)
            if (length(cols) < 2) {
              cols <- weighted_sample(available_cols, min(2, length(available_cols)), replace = FALSE)
            }
          } else {
            cols <- weighted_sample(available_cols, num_cols, replace = TRUE)
          }
        } else {
          cols <- weighted_sample(available_cols, 1)
        }
        
        # If the transformer is multi-component, add all components
        new_genes_to_add <- list()
        if (t_name %in% c("pca", "truncated_svd", "umap", "mca", "famd", "between_group_pca", "genie_centroid_dist", "lumbermark_centroid_dist", "supervised_bgpca", "supervised_mca", "supervised_famd", "fourier_basis")) {
          C <- if (t_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
            sample(2:5, 1)
          } else if (t_name %in% c("mca", "famd", "between_group_pca", "supervised_bgpca", "supervised_mca", "supervised_famd")) {
            sample(2:5, 1)
          } else if (t_name == "fourier_basis") {
            sample(c(4L, 6L, 8L), 1)
          } else {
            max(2L, as.integer(round(log2(length(cols)))))
          }
          
          gini_threshold <- if (t_name == "genie_centroid_dist") round(stats::runif(1, 0.1, 0.9), 2) else NULL
          n_neighbors <- if (t_name == "umap") max(2L, stats::rpois(1, 15)) else NULL
          dens_scale <- if (t_name == "umap") round(stats::runif(1, 0, 1), 2) else NULL
          fbr_scale <- if (t_name == "fourier_basis") sample(c(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), 1) else NULL
          fbr_phase <- if (t_name == "fourier_basis") sample(c(0, round(pi / 6, 4), round(pi / 4, 4), round(pi / 3, 4)), 1) else NULL
          
          for (comp in 1:C) {
            g <- create_gene(t_name, cols)
            g$params$comp_idx <- comp
            if (t_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
              g$params$k <- C
              if (!is.null(gini_threshold)) g$params$gini_threshold <- gini_threshold
            } else if (t_name == "umap") {
              g$params$n_neighbors <- n_neighbors
              g$params$dens_scale <- dens_scale
            } else if (t_name == "fourier_basis") {
              g$params$scale <- fbr_scale
              g$params$phase <- fbr_phase
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
          break
        }
      }
    }
  }
  ind$genes <- topological_sort_genes(ind$genes, c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols))
  ind
}
