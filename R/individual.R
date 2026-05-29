#' Create a single gene
#'
#' @param transformer_name Name of the transformer
#' @param input_cols Vector of input column names
#' @export
create_gene <- function(transformer_name, input_cols) {
  transformer <- evo_transformers[[transformer_name]]
  params <- list()
  if (transformer_name %in% c("pca", "truncated_svd")) {
    params$comp_idx <- sample(1:3, 1)
  } else if (transformer_name == "umap") {
    params$comp_idx <- sample(1:2, 1)
  } else if (transformer_name == "one_hot_encode") {
    params$comp_idx <- sample(1:6, 1)
  } else if (transformer_name == "genie") {
    params$k <- sample(2:5, 1)
  } else if (transformer_name == "lumbermark") {
    params$k <- sample(2:5, 1)
  } else if (transformer_name %in% c("quantile_binning", "quantile_binning_cat")) {
    params$Q <- sample(3:10, 1)
  } else if (transformer_name %in% c("log_binning", "log_binning_cat")) {
    params$base <- sample(2:10, 1)
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
#' @export
gene_to_formula <- function(gene) {
  if (gene$transformer_name == "one_hot_encode") {
    comp_str <- if (gene$params$comp_idx == 6) "other" else as.character(gene$params$comp_idx)
    sprintf("ohe_%s(%s)", comp_str, paste(gene$input_cols, collapse = ", "))
  } else if (!is.null(gene$params$comp_idx)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$comp_idx, paste(gene$input_cols, collapse = ", "))
  } else if (!is.null(gene$params$Q)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$Q, paste(gene$input_cols, collapse = ", "))
  } else if (!is.null(gene$params$base)) {
    sprintf("%s%d(%s)", gene$transformer_name, gene$params$base, paste(gene$input_cols, collapse = ", "))
  } else {
    sprintf("%s(%s)", gene$transformer_name, paste(gene$input_cols, collapse = ", "))
  }
}

#' Convert a gene to a formula string for state caching (ignoring component index)
#'
#' @param gene A gene list
#' @export
gene_to_state_formula <- function(gene) {
  if (gene$transformer_name %in% c("pca", "truncated_svd", "umap")) {
    sprintf("%s(%s)", gene$transformer_name, paste(gene$input_cols, collapse = ", "))
  } else {
    gene_to_formula(gene)
  }
}

#' Convert an individual to a recipe string of formulas
#'
#' @param ind An evo_individual
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
#' @export
mutate <- function(ind, verbose = FALSE, force_add = FALSE, importances = numeric(0), temperature = 1.0, task = "classification", tested_gene_outputs = NULL) {
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
    
    # Baseline for missing features: minimum of known, or 0.01
    baseline <- if (length(importances) > 0) min(importances) else 0.01
    
    weights <- sapply(cols, function(c) {
      val <- if (c %in% names(importances)) importances[[c]] else baseline
      exp(val / temperature)
    })
    
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
    # Remove a random gene
    idx <- sample(seq_along(ind$genes), 1)
    removed <- ind$genes[[idx]]
    ind$genes <- ind$genes[-idx]
    ind$fitness <- NA_real_
    if (verbose) {
      message(sprintf("    [Mutation] Removed gene: %s (%s)", removed$output_col, gene_to_formula(removed)))
    }
  } else if (mut_type == 2) {
    # Modify an existing multivariate gene
    multi_idx <- which(sapply(ind$genes, function(g) evo_transformers[[g$transformer_name]]$type == "multivariate"))
    if (length(multi_idx) > 0) {
      idx <- if (length(multi_idx) == 1) multi_idx else sample(multi_idx, 1)
      gene_to_mod <- ind$genes[[idx]]
      
      gene_num_ex <- setdiff(gene_num, gene_to_mod$output_col)
      avail_num <- c(ind$numeric_cols, gene_num_ex)
      unused <- setdiff(avail_num, gene_to_mod$input_cols)
      
      t_name <- gene_to_mod$transformer_name
      max_cols <- if (t_name %in% c("add", "multiply")) {
        min(5, length(avail_num))
      } else {
        max(2, floor((1 - exp(-1)) * length(avail_num)))
      }
      if (length(unused) > 0 && length(gene_to_mod$input_cols) < max_cols) {
        new_col <- weighted_sample(unused, 1)
        old_formula <- gene_to_formula(gene_to_mod)
        gene_to_mod$input_cols <- c(gene_to_mod$input_cols, new_col)
        
        t_def <- evo_transformers[[gene_to_mod$transformer_name]]
        gene_to_mod$output_col <- t_def$name_generator(gene_to_mod)
        
        # Check if modified gene conflicts with existing output names
        existing_out <- sapply(ind$genes[-idx], function(g) g$output_col)
        if (!(gene_to_mod$output_col %in% existing_out)) {
          ind$genes[[idx]] <- gene_to_mod
          ind$fitness <- NA_real_
          if (verbose) {
            message(sprintf("    [Mutation] Modified gene: added '%s' to %s -> %s", 
                            new_col, old_formula, gene_to_formula(gene_to_mod)))
          }
        } else {
          mut_type <- 3 # Conflict, fallback to add
        }
      } else {
        mut_type <- 3 # No unused columns or limit reached, fallback to add
      }
    } else {
      mut_type <- 3 # No multivariate genes, fallback to add
    }
  }
  
  if (mut_type == 3) {
    # Add a random gene
    allowed_transformers <- names(evo_transformers)
    if (task == "multiclass") {
      allowed_transformers <- setdiff(allowed_transformers, "target_encode")
    }
    t_name <- sample(allowed_transformers, 1)
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
    if (t_name %in% c("pca", "truncated_svd")) {
      for (comp in 1:3) {
        g <- create_gene(t_name, cols)
        g$params$comp_idx <- comp
        g$output_col <- t_def$name_generator(g)
        new_genes_to_add <- c(new_genes_to_add, list(g))
      }
    } else if (t_name == "umap") {
      for (comp in 1:2) {
        g <- create_gene(t_name, cols)
        g$params$comp_idx <- comp
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
