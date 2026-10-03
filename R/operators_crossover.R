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
