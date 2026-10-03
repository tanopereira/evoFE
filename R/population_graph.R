#' Topological sort of genes based on available columns
#'
#' @param genes A list of genes
#' @param original_cols Vector of original column names
#' @return A list of topologically sorted genes
#' @noRd
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
