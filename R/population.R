#' Initialize a population
#'
#' @param pop_size Population size.
#' @param numeric_cols Vector of numeric column names.
#' @param categorical_cols Vector of categorical column names.
#' @param initial_genes Number of initial genes per individual.
#' @param task Task type ("classification", "regression", or "multiclass").
#' @export
initialize_population <- function(pop_size, numeric_cols, categorical_cols, initial_genes = 2, task = "classification") {
  pop <- list()
  for (i in 1:pop_size) {
    ind <- create_individual(genes = list(), numeric_cols = numeric_cols, categorical_cols = categorical_cols)
    # Reserve the first individual as a baseline (original features only)
    if (i > 1) {
      attempts <- 0
      while (length(ind$genes) < initial_genes && attempts < initial_genes * 10) {
        ind <- mutate(ind, force_add = TRUE, task = task, tested_gene_outputs = character(0))
        attempts <- attempts + 1
      }
    }
    pop[[i]] <- ind
  }
  pop
}
