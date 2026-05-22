#' Initialize a population
#' @export
initialize_population <- function(pop_size, numeric_cols, categorical_cols, initial_genes = 2) {
  pop <- list()
  for (i in 1:pop_size) {
    ind <- create_individual(genes = list(), numeric_cols = numeric_cols, categorical_cols = categorical_cols)
    # Reserve the first individual as a baseline (original features only)
    if (i > 1) {
      attempts <- 0
      while (length(ind$genes) < initial_genes && attempts < initial_genes * 10) {
        ind <- mutate(ind, force_add = TRUE)
        attempts <- attempts + 1
      }
    }
    pop[[i]] <- ind
  }
  pop
}
