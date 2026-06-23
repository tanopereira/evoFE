#' Initialize a population
#'
#' @param pop_size Population size.
#' @param numeric_cols Vector of numeric column names.
#' @param categorical_cols Vector of categorical column names.
#' @param datetime_cols Vector of datetime column names.
#' @param initial_genes Number of initial genes per individual.
#' @param task Task type ("classification", "regression", or "multiclass").
#' @param importances Optional numeric vector of feature importances.
#' @param allowed_transformers A character vector of allowed transformer names,
#'   or NULL/"all" to allow all.
#' @return A list of \code{evo_individual} objects of length \code{pop_size}.
#'   The first individual is a baseline with no genes; the remaining
#'   individuals each carry \code{initial_genes} randomly generated genes.
#' @export
initialize_population <- function(pop_size, numeric_cols, categorical_cols, datetime_cols = character(0), initial_genes = 2, task = "classification", importances = NULL, allowed_transformers = NULL) {
  pop <- list()
  for (i in 1:pop_size) {
    ind <- create_individual(genes = list(), numeric_cols = numeric_cols, categorical_cols = categorical_cols, datetime_cols = datetime_cols)
    # Reserve the first individual as a baseline (original features only)
    if (i > 1) {
      attempts <- 0
      while (length(ind$genes) < initial_genes && attempts < initial_genes * 10) {
        ind <- mutate(ind, force_add = TRUE, importances = importances, task = task, tested_gene_outputs = character(0), allowed_transformers = allowed_transformers)
        attempts <- attempts + 1
      }
    }
    pop[[i]] <- ind
  }
  pop
}
