#' Apply feature engineering recipe to new data
#'
#' @param object An evo_recipe object
#' @param newdata A data.frame or data.table
#' @param ... Additional arguments
#' @export
predict.evo_recipe <- function(object, newdata, ...) {
  ind <- object$best_individual
  
  res <- apply_individual(ind, newdata, val_data = NULL, target_col = NULL)
  
  gene_cols <- if (length(ind$genes) > 0) vapply(ind$genes, function(g) g$output_col, character(1)) else character(0)
  features <- c(ind$numeric_cols, ind$categorical_cols, gene_cols)
  
  res$train[, features, with = FALSE]
}
