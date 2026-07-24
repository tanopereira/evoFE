sample_active_mask <- function(cols, importances, temperature, fallback_rate = 0.7) {
  if (length(cols) == 0) return(character(0))
  
  has_imps <- length(importances) > 0 && any(importances > 0)
  
  if (has_imps) {
    threshold <- 1.0 / length(cols)
    vals <- importances[cols]
    vals[is.na(vals) | !is.finite(vals)] <- 0.0
    probs <- 1.0 / (1.0 + exp(-(vals - threshold) / temperature))
    keep <- stats::runif(length(cols)) < probs
  } else {
    keep <- stats::runif(length(cols)) < fallback_rate
  }
  cols[keep]
}

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
#' @param mask_temp_factor Numeric > 0. Temperature scaling factor applied to
#'   feature importances during active mask initialization.  Higher values
#'   flatten the importance distribution (more uniform sampling); lower values
#'   concentrate sampling on the highest-importance features.  Default
#'   \code{0.5}.
#' @return A list of \code{evo_individual} objects of length \code{pop_size}.
#'   The first individual is a baseline with no genes; the remaining
#'   individuals each carry \code{initial_genes} randomly generated genes.
#' @export
initialize_population <- function(pop_size, numeric_cols, categorical_cols, datetime_cols = character(0), initial_genes = 2, task = "classification", importances = NULL, allowed_transformers = NULL, mask_temp_factor = 0.5) {
  pop <- list()
  for (i in 1:pop_size) {
    if (i == 1) {
      ind <- create_individual(
        genes = list(),
        numeric_cols = numeric_cols,
        categorical_cols = categorical_cols,
        datetime_cols = datetime_cols,
        all_numeric_cols = numeric_cols,
        all_categorical_cols = categorical_cols,
        all_datetime_cols = datetime_cols
      )
    } else {
      n_cols <- length(numeric_cols) + length(categorical_cols) + length(datetime_cols)
      threshold <- 1.0 / max(1.0, n_cols)
      temperature <- mask_temp_factor * threshold
      
      init_num <- sample_active_mask(numeric_cols, importances, temperature)
      init_cat <- sample_active_mask(categorical_cols, importances, temperature)
      init_date <- sample_active_mask(datetime_cols, importances, temperature)
      
      total_active <- length(init_num) + length(init_cat) + length(init_date)
      total_avail <- length(numeric_cols) + length(categorical_cols) + length(datetime_cols)
      min_active <- if (total_avail >= 2) 2 else 1
      
      if (total_active < min_active) {
        all_cols <- c(numeric_cols, categorical_cols, datetime_cols)
        active_cols <- c(init_num, init_cat, init_date)
        inactive_cols <- setdiff(all_cols, active_cols)
        if (length(inactive_cols) > 0) {
          to_activate <- sample(inactive_cols, min(length(inactive_cols), min_active - total_active))
          for (col in to_activate) {
            if (col %in% numeric_cols) init_num <- c(init_num, col)
            else if (col %in% categorical_cols) init_cat <- c(init_cat, col)
            else if (col %in% datetime_cols) init_date <- c(init_date, col)
          }
        }
      }
      
      ind <- create_individual(
        genes = list(),
        numeric_cols = init_num,
        categorical_cols = init_cat,
        datetime_cols = init_date,
        all_numeric_cols = numeric_cols,
        all_categorical_cols = categorical_cols,
        all_datetime_cols = datetime_cols
      )
    }
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
