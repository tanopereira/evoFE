#' Create a single gene
#'
#' @param transformer_name Name of the transformer
#' @param input_cols Vector of input column names
#' @return A gene list with elements \code{transformer_name}, \code{input_cols},
#'   \code{params} (transformer-specific parameters), \code{state} (\code{NULL}
#'   until fitted), and \code{output_col} (auto-generated column name).
#' @export
create_gene <- function(transformer_name, input_cols) {
  if (!transformer_name %in% names(evo_transformers)) {
    stop(sprintf(
      "Unknown transformer '%s'. Registered transformers are: %s",
      transformer_name, paste(names(evo_transformers), collapse = ", ")
    ))
  }
  transformer <- evo_transformers[[transformer_name]]
  params <- sample_transformer_params(transformer_name, input_cols)
  gene <- list(
    transformer_name = transformer_name,
    input_cols = input_cols,
    params = params,
    state = NULL # Populated during fitness evaluation if stateful
  )
  gene$output_col <- transformer$name_generator(gene)
  gene
}

#' Create an individual
#'
#' @param genes List of genes
#' @param numeric_cols Vector of active numeric column names visible to this individual.
#' @param categorical_cols Vector of active categorical column names visible to this individual.
#' @param datetime_cols Vector of active datetime column names visible to this individual.
#' @param all_numeric_cols Vector of all numeric column names in the dataset
#'   (superset of \code{numeric_cols}).  Used to initialize the full feature
#'   pool for mask toggling.  Defaults to \code{numeric_cols} when \code{NULL}.
#' @param all_categorical_cols Vector of all categorical column names in the
#'   dataset.  Defaults to \code{categorical_cols} when \code{NULL}.
#' @param all_datetime_cols Vector of all datetime column names in the dataset.
#'   Defaults to \code{datetime_cols} when \code{NULL}.
#' @return An \code{evo_individual} S3 object:
#'   a list with elements \code{genes} (topologically sorted),
#'   \code{numeric_cols}, \code{categorical_cols}, and \code{fitness}
#'   (initialised to \code{NA_real_}).
#' @examples
#' \donttest{
#' ind <- create_individual(
#'   genes = list(),
#'   numeric_cols = c("a", "b"),
#'   categorical_cols = c("c")
#' )
#' print(ind)
#' }
#' @export
create_individual <- function(genes = list(), numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                              all_numeric_cols = NULL, all_categorical_cols = NULL, all_datetime_cols = NULL) {
  if (is.null(all_numeric_cols)) all_numeric_cols <- numeric_cols
  if (is.null(all_categorical_cols)) all_categorical_cols <- categorical_cols
  if (is.null(all_datetime_cols)) all_datetime_cols <- datetime_cols
  
  original_cols <- c(all_numeric_cols, all_categorical_cols, all_datetime_cols)
  sorted_genes <- topological_sort_genes(genes, original_cols)
  structure(
    list(
      genes = sorted_genes,
      numeric_cols = numeric_cols,
      categorical_cols = categorical_cols,
      datetime_cols = datetime_cols,
      all_numeric_cols = all_numeric_cols,
      all_categorical_cols = all_categorical_cols,
      all_datetime_cols = all_datetime_cols,
      raw_fitness = NA_real_,
      penalty = 0.0,
      fitness = NA_real_
    ),
    class = "evo_individual"
  )
}

#' Strip fitted transformer state and fitness from an individual
#'
#' Resets fitness, raw_fitness, predictions, and clears fitted states
#' from supervised genes (or all genes if keep_unsupervised is FALSE)
#' so that the individual can be evaluated cleanly on a new data split
#' or fold without data leakage. Unsupervised transformer states (e.g. PCA,
#' UMAP, clustering) carry no target leakage and are preserved when
#' keep_unsupervised = TRUE.
#'
#' @param ind An \code{evo_individual} object.
#' @param keep_unsupervised Logical. If TRUE (default), fitted states of
#'   unsupervised transformers are preserved since they contain no target
#'   information and do not cause data leakage across splits.
#' @return The modified \code{evo_individual}.
#' @export
strip_individual_state <- function(ind, keep_unsupervised = getOption("evoFE.global_unsupervised", TRUE)) {
  ind$fitness <- NA_real_
  ind$raw_fitness <- NA_real_
  ind$penalty <- 0.0
  ind$val_preds <- NULL
  ind$y_val <- NULL
  if (length(ind$genes) > 0) {
    tainted_cols <- character(0)
    for (i in seq_along(ind$genes)) {
      g <- ind$genes[[i]]
      is_tainted <- is_supervised_transformer(g) || any(g$input_cols %in% tainted_cols)
      if (is_tainted && !is.null(g$output_col)) {
        tainted_cols <- c(tainted_cols, g$output_col)
      }
      if (!isTRUE(keep_unsupervised) || is_tainted) {
        ind$genes[[i]]$state <- NULL
      }
    }
  }
  ind
}

#' Strip training-only leave-one-out vectors from fitted gene states
#' @keywords internal
#' @noRd
strip_loo_state <- function(ind) {
  if (length(ind$genes) > 0) {
    for (i in seq_along(ind$genes)) {
      st <- ind$genes[[i]]$state
      if (is.list(st)) {
        st$loo_train <- NULL
        st$loo_train_list <- NULL
        ind$genes[[i]]$state <- st
      }
    }
  }
  ind
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
#'   feature importances during active mask initialization when \code{mask_raw_at_init = TRUE}.
#'   Higher values flatten the importance distribution (more uniform sampling); lower values
#'   concentrate sampling on the highest-importance features. Default \code{0.5}.
#' @param mask_raw_at_init Logical. If TRUE, samples active raw feature masks during population
#'   initialization based on baseline importances. Default \code{FALSE} (every individual starts with
#'   100\% of raw features active so candidate genes are tested fairly on the full feature set).
#' @return A list of \code{evo_individual} objects of length \code{pop_size}.
#'   The first individual is a baseline with no genes; the remaining
#'   individuals each carry \code{initial_genes} randomly generated genes.
#' @export
initialize_population <- function(pop_size, numeric_cols, categorical_cols, datetime_cols = character(0),
                                  initial_genes = 2, task = "classification", importances = NULL,
                                  allowed_transformers = NULL, mask_temp_factor = 0.5,
                                  mask_raw_at_init = FALSE) {
  pop <- list()
  for (i in 1:pop_size) {
    if (i == 1 || !isTRUE(mask_raw_at_init)) {
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
