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
  params <- list()
  if (transformer_name %in% c("pca", "truncated_svd", "umap")) {
    C <- max(2L, as.integer(round(log2(length(input_cols)))))
    params$comp_idx <- sample(1:C, 1)
    if (transformer_name == "umap") {
      params$n_neighbors <- max(2L, stats::rpois(1, 15))
      params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    }
  } else if (transformer_name %in% c("similarity_encode", "mca", "famd", "between_group_pca", "supervised_bgpca", "supervised_mca", "supervised_famd")) {
    params$comp_idx <- sample(1:5, 1)
  } else if (transformer_name == "minhash_encode") {
    params$comp_idx <- sample(1:8, 1)
  } else if (transformer_name == "gap_encode") {
    params$comp_idx <- sample(1:4, 1)
  } else if (transformer_name == "datetime_cyclic") {
    params$component <- sample(c("hour_sin", "hour_cos", "wday_sin", "wday_cos", "month_sin", "month_cos", "yday_sin", "yday_cos"), 1)
  } else if (transformer_name == "target_quantile_encode") {
    params$q <- sample(c(0.25, 0.5, 0.75), 1)
  } else if (transformer_name == "feature_hash") {
    params$num_bins <- sample(c(4, 8, 16, 32), 1)
    params$comp_idx <- sample(1:params$num_bins, 1)
  } else if (transformer_name %in% c("genie_centroid_dist", "lumbermark_centroid_dist")) {
    params$k <- sample(2:5, 1)
    params$comp_idx <- sample(1:params$k, 1)
    if (transformer_name == "genie_centroid_dist") {
      params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
    }
  } else if (transformer_name == "one_hot_encode") {
    params$comp_idx <- sample(1:6, 1)
  } else if (transformer_name == "genie") {
    params$k <- sample(2:5, 1)
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "umap_genie") {
    params$n_neighbors <- max(2L, stats::rpois(1, 15))
    params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    params$k <- sample(2:5, 1)
    params$gini_threshold <- round(stats::runif(1, 0.1, 0.9), 2)
  } else if (transformer_name == "umap_lumbermark") {
    params$n_neighbors <- max(2L, stats::rpois(1, 15))
    params$dens_scale <- round(stats::runif(1, 0, 1), 2)
    params$k <- sample(2:5, 1)
  } else if (transformer_name == "lumbermark") {
    params$k <- sample(2:5, 1)
  } else if (transformer_name %in% c("quantile_binning", "quantile_binning_cat")) {
    params$Q <- sample(3:10, 1)
  } else if (transformer_name %in% c("log_binning", "log_binning_cat")) {
    params$base <- sample(2:10, 1)
  } else if (transformer_name == "target_encode_multiclass") {
    params$comp_idx <- sample(1:5, 1)
  } else if (transformer_name == "datetime_extract") {
    params$component <- sample(c("year", "month", "day", "hour", "day_of_week", "weekend"), 1)
  } else if (transformer_name == "power") {
    params$p <- sample(c(0.5, 1/3, 2, 3), 1)
  } else if (transformer_name == "displaced_log") {
    params$displacement <- round(stats::runif(1, 10, 1000), 2)
  } else if (transformer_name == "groupby_quantile") {
    params$q <- sample(c(0.25, 0.75), 1)
  } else if (transformer_name == "fourier_basis") {
    params$comp_idx <- sample(1:8, 1)
    params$scale <- sample(c(0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0), 1)
    params$phase <- sample(c(0, round(pi / 6, 4), round(pi / 4, 4), round(pi / 3, 4)), 1)
  } else if (transformer_name == "smooth_clip") {
    params$low_pct <- sample(c(0.01, 0.02, 0.05), 1)
    params$high_pct <- sample(c(0.95, 0.98, 0.99), 1)
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
    for (i in seq_along(ind$genes)) {
      if (!isTRUE(keep_unsupervised) || is_supervised_transformer(ind$genes[[i]])) {
        ind$genes[[i]]$state <- NULL
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
