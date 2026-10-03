#' Compute Complexity Penalty for an Individual
#'
#' Computes the parsimony complexity penalty using Bayesian Information Criterion (BIC)
#' scaling (\code{ln(N) / (2N)}) and optional dynamic convergence scaling based on the
#' relative gap to the metric ceiling.
#'
#' @param n_genes Integer. Number of evolved genes in the individual.
#' @param n_samples Integer. Number of dataset samples (rows).
#' @param running_best_fitness Numeric. Current running best fitness in the population/island.
#' @param baseline_fitness Numeric. Generation 0 baseline fitness (raw features only).
#' @param metric Character or function. Metric being optimized.
#' @param task Character. "classification", "multiclass", or "regression".
#' @param complexity_penalty Numeric. Dimensionless penalty multiplier (default 0).
#' @param complexity_mode Character. "bic_dynamic" (default), "bic", "pac_bayes_dynamic", "pac_bayes", or "none".
#' @param complexity_floor Numeric. Minimum safety floor factor for dynamic penalties (default 0.20, representing 20\% of base penalty).
#' @param complexity_target Character. "all_features" (default, penalizes total active features) or "genes" (penalizes only derived genes).
#' @param n_features Optional integer. Total number of active features (active raw features plus genes). If NULL, defaults to n_genes.
#' @param epsilon_floor Deprecated alias for \code{complexity_floor}.
#' @return Non-negative numeric penalty to subtract from raw fitness.
#' @export
compute_complexity_penalty <- function(n_genes,
                                       n_samples,
                                       running_best_fitness = NULL,
                                       baseline_fitness = NULL,
                                       metric = "default",
                                       task = "classification",
                                       complexity_penalty = 0,
                                       complexity_mode = "bic_dynamic",
                                       complexity_floor = 0.20,
                                       complexity_target = "all_features",
                                       n_features = NULL,
                                       epsilon_floor = 0.20) {
  floor_val <- if (!missing(complexity_floor)) complexity_floor else epsilon_floor
  target_mode <- if (identical(complexity_target, "genes") || identical(complexity_target, "gene_only")) "genes" else "all_features"
  k <- if (target_mode == "genes") {
    n_genes
  } else {
    if (!is.null(n_features)) n_features else n_genes
  }

  if (complexity_penalty <= 0 || k <= 0 || complexity_mode == "none") {
    return(0)
  }

  n_samples <- as.numeric(n_samples)
  if (is.na(n_samples) || n_samples < 2) {
    n_samples <- 2
  }

  # Base penalty factor per feature/gene
  is_dynamic <- grepl("_dynamic$", complexity_mode)
  base_mode <- sub("_dynamic$", "", complexity_mode)

  base_factor <- switch(base_mode,
    "pac_bayes" = complexity_penalty / (2 * sqrt(n_samples)),
    "bic"       = complexity_penalty * (log(n_samples) / (2 * n_samples)),
    0
  )

  if (base_factor <= 0) {
    return(0)
  }

  p <- if (is_dynamic) {
    # Dynamic penalty mode: scale by normalized gap to ideal
    ideal <- if (task %in% c("classification", "multiclass")) 1.0 else 0.0

    if (is.null(baseline_fitness) || is.null(running_best_fitness) ||
      is.na(baseline_fitness) || is.na(running_best_fitness) ||
      !is.finite(baseline_fitness) || !is.finite(running_best_fitness)) {
      base_factor
    } else {
      denom <- ideal - baseline_fitness
      numer <- ideal - running_best_fitness
      gap_ratio <- if (abs(denom) <= 1e-12) 1.0 else numer / denom
      dynamic_factor <- max(floor_val, min(1.0, gap_ratio))
      base_factor * dynamic_factor
    }
  } else {
    base_factor
  }

  # Compound penalty across k features: (1 + p)^k - 1
  # Ensures adding k features simultaneously (e.g. multi-dimensional UMAP / PCA embeddings)
  # has the exact same compound error hurdle as sequential single-feature additions.
  expm1(k * log1p(p))
}
