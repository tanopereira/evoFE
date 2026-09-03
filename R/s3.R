#' Print an evo_recipe object
#'
#' Prints a human-readable summary of the evolutionary feature engineering recipe.
#'
#' @param x An \code{evo_recipe} object.
#' @param ... Additional arguments (currently unused).
#' @return Invisible \code{x}. Called for its side effect of printing the recipe overview.
#' @importFrom stats predict
#' @export
print.evo_recipe <- function(x, ...) {
  cat("An evoFE Recipe\n")
  cat(sprintf("  Evaluator:    %s\n", x$evaluator))
  cat(sprintf("  Task:         %s\n", x$task))
  if (!is.null(x$metric)) {
    cat(sprintf("  Metric:       %s\n", x$metric))
  }
  raw_fit <- x$best_individual$raw_fitness
  pen_val <- x$best_individual$penalty
  if (!is.null(raw_fit) && !is.na(raw_fit)) {
    if (!is.null(pen_val) && is.finite(pen_val) && pen_val > 0) {
      cat(sprintf("  Validation Score: %.4f (Penalized Selection Fitness: %.4f)\n", raw_fit, x$best_individual$fitness))
    } else {
      cat(sprintf("  Validation Score: %.4f\n", raw_fit))
    }
  } else {
    cat(sprintf("  Best Fitness:     %.4f\n", x$best_individual$fitness))
  }
  if (!is.null(x$baseline_fitness) && is.finite(x$baseline_fitness)) {
    gain <- if (!is.null(x$improvement)) x$improvement else (x$best_individual$fitness - x$baseline_fitness)
    headroom_pct <- if (!is.null(x$headroom_closed)) x$headroom_closed * 100 else {
      ideal <- if (x$task %in% c("classification", "multiclass")) 1.0 else 0.0
      denom <- ideal - x$baseline_fitness
      if (abs(denom) > 1e-6) (gain / denom) * 100 else 0.0
    }
    cat(sprintf("  Baseline Score:   %.4f  (Gain: %+.4f | Headroom Closed: %+.1f%%)\n",
                x$baseline_fitness, gain, headroom_pct))
  }
  
  gene_count <- length(x$best_individual$genes)
  cat(sprintf("  Evolved Features: %d\n", gene_count))
  if (gene_count > 0) {
    cat("  Winning Recipe:\n")
    formulas <- sapply(x$best_individual$genes, gene_to_formula)
    for (i in seq_along(formulas)) {
      cat(sprintf("    [%d] %s -> %s\n", i, formulas[i], x$best_individual$genes[[i]]$output_col))
    }
  } else {
    cat("  Winning Recipe: [Original features only]\n")
  }
  invisible(x)
}

#' Summary of an evo_recipe object
#'
#' Computes and formats a detailed summary of the evolutionary feature engineering recipe.
#'
#' @param object An \code{evo_recipe} object.
#' @param ... Additional arguments (currently unused).
#' @return An object of class \code{summary_evo_recipe} containing detailed recipe statistics.
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   seed = 42,
#'   verbose = FALSE
#' )
#'
#' # Print the recipe overview
#' print(recipe)
#'
#' # Inspect a detailed summary
#' recipe_summary <- summary(recipe)
#' print(recipe_summary)
#' }
#' @export
summary.evo_recipe <- function(object, ...) {
  res <- list(
    evaluator = object$evaluator,
    task = object$task,
    metric = object$metric,
    raw_fitness = object$best_individual$raw_fitness,
    penalty = object$best_individual$penalty,
    best_fitness = object$best_individual$fitness,
    baseline_fitness = object$baseline_fitness,
    improvement = object$improvement,
    headroom_closed = object$headroom_closed,
    island_baselines = object$island_baselines,
    island_improvements = object$island_improvements,
    island_headroom_closed = object$island_headroom_closed,
    holdout_fitness = object$best_individual$holdout_fitness,
    search_gap = object$search_gap,
    num_genes = length(object$best_individual$genes),
    genes_summary = data.frame(
      Transformer = character(),
      Inputs = character(),
      Output = character(),
      stringsAsFactors = FALSE
    )
  )
  
  if (length(object$best_individual$genes) > 0) {
    res$genes_summary <- data.frame(
      Transformer = sapply(object$best_individual$genes, function(g) g$transformer_name),
      Inputs = sapply(object$best_individual$genes, function(g) paste(g$input_cols, collapse = ", ")),
      Output = sapply(object$best_individual$genes, function(g) g$output_col),
      stringsAsFactors = FALSE
    )
  }
  
  class(res) <- "summary_evo_recipe"
  res
}

#' Print summary of an evo_recipe object
#'
#' Prints the summary details of the evolutionary feature engineering recipe.
#'
#' @param x A \code{summary_evo_recipe} object.
#' @param ... Additional arguments (currently unused).
#' @return Invisible \code{x}. Called for its side effect of printing the recipe summary.
#' @export
print.summary_evo_recipe <- function(x, ...) {
  cat("=== Evolutionary Feature Engineering Summary ===\n")
  cat(sprintf("ML Evaluator:             %s\n", x$evaluator))
  cat(sprintf("Task type:                %s\n", x$task))
  if (!is.null(x$metric)) {
    cat(sprintf("Optimization Metric:      %s\n", x$metric))
  }
  if (!is.null(x$baseline_fitness) && is.finite(x$baseline_fitness)) {
    cat(sprintf("Baseline Metric Score:    %.6f\n", x$baseline_fitness))
  }
  if (!is.null(x$raw_fitness) && !is.na(x$raw_fitness)) {
    cat(sprintf("Validation Metric Score:  %.6f\n", x$raw_fitness))
    if (!is.null(x$penalty) && is.finite(x$penalty) && x$penalty > 0) {
      cat(sprintf("Complexity Penalty:       %.6f\n", x$penalty))
      cat(sprintf("Selection Fitness:        %.6f\n", x$best_fitness))
    }
  } else {
    cat(sprintf("Best CV/Split Fitness:    %.6f\n", x$best_fitness))
  }
  if (!is.null(x$baseline_fitness) && is.finite(x$baseline_fitness)) {
    gain <- if (!is.null(x$improvement)) x$improvement else (x$best_fitness - x$baseline_fitness)
    headroom_pct <- if (!is.null(x$headroom_closed)) x$headroom_closed * 100 else {
      ideal <- if (x$task %in% c("classification", "multiclass")) 1.0 else 0.0
      denom <- ideal - x$baseline_fitness
      if (abs(denom) > 1e-6) (gain / denom) * 100 else 0.0
    }
    cat(sprintf("Gain over Baseline:       %+.6f (Headroom Closed: %+.1f%%)\n", gain, headroom_pct))
  }
  if (!is.null(x$holdout_fitness) && !is.na(x$holdout_fitness)) {
    cat(sprintf("Best Holdout Fitness:     %.6f\n", x$holdout_fitness))
    if (!is.null(x$search_gap) && is.finite(x$search_gap)) {
      cat(sprintf("Search Gap (Holdout-Val): %+.6f\n", x$search_gap))
    }
  }
  if (!is.null(x$island_baselines) && length(x$island_baselines) > 1) {
    cat("\nIsland Baseline & Headroom Breakdown (Migration Drivers):\n")
    best_vals <- if (!is.null(x$island_improvements)) sprintf("%.4f", x$island_baselines + x$island_improvements) else rep("-", length(x$island_baselines))
    gain_vals <- if (!is.null(x$island_improvements)) sprintf("%+.4f", x$island_improvements) else rep("-", length(x$island_baselines))
    hd_vals <- if (!is.null(x$island_headroom_closed)) sprintf("%+.1f%%", x$island_headroom_closed * 100) else rep("-", length(x$island_baselines))
    df_islands <- data.frame(
      Island = seq_along(x$island_baselines),
      Baseline = sprintf("%.4f", x$island_baselines),
      Best = best_vals,
      Gain = gain_vals,
      Headroom_Closed = hd_vals,
      stringsAsFactors = FALSE
    )
    print(df_islands, row.names = FALSE)
  }
  cat(sprintf("\nNumber of Evolved Features: %d\n\n", x$num_genes))
  
  if (x$num_genes > 0) {
    cat("Feature Transformation Details:\n")
    print(x$genes_summary, row.names = FALSE)
  } else {
    cat("No feature transformations evolved. The model uses original features.\n")
  }
  invisible(x)
}

#' Plot an evo_recipe object
#'
#' Plots either the fitness trajectory over generations or the feature importances of the best individual.
#'
#' @param x An \code{evo_recipe} object.
#' @param type Character string, either \code{"fitness"} (default) to plot the fitness trajectory,
#'   or \code{"importance"} to plot a bar chart of the top feature importances of the winning model.
#' @param ... Additional arguments passed to \code{plot} or \code{barplot}.
#' @return Invisible NULL. Called for its side effect of plotting the fitness curve or feature importances.
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   cv_folds = 2,
#'   seed = 42,
#'   verbose = FALSE
#' )
#'
#' # Plot the fitness curve
#' plot(recipe, type = "fitness")
#'
#' # Plot feature importances
#' plot(recipe, type = "importance")
#' }
#' @export
plot.evo_recipe <- function(x, type = "fitness", ...) {
  type <- match.arg(type, c("fitness", "importance"))
  
  if (type == "fitness") {
    if (is.null(x$fitness_history) || length(x$fitness_history) == 0) {
      stop("No fitness history available in this recipe.")
    }
    y_vals  <- x$fitness_history
    x_vals  <- seq_along(y_vals)
    best_g  <- which.max(y_vals)
    baseline <- if (!is.null(x$baseline_fitness) && is.finite(x$baseline_fitness)) x$baseline_fitness else y_vals[1]
    total_gain <- y_vals[best_g] - baseline
    ideal <- if (identical(x$task, "regression")) 0.0 else 1.0
    denom <- ideal - baseline
    headroom_pct <- if (abs(denom) > 1e-6) (total_gain / denom) * 100 else 0.0
    gain_lbl <- if (!is.null(x$baseline_fitness) && is.finite(x$baseline_fitness)) "Gain vs Base" else "Gain vs Gen 1"
    subtitle <- sprintf("Generations: %d  |  Best: %.4f  |  %s: %+.4f (Headroom: %+.1f%%)",
                        length(y_vals), y_vals[best_g], gain_lbl, total_gain, headroom_pct)

    y_range <- range(c(y_vals, baseline), na.rm = TRUE)
    y_pad   <- max(0.002, diff(y_range) * 0.12)
    ylim    <- c(y_range[1] - y_pad, y_range[2] + y_pad)

    graphics::plot(
      x_vals, y_vals,
      type = "n",
      xlab = "Generation", ylab = "Best Fitness",
      main = "Evolutionary Feature Engineering \u2014 Fitness Curve",
      sub  = subtitle,
      ylim = ylim,
      xaxt = "n",
      ...
    )
    # Integer x-axis ticks
    graphics::axis(1, at = x_vals, labels = x_vals)
    graphics::grid(nx = NA, ny = NULL, lty = 2, col = "#cccccc")

    # Shaded improvement polygon (between baseline and curve)
    poly_x <- c(x_vals, rev(x_vals))
    poly_y <- c(y_vals, rep(baseline, length(x_vals)))
    graphics::polygon(poly_x, poly_y, col = "#cce5ff", border = NA)

    # Baseline dashed line
    graphics::abline(h = baseline, lty = 2, col = "#888888", lwd = 1.2)

    # Fitness line
    graphics::lines(x_vals, y_vals, col = "#1a6fcc", lwd = 2)
    graphics::points(x_vals, y_vals, pch = 19, col = "#1a6fcc", cex = 0.9)

    # Highlight best generation
    graphics::points(best_g, y_vals[best_g], pch = 18, col = "#e6a817", cex = 1.8)
    graphics::text(
      best_g, y_vals[best_g],
      labels = sprintf(" Gen %d\n %.4f", best_g, y_vals[best_g]),
      pos = if (best_g > length(y_vals) / 2) 2 else 4,
      col = "#7a4400", cex = 0.78
    )
  } else if (type == "importance") {
    imp <- x$best_individual$importances
    if (is.null(imp) || length(imp) == 0) {
      message("No feature importances available.")
      return(invisible(NULL))
    }
    
    imp_sorted <- sort(imp, decreasing = TRUE)
    if (length(imp_sorted) > 15) {
      imp_sorted <- imp_sorted[1:15]
    }
    
    old_mar <- graphics::par()$mar
    on.exit(graphics::par(mar = old_mar))
    graphics::par(mar = c(5, 12, 4, 2) + 0.1)
    
    graphics::barplot(
      rev(imp_sorted), horiz = TRUE, las = 1, col = "#2ca02c",
      xlab = "Importance (Gain)",
      main = "Top Feature Importances",
      ...
    )
  }
  invisible(NULL)
}

#' Print an evo_ensemble object
#'
#' Prints a human-readable summary of the Caruana island ensemble.
#'
#' @param x An \code{evo_ensemble} object.
#' @param ... Additional arguments (currently unused).
#' @return Invisible \code{x}. Called for its side effect of printing the ensemble overview.
#' @export
print.evo_ensemble <- function(x, ...) {
  method <- if (!is.null(x$method)) x$method else "caruana"
  cat(sprintf("An evoFE Island Ensemble (%s)\n", if (method == "stack") "Stacked" else "Caruana"))
  cat(sprintf("  Evaluator:            %s\n", x$evaluator))
  cat(sprintf("  Task:                 %s\n", x$task))
  if (!is.null(x$metric)) {
    cat(sprintf("  Metric:               %s\n", x$metric))
  }
  cat(sprintf("  Single Best Fitness:  %.4f\n", x$single_best_fitness))
  cat(sprintf("  Ensemble Fitness:     %.4f\n", x$ensemble_val_fitness))
  if (method == "stack" && !is.null(x$stack_cv_fitness)) {
    cat(sprintf("  Stack CV Fitness:     %.4f  (honest nested estimate)\n", x$stack_cv_fitness))
  }

  active_names <- names(x$weights[x$weights > 0])
  cat(sprintf("  Active Islands:       %d / %d\n", length(active_names), length(x$weights)))

  if (length(active_names) > 0) {
    cat("  Selected Island Weights:\n")
    for (name in active_names) {
      w <- x$weights[[name]]
      rec <- x$active_recipes[[name]]
      n_genes <- if (!is.null(rec$genes)) length(rec$genes) else 0
      cat(sprintf("    [%s] Weight: %5.1f%% | Evolved Features: %d\n", name, w * 100, n_genes))
    }
  }
  invisible(x)
}

#' Summary of an evo_ensemble object
#'
#' Computes and formats a detailed summary of the Caruana island ensemble.
#'
#' @param object An \code{evo_ensemble} object.
#' @param ... Additional arguments (currently unused).
#' @return An object of class \code{summary_evo_ensemble} containing detailed ensemble statistics.
#' @export
summary.evo_ensemble <- function(object, ...) {
  active_names <- names(object$weights[object$weights > 0])

  weights_df <- data.frame(
    Island = active_names,
    Weight = as.numeric(object$weights[active_names]),
    Num_Genes = vapply(active_names, function(name) {
      rec <- object$active_recipes[[name]]
      if (!is.null(rec$genes)) length(rec$genes) else 0
    }, integer(1)),
    stringsAsFactors = FALSE
  )

  res <- list(
    evaluator = object$evaluator,
    task = object$task,
    metric = object$metric,
    method = if (!is.null(object$method)) object$method else "caruana",
    stack_cv_fitness = object$stack_cv_fitness,
    single_best_fitness = object$single_best_fitness,
    ensemble_val_fitness = object$ensemble_val_fitness,
    active_count = length(active_names),
    total_islands = length(object$weights),
    weights_summary = weights_df
  )

  class(res) <- "summary_evo_ensemble"
  res
}

#' Print summary of an evo_ensemble object
#'
#' Prints summary details of the Caruana island ensemble.
#'
#' @param x A \code{summary_evo_ensemble} object.
#' @param ... Additional arguments (currently unused).
#' @return Invisible \code{x}. Called for its side effect of printing the ensemble summary.
#' @export
print.summary_evo_ensemble <- function(x, ...) {
  method <- if (is.null(x$method)) "caruana" else x$method
  cat(sprintf("Summary of evoFE %s Ensemble\n", if (method == "stack") "Stacked" else "Caruana"))
  cat(sprintf("Evaluator: %s | Task: %s | Metric: %s\n", x$evaluator, x$task, x$metric))
  cat(sprintf("Single Best Fitness: %.4f --> Ensemble Fitness: %.4f\n", x$single_best_fitness, x$ensemble_val_fitness))
  if (!is.null(x$stack_cv_fitness)) {
    cat(sprintf("Honest Nested CV Fitness: %.4f\n", x$stack_cv_fitness))
  }
  cat(sprintf("Active Islands: %d of %d\n\n", x$active_count, x$total_islands))
  print(x$weights_summary)
  invisible(x)
}
