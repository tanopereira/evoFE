#' Print an evo_recipe object
#'
#' Prints a human-readable summary of the evolutionary feature engineering recipe.
#'
#' @param x An \code{evo_recipe} object.
#' @param ... Additional arguments (currently unused).
#' @importFrom stats predict
#' @export
print.evo_recipe <- function(x, ...) {
  cat("An evoFE Recipe\n")
  cat(sprintf("  Evaluator:    %s\n", x$evaluator))
  cat(sprintf("  Task:         %s\n", x$task))
  if (!is.null(x$metric)) {
    cat(sprintf("  Metric:       %s\n", x$metric))
  }
  cat(sprintf("  Best Fitness: %.4f\n", x$best_individual$fitness))
  
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
    best_fitness = object$best_individual$fitness,
    holdout_fitness = object$best_individual$holdout_fitness,
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
#' @export
print.summary_evo_recipe <- function(x, ...) {
  cat("=== Evolutionary Feature Engineering Summary ===\n")
  cat(sprintf("ML Evaluator:          %s\n", x$evaluator))
  cat(sprintf("Task type:             %s\n", x$task))
  if (!is.null(x$metric)) {
    cat(sprintf("Optimization Metric:   %s\n", x$metric))
  }
  cat(sprintf("Best CV/Split Fitness: %.6f\n", x$best_fitness))
  if (!is.null(x$holdout_fitness) && !is.na(x$holdout_fitness)) {
    cat(sprintf("Best Holdout Fitness:  %.6f\n", x$holdout_fitness))
  }
  cat(sprintf("Number of Evolved Features: %d\n\n", x$num_genes))
  
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
    baseline <- y_vals[1]
    total_gain <- y_vals[best_g] - baseline
    subtitle <- sprintf("Generations: %d  |  Best: %.4f  |  Gain vs Gen 1: %+.4f",
                        length(y_vals), y_vals[best_g], total_gain)

    y_range <- range(y_vals, na.rm = TRUE)
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
