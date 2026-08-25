#' Caruana Island Ensemble Selection
#'
#' Performs Caruana ensemble selection (greedy forward selection with replacement)
#' over the validation/out-of-fold predictions from evolved islands, creating an
#' optimal multi-model ensemble.
#'
#' @param recipe An \code{evo_recipe} object produced by \code{\link{evolve_features}}.
#' @param data A data.frame or data.table containing the original training data used
#'   during evolution. Required for lazy final model training of surviving islands.
#' @param target_col Character string. Name of the target column. If \code{NULL},
#'   it is inferred from the recipe or data.
#' @param caruana_rounds Positive integer. Number of greedy selection rounds (default: 50).
#' @param bag_samples Logical. If \code{TRUE} (default), uses bootstrap sampling of validation
#'   predictions during selection rounds to prevent validation set overfitting.
#' @param sample_ratio Numeric between 0 and 1. Fraction of validation samples used when
#'   \code{bag_samples = TRUE} (default: 0.8).
#' @param seed Optional integer seed for reproducible bagged sampling. Does not mutate
#'   the user's global RNG state.
#' @param threads Integer. Number of threads to use for model training.
#' @param verbose Logical. Whether to print progress messages.
#' @param ... Additional arguments passed to \code{train_model}.
#'
#' @return An \code{evo_ensemble} object containing:
#'   \item{active_recipes}{Named list of feature engineering recipes for surviving islands.}
#'   \item{active_models}{Named list of trained models for surviving islands.}
#'   \item{weights}{Named numeric vector of Caruana ensemble weights (summing to 1).}
#'   \item{caruana_history}{Data frame of validation loss trajectory across selection rounds.}
#'   \item{single_best_fitness}{Unpenalized validation fitness of the single best island model.}
#'   \item{ensemble_val_fitness}{Validation fitness achieved by the Caruana ensemble.}
#'   \item{task}{The learning task ("classification", "regression", or "multiclass").}
#'   \item{evaluator}{The evaluator model engine used.}
#'   \item{classes}{Target class levels (for multiclass classification).}
#'   \item{metric}{Evaluation metric used.}
#'
#' @examples
#' \donttest{
#' data(mtcars)
#' df <- mtcars
#' df$am <- as.integer(df$am)
#'
#' # Evolve features across 3 islands
#' recipe <- evolve_features(
#'   data = df,
#'   target_col = "am",
#'   task = "classification",
#'   evaluator = "xgboost",
#'   generations = 2,
#'   pop_size = 2,
#'   islands = 3,
#'   cv_folds = 2,
#'   verbose = FALSE
#' )
#'
#' # Build Caruana ensemble from island predictions
#' ens <- ensemble_islands(recipe, data = df, caruana_rounds = 20, verbose = FALSE)
#' print(ens)
#' }
#' @export
ensemble_islands <- function(recipe, data, target_col = NULL,
                             caruana_rounds = 50,
                             bag_samples = TRUE,
                             sample_ratio = 0.8,
                             seed = NULL,
                             threads = 2,
                             verbose = TRUE, ...) {
  # Normalize recipe input: single evo_recipe or list of evo_recipe objects
  if (inherits(recipe, "evo_recipe")) {
    recipe_list <- list(recipe1 = recipe)
  } else if (is.list(recipe)) {
    recipe_list <- recipe
    for (idx in seq_along(recipe_list)) {
      if (!inherits(recipe_list[[idx]], "evo_recipe")) {
        stop(sprintf("Element %d in 'recipe' list is not an object of class 'evo_recipe'.", idx))
      }
    }
  } else {
    stop("Input 'recipe' must be an object of class 'evo_recipe' or a list of 'evo_recipe' objects.")
  }

  if (!is.numeric(caruana_rounds) || caruana_rounds < 1) {
    stop("'caruana_rounds' must be a positive integer >= 1.")
  }
  caruana_rounds <- as.integer(caruana_rounds)

  if (missing(data) || is.null(data)) {
    stop("Argument 'data' (full training dataset) is required for lazy final model fitting.")
  }

  first_recipe <- recipe_list[[1]]

  # Infer target_col
  if (is.null(target_col)) {
    if (!is.null(first_recipe$target_col)) {
      target_col <- first_recipe$target_col
    } else {
      ind <- first_recipe$best_individual
      all_known <- unique(c(ind$all_numeric_cols, ind$all_categorical_cols, ind$all_datetime_cols))
      cand <- setdiff(names(data), all_known)
      if (length(cand) == 1) {
        target_col <- cand
      } else {
        stop("Could not automatically infer 'target_col'. Please specify target_col explicitly.")
      }
    }
  }

  task <- first_recipe$task
  metric <- first_recipe$metric
  classes <- first_recipe$classes
  num_class <- if (!is.null(classes)) length(classes) else NULL

  # Collect validation prediction vectors, targets, and evaluators across all recipes
  val_preds_list <- list()
  cand_metadata <- list()

  for (r_idx in seq_along(recipe_list)) {
    rec <- recipe_list[[r_idx]]
    rec_prefix <- if (!is.null(names(recipe_list)) && names(recipe_list)[r_idx] != "") {
      names(recipe_list)[r_idx]
    } else if (length(recipe_list) > 1) {
      paste0("recipe_", r_idx)
    } else {
      ""
    }

    if (is.null(rec$island_bests) || length(rec$island_bests) == 0) {
      next
    }

    for (i in seq_along(rec$island_bests)) {
      ind_i <- rec$island_bests[[i]]
      if (!is.null(ind_i) && !is.null(ind_i$val_preds)) {
        cand_name <- if (nchar(rec_prefix) > 0) paste0(rec_prefix, "_island_", i) else paste0("island_", i)
        cand_eval <- if (!is.null(ind_i$evaluator)) ind_i$evaluator else rec$evaluator

        val_preds_list[[cand_name]] <- ind_i$val_preds
        cand_metadata[[cand_name]] <- list(
          recipe = rec,
          ind = ind_i,
          evaluator = cand_eval
        )
      }
    }
  }

  if (length(val_preds_list) == 0) {
    stop("No valid validation prediction vectors found across the provided recipe(s).")
  }

  if (length(val_preds_list) < 2) {
    warning("Only 1 candidate island prediction vector available. Ensembling requires >= 2 candidate islands.")
  }

  y_val <- cand_metadata[[1]]$ind$y_val
  if (is.null(y_val)) {
    y_val <- data[[target_col]]
  }

  if (verbose) {
    message(sprintf("\nStarting Caruana Ensemble Selection across %d candidate island models (%d rounds)...", length(val_preds_list), caruana_rounds))
  }

  # Perform Caruana greedy ensemble selection
  selection_res <- caruana_select(
    y_true = y_val,
    val_preds_list = val_preds_list,
    task = task,
    metric = metric,
    rounds = caruana_rounds,
    bag_samples = bag_samples,
    sample_ratio = sample_ratio,
    seed = seed,
    num_class = num_class,
    verbose = verbose
  )

  weights <- selection_res$weights
  active_names <- names(weights[weights > 0])

  # Apples-to-apples comparison: ensemble fitness is an unpenalized validation
  # metric, so compare against the island's raw (unpenalized) validation score
  # rather than the complexity-penalized selection fitness.
  single_best_fitness <- max(vapply(recipe_list, function(r) {
    bi <- r$best_individual
    if (is.null(bi)) return(-Inf)
    if (!is.null(bi$raw_fitness) && is.finite(bi$raw_fitness)) {
      bi$raw_fitness
    } else if (!is.null(bi$fitness) && !is.na(bi$fitness)) {
      bi$fitness
    } else {
      -Inf
    }
  }, double(1)))

  if (verbose) {
    message(sprintf("\nCaruana Selection Complete: %d / %d island models active.", length(active_names), length(val_preds_list)))
    message(sprintf("  Single Best Fitness: %.4f  |  Ensemble Fitness: %.4f", single_best_fitness, selection_res$final_fitness))
  }

  active_recipes <- list()
  active_models <- list()
  active_evaluators <- list()

  # Shared state cache for dataset transformation
  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  dt_full <- data.table::as.data.table(data)

  # Lazy training: Reuse best_model for matching global best island, fit remaining active islands
  for (name in active_names) {
    meta <- cand_metadata[[name]]
    ind_i <- meta$ind
    rec_i <- meta$recipe
    eval_i <- meta$evaluator
    ind_str <- individual_to_recipe_string(ind_i)
    best_ind_str <- individual_to_recipe_string(rec_i$best_individual)

    active_evaluators[[name]] <- eval_i

    # Check if this candidate matches rec_i$best_individual & rec_i$best_model is available
    if (ind_str == best_ind_str && !is.null(rec_i$best_model)) {
      if (verbose) {
        message(sprintf("  [%s] Evaluator: %s | Weight: %5.1f%% | Reusing existing global best model (zero retraining).", name, eval_i, weights[[name]] * 100))
      }
      active_recipes[[name]] <- rec_i$best_individual
      active_models[[name]] <- rec_i$best_model
    } else {
      if (verbose) {
        message(sprintf("  [%s] Evaluator: %s | Weight: %5.1f%% | Lazily training final model on full dataset...", name, eval_i, weights[[name]] * 100))
      }

      res_full <- apply_individual(ind_i, dt_full, NULL, target_col, state_cache = state_cache, allow_prune = TRUE)
      applied_ind <- res_full$ind
      active_recipes[[name]] <- applied_ind

      gene_cols <- if (length(applied_ind$genes) > 0) vapply(applied_ind$genes, function(g) g$output_col, character(1)) else character(0)
      features <- c(applied_ind$numeric_cols, applied_ind$categorical_cols, applied_ind$datetime_cols, gene_cols)
      features <- setdiff(features, target_col)

      x_full <- data.matrix(res_full$train[, features, with = FALSE])
      x_full[!is.finite(x_full)] <- NA
      y_full <- res_full$train[[target_col]]
      if (task == "multiclass") {
        y_full <- as.integer(factor(y_full, levels = classes)) - 1
      }

      # Train model using candidate's specific evaluator and best params
      res_m <- train_model(
        x_full, y_full,
        task = task, evaluator = eval_i,
        threads = threads, num_class = num_class, metric = metric,
        verbose = verbose, best_params = ind_i$best_params, ...
      )
      active_models[[name]] <- res_m$model
    }
  }

  structure(
    list(
      active_recipes = active_recipes,
      active_models = active_models,
      active_evaluators = active_evaluators,
      weights = weights,
      caruana_history = selection_res$history,
      single_best_fitness = single_best_fitness,
      ensemble_val_fitness = selection_res$final_fitness,
      task = task,
      evaluator = first_recipe$evaluator,
      target_col = target_col,
      classes = classes,
      metric = metric
    ),
    class = "evo_ensemble"
  )
}

#' Internal Caruana Greedy Selection Engine
#' @keywords internal
#' @noRd
caruana_select <- function(y_true, val_preds_list, task, metric, rounds = 50,
                           bag_samples = TRUE, sample_ratio = 0.8, seed = NULL,
                           num_class = NULL, verbose = FALSE) {

  n_candidates <- length(val_preds_list)
  candidate_names <- names(val_preds_list)
  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])
  w_fmt <- nchar(as.character(rounds))

  # Helper for safe metric evaluation (higher is better for fitness)
  eval_fitness <- function(y, p) {
    if (is.matrix(p)) {
      mask <- !is.na(p[, 1])
      if (!any(mask)) return(-Inf)
      compute_metric(y[mask], p[mask, , drop = FALSE], task = task, metric = metric, num_class = num_class)
    } else {
      mask <- !is.na(p)
      if (!any(mask)) return(-Inf)
      compute_metric(y[mask], p[mask], task = task, metric = metric)
    }
  }

  # Helper to blend predictions
  blend_preds <- function(current_sum, new_preds, count) {
    if (count == 0) return(new_preds)
    if (is.matrix(current_sum)) {
      (current_sum * count + new_preds) / (count + 1)
    } else {
      (current_sum * count + new_preds) / (count + 1)
    }
  }

  # Local seed wrapper preserving user's global RNG state
  run_with_seed <- function(seed_val, code) {
    if (is.null(seed_val)) return(code())
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) get(".Random.seed", envir = .GlobalEnv) else NULL
    on.exit({
      if (!is.null(old_seed)) {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv)) {
        rm(".Random.seed", envir = .GlobalEnv)
      }
    }, add = TRUE)
    set.seed(seed_val)
    code()
  }

  selected_counts <- stats::setNames(rep(0L, n_candidates), candidate_names)

  # Find initial best candidate model
  initial_scores <- vapply(candidate_names, function(name) {
    eval_fitness(y_true, val_preds_list[[name]])
  }, double(1))

  best_init_idx <- which.max(initial_scores)
  if (length(best_init_idx) == 0 || is.na(best_init_idx)) {
    best_init_idx <- 1L
  }
  best_init_name <- candidate_names[best_init_idx]

  selected_counts[best_init_name] <- 1L
  current_blend <- val_preds_list[[best_init_name]]
  current_score <- initial_scores[best_init_idx]

  if (verbose) {
    message(sprintf("  [Round %*d/%d] Initialized with %s -> Fitness: %.4f", w_fmt, 1, rounds, best_init_name, current_score))
  }

  history <- data.frame(
    round = 1:rounds,
    selected_model = character(rounds),
    fitness = numeric(rounds),
    stringsAsFactors = FALSE
  )

  history$selected_model[1] <- best_init_name
  history$fitness[1] <- current_score

  if (rounds > 1) {
    for (r in 2:rounds) {
      best_cand_name <- NULL
      best_cand_score <- -Inf
      best_cand_blend <- NULL

      run_with_seed(if (!is.null(seed)) seed + r else NULL, function() {
        eval_idx <- if (bag_samples) {
          sample(seq_len(n_obs), size = round(n_obs * sample_ratio), replace = TRUE)
        } else {
          seq_len(n_obs)
        }

        y_eval <- if (is.matrix(y_true)) y_true[eval_idx, ] else y_true[eval_idx]

        for (name in candidate_names) {
          cand_preds <- val_preds_list[[name]]
          candidate_blend <- blend_preds(current_blend, cand_preds, r - 1)

          p_eval <- if (is.matrix(candidate_blend)) candidate_blend[eval_idx, ] else candidate_blend[eval_idx]
          score <- eval_fitness(y_eval, p_eval)

          if (score > best_cand_score) {
            best_cand_score <- score
            best_cand_name <- name
            best_cand_blend <- candidate_blend
          }
        }

        if (!is.null(best_cand_name)) {
          selected_counts[best_cand_name] <<- selected_counts[best_cand_name] + 1L
          current_blend <<- best_cand_blend
          prev_score <- current_score
          current_score <<- eval_fitness(y_true, current_blend)

          if (verbose) {
            imp_tag <- if (!is.na(current_score) && !is.na(prev_score) && current_score > prev_score) " (Improved!)" else ""
            message(sprintf("  [Round %*d/%d] Selected %s -> Ensemble Fitness: %.4f%s", w_fmt, r, rounds, best_cand_name, current_score, imp_tag))
          }
        }

        history$selected_model[r] <<- best_cand_name
        history$fitness[r] <<- current_score
      })
    }
  }

  weights <- selected_counts / rounds
  final_fitness <- eval_fitness(y_true, current_blend)

  list(
    weights = weights,
    history = history,
    final_fitness = final_fitness
  )
}
