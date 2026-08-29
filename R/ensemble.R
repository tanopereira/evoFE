#' Caruana and Stacked Island Ensembling
#'
#' Performs ensemble selection over the validation/out-of-fold predictions from evolved
#' islands, creating an optimal multi-model ensemble. Two methods are available:
#' Caruana greedy forward selection with replacement (default), or non-negative
#' elastic-net stacking with an honest nested cross-validated performance estimate.
#'
#' @param recipe An \code{evo_recipe} object produced by \code{\link{evolve_features}}.
#' @param data A data.frame or data.table containing the original training data used
#'   during evolution. Required for lazy final model training of surviving islands.
#' @param target_col Character string. Name of the target column. If \code{NULL},
#'   it is inferred from the recipe or data.
#' @param method Character string. Ensembling strategy: \code{"caruana"} (greedy forward
#'   selection with replacement) or \code{"stack"} (non-negative elastic-net stacking of
#'   out-of-fold island predictions with an honest nested cross-validated performance
#'   estimate). Default: \code{"caruana"}.
#' @param caruana_rounds Positive integer. Number of greedy selection rounds (default: 50).
#'   Only used when \code{method = "caruana"}.
#' @param bag_samples Logical. If \code{TRUE} (default), uses bootstrap sampling of validation
#'   predictions during selection rounds to prevent validation set overfitting.
#' @param sample_ratio Numeric between 0 and 1. Fraction of validation samples used when
#'   \code{bag_samples = TRUE} (default: 0.8). Only used when \code{method = "caruana"}.
#' @param stack_folds Positive integer. Number of folds for the honest nested evaluation
#'   of the stacked ensemble (default: \code{min(5, cv_folds)} in cv mode when fold
#'   assignments are available, otherwise 5). Only used when \code{method = "stack"}.
#' @param stack_alpha Numeric in \code{[0, 1]}. Elastic-net mixing parameter of the
#'   stacking meta-learner (\code{1} = lasso, \code{0} = ridge). Default: \code{0.5}.
#'   Only used when \code{method = "stack"}.
#' @param seed Optional integer seed for reproducible bagged sampling. Does not mutate
#'   the user's global RNG state.
#' @param threads Integer. Number of threads to use for model training.
#' @param verbose Logical. Whether to print progress messages.
#' @param ... Additional arguments passed to \code{train_model}.
#'
#' @return An \code{evo_ensemble} object containing:
#'   \item{active_recipes}{Named list of feature engineering recipes for surviving islands.}
#'   \item{active_models}{Named list of trained models for surviving islands.}
#'   \item{weights}{Named numeric vector of ensemble weights (summing to 1).}
#'   \item{caruana_history}{Data frame of validation loss trajectory across selection rounds
#'     (only for \code{method = "caruana"}).}
#'   \item{single_best_fitness}{Unpenalized validation fitness of the single best island model.}
#'   \item{ensemble_val_fitness}{Validation fitness achieved by the ensemble.}
#'   \item{method}{The ensembling method used.}
#'   \item{stack_cv_fitness}{Honest nested cross-validated fitness of the stacking procedure
#'     (only for \code{method = "stack"}).}
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
                             method = c("caruana", "stack"),
                             caruana_rounds = 50,
                             bag_samples = TRUE,
                             sample_ratio = 0.8,
                             stack_folds = NULL,
                             stack_alpha = 0.5,
                             seed = NULL,
                             threads = 2,
                             verbose = TRUE, ...) {
  method <- match.arg(method)
  old_threads <- getOption("evoFE.threads")
  on.exit(options(evoFE.threads = old_threads), add = TRUE)
  options(evoFE.threads = threads)

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
    message(sprintf("\nStarting %s ensemble selection across %d candidate island models...",
                    if (method == "caruana") "Caruana" else "stacked",
                    length(val_preds_list)))
  }

  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])

  selection_res <- NULL
  if (method == "caruana") {
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
  } else {
    if (!requireNamespace("glmnet", quietly = TRUE)) {
      stop("Package 'glmnet' is required for method = \"stack\". ",
           "Install it or use method = \"caruana\".")
    }
    stored_folds <- first_recipe$fold_ids
    fold_partition <- NULL
    stack_k <- stack_folds
    if (!is.null(stored_folds) && length(stored_folds) == n_obs &&
        !all(is.na(stored_folds))) {
      fold_partition <- as.integer(stored_folds)
      k_avail <- length(unique(fold_partition))
      if (is.null(stack_k)) stack_k <- min(5L, k_avail)
    } else {
      if (verbose) {
        message("  No stored CV fold assignments matching the out-of-fold predictions; using internal folds for the nested estimate.")
      }
      if (is.null(stack_k)) stack_k <- 5L
    }
    stack_k <- max(2L, min(as.integer(stack_k), length(y_val) - 1L))

    selection_res <- .stack_select(
      y_true = y_val,
      val_preds_list = val_preds_list,
      task = task,
      metric = metric,
      num_class = num_class,
      classes = classes,
      stack_folds = stack_k,
      fold_partition = fold_partition,
      alpha = stack_alpha,
      seed = seed,
      verbose = verbose
    )
  }

  weights <- selection_res$weights
  active_names <- names(weights[weights > 0])

  # Apples-to-apples comparison: ensemble fitness is an unpenalized validation
  # metric, so compare against the island's raw (unpenalized) validation score
  # rather than the complexity-penalized selection fitness.
  single_best_fitness <- max(vapply(cand_metadata, function(m) {
    ind_i <- m$ind
    if (is.null(ind_i)) return(-Inf)
    if (!is.null(ind_i$raw_fitness) && is.finite(ind_i$raw_fitness)) {
      ind_i$raw_fitness
    } else if (!is.null(ind_i$fitness) && !is.na(ind_i$fitness)) {
      ind_i$fitness
    } else {
      -Inf
    }
  }, double(1)))

  if (verbose) {
    message(sprintf("\nEnsemble Selection Complete: %d / %d island models active.", length(active_names), length(val_preds_list)))
    if (method == "stack" && !is.null(selection_res$stack_cv_fitness)) {
      message(sprintf("  Single Best Fitness: %.4f  |  Ensemble Fitness: %.4f  |  Honest CV Fitness: %.4f",
                      single_best_fitness, selection_res$final_fitness, selection_res$stack_cv_fitness))
    } else {
      message(sprintf("  Single Best Fitness: %.4f  |  Ensemble Fitness: %.4f", single_best_fitness, selection_res$final_fitness))
    }
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
      method = method,
      stack_cv_fitness = if (!is.null(selection_res$stack_cv_fitness)) selection_res$stack_cv_fitness else NULL,
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
          sample(seq_len(n_obs), size = max(2L, min(n_obs, round(n_obs * sample_ratio))), replace = TRUE)
        } else {
          seq_len(n_obs)
        }

        y_eval <- if (is.matrix(y_true)) y_true[eval_idx, , drop = FALSE] else y_true[eval_idx]

        for (name in candidate_names) {
          cand_preds <- val_preds_list[[name]]
          candidate_blend <- blend_preds(current_blend, cand_preds, r - 1)

          p_eval <- if (is.matrix(candidate_blend)) candidate_blend[eval_idx, , drop = FALSE] else candidate_blend[eval_idx]
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

#' Internal Non-Negative Elastic-Net Stacking Engine
#'
#' Fits a non-negative elastic-net meta-learner over out-of-fold island predictions
#' and produces an honest nested cross-validated estimate of the deployed stacking
#' procedure (scalar normalized island weights + weighted prediction averaging).
#'
#' @keywords internal
#' @noRd
.stack_select <- function(y_true, val_preds_list, task, metric,
                          num_class = NULL, classes = NULL,
                          stack_folds = 5L, fold_partition = NULL,
                          alpha = 0.5, seed = NULL, verbose = FALSE) {
  n_candidates <- length(val_preds_list)
  candidate_names <- names(val_preds_list)
  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])

  fam <- switch(task,
    regression = "gaussian",
    classification = "binomial",
    multiclass = "multinomial",
    stop(sprintf("Unsupported task '%s' for stacking.", task))
  )

  # Local seed wrapper preserving the user's global RNG state
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

  # Metric helper (higher is better); NA predictions are masked out
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

  # Level-2 design matrix: one block of columns per candidate island
  blocks <- lapply(val_preds_list, function(p) {
    if (is.matrix(p)) p else matrix(p, ncol = 1)
  })
  block_ncols <- vapply(blocks, ncol, integer(1))
  col_end <- cumsum(block_ncols)
  col_start <- col_end - block_ncols + 1L

  X_fit <- do.call(cbind, blocks)
  X_fit[!is.finite(X_fit)] <- 0

  if (task == "regression") {
    y_fit <- as.numeric(y_true)
  } else {
    lv <- if (!is.null(classes)) classes else sort(unique(as.character(y_true)))
    # Multiclass OOF targets may arrive integer-encoded (0-based); map back to labels
    y_lab <- if (is.numeric(y_true) && length(lv) > 0 &&
                 all(stats::na.omit(y_true) %in% (seq_along(lv) - 1))) {
      lv[as.integer(y_true) + 1]
    } else {
      as.character(y_true)
    }
    y_fit <- factor(y_lab, levels = lv)
  }

  # Nesting folds: reuse the evolution partition when provided, else internal balanced folds
  folds <- if (!is.null(fold_partition) && length(fold_partition) == n_obs && anyDuplicated(fold_partition) > 0 &&
               length(unique(fold_partition)) >= 2L) {
    as.integer(factor(fold_partition))
  } else {
    run_with_seed(if (!is.null(seed)) seed + 7919L else NULL, function() {
      sample(rep(seq_len(stack_folds), length.out = n_obs))
    })
  }
  nest_folds <- sort(unique(folds))
  k_fmt <- nchar(as.character(length(nest_folds)))

  aggregate_weights <- function(fit) {
    raw_w <- if (fam == "multinomial") {
      cls_coefs <- glmnet::coef.glmnet(fit, s = "lambda.min")
      mat <- do.call(cbind, lapply(cls_coefs, function(m) as.numeric(m)[-1]))
      mat <- pmax(mat, 0)
      vapply(seq_len(n_candidates), function(j) {
        sum(mat[col_start[j]:col_end[j], , drop = FALSE]) / ncol(mat)
      }, numeric(1))
    } else {
      cf <- as.numeric(glmnet::coef.glmnet(fit, s = "lambda.min"))[-1]
      cf <- pmax(cf, 0)
      vapply(seq_len(n_candidates), function(j) sum(cf[col_start[j]:col_end[j]]), numeric(1))
    }
    total <- sum(raw_w)
    if (!is.finite(total) || total <= 0) {
      return(NULL)
    }
    stats::setNames(raw_w / total, candidate_names)
  }

  blend_with <- function(w) {
    if (!is.matrix(val_preds_list[[1]])) {
      as.vector(X_fit %*% w)
    } else {
      out <- NULL
      for (j in seq_len(n_candidates)) {
        if (w[j] <= 0) next
        p <- val_preds_list[[j]]
        out <- if (is.null(out)) w[j] * p else out + w[j] * p
      }
      out
    }
  }

  fallback_weights <- function(rows = seq_len(n_obs)) {
    y_sub <- if (is.matrix(y_true)) y_true[rows, , drop = FALSE] else y_true[rows]
    solo <- vapply(candidate_names, function(nm) {
      p_sub <- if (is.matrix(val_preds_list[[nm]])) val_preds_list[[nm]][rows, , drop = FALSE] else val_preds_list[[nm]][rows]
      eval_fitness(y_sub, p_sub)
    }, double(1))
    best <- names(which.max(solo))[1]
    stats::setNames(ifelse(candidate_names == best, 1, 0), candidate_names)
  }

  # Class-balanced (or random, for regression) internal CV fold ids for cv.glmnet.
  # All sampling runs under deterministic per-call seeds so results are reproducible
  # for a given `seed` without touching the user's global RNG state.
  sample_counter <- 0L
  seeded_sample <- function(...) {
    sample_counter <<- sample_counter + 1L
    run_with_seed(if (!is.null(seed)) seed + 1000003L * sample_counter else NULL, function() {
      sample(...)
    })
  }

  make_foldid <- function(y_sub, k) {
    k <- max(2L, min(k, length(y_sub)))
    if (fam == "gaussian") {
      return(seeded_sample(rep(seq_len(k), length.out = length(y_sub))))
    }
    fid <- integer(length(y_sub))
    for (lv in levels(droplevels(factor(y_sub)))) {
      ids <- which(as.character(y_sub) == lv)
      if (length(ids) == 0) next
      fid[ids] <- seeded_sample(rep(seq_len(min(k, length(ids))), length.out = length(ids)))
    }
    un <- which(fid == 0)
    if (length(un) > 0) {
      fid[un] <- seeded_sample(rep(seq_len(k), length.out = length(un)))
    }
    if (length(unique(fid[fid > 0])) < min(2L, length(y_sub))) {
      fid <- seeded_sample(rep(seq_len(k), length.out = length(y_sub)))
    }
    fid
  }

  fit_net <- function(rows) {
    y_tr <- y_fit[rows]
    nf <- max(3L, min(10L, length(rows) %/% 4L))
    if (fam != "gaussian") {
      tab <- table(y_tr)
      nf <- min(nf, min(tab[tab > 0]))
    }
    nf <- max(3L, min(nf, length(rows)))
    args <- list(
      x = X_fit[rows, , drop = FALSE],
      y = y_tr,
      family = fam,
      alpha = alpha,
      standardize = FALSE,
      foldid = make_foldid(y_tr, nf)
    )
    if (fam != "multinomial") args$lower.limits <- 0
    do.call(glmnet::cv.glmnet, args)
  }

  if (verbose) {
    message(sprintf("  Stacking %d island models with non-negative elastic net (alpha = %.2f), nested over %d folds...",
                    n_candidates, alpha, length(nest_folds)))
  }

  # Honest nested evaluation of the deployed procedure (scalar weights + weighted blending)
  pooled_preds <- NULL
  pooled_y <- NULL
  fold_fitnesses <- numeric(length(nest_folds))

  for (fi in seq_along(nest_folds)) {
    f <- nest_folds[fi]
    te <- which(folds == f)
    tr <- which(folds != f)

    w_f <- local({
      fit <- fit_net(tr)
      w <- aggregate_weights(fit)
      if (is.null(w)) fallback_weights(tr) else w
    })

    is_mat <- is.matrix(val_preds_list[[1]])
    blended <- blend_with(w_f)
    preds_te <- if (is_mat) blended[te, , drop = FALSE] else blended[te]
    fold_fitnesses[fi] <- eval_fitness(y_true[te], preds_te)

    if (is.null(pooled_preds)) {
      pooled_preds <- if (is_mat) {
        matrix(NA_real_, nrow = n_obs, ncol = num_class)
      } else {
        rep(NA_real_, n_obs)
      }
      pooled_y <- y_true
    }
    if (is_mat) {
      pooled_preds[te, ] <- preds_te
    } else {
      pooled_preds[te] <- preds_te
    }

    if (verbose) {
      message(sprintf("  [Nest Fold %*d/%*d] Fitness: %.4f", k_fmt, fi, k_fmt, length(nest_folds), fold_fitnesses[fi]))
    }
  }

  stack_cv_fitness <- eval_fitness(pooled_y, pooled_preds)

  # Final weights on all level-2 rows
  final_w <- local({
    fit <- fit_net(seq_len(n_obs))
    w <- aggregate_weights(fit)
    if (is.null(w)) fallback_weights(seq_len(n_obs)) else w
  })
  final_fitness <- eval_fitness(y_true, blend_with(final_w))

  if (verbose) {
    message(sprintf("  Stack CV Fitness (honest): %.4f  |  Full-Fit Ensemble Fitness: %.4f", stack_cv_fitness, final_fitness))
  }

  list(
    weights = final_w,
    history = NULL,
    final_fitness = final_fitness,
    stack_cv_fitness = stack_cv_fitness
  )
}
