# Population-level fitness evaluation for the evolutionary search.

#' Evaluate all unevaluated individuals in a population
#' @keywords internal
evaluate_pop <- function(pop, data, target_col, task, cv_folds, evaluation_strategy,
                         split_ids, shared_splits, evaluator,
                         fold_ids, shared_folds, shared_full, state_cache,
                         fitness_cache, threads, verbose, running_best_fitness,
                         metric = "default", allow_prune = TRUE,
                         complexity_penalty = 0, complexity_mode = "bic_dynamic",
                         complexity_floor = 0.20, complexity_target = "all_features",
                         baseline_fitness = NULL, n_samples = NULL, island = NULL,
                         fidelity_tag = "", cv_strategy = "random", time_col = NULL,
                         group_col = NULL, ...) {
  # Initialize running_best_fitness taking into account any already evaluated individuals in pop
  existing_fits <- vapply(pop, function(ind) if (is.null(ind$fitness) || is.na(ind$fitness)) -Inf else ind$fitness, double(1))
  if (length(existing_fits) > 0 && any(!is.infinite(existing_fits))) {
    max_existing <- max(existing_fits[!is.infinite(existing_fits)])
    if (is.null(running_best_fitness) || is.na(running_best_fitness) || max_existing > running_best_fitness) {
      running_best_fitness <- max_existing
    }
  }

  for (i in seq_along(pop)) {
    if (!is.na(pop[[i]]$fitness)) next

    recipe_str <- individual_to_recipe_string(pop[[i]])
    cache_key <- digest::digest(paste0(evaluator, "::", recipe_str, fidelity_tag), algo = "md5", serialize = FALSE)
    cached <- exists(cache_key, envir = fitness_cache, inherits = FALSE)

    if (cached) {
      pop[[i]] <- get(cache_key, envir = fitness_cache)
    } else {
      pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids, shared_splits = shared_splits,
        evaluator = evaluator, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric, verbose = verbose,
        allow_prune = allow_prune, complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        n_samples = n_samples,
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col, ...
      )
      assign(cache_key, pop[[i]], envir = fitness_cache)
    }

    if (verbose) {
      improved <- !is.na(pop[[i]]$fitness) && (is.na(running_best_fitness) || pop[[i]]$fitness > running_best_fitness)
      green_start <- if (supports_color()) "\033[32m" else ""
      red_start <- if (supports_color()) "\033[31m" else ""
      color_reset <- if (supports_color()) "\033[0m" else ""

      new_best_str <- if (improved) " (New Best!)" else ""
      cache_str <- if (cached) " (cached)" else ""
      msg_color <- if (improved) green_start else red_start

      if (!is.null(island)) {
        msg <- sprintf(
          "  [Island %d] Tested Individual %d%s -> Fitness: %.4f%s",
          island, i, new_best_str, pop[[i]]$fitness, cache_str
        )
      } else {
        msg <- sprintf(
          "  Tested Individual %d%s -> Fitness: %.4f%s",
          i, new_best_str, pop[[i]]$fitness, cache_str
        )
      }
      message(paste0(msg_color, msg, color_reset))

      if (improved) {
        running_best_fitness <- pop[[i]]$fitness
      }
    }
  }
  list(pop = pop, running_best_fitness = running_best_fitness)
}

#' Check whether a candidate individual is a duplicate or known-inferior
#' @keywords internal
is_invalid_individual <- function(c_ind, pop_list, cache, best_fit, evaluator = NULL) {
  # Check 1: Duplicate in current generation
  get_out <- function(ind) {
    if (length(ind$genes) == 0) {
      return(character(0))
    }
    sort(vapply(ind$genes, function(g) g$output_col, character(1)))
  }
  c_out <- get_out(c_ind)
  for (existing in pop_list) {
    e_out <- get_out(existing)
    if (length(c_out) == length(e_out) && all(c_out == e_out)) {
      return(TRUE)
    }
  }

  # Check 2: Taboo search <U+2014> reject recipes that are clearly inferior to the best.
  # Use a meaningful epsilon so borderline recipes aren't permanently banned.
  recipe_str <- individual_to_recipe_string(c_ind)
  cand_eval <- if (!is.null(evaluator)) evaluator else if (!is.null(c_ind$evaluator)) c_ind$evaluator else ""
  cache_key <- digest::digest(paste0(cand_eval, "::", recipe_str), algo = "md5", serialize = FALSE)
  if (exists(cache_key, envir = cache, inherits = FALSE)) {
    cached_ind <- get(cache_key, envir = cache)
    known_fit <- cached_ind$fitness
    if (abs(best_fit) <= 1) {
      taboo_threshold <- max(0.00002, 0.1 * (1 - abs(best_fit)))
    } else {
      taboo_threshold <- max(0.00002, 0.01 * abs(best_fit))
    }
    if (!is.infinite(best_fit) && isTRUE(known_fit < best_fit - taboo_threshold)) {
      return(TRUE)
    }
  }

  return(FALSE)
}

#' Evaluate a population with optional multi-fidelity screening
#'
#' When \code{mf_on} is TRUE (warm-up generations), individuals are first scored
#' on row-subsampled folds (\code{lf_*} structures) with low-fidelity results
#' kept in tagged cache entries. The most promising \code{mf_promote_frac}
#' fraction of individuals are then re-evaluated at full fidelity before any
#' selection decision consumes their fitness. When \code{mf_on} is FALSE this
#' is a pass-through to \code{evaluate_pop}.
#' @noRd
evaluate_pop_mf <- function(pop, data, target_col, task, cv_folds, evaluation_strategy,
                            split_ids, shared_splits, evaluator,
                            fold_ids, shared_folds, shared_full, state_cache,
                            fitness_cache, threads, verbose, running_best_fitness,
                            metric = "default", allow_prune = TRUE,
                            complexity_penalty = 0, complexity_mode = "bic_dynamic",
                            complexity_floor = 0.20, complexity_target = "all_features",
                            baseline_fitness = NULL, n_samples = NULL, island = NULL,
                            cv_strategy = "random", time_col = NULL, group_col = NULL,
                            mf_on = FALSE,
                            lf_shared_splits = NULL,
                            lf_shared_folds = NULL, lf_shared_full = NULL,
                            mf_promote_frac = 0.5, ...) {
  .do_eval <- function(p, sh_splits, sh_folds, sh_full, tag, vb, rb) {
    evaluate_pop(p, data, target_col, task, cv_folds, evaluation_strategy,
      split_ids, sh_splits, evaluator,
      fold_ids, sh_folds, sh_full, state_cache,
      fitness_cache, threads, vb, rb,
      metric = metric, allow_prune = allow_prune,
      complexity_penalty = complexity_penalty,
      complexity_mode = complexity_mode,
      complexity_floor = complexity_floor,
      complexity_target = complexity_target,
      baseline_fitness = baseline_fitness,
      n_samples = n_samples, island = island,
      fidelity_tag = tag,
      cv_strategy = cv_strategy, time_col = time_col, group_col = group_col, ...
    )
  }

  if (!mf_on || (is.null(lf_shared_folds) && is.null(lf_shared_full) && is.null(lf_shared_splits))) {
    return(.do_eval(pop, shared_splits, shared_folds, shared_full, "", verbose, running_best_fitness))
  }

  rb_in <- running_best_fitness

  lf_splits_eff <- if (!is.null(lf_shared_splits)) lf_shared_splits else shared_splits
  lf_folds_eff <- if (!is.null(lf_shared_folds)) lf_shared_folds else shared_folds
  lf_full_eff <- if (!is.null(lf_shared_full)) lf_shared_full else shared_full

  # Pass 1: cheap screen on subsampled folds/splits (silent; values are not final)
  res_lf <- .do_eval(pop, lf_splits_eff, lf_folds_eff, lf_full_eff, ":lf", FALSE, -Inf)
  pop_lf <- res_lf$pop

  fits <- vapply(pop_lf, function(x) {
    if (is.null(x$fitness) || is.na(x$fitness)) -Inf else x$fitness
  }, numeric(1))
  n_finite <- sum(is.finite(fits))
  if (n_finite == 0) {
    return(list(pop = pop_lf, running_best_fitness = rb_in))
  }
  n_promote <- max(1L, min(n_finite, ceiling(length(pop_lf) * mf_promote_frac)))
  promo_idx <- order(fits, decreasing = TRUE)[seq_len(n_promote)]
  promo_idx <- promo_idx[is.finite(fits[promo_idx])]

  # Pass 2: full-fidelity re-evaluation of promoted individuals only
  promoted <- pop_lf[promo_idx]
  for (i in seq_along(promoted)) promoted[[i]]$fitness <- NA
  res_ff <- .do_eval(promoted, shared_splits, shared_folds, shared_full, "", verbose, rb_in)

  pop_out <- pop_lf
  pop_out[promo_idx] <- res_ff$pop

  fin_fits <- vapply(pop_out, function(x) {
    if (is.null(x$fitness) || is.na(x$fitness)) -Inf else x$fitness
  }, numeric(1))
  rb_out <- max(c(rb_in, fin_fits[is.finite(fin_fits)]))

  list(pop = pop_out, running_best_fitness = rb_out)
}

#' Tournament selection
