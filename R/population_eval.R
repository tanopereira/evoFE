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
                         group_col = NULL, ind_indices = NULL, ...) {
  # Initialize running_best_fitness taking into account any already evaluated individuals in pop
  existing_fits <- vapply(pop, function(ind) if (is.null(ind$fitness) || is.na(ind$fitness)) -Inf else ind$fitness, double(1))
  if (length(existing_fits) > 0 && any(!is.infinite(existing_fits))) {
    max_existing <- max(existing_fits[!is.infinite(existing_fits)])
    if (is.null(running_best_fitness) || is.na(running_best_fitness) || max_existing > running_best_fitness) {
      running_best_fitness <- max_existing
    }
  }

  for (i in seq_along(pop)) {
    if (!is.na(pop[[i]]$fitness)) {
      if (verbose && !is.null(ind_indices)) {
        ind_idx <- if (length(ind_indices) >= i) ind_indices[i] else i
        is_lf <- identical(fidelity_tag, ":lf")
        if (is_lf) {
          tag_str <- " [MF Screen]"
          score_lbl <- "Score"
          if (!is.null(island)) {
            message(sprintf("  [Island %d]%s Individual %d -> %s: %.4f (prior full-fidelity best)", island, tag_str, ind_idx, score_lbl, pop[[i]]$fitness))
          } else {
            message(sprintf("  %s Individual %d -> %s: %.4f (prior full-fidelity best)", trimws(tag_str), ind_idx, score_lbl, pop[[i]]$fitness))
          }
        } else {
          if (!is.null(island)) {
            message(sprintf("  [Island %d] Tested Individual %d -> Fitness: %.4f (prior full-fidelity best)", island, ind_idx, pop[[i]]$fitness))
          } else {
            message(sprintf("  Tested Individual %d -> Fitness: %.4f (prior full-fidelity best)", ind_idx, pop[[i]]$fitness))
          }
        }
      }
      next
    }

    ind_idx <- if (!is.null(ind_indices) && length(ind_indices) >= i) ind_indices[i] else i

    rec_key <- individual_to_cache_key(pop[[i]])
    cache_key <- digest::digest(paste0(evaluator, "::", rec_key, fidelity_tag), algo = "md5", serialize = FALSE)
    cached <- exists(cache_key, envir = fitness_cache, inherits = FALSE)

    if (cached) {
      cached_ind <- get(cache_key, envir = fitness_cache)
      pop[[i]] <- cached_ind
      pop[[i]]$raw_fitness <- cached_ind$raw_fitness
      pop[[i]]$train_size <- cached_ind$train_size
      is_dynamic <- (complexity_penalty %in% c("bic_dynamic", "pac_bayes_dynamic")) ||
                    (complexity_mode %in% c("bic_dynamic", "pac_bayes_dynamic"))
      if (is_dynamic && !is.null(cached_ind$raw_fitness) && is.finite(cached_ind$raw_fitness)) {
        pop[[i]] <- apply_complexity_penalty(
          raw_score = cached_ind$raw_fitness,
          ind = pop[[i]],
          n_samples = if (!is.null(cached_ind$train_size)) cached_ind$train_size else 100L,
          task = task,
          complexity_penalty = if (is.numeric(complexity_penalty)) complexity_penalty else 1.0,
          complexity_mode = if (is.character(complexity_penalty) && complexity_penalty %in% c("bic_dynamic", "pac_bayes_dynamic")) complexity_penalty else complexity_mode,
          complexity_floor = complexity_floor,
          complexity_target = complexity_target,
          running_best_fitness = running_best_fitness,
          baseline_fitness = baseline_fitness,
          metric = metric
        )
        if (!is.null(cached_ind$sd_penalty_val) && is.finite(cached_ind$sd_penalty_val) && cached_ind$sd_penalty_val > 0) {
          pop[[i]]$penalty <- pop[[i]]$penalty + cached_ind$sd_penalty_val
          pop[[i]]$fitness <- pop[[i]]$fitness - cached_ind$sd_penalty_val
        }
      }
    } else {
      orig_ind <- pop[[i]]
      other_keys <- vapply(pop[-i], individual_to_cache_key, character(1))
      pop[[i]] <- evaluate_fitness(pop[[i]], data, target_col,
        task = task, cv_folds = cv_folds,
        evaluation_strategy = evaluation_strategy,
        split_ids = split_ids, shared_splits = shared_splits,
        evaluator = evaluator, fold_ids = fold_ids,
        shared_folds = shared_folds,
        shared_full = shared_full, state_cache = state_cache,
        threads = threads, metric = metric,
        verbose = if (identical(fidelity_tag, ":lf")) FALSE else verbose,
        allow_prune = allow_prune, complexity_penalty = complexity_penalty,
        complexity_mode = complexity_mode,
        complexity_floor = complexity_floor,
        complexity_target = complexity_target,
        running_best_fitness = running_best_fitness,
        baseline_fitness = baseline_fitness,
        n_samples = n_samples,
        cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
        fitness_cache = fitness_cache, fidelity_tag = fidelity_tag, ...
      )
      collapsed <- isTRUE(attr(pop[[i]], "collapsed_to_cached"))
      attr(pop[[i]], "collapsed_to_cached") <- NULL
      post_key <- individual_to_cache_key(pop[[i]])
      pruned_dup <- FALSE

      if (collapsed && (post_key %in% other_keys)) {
        pruned_dup <- TRUE
        orig_ind$raw_fitness <- -Inf
        orig_ind$penalty <- 0.0
        orig_ind$fitness <- -Inf
        orig_ind$holdout_fitness <- NULL
        pop[[i]] <- orig_ind
        assign(cache_key, pop[[i]], envir = fitness_cache)
        fitness_cache[[rec_key]] <- pop[[i]]
      } else {
        if (collapsed) {
          cached <- TRUE
        }
        assign(cache_key, pop[[i]], envir = fitness_cache)
        fitness_cache[[rec_key]] <- pop[[i]]
        if (!identical(post_key, rec_key)) {
          post_cache_key <- digest::digest(paste0(evaluator, "::", post_key, fidelity_tag), algo = "md5", serialize = FALSE)
          if (!exists(post_cache_key, envir = fitness_cache, inherits = FALSE)) {
            assign(post_cache_key, pop[[i]], envir = fitness_cache)
            fitness_cache[[post_key]] <- pop[[i]]
          }
        }
      }
    }

    if (verbose) {
      improved <- !is.na(pop[[i]]$fitness) && is.finite(pop[[i]]$fitness) &&
                  (is.na(running_best_fitness) || pop[[i]]$fitness > running_best_fitness)
      is_lf <- identical(fidelity_tag, ":lf")
      green_start <- if (supports_color()) "\033[32m" else ""
      red_start <- if (supports_color()) "\033[31m" else ""
      cyan_start <- if (supports_color()) "\033[36m" else ""
      color_reset <- if (supports_color()) "\033[0m" else ""

      new_best_str <- if (improved) {
        if (is_lf) " (Screen Best!)" else " (New Best!)"
      } else {
        ""
      }
      cache_str <- if (exists("pruned_dup", inherits = FALSE) && isTRUE(pruned_dup)) {
        " (pruned: constant/collinear gene)"
      } else if (cached) {
        " (cached)"
      } else {
        ""
      }

      msg_color <- if (is_lf) {
        if (improved) cyan_start else ""
      } else {
        if (improved) green_start else red_start
      }

      score_lbl <- if (is_lf) "Score" else "Fitness"
      tag_str <- if (is_lf) " [MF Screen]" else ""

      if (!is.null(island)) {
        msg <- sprintf(
          "  [Island %d]%s Tested Individual %d%s -> %s: %.4f%s",
          island, tag_str, ind_idx, new_best_str, score_lbl, pop[[i]]$fitness, cache_str
        )
      } else {
        prefix <- if (is_lf) "  [MF Screen] " else "  "
        msg <- sprintf(
          "%sTested Individual %d%s -> %s: %.4f%s",
          prefix, ind_idx, new_best_str, score_lbl, pop[[i]]$fitness, cache_str
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
    sort(vapply(ind$genes, gene_to_formula, character(1), truncate = FALSE))
  }
  get_raw <- function(ind) {
    sort(c(ind$numeric_cols, ind$categorical_cols, ind$datetime_cols))
  }
  c_out <- get_out(c_ind)
  c_raw <- get_raw(c_ind)
  for (existing in pop_list) {
    e_out <- get_out(existing)
    e_raw <- get_raw(existing)
    if (length(c_out) == length(e_out) && all(c_out == e_out) &&
        length(c_raw) == length(e_raw) && all(c_raw == e_raw)) {
      return(TRUE)
    }
  }

  # Check 2: Taboo search — reject recipes that are clearly inferior to the best.
  # Use a meaningful epsilon so borderline recipes aren't permanently banned.
  rec_key <- individual_to_cache_key(c_ind)
  cand_eval <- if (!is.null(evaluator)) evaluator else if (!is.null(c_ind$evaluator)) c_ind$evaluator else ""
  cache_key <- digest::digest(paste0(cand_eval, "::", rec_key), algo = "md5", serialize = FALSE)
  cached_ind <- if (exists(cache_key, envir = cache, inherits = FALSE)) {
    get(cache_key, envir = cache)
  } else if (exists(rec_key, envir = cache, inherits = FALSE)) {
    get(rec_key, envir = cache)
  } else {
    NULL
  }
  if (!is.null(cached_ind)) {
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
  .do_eval <- function(p, sh_splits, sh_folds, sh_full, tag, vb, rb, indices = NULL) {
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
      cv_strategy = cv_strategy, time_col = time_col, group_col = group_col,
      ind_indices = indices, ...
    )
  }

  if (!mf_on || (is.null(lf_shared_folds) && is.null(lf_shared_splits))) {
    return(.do_eval(pop, shared_splits, shared_folds, shared_full, "", verbose, running_best_fitness, indices = seq_along(pop)))
  }

  rb_in <- running_best_fitness

  lf_splits_eff <- if (!is.null(lf_shared_splits)) lf_shared_splits else shared_splits
  lf_folds_eff <- if (!is.null(lf_shared_folds)) lf_shared_folds else shared_folds
  # Full dataset is never downsampled: global unsupervised representations and fold allocations remain global
  lf_full_eff <- shared_full

  # Track which individuals were already evaluated with full-fidelity fitness
  already_evaluated <- !vapply(pop, function(x) is.null(x$fitness) || is.na(x$fitness), logical(1))

  # Determine row counts for display
  lf_train_rows <- if (!is.null(lf_splits_eff$train)) {
    nrow(lf_splits_eff$train)
  } else if (!is.null(lf_folds_eff) && length(lf_folds_eff) > 0 && !is.null(lf_folds_eff[[1]]$train)) {
    nrow(lf_folds_eff[[1]]$train)
  } else {
    NA_integer_
  }

  ff_train_rows <- if (!is.null(shared_splits$train)) {
    nrow(shared_splits$train)
  } else if (!is.null(shared_folds) && length(shared_folds) > 0 && !is.null(shared_folds[[1]]$train)) {
    nrow(shared_folds[[1]]$train)
  } else {
    NA_integer_
  }

  sample_pct_str <- if (!is.na(lf_train_rows) && !is.na(ff_train_rows) && ff_train_rows > 0) {
    sprintf("%.0f%% data: %s rows", 100 * lf_train_rows / ff_train_rows, format(lf_train_rows, big.mark = ","))
  } else if (!is.na(lf_train_rows)) {
    sprintf("%s rows", format(lf_train_rows, big.mark = ","))
  } else {
    "subsampled data"
  }

  isl_pfx <- if (!is.null(island)) sprintf("  [Island %d] ", island) else "  "

  if (verbose) {
    n_uneval <- sum(!already_evaluated)
    cand_lbl <- if (n_uneval == length(pop)) {
      sprintf("%d candidates", length(pop))
    } else {
      sprintf("%d unevaluated, %d prior best", n_uneval, sum(already_evaluated))
    }
    message(sprintf("%s--- Multi-Fidelity Screening (%s, %s) ---", isl_pfx, cand_lbl, sample_pct_str))
  }

  # Pass 1: screen on subsampled folds/splits with real-time individual progress
  res_lf <- .do_eval(pop, lf_splits_eff, lf_folds_eff, lf_full_eff, ":lf", verbose, -Inf, indices = seq_along(pop))
  pop_lf <- res_lf$pop

  fits <- vapply(pop_lf, function(x) {
    if (is.null(x$fitness) || is.na(x$fitness)) -Inf else x$fitness
  }, numeric(1))
  n_finite <- sum(is.finite(fits))
  if (n_finite == 0) {
    if (verbose) {
      message(sprintf("%s[MF Warning] No finite scores obtained during screening.", isl_pfx))
    }
    return(list(pop = pop_lf, running_best_fitness = rb_in))
  }
  n_promote <- max(1L, min(n_finite, ceiling(length(pop_lf) * mf_promote_frac)))
  promo_idx <- order(fits, decreasing = TRUE)[seq_len(n_promote)]
  promo_idx <- promo_idx[is.finite(fits[promo_idx])]
  unpromo_idx <- setdiff(seq_along(pop_lf), promo_idx)

  if (verbose) {
    format_cand_summary <- function(indices, max_display = 6L) {
      if (length(indices) == 0L) return("none")
      items <- vapply(indices, function(idx) {
        sprintf("Ind %d (%.4f)", idx, fits[idx])
      }, character(1))
      if (length(items) > max_display) {
        paste0(paste(items[seq_len(max_display)], collapse = ", "), sprintf(", ... (+%d more)", length(items) - max_display))
      } else {
        paste(items, collapse = ", ")
      }
    }
    message(sprintf(
      "%s[MF Promotion] Top %d/%d candidates promoted to full fidelity: %s",
      isl_pfx, length(promo_idx), length(pop_lf), format_cand_summary(promo_idx)
    ))
    if (length(unpromo_idx) > 0L) {
      message(sprintf(
        "%s[MF Filtered] %d candidates filtered out: %s",
        isl_pfx, length(unpromo_idx), format_cand_summary(unpromo_idx)
      ))
    }
  }

  # Pass 2: full-fidelity re-evaluation of promoted individuals only
  if (verbose) {
    ff_rows_lbl <- if (!is.na(ff_train_rows)) sprintf("%s rows", format(ff_train_rows, big.mark = ",")) else "100% data"
    message(sprintf("%s--- Full-Fidelity Evaluation (%d promoted candidates on %s) ---", isl_pfx, length(promo_idx), ff_rows_lbl))
  }

  promoted <- pop_lf[promo_idx]
  for (i in seq_along(promoted)) {
    orig_i <- promo_idx[i]
    if (!already_evaluated[orig_i]) {
      promoted[[i]]$fitness <- NA_real_
    }
  }
  res_ff <- .do_eval(promoted, shared_splits, shared_folds, shared_full, "", verbose, rb_in, indices = promo_idx)

  pop_out <- pop_lf
  pop_out[promo_idx] <- res_ff$pop

  unpromo_idx <- setdiff(seq_along(pop_out), promo_idx)
  for (idx in unpromo_idx) {
    pop_out[[idx]]$lf_fitness <- pop_out[[idx]]$fitness
    pop_out[[idx]]$fitness <- -Inf
    pop_out[[idx]]$raw_fitness <- NA_real_
  }

  fin_fits <- vapply(pop_out, function(x) {
    if (is.null(x$fitness) || is.na(x$fitness)) -Inf else x$fitness
  }, numeric(1))
  rb_out <- max(c(rb_in, fin_fits[is.finite(fin_fits)]))

  list(pop = pop_out, running_best_fitness = rb_out)
}

#' Tournament selection
