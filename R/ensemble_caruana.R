#' Internal Caruana Greedy Selection Engine
#' @keywords internal
#' @noRd
caruana_select <- function(y_true, val_preds_list, task, metric, rounds = 50,
                           patience = 15, bag_samples = FALSE, bag_bags = 5,
                           sample_ratio = 0.8, seed = NULL,
                           num_class = NULL, verbose = FALSE) {

  n_candidates <- length(val_preds_list)
  candidate_names <- names(val_preds_list)
  n_obs <- if (is.matrix(val_preds_list[[1]])) nrow(val_preds_list[[1]]) else length(val_preds_list[[1]])
  w_fmt <- nchar(as.character(rounds))

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

  blend_preds <- function(current_sum, new_preds, count) {
    if (count == 0) return(new_preds)
    (current_sum * count + new_preds) / (count + 1)
  }

  run_with_seed <- function(seed_val, code) {
    with_seed(seed_val, code)
  }

  run_greedy_trajectory <- function(y_eval, preds_eval, max_rounds, early_patience, is_verbose = FALSE) {
    init_scores <- vapply(candidate_names, function(nm) eval_fitness(y_eval, preds_eval[[nm]]), double(1))
    best_init_idx <- which.max(init_scores)
    if (length(best_init_idx) == 0 || is.na(best_init_idx)) best_init_idx <- 1L
    best_init_name <- candidate_names[best_init_idx]

    counts <- stats::setNames(rep(0L, n_candidates), candidate_names)
    counts[best_init_name] <- 1L
    current_blend <- preds_eval[[best_init_name]]
    current_score <- init_scores[best_init_idx]

    best_score <- current_score
    best_counts <- counts
    best_step <- 1L
    no_improve <- 0L

    if (is_verbose) {
      message(sprintf("  [Round %*d/%d] Initialized with %s -> Fitness: %.4f", w_fmt, 1, max_rounds, best_init_name, current_score))
    }

    history <- data.frame(
      round = 1:max_rounds,
      selected_model = character(max_rounds),
      fitness = numeric(max_rounds),
      stringsAsFactors = FALSE
    )
    history$selected_model[1] <- best_init_name
    history$fitness[1] <- current_score

    if (max_rounds > 1) {
      for (r in 2:max_rounds) {
        best_cand <- NULL
        best_cand_score <- -Inf
        best_cand_blend <- NULL

        for (nm in candidate_names) {
          cand_p <- preds_eval[[nm]]
          cand_blend <- blend_preds(current_blend, cand_p, r - 1)
          score <- eval_fitness(y_eval, cand_blend)
          if (score > best_cand_score) {
            best_cand_score <- score
            best_cand <- nm
            best_cand_blend <- cand_blend
          }
        }

        if (!is.null(best_cand)) {
          counts[best_cand] <- counts[best_cand] + 1L
          current_blend <- best_cand_blend
          current_score <- best_cand_score

          if (current_score > best_score + 1e-7) {
            best_score <- current_score
            best_counts <- counts
            best_step <- r
            no_improve <- 0L
          } else {
            no_improve <- no_improve + 1L
          }

          if (is_verbose) {
            imp_tag <- if (current_score >= best_score) " (Improved!)" else ""
            message(sprintf("  [Round %*d/%d] Selected %s -> Ensemble Fitness: %.4f%s", w_fmt, r, max_rounds, best_cand, current_score, imp_tag))
          }
        }

        history$selected_model[r] <- if (!is.null(best_cand)) best_cand else best_init_name
        history$fitness[r] <- current_score

        if (no_improve >= early_patience) {
          if (is_verbose) {
            message(sprintf("  Early stopping at round %d (no improvement for %d rounds; best fitness: %.4f at round %d).",
                            r, early_patience, best_score, best_step))
          }
          history <- history[1:r, , drop = FALSE]
          break
        }
      }
    }

    list(weights = best_counts / sum(best_counts), best_score = best_score, best_step = best_step, history = history)
  }

  selection_res <- if (!bag_samples || bag_bags <= 1) {
    run_with_seed(seed, function() {
      run_greedy_trajectory(y_true, val_preds_list, rounds, patience, is_verbose = verbose)
    })
  } else {
    run_with_seed(seed, function() {
      bag_w_list <- vector("list", bag_bags)
      if (verbose) {
        message(sprintf("  Running Bagged Caruana Ensemble Selection over %d bootstrap bags...", bag_bags))
      }
      for (b in seq_len(bag_bags)) {
        boot_idx <- sample(seq_len(n_obs), size = max(2L, min(n_obs, round(n_obs * sample_ratio))), replace = TRUE)
        y_boot <- if (is.matrix(y_true)) y_true[boot_idx, , drop = FALSE] else y_true[boot_idx]
        preds_boot <- lapply(val_preds_list, function(p) {
          if (is.matrix(p)) p[boot_idx, , drop = FALSE] else p[boot_idx]
        })
        res_b <- run_greedy_trajectory(y_boot, preds_boot, rounds, patience, is_verbose = FALSE)
        bag_w_list[[b]] <- res_b$weights
      }
      avg_w <- Reduce("+", bag_w_list) / bag_bags
      list(weights = avg_w, history = NULL)
    })
  }

  final_w <- selection_res$weights
  final_blend <- NULL
  for (nm in candidate_names) {
    w <- final_w[[nm]]
    if (w <= 0) next
    p <- val_preds_list[[nm]]
    final_blend <- if (is.null(final_blend)) w * p else final_blend + w * p
  }
  final_fitness <- eval_fitness(y_true, final_blend)

  list(
    weights = final_w,
    history = selection_res$history,
    final_fitness = final_fitness
  )
}
