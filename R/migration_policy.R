#' Migration Policy Constructors and Generics for Island Models
#'
#' Functions to specify and resolve migration dynamics, target selection, and admission rules.
#'
#' @param temperature Numeric. Softmax temperature for Gibbs routing (default: 0.5).
#' @param weight_by Character. Criterion for Gibbs push weighting ("stagnation" or "fitness").
#' @param stagnation_threshold Integer. Stagnation generation threshold for Gibbs pull (default: 3).
#' @param min_fitness_threshold Character. Tier admission rule (default: "min_peer").
#' @param topology An \code{evo_topology} object or string name.
#' @param policy An \code{evo_policy} object or string name.
#' @param payload Character. Payload strategy: "gene_only" (default) or "full_individual".
#' @param state List containing current evolution state (pop_list, island_best_fitness, island_gens_without_improvement).
#' @param ... Additional arguments.
#'
#' @return An object of class \code{evo_policy} or \code{evo_migration_config}, or a transaction list.
#' @name migration_policy
NULL

.calc_feature_distance <- function(ind_j, ind_k) {
  if (is.null(ind_j) || is.null(ind_k) || length(ind_j$genes) == 0L || length(ind_k$genes) == 0L) {
    return(1.0)
  }
  f_j <- vapply(ind_j$genes, gene_to_formula, character(1))
  f_k <- vapply(ind_k$genes, gene_to_formula, character(1))
  union_len <- length(union(f_j, f_k))
  if (union_len == 0L) return(1.0)
  intersect_len <- length(intersect(f_j, f_k))
  1.0 - (intersect_len / union_len)
}

#' @rdname migration_policy
#' @export
policy_push_uniform <- function() {
  structure(
    list(type = "push_uniform"),
    class = c("evo_policy_push_uniform", "evo_policy")
  )
}

#' @rdname migration_policy
#' @export
policy_gibbs_push <- function(temperature = 0.5, weight_by = c("stagnation", "fitness", "feature_distance", "uniform")) {
  temperature <- as.numeric(temperature)
  if (is.na(temperature) || temperature <= 0) stop("temperature must be a positive numeric value")
  weight_by <- match.arg(weight_by)

  structure(
    list(type = "gibbs_push", temperature = temperature, weight_by = weight_by),
    class = c("evo_policy_gibbs_push", "evo_policy")
  )
}

#' @rdname migration_policy
#' @export
policy_gibbs_pull <- function(temperature = 0.5, stagnation_threshold = 3, weight_by = c("fitness", "feature_distance")) {
  temperature <- as.numeric(temperature)
  stagnation_threshold <- as.integer(stagnation_threshold)
  if (is.na(temperature) || temperature <= 0) stop("temperature must be a positive numeric value")
  if (is.na(stagnation_threshold) || stagnation_threshold < 1L) stop("stagnation_threshold must be a positive integer")
  weight_by <- match.arg(weight_by)

  structure(
    list(type = "gibbs_pull", temperature = temperature, stagnation_threshold = stagnation_threshold, weight_by = weight_by),
    class = c("evo_policy_gibbs_pull", "evo_policy")
  )
}

#' @rdname migration_policy
#' @export
policy_tiered_admission <- function(min_fitness_threshold = "min_peer") {
  structure(
    list(type = "tiered_admission", min_fitness_threshold = min_fitness_threshold),
    class = c("evo_policy_tiered_admission", "evo_policy")
  )
}

#' Configurable Container for Topology, Policy, and Payload Strategy
#' @rdname migration_policy
#' @export
migration_config <- function(topology = topology_ring(), policy = policy_push_uniform(), payload = "gene_only") {
  if (is.character(topology)) {
    topology <- switch(topology,
      "ring" = topology_ring(),
      "grid" = topology_grid(),
      "torus" = topology_torus(),
      "hypercube" = topology_hypercube(),
      "tiered" = topology_tiered(),
      "hfc" = topology_tiered(),
      "complete" = topology_complete(),
      "feature_distance" = topology_complete(),
      topology_ring()
    )
  }

  if (is.character(policy)) {
    policy <- switch(policy,
      "push_uniform" = policy_push_uniform(),
      "gibbs_stagnation" = policy_gibbs_push(weight_by = "stagnation"),
      "gibbs_fitness" = policy_gibbs_push(weight_by = "fitness"),
      "gibbs_feature_distance" = policy_gibbs_push(weight_by = "feature_distance"),
      "feature_distance" = policy_gibbs_push(weight_by = "feature_distance"),
      "dual_gibbs_pull" = policy_gibbs_pull(weight_by = "fitness"),
      "gibbs_pull_fitness" = policy_gibbs_pull(weight_by = "fitness"),
      "gibbs_pull_feature_distance" = policy_gibbs_pull(weight_by = "feature_distance"),
      "tiered_admission" = policy_tiered_admission(),
      policy_push_uniform()
    )
  }

  if (!inherits(topology, "evo_topology")) stop("topology must inherit from evo_topology")
  if (!inherits(policy, "evo_policy")) stop("policy must inherit from evo_policy")
  if (inherits(policy, "evo_policy_tiered_admission") && !inherits(topology, "evo_topology_tiered")) {
    stop("policy_tiered_admission can only be used with tiered topologies (topology_tiered).")
  }
  if (!payload %in% c("gene_only", "full_individual")) stop("payload must be 'gene_only' or 'full_individual'")

  structure(
    list(topology = topology, policy = policy, payload = payload),
    class = "evo_migration_config"
  )
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions <- function(policy, topology, state, ...) {
  UseMethod("resolve_migration_transactions")
}

# Helper to resolve vertical promotion in tiered topologies
.resolve_tiered_promotions <- function(topology, state, policy = NULL) {
  txs <- list()
  if (!inherits(topology, "evo_topology_tiered")) return(txs)
  t_all <- topology$tier_partition
  t_all <- t_all[vapply(t_all, function(x) length(x) > 0, logical(1))]
  N <- topology$islands

  threshold_mode <- "min_peer"
  if (!is.null(policy) && !is.null(policy$min_fitness_threshold)) {
    threshold_mode <- policy$min_fitness_threshold
  }

  for (j in 1:N) {
    tier_matches <- which(vapply(t_all, function(t_islands) j %in% t_islands, logical(1)))
    if (length(tier_matches) == 0L) next
    tier_idx <- tier_matches[1]
    if (tier_idx >= length(t_all)) next

    curr_tier_islands <- t_all[[tier_idx]]
    local_pos <- which(curr_tier_islands == j)
    from_t <- tier_idx - 1L

    next_tier_islands <- t_all[[tier_idx + 1L]]
    next_tier_fits <- state$island_best_fitness[next_tier_islands]
    next_tier_fits_clean <- next_tier_fits[!is.na(next_tier_fits)]

    if (length(next_tier_fits_clean) == 0L) {
      next_tier_thresh <- -Inf
    } else if (is.numeric(threshold_mode)) {
      next_tier_thresh <- threshold_mode
    } else {
      next_tier_thresh <- switch(threshold_mode,
        "min_peer"    = min(next_tier_fits_clean),
        "mean_peer"   = mean(next_tier_fits_clean),
        "median_peer" = stats::median(next_tier_fits_clean),
        min(next_tier_fits_clean)
      )
    }

    j_fit <- state$island_best_fitness[j]
    if (!is.na(j_fit) && j_fit > next_tier_thresh) {
      parent_idx <- min(length(next_tier_islands), ((local_pos - 1L) %/% 2L) + 1L)
      dest <- next_tier_islands[parent_idx]
      txs[[length(txs) + 1L]] <- list(
        from = j, to = dest, is_pull = FALSE, is_promotion = TRUE,
        from_tier = from_t, to_tier = from_t + 1L
      )
    }
  }
  txs
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_push_uniform <- function(policy, topology, state, ...) {
  policy_gibbs <- policy_gibbs_push(weight_by = "uniform")
  # Forward to evo_policy_gibbs_push using uniform weight strategy
  resolve_migration_transactions.evo_policy_gibbs_push(policy_gibbs, topology, state, ...)
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_gibbs_push <- function(policy, topology, state, ...) {
  txs <- list()
  N <- topology$islands
  if (N <= 1L) return(txs)

  temp <- policy$temperature
  weight_by <- policy$weight_by

  calc_weights <- function(j, nbrs) {
    if (weight_by == "uniform") {
      rep(1 / length(nbrs), length(nbrs))
    } else if (weight_by == "stagnation") {
      stags <- state$island_gens_without_improvement[nbrs]
      stags[is.na(stags)] <- 0
      max_s <- max(stags)
      logits <- (stags - max_s) / temp
      exp(logits) / sum(exp(logits))
    } else if (weight_by == "feature_distance") {
      ind_j <- if (!is.null(state$pop_list[[j]]) && length(state$pop_list[[j]]) > 0L) state$pop_list[[j]][[1]] else NULL
      dists <- vapply(nbrs, function(k) {
        ind_k <- if (!is.null(state$pop_list[[k]]) && length(state$pop_list[[k]]) > 0L) state$pop_list[[k]][[1]] else NULL
        .calc_feature_distance(ind_j, ind_k)
      }, numeric(1))
      max_d <- max(dists)
      logits <- (dists - max_d) / temp
      exp(logits) / sum(exp(logits))
    } else { # fitness
      fits <- state$island_best_fitness[nbrs]
      valid_fits <- fits[!is.na(fits)]
      fits[is.na(fits)] <- if (length(valid_fits) > 0L) min(valid_fits) else 0
      diffs <- max(fits) - fits
      logits <- (diffs - max(diffs)) / temp
      exp(logits) / sum(exp(logits))
    }
  }

  if (inherits(topology, "evo_topology_tiered")) {
    txs <- c(txs, .resolve_tiered_promotions(topology, state, policy))
    t_all <- topology$tier_partition
    for (t_islands in t_all) {
      if (length(t_islands) > 1L) {
        for (j in t_islands) {
          neighbors <- setdiff(t_islands, j)
          probs <- calc_weights(j, neighbors)
          if (any(is.na(probs)) || sum(probs) == 0) probs <- rep(1 / length(neighbors), length(neighbors))
          dest <- if (length(neighbors) == 1L) neighbors[1] else sample(neighbors, 1L, prob = probs)
          txs[[length(txs) + 1L]] <- list(from = j, to = dest, is_pull = FALSE)
        }
      }
    }
    return(txs)
  }

  for (j in 1:N) {
    neighbors <- get_neighbors(topology, j, state)
    if (length(neighbors) > 0L) {
      probs <- calc_weights(j, neighbors)
      if (any(is.na(probs)) || sum(probs) == 0) {
        probs <- rep(1 / length(neighbors), length(neighbors))
      }

      dest <- if (length(neighbors) == 1L) neighbors[1] else sample(neighbors, 1L, prob = probs)
      txs[[length(txs) + 1L]] <- list(from = j, to = dest, is_pull = FALSE)
    }
  }
  txs
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_gibbs_pull <- function(policy, topology, state, ...) {
  txs <- list()
  N <- topology$islands
  if (N <= 1L) return(txs)

  temp <- policy$temperature
  stag_thresh <- policy$stagnation_threshold
  weight_by <- policy$weight_by
  if (is.null(weight_by)) weight_by <- "fitness"

  calc_weights <- function(j, nbrs) {
    if (weight_by == "feature_distance") {
      ind_j <- if (!is.null(state$pop_list[[j]]) && length(state$pop_list[[j]]) > 0L) state$pop_list[[j]][[1]] else NULL
      dists <- vapply(nbrs, function(k) {
        ind_k <- if (!is.null(state$pop_list[[k]]) && length(state$pop_list[[k]]) > 0L) state$pop_list[[k]][[1]] else NULL
        .calc_feature_distance(ind_j, ind_k)
      }, numeric(1))
      max_d <- max(dists)
      logits <- (dists - max_d) / temp
      exp(logits) / sum(exp(logits))
    } else { # fitness
      donor_fits <- state$island_best_fitness[nbrs]
      donor_fits[is.na(donor_fits)] <- -Inf
      max_f <- max(donor_fits)
      if (is.finite(max_f)) {
        logits <- (donor_fits - max_f) / temp
        exp(logits) / sum(exp(logits))
      } else {
        rep(1 / length(nbrs), length(nbrs))
      }
    }
  }

  if (inherits(topology, "evo_topology_tiered")) {
    txs <- c(txs, .resolve_tiered_promotions(topology, state, policy))
    t_all <- topology$tier_partition
    for (t_islands in t_all) {
      if (length(t_islands) > 1L) {
        for (j in t_islands) {
          s_j <- state$island_gens_without_improvement[j]
          p_pull <- 1 / (1 + exp(-(s_j - stag_thresh) / temp))
          if (stats::runif(1) < p_pull) {
            candidates <- setdiff(t_islands, j)
            probs <- calc_weights(j, candidates)
            if (any(is.na(probs)) || sum(probs) == 0) probs <- rep(1 / length(candidates), length(candidates))
            donor <- if (length(candidates) == 1L) candidates[1] else sample(candidates, 1L, prob = probs)
            txs[[length(txs) + 1L]] <- list(from = donor, to = j, is_pull = TRUE)
          }
        }
      }
    }
    return(txs)
  }

  for (j in 1:N) {
    s_j <- state$island_gens_without_improvement[j]
    p_pull <- 1 / (1 + exp(-(s_j - stag_thresh) / temp))
    if (stats::runif(1) < p_pull) {
      candidates <- get_neighbors(topology, j, state)
      if (length(candidates) > 0L) {
        probs <- calc_weights(j, candidates)
        if (any(is.na(probs)) || sum(probs) == 0) {
          probs <- rep(1 / length(candidates), length(candidates))
        }

        donor <- if (length(candidates) == 1L) candidates[1] else sample(candidates, 1L, prob = probs)
        txs[[length(txs) + 1L]] <- list(from = donor, to = j, is_pull = TRUE)
      }
    }
  }
  txs
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_tiered_admission <- function(policy, topology, state, ...) {
  if (!inherits(topology, "evo_topology_tiered")) {
    stop("policy_tiered_admission can only be used with tiered topologies (topology_tiered).")
  }
  resolve_migration_transactions.evo_policy_push_uniform(policy, topology, state, ...)
}
