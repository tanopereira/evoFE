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
policy_gibbs_push <- function(temperature = 0.5, weight_by = c("stagnation", "fitness")) {
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
policy_gibbs_pull <- function(temperature = 0.5, stagnation_threshold = 3) {
  temperature <- as.numeric(temperature)
  stagnation_threshold <- as.integer(stagnation_threshold)
  if (is.na(temperature) || temperature <= 0) stop("temperature must be a positive numeric value")
  if (is.na(stagnation_threshold) || stagnation_threshold < 1L) stop("stagnation_threshold must be a positive integer")

  structure(
    list(type = "gibbs_pull", temperature = temperature, stagnation_threshold = stagnation_threshold),
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
      "hypercube" = topology_hypercube(),
      "tiered" = topology_tiered(),
      "hfc" = topology_tiered(),
      "complete" = topology_complete(),
      "feature_distance" = topology_feature_distance(),
      topology_ring()
    )
  }

  if (is.character(policy)) {
    policy <- switch(policy,
      "push_uniform" = policy_push_uniform(),
      "gibbs_stagnation" = policy_gibbs_push(weight_by = "stagnation"),
      "gibbs_fitness" = policy_gibbs_push(weight_by = "fitness"),
      "dual_gibbs_pull" = policy_gibbs_pull(),
      "tiered_admission" = policy_tiered_admission(),
      policy_push_uniform()
    )
  }

  if (!inherits(topology, "evo_topology")) stop("topology must inherit from evo_topology")
  if (!inherits(policy, "evo_policy")) stop("policy must inherit from evo_policy")
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

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_push_uniform <- function(policy, topology, state, ...) {
  txs <- list()
  N <- topology$islands
  if (N <= 1L) return(txs)

  for (j in 1:N) {
    neighbors <- get_neighbors(topology, j, state)
    if (length(neighbors) > 0L) {
      dest <- if (length(neighbors) == 1L) neighbors[1] else sample(neighbors, 1L)
      txs[[length(txs) + 1L]] <- list(from = j, to = dest, is_pull = FALSE)
    }
  }
  txs
}

#' @rdname migration_policy
#' @export
resolve_migration_transactions.evo_policy_gibbs_push <- function(policy, topology, state, ...) {
  txs <- list()
  N <- topology$islands
  if (N <= 1L) return(txs)

  temp <- policy$temperature
  weight_by <- policy$weight_by

  for (j in 1:N) {
    neighbors <- get_neighbors(topology, j, state)
    if (length(neighbors) > 0L) {
      if (weight_by == "stagnation") {
        stags <- state$island_gens_without_improvement[neighbors]
        stags[is.na(stags)] <- 0
        max_s <- max(stags)
        logits <- (stags - max_s) / temp
        probs <- exp(logits) / sum(exp(logits))
      } else { # fitness
        fits <- state$island_best_fitness[neighbors]
        valid_fits <- fits[!is.na(fits)]
        if (length(valid_fits) > 0L) {
          fits[is.na(fits)] <- min(valid_fits)
        } else {
          fits[] <- 0
        }
        max_f <- max(fits)
        diffs <- max_f - fits
        max_d <- max(diffs)
        if (all(diffs == 0)) {
          probs <- rep(1 / length(neighbors), length(neighbors))
        } else {
          logits <- (diffs - max_d) / temp
          probs <- exp(logits) / sum(exp(logits))
        }
      }
      
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

  for (j in 1:N) {
    s_j <- state$island_gens_without_improvement[j]
    p_pull <- 1 / (1 + exp(-(s_j - stag_thresh) / temp))
    if (stats::runif(1) < p_pull) {
      candidates <- setdiff(1:N, j)
      if (length(candidates) > 0L) {
        donor_fits <- state$island_best_fitness[candidates]
        donor_fits[is.na(donor_fits)] <- -Inf
        max_f <- max(donor_fits)
        if (is.finite(max_f)) {
          logits <- (donor_fits - max_f) / temp
          probs <- exp(logits) / sum(exp(logits))
        } else {
          probs <- rep(1 / length(candidates), length(candidates))
        }
        
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
  txs <- list()
  N <- topology$islands
  if (N <= 1L) return(txs)

  if (!inherits(topology, "evo_topology_tiered")) {
    topology <- topology_tiered(N)
  }

  t_all <- topology$tier_partition
  t_all <- t_all[vapply(t_all, function(x) length(x) > 0, logical(1))]

  for (j in 1:N) {
    tier_idx <- which(vapply(t_all, function(t_islands) j %in% t_islands, logical(1)))
    if (length(tier_idx) == 0L) next

    curr_tier_islands <- t_all[[tier_idx]]
    local_pos <- which(curr_tier_islands == j)
    from_t <- tier_idx - 1L

    # 1. Vertical Promotion check to next higher tier
    if (tier_idx < length(t_all)) {
      next_tier_islands <- t_all[[tier_idx + 1L]]
      next_tier_fits <- state$island_best_fitness[next_tier_islands]
      min_next_fit <- if (all(is.na(next_tier_fits))) -Inf else min(next_tier_fits, na.rm = TRUE)

      j_fit <- state$island_best_fitness[j]
      if (!is.na(j_fit) && j_fit >= min_next_fit) {
        target_size <- length(next_tier_islands)
        slot_idx <- ((local_pos - 1L) %/% ceiling(length(curr_tier_islands) / target_size)) + 1L
        target_slot <- min(slot_idx, target_size)
        dest <- next_tier_islands[target_slot]
        txs[[length(txs) + 1L]] <- list(
          from = j, to = dest, is_pull = FALSE, is_promotion = TRUE,
          from_tier = from_t, to_tier = from_t + 1L
        )
      }
    }

    # 2. Horizontal Intra-Tier Peer Migration
    if (length(curr_tier_islands) > 1L) {
      candidates <- setdiff(curr_tier_islands, j)
      dest <- if (length(candidates) == 1L) candidates[1] else sample(candidates, 1L)
      txs[[length(txs) + 1L]] <- list(
        from = j, to = dest, is_pull = FALSE, is_promotion = FALSE,
        from_tier = from_t, to_tier = from_t
      )
    }
  }
  txs
}
