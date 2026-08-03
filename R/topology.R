#' Graph Topology Constructors and Generics for Island Models
#'
#' Functions to specify and query graph topologies connecting evolution islands.
#' Topologies define island adjacency ($G = (V, E)$) independent of migration movement policies.
#'
#' @param islands Integer. Number of active islands (default: 10).
#' @param rows Optional integer. Number of grid rows.
#' @param cols Optional integer. Number of grid columns.
#' @param dimension Optional integer. Hypercube dimension.
#' @param tiers Integer. Number of tiers for pyramid topology (default: 3).
#' @param topology An \code{evo_topology} object.
#' @param island_id Integer. 1-indexed island ID (1 to \code{islands}).
#' @param state Optional list containing runtime evolution state (populations, fitnesses, stagnation counts).
#' @param ... Additional arguments passed to methods.
#'
#' @return An object of class \code{evo_topology} (and specific subclass), or neighbor node IDs.
#' @name topology
NULL

#' @rdname topology
#' @export
topology_ring <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  structure(
    list(islands = islands, type = "ring"),
    class = c("evo_topology_ring", "evo_topology")
  )
}

#' @rdname topology
#' @export
topology_grid <- function(islands = 10, rows = NULL, cols = NULL) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")

  if (islands == 1L) {
    r_size <- 1L
    c_size <- 1L
    row_counts <- 1L
  } else {
    r_size <- if (!is.null(rows)) as.integer(rows) else max(1L, as.integer(floor(sqrt(islands))))
    c_size <- if (!is.null(cols)) as.integer(cols) else as.integer(ceiling(islands / r_size))
    
    # Symmetrical Staggered Row Distribution for non-square island counts
    base_per_row <- islands %/% r_size
    rem <- islands %% r_size
    row_counts <- rep(base_per_row, r_size)
    if (rem > 0) {
      # Distribute remainders starting from middle row outward
      mid <- (r_size + 1L) %/% 2L
      offset_order <- order(abs((1:r_size) - mid))
      for (k in 1:rem) {
        idx <- offset_order[k]
        row_counts[idx] <- row_counts[idx] + 1L
      }
    }
  }

  structure(
    list(islands = islands, type = "grid", rows = r_size, cols = c_size, row_counts = row_counts),
    class = c("evo_topology_grid", "evo_topology")
  )
}

#' @rdname topology
#' @export
topology_hypercube <- function(islands = 10, dimension = NULL) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")

  dim_val <- if (!is.null(dimension)) {
    as.integer(dimension)
  } else {
    max(1L, as.integer(floor(log2(islands))))
  }

  structure(
    list(islands = islands, type = "hypercube", dimension = dim_val),
    class = c("evo_topology_hypercube", "evo_topology")
  )
}

#' Partition N islands into 3 HFC tiers (Entry 50%, Intermediate 25%, Apex 25%)
#' @param islands Integer. Total number of active islands.
#' @return List with components \code{tier0}, \code{tier1}, and \code{tier2}.
#' @keywords internal
.partition_hfc_tiers <- function(islands) {
  if (islands <= 1) {
    return(list(tier0 = 1:islands, tier1 = integer(0), tier2 = integer(0)))
  } else if (islands == 2) {
    return(list(tier0 = 1L, tier1 = integer(0), tier2 = 2L))
  } else if (islands == 3) {
    return(list(tier0 = 1L, tier1 = 2L, tier2 = 3L))
  } else if (islands == 4) {
    return(list(tier0 = 1:2, tier1 = 3L, tier2 = 4L))
  } else {
    n_tier0 <- max(1L, as.integer(ceiling(0.50 * islands)))
    rem <- islands - n_tier0
    n_tier1 <- max(1L, as.integer(ceiling(0.60 * rem)))
    n_tier2 <- max(1L, islands - (n_tier0 + n_tier1))

    while (n_tier1 < n_tier2 && n_tier1 > 1) {
      n_tier1 <- n_tier1 + 1L
      n_tier2 <- n_tier2 - 1L
    }
    while (n_tier0 < n_tier1 && n_tier0 > 1) {
      n_tier0 <- n_tier0 + 1L
      n_tier1 <- n_tier1 - 1L
    }

    t0 <- 1:n_tier0
    t1 <- (n_tier0 + 1):(n_tier0 + n_tier1)
    t2 <- (n_tier0 + n_tier1 + 1):islands
    list(tier0 = t0, tier1 = t1, tier2 = t2)
  }
}

#' Partition N islands into K tiers with geometric decay & monotonicity constraint
#' @param islands Integer. Number of active islands.
#' @param tiers Integer. Number of tiers requested.
#' @return List of tier integer vectors.
#' @keywords internal
.partition_k_tiers <- function(islands, tiers = 3) {
  islands <- as.integer(islands)
  tiers <- max(2L, min(as.integer(tiers), islands))

  if (islands <= 1L) {
    res <- vector("list", tiers)
    res[[1]] <- 1L
    for (k in 2:tiers) res[[k]] <- integer(0)
    names(res) <- paste0("tier", 0:(tiers - 1L))
    return(res)
  }

  if (tiers == 3L) {
    return(.partition_hfc_tiers(islands))
  }

  # Geometric weights: w_k = 0.5^k
  weights <- 0.5^(0:(tiers - 1L))
  probs <- weights / sum(weights)
  counts <- pmax(1L, as.integer(round(probs * islands)))

  diff_total <- islands - sum(counts)
  if (diff_total != 0) {
    counts[1] <- max(1L, counts[1] + diff_total)
  }

  res <- vector("list", tiers)
  curr_start <- 1L
  for (k in 1:tiers) {
    n_k <- counts[k]
    if (n_k > 0 && curr_start <= islands) {
      curr_end <- min(islands, curr_start + n_k - 1L)
      res[[k]] <- curr_start:curr_end
      curr_start <- curr_end + 1L
    } else {
      res[[k]] <- integer(0)
    }
  }
  names(res) <- paste0("tier", 0:(tiers - 1L))
  res
}

#' @rdname topology
#' @export
topology_tiered <- function(islands = 10, tiers = 3) {
  islands <- as.integer(islands)
  tiers <- as.integer(tiers)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  if (is.na(tiers) || tiers < 2L) stop("tiers must be an integer >= 2")

  tier_partition <- .partition_k_tiers(islands, tiers)

  structure(
    list(islands = islands, type = "tiered", tiers = length(tier_partition), tier_partition = tier_partition),
    class = c("evo_topology_tiered", "evo_topology")
  )
}

#' @rdname topology
#' @export
topology_complete <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  structure(
    list(islands = islands, type = "complete"),
    class = c("evo_topology_complete", "evo_topology")
  )
}

#' @rdname topology
#' @export
topology_feature_distance <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  structure(
    list(islands = islands, type = "feature_distance"),
    class = c("evo_topology_feature_distance", "evo_topology")
  )
}

#' @rdname topology
#' @export
get_neighbors <- function(topology, island_id, state = NULL, ...) {
  UseMethod("get_neighbors")
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_ring <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))
  dest <- (island_id %% N) + 1L
  setdiff(dest, island_id)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_grid <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))

  r_size <- topology$rows
  c_size <- topology$cols
  r <- (island_id - 1L) %/% c_size
  c <- (island_id - 1L) %% c_size

  n_r <- (r - 1L + r_size) %% r_size
  s_r <- (r + 1L) %% r_size
  e_c <- (c + 1L) %% c_size
  w_c <- (c - 1L + c_size) %% c_size

  n_idx <- n_r * c_size + c + 1L
  s_idx <- s_r * c_size + c + 1L
  e_idx <- r * c_size + e_c + 1L
  w_idx <- r * c_size + w_c + 1L

  grid_neighbors <- setdiff(unique(c(n_idx, s_idx, e_idx, w_idx)), island_id)
  grid_neighbors <- grid_neighbors[grid_neighbors <= N]

  if (length(grid_neighbors) == 0L) {
    grid_neighbors <- setdiff(1:N, island_id)
  }
  grid_neighbors
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_hypercube <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))

  dim_val <- topology$dimension
  mask <- 2^(0:(dim_val - 1L))
  neighbors <- (bitwXor(island_id - 1L, mask) %% N) + 1L
  neighbors <- setdiff(unique(neighbors), island_id)
  if (length(neighbors) == 0L) {
    neighbors <- setdiff(1:N, island_id)
  }
  neighbors
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_tiered <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))

  t_all <- topology$tier_partition
  t_all <- t_all[vapply(t_all, function(x) length(x) > 0, logical(1))]

  tier_idx <- which(vapply(t_all, function(t_islands) island_id %in% t_islands, logical(1)))
  if (length(tier_idx) == 0L) return(setdiff(1:N, island_id))

  curr_tier_islands <- t_all[[tier_idx]]
  peers <- setdiff(curr_tier_islands, island_id)

  upward <- integer(0)
  if (tier_idx < length(t_all)) {
    upward <- t_all[[tier_idx + 1L]]
  }

  unique(c(peers, upward))
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_complete <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))
  setdiff(1:N, island_id)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_feature_distance <- function(topology, island_id, state = NULL, ...) {
  N <- topology$islands
  if (N <= 1L) return(integer(0))
  setdiff(1:N, island_id)
}
