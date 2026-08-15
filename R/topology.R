#' Graph Topology Constructors and Generics for Island Models
#'
#' Functions to specify and query graph topologies connecting evolution islands.
#' Topologies define island adjacency ($G = (V, E)$) independent of migration movement policies.
#' Every topology object pre-builds and exposes an explicit \code{adj_list} and \code{adj_matrix}.
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

# Internal helper to populate adj_list and adj_matrix on any topology object
.populate_adjacency <- function(topology, neighbor_fn) {
  N <- topology$islands
  adj_list <- vector("list", N)
  mat <- matrix(0L, nrow = N, ncol = N)
  if (N > 0L) {
    rownames(mat) <- colnames(mat) <- paste0("island_", 1:N)
    for (i in 1:N) {
      nbrs <- as.integer(neighbor_fn(i))
      nbrs <- setdiff(unique(nbrs), i)
      nbrs <- nbrs[nbrs >= 1L & nbrs <= N]
      adj_list[[i]] <- nbrs
      if (length(nbrs) > 0L) {
        mat[i, nbrs] <- 1L
      }
    }
  }
  topology$adj_list <- adj_list
  topology$adj_matrix <- mat
  topology
}

#' @rdname topology
#' @export
topology_ring <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  top <- structure(
    list(islands = islands, type = "ring"),
    class = c("evo_topology_ring", "evo_topology")
  )
  .populate_adjacency(top, function(i) if (islands <= 1L) integer(0) else (i %% islands) + 1L)
}

.grid_neighbors_calc <- function(top, island_id) {
  N <- top$islands
  if (N <= 1L) return(integer(0))

  r_islands <- top$row_islands
  r_size <- length(r_islands)
  r_idx <- which(vapply(r_islands, function(isls) island_id %in% isls, logical(1)))
  if (length(r_idx) == 0L) return(setdiff(1:N, island_id))

  curr_row <- r_islands[[r_idx]]
  p <- which(curr_row == island_id)
  n_curr <- length(curr_row)

  neighbors <- integer(0)
  if (p > 1L) neighbors <- c(neighbors, curr_row[p - 1L])
  if (p < n_curr) neighbors <- c(neighbors, curr_row[p + 1L])

  get_vertical_neighbors <- function(target_row) {
    n_target <- length(target_row)
    if (n_target == 0L) return(integer(0))
    my_start <- (p - 1) / n_curr
    my_end <- p / n_curr
    res <- integer(0)
    for (k in seq_len(n_target)) {
      t_start <- (k - 1) / n_target
      t_end <- k / n_target
      if (min(my_end, t_end) - max(my_start, t_start) > 1e-9) {
        res <- c(res, target_row[k])
      }
    }
    res
  }

  if (r_idx > 1L) neighbors <- c(neighbors, get_vertical_neighbors(r_islands[[r_idx - 1L]]))
  if (r_idx < r_size) neighbors <- c(neighbors, get_vertical_neighbors(r_islands[[r_idx + 1L]]))
  unique(setdiff(neighbors, island_id))
}

.torus_neighbors_calc <- function(top, island_id) {
  N <- top$islands
  if (N <= 1L) return(integer(0))

  r_islands <- top$row_islands
  r_size <- length(r_islands)
  r_idx <- which(vapply(r_islands, function(isls) island_id %in% isls, logical(1)))
  if (length(r_idx) == 0L) return(setdiff(1:N, island_id))

  curr_row <- r_islands[[r_idx]]
  p <- which(curr_row == island_id)
  n_curr <- length(curr_row)

  neighbors <- integer(0)

  # Horizontal neighbors (with wrap-around)
  if (n_curr > 1L) {
    left_p <- if (p == 1L) n_curr else p - 1L
    right_p <- if (p == n_curr) 1L else p + 1L
    neighbors <- c(neighbors, curr_row[left_p], curr_row[right_p])
  }

  get_vertical_neighbors <- function(target_row) {
    n_target <- length(target_row)
    if (n_target == 0L) return(integer(0))
    my_start <- (p - 1) / n_curr
    my_end <- p / n_curr
    res <- integer(0)
    for (k in seq_len(n_target)) {
      t_start <- (k - 1) / n_target
      t_end <- k / n_target
      if (min(my_end, t_end) - max(my_start, t_start) > 1e-9) {
        res <- c(res, target_row[k])
      }
    }
    res
  }

  # Vertical neighbors (with wrap-around)
  if (r_size > 1L) {
    up_r <- if (r_idx == 1L) r_size else r_idx - 1L
    down_r <- if (r_idx == r_size) 1L else r_idx + 1L
    neighbors <- c(neighbors, get_vertical_neighbors(r_islands[[up_r]]))
    if (down_r != up_r) {
      neighbors <- c(neighbors, get_vertical_neighbors(r_islands[[down_r]]))
    }
  }

  unique(setdiff(neighbors, island_id))
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
    r_size <- if (!is.null(rows)) as.integer(rows) else max(1L, as.integer(round(sqrt(islands))))
    c_size <- if (!is.null(cols)) as.integer(cols) else as.integer(ceiling(islands / r_size))
    
    base_per_row <- islands %/% r_size
    rem <- islands %% r_size
    row_counts <- rep(base_per_row, r_size)
    if (rem > 0) {
      mid <- (r_size + 1) / 2
      offset_order <- order(abs((1:r_size) - mid))
      for (k in 1:rem) {
        idx <- offset_order[k]
        row_counts[idx] <- row_counts[idx] + 1L
      }
    }
  }

  row_islands <- list()
  curr <- 1L
  for (r in 1:r_size) {
    n_r <- row_counts[r]
    row_islands[[r]] <- curr:(curr + n_r - 1L)
    curr <- curr + n_r
  }

  top <- structure(
    list(islands = islands, type = "grid", rows = r_size, cols = c_size, row_counts = row_counts, row_islands = row_islands),
    class = c("evo_topology_grid", "evo_topology")
  )
  .populate_adjacency(top, function(i) .grid_neighbors_calc(top, i))
}

#' @rdname topology
#' @export
topology_torus <- function(islands = 10, rows = NULL, cols = NULL) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")

  if (islands == 1L) {
    r_size <- 1L
    c_size <- 1L
    row_counts <- 1L
  } else {
    r_size <- if (!is.null(rows)) as.integer(rows) else max(1L, as.integer(round(sqrt(islands))))
    c_size <- if (!is.null(cols)) as.integer(cols) else as.integer(ceiling(islands / r_size))
    
    base_per_row <- islands %/% r_size
    rem <- islands %% r_size
    row_counts <- rep(base_per_row, r_size)
    if (rem > 0) {
      mid <- (r_size + 1) / 2
      offset_order <- order(abs((1:r_size) - mid))
      for (k in 1:rem) {
        idx <- offset_order[k]
        row_counts[idx] <- row_counts[idx] + 1L
      }
    }
  }

  row_islands <- list()
  curr <- 1L
  for (r in 1:r_size) {
    n_r <- row_counts[r]
    row_islands[[r]] <- curr:(curr + n_r - 1L)
    curr <- curr + n_r
  }

  top <- structure(
    list(islands = islands, type = "torus", rows = r_size, cols = c_size, row_counts = row_counts, row_islands = row_islands),
    class = c("evo_topology_torus", "evo_topology_grid", "evo_topology")
  )
  .populate_adjacency(top, function(i) .torus_neighbors_calc(top, i))
}

.hypercube_neighbors_calc <- function(top, island_id) {
  N <- top$islands
  if (N <= 1L) return(integer(0))
  dim_val <- top$dimension
  mask <- 2^(0:(dim_val - 1L))
  neighbors <- (bitwXor(island_id - 1L, mask) %% N) + 1L
  setdiff(unique(neighbors), island_id)
}

#' @rdname topology
#' @export
topology_hypercube <- function(islands = NULL, dimension = NULL) {
  if (is.null(islands) && is.null(dimension)) {
    islands <- 8L
    dimension <- 3L
  } else if (!is.null(dimension) && is.null(islands)) {
    dimension <- as.integer(dimension)
    if (is.na(dimension) || dimension < 1L) stop("dimension must be a positive integer >= 1")
    islands <- 2L^dimension
  } else if (is.null(dimension) && !is.null(islands)) {
    islands <- as.integer(islands)
    if (is.na(islands) || islands < 1L) stop("islands must be a positive integer >= 1")
    dimension <- max(1L, as.integer(ceiling(log2(islands))))
  } else {
    islands <- as.integer(islands)
    dimension <- as.integer(dimension)
  }

  top <- structure(
    list(islands = as.integer(islands), type = "hypercube", dimension = as.integer(dimension)),
    class = c("evo_topology_hypercube", "evo_topology")
  )
  .populate_adjacency(top, function(i) .hypercube_neighbors_calc(top, i))
}

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

  weights <- 0.5^(0:(tiers - 1L))
  probs <- weights / sum(weights)
  counts <- pmax(1L, as.integer(round(probs * islands)))

  diff_total <- islands - sum(counts)
  if (diff_total > 0) {
    counts[1] <- counts[1] + diff_total
  } else if (diff_total < 0) {
    for (k in tiers:1) {
      reduce <- min(counts[k] - 1L, -diff_total)
      counts[k] <- counts[k] - reduce
      diff_total <- diff_total + reduce
      if (diff_total == 0) break
    }
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

.tiered_neighbors_calc <- function(top, island_id) {
  N <- top$islands
  if (N <= 1L) return(integer(0))

  t_all <- top$tier_partition
  t_all <- t_all[vapply(t_all, function(x) length(x) > 0, logical(1))]

  tier_matches <- which(vapply(t_all, function(t_islands) island_id %in% t_islands, logical(1)))
  if (length(tier_matches) == 0L) return(setdiff(1:N, island_id))
  tier_idx <- tier_matches[1]

  curr_tier_islands <- t_all[[tier_idx]]
  local_pos <- which(curr_tier_islands == island_id)
  peers <- setdiff(curr_tier_islands, island_id)

  upward <- integer(0)
  if (tier_idx < length(t_all)) {
    next_tier_islands <- t_all[[tier_idx + 1L]]
    parent_idx <- min(length(next_tier_islands), ((local_pos - 1L) %/% 2L) + 1L)
    upward <- next_tier_islands[parent_idx]
  }

  unique(c(peers, upward))
}

#' @rdname topology
#' @export
topology_tiered <- function(islands = 10, tiers = 3) {
  islands <- as.integer(islands)
  tiers <- as.integer(tiers)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  if (is.na(tiers) || tiers < 2L) stop("tiers must be an integer >= 2")

  tier_partition <- .partition_k_tiers(islands, tiers)
  top <- structure(
    list(islands = islands, type = "tiered", tiers = length(tier_partition), tier_partition = tier_partition),
    class = c("evo_topology_tiered", "evo_topology")
  )
  .populate_adjacency(top, function(i) .tiered_neighbors_calc(top, i))
}

#' @rdname topology
#' @export
topology_complete <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  top <- structure(
    list(islands = islands, type = "complete"),
    class = c("evo_topology_complete", "evo_topology")
  )
  .populate_adjacency(top, function(i) setdiff(1:islands, i))
}

topology_feature_distance <- function(islands = 10) {
  islands <- as.integer(islands)
  if (is.na(islands) || islands < 1L) stop("islands must be a positive integer")
  top <- structure(
    list(islands = islands, type = "feature_distance"),
    class = c("evo_topology_feature_distance", "evo_topology")
  )
  .populate_adjacency(top, function(i) setdiff(1:islands, i))
}

#' Custom Graph Topology Constructor
#' @param adj N x N binary matrix or list of integer vectors specifying adjacency.
#' @return Object of class \code{evo_topology_custom}.
#' @export
topology_custom <- function(adj) {
  if (is.matrix(adj)) {
    N <- nrow(adj)
    mat <- (adj != 0L) * 1L
    rownames(mat) <- colnames(mat) <- paste0("island_", 1:N)
    adj_list <- lapply(1:N, function(i) which(mat[i, ] != 0L & (1:N) != i))
  } else if (is.list(adj)) {
    N <- length(adj)
    adj_list <- lapply(adj, as.integer)
    mat <- matrix(0L, nrow = N, ncol = N)
    rownames(mat) <- colnames(mat) <- paste0("island_", 1:N)
    for (i in 1:N) {
      nbrs <- adj_list[[i]]
      if (length(nbrs) > 0L) mat[i, nbrs] <- 1L
    }
  } else {
    stop("adj must be an adjacency matrix or list")
  }

  structure(
    list(islands = as.integer(N), type = "custom", adj_list = adj_list, adj_matrix = mat),
    class = c("evo_topology_custom", "evo_topology")
  )
}

#' @rdname topology
#' @export
get_neighbors <- function(topology, island_id, state = NULL, ...) {
  UseMethod("get_neighbors")
}

#' @rdname topology
#' @export
get_neighbors.evo_topology <- function(topology, island_id, state = NULL, ...) {
  if (!is.null(topology$adj_list) && island_id >= 1L && island_id <= length(topology$adj_list)) {
    return(topology$adj_list[[island_id]])
  }
  integer(0)
}

#' @rdname topology
#' @export
get_in_neighbors <- function(topology, island_id, state = NULL, ...) {
  UseMethod("get_in_neighbors")
}

#' @rdname topology
#' @export
get_in_neighbors.evo_topology <- function(topology, island_id, state = NULL, ...) {
  mat <- as.matrix(topology)
  N <- topology$islands
  if (island_id >= 1L && island_id <= N) {
    return(unname(which(mat[, island_id] == 1L)))
  }
  integer(0)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_ring <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_grid <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_torus <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_hypercube <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_tiered <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_complete <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' @rdname topology
#' @export
get_neighbors.evo_topology_custom <- function(topology, island_id, state = NULL, ...) {
  get_neighbors.evo_topology(topology, island_id, state, ...)
}

#' Convert evo_topology to Adjacency Matrix
#' @param x An evo_topology object.
#' @param ... Unused.
#' @return N x N integer adjacency matrix.
#' @export
#' @exportS3Method as.matrix evo_topology
as.matrix.evo_topology <- function(x, ...) {
  if (!is.null(x$adj_matrix)) {
    return(x$adj_matrix)
  }
  N <- x$islands
  mat <- matrix(0L, nrow = N, ncol = N)
  if (!is.null(x$adj_list)) {
    for (i in 1:N) {
      nbrs <- x$adj_list[[i]]
      if (length(nbrs) > 0L) mat[i, nbrs] <- 1L
    }
  }
  rownames(mat) <- colnames(mat) <- paste0("island_", 1:N)
  mat
}
