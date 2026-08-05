test_that("topology constructors initialize correctly and set expected S3 classes", {
  ring <- topology_ring(10)
  expect_s3_class(ring, "evo_topology_ring")
  expect_equal(ring$islands, 10L)

  grid <- topology_grid(10)
  expect_s3_class(grid, "evo_topology_grid")
  expect_equal(grid$islands, 10L)
  expect_equal(grid$rows, 3L)
  expect_equal(grid$cols, 4L)
  expect_equal(grid$row_counts, c(3L, 4L, 3L)) # 3-4-3 symmetrical lattice

  cube <- topology_hypercube(10)
  expect_s3_class(cube, "evo_topology_hypercube")
  expect_equal(cube$islands, 10L)
  expect_equal(cube$dimension, 4L)

  tiered <- topology_tiered(10, tiers = 3)
  expect_s3_class(tiered, "evo_topology_tiered")
  expect_equal(tiered$islands, 10L)
  expect_equal(length(tiered$tier_partition), 3L)

  tiered4 <- topology_tiered(10, tiers = 4)
  expect_equal(length(tiered4$tier_partition), 4L)

  comp <- topology_complete(10)
  expect_s3_class(comp, "evo_topology_complete")

  feat <- evoFE:::topology_feature_distance(10)
  expect_s3_class(feat, "evo_topology_feature_distance")
})

test_that("get_neighbors handles single island (N=1) without error across all topologies", {
  topos <- list(
    topology_ring(1),
    topology_grid(1),
    topology_hypercube(1),
    topology_tiered(1, 3),
    topology_complete(1),
    evoFE:::topology_feature_distance(1)
  )

  for (topo in topos) {
    nb <- get_neighbors(topo, 1L)
    expect_equal(nb, integer(0))
  }
})

test_that("get_neighbors resolves valid spatial neighbors for prime island grid (N=7)", {
  grid7 <- topology_grid(7)
  expect_equal(grid7$rows, 3L)
  expect_equal(grid7$cols, 3L)

  for (i in 1:7) {
    nb <- get_neighbors(grid7, i)
    expect_true(length(nb) > 0)
    expect_false(i %in% nb)
    expect_true(all(nb >= 1 & nb <= 7))
  }
})

test_that("non-power-of-2 hypercube (N=10) resolves valid neighbors without self-loops", {
  cube10 <- topology_hypercube(10)
  for (i in 1:10) {
    nb <- get_neighbors(cube10, i)
    expect_true(length(nb) > 0)
    expect_false(i %in% nb)
    expect_true(all(nb >= 1 & nb <= 10))
  }
})

test_that("partition_k_tiers allocates monotonic pyramid tiers for various N and K", {
  part3 <- .partition_k_tiers(10, 3)
  expect_equal(length(part3$tier0), 6)
  expect_equal(length(part3$tier1), 3)
  expect_equal(length(part3$tier2), 1)

  part4 <- .partition_k_tiers(20, 4)
  expect_equal(sum(vapply(part4, length, integer(1))), 20)

  # Edge case: more tiers than islands (N=3, K=5)
  part_clamp <- .partition_k_tiers(3, 5)
  expect_equal(sum(vapply(part_clamp, length, integer(1))), 3)
})

test_that("migration policies and transaction resolution dispatch correctly", {
  state <- list(
    pop_list = vector("list", 10),
    island_best_fitness = rep(0.5, 10),
    island_gens_without_improvement = rep(0L, 10)
  )

  ring <- topology_ring(10)

  # Push uniform policy
  p_push <- policy_push_uniform()
  tx_push <- resolve_migration_transactions(p_push, ring, state)
  expect_equal(length(tx_push), 10)
  expect_equal(tx_push[[1]]$from, 1)
  expect_equal(tx_push[[1]]$to, 2)

  # Gibbs push policy with zero variance fallback
  p_gibbs <- policy_gibbs_push(weight_by = "fitness")
  tx_gibbs <- resolve_migration_transactions(p_gibbs, ring, state)
  expect_equal(length(tx_gibbs), 10)

  # Tiered admission policy
  tiered_topo <- topology_tiered(10, 3)
  p_tiered <- policy_tiered_admission()
  tx_tiered <- resolve_migration_transactions(p_tiered, tiered_topo, state)
  expect_true(length(tx_tiered) > 0)
})

test_that("migration_config creates comopsable objects and accepts string names", {
  cfg <- migration_config("hypercube", "dual_gibbs_pull", "gene_only")
  expect_s3_class(cfg, "evo_migration_config")
  expect_s3_class(cfg$topology, "evo_topology_hypercube")
  expect_s3_class(cfg$policy, "evo_policy_gibbs_pull")
  expect_equal(cfg$payload, "gene_only")
})

test_that("evolve_features accepts custom migration_config object", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  custom_mig <- migration_config(
    topology = topology_hypercube(10),
    policy = policy_gibbs_pull(stagnation_threshold = 2)
  )

  set.seed(42)
  rec <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 4,
    cv_folds = 2,
    islands = 10,
    migration_interval = 1,
    migration = custom_mig,
    verbose = FALSE
  )

  expect_s3_class(rec, "evo_recipe")
})

test_that("all topologies pre-build explicit adj_list, adj_matrix, and topology_custom works", {
  topos <- list(
    topology_ring(4),
    topology_grid(4),
    topology_hypercube(dimension = 2),
    topology_tiered(4, tiers = 2),
    topology_complete(4)
  )

  for (topo in topos) {
    expect_true(!is.null(topo$adj_list))
    expect_equal(length(topo$adj_list), 4L)
    mat <- as.matrix(topo)
    expect_equal(dim(mat), c(4L, 4L))
    expect_equal(unname(diag(mat)), rep(0L, 4L))
  }

  # Test topology_custom
  m <- matrix(c(0, 1, 1, 0), nrow = 2, ncol = 2)
  custom_topo <- topology_custom(m)
  expect_s3_class(custom_topo, "evo_topology_custom")
  expect_equal(custom_topo$islands, 2L)
  expect_equal(unname(get_neighbors(custom_topo, 1)), 2L)
  expect_equal(unname(get_neighbors(custom_topo, 2)), 1L)
})

test_that("topology_torus connects horizontal and vertical wrap-around neighbors correctly", {
  torus9 <- topology_torus(islands = 9, rows = 3, cols = 3)
  expect_s3_class(torus9, "evo_topology_torus")
  expect_s3_class(torus9, "evo_topology_grid")

  # 3x3 Grid:
  # 1 2 3
  # 4 5 6
  # 7 8 9

  # Island 1 connects to 2, 3 (horizontal wrap), 4, 7 (vertical wrap)
  n1 <- get_neighbors(torus9, 1)
  expect_setequal(n1, c(2, 3, 4, 7))

  # Island 3 connects to 2, 1 (horizontal wrap), 6, 9 (vertical wrap)
  n3 <- get_neighbors(torus9, 3)
  expect_setequal(n3, c(1, 2, 6, 9))

  # Island 7 connects to 8, 9 (horizontal wrap), 4, 1 (vertical wrap)
  n7 <- get_neighbors(torus9, 7)
  expect_setequal(n7, c(1, 4, 8, 9))

  # Island 9 connects to 8, 7 (horizontal wrap), 6, 3 (vertical wrap)
  n9 <- get_neighbors(torus9, 9)
  expect_setequal(n9, c(3, 6, 7, 8))
})
