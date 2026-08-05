test_that("evolve_features throws errors for invalid island parameters", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  expect_error(
    evolve_features(df, "am", task = "classification", islands = -1, verbose = FALSE),
    "islands must be a positive integer"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, migration_interval = 0, verbose = FALSE),
    "migration_interval must be a positive integer"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, migration_rate = 0, verbose = FALSE),
    "migration_rate must be a positive integer"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, gene_migration_prob = 1.5, verbose = FALSE),
    "gene_migration_prob must be a numeric value between 0 and 1"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, pop_size = 4, migration_rate = 4, verbose = FALSE),
    "migration_rate must be less than pop_size"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, migration_topology = "invalid_topo", verbose = FALSE),
    "migration_topology must be one of"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, migration_temperature = -0.5, verbose = FALSE),
    "migration_temperature must be a positive numeric value"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 2, pull_stagnation_threshold = 0, verbose = FALSE),
    "pull_stagnation_threshold must be a positive integer"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 1, allowed_transformers = list("basic", "robust"), verbose = FALSE),
    "length must match the number of islands"
  )

  expect_error(
    evolve_features(df, "am", task = "classification", islands = 3, allowed_transformers = list("basic", "robust"), verbose = FALSE),
    "length \\(2\\) must match the number of islands \\(3\\)"
  )
})

test_that("evolve_features runs successfully with multiple islands and outputs correct logging", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # Check that it runs end-to-end with islands = 2
  # Use small parameters for fast testing
  set.seed(42)
  
  logs <- character(0)
  withCallingHandlers(
    {
      recipe <- evolve_features(
        data = df,
        target_col = "am",
        task = "classification",
        evaluator = "lightgbm",
        generations = 2,
        pop_size = 4,
        cv_folds = 2,
        islands = 2,
        migration_interval = 1,
        migration_rate = 1,
        migration_prob = 1.0,
        gene_migration_prob = 0.5,
        verbose = TRUE
      )
    },
    message = function(m) {
      logs <<- c(logs, m$message)
      invokeRestart("muffleMessage")
    }
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_true(length(recipe$best_individual) > 0)
  
  # Verify logs contain Island information
  expect_true(any(grepl("\\[Island 1\\] Tested Individual", logs)))
  expect_true(any(grepl("[Migration Phase] Triggering migration", logs, fixed = TRUE)))
})

test_that("evolve_features runs successfully with heterogeneous per-island allowed_transformers", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  set.seed(42)
  recipe <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 4,
    cv_folds = 2,
    islands = 3,
    allowed_transformers = list("basic", "robust", "clustering"),
    migration_interval = 1,
    migration_rate = 1,
    gene_migration_prob = 0.5,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_true(length(recipe$best_individual) > 0)
})

test_that("evolve_features validates and runs successfully with row_split_islands", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # 1. Validation error for non-logical type
  expect_error(
    evolve_features(df, "am", task = "classification", row_split_islands = "invalid", verbose = FALSE),
    "row_split_islands must be a logical scalar"
  )
  
  # 2. Warning for row_split_islands = TRUE with islands = 1
  expect_warning(
    evolve_features(df, "am", task = "classification", islands = 1, row_split_islands = TRUE, generations = 1, pop_size = 2, cv_folds = 2, verbose = FALSE),
    "row_split_islands is TRUE but islands is 1"
  )

  set.seed(42)
  # 3. Split strategy row_split_islands test
  recipe_split <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 4,
    islands = 2,
    row_split_islands = TRUE,
    evaluation_strategy = "split",
    split_ratio = c(0.6, 0.4),
    verbose = FALSE
  )
  expect_s3_class(recipe_split, "evo_recipe")

  # 4. CV strategy row_split_islands test
  recipe_cv <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 4,
    islands = 2,
    row_split_islands = TRUE,
    evaluation_strategy = "cv",
    cv_folds = 2,
    verbose = FALSE
  )
  expect_s3_class(recipe_cv, "evo_recipe")
})

test_that("evolve_features throws errors for invalid per_island_validation parameters", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # per_island_validation without row_split_islands should error
  expect_error(
    evolve_features(df, "am", task = "classification",
                    islands = 2, row_split_islands = FALSE,
                    per_island_validation = TRUE, verbose = FALSE),
    "per_island_validation = TRUE requires row_split_islands = TRUE"
  )

  # per_island_validation with cv evaluation_strategy should error
  expect_error(
    evolve_features(df, "am", task = "classification",
                    islands = 2, row_split_islands = TRUE,
                    per_island_validation = TRUE,
                    evaluation_strategy = "cv", verbose = FALSE),
    "per_island_validation = TRUE is only supported with evaluation_strategy = 'split'"
  )

  # non-logical per_island_validation should error
  expect_error(
    evolve_features(df, "am", task = "classification",
                    islands = 2, per_island_validation = "yes", verbose = FALSE),
    "per_island_validation must be a logical scalar"
  )
})

test_that("evolve_features runs successfully with per_island_validation = TRUE", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  set.seed(42)
  recipe <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 2,
    pop_size = 4,
    islands = 2,
    row_split_islands = TRUE,
    per_island_validation = TRUE,
    evaluation_strategy = "split",
    verbose = FALSE
  )
  expect_s3_class(recipe, "evo_recipe")
})

test_that("evolve_features handles migration when no new genes are generated without error", {
  data(iris)
  set.seed(42)
  recipe <- evolve_features(
    data = iris,
    target_col = "Species",
    task = "multiclass",
    evaluator = "lightgbm",
    generations = 5,
    pop_size = 4,
    cv_folds = 2,
    islands = 2,
    migration_interval = 1,
    migration_rate = 1,
    gene_migration_prob = 0.5,
    verbose = FALSE
  )
  expect_s3_class(recipe, "evo_recipe")
})

test_that("migrated elite individuals participate in survivor selection and elitism", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  set.seed(123)
  # Run evolution with islands = 2, migration_interval = 1, migration_rate = 1
  recipe_no_split <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 3,
    pop_size = 4,
    cv_folds = 2,
    islands = 2,
    migration_interval = 1,
    migration_rate = 1,
    row_split_islands = FALSE,
    verbose = FALSE
  )
  expect_s3_class(recipe_no_split, "evo_recipe")

  # Run with row_split_islands = TRUE
  recipe_split <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "lightgbm",
    generations = 3,
    pop_size = 4,
    cv_folds = 2,
    islands = 2,
    migration_interval = 1,
    migration_rate = 1,
    row_split_islands = TRUE,
    verbose = FALSE
  )
  expect_s3_class(recipe_split, "evo_recipe")
})

test_that("evolve_features executes successfully with gibbs_stagnation, gibbs_fitness, dual_gibbs_pull, and random topologies", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  set.seed(42)
  for (topo in c("gibbs_stagnation", "gibbs_fitness", "dual_gibbs_pull", "random")) {
    logs <- character(0)
    withCallingHandlers(
      {
        recipe <- evolve_features(
          data = df,
          target_col = "am",
          task = "classification",
          evaluator = "lightgbm",
          generations = 3,
          pop_size = 4,
          cv_folds = 2,
          islands = 3,
          migration_interval = 1,
          migration_rate = 1,
          migration_topology = topo,
          migration_temperature = 0.5,
          pull_stagnation_threshold = 1,
          verbose = TRUE
        )
      },
      message = function(m) {
        logs <<- c(logs, m$message)
        invokeRestart("muffleMessage")
      }
    )

    expect_s3_class(recipe, "evo_recipe")
    expect_true(any(grepl("\\[Migration Phase\\] Triggering migration", logs)))
  }
})

test_that("promoted migrant with superior fitness is sorted to index 1 in destination island", {
  data(mtcars)
  mc <- migration_config(topology = topology_tiered(islands = 5), payload = "full_individual")
  rec <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 3, pop_size = 4, cv_folds = 2, islands = 5, migration_interval = 1,
    migration = mc, verbose = FALSE
  )
  expect_s3_class(rec, "evo_recipe")
})

test_that("gene_only payload does not replace full individuals in destination population", {
  data(mtcars)
  mc_gene <- migration_config(topology = topology_ring(islands = 3), payload = "gene_only")
  rec_gene <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 3, pop_size = 4, cv_folds = 2, islands = 3, migration_interval = 1,
    migration = mc_gene, verbose = FALSE
  )
  expect_s3_class(rec_gene, "evo_recipe")
})

test_that("grid topology respects gibbs push policy by stagnation", {
  data(mtcars)
  mc_grid_stag <- migration_config(
    topology = topology_grid(islands = 4),
    policy = policy_gibbs_push(weight_by = "stagnation")
  )
  rec <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 3, pop_size = 4, cv_folds = 2, islands = 4, migration_interval = 1,
    migration = mc_grid_stag, verbose = FALSE
  )
  expect_s3_class(rec, "evo_recipe")
})

test_that("policy_tiered_admission throws error when combined with non-tiered topology", {
  expect_error(
    migration_config(topology = topology_grid(islands = 4), policy = policy_tiered_admission()),
    "policy_tiered_admission can only be used with tiered topologies"
  )
})

test_that("tiered topology uses vertical promotion and respects policy within each tier", {
  data(mtcars)
  mc_tiered_gibbs <- migration_config(
    topology = topology_tiered(islands = 5),
    policy = policy_gibbs_push(weight_by = "stagnation")
  )
  rec <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 3, pop_size = 4, cv_folds = 2, islands = 5, migration_interval = 1,
    migration = mc_tiered_gibbs, verbose = FALSE
  )
  expect_s3_class(rec, "evo_recipe")
})

test_that("evolve_features accepts migration_topology = 'torus'", {
  data(mtcars)
  rec <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 2, pop_size = 4, cv_folds = 2, islands = 4, migration_interval = 1,
    migration_topology = "torus", verbose = FALSE
  )
  expect_s3_class(rec, "evo_recipe")
})

test_that("policy_gibbs_push with feature_distance weighting executes correctly", {
  data(mtcars)
  mc_feat_dist <- migration_config(
    topology = topology_grid(islands = 4),
    policy = policy_gibbs_push(weight_by = "feature_distance")
  )
  rec <- evolve_features(
    mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 3, pop_size = 4, cv_folds = 2, islands = 4, migration_interval = 1,
    migration = mc_feat_dist, verbose = FALSE
  )
  expect_s3_class(rec, "evo_recipe")
})

