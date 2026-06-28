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
  expect_true(any(grepl("\\[Island 2\\] Tested Individual", logs)))
  expect_true(any(grepl("\\[Migration Phase\\] Triggering migration", logs)))
  expect_true(any(grepl("Migrating top 1 recipe\\(s\\) from Island 1 to Island 2", logs)))
})
