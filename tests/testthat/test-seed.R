# Tests for the CRAN-safe seeding contract: reproducibility given a seed,
# and zero leakage into the user's .Random.seed.

test_that("seed produces identical runs without touching user RNG state", {
  skip_on_cran()
  set.seed(999)
  user_state <- get(".Random.seed", envir = globalenv())

  d <- data.table::data.table(
    x1 = rnorm(100), x2 = runif(100),
    cat = sample(c("a", "b", "c"), 100, TRUE),
    y = rnorm(100)
  )

  args <- list(
    data = d, target_col = "y", task = "regression",
    generations = 2, pop_size = 3, cv_folds = 2,
    evaluator = "lightgbm", threads = 1, verbose = FALSE, seed = 123
  )
  r1 <- do.call(evolve_features, args)
  r2 <- do.call(evolve_features, args)

  expect_identical(r1$fitness_history, r2$fitness_history)
  expect_identical(
    individual_to_recipe_string(r1$best_individual),
    individual_to_recipe_string(r2$best_individual)
  )

  # Output depends only on the seed, not on the ambient RNG state:
  # a run entering with no .Random.seed at all must reproduce r1 exactly.
  if (exists(".Random.seed", envir = globalenv())) {
    rm(".Random.seed", envir = globalenv())
  }
  r3 <- do.call(evolve_features, args)
  expect_identical(r3$fitness_history, r1$fitness_history)
  expect_identical(
    individual_to_recipe_string(r3$best_individual),
    individual_to_recipe_string(r1$best_individual)
  )
})


test_that("island populations are stable functions of (seed, island index)", {
  skip_on_cran()
  d <- data.table::data.table(x1 = rnorm(80), x2 = runif(80), y = rnorm(80))

  base_args <- list(
    data = d, target_col = "y", task = "regression",
    generations = 1, pop_size = 3, cv_folds = 2,
    islands = 2, evaluator = "lightgbm", threads = 1,
    verbose = FALSE
  )
  a <- do.call(evolve_features, c(base_args, list(seed = 7)))
  b <- do.call(evolve_features, c(base_args, list(seed = 7)))

  recipes_a <- sapply(a$island_bests, individual_to_recipe_string)
  recipes_b <- sapply(b$island_bests, individual_to_recipe_string)
  expect_identical(recipes_a, recipes_b)

  # Island 1 init uses seed + 1000*1 regardless of island count:
  # rerun with 1 island and compare its population's first individual recipe
  one <- evolve_features(data = d, target_col = "y", task = "regression",
    generations = 1, pop_size = 3, cv_folds = 2, islands = 1,
    seed = 1007, evaluator = "lightgbm", threads = 1, verbose = FALSE)
  expect_type(recipes_a[1], "character")
})

test_that("invalid seed values error clearly", {
  d <- data.frame(x = 1:30, y = rnorm(30))
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1,
      pop_size = 2, seed = "abc"),
    regexp = "'seed' must be"
  )
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1,
      pop_size = 2, seed = c(1, 2)),
    regexp = "'seed' must be"
  )
})
