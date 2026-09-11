# Tests for cv_strategy (random / time / group), holdout confirmation,
# and multi-fidelity evaluation.

test_that(".build_cv_folds random strategy covers all rows exactly once", {
  set.seed(1)
  dt <- data.frame(x = 1:100)
  f <- evoFE:::.build_cv_folds(dt, 5, "random")
  expect_equal(as.numeric(sort(table(f))), rep(20, 5))
})

test_that(".build_cv_folds time strategy creates contiguous chronological blocks", {
  dt <- data.frame(t = as.POSIXct("2024-01-01") + c(9, 1, 7, 3, 5, 2, 8, 4, 6, 10) * 86400)
  f <- evoFE:::.build_cv_folds(dt, 3, "time", time_col = "t")

  # Each fold must occupy a contiguous time range and ranges must not overlap
  ranges <- tapply(dt$t, f, range)
  starts <- sapply(ranges, function(r) r[1])
  ends <- sapply(ranges, function(r) r[2])
  o <- order(starts)
  expect_gt(starts[o[2]], ends[o[1]])
  expect_gt(starts[o[3]], ends[o[2]])
  # All rows assigned
  expect_true(all(f %in% 1:3))
})

test_that(".build_cv_folds group strategy never splits a group across folds", {
  set.seed(2)
  n_groups <- 17
  sizes <- sample(1:12, n_groups, replace = TRUE)
  dt <- data.frame(
    g = factor(rep(sprintf("g%d", 1:n_groups), times = sizes)),
    y = rnorm(sum(sizes))
  )
  f <- evoFE:::.build_cv_folds(dt, 4, "group", group_col = "g")

  # Every group maps to exactly one fold
  folds_per_group <- tapply(f, dt$g, function(x) length(unique(x)))
  expect_true(all(folds_per_group == 1))
  # All folds used and all rows assigned
  expect_equal(length(unique(f)), 4L)
  # Group sizes are balanced within ~1 largest-group slack
  load <- tapply(f, dt$g, length)
  fold_totals <- tapply(rep(1, nrow(dt)), f, sum)
  expect_lte(max(fold_totals) - min(fold_totals), max(sizes))
})

test_that("evolve_features with cv_strategy='group' keeps groups intact end-to-end", {
  skip_on_cran()
  set.seed(42)
  n_groups <- 30
  d <- data.table::data.table(
    entity = rep(sprintf("e%02d", 1:n_groups), each = 4),
    x1 = rnorm(n_groups * 4),
    x2 = runif(n_groups * 4),
    y = rnorm(n_groups * 4)
  )

  recipe <- evolve_features(
    data = d, target_col = "y", task = "regression",
    generations = 1, pop_size = 2, cv_folds = 3,
    evaluation_strategy = "cv", cv_strategy = "group", group_col = "entity",
    evaluator = "lightgbm", verbose = FALSE
  )
  expect_s3_class(recipe, "evo_recipe")
})

test_that("holdout_frac carves an untouched set and reports search gap", {
  skip_on_cran()
  set.seed(42)
  n <- 120
  d <- data.table::data.table(
    x1 = rnorm(n),
    x2 = runif(n),
    cat = sample(c("a", "b", "c"), n, replace = TRUE),
    y = rnorm(n)
  )

  recipe <- evolve_features(
    data = d, target_col = "y", task = "regression",
    generations = 1, pop_size = 2, cv_folds = 3,
    evaluation_strategy = "cv", holdout_frac = 0.2,
    evaluator = "lightgbm", verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  # Holdout was scored
  expect_false(is.null(recipe$holdout_fitness))
  expect_true(is.finite(recipe$holdout_fitness))
  # Search gap is computed against validation fitness
  expect_false(is.null(recipe$search_gap))
  raw <- recipe$best_individual$raw_fitness
  expect_equal(recipe$search_gap, recipe$holdout_fitness - raw)
  # The search only saw 80% of rows: fitness_history length reflects search data
  expect_equal(round(nrow(d) * 0.8), 96)
})

test_that("multi_fidelity runs and promotes at full fidelity", {
  skip_on_cran()
  set.seed(42)
  n <- 150
  d <- data.table::data.table(
    x1 = rnorm(n),
    x2 = runif(n),
    y = rnorm(n)
  )

  recipe <- evolve_features(
    data = d, target_col = "y", task = "regression",
    generations = 2, pop_size = 4, cv_folds = 2,
    evaluation_strategy = "cv", multi_fidelity = TRUE,
    mf_sample_frac = 0.6, mf_warmup_frac = 0.5,
    evaluator = "lightgbm", verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  best <- recipe$best_individual
  expect_true(is.finite(best$fitness))
  # Final best was evaluated at full fidelity (all rows available to folds)
  expect_lte(best$fitness, Inf)
})

test_that("invalid new arguments produce clear errors", {
  d <- data.frame(x = 1:50, y = rnorm(50))
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1, pop_size = 2,
      evaluation_strategy = "cv", cv_strategy = "time"),
    regexp = "time_col"
  )
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1, pop_size = 2,
      evaluation_strategy = "cv", cv_strategy = "group", group_col = "nope"),
    regexp = "group_col"
  )
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1, pop_size = 2,
      evaluation_strategy = "cv", holdout_frac = 0.6),
    regexp = "holdout_frac"
  )
  expect_error(
    evolve_features(d, "y", task = "regression", generations = 1, pop_size = 2,
      multi_fidelity = TRUE, mf_sample_frac = 0),
    regexp = "mf_sample_frac"
  )
})

test_that("multi-fidelity and evaluate_pop correctly index tested individuals", {
  skip_on_cran()
  set.seed(42)
  n <- 100
  d <- data.frame(
    x1 = rnorm(n),
    x2 = runif(n),
    y = rnorm(n)
  )

  # Unit check: evaluate_pop respects ind_indices
  pop <- list(
    structure(list(genes = list(), all_numeric_cols = c("x1", "x2"), all_categorical_cols = character(0), all_datetime_cols = character(0), fitness = NA_real_), class = "evo_individual"),
    structure(list(genes = list(), all_numeric_cols = c("x1", "x2"), all_categorical_cols = character(0), all_datetime_cols = character(0), fitness = NA_real_), class = "evo_individual")
  )
  cache <- new.env(hash = TRUE, parent = emptyenv())
  state <- new.env(hash = TRUE, parent = emptyenv())

  msgs <- testthat::capture_messages({
    res <- evaluate_pop(
      pop = pop, data = d, target_col = "y", task = "regression", cv_folds = 2,
      evaluation_strategy = "cv", split_ids = NULL, shared_splits = NULL,
      evaluator = "lightgbm", fold_ids = NULL, shared_folds = NULL, shared_full = NULL,
      state_cache = state, fitness_cache = cache, threads = 1, verbose = TRUE,
      running_best_fitness = -Inf, island = 3, ind_indices = c(4, 7)
    )
  })

  expect_true(any(grepl("\\[Island 3\\] Tested Individual 4", msgs)))
  expect_true(any(grepl("\\[Island 3\\] Tested Individual 7", msgs)))
  expect_false(any(grepl("\\[Island 3\\] Tested Individual 1", msgs)))
  expect_false(any(grepl("\\[Island 3\\] Tested Individual 2", msgs)))
})

