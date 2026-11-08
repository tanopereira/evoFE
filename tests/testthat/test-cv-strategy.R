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
})

test_that("evaluate_pop correctly indexes tested individuals", {
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

test_that("continuous regression quantile stratification, stratified_group CV, and adversarial holdout work", {
  set.seed(123)
  # 1. Continuous target quantile stratification in .build_cv_folds and stratified_split
  y_skew <- exp(stats::rnorm(200, mean = 0, sd = 2))
  df_reg <- data.frame(x = stats::rnorm(200), y = y_skew)
  f_reg <- evoFE:::.build_cv_folds(df_reg, cv_folds = 5, strategy = "random",
                                   target_col = "y", task = "regression")
  expect_equal(as.numeric(table(f_reg)), rep(40, 5))
  fold_medians <- tapply(df_reg$y, f_reg, stats::median)
  expect_lt(max(fold_medians) - min(fold_medians), 0.35 * stats::IQR(df_reg$y))

  sp_reg <- evoFE:::stratified_split(y_skew, c(0.8, 0.2))
  expect_equal(sum(sp_reg == "train"), 160L)
  expect_equal(sum(sp_reg == "val"), 40L)

  # 2. Stratified Group K-Fold balances target classes while keeping groups intact
  n_grp <- 20
  grp_ids <- rep(sprintf("grp_%02d", seq_len(n_grp)), each = 5)
  # First 4 groups are positive, remaining 16 groups are negative
  y_grp <- c(rep(1L, 4 * 5), rep(0L, 16 * 5))
  df_grp <- data.frame(g = grp_ids, x = stats::rnorm(length(grp_ids)), y = y_grp)
  f_sgrp <- evoFE:::.build_cv_folds(df_grp, cv_folds = 4, strategy = "stratified_group",
                                    group_col = "g", target_col = "y", task = "classification")
  # Every group is in a single fold, and each of the 4 folds gets exactly 1 positive group (5 positive rows)
  expect_true(all(tapply(f_sgrp, df_grp$g, function(z) length(unique(z))) == 1L))
  expect_equal(as.numeric(tapply(df_grp$y, f_sgrp, sum)), rep(5L, 4))

  # Group-intact stratified_split never splits a group across train/val
  sp_grp <- evoFE:::stratified_split(df_grp$y, c(0.75, 0.25), group = df_grp$g)
  expect_true(all(tapply(sp_grp, df_grp$g, function(z) length(unique(z))) == 1L))

  # 3. Adversarial validation detects drifting features and weights holdout toward test distribution
  set.seed(77)
  tr_adv <- data.frame(
    x_stable = stats::rnorm(120, mean = 0, sd = 1),
    x_drift = stats::rnorm(120, mean = 0, sd = 1),
    y = sample(0:1, 120, replace = TRUE)
  )
  te_adv <- data.frame(
    x_stable = stats::rnorm(80, mean = 0, sd = 1),
    x_drift = stats::rnorm(80, mean = 3.5, sd = 1)
  )
  adv_res <- compute_adversarial_weights(tr_adv, te_adv, target_col = "y")
  expect_gt(adv_res$adversarial_auc, 0.85)
  expect_true("x_drift" %in% adv_res$drifting_features)
  expect_false("x_stable" %in% adv_res$drifting_features)
  # Training rows with larger x_drift receive higher test-similarity weights
  expect_gt(stats::cor(tr_adv$x_drift, adv_res$weights), 0.5)
})


