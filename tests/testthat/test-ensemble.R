test_that("caruana_select selects candidates greedily and returns normalized weights", {
  set.seed(42)
  n <- 100
  y_true <- rnorm(n, mean = 10, sd = 2)

  # Generate candidate predictions with different errors
  cand1 <- y_true + rnorm(n, mean = 0, sd = 1.0)
  cand2 <- y_true + rnorm(n, mean = 0, sd = 1.5)
  cand3 <- y_true + rnorm(n, mean = 0, sd = 2.0)

  val_preds_list <- list(island_1 = cand1, island_2 = cand2, island_3 = cand3)

  res <- caruana_select(
    y_true = y_true,
    val_preds_list = val_preds_list,
    task = "regression",
    metric = "default",
    rounds = 20,
    bag_samples = FALSE
  )

  expect_named(res$weights, c("island_1", "island_2", "island_3"))
  expect_equal(sum(res$weights), 1.0)
  expect_equal(nrow(res$history), 20)
  expect_true(res$weights["island_1"] > 0)
})

test_that("ensemble_islands validates inputs correctly", {
  data(mtcars)

  expect_error(
    ensemble_islands("not_a_recipe", data = mtcars),
    "must be an object of class 'evo_recipe'"
  )

  # Fake recipe without island_bests
  fake_recipe <- structure(list(best_individual = list(all_numeric_cols = "mpg")), class = "evo_recipe")
  expect_error(
    ensemble_islands(fake_recipe, data = mtcars, target_col = "mpg"),
    "No valid validation prediction vectors found"
  )

  # Non-existent target column
  expect_error(
    ensemble_islands(fake_recipe, data = mtcars, target_col = "nonexistent_col"),
    "Target column 'nonexistent_col' not found in 'data'"
  )
})

test_that("ensemble_islands works for binary classification and predict_model/print/summary S3 methods", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = "xgboost",
    generations = 2,
    pop_size = 2,
    islands = 3,
    cv_folds = 2,
    verbose = FALSE
  )

  expect_true(!is.null(recipe$island_bests))
  expect_equal(length(recipe$island_bests), 3)

  ens <- ensemble_islands(recipe, data = df, caruana_rounds = 10, verbose = FALSE)

  expect_s3_class(ens, "evo_ensemble")
  expect_true(sum(ens$weights) == 1.0)
  expect_true(length(ens$active_models) > 0)

  # Positional method = "caruana"
  ens_pos_caruana <- ensemble_islands(recipe, df, "caruana", caruana_rounds = 5, verbose = FALSE)
  expect_s3_class(ens_pos_caruana, "evo_ensemble")
  expect_equal(ens_pos_caruana$method, "caruana")

  # Positional method = "stack"
  ens_pos_stack <- ensemble_islands(recipe, df, "stack", seed = 1, verbose = FALSE)
  expect_s3_class(ens_pos_stack, "evo_ensemble")
  expect_equal(ens_pos_stack$method, "stack")

  # Predict model
  preds <- predict_model(ens, df[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)

  # Predict features
  feat_dt <- predict(ens, df[1:5, ])
  expect_s3_class(feat_dt, "data.table")
  expect_equal(nrow(feat_dt), 5)

  # S3 output checks
  expect_output(print(ens), "An evoFE Island Ensemble \\(Caruana\\)")
  sum_ens <- summary(ens)
  expect_s3_class(sum_ens, "summary_evo_ensemble")
  expect_output(print(sum_ens), "Summary of evoFE Caruana Ensemble")})

test_that("ensemble_islands works for regression tasks", {
  data(mtcars)

  recipe <- evolve_features(
    data = mtcars,
    target_col = "mpg",
    task = "regression",
    evaluator = "xgboost",
    generations = 2,
    pop_size = 2,
    islands = 2,
    cv_folds = 2,
    verbose = FALSE
  )

  ens <- ensemble_islands(recipe, data = mtcars, caruana_rounds = 10, verbose = FALSE)

  expect_s3_class(ens, "evo_ensemble")
  preds <- predict_model(ens, mtcars[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)
})

test_that("evolve_features supports per-island evaluators vector and ensemble_islands ensembles them", {
  testthat::skip_if_not_installed("glmnet")
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    data = df,
    target_col = "am",
    task = "classification",
    evaluator = c("xgboost", "lightgbm", "lm"),
    generations = 2,
    pop_size = 2,
    islands = 3,
    cv_folds = 2,
    verbose = FALSE
  )

  expect_equal(length(recipe$island_bests), 3)
  expect_equal(recipe$island_bests[[1]]$evaluator, "xgboost")
  expect_equal(recipe$island_bests[[2]]$evaluator, "lightgbm")
  expect_equal(recipe$island_bests[[3]]$evaluator, "lm")

  ens <- ensemble_islands(recipe, data = df, caruana_rounds = 10, verbose = FALSE)

  expect_s3_class(ens, "evo_ensemble")
  preds <- predict_model(ens, df[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)
})

test_that("ensemble_islands supports cross-recipe ensembling from a list of evo_recipe objects", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe_xgb <- evolve_features(
    data = df, target_col = "am", task = "classification", evaluator = "xgboost",
    generations = 2, pop_size = 2, islands = 2, cv_folds = 2, verbose = FALSE
  )

  recipe_lgb <- evolve_features(
    data = df, target_col = "am", task = "classification", evaluator = "lightgbm",
    generations = 2, pop_size = 2, islands = 2, cv_folds = 2, verbose = FALSE
  )

  ens <- ensemble_islands(
    recipe = list(xgb = recipe_xgb, lgb = recipe_lgb),
    data = df,
    caruana_rounds = 15,
    verbose = FALSE
  )

  expect_s3_class(ens, "evo_ensemble")
  expect_true(length(ens$active_models) > 0)

  preds <- predict_model(ens, df[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)
})

test_that("ensemble_islands method = 'stack' runs on regression with honest CV fitness", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("glmnet")
  data(mtcars)
  set.seed(42)

  rec <- evolve_features(
    data = mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 1, pop_size = 4, cv_folds = 3, islands = 3, verbose = FALSE
  )

  ens <- ensemble_islands(rec, data = mtcars, method = "stack", seed = 1, verbose = FALSE)

  expect_s3_class(ens, "evo_ensemble")
  expect_identical(ens$method, "stack")
  expect_null(ens$caruana_history)
  expect_true(is.finite(ens$stack_cv_fitness))
  expect_true(is.finite(ens$ensemble_val_fitness))
  expect_true(all(ens$weights >= 0))
  expect_equal(sum(ens$weights), 1, tolerance = 1e-8)
  expect_true(length(ens$active_models) > 0)

  preds <- predict_model(ens, mtcars[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)
})

test_that("stacked weights are reproducible under a fixed seed", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("glmnet")
  data(mtcars)
  set.seed(42)

  rec <- evolve_features(
    data = mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 1, pop_size = 4, cv_folds = 3, islands = 3, verbose = FALSE
  )

  a <- ensemble_islands(rec, data = mtcars, method = "stack", seed = 11, verbose = FALSE)
  b <- ensemble_islands(rec, data = mtcars, method = "stack", seed = 11, verbose = FALSE)

  expect_equal(a$weights, b$weights)
  expect_equal(a$stack_cv_fitness, b$stack_cv_fitness)
})

test_that("ensemble_islands method = 'stack' supports binary classification and multiclass", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("glmnet")

  df_bin <- mtcars
  df_bin$am <- as.integer(df_bin$am)
  set.seed(42)
  rec_bin <- evolve_features(
    data = df_bin, target_col = "am", task = "classification", evaluator = "lm",
    generations = 1, pop_size = 4, cv_folds = 3, islands = 2, verbose = FALSE
  )
  ens_bin <- ensemble_islands(rec_bin, data = df_bin, method = "stack", seed = 2, verbose = FALSE)
  expect_s3_class(ens_bin, "evo_ensemble")
  expect_true(is.finite(ens_bin$stack_cv_fitness))

  set.seed(7)
  rec_multi <- evolve_features(
    data = iris, target_col = "Species", task = "multiclass", evaluator = "xgboost",
    generations = 1, pop_size = 3, cv_folds = 3, islands = 3, verbose = FALSE
  )
  ens_multi <- ensemble_islands(rec_multi, data = iris, method = "stack",
                                stack_folds = 3, seed = 3, verbose = FALSE)
  expect_s3_class(ens_multi, "evo_ensemble")
  expect_true(is.finite(ens_multi$stack_cv_fitness))
  expect_true(all(ens_multi$weights >= 0))
  expect_equal(sum(ens_multi$weights), 1, tolerance = 1e-8)

  preds <- predict_model(ens_multi, iris[1:6, ])
  expect_true(is.numeric(preds))
})

test_that("stack method falls back to internal folds in split mode", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("glmnet")
  data(mtcars)
  set.seed(42)

  rec <- evolve_features(
    data = mtcars, target_col = "mpg", task = "regression", evaluator = "lm",
    generations = 1, pop_size = 4, islands = 2,
    evaluation_strategy = "split", verbose = FALSE
  )

  msgs <- character(0)
  ens <- withCallingHandlers(
    ensemble_islands(rec, data = mtcars, method = "stack", seed = 4, verbose = TRUE),
    message = function(m) {
      msgs <<- c(msgs, m$message)
      invokeRestart("muffleMessage")
    }
  )

  expect_s3_class(ens, "evo_ensemble")
  expect_true(any(grepl("internal folds", msgs)))
  expect_true(is.finite(ens$stack_cv_fitness))
})

test_that("ensemble_islands harmonizes mixed recipes with different evaluation strategies (split + cv)", {
  testthat::skip_on_cran()
  testthat::skip_if_not_installed("glmnet")
  data(iris)
  set.seed(42)

  rec1 <- evolve_features(
    data = iris, target_col = "Species", task = "multiclass", evaluator = "lightgbm",
    generations = 1, pop_size = 2, islands = 2, evaluation_strategy = "split", verbose = FALSE
  )
  rec2 <- evolve_features(
    data = iris, target_col = "Species", task = "multiclass", evaluator = "lightgbm",
    generations = 1, pop_size = 2, islands = 2, evaluation_strategy = "cv", cv_folds = 2, verbose = FALSE
  )

  res <- list(recipe1 = rec1, recipe2 = rec2)

  ens_stack <- ensemble_islands(res, data = iris, method = "stack", threads = 2, verbose = FALSE)
  expect_s3_class(ens_stack, "evo_ensemble")
  expect_true(is.finite(ens_stack$stack_cv_fitness))
  expect_equal(sum(ens_stack$weights), 1, tolerance = 1e-8)
  preds_s <- predict_model(ens_stack, iris[1:5, ])
  expect_true(is.matrix(preds_s) || is.numeric(preds_s))

  ens_car <- ensemble_islands(res, data = iris, method = "caruana", caruana_rounds = 10, threads = 2, verbose = FALSE)
  expect_s3_class(ens_car, "evo_ensemble")
  expect_true(is.finite(ens_car$ensemble_val_fitness))
  expect_equal(sum(ens_car$weights), 1, tolerance = 1e-8)
  preds_c <- predict_model(ens_car, iris[1:5, ])
  expect_true(is.matrix(preds_c) || is.numeric(preds_c))
})

