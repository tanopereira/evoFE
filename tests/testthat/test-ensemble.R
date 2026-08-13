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
  fake_recipe <- structure(list(best_individual = list()), class = "evo_recipe")
  expect_error(
    ensemble_islands(fake_recipe, data = mtcars),
    "does not contain island prediction history"
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

  # Predict model
  preds <- predict_model(ens, df[1:5, ])
  expect_true(is.numeric(preds))
  expect_equal(length(preds), 5)

  # Predict features
  feat_dt <- predict(ens, df[1:5, ])
  expect_s3_class(feat_dt, "data.table")
  expect_equal(nrow(feat_dt), 5)

  # S3 output checks
  expect_output(print(ens), "An evoFE Caruana Island Ensemble")
  sum_ens <- summary(ens)
  expect_s3_class(sum_ens, "summary_evo_ensemble")
  expect_output(print(sum_ens), "Summary of evoFE Caruana Ensemble")
})

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
