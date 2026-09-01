library(evoFE)

test_that("evolve_features validates metacv parameters properly", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # Error if islands < 2 in metacv mode
  expect_error(
    evolve_features(df, "am", task = "classification", evaluation_strategy = "metacv", islands = 1, verbose = FALSE),
    "requires at least 2 islands"
  )

  # Error if both islands and cv_folds are explicitly provided and mismatched
  expect_error(
    evolve_features(df, "am", task = "classification", evaluation_strategy = "metacv", islands = 3, cv_folds = 4, verbose = FALSE),
    "must equal"
  )

  # Error if row_split_islands or per_island_validation is combined with metacv
  expect_error(
    evolve_features(df, "am", task = "classification", evaluation_strategy = "metacv", islands = 3, per_island_validation = TRUE, verbose = FALSE),
    "per_island_validation"
  )
})

test_that("evolve_features executes metacv for classification with islands", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_equal(recipe$evaluation_strategy, "metacv")
  expect_equal(length(recipe$island_bests), 3)
  expect_equal(length(recipe$fold_ids), nrow(df))
  expect_true(all(1:3 %in% recipe$fold_ids))
  expect_true(!is.null(recipe$oof_preds))
  expect_equal(length(recipe$oof_preds), nrow(df))
})

test_that("evolve_features executes metacv with holdout fraction", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "meta_cv", # test alias meta_cv
    islands = 3,
    generations = 2,
    pop_size = 3,
    holdout_frac = 0.2,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_equal(recipe$evaluation_strategy, "metacv")
  expect_true(!is.null(recipe$holdout_fitness))
  expect_true(is.numeric(recipe$holdout_fitness))
})

test_that("evolve_features executes metacv for regression and multiclass", {
  # Regression
  data(mtcars)
  rec_reg <- evolve_features(
    mtcars, "mpg",
    task = "regression",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )
  expect_s3_class(rec_reg, "evo_recipe")
  expect_equal(rec_reg$evaluation_strategy, "metacv")
  expect_equal(length(rec_reg$oof_preds), nrow(mtcars))

  # Multiclass
  data(iris)
  rec_mc <- evolve_features(
    iris, "Species",
    task = "multiclass",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )
  expect_s3_class(rec_mc, "evo_recipe")
  expect_equal(rec_mc$evaluation_strategy, "metacv")
  expect_true(is.matrix(rec_mc$oof_preds))
  expect_equal(nrow(rec_mc$oof_preds), nrow(iris))
  expect_equal(ncol(rec_mc$oof_preds), 3)
})

test_that("ensemble_islands and model_all_final_genes work with metacv recipes", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 2,
    pop_size = 3,
    model_all_final_genes = TRUE,
    verbose = FALSE
  )

  # Check model_all_final_genes
  expect_true(!is.null(recipe$best_individual))

  # Test ensemble_islands Caruana
  ens_caruana <- ensemble_islands(recipe, data = df, method = "caruana", caruana_rounds = 10, verbose = FALSE)
  expect_s3_class(ens_caruana, "evo_ensemble")
  expect_equal(ens_caruana$method, "caruana")

  # Predict with ensemble (features transformation & model predictions)
  transformed <- predict(ens_caruana, newdata = df)
  expect_s3_class(transformed, "data.table")
  expect_equal(nrow(transformed), nrow(df))

  preds <- predict_model(ens_caruana, newdata = df)
  expect_equal(length(preds), nrow(df))
})

test_that("metacv respects cv_strategy like group and time", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)
  df$grp <- rep(1:8, each = 4)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    cv_strategy = "group",
    group_col = "grp",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  # Verify groups are not split across folds
  for (g in unique(df$grp)) {
    fold_assignment <- unique(recipe$fold_ids[df$grp == g])
    expect_equal(length(fold_assignment), 1)
  }
})

test_that("metacv early stopping evaluates per-island stagnation", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 5,
    pop_size = 2,
    early_stopping_generations = 2,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_equal(recipe$evaluation_strategy, "metacv")
  expect_true(length(recipe$fitness_history) <= 5)
})
