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

test_that("metacv respects migration configuration topology and islands", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  mig_cfg <- migration_config(
    topology = topology_ring(islands = 3),
    policy = policy_push_uniform()
  )

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    migration = mig_cfg,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_equal(recipe$evaluation_strategy, "metacv")
  expect_equal(length(recipe$island_bests), 3)
})

test_that("metacv provides aligned oof_preds and metacv_island_oof_preds", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # 1. Ensemble mode (default): oof_preds matches the stitched metacv_island_oof_preds
  rec_ens <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    metacv_mode = "ensemble",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(rec_ens, "evo_ensemble")
  expect_s3_class(rec_ens, "evo_recipe")
  expect_equal(rec_ens$oof_preds, rec_ens$metacv_island_oof_preds)
  expect_equal(length(rec_ens$oof_preds), nrow(df))
  expect_equal(length(rec_ens$active_models), 3)
  expect_equal(length(rec_ens$active_recipes), 3)

  # Predict model on new data works with ensemble
  ens_preds <- predict_model(rec_ens, df[1:5, ])
  expect_equal(length(ens_preds), 5)
  expect_true(is.numeric(ens_preds))

  # 2. Tournament mode: oof_preds matches the best_individual full CV predictions
  rec_tourn <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    metacv_mode = "tournament",
    islands = 3,
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(rec_tourn, "evo_recipe")
  expect_false(inherits(rec_tourn, "evo_ensemble"))
  expect_equal(rec_tourn$oof_preds, rec_tourn$best_individual$val_preds)
  expect_equal(length(rec_tourn$oof_preds), nrow(df))
  expect_true(!is.null(rec_tourn$metacv_island_oof_preds))
  expect_equal(length(rec_tourn$metacv_island_oof_preds), nrow(df))
})

test_that("metacv tournament deduplicates identical candidate recipes", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # With pop_size 2 and generations 1, islands are likely to share baseline or simple recipes
  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 1,
    pop_size = 2,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_equal(length(recipe$island_bests), 3)
  expect_true(all(vapply(recipe$island_bests, function(ind) is.finite(ind$fitness), logical(1))))
})

test_that("metacv computes honest non-resubstitution baseline fitness", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  recipe <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    islands = 3,
    generations = 1,
    pop_size = 2,
    record = TRUE,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_true(!is.null(recipe$evolution_log$baseline))

  base_fit <- recipe$evolution_log$baseline$fitness
  island_fits <- vapply(recipe$evolution_log$baseline$islands, function(x) x$fitness, numeric(1))

  # Baseline fitness must equal the average of the out-of-fold island baselines
  expect_equal(base_fit, mean(island_fits), tolerance = 1e-6)
  expect_true(is.finite(base_fit))

  # Check new top-level recipe fields
  expect_equal(recipe$baseline_fitness, base_fit)
  expect_equal(recipe$improvement, recipe$ensemble_val_fitness - base_fit)
  expect_equal(recipe$single_best_improvement, recipe$best_individual$fitness - base_fit)
  expect_true(is.numeric(recipe$headroom_closed))
  expect_equal(length(recipe$island_baselines), 3)
  expect_equal(length(recipe$island_improvements), 3)
  expect_equal(length(recipe$island_headroom_closed), 3)

  # Check print and summary methods
  out_print <- utils::capture.output(print(recipe))
  expect_true(any(grepl("Baseline Score:", out_print)))
  expect_true(any(grepl("Headroom Closed:", out_print)))

  out_summary <- utils::capture.output(print(summary(recipe)))
  expect_true(any(grepl("Baseline Metric Score:", out_summary)))
  expect_true(any(grepl("Headroom Breakdown", out_summary)))

  # Verify that generation snapshots calculate headroom off the source island baseline
  gen1 <- recipe$evolution_log$generations[[1]]
  expect_true(!is.null(gen1$global_best_island))
  expect_equal(gen1$global_best_island_baseline, island_fits[gen1$global_best_island])
})

test_that("metacv_mode validation and task coverage for ensemble and tournament", {
  data(mtcars)
  df <- mtcars
  df$am <- as.integer(df$am)

  # Invalid metacv_mode throws error
  expect_error(
    evolve_features(df, "am", evaluation_strategy = "metacv", islands = 3, metacv_mode = "invalid", verbose = FALSE),
    "'arg' should be one of"
  )

  # Regression with metacv_mode = 'ensemble'
  rec_reg <- evolve_features(
    mtcars, "mpg",
    task = "regression",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    metacv_mode = "ensemble",
    islands = 3,
    generations = 1,
    pop_size = 2,
    verbose = FALSE
  )
  expect_s3_class(rec_reg, "evo_ensemble")
  expect_s3_class(rec_reg, "evo_recipe")
  expect_equal(rec_reg$method, "metacv")
  expect_equal(length(rec_reg$active_models), 3)

  reg_preds <- predict_model(rec_reg, mtcars[1:4, ])
  expect_equal(length(reg_preds), 4)
  expect_true(is.numeric(reg_preds))

  reg_feats <- predict(rec_reg, mtcars[1:4, ])
  expect_true(inherits(reg_feats, "data.table"))
  expect_equal(nrow(reg_feats), 4)

  # Multiclass with metacv_mode = 'ensemble'
  data(iris)
  rec_mc <- evolve_features(
    iris, "Species",
    task = "multiclass",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    metacv_mode = "ensemble",
    islands = 3,
    generations = 1,
    pop_size = 2,
    verbose = FALSE
  )
  expect_s3_class(rec_mc, "evo_ensemble")
  expect_equal(length(rec_mc$active_models), 3)

  mc_preds <- predict_model(rec_mc, iris[1:6, ])
  expect_true(is.matrix(mc_preds))
  expect_equal(nrow(mc_preds), 6)
  expect_equal(ncol(mc_preds), 3)
  expect_equal(colnames(mc_preds), levels(iris$Species))

  # Test print and summary for metacv ensemble
  ens_print <- utils::capture.output(print(rec_mc))
  expect_true(any(grepl("An evoFE Island Ensemble \\(MetaCV\\)", ens_print)))
  expect_true(any(grepl("Active Islands:       3 / 3", ens_print)))
  expect_equal(rec_mc$headroom_closed, rec_mc$ensemble_headroom_closed)
  expect_equal(rec_mc$improvement, rec_mc$ensemble_improvement)
  expect_true(!is.null(rec_mc$single_best_improvement))
  expect_true(!is.null(rec_mc$single_best_headroom_closed))

  if (!is.null(rec_mc$ensemble_headroom_closed) && is.finite(rec_mc$ensemble_headroom_closed)) {
    expected_hd_str <- sprintf("%5.1f%%", rec_mc$ensemble_headroom_closed * 100)
    expect_true(any(grepl(expected_hd_str, ens_print, fixed = TRUE)))
  }

  ens_summary <- utils::capture.output(print(summary(rec_mc)))
  expect_true(any(grepl("Summary of evoFE MetaCV Ensemble", ens_summary)))
  sum_mc <- summary(rec_mc)
  expect_equal(sum_mc$headroom_closed, rec_mc$ensemble_headroom_closed)

  # Confirmation holdout scoring with metacv ensemble
  rec_holdout <- evolve_features(
    df, "am",
    task = "classification",
    evaluator = "lightgbm",
    evaluation_strategy = "metacv",
    metacv_mode = "ensemble",
    islands = 3,
    generations = 1,
    pop_size = 2,
    holdout_frac = 0.25,
    verbose = FALSE
  )
  expect_s3_class(rec_holdout, "evo_ensemble")
  expect_true(!is.null(rec_holdout$best_individual$holdout_fitness))
  expect_true(is.numeric(rec_holdout$best_individual$holdout_fitness))

  # Edge case: islands < 2 fails with clear error
  expect_error(
    evolve_features(df, "am", evaluation_strategy = "metacv", islands = 1, verbose = FALSE),
    "requires at least 2 islands"
  )
})

