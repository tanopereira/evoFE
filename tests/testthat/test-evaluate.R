test_that("compute_calibrated_rmse works correctly", {
  # 1. Standard case
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(2, 4, 6, 8, 10) # perfect linear prediction (y = 0.5 * y_pred)
  expect_equal(compute_calibrated_rmse(y_true, y_pred), 0.0, tolerance = 1e-7)

  # 2. Complete noise case (uncorrelated)
  set.seed(42)
  y_true <- rnorm(100, mean = 10, sd = 2)
  y_pred <- rnorm(100, mean = 0, sd = 1)
  # Calibrated RMSE should be very close to SD(y_true) * sqrt(1 - R^2)
  r <- cor(y_pred, y_true)
  expected <- sd(y_true) * sqrt(1 - r^2)
  expect_equal(compute_calibrated_rmse(y_true, y_pred), expected, tolerance = 1e-9)

  # 3. Constant prediction edge case (no variance)
  y_pred_const <- rep(5, 5)
  y_true <- c(1, 2, 3, 4, 5)
  expected_const <- sqrt(mean((y_true - 5)^2))
  expect_equal(compute_calibrated_rmse(y_true, y_pred_const), expected_const, tolerance = 1e-9)

  # 4. Single element edge case
  expect_equal(compute_calibrated_rmse(c(1), c(2)), 0.0)
})

test_that("compute_calibrated_mae works correctly", {
  # 1. Perfect linear relation
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.1, 2.1, 3.1, 4.1, 5.1) # exact shift of +0.1
  # Optimal calibration: a = -0.1, b = 1.0, MAE = 0
  expect_equal(compute_calibrated_mae(y_true, y_pred), 0.0, tolerance = 1e-5)

  # 2. Standard case
  y_true <- c(1, 2, 3, 4, 5)
  y_pred <- c(1.5, 1.8, 3.2, 4.5, 4.8)
  val <- compute_calibrated_mae(y_true, y_pred)
  expect_true(val >= 0.0)
  expect_true(val <= mean(abs(y_true - y_pred))) # Calibrated MAE should be <= raw MAE

  # 3. Constant prediction edge case
  y_pred_const <- rep(5, 5)
  y_true <- c(1, 2, 3, 4, 5)
  # Optimal L1 calibration: a = median(y_true) = 3, b = 0. MAE = mean(abs(y_true - 3)) = (2+1+0+1+2)/5 = 1.2
  expect_equal(compute_calibrated_mae(y_true, y_pred_const), 1.2, tolerance = 1e-5)

  # 4. NA handling
  y_true <- c(1, 2, NA, 4, 5)
  y_pred <- c(1, 2, 3, 4, NA)
  val_na <- compute_calibrated_mae(y_true, y_pred)
  # valid pairs are c(1, 2, 4) vs c(1, 2, 4) -> perfect
  expect_equal(val_na, 0.0, tolerance = 1e-5)
})

test_that("realmlp evaluator returns non-null feature importances", {
  set.seed(42)
  d <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    y = rnorm(50)
  )

  ev <- evoFE::evo_evaluators[["realmlp"]]
  res <- ev$train_func(
    x_train = d[, c("x1", "x2")],
    y_train = d$y,
    task = "regression",
    seed = 42
  )

  expect_true(!is.null(res$importances))
  expect_equal(length(res$importances), 2)
  expect_named(res$importances, c("x1", "x2"))
  expect_equal(sum(res$importances), 1.0, tolerance = 1e-5)
  expect_true(all(res$importances >= 0))
})

test_that("realmlp evaluator emits progress messages when verbose and stays silent when not", {
  set.seed(42)
  d <- data.frame(
    x1 = rnorm(40),
    x2 = rnorm(40),
    y = rnorm(40)
  )

  ev <- evoFE::evo_evaluators[["realmlp"]]

  # 1. Verbose mode emits starting, epoch progress, and completion messages
  out_verb <- testthat::capture_output({
    msgs <- testthat::capture_messages({
      res_verb <- ev$train_func(
        x_train = d[, c("x1", "x2")],
        y_train = d$y,
        task = "regression",
        seed = 42,
        nrounds = 20,
        verbose = TRUE
      )
    })
  })

  expect_true(any(grepl("\\[RealMLP C\\+\\+ regression\\] Fitted.*rows", msgs)))
  expect_true(grepl("\\[RealMLP C\\+\\+ regression\\] Starting training:", out_verb))
  expect_true(grepl("Epoch\\s+\\d+/20", out_verb))

  # 2. Silent mode emits no messages or stdout
  out_silent <- testthat::capture_output({
    msgs_silent <- testthat::capture_messages({
      res_quiet <- ev$train_func(
        x_train = d[, c("x1", "x2")],
        y_train = d$y,
        task = "regression",
        seed = 42,
        nrounds = 20,
        verbose = FALSE
      )
    })
  })

  expect_equal(length(msgs_silent), 0)
  expect_equal(nchar(out_silent), 0)
})

test_that("realmlp evaluator works for binary classification and predict_func", {
  set.seed(42)
  n <- 60
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  prob <- 1 / (1 + exp(-(1.5 * x1 - 2.0 * x2)))
  y_bin <- as.factor(ifelse(prob > 0.5, "yes", "no"))
  d <- data.frame(x1 = x1, x2 = x2, y = y_bin)

  train_idx <- 1:45
  val_idx <- 46:60

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = d[train_idx, c("x1", "x2")],
    y_train = d$y[train_idx],
    x_val = d[val_idx, c("x1", "x2")],
    task = "classification",
    seed = 42,
    nrounds = 30,
    verbose = FALSE
  )

  expect_true(!is.null(fit$model))
  expect_equal(length(fit$predictions), length(val_idx))
  expect_true(all(fit$predictions >= 0 & fit$predictions <= 1))
  expect_equal(length(fit$importances), 2)
  expect_equal(sum(fit$importances), 1.0, tolerance = 1e-5)

  # Test predict_func on new data
  new_preds <- ev$predict_func(fit$model, d[val_idx, c("x1", "x2")], task = "classification")
  expect_equal(new_preds, fit$predictions, tolerance = 1e-6)

  # Test NA imputation in predict_func
  d_na <- d[val_idx, c("x1", "x2")]
  d_na[1, 1] <- NA
  preds_with_na <- ev$predict_func(fit$model, d_na, task = "classification")
  expect_true(!is.na(preds_with_na[1]))
})

test_that("realmlp evaluator works for multiclass classification", {
  set.seed(123)
  n <- 90
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  # 3 distinct classes
  y_cat <- factor(
    ifelse(x1 + x2 > 0.5, "ClassA",
           ifelse(x1 - x2 > 0.5, "ClassB", "ClassC")),
    levels = c("ClassA", "ClassB", "ClassC")
  )
  d <- data.frame(x1 = x1, x2 = x2, y = y_cat)

  train_idx <- 1:70
  val_idx <- 71:90

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = d[train_idx, c("x1", "x2")],
    y_train = d$y[train_idx],
    x_val = d[val_idx, c("x1", "x2")],
    task = "multiclass",
    seed = 123,
    nrounds = 30,
    verbose = FALSE
  )

  expect_true(!is.null(fit$model))
  expect_true(is.matrix(fit$predictions))
  expect_equal(nrow(fit$predictions), length(val_idx))
  expect_equal(ncol(fit$predictions), 3)
  expect_equal(colnames(fit$predictions), c("ClassA", "ClassB", "ClassC"))
  expect_equal(rowSums(fit$predictions), rep(1.0, length(val_idx)), tolerance = 1e-4)

  # Test predict_func
  new_preds <- ev$predict_func(fit$model, d[val_idx, c("x1", "x2")], task = "multiclass")
  expect_equal(new_preds, fit$predictions, tolerance = 1e-6)
})

test_that("realmlp evaluator early stopping works with validation data", {
  set.seed(42)
  n <- 80
  d <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = d[1:50, c("x1", "x2")],
    y_train = d$y[1:50],
    x_val = d[51:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 50,
    early_stopping_rounds = 5,
    y_val = d$y[51:80],
    verbose = FALSE
  )

  expect_true(!is.null(fit$model))
  expect_equal(length(fit$predictions), 30)
  expect_true(!is.null(fit$importances))
  expect_true(!is.null(fit$best_iteration))
  expect_true(fit$best_iteration <= 50)
  expect_equal(fit$best_iteration, fit$model$best_iteration)
})

test_that("evolve_features preserves configured epochs for final realmlp model when early stopping triggers", {
  set.seed(42)
  n <- 60
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )

  recipe <- evolve_features(
    data = df,
    target_col = "y",
    task = "regression",
    evaluator = "realmlp",
    generations = 1,
    pop_size = 2,
    evaluation_strategy = "split",
    split_ratio = c(0.7, 0.3, 0),
    epochs = 100,
    early_stopping_rounds = 3,
    verbose = FALSE,
    seed = 42
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_true(!is.null(recipe$best_iteration))
  expect_true(recipe$best_iteration < 100)
  expect_equal(recipe$best_individual$best_iteration, recipe$best_iteration)
  expect_true(!is.null(recipe$best_model))
  expect_equal(recipe$best_model$n_epochs, 100L)
  expect_equal(recipe$best_model$best_iteration, recipe$best_iteration)
  expect_equal(recipe$best_model$best_epoch, recipe$best_iteration)

  # Verify ensemble_islands also stops lazy island refits at each island's validated best_iteration
  # while preserving the 100-epoch coslog schedule horizon
  recipe_isl <- evolve_features(
    data = df,
    target_col = "y",
    task = "regression",
    evaluator = "realmlp",
    generations = 1,
    pop_size = 2,
    islands = 2,
    evaluation_strategy = "split",
    split_ratio = c(0.7, 0.3, 0),
    epochs = 100,
    early_stopping_rounds = 3,
    verbose = FALSE,
    seed = 42
  )
  ens <- ensemble_islands(recipe_isl, data = df, method = "equal", verbose = FALSE)
  for (nm in names(ens$active_models)) {
    expect_equal(ens$active_models[[nm]]$n_epochs, 100L)
    expect_equal(ens$active_models[[nm]]$best_iteration, ens$active_recipes[[nm]]$best_iteration)
  }
})

test_that("evolve_features scales best_iteration by data size for tree evaluators", {
  skip_if_not_installed("lightgbm")
  set.seed(42)
  n <- 100
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )

  recipe <- evolve_features(
    data = df,
    target_col = "y",
    task = "regression",
    evaluator = "lightgbm",
    generations = 1,
    pop_size = 2,
    evaluation_strategy = "split",
    split_ratio = c(0.8, 0.2, 0),
    nrounds = 100,
    early_stopping_rounds = 3,
    verbose = FALSE,
    seed = 42
  )

  expect_s3_class(recipe, "evo_recipe")
  expect_true(!is.null(recipe$best_iteration))
  expect_true(!is.null(recipe$best_individual$best_iteration))
  expect_true(!is.null(recipe$best_individual$train_size))
  expect_equal(recipe$best_individual$train_size, 80)
  # Scaled by 100 / 80 = 1.25
  expected_scaled <- as.integer(round(recipe$best_individual$best_iteration * 1.25))
  expect_equal(recipe$best_iteration, expected_scaled)
})

test_that("realmlp evaluator trains and achieves high accuracy on iris", {
  data(iris)
  set.seed(42)
  train_idx <- c(1:40, 51:90, 101:140)
  test_idx <- c(41:50, 91:100, 141:150)

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = iris[train_idx, 1:4],
    y_train = iris$Species[train_idx],
    task = "multiclass",
    seed = 42,
    nrounds = 50,
    verbose = FALSE
  )

  preds <- ev$predict_func(fit$model, iris[test_idx, 1:4], task = "multiclass")
  pred_cls <- colnames(preds)[max.col(preds)]
  acc <- mean(pred_cls == as.character(iris$Species[test_idx]))
  expect_gt(acc, 0.85)
})

test_that("realmlp evaluator respects seed parameter and has deterministic default without mutating RNG", {
  data(iris)
  train_idx <- 1:100
  test_idx <- 101:150
  ev <- evoFE::evo_evaluators[["realmlp"]]

  # 1. Explicit seed reproducibility
  fit_s1 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", seed = 42, nrounds = 20, verbose = FALSE)
  fit_s2 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", seed = 42, nrounds = 20, verbose = FALSE)
  p1 <- ev$predict_func(fit_s1$model, iris[test_idx, 1:4], task = "multiclass")
  p2 <- ev$predict_func(fit_s2$model, iris[test_idx, 1:4], task = "multiclass")
  expect_identical(p1, p2)

  # 2. Different seeds produce different results
  fit_s3 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", seed = 999, nrounds = 20, verbose = FALSE)
  p3 <- ev$predict_func(fit_s3$model, iris[test_idx, 1:4], task = "multiclass")
  expect_false(identical(p1, p3))

  # 3. realmlp_seed parameter works like seed
  fit_s4 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", realmlp_seed = 999, nrounds = 20, verbose = FALSE)
  p4 <- ev$predict_func(fit_s4$model, iris[test_idx, 1:4], task = "multiclass")
  expect_identical(p3, p4)

  # 4. Deterministic default (42L) when seed is omitted, without mutating R's .Random.seed
  set.seed(12345)
  rng_before <- .Random.seed
  fit_d1 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", nrounds = 20, verbose = FALSE)
  p_d1 <- ev$predict_func(fit_d1$model, iris[test_idx, 1:4], task = "multiclass")
  rng_after <- .Random.seed
  expect_identical(rng_before, rng_after)

  # Default matches explicit seed = 42
  expect_identical(p_d1, p1)

  # Second call without seed also identical and doesn't mutate RNG
  fit_d2 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", nrounds = 20, verbose = FALSE)
  p_d2 <- ev$predict_func(fit_d2$model, iris[test_idx, 1:4], task = "multiclass")
  expect_identical(p_d1, p_d2)
  expect_identical(rng_before, .Random.seed)
})

test_that("realmlp evaluator errors gracefully on unsupported task", {
  set.seed(42)
  d <- data.frame(x1 = rnorm(20), x2 = rnorm(20), y = rnorm(20))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  expect_error(
    ev$train_func(d[, c("x1", "x2")], d$y, task = "unknown", nrounds = 5, verbose = FALSE),
    "Unsupported task"
  )
})

test_that("realmlp evaluator handles NA values in training data", {
  set.seed(42)
  n <- 60
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), y = rnorm(n))
  # Inject NAs into training features
  d$x1[c(1, 5, 10)] <- NA
  d$x2[c(3, 7)] <- NA

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = d[1:40, c("x1", "x2")],
    y_train = d$y[1:40],
    x_val = d[41:60, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 10,
    verbose = FALSE
  )

  expect_true(!is.null(fit$model))
  expect_equal(length(fit$predictions), 20)
  expect_true(!any(is.na(fit$predictions)))
})

test_that("realmlp evaluator supports configurable hidden_dim parameter", {
  set.seed(42)
  d <- data.frame(x1 = rnorm(30), x2 = rnorm(30), y = rnorm(30))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  fit_128 <- ev$train_func(
    x_train = d[1:20, c("x1", "x2")],
    y_train = d$y[1:20],
    x_val = d[21:30, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 5,
    hidden_dim = 128,
    verbose = FALSE
  )

  expect_true(!is.null(fit_128$model))
  expect_equal(fit_128$model$model_state$hidden_dim, 128L)
  expect_equal(length(fit_128$predictions), 10)
})

test_that("realmlp evaluator supports configurable batch_size parameter", {
  set.seed(42)
  d <- data.frame(x1 = rnorm(80), x2 = rnorm(80), y = rnorm(80))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  # Test custom batch_size = 16
  fit_b16 <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 5,
    batch_size = 16,
    verbose = FALSE
  )

  expect_true(!is.null(fit_b16$model))
  expect_equal(length(fit_b16$predictions), 20)

  # Test large batch_size (full-batch equivalent)
  fit_b60 <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 5,
    batch_size = 60,
    verbose = FALSE
  )

  expect_true(!is.null(fit_b60$model))
  expect_equal(length(fit_b60$predictions), 20)
})

test_that("realmlp evaluator supports configurable n_cycles parameter and auto-scaling", {
  set.seed(42)
  d <- data.frame(x1 = rnorm(80), x2 = rnorm(80), y = rnorm(80))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  # Default is 4 cycles
  fit_default <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 5,
    verbose = FALSE
  )
  expect_equal(fit_default$model$n_cycles, 4L)

  # Explicit 2 cycles
  fit_c2 <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 5,
    realmlp_n_cycles = 2,
    verbose = FALSE
  )
  expect_equal(fit_c2$model$n_cycles, 2L)

  # Auto-scaling with n_cycles = 0 (nrounds = 50 < 64 -> 2 cycles)
  fit_auto_small <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 50,
    n_cycles = 0,
    verbose = FALSE
  )
  expect_equal(fit_auto_small$model$n_cycles, 2L)

  # Auto-scaling with n_cycles = "auto" (nrounds = 100 < 150 -> 3 cycles)
  fit_auto_med <- ev$train_func(
    x_train = d[1:60, c("x1", "x2")],
    y_train = d$y[1:60],
    x_val = d[61:80, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 100,
    n_cycles = "auto",
    verbose = FALSE
  )
  expect_equal(fit_auto_med$model$n_cycles, 3L)
})

test_that("realmlp evaluator supports cal-rmse and cal-mae regression metrics with early stopping", {
  set.seed(42)
  n <- 80
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), y = rnorm(n))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  fit_rmse <- ev$train_func(
    x_train = d[1:50, c("x1", "x2")],
    y_train = d$y[1:50],
    x_val = d[51:80, c("x1", "x2")],
    y_val = d$y[51:80],
    task = "regression",
    metric = "cal_rmse",
    seed = 42,
    nrounds = 20,
    early_stopping_rounds = 4,
    verbose = FALSE
  )
  expect_true(!is.null(fit_rmse$model))
  expect_true(!is.null(fit_rmse$best_iteration))
  expect_true(fit_rmse$best_iteration <= 20)

  fit_mae <- ev$train_func(
    x_train = d[1:50, c("x1", "x2")],
    y_train = d$y[1:50],
    x_val = d[51:80, c("x1", "x2")],
    y_val = d$y[51:80],
    task = "regression",
    metric = "cal_mae",
    seed = 42,
    nrounds = 20,
    early_stopping_rounds = 4,
    verbose = FALSE
  )
  expect_true(!is.null(fit_mae$model))
  expect_true(!is.null(fit_mae$best_iteration))
  expect_true(fit_mae$best_iteration <= 20)
})

test_that("realmlp feature importances sharply separate signal from noise", {
  set.seed(42)
  n <- 200
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y <- 5 * x1 + rnorm(n, sd = 0.1)
  d <- data.frame(x1 = x1, x2 = x2, y = y)

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = d[1:150, c("x1", "x2")],
    y_train = d$y[1:150],
    x_val = d[151:200, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 30,
    verbose = FALSE
  )

  expect_true(!is.null(fit$importances))
  expect_true(fit$importances["x1"] > 0.70)
  expect_true(fit$importances["x1"] > 5 * fit$importances["x2"])
})

test_that("realmlp is numerically robust against near-zero variance features, Inf, and OOD validation outliers", {
  set.seed(42)
  n_tr <- 100
  n_val <- 30

  # x1: real signal
  # x2: normal noise
  # x3: near-zero variance feature (std ~ 1e-7, below standard deviation threshold 1e-5)
  x_tr <- data.frame(
    x1 = rnorm(n_tr),
    x2 = rnorm(n_tr),
    x3 = 1.0 + rnorm(n_tr) * 1e-7
  )
  y_tr <- 73000 + 30000 * x_tr$x1 + rnorm(n_tr, sd = 1000)

  x_v <- data.frame(
    x1 = rnorm(n_val),
    x2 = rnorm(n_val),
    x3 = rnorm(n_val)
  )
  # Inject extreme values and Infs in validation set
  x_v$x3[1] <- 100.0
  x_v$x2[2] <- Inf
  x_v$x1[3] <- -Inf

  y_v <- 73000 + 30000 * x_v$x1 + rnorm(n_val, sd = 1000)
  y_v[3] <- 73000

  ev <- evoFE::evo_evaluators[["realmlp"]]
  fit <- ev$train_func(
    x_train = x_tr,
    y_train = y_tr,
    x_val = x_v,
    y_val = y_v,
    task = "regression",
    seed = 42,
    nrounds = 10,
    early_stopping_rounds = 3,
    verbose = FALSE
  )

  expect_true(!is.null(fit$model))
  expect_equal(length(fit$predictions), n_val)
  expect_true(all(is.finite(fit$predictions)))
  expect_lt(max(abs(fit$predictions)), 1e7)

  # Check predict_func also handles OOD and non-finite safely
  x_test <- data.frame(
    x1 = c(1.0, NaN, 1e8),
    x2 = c(Inf, -Inf, 0.0),
    x3 = c(50.0, 1.0, -100.0)
  )
  preds_test <- ev$predict_func(fit$model, x_test, task = "regression")
  expect_true(all(is.finite(preds_test)))
  expect_lt(max(abs(preds_test)), 1e7)
})

test_that("evaluate_pop retains finite fitness and does not assign -Inf when gene collapses to cached individual", {
  set.seed(42)
  df <- data.frame(x1 = stats::rnorm(50), x2 = stats::rnorm(50), target = stats::rnorm(50))
  ind1 <- create_individual(genes = list(), numeric_cols = c("x1", "x2"))
  g_red <- create_gene("add", c("x1", "x1"))
  ind2 <- create_individual(genes = list(g_red), numeric_cols = c("x1", "x2"))
  pop <- list(ind1, ind2)
  cache <- new.env(hash = TRUE, parent = emptyenv())
  state <- new.env(hash = TRUE, parent = emptyenv())
  s_ids <- c(rep("train", 35), rep("val", 15))

  msgs <- testthat::capture_messages({
    res <- evaluate_pop(
      pop = pop, data = df, target_col = "target", task = "regression",
      cv_folds = 2, evaluation_strategy = "split", split_ids = s_ids,
      shared_splits = NULL, evaluator = "lightgbm", fold_ids = NULL,
      shared_folds = NULL, shared_full = NULL, state_cache = state,
      fitness_cache = cache, threads = 1, verbose = TRUE, running_best_fitness = -Inf
    )
  })

  expect_true(is.finite(res$pop[[1]]$fitness))
  expect_true(is.finite(res$pop[[2]]$fitness))
  expect_equal(res$pop[[1]]$fitness, res$pop[[2]]$fitness)
  expect_false(res$pop[[2]]$fitness == -Inf)

  # Check that output message does not contain -Inf or pruned: constant/collinear gene
  expect_false(any(grepl("Fitness: -Inf", msgs)))
  expect_true(any(grepl("Tested Individual 2 -> Fitness: [^ ]+ \\(pruned: cached\\)", msgs)))

  # Check cache holds valid finite fitness
  k2 <- individual_to_cache_key(ind2)
  expect_true(exists(k2, envir = cache))
  expect_true(is.finite(cache[[k2]]$fitness))
})

test_that("realmlp defaults learning rate to 0.20 for regression and 0.04 for classification when lr <= 0", {
  set.seed(42)
  d_reg <- data.frame(x1 = rnorm(50), x2 = rnorm(50), y = rnorm(50))
  ev <- evoFE::evo_evaluators[["realmlp"]]

  fit_reg <- ev$train_func(
    x_train = d_reg[1:40, c("x1", "x2")],
    y_train = d_reg$y[1:40],
    x_val = d_reg[41:50, c("x1", "x2")],
    task = "regression",
    seed = 42,
    nrounds = 2,
    realmlp_lr = -1.0,
    verbose = FALSE
  )
  expect_equal(fit_reg$model$base_lr, 0.20)

  d_cls <- data.frame(x1 = rnorm(50), x2 = rnorm(50), y = sample(c(0, 1), 50, replace = TRUE))
  fit_cls <- ev$train_func(
    x_train = d_cls[1:40, c("x1", "x2")],
    y_train = d_cls$y[1:40],
    x_val = d_cls[41:50, c("x1", "x2")],
    task = "classification",
    seed = 42,
    nrounds = 2,
    realmlp_lr = -1.0,
    verbose = FALSE
  )
  expect_equal(fit_cls$model$base_lr, 0.04)
})

test_that("catboost handles zero and negative early_stopping_rounds without C++ abort", {
  skip_if(system.file(package = "catboost") == "", "catboost package is not installed")
  set.seed(42)
  x_tr <- matrix(rnorm(40), nrow = 20, ncol = 2)
  colnames(x_tr) <- c("x1", "x2")
  y_tr <- rbinom(20, 1, 0.5)
  x_va <- matrix(rnorm(20), nrow = 10, ncol = 2)
  colnames(x_va) <- c("x1", "x2")
  y_va <- rbinom(10, 1, 0.5)

  # 1. early_stopping_rounds = 0 (both with and without validation data)
  expect_no_error({
    fit0 <- train_model(x_tr, y_tr, x_val = x_va, y_val = y_va,
                        evaluator = "catboost", nrounds = 5,
                        early_stopping_rounds = 0L)
  })

  expect_no_error({
    fit_noval <- train_model(x_tr, y_tr, x_val = NULL, y_val = NULL,
                             evaluator = "catboost", nrounds = 5,
                             early_stopping_rounds = 10L)
  })

  # 2. early_stopping_round (singular alias) = 0
  expect_no_error({
    fit_sing <- train_model(x_tr, y_tr, x_val = x_va, y_val = y_va,
                            evaluator = "catboost", nrounds = 5,
                            early_stopping_round = 0L)
  })

  # 3. Parameter aliases and out-of-range depths
  expect_no_error({
    fit_aliases <- train_model(
      x_tr, y_tr, x_val = x_va, y_val = y_va,
      evaluator = "catboost", nrounds = 5,
      max_depth = 20L,
      colsample_bytree = 0.75,
      tree_method = "hist",
      subsample = 0.8,
      bagging_temperature = 0.5
    )
  })
  expect_true(!is.null(fit_aliases$model))
  expect_equal(fit_aliases$model$feature_count, 2L)

  # 4. Incompatible or custom metrics passed as eval_metric do not crash CatBoost
  expect_no_error({
    fit_metric <- train_model(
      x_tr, y_tr, x_val = x_va, y_val = y_va,
      evaluator = "catboost", nrounds = 5,
      eval_metric = "ts_refinement"
    )
  })
})

test_that("lm evaluator supports configurable alpha, s = 'lambda.1se', metric-aligned type.measure, and single-row multiclass predict", {
  testthat::skip_if_not_installed("glmnet")

  set.seed(42)
  n <- 60
  x <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n)
  )
  y_reg <- 2 * x$x1 - 1.5 * x$x2 + rnorm(n, sd = 0.1)

  ev <- evoFE::get_evaluator("lm")

  # 1. Custom alpha = 1.0 (Lasso) and alpha = 0.0 (Ridge)
  fit_lasso <- ev$train_func(x_train = x, y_train = y_reg, task = "regression", alpha = 1.0)
  expect_equal(fit_lasso$model$s, "lambda.min")
  expect_equal(fit_lasso$model$call$alpha, 1.0)
  pred_lasso <- ev$predict_func(fit_lasso$model, x_new = x[1:5, ], task = "regression")
  expect_length(pred_lasso, 5)
  expect_true(all(is.finite(pred_lasso)))

  fit_ridge <- ev$train_func(x_train = x, y_train = y_reg, task = "regression", alpha = 0.0)
  expect_equal(fit_ridge$model$call$alpha, 0.0)
  pred_ridge <- ev$predict_func(fit_ridge$model, x_new = x[1:5, ], task = "regression")
  expect_length(pred_ridge, 5)
  expect_true(all(is.finite(pred_ridge)))

  # 2. Configurable s = "lambda.1se"
  fit_1se <- ev$train_func(x_train = x, y_train = y_reg, task = "regression", s = "lambda.1se")
  expect_equal(fit_1se$model$s, "lambda.1se")
  pred_1se <- ev$predict_func(fit_1se$model, x_new = x[1:5, ], task = "regression")
  expect_length(pred_1se, 5)
  expect_true(all(is.finite(pred_1se)))

  # Override s in predict_func
  pred_min <- ev$predict_func(fit_1se$model, x_new = x[1:5, ], task = "regression", s = "lambda.min")
  expect_length(pred_min, 5)
  expect_true(all(is.finite(pred_min)))

  # 3. Metric-aligned type.measure in cv.glmnet
  # Classification with metric = "auc" -> type.measure = "auc"
  y_bin <- as.integer(x$x1 + x$x2 > 0)
  fit_auc <- ev$train_func(x_train = x, y_train = y_bin, task = "classification", metric = "auc", nfolds = 3)
  expect_equal(names(fit_auc$model$name), "auc")

  # Regression with metric = "mae" -> type.measure = "mae"
  fit_mae <- ev$train_func(x_train = x, y_train = y_reg, task = "regression", metric = "mae", nfolds = 3)
  expect_equal(names(fit_mae$model$name), "mae")

  # Regression with default metric (rmse) -> type.measure = "mse"
  fit_mse <- ev$train_func(x_train = x, y_train = y_reg, task = "regression", metric = "rmse", nfolds = 3)
  expect_equal(names(fit_mse$model$name), "mse")

  # 4. Multiclass and single-row predict matrix shape
  y_multi <- as.factor(rep(c("A", "B", "C"), length.out = n))
  fit_multi <- ev$train_func(x_train = x, y_train = y_multi, task = "multiclass", num_class = 3, nfolds = 3)
  expect_equal(fit_multi$model$s, "lambda.min")

  # Single-row input to predict_func
  single_row <- x[1, , drop = FALSE]
  pred_single <- ev$predict_func(fit_multi$model, x_new = single_row, task = "multiclass")
  expect_true(is.matrix(pred_single))
  expect_equal(nrow(pred_single), 1L)
  expect_equal(ncol(pred_single), 3L)

  # Multi-row input to predict_func
  multi_rows <- x[1:4, , drop = FALSE]
  pred_multi <- ev$predict_func(fit_multi$model, x_new = multi_rows, task = "multiclass")
  expect_true(is.matrix(pred_multi))
  expect_equal(nrow(pred_multi), 4L)
  expect_equal(ncol(pred_multi), 3L)

  # 5. Degeneracy handling (single-class edge case)
  y_degen <- rep(1L, n)
  fit_degen <- ev$train_func(x_train = x, y_train = y_degen, task = "classification")
  expect_true(isTRUE(fit_degen$model$is_degenerate))
  pred_degen <- ev$predict_func(fit_degen$model, x_new = single_row, task = "classification")
  expect_equal(pred_degen, 1.0)
})




