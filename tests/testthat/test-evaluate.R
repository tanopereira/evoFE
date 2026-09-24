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

test_that("realmlp evaluator respects seed parameter and global set.seed", {
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

  # 3. Global set.seed reproducibility when seed parameter is omitted
  set.seed(777)
  fit_g1 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", nrounds = 20, verbose = FALSE)
  p_g1 <- ev$predict_func(fit_g1$model, iris[test_idx, 1:4], task = "multiclass")

  set.seed(777)
  fit_g2 <- ev$train_func(iris[train_idx, 1:4], iris$Species[train_idx], task = "multiclass", nrounds = 20, verbose = FALSE)
  p_g2 <- ev$predict_func(fit_g2$model, iris[test_idx, 1:4], task = "multiclass")

  expect_identical(p_g1, p_g2)
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

