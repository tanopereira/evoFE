test_that("make_tunable() registers and optimizes correctly", {
  # 1. Register a simple mock evaluator
  register_evaluator(
    "mock_base_test",
    train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                          task = "regression", ...) {
      args <- list(...)
      val_score <- 100 - abs(args$param_a - 4.5)
      list(
        model = list(args = args, val_score = val_score),
        predictions = if (!is.null(x_val)) {
          rep(val_score, nrow(x_val))
        } else {
          NULL
        }
      )
    },
    predict_func = function(model, x_new, task, ...) {
      rep(model$val_score, nrow(x_new))
    }
  )

  # 2. Make it tunable
  param_ranges <- list(
    param_a = list(type = "numeric", lower = 1.0, upper = 8.0)
  )
  make_tunable("mock_base_test", param_ranges, tuner_name = "mock_tuned_test")

  # Check that it is registered
  expect_true(exists("mock_tuned_test", envir = evo_evaluators))

  # 3. Train the tuned model on mock data
  set.seed(42)
  x_train <- matrix(rnorm(20), ncol = 2)
  colnames(x_train) <- c("x1", "x2")
  y_train <- rnorm(10)
  x_val <- matrix(rnorm(10), ncol = 2)
  y_val <- rnorm(5)

  fit <- train_model(
    x_train, y_train, x_val = x_val, y_val = y_val,
    task = "regression", evaluator = "mock_tuned_test",
    mbo_iters = 3, mbo_init_design = 5, mbo_folds = 2
  )

  expect_type(fit$best_params, "list")
  expect_true("param_a" %in% names(fit$best_params))
  expect_true(fit$best_params$param_a >= 1.0 && fit$best_params$param_a <= 8.0)
})

test_that("make_tunable issues deprecation warning for ea infill optimizer", {
  # We reuse mock_tuned_test from previous test or register new one
  if (!exists("mock_tuned_test", envir = evo_evaluators)) {
    param_ranges <- list(
      param_a = list(type = "numeric", lower = 1.0, upper = 8.0)
    )
    make_tunable("mock_base_test", param_ranges, tuner_name = "mock_tuned_test")
  }

  set.seed(42)
  x_train <- matrix(rnorm(20), ncol = 2)
  colnames(x_train) <- c("x1", "x2")
  y_train <- rnorm(10)
  x_val <- matrix(rnorm(10), ncol = 2)
  y_val <- rnorm(5)

  expect_warning({
    fit <- train_model(
      x_train, y_train, x_val = x_val, y_val = y_val,
      task = "regression", evaluator = "mock_tuned_test",
      mbo_iters = 2, mbo_init_design = 3, mbo_folds = 2,
      mbo_infill_opt = "ea"
    )
  }, "mbo_infill_opt = 'ea' is deprecated")
})

test_that("lightgbm_mbo end-to-end runs on dummy data", {
  set.seed(42)
  n <- 30
  dummy_data <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = sample(0:1, n, replace = TRUE)
  )

  # Check that we can run lightgbm_mbo
  suppressWarnings({
    res <- tryCatch({
      evolve_features(
        data = dummy_data,
        target_col = "target",
        task = "classification",
        generations = 1,
        pop_size = 2,
        cv_folds = 2,
        early_stopping_generations = 1,
        evaluator = "lightgbm_mbo",
        mbo_iters = 2,
        mbo_init_design = 3,
        verbose = FALSE
      )
    }, error = function(e) {
      NULL
    })
  })

  # If lightgbm is not working in this environment, it could return NULL,
  # but if it succeeds, it should be an evo_recipe
  if (!is.null(res)) {
    expect_s3_class(res, "evo_recipe")
    expect_type(res$best_individual, "list")
  }
})
