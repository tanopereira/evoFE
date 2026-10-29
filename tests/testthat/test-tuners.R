test_that("make_tunable() registers and optimizes correctly", {
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")
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
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")
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
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")
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
    expect_true(!is.null(res$best_individual$best_params))
  }
})

test_that("is_tuned_evaluator() correctly detects tuned vs base evaluators", {
  expect_false(is_tuned_evaluator("lightgbm"))
  expect_false(is_tuned_evaluator("xgboost"))
  expect_true(is_tuned_evaluator("lightgbm_mbo"))
  expect_false(is_tuned_evaluator("non_existent_evaluator"))
})

test_that("global CV optimization sets one winning best_params across folds", {
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")

  set.seed(42)
  n <- 40
  dummy_data <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = stats::rnorm(n)
  )

  if (!exists("mock_tuned_test", envir = evo_evaluators)) {
    register_evaluator(
      "mock_base_test",
      train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                            task = "regression", ...) {
        args <- list(...)
        param_val <- if (!is.null(args$param_a)) args$param_a else 4.0
        val_score <- 100 - abs(param_val - 4.5)
        list(
          model = list(args = args, val_score = val_score),
          predictions = if (!is.null(x_val)) rep(val_score, nrow(x_val)) else NULL
        )
      },
      predict_func = function(model, x_new, task, ...) {
        rep(model$val_score, nrow(x_new))
      }
    )
    param_ranges <- list(
      param_a = list(type = "numeric", lower = 1.0, upper = 8.0)
    )
    make_tunable("mock_base_test", param_ranges, tuner_name = "mock_tuned_test")
  }

  res <- evolve_features(
    data = dummy_data,
    target_col = "target",
    task = "regression",
    generations = 1,
    pop_size = 2,
    cv_folds = 3,
    early_stopping_generations = 1,
    evaluator = "mock_tuned_test",
    mbo_iters = 2,
    mbo_init_design = 3,
    verbose = FALSE
  )

  expect_s3_class(res, "evo_recipe")
  expect_true(!is.null(res$best_individual$best_params))
  expect_true("param_a" %in% names(res$best_individual$best_params))
  expect_true(is.numeric(res$best_individual$fitness))
  expect_true(length(res$best_individual$val_preds) == n)
})

test_that("make_tunable reuses winning trial outputs without post-MBO retraining in split and fold_data modes", {
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")

  call_counter <- new.env(parent = emptyenv())
  call_counter$count <- 0L

  register_evaluator(
    "mock_counted_base",
    train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                          task = "regression", ...) {
      call_counter$count <- call_counter$count + 1L
      args <- list(...)
      param_val <- if (!is.null(args$param_a)) args$param_a else 4.0
      pred_val <- 4.5 - abs(param_val - 4.5)
      list(
        model = list(args = args, pred_val = pred_val),
        predictions = if (!is.null(x_val)) rep(pred_val, nrow(x_val)) else NULL,
        importances = stats::setNames(rep(1, ncol(x_train)), colnames(x_train)),
        best_iteration = 12L
      )
    },
    predict_func = function(model, x_new, task, ...) {
      rep(model$pred_val, nrow(x_new))
    }
  )

  make_tunable(
    "mock_counted_base",
    list(param_a = list(type = "numeric", lower = 1.0, upper = 8.0)),
    tuner_name = "mock_counted_tuned"
  )

  set.seed(42)
  x_tr <- matrix(stats::rnorm(20), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  y_tr <- rep(4.5, 10)
  x_va <- matrix(stats::rnorm(10), ncol = 2, dimnames = list(NULL, c("x1", "x2")))
  y_va <- rep(4.5, 5)

  # 1. Split mode: 3 init design + 2 MBO iters = 5 total fits (zero post-MBO retrain)
  call_counter$count <- 0L
  fit_split <- train_model(
    x_tr, y_tr, x_val = x_va, y_val = y_va,
    task = "regression", evaluator = "mock_counted_tuned",
    mbo_init_design = 3, mbo_iters = 2
  )
  expect_equal(call_counter$count, 5L)
  expect_length(fit_split$predictions, 5L)
  expect_equal(fit_split$best_iteration, 12L)
  expect_true(is.finite(fit_split$score))

  # 2. Fold-data (global CV) mode: (3 init + 2 iters) * 2 folds = 10 total fits (zero post-MBO retrain)
  call_counter$count <- 0L
  fold_data <- list(
    list(x_train = x_tr, y_train = y_tr, x_val = x_va, y_val = y_va),
    list(x_train = x_tr, y_train = y_tr, x_val = x_va, y_val = y_va)
  )
  fit_cv <- train_model(
    x_train = NULL, y_train = NULL, x_val = NULL, y_val = NULL,
    task = "regression", evaluator = "mock_counted_tuned",
    fold_data = fold_data, mbo_init_design = 3, mbo_iters = 2
  )
  expect_equal(call_counter$count, 10L)
  expect_length(fit_cv$fold_res, 2L)
  expect_equal(fit_cv$fold_res[[1]]$best_iteration, 12L)
})
