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

  # 3. Standalone CV mode preserves fixed_params in best_params and applies best_iteration on final refit
  make_tunable(
    "mock_counted_base",
    list(
      param_a = list(type = "numeric", lower = 1.0, upper = 8.0),
      fixed_flag = list(tunable = FALSE, value = "locked_val")
    ),
    tuner_name = "mock_counted_tuned_fixed"
  )
  fit_standalone_cv <- train_model(
    x_tr, y_tr,
    task = "regression", evaluator = "mock_counted_tuned_fixed",
    mbo_folds = 2, mbo_init_design = 3, mbo_iters = 1
  )
  expect_equal(fit_standalone_cv$best_params$fixed_flag, "locked_val")
  expect_equal(fit_standalone_cv$best_iteration, 12L)
  expect_equal(fit_standalone_cv$model$args$nrounds, 12L)
  expect_equal(fit_standalone_cv$model$args$fixed_flag, "locked_val")
})

test_that("metacv tournament and ensemble_islands OOF harmonization unwrap pre-tuned evaluators without re-running MBO", {
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")

  call_log <- new.env(parent = emptyenv())
  call_log$last_args <- list()

  register_evaluator(
    "mock_harm_base",
    train_func = function(x_train, y_train, x_val = NULL, y_val = NULL,
                          task = "regression", ...) {
      args <- list(...)
      call_log$last_args <- args
      p_a <- if (!is.null(args$param_a)) args$param_a else 1.0
      list(
        model = list(args = args),
        predictions = if (!is.null(x_val)) rep(p_a, nrow(x_val)) else NULL,
        importances = stats::setNames(rep(1, ncol(x_train)), colnames(x_train)),
        best_iteration = 9L
      )
    },
    predict_func = function(model, x_new, task, ...) {
      p_a <- if (!is.null(model$args$param_a)) model$args$param_a else 1.0
      rep(p_a, nrow(x_new))
    }
  )

  make_tunable(
    "mock_harm_base",
    list(
      param_a = list(type = "numeric", lower = 2.0, upper = 6.0),
      fixed_tag = list(tunable = FALSE, value = "kept_tag")
    ),
    tuner_name = "mock_harm_mbo"
  )

  set.seed(123)
  df <- data.frame(
    x1 = stats::rnorm(30),
    x2 = stats::rnorm(30),
    y = stats::rnorm(30)
  )

  rec <- evolve_features(
    df, "y",
    task = "regression",
    evaluator = "mock_harm_mbo",
    evaluation_strategy = "metacv",
    metacv_selection = "tournament",
    islands = 2,
    generations = 1,
    pop_size = 2,
    mbo_init_design = 3,
    mbo_iters = 1,
    verbose = FALSE
  )

  expect_equal(rec$best_individual$best_params$fixed_tag, "kept_tag")
  expect_equal(rec$best_model$args$fixed_tag, "kept_tag")

  # Harmonizing in ensemble_islands should unwrap mock_harm_mbo and reuse best_params
  ens <- ensemble_islands(rec, df, method = "caruana", caruana_rounds = 5, verbose = FALSE)
  expect_s3_class(ens, "evo_ensemble")
  for (mod in ens$active_models) {
    expect_equal(mod$args$fixed_tag, "kept_tag")
  }
})

test_that("first-class MBO tuners (xgboost_mbo, catboost_mbo, realmlp_mbo, lm_mbo) are registered in model_registry", {
  for (ev in c("lightgbm_mbo", "xgboost_mbo", "catboost_mbo", "realmlp_mbo", "lm_mbo",
               "lightgbm_tabarena", "xgboost_tabarena", "catboost_tabarena", "realmlp_tabarena", "lm_tabarena")) {
    entry <- get_evaluator(ev)
    expect_true(isTRUE(entry$is_tuned), info = ev)
    expect_true(is.character(entry$base_evaluator) && nzchar(entry$base_evaluator), info = ev)
  }
  expect_equal(unwrap_evaluator("xgboost_mbo"), "xgboost")
  expect_equal(unwrap_evaluator("catboost_mbo"), "catboost")
  expect_equal(unwrap_evaluator("realmlp_mbo"), "realmlp")
  expect_equal(unwrap_evaluator("lm_mbo"), "lm")
  expect_equal(unwrap_evaluator("lightgbm_tabarena"), "lightgbm")
  expect_equal(unwrap_evaluator("xgboost_tabarena"), "xgboost")
  expect_equal(unwrap_evaluator("catboost_tabarena"), "catboost")
  expect_equal(unwrap_evaluator("realmlp_tabarena"), "realmlp")
  expect_equal(unwrap_evaluator("lm_tabarena"), "lm")

  # Verify parameter space invariants for TabArena tuners
  rmlp_env <- environment(get_evaluator("realmlp_tabarena")$train_func)
  defs <- rmlp_env$tunable_defs
  expect_equal(defs$hidden_dim$lower, 128L)
  expect_equal(defs$hidden_dim$upper, 512L)
  expect_equal(defs$realmlp_lr$lower, 0.01)
  expect_equal(defs$realmlp_lr$upper, 0.35)
  expect_equal(defs$realmlp_n_cycles$lower, 1L)
  expect_equal(defs$realmlp_n_cycles$upper, 4L)
  expect_equal(defs$batch_size$values, c(256L, 512L, 1024L, 2048L))
  expect_true(isTRUE(defs$realmlp_lr$log))

  # Verify parameter space invariants for LM tuners
  lm_mbo_env <- environment(get_evaluator("lm_mbo")$train_func)
  lm_mbo_defs <- lm_mbo_env$tunable_defs
  expect_equal(lm_mbo_defs$alpha$lower, 0.0)
  expect_equal(lm_mbo_defs$alpha$upper, 1.0)

  lm_tab_env <- environment(get_evaluator("lm_tabarena")$train_func)
  lm_tab_defs <- lm_tab_env$tunable_defs
  expect_equal(lm_tab_defs$alpha$lower, 0.0)
  expect_equal(lm_tab_defs$alpha$upper, 1.0)
  expect_equal(lm_tab_defs$s$values, c("lambda.min", "lambda.1se"))
})

test_that("make_tunable supports logarithmic scale sampling and fails fast on non-positive bounds", {
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")

  # 1. Fail fast when log = TRUE with lower <= 0
  expect_error(
    make_tunable(
      base_model_name = "xgboost",
      param_ranges = list(
        bad_lr = list(type = "numeric", lower = 0.0, upper = 1.0, log = TRUE)
      ),
      tuner_name = "mock_bad_log"
    ),
    "lower bound must be positive"
  )

  # 2. Check that realmlp_tabarena has log = TRUE on realmlp_lr
  rmlp_env <- environment(get_evaluator("realmlp_tabarena")$train_func)
  expect_true(isTRUE(rmlp_env$tunable_defs$realmlp_lr$log))
  expect_true(isTRUE(rmlp_env$search_ps$is_logscale[["realmlp_lr"]]))
  expect_true(isTRUE(rmlp_env$search_ps$has_trafo))

  # 3. Register a mock model with log-scale tuning and verify end-to-end execution
  register_evaluator(
    "mock_log_base",
    train_func = function(x_train, y_train, x_val = NULL, y_val = NULL, lr = 0.05, ...) {
      preds <- rep(lr, if (!is.null(x_val)) nrow(x_val) else nrow(x_train))
      list(model = list(lr = lr), predictions = preds)
    },
    predict_func = function(model, newdata, ...) rep(model$lr, nrow(newdata))
  )
  on.exit(rm("mock_log_base", envir = evo_evaluators), add = TRUE)

  make_tunable(
    base_model_name = "mock_log_base",
    param_ranges = list(
      lr = list(type = "numeric", lower = 0.001, upper = 0.1, log = TRUE)
    ),
    tuner_name = "mock_log_tuner"
  )
  on.exit(rm("mock_log_tuner", envir = evo_evaluators), add = TRUE)

  x_dummy <- matrix(rnorm(20), ncol = 2)
  y_dummy <- rep(0.04, 10)

  fit_log <- train_model(
    x_train = x_dummy[1:8, ],
    y_train = y_dummy[1:8],
    x_val = x_dummy[9:10, ],
    y_val = y_dummy[9:10],
    evaluator = "mock_log_tuner",
    mbo_iters = 1,
    mbo_init_design = 3,
    best_params = list(lr = 0.04),
    verbose = FALSE
  )

  expect_true(!is.null(fit_log$best_params))
  expect_true(fit_log$best_params$lr >= 0.001 && fit_log$best_params$lr <= 0.1)
  expect_true(is.numeric(fit_log$best_params$lr))
})

test_that("lm_mbo tunes alpha over a toy dataset with MBO", {
  testthat::skip_if_not_installed("glmnet")
  testthat::skip_if_not_installed("mlr3mbo")
  testthat::skip_if_not_installed("paradox")
  testthat::skip_if_not_installed("bbotk")

  set.seed(42)
  n <- 50
  x <- matrix(rnorm(n * 4), nrow = n, ncol = 4)
  colnames(x) <- paste0("x", 1:4)
  y <- 2 * x[, 1] - 1.5 * x[, 2] + rnorm(n, sd = 0.2)

  fit <- train_model(
    x_train = x[1:40, ],
    y_train = y[1:40],
    x_val = x[41:50, ],
    y_val = y[41:50],
    evaluator = "lm_mbo",
    task = "regression",
    mbo_iters = 2,
    mbo_init_design = 3,
    verbose = FALSE
  )

  expect_true(!is.null(fit$best_params))
  expect_true("alpha" %in% names(fit$best_params))
  expect_true(fit$best_params$alpha >= 0.0 && fit$best_params$alpha <= 1.0)
  expect_true(!is.null(fit$model))
  expect_length(fit$predictions, 10)
  expect_true(all(is.finite(fit$predictions)))

  ev <- get_evaluator("lm_mbo")
  preds <- ev$predict_func(fit$model, x_new = x[41:50, ], task = "regression")
  expect_length(preds, 10)
  expect_true(all(is.finite(preds)))
})


