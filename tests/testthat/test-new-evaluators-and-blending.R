# Tests for Wave 1-2 Python-free evaluators, cat_schema, native offsets,
# fast lm + rcpp_impute_and_col_stats, and Stage 5 logit stacking / residual blender.

.pred_ev <- function(model, x_new, task, evaluator, ...) {
  get_evaluator(evaluator)$predict_func(model, x_new, task = task, ...)
}

test_that("tabm handles regression, binary, multiclass, mixed cat_schema + NAs, tiny N=8, and MBO/TabArena", {
  set.seed(101)
  n_tr <- 72L
  n_va <- 24L
  n_te <- 16L

  make_mixed <- function(n) {
    df <- data.frame(
      num1 = stats::rnorm(n),
      num2 = stats::runif(n, -2, 2),
      cat1 = factor(sample(c("A", "B", "C"), n, replace = TRUE)),
      cat2 = sample(c("X", "Y"), n, replace = TRUE),
      stringsAsFactors = FALSE
    )
    df$num1[sample.int(n, max(1L, n %/% 10L))] <- NA_real_
    df$cat1[sample.int(n, max(1L, n %/% 12L))] <- NA
    df
  }

  x_tr <- make_mixed(n_tr)
  x_va <- make_mixed(n_va)
  x_te <- make_mixed(n_te)
  # Inject unseen factor level in test
  x_te$cat1 <- factor(c("UNSEEN_LVL", as.character(x_te$cat1[-1L])),
                      levels = c(levels(x_tr$cat1), "UNSEEN_LVL"))

  # Trait check: scale_iterations_with_data == FALSE
  expect_false(evoFE:::.evaluator_scales_iterations("tabm"))
  expect_false(evoFE:::.evaluator_scales_iterations("tabm_mbo"))
  expect_false(evoFE:::.evaluator_scales_iterations("tabm_tabarena"))

  # 1. Regression
  y_tr_reg <- 2 * ifelse(is.na(x_tr$num1), 0, x_tr$num1) - x_tr$num2 + stats::rnorm(n_tr, sd = 0.2)
  y_va_reg <- 2 * ifelse(is.na(x_va$num1), 0, x_va$num1) - x_va$num2 + stats::rnorm(n_va, sd = 0.2)
  fit_reg <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "tabm",
    epochs = 5L, k_ensemble = 4L, d_block = 16L, n_blocks = 1L, n_bins = 8L,
    threads = 1L, seed = 42L
  )
  expect_length(fit_reg$predictions, n_va)
  expect_true(all(is.finite(fit_reg$predictions)))
  p_reg <- .pred_ev(fit_reg$model, x_te, task = "regression", evaluator = "tabm")
  expect_length(p_reg, n_te)
  expect_true(all(is.finite(p_reg)))

  # 2. Binary classification
  y_tr_bin <- as.integer(ifelse(is.na(x_tr$num1), 0, x_tr$num1) + x_tr$num2 > 0)
  y_va_bin <- as.integer(ifelse(is.na(x_va$num1), 0, x_va$num1) + x_va$num2 > 0)
  fit_bin <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "tabm",
    epochs = 5L, k_ensemble = 4L, d_block = 16L, n_blocks = 1L, n_bins = 8L,
    threads = 1L, seed = 42L
  )
  expect_length(fit_bin$predictions, n_va)
  expect_true(all(is.finite(fit_bin$predictions) & fit_bin$predictions >= 0 & fit_bin$predictions <= 1))
  p_bin <- .pred_ev(fit_bin$model, x_te, task = "classification", evaluator = "tabm")
  expect_length(p_bin, n_te)
  expect_true(all(is.finite(p_bin) & p_bin >= 0 & p_bin <= 1))

  # 3. Multiclass (character & factor target)
  cls_names <- c("alpha", "beta", "gamma")
  y_tr_mc_chr <- sample(cls_names, n_tr, replace = TRUE)
  y_va_mc_chr <- sample(cls_names, n_va, replace = TRUE)
  fit_mc <- train_model(
    x_tr, y_tr_mc_chr, x_val = x_va, y_val = y_va_mc_chr,
    task = "multiclass", num_class = 3L, evaluator = "tabm",
    epochs = 5L, k_ensemble = 4L, d_block = 16L, n_blocks = 1L, n_bins = 8L,
    threads = 1L, seed = 42L
  )
  expect_true(is.matrix(fit_mc$predictions))
  expect_equal(dim(fit_mc$predictions), c(n_va, 3L))
  expect_equal(sort(colnames(fit_mc$predictions)), cls_names)
  expect_equal(rowSums(fit_mc$predictions), rep(1.0, n_va), tolerance = 1e-4)

  y_tr_mc_fac <- factor(y_tr_mc_chr, levels = cls_names)
  fit_mc_fac <- train_model(
    x_tr, y_tr_mc_fac, x_val = NULL,
    task = "multiclass", num_class = 3L, evaluator = "tabm",
    epochs = 5L, k_ensemble = 4L, d_block = 16L, n_blocks = 1L, n_bins = 8L,
    threads = 1L, seed = 42L
  )
  p_mc <- .pred_ev(fit_mc_fac$model, x_te, task = "multiclass", evaluator = "tabm")
  expect_true(is.matrix(p_mc))
  expect_equal(dim(p_mc), c(n_te, 3L))
  expect_equal(colnames(p_mc), cls_names)
  expect_equal(rowSums(p_mc), rep(1.0, n_te), tolerance = 1e-4)

  # 4. Tiny N = 8 edge case
  x_tiny <- x_tr[1:8, , drop = FALSE]
  y_tiny <- y_tr_reg[1:8]
  fit_tiny <- train_model(
    x_tiny, y_tiny, x_val = x_tiny[1:4, , drop = FALSE], y_val = y_tiny[1:4],
    task = "regression", evaluator = "tabm",
    epochs = 5L, k_ensemble = 4L, d_block = 16L, n_blocks = 1L, n_bins = 8L,
    threads = 1L, seed = 42L
  )
  p_tiny <- .pred_ev(fit_tiny$model, x_tiny, task = "regression", evaluator = "tabm")
  expect_length(p_tiny, 8L)
  expect_true(all(is.finite(p_tiny)))

  # 5. tabm_mbo and tabm_tabarena
  skip_if_not_installed("mlr3mbo")
  skip_if_not_installed("paradox")
  skip_if_not_installed("bbotk")

  fit_mbo <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "tabm_mbo",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L,
    epochs = 5L, n_bins = 8L, threads = 1L, seed = 42L
  )
  expect_type(fit_mbo$best_params, "list")
  expect_true(all(is.finite(fit_mbo$predictions)))

  fit_ta <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "tabm_tabarena",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L,
    epochs = 5L, threads = 1L, seed = 42L
  )
  expect_type(fit_ta$best_params, "list")
  expect_true(all(is.finite(fit_ta$predictions)))
})

test_that("rff supports rff/nystrom, rbf/laplacian kernels, native offsets, and MBO/TabArena", {
  set.seed(102)
  n_tr <- 64L
  n_va <- 20L
  n_te <- 12L

  x_tr <- matrix(stats::rnorm(n_tr * 4L), ncol = 4L)
  x_va <- matrix(stats::rnorm(n_va * 4L), ncol = 4L)
  x_te <- matrix(stats::rnorm(n_te * 4L), ncol = 4L)
  colnames(x_tr) <- colnames(x_va) <- colnames(x_te) <- paste0("f", 1:4)

  expect_false(evoFE:::.evaluator_scales_iterations("rff"))
  expect_false(evoFE:::.evaluator_scales_iterations("rff_mbo"))
  expect_false(evoFE:::.evaluator_scales_iterations("rff_tabarena"))

  y_tr_reg <- sin(x_tr[, 1L]) + 0.5 * x_tr[, 2L]^2 + stats::rnorm(n_tr, sd = 0.1)
  y_va_reg <- sin(x_va[, 1L]) + 0.5 * x_va[, 2L]^2 + stats::rnorm(n_va, sd = 0.1)

  for (method in c("rff", "nystrom")) {
    for (kern in c("rbf", "laplacian")) {
      fit <- train_model(
        x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
        task = "regression", evaluator = "rff",
        approx_method = method, kernel = kern, n_features = 32L,
        threads = 1L, seed = 7L
      )
      expect_length(fit$predictions, n_va)
      expect_true(all(is.finite(fit$predictions)))
      p <- .pred_ev(fit$model, x_te, task = "regression", evaluator = "rff")
      expect_length(p, n_te)
      expect_true(all(is.finite(p)))
    }
  }

  # Native offset verification: regression, binary, multiclass
  off_tr_reg <- rep(50.0, n_tr)
  off_va_reg <- rep(50.0, n_va)
  off_te_reg <- rep(50.0, n_te)
  fit_off_reg <- train_model(
    x_tr, y_tr_reg + off_tr_reg, x_val = x_va, y_val = y_va_reg + off_va_reg,
    task = "regression", evaluator = "rff", n_features = 32L,
    train_offset = off_tr_reg, offset_val = off_va_reg,
    threads = 1L, seed = 7L
  )
  p_off_reg <- .pred_ev(
    fit_off_reg$model, x_te, task = "regression", evaluator = "rff",
    newoffset = off_te_reg
  )
  p_zero_reg <- .pred_ev(
    fit_off_reg$model, x_te, task = "regression", evaluator = "rff",
    newoffset = rep(0.0, n_te)
  )
  expect_gt(mean(p_off_reg) - mean(p_zero_reg), 40.0)

  # Binary classification with offset
  y_tr_bin <- as.integer(x_tr[, 1L] > 0)
  y_va_bin <- as.integer(x_va[, 1L] > 0)
  fit_off_bin <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "rff", n_features = 32L,
    train_offset = rep(0.0, n_tr), offset_val = rep(2.0, n_va),
    threads = 1L, seed = 7L
  )
  expect_true(all(is.finite(fit_off_bin$predictions) &
                    fit_off_bin$predictions > 0 & fit_off_bin$predictions < 1))
  p_hi <- .pred_ev(
    fit_off_bin$model, x_te, task = "classification", evaluator = "rff",
    newoffset = rep(5.0, n_te)
  )
  p_lo <- .pred_ev(
    fit_off_bin$model, x_te, task = "classification", evaluator = "rff",
    newoffset = rep(-5.0, n_te)
  )
  expect_true(all(p_hi > p_lo))

  # Multiclass with offset
  y_tr_mc <- factor(rep(c("c0", "c1", "c2"), length.out = n_tr))
  y_va_mc <- factor(rep(c("c0", "c1", "c2"), length.out = n_va))
  off_tr_mc <- matrix(0.0, nrow = n_tr, ncol = 3L)
  off_va_mc <- matrix(0.0, nrow = n_va, ncol = 3L)
  off_te_mc <- matrix(0.0, nrow = n_te, ncol = 3L)
  off_te_mc[, 2L] <- 6.0
  fit_off_mc <- train_model(
    x_tr, y_tr_mc, x_val = x_va, y_val = y_va_mc,
    task = "multiclass", num_class = 3L, evaluator = "rff", n_features = 32L,
    train_offset = off_tr_mc, offset_val = off_va_mc,
    threads = 1L, seed = 7L
  )
  expect_equal(dim(fit_off_mc$predictions), c(n_va, 3L))
  p_off_mc <- .pred_ev(
    fit_off_mc$model, x_te, task = "multiclass", evaluator = "rff",
    newoffset = off_te_mc
  )
  expect_equal(dim(p_off_mc), c(n_te, 3L))
  expect_equal(rowSums(p_off_mc), rep(1.0, n_te), tolerance = 1e-4)
  expect_true(all(p_off_mc[, 2L] > 0.7))

  # rff_mbo and rff_tabarena
  skip_if_not_installed("mlr3mbo")
  skip_if_not_installed("paradox")
  skip_if_not_installed("bbotk")

  fit_mbo <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "rff_mbo",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L,
    n_features = 32L, threads = 1L, seed = 7L
  )
  expect_type(fit_mbo$best_params, "list")
  expect_true(all(is.finite(fit_mbo$predictions)))

  fit_ta <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "rff_tabarena",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L,
    threads = 1L, seed = 7L
  )
  expect_type(fit_ta$best_params, "list")
  expect_true(all(is.finite(fit_ta$predictions)))
})

test_that("spline_lm handles sparse B-splines + OHE, unseen levels/NAs, offsets, edge cases, and MBO/TabArena", {
  skip_if_not_installed("glmnet")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("splines")

  set.seed(103)
  n_tr <- 72L
  n_va <- 24L
  n_te <- 12L

  make_df <- function(n) {
    df <- data.frame(
      x1 = stats::rnorm(n),
      x2 = stats::runif(n, 0, 10),
      cat1 = factor(sample(c("L1", "L2", "L3"), n, replace = TRUE)),
      stringsAsFactors = FALSE
    )
    df$x1[sample.int(n, max(1L, n %/% 10L))] <- NA_real_
    df$cat1[1L] <- NA
    df
  }

  x_tr <- make_df(n_tr)
  x_va <- make_df(n_va)
  x_te <- make_df(n_te)
  x_te$cat1 <- factor(c("UNSEEN", as.character(x_te$cat1[-1L])),
                      levels = c(levels(x_tr$cat1), "UNSEEN"))

  expect_false(evoFE:::.evaluator_scales_iterations("spline_lm"))
  expect_false(evoFE:::.evaluator_scales_iterations("spline_lm_mbo"))
  expect_false(evoFE:::.evaluator_scales_iterations("spline_lm_tabarena"))

  # 1. Regression + offset
  y_tr_reg <- sin(ifelse(is.na(x_tr$x1), 0, x_tr$x1)) + 0.1 * x_tr$x2 + stats::rnorm(n_tr, sd = 0.2)
  y_va_reg <- sin(ifelse(is.na(x_va$x1), 0, x_va$x1)) + 0.1 * x_va$x2 + stats::rnorm(n_va, sd = 0.2)
  fit_reg <- train_model(
    x_tr, y_tr_reg + 20.0, x_val = x_va, y_val = y_va_reg + 20.0,
    task = "regression", evaluator = "spline_lm",
    train_offset = rep(20.0, n_tr), offset_val = rep(20.0, n_va),
    threads = 1L
  )
  expect_true(inherits(fit_reg$model$glmnet_model, "glmnet"))
  expect_length(fit_reg$predictions, n_va)
  p_reg_off <- .pred_ev(fit_reg$model, x_te, task = "regression", evaluator = "spline_lm",
                                 newoffset = rep(20.0, n_te))
  p_reg_zero <- .pred_ev(fit_reg$model, x_te, task = "regression", evaluator = "spline_lm",
                                  newoffset = rep(0.0, n_te))
  expect_equal(p_reg_off - p_reg_zero, rep(20.0, n_te), tolerance = 1e-5)

  # Single-row predict edge case
  p_one <- .pred_ev(fit_reg$model, x_te[1L, , drop = FALSE], task = "regression",
                             evaluator = "spline_lm", newoffset = 20.0)
  expect_length(p_one, 1L)
  expect_true(is.finite(p_one))

  # Single-col edge case
  x_1c_tr <- x_tr[, "x1", drop = FALSE]
  x_1c_va <- x_va[, "x1", drop = FALSE]
  fit_1c <- train_model(
    x_1c_tr, y_tr_reg, x_val = x_1c_va, y_val = y_va_reg,
    task = "regression", evaluator = "spline_lm", threads = 1L
  )
  p_1c <- .pred_ev(fit_1c$model, x_te[, "x1", drop = FALSE],
                            task = "regression", evaluator = "spline_lm")
  expect_length(p_1c, n_te)
  expect_true(all(is.finite(p_1c)))

  # 2. Binary classification ("logit-started LR" offset)
  y_tr_bin <- as.integer(ifelse(is.na(x_tr$x1), 0, x_tr$x1) > 0)
  y_va_bin <- as.integer(ifelse(is.na(x_va$x1), 0, x_va$x1) > 0)
  fit_bin <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "spline_lm",
    train_offset = rep(0.0, n_tr), offset_val = rep(0.0, n_va),
    threads = 1L
  )
  p_bin_hi <- .pred_ev(fit_bin$model, x_te, task = "classification", evaluator = "spline_lm",
                                newoffset = rep(5.0, n_te))
  p_bin_lo <- .pred_ev(fit_bin$model, x_te, task = "classification", evaluator = "spline_lm",
                                newoffset = rep(-5.0, n_te))
  expect_true(all(p_bin_hi > p_bin_lo))

  # 3. Multiclass + offset
  y_tr_mc <- factor(rep(c("A", "B", "C"), length.out = n_tr))
  y_va_mc <- factor(rep(c("A", "B", "C"), length.out = n_va))
  off_tr_mc <- matrix(0.0, nrow = n_tr, ncol = 3L)
  off_va_mc <- matrix(0.0, nrow = n_va, ncol = 3L)
  off_te_mc <- matrix(0.0, nrow = n_te, ncol = 3L)
  off_te_mc[, 3L] <- 6.0
  fit_mc <- train_model(
    x_tr, y_tr_mc, x_val = x_va, y_val = y_va_mc,
    task = "multiclass", num_class = 3L, evaluator = "spline_lm",
    train_offset = off_tr_mc, offset_val = off_va_mc, threads = 1L
  )
  p_mc <- .pred_ev(fit_mc$model, x_te, task = "multiclass", evaluator = "spline_lm",
                            newoffset = off_te_mc)
  expect_equal(dim(p_mc), c(n_te, 3L))
  expect_true(all(p_mc[, 3L] > 0.7))

  # 4. spline_lm_mbo and spline_lm_tabarena
  skip_if_not_installed("mlr3mbo")
  skip_if_not_installed("paradox")
  skip_if_not_installed("bbotk")

  fit_mbo <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "spline_lm_mbo",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L, threads = 1L
  )
  expect_type(fit_mbo$best_params, "list")
  expect_true(all(is.finite(fit_mbo$predictions)))

  fit_ta <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "spline_lm_tabarena",
    mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L, threads = 1L
  )
  expect_type(fit_ta$best_params, "list")
  expect_true(all(is.finite(fit_ta$predictions)))
})

test_that("lightgbm_rf and ranger handle NAs, dropped multiclass levels, traits, and MBO/TabArena", {
  skip_if_not_installed("lightgbm")
  skip_if_not_installed("ranger")

  set.seed(104)
  n_tr <- 72L
  n_va <- 24L
  n_te <- 12L

  make_df <- function(n) {
    df <- data.frame(
      x1 = stats::rnorm(n),
      x2 = stats::runif(n),
      cat1 = factor(sample(c("A", "B", "C"), n, replace = TRUE))
    )
    df$x1[sample.int(n, max(1L, n %/% 8L))] <- NA_real_
    df$cat1[1L] <- NA
    df
  }

  x_tr <- make_df(n_tr)
  x_va <- make_df(n_va)
  x_te <- make_df(n_te)

  # Trait checks: scale_iterations_with_data == FALSE
  for (ev in c("lightgbm_rf", "lightgbm_rf_mbo", "lightgbm_rf_tabarena",
               "ranger", "ranger_mbo", "ranger_tabarena")) {
    expect_false(evoFE:::.evaluator_scales_iterations(ev))
  }

  # lightgbm_rf works without aborting (bagging_freq >= 1L) across reg/bin/mc
  y_tr_reg <- x_tr$x2 * 3 + stats::rnorm(n_tr, sd = 0.2)
  y_va_reg <- x_va$x2 * 3 + stats::rnorm(n_va, sd = 0.2)
  fit_lgb_rf <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "lightgbm_rf", nrounds = 15L, threads = 1L
  )
  expect_length(fit_lgb_rf$predictions, n_va)
  expect_true(all(is.finite(fit_lgb_rf$predictions)))

  y_tr_bin <- as.integer(x_tr$x2 > 0.5)
  y_va_bin <- as.integer(x_va$x2 > 0.5)
  fit_lgb_rf_bin <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "lightgbm_rf", nrounds = 15L, threads = 1L
  )
  expect_true(all(is.finite(fit_lgb_rf_bin$predictions) &
                    fit_lgb_rf_bin$predictions >= 0 & fit_lgb_rf_bin$predictions <= 1))

  # ranger handles NAs + save.memory = FALSE + dropped multiclass levels in extratrees
  y_tr_dropped <- factor(rep(c("c1", "c2"), length.out = n_tr), levels = c("c1", "c2", "c3_empty"))
  y_va_dropped <- factor(rep(c("c1", "c2"), length.out = n_va), levels = c("c1", "c2", "c3_empty"))
  fit_rng_mc <- train_model(
    x_tr, y_tr_dropped, x_val = x_va, y_val = y_va_dropped,
    task = "multiclass", num_class = 3L, evaluator = "ranger",
    nrounds = 15L, splitrule = "extratrees", threads = 1L
  )
  expect_false(isTRUE(fit_rng_mc$model$ranger_fit$save.memory))
  expect_equal(dim(fit_rng_mc$predictions), c(n_va, 3L))
  expect_equal(colnames(fit_rng_mc$predictions), c("c1", "c2", "c3_empty"))
  expect_true(all(fit_rng_mc$predictions[, "c3_empty"] == 0))
  p_rng_mc <- .pred_ev(fit_rng_mc$model, x_te, task = "multiclass", evaluator = "ranger")
  expect_equal(dim(p_rng_mc), c(n_te, 3L))
  expect_true(all(is.finite(p_rng_mc)))

  # MBO & TabArena tuners for both
  skip_if_not_installed("mlr3mbo")
  skip_if_not_installed("paradox")
  skip_if_not_installed("bbotk")

  for (ev in c("lightgbm_rf_mbo", "lightgbm_rf_tabarena", "ranger_mbo", "ranger_tabarena")) {
    fit_t <- train_model(
      x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
      task = "regression", evaluator = ev,
      mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L,
      nrounds = 15L, threads = 1L
    )
    expect_type(fit_t$best_params, "list")
    expect_true(all(is.finite(fit_t$predictions)))
  }
})

test_that("rcpp_impute_and_col_stats and fast lm work with single-path/lambda, offsets, and MBO/TabArena", {
  skip_if_not_installed("glmnet")

  # 1. rcpp_impute_and_col_stats: zero-copy finite behavior + NA imputation + sample SD fallback
  mat_clean <- matrix(c(1, 2, 3, 4, 5, 5, 5, 5), ncol = 2L)
  res_clean <- evoFE:::rcpp_impute_and_col_stats(mat_clean, compute_sd = TRUE, threads = 1L)
  expect_identical(res_clean$mat, mat_clean)
  expect_equal(res_clean$col_means, c(2.5, 5.0))
  # Column 2 is constant -> SD falls back to 1.0
  expect_equal(res_clean$col_sds, c(stats::sd(1:4), 1.0), tolerance = 1e-12)

  mat_na <- matrix(c(1, NA_real_, 3, Inf, 10, 20, NA_real_, 40), ncol = 2L)
  res_na <- evoFE:::rcpp_impute_and_col_stats(mat_na, compute_sd = TRUE, threads = 1L)
  expect_false(any(!is.finite(res_na$mat)))
  expect_equal(res_na$col_means, c(2.0, 70.0 / 3.0), tolerance = 1e-12)
  expect_equal(res_na$mat[c(2L, 4L), 1L], c(2.0, 2.0), tolerance = 1e-12)
  expect_equal(res_na$mat[3L, 2L], 70.0 / 3.0, tolerance = 1e-12)
  expect_equal(res_na$col_sds[1L], stats::sd(res_na$mat[, 1L]), tolerance = 1e-12)

  # Single finite value in column -> sample SD fallback = 1.0
  mat_1fin <- matrix(c(7.0, NA_real_, NA_real_, NA_real_), ncol = 1L)
  res_1fin <- evoFE:::rcpp_impute_and_col_stats(mat_1fin, compute_sd = TRUE, threads = 1L)
  expect_equal(res_1fin$col_means, 7.0)
  expect_equal(res_1fin$col_sds, 1.0)

  # 2. Fast lm single-path with x_val (no cv.glmnet) and single-lambda
  set.seed(105)
  n_tr <- 60L
  n_va <- 20L
  n_te <- 10L
  x_tr <- matrix(stats::rnorm(n_tr * 4L), ncol = 4L)
  x_va <- matrix(stats::rnorm(n_va * 4L), ncol = 4L)
  x_te <- matrix(stats::rnorm(n_te * 4L), ncol = 4L)
  colnames(x_tr) <- colnames(x_va) <- colnames(x_te) <- paste0("v", 1:4)

  y_tr_reg <- 1.5 * x_tr[, 1L] - 0.8 * x_tr[, 2L] + stats::rnorm(n_tr, sd = 0.2)
  y_va_reg <- 1.5 * x_va[, 1L] - 0.8 * x_va[, 2L] + stats::rnorm(n_va, sd = 0.2)

  fit_path <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "lm", threads = 1L
  )
  expect_true(inherits(fit_path$model, "glmnet"))
  expect_false(inherits(fit_path$model, "cv.glmnet"))
  expect_length(fit_path$best_lambda, 1L)
  expect_true(is.finite(fit_path$best_lambda))

  fit_scalar <- train_model(
    x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
    task = "regression", evaluator = "lm", lambda = 0.05, threads = 1L
  )
  expect_equal(fit_scalar$best_lambda, 0.05)

  # Native offset in reg / binary / multiclass
  fit_lm_off_reg <- train_model(
    x_tr, y_tr_reg + 15.0, x_val = x_va, y_val = y_va_reg + 15.0,
    task = "regression", evaluator = "lm", lambda = 0.05,
    train_offset = rep(15.0, n_tr), offset_val = rep(15.0, n_va), threads = 1L
  )
  p_off <- .pred_ev(fit_lm_off_reg$model, x_te, task = "regression", evaluator = "lm",
                             newoffset = rep(15.0, n_te))
  p_zero <- .pred_ev(fit_lm_off_reg$model, x_te, task = "regression", evaluator = "lm",
                              newoffset = rep(0.0, n_te))
  expect_equal(p_off - p_zero, rep(15.0, n_te), tolerance = 1e-5)

  y_tr_bin <- as.integer(x_tr[, 1L] > 0)
  y_va_bin <- as.integer(x_va[, 1L] > 0)
  fit_lm_off_bin <- train_model(
    x_tr, y_tr_bin, x_val = x_va, y_val = y_va_bin,
    task = "classification", evaluator = "lm", lambda = 0.05,
    train_offset = rep(0.0, n_tr), offset_val = rep(0.0, n_va), threads = 1L
  )
  p_bin_hi <- .pred_ev(fit_lm_off_bin$model, x_te, task = "classification", evaluator = "lm",
                                newoffset = rep(5.0, n_te))
  p_bin_lo <- .pred_ev(fit_lm_off_bin$model, x_te, task = "classification", evaluator = "lm",
                                newoffset = rep(-5.0, n_te))
  expect_true(all(p_bin_hi > p_bin_lo))

  y_tr_mc <- factor(rep(c("A", "B", "C"), length.out = n_tr))
  y_va_mc <- factor(rep(c("A", "B", "C"), length.out = n_va))
  off_tr_mc <- matrix(0.0, nrow = n_tr, ncol = 3L)
  off_va_mc <- matrix(0.0, nrow = n_va, ncol = 3L)
  off_te_mc <- matrix(0.0, nrow = n_te, ncol = 3L)
  off_te_mc[, 1L] <- 6.0
  fit_lm_off_mc <- train_model(
    x_tr, y_tr_mc, x_val = x_va, y_val = y_va_mc,
    task = "multiclass", num_class = 3L, evaluator = "lm", lambda = 0.05,
    train_offset = off_tr_mc, offset_val = off_va_mc, threads = 1L
  )
  p_mc <- .pred_ev(fit_lm_off_mc$model, x_te, task = "multiclass", evaluator = "lm",
                            newoffset = off_te_mc)
  expect_equal(dim(p_mc), c(n_te, 3L))
  expect_true(all(p_mc[, 1L] > 0.7))

  # lm_mbo & lm_tabarena compatibility
  skip_if_not_installed("mlr3mbo")
  skip_if_not_installed("paradox")
  skip_if_not_installed("bbotk")

  for (ev in c("lm_mbo", "lm_tabarena")) {
    fit_m <- train_model(
      x_tr, y_tr_reg, x_val = x_va, y_val = y_va_reg,
      task = "regression", evaluator = ev,
      mbo_iters = 2L, mbo_init_design = 3L, mbo_folds = 2L, threads = 1L
    )
    expect_type(fit_m$best_params, "list")
    expect_true(all(is.finite(fit_m$predictions)))
  }
})

test_that("End-to-end cat_schema preserves logical flag_active and handles unseen levels across evaluators", {
  skip_if_not_installed("lightgbm")
  skip_if_not_installed("xgboost")
  skip_if_not_installed("ranger")
  skip_if_not_installed("glmnet")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("splines")

  set.seed(106)
  n_tr <- 60L
  n_te <- 15L

  make_mixed_df <- function(n) {
    data.frame(
      num1 = stats::rnorm(n),
      num2 = stats::runif(n, 1, 5),
      flag_active = sample(c(TRUE, FALSE), n, replace = TRUE),
      cat_grp = factor(sample(c("g1", "g2", "g3"), n, replace = TRUE)),
      stringsAsFactors = FALSE
    )
  }

  tr_df <- make_mixed_df(n_tr)
  tr_df$target <- as.integer(tr_df$num1 + ifelse(tr_df$flag_active, 0.8, -0.8) > 0)

  te_df <- make_mixed_df(n_te)
  te_df$cat_grp <- factor(c("UNSEEN_GRP", as.character(te_df$cat_grp[-1L])),
                          levels = c("g1", "g2", "g3", "UNSEEN_GRP"))
  te_df$num1[1L] <- NA_real_

  evs <- c("lightgbm", "xgboost", "tabm", "ranger", "spline_lm")
  if (requireNamespace("catboost", quietly = TRUE)) {
    evs <- c(evs, "catboost")
  }

  for (ev in evs) {
    rec <- suppressWarnings(evolve_features(
      data = tr_df,
      target_col = "target",
      task = "classification",
      evaluator = ev,
      generations = 1L,
      pop_size = 2L,
      cv_folds = 2L,
      bag_folds = TRUE,
      nrounds = 15L,
      epochs = 5L,
      threads = 1L,
      seed = 42L,
      verbose = FALSE
    ))
    expect_s3_class(rec, "evo_recipe")

    feat_te <- predict(rec, te_df)
    expect_s3_class(feat_te, "data.table")
    expect_equal(nrow(feat_te), n_te)
    # flag_active is preserved as logical in data.table and encoded as numeric {0, 1} in cat_schema
    if ("flag_active" %in% names(feat_te)) {
      expect_true(is.logical(feat_te$flag_active) || is.numeric(feat_te$flag_active))
    }
    san_tr <- evoFE:::.sanitize_feature_matrix(tr_df[, setdiff(names(tr_df), "target")])
    cs_tr <- attr(san_tr, "cat_schema", exact = TRUE)
    expect_false("flag_active" %in% names(cs_tr$levels))
    expect_true(all(san_tr[, "flag_active"] %in% c(0, 1)))

    p_bag <- predict_model(rec, te_df, bag_folds = TRUE)
    p_full <- predict_model(rec, te_df, bag_folds = FALSE)
    expect_length(p_bag, n_te)
    expect_length(p_full, n_te)
    expect_true(all(is.finite(p_bag) & p_bag >= 0 & p_bag <= 1))
    expect_true(all(is.finite(p_full) & p_full >= 0 & p_full <= 1))
  }
})

test_that("Explicit init_score / base_margin / baseline offsets work for lightgbm, xgboost, and catboost", {
  set.seed(107)
  n_tr <- 60L
  n_va <- 20L
  n_te <- 10L
  x_tr <- matrix(stats::rnorm(n_tr * 3L), ncol = 3L)
  x_va <- matrix(stats::rnorm(n_va * 3L), ncol = 3L)
  x_te <- matrix(stats::rnorm(n_te * 3L), ncol = 3L)
  colnames(x_tr) <- colnames(x_va) <- colnames(x_te) <- c("a", "b", "c")

  y_tr_reg <- x_tr[, 1L] + stats::rnorm(n_tr, sd = 0.2)
  y_va_reg <- x_va[, 1L] + stats::rnorm(n_va, sd = 0.2)
  y_tr_bin <- as.integer(x_tr[, 1L] > 0)
  y_va_bin <- as.integer(x_va[, 1L] > 0)
  y_tr_mc <- rep(0:2, length.out = n_tr)
  y_va_mc <- rep(0:2, length.out = n_va)

  gbdt_evs <- list(
    list(ev = "lightgbm", tr_arg = "init_score", va_arg = "val_init_score", te_arg = "newoffset"),
    list(ev = "xgboost",  tr_arg = "base_margin", va_arg = "val_base_margin", te_arg = "new_base_margin")
  )
  if (requireNamespace("catboost", quietly = TRUE)) {
    gbdt_evs <- c(gbdt_evs, list(
      list(ev = "catboost", tr_arg = "baseline", va_arg = "val_baseline", te_arg = "newoffset")
    ))
  }

  for (spec in gbdt_evs) {
    skip_if_not_installed(spec$ev)

    # Regression offset
    reg_args <- list(
      x_train = x_tr, y_train = y_tr_reg + 30.0,
      x_val = x_va, y_val = y_va_reg + 30.0,
      task = "regression", evaluator = spec$ev, nrounds = 15L, threads = 1L
    )
    reg_args[[spec$tr_arg]] <- rep(30.0, n_tr)
    reg_args[[spec$va_arg]] <- rep(30.0, n_va)
    fit_r <- do.call(train_model, reg_args)

    p_args_hi <- list(model = fit_r$model, x_new = x_te, task = "regression", evaluator = spec$ev)
    p_args_hi[[spec$te_arg]] <- rep(30.0, n_te)
    p_args_lo <- list(model = fit_r$model, x_new = x_te, task = "regression", evaluator = spec$ev)
    p_args_lo[[spec$te_arg]] <- rep(0.0, n_te)
    expect_equal(do.call(.pred_ev, p_args_hi) - do.call(.pred_ev, p_args_lo),
                 rep(30.0, n_te), tolerance = 1e-4)

    # Binary offset
    bin_args <- list(
      x_train = x_tr, y_train = y_tr_bin,
      x_val = x_va, y_val = y_va_bin,
      task = "classification", evaluator = spec$ev, nrounds = 15L, threads = 1L
    )
    bin_args[[spec$tr_arg]] <- rep(0.0, n_tr)
    bin_args[[spec$va_arg]] <- rep(0.0, n_va)
    fit_b <- do.call(train_model, bin_args)

    pb_hi <- list(model = fit_b$model, x_new = x_te, task = "classification", evaluator = spec$ev)
    pb_hi[[spec$te_arg]] <- rep(5.0, n_te)
    pb_lo <- list(model = fit_b$model, x_new = x_te, task = "classification", evaluator = spec$ev)
    pb_lo[[spec$te_arg]] <- rep(-5.0, n_te)
    expect_true(all(do.call(.pred_ev, pb_hi) > do.call(.pred_ev, pb_lo)))

    # Multiclass offset
    off_tr_mc <- matrix(0.0, nrow = n_tr, ncol = 3L)
    off_va_mc <- matrix(0.0, nrow = n_va, ncol = 3L)
    off_te_mc <- matrix(0.0, nrow = n_te, ncol = 3L)
    off_te_mc[, 2L] <- 6.0
    mc_args <- list(
      x_train = x_tr, y_train = y_tr_mc,
      x_val = x_va, y_val = y_va_mc,
      task = "multiclass", num_class = 3L, evaluator = spec$ev, nrounds = 15L, threads = 1L
    )
    mc_args[[spec$tr_arg]] <- off_tr_mc
    mc_args[[spec$va_arg]] <- off_va_mc
    fit_m <- do.call(train_model, mc_args)

    pm_args <- list(model = fit_m$model, x_new = x_te, task = "multiclass", evaluator = spec$ev)
    pm_args[[spec$te_arg]] <- off_te_mc
    p_m <- do.call(.pred_ev, pm_args)
    expect_equal(dim(p_m), c(n_te, 3L))
    expect_true(all(p_m[, 2L] > 0.7))
  }
})

test_that("Stage 5 logit-space stacking, residual blender, pipeline consistency, and zero Python imports", {
  skip_if_not_installed("lightgbm")
  skip_if_not_installed("glmnet")
  skip_if_not_installed("Matrix")
  skip_if_not_installed("splines")

  # 1. .resolve_stack_space
  expect_equal(evoFE:::.resolve_stack_space("prob", "classification"), "prob")
  expect_equal(evoFE:::.resolve_stack_space("logit", "classification"), "logit")
  expect_equal(evoFE:::.resolve_stack_space("auto", "classification", rank_average = FALSE), "logit")
  expect_equal(evoFE:::.resolve_stack_space("auto", "multiclass", rank_average = FALSE), "logit")
  expect_equal(evoFE:::.resolve_stack_space("auto", "classification", rank_average = TRUE), "prob")
  expect_equal(evoFE:::.resolve_stack_space("auto", "regression"), "prob")

  # 2. ensemble_islands with stack_space = "logit" and residual_blender = "lm" / "spline_lm"
  set.seed(108)
  n <- 80L
  df <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    x3 = stats::runif(n, -1, 1)
  )
  df$target <- as.integer(df$x1 + 0.5 * df$x2^2 + stats::rnorm(n, sd = 0.3) > 0)

  rec <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    evaluator = c("lightgbm", "spline_lm"),
    generations = 1L,
    pop_size = 2L,
    islands = 2L,
    cv_folds = 2L,
    nrounds = 15L,
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  )

  for (rb in c("lm", "spline_lm")) {
    ens <- ensemble_islands(
      rec, data = df, method = "stack",
      stack_space = "logit", residual_blender = rb,
      seed = 42L, verbose = FALSE
    )
    expect_s3_class(ens, "evo_ensemble")
    expect_true(all(ens$weights >= 0))
    expect_equal(sum(ens$weights), 1.0, tolerance = 1e-6)
    p <- predict_model(ens, df[1:10, ])
    expect_length(p, 10L)
    expect_true(all(is.finite(p) & p >= 0 & p <= 1))
  }

  # 3. run_competition_pipeline with Python-free evaluators + stage5_stack_space = "auto" + residual_blender = "lm"
  tr_df <- df[1:64, ]
  te_df <- df[65:80, c("x1", "x2", "x3")]
  pipe <- suppressWarnings(run_competition_pipeline(
    train = tr_df,
    test = te_df,
    target_col = "target",
    task = "classification",
    preset = "fast",
    preset_overrides = list(
      s2_gen = 1L, s2_pop = 2L,
      s2_nrounds = 15L, s4_lgb_nrounds = 15L
    ),
    evaluators = c("lightgbm", "lightgbm_rf", "spline_lm", "rff"),
    cv_folds = 2L,
    bag_folds = TRUE,
    stage5_stack_space = "auto",
    stage5_residual_blender = "lm",
    threads = 1L,
    seed = 42L,
    verbose = FALSE
  ))
  expect_true(is.list(pipe))
  expect_length(pipe$predictions, nrow(te_df))
  expect_true(all(is.finite(pipe$predictions)))
  p_pipe <- predict_model(pipe$final_model, pipe$test_mega)
  expect_equal(p_pipe, pipe$predictions, tolerance = 1e-8)

  # 4. Zero Python imports in DESCRIPTION
  desc_path <- testthat::test_path("..", "..", "DESCRIPTION")
  if (!file.exists(desc_path)) {
    desc_path <- system.file("DESCRIPTION", package = "evoFE")
  }
  expect_true(file.exists(desc_path))
  dcf <- read.dcf(desc_path)
  imp_str <- paste(c(dcf[, intersect(c("Imports", "Depends"), colnames(dcf))]), collapse = " ")
  expect_false(grepl("reticulate|python|tensorflow|torch", imp_str, ignore.case = TRUE))
})

test_that("TabM logs progress at verbose = TRUE or evoFE.tabm_verbose without leaking into GBDTs", {
  set.seed(404)
  n <- 40L
  x <- matrix(stats::rnorm(n * 4L), ncol = 4L)
  colnames(x) <- paste0("f", 1:4)
  y <- as.numeric(x[, 1] + 0.5 * x[, 2] + stats::rnorm(n, sd = 0.2))

  expect_silent(train_model(
    x_train = x[1:30, ], y_train = y[1:30],
    x_val = x[31:40, ], y_val = y[31:40],
    task = "regression", evaluator = "tabm",
    epochs = 6L, early_stopping_rounds = 0L, threads = 1L, verbose = FALSE
  ))

  out_true <- capture.output(train_model(
    x_train = x[1:30, ], y_train = y[1:30],
    x_val = x[31:40, ], y_val = y[31:40],
    task = "regression", evaluator = "tabm",
    epochs = 6L, early_stopping_rounds = 0L, threads = 1L, verbose = TRUE
  ))
  expect_true(any(grepl("\\[TabM C\\+\\+", out_true)))

  old_opt <- getOption("evoFE.tabm_verbose")
  options(evoFE.tabm_verbose = 2L)
  on.exit(options(evoFE.tabm_verbose = old_opt), add = TRUE)

  out_opt <- capture.output(train_model(
    x_train = x[1:30, ], y_train = y[1:30],
    x_val = x[31:40, ], y_val = y[31:40],
    task = "regression", evaluator = "tabm",
    epochs = 6L, early_stopping_rounds = 0L, threads = 1L, verbose = FALSE
  ))
  expect_true(any(grepl("\\[TabM C\\+\\+.*Epoch", out_opt)))

  # LightGBM must remain silent when only evoFE.tabm_verbose is enabled
  out_lgb <- capture.output(train_model(
    x_train = x[1:30, ], y_train = y[1:30],
    x_val = x[31:40, ], y_val = y[31:40],
    task = "regression", evaluator = "lightgbm",
    nrounds = 5L, early_stopping_rounds = 0L, threads = 1L, verbose = TRUE
  ))
  expect_false(any(grepl("\\[LightGBM\\]", out_lgb)))
  options(evoFE.tabm_verbose = old_opt)

  # run_competition_pipeline accepts per-evaluator verbose override in evaluator_args
  df_tr <- data.frame(x, y = y)[1:30, ]
  df_te <- data.frame(x)[31:40, , drop = FALSE]
  expect_no_error(suppressMessages(run_competition_pipeline(
    train = df_tr,
    test = df_te,
    target_col = "y",
    task = "regression",
    preset = "fast",
    preset_overrides = list(s2_gen = 1L, s2_pop = 2L, s2_mlp_gen = 1L, s2_mlp_pop = 2L,
                            s2_mlp_epochs = 4L, s4_mlp_epochs = 4L),
    evaluators = c("tabm"),
    evaluator_args = list(tabm = list(tabm_verbose = 0L, verbose = FALSE)),
    cv_folds = 2L,
    threads = 1L,
    verbose = FALSE
  )))
})
