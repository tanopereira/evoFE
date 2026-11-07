test_that("Item 1: Regression competition metrics (rmsle, r2, mape, smape, qwk) are exact and oriented", {
  y_true <- c(0, 1, 3, 7)
  y_perf <- c(0, 1, 3, 7)
  y_good <- c(0.1, 1.1, 2.8, 6.5)
  y_bad  <- c(2.0, 4.0, 0.5, 1.0)

  # RMSLE: -sqrt(mean((log1p(y) - log1p(yhat))^2))
  expected_rmsle <- -sqrt(mean((log1p(y_true) - log1p(y_good))^2))
  expect_equal(evoFE:::compute_metric(y_true, y_perf, "regression", "rmsle"), 0.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_true, y_good, "regression", "rmsle"), expected_rmsle, tolerance = 1e-12)
  expect_gt(
    evoFE:::compute_metric(y_true, y_good, "regression", "rmsle"),
    evoFE:::compute_metric(y_true, y_bad, "regression", "rmsle")
  )

  # R2: 1 - SS_res / SS_tot (natural higher-is-better, ceiling 1.0)
  expected_r2 <- 1.0 - sum((y_true - y_good)^2) / sum((y_true - mean(y_true))^2)
  expect_equal(evoFE:::compute_metric(y_true, y_perf, "regression", "r2"), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_true, y_good, "regression", "rsq"), expected_r2, tolerance = 1e-12)
  expect_gt(
    evoFE:::compute_metric(y_true, y_good, "regression", "r2"),
    evoFE:::compute_metric(y_true, y_bad, "regression", "r2")
  )

  # MAPE & SMAPE (negative-loss orientation, ceiling 0.0, safe at zero)
  y_pos <- c(2, 4, 8, 10)
  p_pos <- c(2.2, 3.6, 8.8, 9.0) # 10% relative error on every element
  expect_equal(evoFE:::compute_metric(y_pos, p_pos, "regression", "mape"), -0.10, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_true, y_perf, "regression", "mape"), 0.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_true, y_perf, "regression", "smape"), 0.0, tolerance = 1e-12)
  expect_true(is.finite(evoFE:::compute_metric(c(0, 0, 5), c(0, 1, 5), "regression", "smape")))

  # QWK on regression ordinal ratings (0..3)
  y_ord <- c(0, 1, 2, 3, 0, 1, 2, 3)
  expect_equal(evoFE:::compute_metric(y_ord, y_ord, "regression", "qwk"), 1.0, tolerance = 1e-12)
  # Off-by-1 predictions beat off-by-3 predictions under quadratic weights
  p_off1 <- c(1, 1, 2, 2, 0, 1, 2, 3)
  p_off3 <- c(3, 1, 2, 0, 0, 1, 2, 3)
  expect_gt(
    evoFE:::compute_metric(y_ord, p_off1, "regression", "qwk"),
    evoFE:::compute_metric(y_ord, p_off3, "regression", "qwk")
  )

  # Smearing & ceiling contracts
  expect_true(evoFE:::metric_uses_smearing("rmse", "regression"))
  expect_true(evoFE:::metric_uses_smearing("r2", "regression"))
  expect_false(evoFE:::metric_uses_smearing("rmsle", "regression"))
  expect_false(evoFE:::metric_uses_smearing("mape", "regression"))
  expect_false(evoFE:::metric_uses_smearing("smape", "regression"))
  expect_false(evoFE:::metric_uses_smearing("qwk", "regression"))

  expect_equal(evoFE:::metric_ideal_ceiling("r2", "regression"), 1.0)
  expect_equal(evoFE:::metric_ideal_ceiling("qwk", "regression"), 1.0)
  expect_equal(evoFE:::metric_ideal_ceiling("rmsle", "regression"), 0.0)
  expect_equal(evoFE:::metric_ideal_ceiling("smape", "regression"), 0.0)
})

test_that("Item 1: Binary classification competition metrics (aucpr, accuracy, qwk) & threshold calibration", {
  y_bin <- c(0L, 0L, 0L, 0L, 1L, 1L)
  p_perf <- c(0.05, 0.10, 0.15, 0.20, 0.85, 0.95)
  p_tied <- rep(0.5, length(y_bin))

  expect_equal(evoFE:::compute_metric(y_bin, p_perf, "classification", "aucpr"), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_bin, p_perf, "classification", "pr_auc"), 1.0, tolerance = 1e-12)
  # All-tied predictions yield exact positive prevalence (2/6) regardless of row order
  expect_equal(evoFE:::compute_metric(y_bin, p_tied, "classification", "average_precision"), 2 / 6, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(rev(y_bin), p_tied, "classification", "aucpr"), 2 / 6, tolerance = 1e-12)
  expect_true(evoFE:::metric_is_rank_invariant("aucpr", "classification"))

  # Accuracy & threshold optimization
  p_shifted <- c(0.10, 0.15, 0.20, 0.25, 0.35, 0.40) # separable at tau = 0.30, all 0 at tau = 0.50
  expect_equal(evoFE:::compute_metric(y_bin, p_shifted, "classification", "accuracy"), 4 / 6, tolerance = 1e-12)
  opt_acc <- optimize_threshold(y_bin, p_shifted, metric = "accuracy")
  expect_equal(opt_acc$best_score, 1.0, tolerance = 1e-12)
  expect_true(opt_acc$threshold > 0.25 && opt_acc$threshold < 0.35)
  cal_acc <- evoFE:::fit_post_calibrator(y_bin, p_shifted, task = "classification", metric = "acc")
  expect_equal(cal_acc$type, "threshold_shift")
  p_cal <- evoFE:::apply_post_calibrator(p_shifted, cal_acc)
  expect_equal(evoFE:::compute_metric(y_bin, p_cal, "classification", "acc"), 1.0, tolerance = 1e-12)

  # Binary QWK equals Cohen's kappa at threshold 0.5
  expect_equal(evoFE:::compute_metric(y_bin, p_perf, "classification", "qwk"), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(c(0L, 0L, 1L, 1L), c(0.1, 0.9, 0.1, 0.9), "classification", "qwk"), 0.0, tolerance = 1e-12)
})

test_that("Item 1: Multiclass competition metrics (macro_f1, accuracy, qwk)", {
  y_mc <- c(0L, 0L, 1L, 1L, 2L, 2L)
  P_perf <- rbind(
    c(0.90, 0.05, 0.05),
    c(0.85, 0.10, 0.05),
    c(0.05, 0.90, 0.05),
    c(0.10, 0.80, 0.10),
    c(0.05, 0.05, 0.90),
    c(0.05, 0.10, 0.85)
  )
  expect_equal(evoFE:::compute_metric(y_mc, P_perf, "multiclass", "macro_f1", num_class = 3L), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_mc, P_perf, "multiclass", "f1_macro", num_class = 3L), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_mc, P_perf, "multiclass", "accuracy", num_class = 3L), 1.0, tolerance = 1e-12)
  expect_equal(evoFE:::compute_metric(y_mc, P_perf, "multiclass", "qwk", num_class = 3L), 1.0, tolerance = 1e-12)

  # Ambiguous bimodal row (0.48, 0.02, 0.50) on true class 0:
  # Expected value 0*0.48 + 1*0.02 + 2*0.50 = 1.02 -> rounds to 1 (error 1), whereas argmax=2 has error 2
  P_near <- P_perf
  P_near[1, ] <- c(0.48, 0.02, 0.50)
  P_far <- P_perf
  P_far[1, ] <- c(0.01, 0.01, 0.98)
  expect_gt(
    evoFE:::compute_metric(y_mc, P_near, "multiclass", "qwk", num_class = 3L),
    evoFE:::compute_metric(y_mc, P_far, "multiclass", "qwk", num_class = 3L)
  )
})

test_that("Item 1: RealMLP C++ early stopping supports new competition metrics", {
  set.seed(12)
  X_tr <- matrix(rnorm(120), nrow = 40, ncol = 3)
  colnames(X_tr) <- c("x1", "x2", "x3")
  X_va <- matrix(rnorm(60), nrow = 20, ncol = 3)
  colnames(X_va) <- c("x1", "x2", "x3")

  # Regression: rmsle & qwk
  y_reg_tr <- abs(X_tr[, 1]) * 3 + 1
  y_reg_va <- abs(X_va[, 1]) * 3 + 1
  m_rmsle <- train_model(
    X_tr, y_reg_tr, X_va, y_reg_va, task = "regression", evaluator = "realmlp",
    metric = "rmsle", epochs = 6L, early_stopping_rounds = 3L, hidden_dim = 16L, threads = 1L, verbose = FALSE
  )
  expect_length(m_rmsle$predictions, 20L)
  expect_true(all(is.finite(m_rmsle$predictions)))

  # Binary: aucpr
  y_bin_tr <- as.integer(X_tr[, 1] + 0.5 * X_tr[, 2] > 0)
  y_bin_va <- as.integer(X_va[, 1] + 0.5 * X_va[, 2] > 0)
  m_aucpr <- train_model(
    X_tr, y_bin_tr, X_va, y_bin_va, task = "classification", evaluator = "realmlp",
    metric = "aucpr", epochs = 6L, early_stopping_rounds = 3L, hidden_dim = 16L, threads = 1L, verbose = FALSE
  )
  expect_length(m_aucpr$predictions, 20L)
  expect_true(all(is.finite(m_aucpr$predictions)))

  # Multiclass: macro_f1 & qwk
  y_mc_tr <- as.integer(cut(X_tr[, 1], breaks = c(-Inf, -0.4, 0.4, Inf), labels = FALSE)) - 1L
  y_mc_va <- as.integer(cut(X_va[, 1], breaks = c(-Inf, -0.4, 0.4, Inf), labels = FALSE)) - 1L
  m_mf1 <- train_model(
    X_tr, y_mc_tr, X_va, y_mc_va, task = "multiclass", num_class = 3L, evaluator = "realmlp",
    metric = "macro_f1", epochs = 6L, early_stopping_rounds = 3L, hidden_dim = 16L, threads = 1L, verbose = FALSE
  )
  expect_equal(dim(m_mf1$predictions), c(20L, 3L))
  m_qwk <- train_model(
    X_tr, y_mc_tr, X_va, y_mc_va, task = "multiclass", num_class = 3L, evaluator = "realmlp",
    metric = "qwk", epochs = 6L, early_stopping_rounds = 3L, hidden_dim = 16L, threads = 1L, verbose = FALSE
  )
  expect_equal(dim(m_qwk$predictions), c(20L, 3L))
})

test_that("Item 4: confirm_top_k runs deterministic 2nd-split confirmation and handles group & time CV", {
  set.seed(77)
  n <- 90L
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    grp = rep(paste0("g", seq_len(15L)), each = 6L),
    tm = seq_len(n)
  )
  df$y <- as.integer(df$x1 * df$x2 + 0.5 * df$x1 > 0)

  # Group fold randomization check
  f1 <- with_seed(1L, evoFE:::.build_cv_folds(df, 3L, strategy = "group", group_col = "grp", target_col = "y", task = "classification", shuffle_groups = TRUE))
  f2 <- with_seed(2L, evoFE:::.build_cv_folds(df, 3L, strategy = "group", group_col = "grp", target_col = "y", task = "classification", shuffle_groups = TRUE))
  expect_false(identical(f1, f2))
  # Group integrity preserved in both
  expect_true(all(tapply(f1, df$grp, function(z) length(unique(z))) == 1L))
  expect_true(all(tapply(f2, df$grp, function(z) length(unique(z))) == 1L))

  rec1 <- evolve_features(
    data = df[, c("x1", "x2", "y")], target_col = "y", task = "classification",
    generations = 2L, pop_size = 4L, cv_folds = 3L, evaluation_strategy = "cv",
    evaluator = "lightgbm", nrounds = 12L, threads = 1L,
    allowed_transformers = c("multiply", "add", "log"),
    confirm_top_k = 3L, metric = "aucpr", seed = 202L, verbose = FALSE
  )
  rec2 <- evolve_features(
    data = df[, c("x1", "x2", "y")], target_col = "y", task = "classification",
    generations = 2L, pop_size = 4L, cv_folds = 3L, evaluation_strategy = "cv",
    evaluator = "lightgbm", nrounds = 12L, threads = 1L,
    allowed_transformers = c("multiply", "add", "log"),
    confirm_top_k = 3L, metric = "aucpr", seed = 202L, verbose = FALSE
  )
  expect_true(is.finite(rec1$confirm_fitness))
  expect_true(is.finite(rec1$mean_cv_fitness))
  expect_equal(rec1$mean_cv_fitness, 0.5 * (rec1$best_individual$fitness + rec1$confirm_fitness), tolerance = 1e-12)
  expect_equal(rec1$mean_cv_fitness, rec2$mean_cv_fitness, tolerance = 1e-12)
  expect_length(rec1$oof_preds, n)

  # Time CV is deterministic, so confirm_top_k gracefully skips without error
  rec_time <- evolve_features(
    data = df[, c("x1", "x2", "tm", "y")], target_col = "y", task = "classification",
    generations = 1L, pop_size = 3L, cv_folds = 3L, evaluation_strategy = "cv",
    cv_strategy = "time", time_col = "tm",
    evaluator = "lightgbm", nrounds = 10L, threads = 1L,
    allowed_transformers = c("multiply", "add"),
    confirm_top_k = 2L, seed = 99L, verbose = FALSE
  )
  expect_null(rec_time$confirm_fitness)
})

test_that("Item 5: frac_part extracts exact fractional part and preserves NA", {
  expect_true("frac_part" %in% names(evo_transformers))
  expect_true("frac_part" %in% evoFE:::resolve_allowed_transformers("basic"))
  expect_true("frac_part" %in% evoFE:::resolve_allowed_transformers("robust"))

  dt <- data.table::data.table(
    x = c(-2.75, 0.0, 3.125, NA_real_, Inf),
    y = c(1, 0, 1, 0, 1)
  )
  g <- create_gene("frac_part", "x")
  out <- evo_transformers$frac_part$apply_func(dt, g, NULL)
  expect_equal(out[1:3], c(0.75, 0.0, 0.125), tolerance = 1e-12)
  expect_true(is.na(out[4]))
  expect_true(is.na(out[5]))
})

test_that("Item 5: ridge_oof_encode matches brute-force LOO to machine precision and avoids noise memorization", {
  expect_true("ridge_oof_encode" %in% names(evo_transformers))
  expect_true(is_supervised_transformer("ridge_oof_encode"))
  expect_true("ridge_oof_encode" %in% evoFE:::resolve_allowed_transformers("basic"))
  expect_true("ridge_oof_encode" %in% evoFE:::resolve_allowed_transformers("robust"))
  expect_false("ridge_oof_encode" %in% evoFE:::filter_transformers_for_task(names(evo_transformers), "multiclass"))

  set.seed(314)
  n <- 80L
  dt <- data.table::data.table(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n)
  )
  dt[, y := 1.5 * x1 - 1.2 * x2 + 0.9 * (x1^2 + x2^2) + rnorm(n, sd = 0.25)]

  for (deg in c(1L, 2L)) {
    g <- create_gene("ridge_oof_encode", c("x1", "x2", "x3"))
    g$params$degree <- deg
    g$params$lambda <- 1.0
    g$output_col <- evo_transformers$ridge_oof_encode$name_generator(g)

    st_full <- evo_transformers$ridge_oof_encode$fit_func(dt, g, "y")
    expect_true(st_full$valid)
    expect_length(st_full$loo_train, n)

    # Verify Sherman-Morrison LOO matches brute-force leave-one-out slope refit on sample rows
    Z <- scale(as.matrix(dt[, .(x1, x2, x3)]), center = st_full$mu, scale = st_full$sigma)
    Z <- pmin(pmax(Z, -10.0), 10.0)
    B <- if (deg == 2L) cbind(Z, (Z * Z - 1.0) * sqrt(0.5)) else Z
    Phi <- sweep(B, 2L, st_full$phi_mean, "-")
    P <- diag(1.0, ncol(Phi))
    yc <- dt$y - st_full$global_mean
    for (i in c(1L, 17L, 42L, 80L)) {
      Phi_mi <- Phi[-i, , drop = FALSE]
      beta_mi <- solve(crossprod(Phi_mi) + P, crossprod(Phi_mi, yc[-i]))
      pred_mi <- st_full$global_mean + sum(Phi[i, ] * beta_mi)
      pred_mi <- min(max(pred_mi, st_full$clip_bounds[1]), st_full$clip_bounds[2])
      expect_equal(st_full$loo_train[i], pred_mi, tolerance = 1e-10)
    }

    if (deg == 2L) {
      expect_gt(cor(st_full$loo_train, dt$y), 0.90)
    }
    expect_match(gene_to_formula(g), "^ridge_oof_d[12]_l")
  }

  # Zero target memorization on pure independent noise
  dt_noise <- data.table::data.table(x1 = rnorm(200), x2 = rnorm(200), y = rnorm(200))
  gn <- create_gene("ridge_oof_encode", c("x1", "x2"))
  gn$params$degree <- 2L
  gn$params$lambda <- 10.0
  res_n <- apply_gene(gn, data.table::copy(dt_noise), NULL, target_col = "y", global_unsupervised = FALSE)
  expect_lt(abs(cor(res_n$train[[gn$output_col]], dt_noise$y)), 0.20)

  # Compact state after strip_loo_state
  ind_tmp <- create_individual(genes = list(res_n$gene), numeric_cols = c("x1", "x2"))
  ind_stripped <- evoFE:::strip_loo_state(ind_tmp)
  expect_null(ind_stripped$genes[[1]]$state$loo_train)
  preds_te <- evo_transformers$ridge_oof_encode$apply_func(dt_noise[1:10], gn, ind_stripped$genes[[1]]$state)
  expect_length(preds_te, 10L)
  expect_true(all(is.finite(preds_te)))
})
