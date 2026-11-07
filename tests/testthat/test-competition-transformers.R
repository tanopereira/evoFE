testthat::test_that("knn_target_encode works on 1D, 2D, and 3D numeric columns without leakage", {
  set.seed(42)
  n <- 30
  df <- data.frame(
    x1 = rnorm(n),
    x2 = rnorm(n),
    x3 = rnorm(n),
    y = as.numeric(rnorm(n) > 0)
  )

  # Test 1D column
  g1 <- create_gene("knn_target_encode", "x1")
  g1$params$k <- 3L
  res1 <- apply_gene(g1, train_data = data.table::as.data.table(df), val_data = data.table::as.data.table(df), target_col = "y")
  expect_true(g1$output_col %in% names(res1$train))
  expect_true(g1$output_col %in% names(res1$val))
  expect_equal(nrow(res1$train), n)
  expect_true(all(is.finite(res1$train[[g1$output_col]])))

  # Verify LOO training does not leak self-target:
  # With 2 nearest points possessing different targets, LOO prediction must be strictly between 0 and 1
  df_pair <- data.frame(
    x1 = c(0.0, 0.0001, 10.0, 10.0001),
    x2 = c(0.0, 0.0001, 10.0, 10.0001),
    y = c(1.0, 0.0, 1.0, 0.0)
  )
  g_pair <- create_gene("knn_target_encode", c("x1", "x2"))
  g_pair$params$k <- 1L
  res_pair <- apply_gene(g_pair, train_data = data.table::as.data.table(df_pair), val_data = data.table::as.data.table(df_pair), target_col = "y")
  # For row 1 (y=1), nearest neighbor in training is row 2 (y=0)
  # So row 1's LOO prediction must be pulled toward 0, not 1!
  expect_true(res_pair$train[[g_pair$output_col]][1] < res_pair$train[[g_pair$output_col]][2])

  # Test 3D column
  g3 <- create_gene("knn_target_encode", c("x1", "x2", "x3"))
  g3$params$k <- 5L
  res3 <- apply_gene(g3, train_data = data.table::as.data.table(df), val_data = data.table::as.data.table(df), target_col = "y")
  expect_true(g3$output_col %in% names(res3$train))
  expect_true(all(is.finite(res3$train[[g3$output_col]])))

  # Constant input must be rejected (never leak row-order y[1:(k+1)])
  df_const <- data.frame(x1 = rep(5.0, 20), y = c(rep(1.0, 10), rep(0.0, 10)))
  g_const <- create_gene("knn_target_encode", "x1")
  expect_error(
    apply_gene(g_const, train_data = data.table::as.data.table(df_const), target_col = "y", validate_genes = TRUE),
    "Constant column"
  )

  # Exact coordinate ties must be invariant to row order (no timestamp/index bias)
  df_ties <- data.frame(
    x1 = c(rep(1.0, 12), rep(2.0, 12)),
    y = c(rep(1.0, 6), rep(0.0, 6), rep(0.0, 6), rep(1.0, 6))
  )
  perm <- c(12:1, 24:13)
  g_tie <- create_gene("knn_target_encode", "x1")
  g_tie$params$k <- 3L
  r_orig <- apply_gene(g_tie, train_data = data.table::as.data.table(df_ties), target_col = "y", validate_genes = FALSE)
  r_perm <- apply_gene(g_tie, train_data = data.table::as.data.table(df_ties[perm, ]), target_col = "y", validate_genes = FALSE)
  expect_equal(r_orig$train[[g_tie$output_col]][perm], r_perm$train[[g_tie$output_col]], tolerance = 1e-12)

  # Landmark reference set respects evoFE.max_clustering_size
  old_opt <- options(evoFE.max_clustering_size = 10L)
  on.exit(options(old_opt), add = TRUE)
  res_cap <- apply_gene(g1, train_data = data.table::as.data.table(df), target_col = "y", validate_genes = FALSE)
  expect_lte(nrow(res_cap$gene$state$u_ref), 10L)
})

testthat::test_that("round_transform correctly discretizes continuous features with NA preservation", {
  x_test <- c(3.14159, -2.71828, 42.0, 0.055, NA, Inf, -Inf)
  df <- data.frame(num = x_test)

  # Digits = 0 (integer rounding)
  g0 <- create_gene("round_transform", "num")
  g0$params$digits <- 0L
  res0 <- apply_gene(g0, train_data = data.table::as.data.table(df), target_col = NULL)
  out0 <- res0$train[[g0$output_col]]
  expect_equal(out0[1:4], c(3, -3, 42, 0))
  expect_true(all(is.na(out0[5:7])))

  # Digits = 2
  g2 <- create_gene("round_transform", "num")
  g2$params$digits <- 2L
  res2 <- apply_gene(g2, train_data = data.table::as.data.table(df), target_col = NULL)
  out2 <- res2$train[[g2$output_col]]
  expect_equal(out2[1:4], c(3.14, -2.72, 42.0, 0.06))
  expect_true(all(is.na(out2[5:7])))

  # Formula formatting
  expect_match(gene_to_formula(g2), "^round_transform_d2\\(num\\)$")
})

testthat::test_that("decimal_digit handles IEEE-754 floats accurately and preserves NA", {
  x_test <- c(3.14159, -3.14159, 42.0, 0.05, NA, Inf)
  df <- data.frame(num = x_test)

  # Position 1
  g1 <- create_gene("decimal_digit", "num")
  g1$params$digits <- 1L
  res1 <- apply_gene(g1, train_data = data.table::as.data.table(df), target_col = NULL)
  out1 <- res1$train[[g1$output_col]]
  expect_equal(out1[1:4], c(1, 1, 0, 0))
  expect_true(all(is.na(out1[5:6])))

  # Position 2
  g2 <- create_gene("decimal_digit", "num")
  g2$params$digits <- 2L
  res2 <- apply_gene(g2, train_data = data.table::as.data.table(df), target_col = NULL)
  out2 <- res2$train[[g2$output_col]]
  expect_equal(out2[1:4], c(4, 4, 0, 5))
  expect_true(all(is.na(out2[5:6])))

  # IEEE-754 binary float edge cases that fail naive floor(abs(x) * 10^d) %% 10
  df_ieee <- data.frame(num = c(0.29, 0.58, 1.13, 2.3, 0.07, 4.1, 9.95, 123456.29))
  res_ieee <- apply_gene(g2, train_data = data.table::as.data.table(df_ieee), target_col = NULL)
  expect_equal(res_ieee$train[[g2$output_col]], c(9, 8, 3, 0, 7, 0, 5, 9))

  # Position 4
  g4 <- create_gene("decimal_digit", "num")
  g4$params$digits <- 4L
  res4 <- apply_gene(g4, train_data = data.table::as.data.table(df), target_col = NULL)
  out4 <- res4$train[[g4$output_col]]
  expect_equal(out4[1], 5)

  # Formula formatting
  expect_match(gene_to_formula(g2), "^decimal_digit_d2\\(num\\)$")
})

testthat::test_that("mutation and parameter operators handle new competition transformers cleanly", {
  ind <- create_individual(
    numeric_cols = c("num1", "num2", "num3"),
    categorical_cols = c("cat1", "cat2")
  )

  # Test mutate force add for new transformers
  set.seed(123)
  for (t_name in c("knn_target_encode", "round_transform", "decimal_digit")) {
    mut_ind <- mutate(ind, force_add = TRUE, allowed_transformers = c(t_name), task = "classification")
    expect_equal(length(mut_ind$genes), 1L)
    expect_equal(mut_ind$genes[[1]]$transformer_name, t_name)
    expect_true(length(mut_ind$genes[[1]]$input_cols) >= 1L)
  }

  # Test mutate modify
  g_rnd <- create_gene("round_transform", "num1")
  g_rnd$params$digits <- 1L
  ind_rnd <- create_individual(genes = list(g_rnd), numeric_cols = c("num1", "num2"))
  set.seed(42)
  mod_ind <- mutate(ind_rnd, force_add = FALSE, allowed_transformers = c("round_transform"))
  expect_true(length(mod_ind$genes) >= 1L)
})

testthat::test_that("evolve_features executes seamlessly with competition transformers", {
  set.seed(99)
  n <- 40
  train_df <- data.frame(
    feat1 = rnorm(n),
    feat2 = rnorm(n) + 0.12345,
    target = as.numeric(rnorm(n) > 0)
  )

  res <- evolve_features(
    data = train_df,
    target_col = "target",
    task = "classification",
    generations = 1,
    pop_size = 3,
    cv_folds = 2,
    threads = 1,
    allowed_transformers = c("knn_target_encode", "round_transform", "decimal_digit"),
    verbose = FALSE
  )

  expect_s3_class(res, "evo_recipe")
  expect_true(is.finite(res$best_individual$fitness))

  # Test predict with the evolved recipe
  trans_data <- predict(res, newdata = train_df)
  expect_equal(nrow(trans_data), n)

  # Test predict_model with the final trained model
  preds <- predict_model(res, newdata = train_df)
  expect_equal(length(preds), n)
  expect_true(all(is.finite(preds)))

  # Final recipe genes must not retain training-length loo_train vectors
  for (g in res$best_individual$genes) {
    if (!is.null(g$state)) {
      expect_null(g$state$loo_train)
      expect_null(g$state$loo_train_list)
    }
  }
})

testthat::test_that("pooled_target_encode, target_quantile_encode LOO and strict metric dispatch work", {
  # Singleton category ("rare") with extreme y = 100 must map to global prior on train (LOO), not memorize 100
  dt <- data.table::data.table(
    cat = c("rare", rep("A", 5), rep("B", 5)),
    y = c(100.0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14)
  )

  g_pte <- create_gene("pooled_target_encode", "cat")
  r_pte <- apply_gene(g_pte, data.table::copy(dt), data.table::copy(dt), target_col = "y", validate_genes = FALSE)
  expect_equal(r_pte$train[[g_pte$output_col]][1], mean(dt$y), tolerance = 1e-10)
  expect_gt(r_pte$val[[g_pte$output_col]][1], r_pte$train[[g_pte$output_col]][1])

  g_tqe <- create_gene("target_quantile_encode", "cat")
  g_tqe$params$q <- 0.5
  r_tqe <- apply_gene(g_tqe, data.table::copy(dt), data.table::copy(dt), target_col = "y", validate_genes = FALSE)
  expect_equal(r_tqe$train[[g_tqe$output_col]][1], stats::median(dt$y), tolerance = 1e-10)
  expect_gt(r_tqe$val[[g_tqe$output_col]][1], r_tqe$train[[g_tqe$output_col]][1])

  # Strict metric validation rejects unknown metrics instead of silently falling back
  expect_error(evoFE:::compute_metric(c(1, 2), c(1.1, 1.9), "regression", "rmsle"), "not supported")
  expect_error(evoFE:::validate_metric("unknown_xyz", "classification"), "not supported")
  expect_silent(evoFE:::validate_metric("rmse", "regression"))
  expect_silent(evoFE:::validate_metric("logloss", "classification"))
})

testthat::test_that("MEDIUM fixes: schema sync, leak-free smearing, multiclass rank_average sum, and CV seed propagation", {
  withr::local_seed(42)

  # 1. Datetime & target_quantile_encode parameter mutation schemas match implementations
  g_dt <- create_gene("datetime_extract", "d")
  g_dt$params$component <- "year"
  dt_muts <- replicate(30, evoFE:::mutate_transformer_param(g_dt, "component"))
  expect_true(all(dt_muts %in% c("year", "month", "day", "hour", "day_of_week", "weekend")))

  g_cyc <- create_gene("datetime_cyclic", "d")
  g_cyc$params$component <- "hour_sin"
  cyc_muts <- replicate(30, evoFE:::mutate_transformer_param(g_cyc, "component"))
  expect_true(all(cyc_muts %in% c("hour_sin", "hour_cos", "wday_sin", "wday_cos", "month_sin", "month_cos", "yday_sin", "yday_cos")))

  g_tqe <- create_gene("target_quantile_encode", "c")
  g_tqe$params$q <- 0.25
  tqe_muts <- replicate(30, evoFE:::mutate_transformer_param(g_tqe, "q"))
  expect_true(0.5 %in% tqe_muts)

  expect_equal(evoFE:::transformer_max_cols("knn_target_encode", 10, mode = "modify"), 3)
  expect_equal(evoFE:::transformer_max_cols("row_min", 10, mode = "modify"), 5)

  # Restrictive allowed_transformers incompatible with task must not silently fall back to all transformers
  ind_mc <- create_individual(numeric_cols = c("x1", "x2"), categorical_cols = c("c1", "c2"))
  mut_empty1 <- mutate(ind_mc, force_add = TRUE, task = "multiclass", allowed_transformers = "target_encode")
  expect_length(mut_empty1$genes, 0L)
  mut_empty2 <- mutate(ind_mc, force_add = TRUE, task = "multiclass", allowed_transformers = c("knn_target_encode", "woe_encode"))
  expect_length(mut_empty2$genes, 0L)

  # 2. Leak-free Duan smearing does not peek at validation targets during evaluate_fitness
  register_evaluator(
    "_test_const_zero",
    train_func = function(x_train, y_train, x_val = NULL, ...) {
      list(
        model = list(c = 0.0),
        predictions = if (!is.null(x_val)) rep(0.0, nrow(x_val)) else NULL,
        importances = setNames(rep(1 / ncol(x_train), ncol(x_train)), colnames(x_train))
      )
    },
    predict_func = function(model, x_new, ...) rep(0.0, nrow(x_new))
  )
  on.exit(if (exists("_test_const_zero", envir = evo_evaluators)) rm("_test_const_zero", envir = evo_evaluators), add = TRUE)

  df_sm <- data.table::data.table(x1 = rnorm(20), x2 = rnorm(20), y = c(rep(0.0, 10), rep(100.0, 10)))
  ind_sm <- create_individual(numeric_cols = c("x1", "x2"))
  ev_sm <- evoFE:::evaluate_fitness(
    ind_sm, df_sm, target_col = "y", task = "regression",
    evaluation_strategy = "split", split_ids = c(rep("train", 10), rep("val", 10)),
    evaluator = "_test_const_zero", metric = "rmse", target_transform = "log1p", threads = 1L
  )
  expect_equal(as.numeric(ev_sm$val_preds), rep(0.0, 10), tolerance = 1e-12)

  # 3. Multiclass rank_average normalizes row probabilities to sum to 1
  mc_df <- data.table::data.table(x1 = rnorm(30), x2 = rnorm(30), y = factor(rep(c("a", "b", "c"), 10)))
  register_evaluator(
    "_test_seed_mc",
    train_func = function(x_train, y_train, x_val = NULL, task = "multiclass", num_class = 3L, ...) {
      dots <- list(...)
      seen_seeds <<- c(seen_seeds, if (is.null(dots$seed)) NA_integer_ else as.integer(dots$seed))
      n_va <- if (!is.null(x_val)) nrow(x_val) else 0L
      p_va <- if (n_va > 0L) matrix(c(0.6, 0.3, 0.1), nrow = n_va, ncol = 3L, byrow = TRUE) else NULL
      list(model = list(ok = TRUE), predictions = p_va, importances = setNames(rep(0.5, ncol(x_train)), colnames(x_train)))
    },
    predict_func = function(model, x_new, ...) {
      matrix(seq_len(nrow(x_new) * 3L) / (nrow(x_new) * 3L + 1L), nrow = nrow(x_new), ncol = 3L)
    }
  )
  on.exit(if (exists("_test_seed_mc", envir = evo_evaluators)) rm("_test_seed_mc", envir = evo_evaluators), add = TRUE)

  seen_seeds <- integer(0)
  rec_mc <- evolve_features(
    mc_df, target_col = "y", task = "multiclass", evaluator = "_test_seed_mc",
    generations = 1, pop_size = 2, cv_folds = 2, islands = 2,
    seed = 777L, threads = 1L, verbose = FALSE
  )
  # First CV call (Generation 0 baseline) must receive seed = 777L
  expect_equal(seen_seeds[1L], 777L)

  ens_mc <- ensemble_islands(rec_mc, mc_df, method = "equal", rank_average = TRUE, threads = 1L, verbose = FALSE)
  preds_mc <- predict_model(ens_mc, mc_df[1:6, ])
  expect_equal(as.numeric(rowSums(preds_mc)), rep(1.0, 6), tolerance = 1e-12)

  # 4. RealMLP occlusion importance subsampling (>1000 rows) is immune to class/time sorting
  y_sort <- c(rep(0.0, 1050), rep(1.0, 150))
  x_sort <- cbind(sig = y_sort * 4.0 + rnorm(1200, sd = 0.05), noise = rnorm(1200))
  rmlp_fit <- evoFE:::rcpp_realmlp_train(
    x_train = x_sort, y_train = y_sort,
    task = "classification", n_epochs = 8L, batch_size = 128L,
    lr = 0.04, seed = 42L, threads = 1L, verbose = 0L
  )
  expect_gt(rmlp_fit$importances[1L], rmlp_fit$importances[2L] * 2.0)
})
