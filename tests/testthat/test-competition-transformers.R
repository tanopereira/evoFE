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
})

testthat::test_that("round_transform correctly discretizes continuous features with robust edge cases", {
  x_test <- c(3.14159, -2.71828, 42.0, 0.055, NA, Inf, -Inf)
  df <- data.frame(num = x_test)

  # Digits = 0 (integer rounding)
  g0 <- create_gene("round_transform", "num")
  g0$params$digits <- 0L
  res0 <- apply_gene(g0, train_data = data.table::as.data.table(df), target_col = NULL)
  out0 <- res0$train[[g0$output_col]]
  expect_equal(out0[1:4], c(3, -3, 42, 0))
  expect_true(all(is.finite(out0)))

  # Digits = 2
  g2 <- create_gene("round_transform", "num")
  g2$params$digits <- 2L
  res2 <- apply_gene(g2, train_data = data.table::as.data.table(df), target_col = NULL)
  out2 <- res2$train[[g2$output_col]]
  expect_equal(out2[1:4], c(3.14, -2.72, 42.0, 0.06))
  expect_true(all(is.finite(out2)))

  # Formula formatting
  expect_match(gene_to_formula(g2), "^round_transform_d2\\(num\\)$")
})

testthat::test_that("decimal_digit correctly extracts decimal positions with robust edge cases", {
  x_test <- c(3.14159, -3.14159, 42.0, 0.05, NA, Inf)
  df <- data.frame(num = x_test)

  # Position 1
  g1 <- create_gene("decimal_digit", "num")
  g1$params$digits <- 1L
  res1 <- apply_gene(g1, train_data = data.table::as.data.table(df), target_col = NULL)
  out1 <- res1$train[[g1$output_col]]
  expect_equal(out1[1:4], c(1, 1, 0, 0))
  expect_true(all(is.finite(out1)))

  # Position 2
  g2 <- create_gene("decimal_digit", "num")
  g2$params$digits <- 2L
  res2 <- apply_gene(g2, train_data = data.table::as.data.table(df), target_col = NULL)
  out2 <- res2$train[[g2$output_col]]
  expect_equal(out2[1:4], c(4, 4, 0, 5))
  expect_true(all(is.finite(out2)))

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
})
