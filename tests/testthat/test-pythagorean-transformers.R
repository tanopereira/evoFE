test_that("row_min and row_max compute fused row-wise bounds accurately", {
  dt <- data.table::data.table(
    s1 = c(5, 4, 2, 3),
    s2 = c(5, 2, 2, 4),
    s3 = c(1, 4, 5, 3)
  )

  g_min <- create_gene("row_min", c("s1", "s2", "s3"))
  res_min <- apply_gene(g_min, dt)
  expect_equal(res_min$train[[g_min$output_col]], c(1, 2, 2, 3))

  g_max <- create_gene("row_max", c("s1", "s2", "s3"))
  res_max <- apply_gene(g_max, dt)
  expect_equal(res_max$train[[g_max$output_col]], c(5, 4, 5, 4))
})

test_that("relative_rating calculates primary minus baseline of other items", {
  dt <- data.table::data.table(
    food = c(1, 5, 4),
    seat = c(5, 1, 4),
    wifi = c(5, 1, 4)
  )

  g_rel <- create_gene("relative_rating", c("food", "seat", "wifi"))
  res_rel <- apply_gene(g_rel, dt)
  # row 1: food=1, others=(5, 5) -> mean=5 -> 1 - 5 = -4
  # row 2: food=5, others=(1, 1) -> mean=1 -> 5 - 1 = +4
  # row 3: food=4, others=(4, 4) -> mean=4 -> 4 - 4 = 0
  expect_equal(res_rel$train[[g_rel$output_col]], c(-4, 4, 0))
})

test_that("geometric_mean computes multiplicative Cobb-Douglas utility correctly", {
  dt <- data.table::data.table(
    a = c(5, 2, 4),
    b = c(5, 8, 4),
    c = c(1, 4, 4)
  )

  g_gm <- create_gene("geometric_mean", c("a", "b", "c"))
  res_gm <- apply_gene(g_gm, dt)
  vals <- res_gm$train[[g_gm$output_col]]

  # row 1: (5 * 5 * 1)^(1/3) = 25^(1/3)
  expect_equal(vals[1], 25^(1/3), tolerance = 1e-5)
  # row 2: (2 * 8 * 4)^(1/3) = 64^(1/3) = 4
  expect_equal(vals[2], 4.0, tolerance = 1e-5)
  # row 3: (4 * 4 * 4)^(1/3) = 4
  expect_equal(vals[3], 4.0, tolerance = 1e-5)
})

test_that("harmonic_mean computes smooth soft bottleneck correctly", {
  dt <- data.table::data.table(
    a = c(5, 2, 4),
    b = c(5, 2, 4),
    c = c(1, 2, 4)
  )

  g_hm <- create_gene("harmonic_mean", c("a", "b", "c"))
  res_hm <- apply_gene(g_hm, dt)
  vals <- res_hm$train[[g_hm$output_col]]

  # row 1: 3 / (1/5 + 1/5 + 1/1) = 3 / 1.4 = 2.142857
  expect_equal(vals[1], 3 / 1.4, tolerance = 1e-5)
  # row 2: all 2 -> 2.0
  expect_equal(vals[2], 2.0, tolerance = 1e-5)
  # row 3: all 4 -> 4.0
  expect_equal(vals[3], 4.0, tolerance = 1e-5)
})

test_that("pythagorean_imbalance computes AM - HM and satisfies inequality", {
  dt <- data.table::data.table(
    a = c(5, 4, 10, 1),
    b = c(5, 4, 2, 1),
    c = c(1, 4, 1, 100)
  )

  g_imb <- create_gene("pythagorean_imbalance", c("a", "b", "c"))
  res_imb <- apply_gene(g_imb, dt)
  vals <- res_imb$train[[g_imb$output_col]]

  # Universal inequality: AM >= HM -> AM - HM >= 0
  expect_true(all(vals >= -1e-6))
  # Row 2 has identical values (4, 4, 4) -> imbalance == 0
  expect_equal(vals[2], 0.0, tolerance = 1e-5)
  # Row 1: AM = (5+5+1)/3 = 11/3; HM = 3/1.4; AM - HM = 11/3 - 3/1.4
  expect_equal(vals[1], (11 / 3) - (3 / 1.4), tolerance = 1e-5)
})

test_that("numeric_freq learns frequency counts and projects to unseen test data", {
  train <- data.table::data.table(
    distance = c(500, 500, 500, 1200, 1200, 300),
    target = c(0, 1, 0, 1, 0, 1)
  )
  test <- data.table::data.table(
    distance = c(500, 1200, 9999), # 9999 is unseen in training
    target = c(0, 1, 0)
  )

  g_freq <- create_gene("numeric_freq", "distance")
  res <- apply_gene(g_freq, train, val_data = test, target_col = "target")

  # Train counts: 500 -> 3, 1200 -> 2, 300 -> 1
  expect_equal(res$train[[g_freq$output_col]], c(3, 3, 3, 2, 2, 1))

  # Test counts: 500 -> 3, 1200 -> 2, 9999 -> median of (1, 2, 3) = 2
  expect_equal(res$val[[g_freq$output_col]], c(3, 2, 2))
})


test_that("evolution runs successfully with new transformers", {
  set.seed(42)
  n <- 80
  train <- data.table::data.table(
    food = sample(1:5, n, replace = TRUE),
    seat = sample(1:5, n, replace = TRUE),
    clean = sample(1:5, n, replace = TRUE),
    distance = sample(c(240, 500, 1080), n, replace = TRUE),
    target = sample(0:1, n, replace = TRUE)
  )

  rec <- evolve_features(
    data = train,
    target_col = "target",
    task = "classification",
    evaluator = "lightgbm",
    allowed_transformers = c("row_min", "row_max", "relative_rating",
                             "geometric_mean", "harmonic_mean",
                             "pythagorean_imbalance", "numeric_freq"),
    generations = 2,
    pop_size = 3,
    verbose = FALSE
  )

  expect_s3_class(rec, "evo_recipe")
  preds <- predict_model(rec, train)
  expect_equal(length(preds), n)
})
