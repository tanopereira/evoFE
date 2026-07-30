test_that("similarity_encode works correctly", {
  library(data.table)

  df_train <- data.table(
    city = c("New York", "New York City", "N.Y.", "London", "London", "Paris", "Paris", "Paris")
  )
  df_test <- data.table(
    city = c("New York", "Paris", "Berlin", NA)
  )

  gene <- create_gene("similarity_encode", "city")
  expect_equal(gene$transformer_name, "similarity_encode")

  state <- evo_transformers$similarity_encode$fit_func(df_train, gene)
  expect_true(state$valid)
  expect_true(length(state$prototypes) > 0)

  res_train <- evo_transformers$similarity_encode$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  expect_true(is.numeric(res_train))

  res_test <- evo_transformers$similarity_encode$apply_func(df_test, gene, state)
  expect_equal(length(res_test), nrow(df_test))
  expect_true(is.numeric(res_test))
  expect_equal(res_test[4], 0) # NA returns 0
})

test_that("minhash_encode works correctly", {
  library(data.table)

  df <- data.table(
    title = c("Data Scientist", "Data Engineer", "Machine Learning Engineer", NA, "Software Engineer")
  )

  gene <- create_gene("minhash_encode", "title")
  expect_equal(gene$transformer_name, "minhash_encode")
  expect_true("comp_idx" %in% names(gene$params))

  state <- evo_transformers$minhash_encode$fit_func(df, gene)
  expect_true(state$valid)

  res <- evo_transformers$minhash_encode$apply_func(df, gene, state)
  expect_equal(length(res), nrow(df))
  expect_true(is.numeric(res))
  expect_true(all(res >= 0 & res <= 1, na.rm = TRUE))
  expect_equal(res[4], 0) # NA returns 0
})

test_that("gap_encode works correctly", {
  library(data.table)

  df_train <- data.table(
    desc = c("deep learning model", "machine learning pipeline", "statistical modeling", "deep neural network", "data engineering pipeline")
  )
  df_test <- data.table(
    desc = c("deep learning", "unseen text string", NA)
  )

  gene <- create_gene("gap_encode", "desc")
  expect_equal(gene$transformer_name, "gap_encode")

  state <- evo_transformers$gap_encode$fit_func(df_train, gene)
  expect_true(state$valid)
  expect_true(length(state$top_grams) > 0)

  res_train <- evo_transformers$gap_encode$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  expect_true(is.numeric(res_train))

  res_test <- evo_transformers$gap_encode$apply_func(df_test, gene, state)
  expect_equal(length(res_test), nrow(df_test))
  expect_true(is.numeric(res_test))
})

test_that("datetime_cyclic works correctly", {
  library(data.table)

  dates <- as.POSIXct(c("2026-01-01 00:00:00", "2026-06-15 12:00:00", "2026-12-31 23:59:59", NA))
  df <- data.table(ts = dates)

  components <- c("hour_sin", "hour_cos", "wday_sin", "wday_cos", "month_sin", "month_cos", "yday_sin", "yday_cos")

  for (comp in components) {
    gene <- create_gene("datetime_cyclic", "ts")
    gene$params$component <- comp
    res <- evo_transformers$datetime_cyclic$apply_func(df, gene)

    expect_equal(length(res), nrow(df))
    expect_true(is.numeric(res))
    expect_true(all(res >= -1 & res <= 1))
    expect_equal(res[4], 0) # NA returns 0
  }
})

test_that("target_quantile_encode works correctly", {
  library(data.table)

  df_train <- data.table(
    cat = c("A", "A", "A", "B", "B", "B", "C", "C"),
    y = c(10, 20, 30, 100, 200, 300, 5, 15)
  )
  df_test <- data.table(
    cat = c("A", "B", "Unseen", NA)
  )

  gene <- create_gene("target_quantile_encode", "cat")
  expect_equal(gene$transformer_name, "target_quantile_encode")

  state <- evo_transformers$target_quantile_encode$fit_func(df_train, gene, "y")
  expect_true(!is.null(state$mapping))

  res_train <- evo_transformers$target_quantile_encode$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  expect_true(is.numeric(res_train))

  res_test <- evo_transformers$target_quantile_encode$apply_func(df_test, gene, state)
  expect_equal(length(res_test), nrow(df_test))
  expect_equal(res_test[3], state$global_q) # Unseen maps to global quantile
  expect_equal(res_test[4], state$global_q) # NA maps to global quantile
})

test_that("cat_interaction_target_encode works correctly", {
  library(data.table)

  df_train <- data.table(
    cat1 = c("X", "X", "Y", "Y", "Z", "Z"),
    cat2 = c("A", "B", "A", "B", "A", "B"),
    y = c(1, 2, 10, 20, 100, 200)
  )
  df_test <- data.table(
    cat1 = c("X", "Y", "W", NA),
    cat2 = c("A", "B", "A", "B")
  )

  gene <- create_gene("cat_interaction_target_encode", c("cat1", "cat2"))
  expect_equal(gene$transformer_name, "cat_interaction_target_encode")

  state <- evo_transformers$cat_interaction_target_encode$fit_func(df_train, gene, "y")
  expect_true(!is.null(state$mapping))

  res_train <- evo_transformers$cat_interaction_target_encode$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))

  res_test <- evo_transformers$cat_interaction_target_encode$apply_func(df_test, gene, state)
  expect_equal(length(res_test), nrow(df_test))
  expect_equal(res_test[3], state$global_mean) # Unseen joint pair maps to global mean
})

test_that("skrub-inspired encoders run end-to-end in evolve_features", {
  library(data.table)

  set.seed(42)
  df <- data.table(
    city = sample(c("New York", "N.Y.", "London", "Paris"), 30, replace = TRUE),
    date_col = as.POSIXct("2026-01-01") + sample(1:864000, 30),
    cat2 = sample(c("TypeA", "TypeB"), 30, replace = TRUE),
    target = rnorm(30)
  )

  recipe <- evolve_features(
    data = df,
    target_col = "target",
    task = "regression",
    pop_size = 4,
    generations = 2,
    cv_folds = 2,
    verbose = FALSE
  )

  expect_s3_class(recipe, "evo_recipe")
  preds <- predict(recipe, df)
  expect_equal(nrow(preds), nrow(df))
  
  target_preds <- predict_model(recipe, df)
  expect_equal(length(target_preds), nrow(df))
})
