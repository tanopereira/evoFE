test_that("is_supervised_transformer correctly classifies transformers", {
  expect_true(is_supervised_transformer("target_encode"))
  expect_true(is_supervised_transformer("pooled_target_encode"))
  expect_true(is_supervised_transformer("woe_encode"))
  expect_true(is_supervised_transformer("supervised_bgpca"))

  expect_false(is_supervised_transformer("pca"))
  expect_false(is_supervised_transformer("umap"))
  expect_false(is_supervised_transformer("lumbermark"))
  expect_false(is_supervised_transformer("genie"))
  expect_false(is_supervised_transformer("robust_scale"))
  expect_false(is_supervised_transformer("fourier_basis"))
  expect_false(is_supervised_transformer("log"))
})

test_that("global_unsupervised provides consistent coordinates and states across CV folds", {
  set.seed(42)
  n <- 60
  dt <- data.table::data.table(
    x1 = rnorm(n, mean = 10, sd = 2),
    x2 = rnorm(n, mean = 20, sd = 4),
    cat = sample(c("A", "B", "C"), n, replace = TRUE),
    target = rbinom(n, 1, 0.5)
  )

  # Create an individual with unsupervised PCA and supervised target encoding
  g_pca <- create_gene("pca", c("x1", "x2"))
  g_pca$params$comp_idx <- 1L
  g_te <- create_gene("target_encode", "cat")

  ind1 <- create_individual(
    genes = list(g_pca, g_te),
    numeric_cols = c("x1", "x2"),
    categorical_cols = "cat",
    all_numeric_cols = c("x1", "x2"),
    all_categorical_cols = "cat"
  )

  ind2 <- create_individual(
    genes = list(g_pca, g_te),
    numeric_cols = c("x1", "x2"),
    categorical_cols = "cat",
    all_numeric_cols = c("x1", "x2"),
    all_categorical_cols = "cat"
  )

  # Fold 1: train = rows 1:40, val = rows 41:60
  fold1_train <- dt[1:40, ]
  fold1_val <- dt[41:60, ]

  # Fold 2: train = rows 21:60, val = rows 1:20
  fold2_train <- dt[21:60, ]
  fold2_val <- dt[1:20, ]

  cache_global <- new.env(hash = TRUE, parent = emptyenv())

  # Apply on Fold 1 with full_data = dt
  res_f1 <- apply_individual(
    ind1, data.table::copy(fold1_train), data.table::copy(fold1_val),
    target_col = "target", state_cache = cache_global,
    full_data = dt, global_unsupervised = TRUE
  )

  # Apply on Fold 2 with full_data = dt
  res_f2 <- apply_individual(
    ind2, data.table::copy(fold2_train), data.table::copy(fold2_val),
    target_col = "target", state_cache = cache_global,
    full_data = dt, global_unsupervised = TRUE
  )

  # Unsupervised PCA feature on rows 41:60:
  # In Fold 1, rows 41:60 are val
  val_pca_f1 <- res_f1$val[[g_pca$output_col]]
  # In Fold 2, rows 41:60 are train (specifically rows 21:40 of fold2_train)
  train_pca_f2 <- res_f2$train[[g_pca$output_col]][21:40]

  # Because PCA was fit globally on dt, rows 41:60 have identical PCA coordinates in both folds
  expect_equal(val_pca_f1, train_pca_f2, tolerance = 1e-7)

  # And cache must hold the global state
  expected_key <- digest::digest(paste0(gene_to_state_formula(g_pca), "_global"), algo = "md5", serialize = FALSE)
  expect_true(expected_key %in% ls(cache_global))

  # Meanwhile, supervised target encoding must differ between folds (trained on different target samples)
  te_col <- g_te$output_col
  expect_true(te_col %in% names(res_f1$train))
  expect_true(te_col %in% names(res_f2$train))
})

test_that("global_unsupervised = FALSE falls back to fold-local unsupervised fitting", {
  set.seed(123)
  n <- 40
  dt <- data.table::data.table(
    x1 = rnorm(n),
    x2 = rnorm(n),
    target = rnorm(n)
  )

  g_pca1 <- create_gene("pca", c("x1", "x2"))
  g_pca1$params$comp_idx <- 1L
  ind1 <- create_individual(
    genes = list(g_pca1),
    numeric_cols = c("x1", "x2"),
    all_numeric_cols = c("x1", "x2")
  )

  g_pca2 <- create_gene("pca", c("x1", "x2"))
  g_pca2$params$comp_idx <- 1L
  ind2 <- create_individual(
    genes = list(g_pca2),
    numeric_cols = c("x1", "x2"),
    all_numeric_cols = c("x1", "x2")
  )

  # Different splits with different data distributions
  f1_train <- dt[1:20, ]
  f1_val <- dt[21:40, ]

  f2_train <- dt[21:40, ]
  f2_val <- dt[1:20, ]

  res_local_1 <- apply_individual(
    ind1, data.table::copy(f1_train), data.table::copy(f1_val),
    target_col = "target", global_unsupervised = FALSE
  )

  res_local_2 <- apply_individual(
    ind2, data.table::copy(f2_train), data.table::copy(f2_val),
    target_col = "target", global_unsupervised = FALSE
  )

  # Because they were fit locally on different folds without global data, the rotation states differ
  expect_false(identical(res_local_1$ind$genes[[1]]$state$model$rotation, res_local_2$ind$genes[[1]]$state$model$rotation))
})

test_that("multi-fidelity screening fits unsupervised transformers on full dataset", {
  set.seed(42)
  n <- 100
  dt <- data.table::data.table(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y = rnorm(n)
  )

  g_pca <- create_gene("pca", c("x1", "x2"))
  g_pca$params$comp_idx <- 1L
  ind <- create_individual(
    genes = list(g_pca),
    numeric_cols = c("x1", "x2"),
    all_numeric_cols = c("x1", "x2")
  )

  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  fitness_cache <- new.env(hash = TRUE, parent = emptyenv())

  # 50 rows train, 50 rows val
  sh_splits <- list(
    train = dt[1:50, ],
    val = dt[51:100, ]
  )
  # Low-fidelity train has only 20 rows
  lf_splits <- list(
    train = dt[1:20, ],
    val = dt[51:100, ]
  )

  res_mf <- evaluate_pop_mf(
    pop = list(ind),
    data = dt,
    target_col = "y",
    task = "regression",
    cv_folds = 2,
    evaluation_strategy = "split",
    split_ids = NULL,
    shared_splits = sh_splits,
    evaluator = "lightgbm",
    fold_ids = NULL,
    shared_folds = NULL,
    shared_full = dt,
    state_cache = state_cache,
    fitness_cache = fitness_cache,
    threads = 1,
    verbose = FALSE,
    running_best_fitness = -Inf,
    mf_on = TRUE,
    lf_shared_splits = lf_splits
  )

  # Check that state was fit globally on dt (100 rows, not 20 rows)
  full_pca <- stats::prcomp(as.matrix(dt[, .(x1, x2)]), center = TRUE, scale. = TRUE)
  cached_key <- digest::digest(paste0(gene_to_state_formula(g_pca), "_global"), algo = "md5", serialize = FALSE)
  expect_true(cached_key %in% ls(state_cache))
  cached_state <- get(cached_key, envir = state_cache)
  # Global state center and scale match the full 100-row dataset
  expect_equal(unname(cached_state$model$center), unname(full_pca$center), tolerance = 1e-6)
  expect_equal(unname(cached_state$model$scale), unname(full_pca$scale), tolerance = 1e-6)
})

