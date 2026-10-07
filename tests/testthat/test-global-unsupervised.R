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

test_that("global_unsupervised slices subsets by .row_id and persists in full_data", {
  set.seed(42)
  n <- 50
  dt <- data.table::data.table(
    x1 = rnorm(n),
    x2 = rnorm(n),
    target = rnorm(n)
  )

  tr_idx <- 1:35
  va_idx <- 36:50
  train_dt <- data.table::copy(dt[tr_idx, ])
  val_dt <- data.table::copy(dt[va_idx, ])
  data.table::setattr(train_dt, ".row_id", tr_idx)
  data.table::setattr(val_dt, ".row_id", va_idx)

  g_pca <- create_gene("pca", c("x1", "x2"))
  g_pca$params$comp_idx <- 1L
  ind <- create_individual(
    genes = list(g_pca),
    numeric_cols = c("x1", "x2"),
    all_numeric_cols = c("x1", "x2")
  )

  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  res <- apply_individual(
    ind, train_dt, val_dt,
    target_col = "target", state_cache = state_cache,
    full_data = dt, global_unsupervised = TRUE
  )

  out_col <- g_pca$output_col
  expect_true(out_col %in% names(dt))
  expect_true(out_col %in% names(res$train))
  expect_true(out_col %in% names(res$val))

  # Train and val columns must be exact slices of the global column
  expect_equal(res$train[[out_col]], dt[[out_col]][tr_idx])
  expect_equal(res$val[[out_col]], dt[[out_col]][va_idx])

  # A second evaluation with the same gene reuses dt[[out_col]] directly
  train_dt2 <- data.table::copy(dt[tr_idx, ])
  val_dt2 <- data.table::copy(dt[va_idx, ])
  data.table::setattr(train_dt2, ".row_id", tr_idx)
  data.table::setattr(val_dt2, ".row_id", va_idx)

  res2 <- apply_individual(
    ind, train_dt2, val_dt2,
    target_col = "target", state_cache = state_cache,
    full_data = dt, global_unsupervised = TRUE
  )
  expect_equal(res2$train[[out_col]], dt[[out_col]][tr_idx])
  expect_equal(res2$val[[out_col]], dt[[out_col]][va_idx])
})

test_that("rejected constant column does not pollute full_data", {
  n <- 30
  dt <- data.table::data.table(
    x1 = rep(5.0, n),
    x2 = rnorm(n),
    target = rnorm(n)
  )

  tr_idx <- 1:20
  va_idx <- 21:30
  train_dt <- data.table::copy(dt[tr_idx, ])
  val_dt <- data.table::copy(dt[va_idx, ])
  data.table::setattr(train_dt, ".row_id", tr_idx)
  data.table::setattr(val_dt, ".row_id", va_idx)

  # Scale on constant column produces NaN / constant which gets rejected
  g_scale <- create_gene("robust_scale", "x1")
  ind <- create_individual(
    genes = list(g_scale),
    numeric_cols = c("x1", "x2"),
    all_numeric_cols = c("x1", "x2")
  )

  out_col <- g_scale$output_col
  res <- apply_individual(
    ind, train_dt, val_dt,
    target_col = "target", allow_prune = TRUE,
    full_data = dt, global_unsupervised = TRUE
  )

  # Gene should be pruned and output_col must NOT exist in dt
  expect_false(out_col %in% names(dt))
  expect_false(out_col %in% names(res$train))
})

test_that("strip_individual_state preserves unsupervised states and strips supervised states", {
  g_pca <- create_gene("pca", c("x1", "x2"))
  g_pca$state <- list(model = "mock_pca_state")

  g_te <- create_gene("target_encode", "cat")
  g_te$state <- list(encoding_map = c(A = 0.5, B = 0.2))

  ind <- create_individual(
    genes = list(g_pca, g_te),
    numeric_cols = c("x1", "x2"),
    categorical_cols = "cat"
  )
  ind$fitness <- 0.95
  ind$val_preds <- c(1, 0, 1)

  stripped <- strip_individual_state(ind, keep_unsupervised = TRUE)

  # Fitness and predictions must be reset
  expect_true(is.na(stripped$fitness))
  expect_null(stripped$val_preds)

  # Unsupervised PCA state is preserved
  expect_equal(stripped$genes[[1]]$state$model, "mock_pca_state")

  # Supervised target encode state is stripped
  expect_null(stripped$genes[[2]]$state)
})

test_that("cross-island evaluation reuses global state cache and avoids redundant fitting", {
  set.seed(123)
  n <- 60
  dt <- data.table::data.table(
    x1 = rnorm(n),
    x2 = rnorm(n),
    target = rnorm(n)
  )

  global_cache <- new.env(hash = TRUE, parent = emptyenv())
  data.table::setattr(dt, ".global_state_cache", global_cache)

  island1_cache <- new.env(hash = TRUE, parent = global_cache)
  island2_cache <- new.env(hash = TRUE, parent = global_cache)

  # Island 1 partitions (rows 1:40 train, 41:60 val)
  tr1_idx <- 1:40
  va1_idx <- 41:60
  tr1 <- data.table::copy(dt[tr1_idx, ])
  va1 <- data.table::copy(dt[va1_idx, ])
  data.table::setattr(tr1, ".row_id", tr1_idx)
  data.table::setattr(va1, ".row_id", va1_idx)

  # Custom transformer with call counter to prove fit_func is only called once
  fit_counter <- 0L
  t_custom <- create_transformer(
    name = "test_counted_trans",
    type = "multivariate",
    fit_func = function(data, gene, target_col = NULL) {
      fit_counter <<- fit_counter + 1L
      list(offset = 42.0)
    },
    apply_func = function(data, gene, state = NULL) {
      sin(data[[gene$input_cols[1]]]) * cos(data[[gene$input_cols[2]]]) + state$offset
    },
    name_generator = function(gene) "test_counted_col"
  )
  evo_transformers$test_counted_trans <- t_custom
  on.exit(rm("test_counted_trans", envir = evo_transformers), add = TRUE)

  g_counted <- create_gene("test_counted_trans", c("x1", "x2"))
  ind1 <- create_individual(genes = list(g_counted), numeric_cols = c("x1", "x2"))

  # Island 1 applies individual
  res1 <- apply_individual(ind1, tr1, va1, target_col = "target",
                           state_cache = island1_cache, full_data = dt, global_unsupervised = TRUE)

  expect_equal(fit_counter, 1L)
  expect_true("test_counted_col" %in% names(dt))
  expect_equal(res1$train[["test_counted_col"]], dt[["test_counted_col"]][tr1_idx])

  # Island 2 receives migrated individual or evaluates new individual with same gene
  tr2_idx <- 21:60
  va2_idx <- 1:20
  tr2 <- data.table::copy(dt[tr2_idx, ])
  va2 <- data.table::copy(dt[va2_idx, ])
  data.table::setattr(tr2, ".row_id", tr2_idx)
  data.table::setattr(va2, ".row_id", va2_idx)

  ind2 <- create_individual(genes = list(g_counted), numeric_cols = c("x1", "x2"))
  res2 <- apply_individual(ind2, tr2, va2, target_col = "target",
                           state_cache = island2_cache, full_data = dt, global_unsupervised = TRUE)

  # fit_counter must STILL be 1 (fit_func was NOT called again!)
  expect_equal(fit_counter, 1L)
  expect_equal(res2$train[["test_counted_col"]], dt[["test_counted_col"]][tr2_idx])
  expect_equal(res2$val[["test_counted_col"]], dt[["test_counted_col"]][va2_idx])
})



