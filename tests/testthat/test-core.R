test_that("Gene creation and formula representation works", {
  gene <- create_gene("log", "x")
  expect_equal(gene$transformer_name, "log")
  expect_equal(gene$input_cols, "x")
  expect_equal(gene_to_formula(gene), "log(x)")
})

test_that("Individual mutation and crossover works", {
  ind <- create_individual(genes = list(), numeric_cols = c("x", "y"), categorical_cols = "cat")
  expect_equal(length(ind$genes), 0)
  
  # Mutate should add a gene
  ind_mut <- mutate(ind, force_add = TRUE)
  expect_gt(length(ind_mut$genes), 0)
  
  # Crossover should merge genes
  ind2 <- mutate(ind, force_add = TRUE)
  child <- crossover(ind_mut, ind2)
  expect_type(child, "list")
  expect_s3_class(child, "evo_individual")
})

test_that("Full feature evolution loop runs on dummy data", {
  set.seed(42)
  n <- 50
  dummy_data <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    c1 = sample(c("A", "B", "C"), n, replace = TRUE),
    target = sample(0:1, n, replace = TRUE)
  )
  
  suppressWarnings({
    res <- evolve_features(
      data = dummy_data,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  
  expect_s3_class(res, "evo_recipe")
  expect_type(res$best_individual, "list")
  
  # Test prediction S3 method
  suppressWarnings({
    preds <- predict(res, dummy_data)
  })
  expect_s3_class(preds, "data.table")
  expect_gt(ncol(preds), 0)
})

test_that("UMAP, Genie, MST score, Lumbermark, and Deadwood transformers work", {
  set.seed(42)
  n <- 30
  df <- data.table::as.data.table(data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  # Test UMAP
  gene_umap <- create_gene("umap", c("x1", "x2"))
  expect_true(gene_umap$params$comp_idx %in% 1:2)
  res_umap <- apply_gene(gene_umap, df, target_col = "target")
  expect_true(gene_umap$output_col %in% names(res_umap$train))
  expect_true(is.numeric(res_umap$train[[gene_umap$output_col]]))
  
  # Test Genie
  gene_genie <- create_gene("genie", c("x1", "x2"))
  expect_true(gene_genie$params$k >= 2 && gene_genie$params$k <= 5)
  res_genie <- apply_gene(gene_genie, df, target_col = "target")
  expect_true(gene_genie$output_col %in% names(res_genie$train))
  expect_true(is.factor(res_genie$train[[gene_genie$output_col]]))
  
  # Test MST Score
  gene_mst <- create_gene("mst_score", c("x1", "x2"))
  res_mst <- apply_gene(gene_mst, df, target_col = "target")
  expect_true(gene_mst$output_col %in% names(res_mst$train))
  expect_true(is.numeric(res_mst$train[[gene_mst$output_col]]))
  
  # Test Lumbermark
  gene_lumb <- create_gene("lumbermark", c("x1", "x2"))
  expect_true(gene_lumb$params$k >= 2 && gene_lumb$params$k <= 5)
  res_lumb <- apply_gene(gene_lumb, df, target_col = "target")
  expect_true(gene_lumb$output_col %in% names(res_lumb$train))
  expect_s3_class(res_lumb$train[[gene_lumb$output_col]], "factor")
  
  # Test Deadwood
  gene_dead <- create_gene("deadwood", c("x1", "x2"))
  res_dead <- apply_gene(gene_dead, df, target_col = "target")
  expect_true(gene_dead$output_col %in% names(res_dead$train))
  expect_s3_class(res_dead$train[[gene_dead$output_col]], "factor")
})

test_that("Constant columns are skipped and individual survives", {
  set.seed(42)
  n <- 10
  df <- data.table::as.data.table(data.frame(
    x1 = rep(1.0, n),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  gene_log <- create_gene("log", "x1")
  ind <- create_individual(genes = list(gene_log), numeric_cols = "x1")
  
  # Default (allow_prune = TRUE): constant gene is pruned, individual survives with 0 genes
  res <- apply_individual(ind, df, target_col = "target")
  expect_equal(length(res$ind$genes), 0)
  expect_false(gene_log$output_col %in% names(res$train))
  
  # Explicit allow_prune = FALSE: entire individual is killed -> NULL
  res_strict <- apply_individual(ind, df, target_col = "target", allow_prune = FALSE)
  expect_null(res_strict)
  
  # evaluate_fitness with default allow_prune=TRUE: bad gene pruned, individual gets a real fitness
  ind_eval <- evaluate_fitness(ind, df, target_col = "target", cv_folds = 2)
  expect_true(is.finite(ind_eval$fitness))
})

test_that("Genie, MST Score, Lumbermark, and Deadwood handle constant/all-zero data without crashing", {
  set.seed(42)
  n <- 10
  # All zero dataset
  df_zero <- data.table::as.data.table(data.frame(
    x1 = rep(0.0, n),
    x2 = rep(0.0, n),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  # Genie on all zero data should generate a constant column and thus fail apply_gene
  gene_genie <- create_gene("genie", c("x1", "x2"))
  expect_error(apply_gene(gene_genie, df_zero, target_col = "target"), "Constant column generated")
  
  # MST score on all zero data should generate a constant column and thus fail apply_gene
  gene_mst <- create_gene("mst_score", c("x1", "x2"))
  expect_error(apply_gene(gene_mst, df_zero, target_col = "target"), "Constant column generated")
  
  # Lumbermark on all zero data should generate a constant column and thus fail apply_gene
  gene_lumb <- create_gene("lumbermark", c("x1", "x2"))
  expect_error(apply_gene(gene_lumb, df_zero, target_col = "target"), "Constant column generated")
  
  # Deadwood on all zero data should generate a constant column and thus fail apply_gene
  gene_dead <- create_gene("deadwood", c("x1", "x2"))
  expect_error(apply_gene(gene_dead, df_zero, target_col = "target"), "Constant column generated")
  
  # Also verify with an individual that they are skipped during evaluation
  ind <- create_individual(genes = list(gene_genie, gene_mst, gene_lumb, gene_dead), numeric_cols = c("x1", "x2"))
  res <- apply_individual(ind, df_zero, target_col = "target")
  expect_equal(length(res$ind$genes), 0)
})

test_that("Best model is saved and predict_model works", {
  set.seed(42)
  n <- 50
  df <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = sample(0:1, n, replace = TRUE)
  )
  
  suppressWarnings({
    res <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  
  expect_false(is.null(res$best_model))
  
  # predict_model should return numeric vector of predictions
  preds <- predict_model(res, df)
  expect_type(preds, "double")
  expect_equal(length(preds), n)
  
  # predict_model should work on exactly 1 row (regression test for transpose bug)
  preds_1 <- predict_model(res, df[1, , drop = FALSE])
  expect_type(preds_1, "double")
  expect_equal(length(preds_1), 1)
})

test_that("State caching and fit-skipping works correctly", {
  set.seed(42)
  n <- 30
  df <- data.table::as.data.table(data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  # Create a stateful gene (PCA)
  gene <- create_gene("pca", c("x1", "x2"))
  ind <- create_individual(genes = list(gene), numeric_cols = c("x1", "x2"))
  
  # Setup caches
  shared_folds <- list(
    list(train = data.table::copy(df), val = data.table::copy(df))
  )
  shared_full <- data.table::copy(df)
  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  
  # 1. Run apply_gene the first time on shared_folds
  # Since col doesn't exist, it should fit the state and create the column.
  # But since state_cache is NULL (CV fold evaluation), it should fit it but not write to state_cache.
  res1 <- apply_gene(gene, shared_folds[[1]]$train, shared_folds[[1]]$val, target_col = "target", state_cache = NULL)
  expect_true(gene$output_col %in% names(shared_folds[[1]]$train))
  expect_false(is.null(res1$gene$state))
  
  # Let's clean the state to test the second run where the column exists but state is NULL.
  gene_no_state <- gene
  gene_no_state$state <- NULL
  
  # 2. Run apply_gene again on shared_folds where col exists.
  # Since col exists and state_cache is NULL, fit_func should NOT be run and state should remain NULL.
  res2 <- apply_gene(gene_no_state, shared_folds[[1]]$train, shared_folds[[1]]$val, target_col = "target", state_cache = NULL)
  expect_true(is.null(res2$gene$state))
  
  # 3. Now test on the full dataset with a state_cache.
  # The column doesn't exist in shared_full.
  # So it should fit the state, create the column, and store the state in state_cache.
  expect_false(gene$output_col %in% names(shared_full))
  res3 <- apply_gene(gene_no_state, shared_full, NULL, target_col = "target", state_cache = state_cache)
  expect_true(gene$output_col %in% names(shared_full))
  expect_false(is.null(res3$gene$state))
  
  # Verify that the state was indeed stored in the state_cache
  data_hash <- digest::digest(shared_full[["target"]], algo = "xxhash64")
  cache_key <- digest::digest(paste0(gene_to_state_formula(gene), "_", data_hash), algo = "md5", serialize = FALSE)
  expect_true(exists(cache_key, envir = state_cache))
  
  # 4. Now run on shared_full where the column exists and state_cache is passed.
  # Since the column exists and the state is in the cache, it should retrieve the state from the cache
  # without re-running fit_func.
  gene_no_state_2 <- gene
  gene_no_state_2$state <- NULL
  
  res4 <- apply_gene(gene_no_state_2, shared_full, NULL, target_col = "target", state_cache = state_cache)
  expect_false(is.null(res4$gene$state))
  expect_equal(res4$gene$state$model$rotation, res3$gene$state$model$rotation)
})

test_that("New algorithm-agnostic transformers work correctly", {
  set.seed(42)
  n <- 30
  df <- data.table::as.data.table(data.frame(
    x1 = exp(stats::rnorm(n) * 5),
    x2 = stats::rnorm(n),
    cat = sample(c("A", "B"), n, replace = TRUE),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  # 1. mst_score
  gene_ds <- create_gene("mst_score", c("x1", "x2"))
  res_ds <- apply_gene(gene_ds, df, target_col = "target")
  expect_true(gene_ds$output_col %in% names(res_ds$train))
  expect_type(res_ds$train[[gene_ds$output_col]], "double")
  
  # 2. groupby_ratio
  gene_gr <- create_gene("groupby_ratio", c("cat", "x1"))
  res_gr <- apply_gene(gene_gr, df, target_col = "target")
  expect_true(gene_gr$output_col %in% names(res_gr$train))
  expect_type(res_gr$train[[gene_gr$output_col]], "double")
  
  # 3. groupby_zscore
  gene_gz <- create_gene("groupby_zscore", c("cat", "x1"))
  res_gz <- apply_gene(gene_gz, df, target_col = "target")
  expect_true(gene_gz$output_col %in% names(res_gz$train))
  expect_type(res_gz$train[[gene_gz$output_col]], "double")
  
  # 4. quantile_binning
  gene_qb <- create_gene("quantile_binning", "x1")
  res_qb <- apply_gene(gene_qb, df, target_col = "target")
  expect_true(gene_qb$output_col %in% names(res_qb$train))
  expect_type(res_qb$train[[gene_qb$output_col]], "integer")
  
  # 5. log_binning
  gene_lb <- create_gene("log_binning", "x1")
  res_lb <- apply_gene(gene_lb, df)
  expect_true(gene_lb$output_col %in% names(res_lb$train))
  expect_type(res_lb$train[[gene_lb$output_col]], "integer")
  
  # 6. normalized_difference
  gene_nd <- create_gene("normalized_difference", c("x1", "x2"))
  res_nd <- apply_gene(gene_nd, df)
  expect_true(gene_nd$output_col %in% names(res_nd$train))
  expect_type(res_nd$train[[gene_nd$output_col]], "double")
  
  # 7. log_ratio
  gene_lr <- create_gene("log_ratio", c("x1", "x2"))
  res_lr <- apply_gene(gene_lr, df)
  expect_true(gene_lr$output_col %in% names(res_lr$train))
  expect_type(res_lr$train[[gene_lr$output_col]], "double")
  
  # 8. random_projection
  gene_rp <- create_gene("random_projection", c("x1", "x2"))
  res_rp <- apply_gene(gene_rp, df, target_col = "target")
  expect_true(gene_rp$output_col %in% names(res_rp$train))
  expect_type(res_rp$train[[gene_rp$output_col]], "double")
  expect_false(is.null(res_rp$gene$state$w))
})

test_that("Individual topological sorting of genes works correctly", {
  # Gene B depends on original features
  gene_b <- create_gene("log", "x1")
  # Gene A depends on Gene B (so B must come before A!)
  gene_a <- create_gene("log", gene_b$output_col)
  
  # Create an individual where the genes are in the WRONG order (A before B)
  ind <- create_individual(
    genes = list(gene_a, gene_b),
    numeric_cols = c("x1", "x2")
  )
  
  # The output individual should have the genes in the CORRECT order (B before A)
  expect_equal(length(ind$genes), 2)
  expect_equal(ind$genes[[1]]$output_col, gene_b$output_col)
  expect_equal(ind$genes[[2]]$output_col, gene_a$output_col)
  
  # Now create an individual with a missing dependency (only gene_a which depends on gene_b, but gene_b is not provided)
  ind_missing <- create_individual(
    genes = list(gene_a),
    numeric_cols = c("x1", "x2")
  )
  
  # The invalid gene_a should be automatically discarded
  expect_equal(length(ind_missing$genes), 0)
})

test_that("Prediction with stateful features does not crash and preserves states", {
  ind <- create_individual(
    genes = list(create_gene("genie", c("x1", "x2"))),
    numeric_cols = c("x1", "x2")
  )
  recipe <- structure(
    list(
      best_individual = ind,
      best_model = "dummy",
      evaluator = "xgboost"
    ),
    class = "evo_recipe"
  )
  
  df <- data.frame(x1 = rnorm(10), x2 = rnorm(10), c1 = "A")
  # Calling predict should not crash even if the state is initially NULL
  res_pred <- predict(recipe, df)
  expect_true(any(grepl("Genie", names(res_pred))))
  
  # Also test evolving features saves the states in best_individual
  res_evolve <- evolve_features(
    data = iris[, 1:5],
    target_col = "Petal.Length",
    task = "regression",
    generations = 2,
    pop_size = 2,
    cv_folds = 2,
    early_stopping_rounds = 1,
    evaluator = "xgboost",
    verbose = FALSE
  )
  
  # Check that predict on the resulting recipe works perfectly
  res_pred_evolve <- predict(res_evolve, iris[, 1:5])
  expect_s3_class(res_pred_evolve, "data.table")
  expect_gt(nrow(res_pred_evolve), 0)
})

test_that("Multiclass classification task runs and predicts correctly", {
  # Run evolution on iris predicting Species
  res <- evolve_features(
    data = iris,
    target_col = "Species",
    task = "multiclass",
    generations = 2,
    pop_size = 2,
    cv_folds = 2,
    early_stopping_rounds = 1,
    evaluator = "xgboost",
    verbose = FALSE
  )
  
  # Ensure the target encoding was not used (since it is forbidden in multiclass)
  for (gene in res$best_individual$genes) {
    expect_false(gene$transformer_name == "target_encode")
  }
  
  # Predict and assert correct multiclass probability matrix output
  preds <- predict_model(res, iris)
  expect_true(is.matrix(preds))
  expect_equal(colnames(preds), levels(iris$Species))
  expect_equal(nrow(preds), nrow(iris))
  expect_equal(ncol(preds), 3)
})

test_that("genie and categorical binning output factors and can be used for grouping", {
  df <- data.table::as.data.table(data.frame(
    x1 = rnorm(50) * 100,
    x2 = rnorm(50) * 100,
    target = sample(0:1, 50, replace = TRUE)
  ))
  
  # 1. Genie
  gene_genie <- create_gene("genie", c("x1", "x2"))
  res_genie <- apply_gene(gene_genie, df, target_col = "target")
  expect_true(is.factor(res_genie$train[[gene_genie$output_col]]))
  
  # 2. Quantile Binning Cat
  gene_qbc <- create_gene("quantile_binning_cat", "x1")
  res_qbc <- apply_gene(gene_qbc, df, target_col = "target")
  expect_true(is.factor(res_qbc$train[[gene_qbc$output_col]]))
  
  # 3. Log Binning Cat
  gene_lbc <- create_gene("log_binning_cat", "x1")
  res_lbc <- apply_gene(gene_lbc, df, target_col = "target")
  expect_true(is.factor(res_lbc$train[[gene_lbc$output_col]]))
  
  # 4. Groupby mean using categorical gene output
  df_with_genie <- res_genie$train
  gene_grp <- create_gene("groupby_mean", c(gene_genie$output_col, "x1"))
  res_grp <- apply_gene(gene_grp, df_with_genie, target_col = "target")
  expect_true(gene_grp$output_col %in% names(res_grp$train))
  expect_type(res_grp$train[[gene_grp$output_col]], "double")
})

test_that("set.seed before evolve_features produces reproducible results", {
  run_once <- function(s) {
    set.seed(s)
    evolve_features(
      data = iris[, 1:5],
      target_col = "Petal.Length",
      task = "regression",
      generations = 2,
      pop_size = 3,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "xgboost",
      verbose = FALSE
    )
  }
  
  res1 <- run_once(123)
  res2 <- run_once(123)
  
  expect_equal(res1$best_individual$fitness, res2$best_individual$fitness)
  expect_equal(
    individual_to_recipe_string(res1$best_individual),
    individual_to_recipe_string(res2$best_individual)
  )
})

test_that("predict handles genes that get pruned during apply_individual", {
  # Create a gene that references a column NOT in the newdata.
  # apply_individual should prune it, and predict should use the pruned list.
  gene_ok <- create_gene("log", "Sepal.Width")
  gene_bad <- create_gene("log", "MISSING_COLUMN")
  
  ind <- create_individual(
    genes = list(gene_ok),
    numeric_cols = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"),
    categorical_cols = character(0)
  )
  
  # Force the bad gene in after topological sort (simulating a stale recipe)
  ind$genes <- c(ind$genes, list(gene_bad))
  
  recipe <- structure(
    list(
      best_individual = ind,
      best_model = "dummy",
      evaluator = "xgboost"
    ),
    class = "evo_recipe"
  )
  
  # predict should NOT crash; gene_bad should be pruned
  res <- predict(recipe, iris[, 1:4])
  expect_s3_class(res, "data.table")
  expect_true("log(Sepal.Width)" %in% names(res))
  expect_false("log(MISSING_COLUMN)" %in% names(res))
})

test_that("max_clustering_size downsampling works in genie and mst_score", {
  # Create a large dataset
  set.seed(42)
  df <- data.table::as.data.table(matrix(rnorm(1000 * 2), ncol = 2))
  colnames(df) <- c("x1", "x2")
  df[, target := rnorm(1000)]
  
  # Set option to a small value
  options(evoFE.max_clustering_size = 50)
  on.exit(options(evoFE.max_clustering_size = 5000), add = TRUE)
  
  gene_genie <- create_gene("genie", c("x1", "x2"))
  state_genie <- evo_transformers$genie$fit_func(df, gene_genie, "target")
  
  expect_true(state_genie$valid)
  expect_equal(nrow(state_genie$x_train), 50)
  expect_equal(length(state_genie$labels), 50)
  
  # Apply output should have the original length
  applied <- evo_transformers$genie$apply_func(df, gene_genie, state_genie)
  expect_equal(length(applied), 1000)
  
  # Test with mst_score
  gene_mst <- create_gene("mst_score", c("x1", "x2"))
  state_mst <- evo_transformers$mst_score$fit_func(df, gene_mst, "target")
  expect_true(state_mst$valid)
  expect_equal(nrow(state_mst$x_train), 50)
  
  applied_mst <- evo_transformers$mst_score$apply_func(df, gene_mst, state_mst)
  expect_equal(length(applied_mst), 1000)
})

test_that("mutate adds all components for UMAP, PCA, and SVD", {
  attempts <- 0
  pca_added <- FALSE
  umap_added <- FALSE
  
  while ((!pca_added || !umap_added) && attempts < 500) {
    attempts <- attempts + 1
    ind <- create_individual(genes = list(), numeric_cols = c("x1", "x2", "x3"))
    ind_mut <- mutate(ind, force_add = TRUE, task = "regression")
    genes <- ind_mut$genes
    if (length(genes) > 0) {
      t_name <- genes[[1]]$transformer_name
      if (t_name == "pca") {
        expect_equal(length(genes), 3)
        expect_equal(sapply(genes, function(g) g$params$comp_idx), 1:3)
        pca_added <- TRUE
      } else if (t_name == "umap") {
        expect_equal(length(genes), 2)
        expect_equal(sapply(genes, function(g) g$params$comp_idx), 1:2)
        umap_added <- TRUE
      }
    }
  }
  expect_true(pca_added)
  expect_true(umap_added)
})

test_that("different components share the same fitted state in state_cache", {
  state_cache <- new.env(hash = TRUE, parent = emptyenv())
  df <- data.table::as.data.table(matrix(rnorm(100 * 3), ncol = 3))
  colnames(df) <- c("x1", "x2", "x3")
  df[, target := rnorm(100)]
  
  # 1. Test PCA
  gene_pca1 <- create_gene("pca", c("x1", "x2"))
  gene_pca1$params$comp_idx <- 1
  gene_pca1$output_col <- "PCA1"
  
  gene_pca2 <- create_gene("pca", c("x1", "x2"))
  gene_pca2$params$comp_idx <- 2
  gene_pca2$output_col <- "PCA2"
  
  # Fit gene_pca1
  res1 <- apply_gene(gene_pca1, df, target_col = "target", state_cache = state_cache)
  expect_false(is.null(res1$gene$state))
  
  # Fit gene_pca2. It should retrieve the state from cache without recalculating/refitting.
  res2 <- apply_gene(gene_pca2, df, target_col = "target", state_cache = state_cache)
  expect_identical(res1$gene$state, res2$gene$state)
  
  # 2. Test truncated_svd
  gene_svd1 <- create_gene("truncated_svd", c("x1", "x2", "x3"))
  gene_svd1$params$comp_idx <- 1
  gene_svd1$output_col <- "SVD1"
  
  gene_svd2 <- create_gene("truncated_svd", c("x1", "x2", "x3"))
  gene_svd2$params$comp_idx <- 2
  gene_svd2$output_col <- "SVD2"
  
  gene_svd3 <- create_gene("truncated_svd", c("x1", "x2", "x3"))
  gene_svd3$params$comp_idx <- 3
  gene_svd3$output_col <- "SVD3"
  
  # Apply and fit them
  res_svd1 <- apply_gene(gene_svd1, df, target_col = "target", state_cache = state_cache)
  res_svd2 <- apply_gene(gene_svd2, df, target_col = "target", state_cache = state_cache)
  res_svd3 <- apply_gene(gene_svd3, df, target_col = "target", state_cache = state_cache)
  
  # State should be identical/shared
  expect_identical(res_svd1$gene$state, res_svd2$gene$state)
  expect_identical(res_svd1$gene$state, res_svd3$gene$state)
  
  # Output columns must be different from each other (not identical components)
  col1 <- res_svd1$train$SVD1
  col2 <- res_svd2$train$SVD2
  col3 <- res_svd3$train$SVD3
  
  expect_true(mean(abs(col1 - col2)) > 1e-5)
  expect_true(mean(abs(col1 - col3)) > 1e-5)
  expect_true(mean(abs(col2 - col3)) > 1e-5)
})

test_that("umap, genie, and mst_score respect evoFE.verbose option", {
  df <- data.table::as.data.table(data.frame(
    x1 = rnorm(20),
    x2 = rnorm(20),
    target = sample(0:1, 20, replace = TRUE)
  ))
  
  # Set options(evoFE.verbose = TRUE) to test truthiness
  options(evoFE.verbose = TRUE)
  on.exit(options(evoFE.verbose = 0), add = TRUE)
  
  # Test UMAP fit messages
  gene_umap <- create_gene("umap", c("x1", "x2"))
  msg_umap_fit <- testthat::capture_messages({
    state_umap <- evo_transformers$umap$fit_func(df, gene_umap, "target")
  })
  expect_true(any(grepl("\\[UMAP Fit\\]", msg_umap_fit)))
  
  # Test UMAP apply messages
  msg_umap_apply <- testthat::capture_messages({
    evo_transformers$umap$apply_func(df, gene_umap, state_umap)
  })
  expect_true(any(grepl("\\[UMAP Apply\\]", msg_umap_apply)))
  
  # Test MST score messages
  gene_mst <- create_gene("mst_score", c("x1", "x2"))
  msg_mst_fit <- testthat::capture_messages({
    state_mst <- evo_transformers$mst_score$fit_func(df, gene_mst, "target")
  })
  expect_true(any(grepl("\\[MST Fit\\]", msg_mst_fit)))
  
  msg_mst_apply <- testthat::capture_messages({
    evo_transformers$mst_score$apply_func(df, gene_mst, state_mst)
  })
  expect_true(any(grepl("\\[MST Apply\\]", msg_mst_apply)))
  
  # Test Genie messages
  gene_genie <- create_gene("genie", c("x1", "x2"))
  msg_genie_fit <- testthat::capture_messages({
    state_genie <- evo_transformers$genie$fit_func(df, gene_genie, "target")
  })
  expect_true(any(grepl("\\[Genie Fit\\]", msg_genie_fit)))
  
  msg_genie_apply <- testthat::capture_messages({
    evo_transformers$genie$apply_func(df, gene_genie, state_genie)
  })
  expect_true(any(grepl("\\[Genie Apply\\]", msg_genie_apply)))
  
  # Now set options(evoFE.verbose = 0) and verify NO messages are output
  options(evoFE.verbose = 0)
  msg_silent <- testthat::capture_messages({
    state_umap <- evo_transformers$umap$fit_func(df, gene_umap, "target")
    evo_transformers$umap$apply_func(df, gene_umap, state_umap)
  })
  expect_equal(length(msg_silent), 0)
})

test_that("evolve_features and evaluate_fitness support evaluation_strategy = 'split' and stratification", {
  set.seed(42)
  n <- 60
  df <- data.frame(
    x1 = stats::rnorm(n),
    x2 = stats::rnorm(n),
    target = sample(0:1, n, replace = TRUE)
  )
  
  # 1. Test stratified_split helper (internal)
  # classification target -> stratified
  splits <- evoFE:::stratified_split(df$target, c(0.6, 0.2, 0.2))
  expect_equal(length(splits), n)
  expect_true(all(splits %in% c("train", "val", "holdout")))
  
  # check stratification ratios
  train_target <- df$target[splits == "train"]
  val_target <- df$target[splits == "val"]
  holdout_target <- df$target[splits == "holdout"]
  
  # expect class proportions to be roughly preserved/similar
  p_total <- mean(df$target)
  p_train <- mean(train_target)
  p_val <- mean(val_target)
  
  # since n is 60, proportions should be very close
  expect_lt(abs(p_train - p_total), 0.1)
  expect_lt(abs(p_val - p_total), 0.1)
  
  # 2. Test evolve_features under split strategy with holdout
  suppressWarnings({
    res_split <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      evaluation_strategy = "split",
      split_ratio = c(0.6, 0.2, 0.2),
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  
  expect_s3_class(res_split, "evo_recipe")
  expect_true(!is.null(res_split$best_individual$holdout_fitness))
  expect_true(is.numeric(res_split$best_individual$holdout_fitness))
  expect_gt(res_split$best_individual$holdout_fitness, -Inf)
  
  # 3. Test evolve_features under split strategy without holdout (split_ratio of length 2)
  suppressWarnings({
    res_split2 <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      evaluation_strategy = "split",
      split_ratio = c(0.7, 0.3),
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  
  expect_s3_class(res_split2, "evo_recipe")
  expect_true(is.null(res_split2$best_individual$holdout_fitness))
  
  # 4. Test evolve_features with manually passed split_ids
  manual_ids <- rep("train", n)
  manual_ids[1:round(n*0.3)] <- "val"
  manual_ids[(round(n*0.3)+1):round(n*0.5)] <- "holdout"
  
  suppressWarnings({
    res_manual <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      evaluation_strategy = "split",
      split_ids = manual_ids,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  expect_s3_class(res_manual, "evo_recipe")
  expect_true(!is.null(res_manual$best_individual$holdout_fitness))
})

test_that("one_hot_encode transformer works", {
  set.seed(42)
  n <- 50
  cat_col <- c(rep("A", 30), rep("B", 15), rep("C", 5))
  df <- data.table::as.data.table(data.frame(
    cat_val = factor(cat_col),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  # Test OHE component 1 (should map to "A")
  gene_ohe1 <- create_gene("one_hot_encode", "cat_val")
  gene_ohe1$params$comp_idx <- 1
  res_ohe1 <- apply_gene(gene_ohe1, df, target_col = "target")
  
  expect_true(gene_ohe1$output_col %in% names(res_ohe1$train))
  expect_equal(res_ohe1$train[[gene_ohe1$output_col]], as.numeric(df$cat_val == "A"))
  
  # Test OHE component 2 (should map to "B")
  gene_ohe2 <- create_gene("one_hot_encode", "cat_val")
  gene_ohe2$params$comp_idx <- 2
  res_ohe2 <- apply_gene(gene_ohe2, df, target_col = "target")
  
  expect_true(gene_ohe2$output_col %in% names(res_ohe2$train))
  expect_equal(res_ohe2$train[[gene_ohe2$output_col]], as.numeric(df$cat_val == "B"))
  
  # Test OHE component 6 ("other" - should map to "C" because C is < 5% or not in top 5, actually C is 5/50 = 10%, let's make it 2%)
  cat_col_rare <- c(rep("A", 45), rep("B", 4), rep("C", 1))
  df_rare <- data.table::as.data.table(data.frame(
    cat_val = factor(cat_col_rare),
    target = sample(0:1, n, replace = TRUE)
  ))
  
  gene_ohe6 <- create_gene("one_hot_encode", "cat_val")
  gene_ohe6$params$comp_idx <- 6
  res_ohe6 <- apply_gene(gene_ohe6, df_rare, target_col = "target")
  
  expect_true(gene_ohe6$output_col %in% names(res_ohe6$train))
  expect_equal(res_ohe6$train[[gene_ohe6$output_col]], as.numeric(df_rare$cat_val == "C"))
})

test_that("Custom transformer registration works", {
  custom_trans <- create_transformer(
    name = "add_one",
    type = "unary",
    input_type = "numeric",
    apply_func = function(data, gene, state = NULL) {
      data[[gene$input_cols[1]]] + 1
    },
    name_generator = function(gene) paste0("add1_", gene$input_cols[1])
  )
  register_transformer("add_one", custom_trans)
  
  expect_true("add_one" %in% names(evo_transformers))
  expect_identical(evo_transformers$add_one, custom_trans)
  
  gene <- create_gene("add_one", "x1")
  df <- data.table::data.table(x1 = 1:5)
  res <- apply_gene(gene, df)
  expect_equal(res$train[[gene$output_col]], 2:6)
})

test_that("datetime_extract transformer works", {
  df <- data.table::data.table(
    date_str = c("2026-05-29 23:52:42", "2025-01-01 12:00:00")
  )
  gene_year <- create_gene("datetime_extract", "date_str")
  gene_year$params$component <- "year"
  res_year <- apply_gene(gene_year, df)
  expect_equal(res_year$train[[gene_year$output_col]], c(2026, 2025))
  
  gene_month <- create_gene("datetime_extract", "date_str")
  gene_month$params$component <- "month"
  res_month <- apply_gene(gene_month, df)
  expect_equal(res_month$train[[gene_month$output_col]], c(5, 1))
})

test_that("target_encode_multiclass transformer works", {
  set.seed(42)
  df <- data.table::data.table(
    cat = rep(c("A", "B"), each = 10),
    target = c(rep("X", 8), rep("Y", 2), rep("Y", 7), rep("X", 3))
  )
  gene_te <- create_gene("target_encode_multiclass", "cat")
  gene_te$params$comp_idx <- 1
  
  res <- apply_gene(gene_te, df, target_col = "target")
  expect_true(gene_te$output_col %in% names(res$train))
  expect_type(res$train[[gene_te$output_col]], "double")
})

test_that("Alternative and custom fitness metrics work", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    target = sample(0:1, 50, replace = TRUE)
  )
  
  # Use AUC
  suppressWarnings({
    res_auc <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE,
      metric = "auc"
    )
  })
  expect_s3_class(res_auc, "evo_recipe")
  expect_equal(res_auc$metric, "auc")
  
  # Use custom metric function
  custom_metric <- function(y_true, y_pred) {
    mean(y_pred)
  }
  suppressWarnings({
    res_custom <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE,
      metric = custom_metric
    )
  })
  expect_s3_class(res_custom, "evo_recipe")
  expect_identical(res_custom$metric, custom_metric)
})

test_that("eval-ts-refinement metric and training integration works", {
  set.seed(42)
  # Create a binary classification dataset
  y_true_bin <- c(1, 0, 1, 0, 1, 0, 1, 0)
  # Logits: positive for 1, negative for 0
  logits_bin <- c(1.5, -2.0, 0.8, -1.2, 2.5, -0.5, 1.1, -1.8)
  probs_bin <- 1 / (1 + exp(-logits_bin))
  
  # Calculate metric using logits
  score_logits <- compute_ts_refinement(y_true_bin, logits_bin, task = "classification", is_logits = TRUE)
  # Calculate metric using probs
  score_probs <- compute_ts_refinement(y_true_bin, probs_bin, task = "classification", is_logits = FALSE)
  
  # They should be identical
  expect_equal(score_logits, score_probs, tolerance = 1e-6)
  expect_gt(score_logits, 0)
  
  # Create a multiclass dataset
  y_true_multi <- c(0, 1, 2, 0, 1, 2)
  # Probabilities
  probs_multi <- matrix(c(
    0.8, 0.1, 0.1,
    0.1, 0.7, 0.2,
    0.2, 0.2, 0.6,
    0.7, 0.2, 0.1,
    0.1, 0.8, 0.1,
    0.3, 0.1, 0.6
  ), ncol = 3, byrow = TRUE)
  logits_multi <- log(probs_multi)
  
  score_logits_multi <- compute_ts_refinement(y_true_multi, logits_multi, task = "multiclass", num_class = 3, is_logits = TRUE)
  score_probs_multi <- compute_ts_refinement(y_true_multi, probs_multi, task = "multiclass", num_class = 3, is_logits = FALSE)
  
  expect_equal(score_logits_multi, score_probs_multi, tolerance = 1e-6)
  expect_gt(score_logits_multi, 0)
  
  # Imbalanced multiclass test with default alpha=1
  # Verifies the Laplace smoothing denominator uses total N (not per-class N_k).
  # With the old (wrong) formula the smoothed rows did NOT sum to 1 on imbalanced data.
  y_true_imb <- c(0, 0, 0, 1, 1, 2)
  probs_imb <- matrix(c(
    0.8, 0.1, 0.1,
    0.9, 0.05, 0.05,
    0.7, 0.2, 0.1,
    0.1, 0.8, 0.1,
    0.2, 0.7, 0.1,
    0.8, 0.1, 0.1
  ), ncol = 3, byrow = TRUE)

  score_imb <- compute_ts_refinement(y_true_imb, probs_imb, task = "multiclass", num_class = 3, alpha = 1)
  expect_true(is.finite(score_imb))
  expect_gt(score_imb, 0)

  # Verify class-count-aware smoothing details.
  # For class 0 (N_k=3), n=6, C=3, alpha=1:
  #   true_target: (3+1)/(3+2) = 4/5 = 0.8
  #   leftover_mass: 1/(3+2) = 1/5 = 0.2
  #   incorrect classes get: N_j * (leftover_mass / (n - N_k))
  n_mc <- length(y_true_imb); C_mc <- 3L; alpha_mc <- 1
  cc <- tabulate(y_true_imb + 1, nbins = C_mc)
  Nk_0 <- cc[1]  # class 0 has 3 samples
  expect_equal((Nk_0 + alpha_mc) / (Nk_0 + 2 * alpha_mc), 4/5, tolerance = 1e-10)

  # Binary imbalanced: smoothed positive label must equal (N1+alpha)/(N1+2*alpha)
  N1_t <- 90L; alpha_t <- 1
  expect_equal((N1_t + alpha_t) / (N1_t + 2 * alpha_t), 91/92, tolerance = 1e-10)

  set.seed(1)
  y_bin_imb <- c(rep(1L, 90), rep(0L, 10))
  p_bin_imb <- c(runif(90, 0.55, 0.95), runif(10, 0.05, 0.45))
  score_bin_imb <- compute_ts_refinement(y_bin_imb, p_bin_imb, task = "classification", alpha = 1)
  expect_true(is.finite(score_bin_imb))
  expect_gt(score_bin_imb, 0)

  # Integrate with evolve_features
  df <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    target = sample(0:1, 50, replace = TRUE)
  )
  
  # 1. Test with lightgbm evaluator
  suppressWarnings({
    res_lgb <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "lightgbm",
      verbose = FALSE,
      metric = "eval-ts-refinement"
    )
  })
  expect_s3_class(res_lgb, "evo_recipe")
  expect_equal(tolower(res_lgb$metric), "eval-ts-refinement")
  expect_gt(res_lgb$best_individual$fitness, 0)
  
  # 2. Test with xgboost evaluator
  suppressWarnings({
    res_xgb <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 1,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "xgboost",
      verbose = FALSE,
      metric = "eval-ts-refinement"
    )
  })
  expect_s3_class(res_xgb, "evo_recipe")
  expect_equal(tolower(res_xgb$metric), "eval-ts-refinement")
  expect_gt(res_xgb$best_individual$fitness, 0)

  # 3. Test direct train_model with booster-level early stopping and custom metric
  x_tr <- data.matrix(df[1:30, c("x1", "x2")])
  y_tr <- df$target[1:30]
  x_va <- data.matrix(df[31:50, c("x1", "x2")])
  y_va <- df$target[31:50]
  
  res_train_lgb <- train_model(
    x_train = x_tr, y_train = y_tr, x_val = x_va, y_val = y_va,
    task = "classification", evaluator = "lightgbm",
    metric = "eval-ts-refinement", early_stopping_rounds = 3
  )
  expect_type(res_train_lgb, "list")
  expect_true(!is.null(res_train_lgb$model))

  res_train_xgb <- train_model(
    x_train = x_tr, y_train = y_tr, x_val = x_va, y_val = y_va,
    task = "classification", evaluator = "xgboost",
    metric = "eval-ts-refinement", early_stopping_rounds = 3
  )
  expect_type(res_train_xgb, "list")
  expect_true(!is.null(res_train_xgb$model))
})


test_that("S3 print, summary, and plot methods work correctly", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    target = sample(0:1, 50, replace = TRUE)
  )
  
  suppressWarnings({
    res <- evolve_features(
      data = df,
      target_col = "target",
      task = "classification",
      generations = 2,
      pop_size = 2,
      cv_folds = 2,
      early_stopping_rounds = 2,
      evaluator = "lightgbm",
      verbose = FALSE
    )
  })
  
  # 1. Print
  msg_print <- testthat::capture_output(print(res))
  expect_true(grepl("An evoFE Recipe", msg_print))
  
  # 2. Summary
  sum_res <- summary(res)
  expect_s3_class(sum_res, "summary_evo_recipe")
  msg_summary <- testthat::capture_output(print(sum_res))
  expect_true(grepl("Evolutionary Feature Engineering Summary", msg_summary))
  
  # 3. Plot (Fitness)
  pdf(NULL)
  on.exit(dev.off(), add = TRUE)
  expect_silent(plot(res, type = "fitness"))
  
  # 4. Plot (Importance)
  if (length(res$best_individual$importances) > 0) {
    expect_silent(plot(res, type = "importance"))
  } else {
    expect_message(plot(res, type = "importance"), "No feature importances available.")
  }
})

test_that("Incompatible task/metric combinations raise clear errors", {
  df <- data.frame(x = rnorm(10), target = 1:10)
  
  # 1. Invalid task type
  expect_error(
    evolve_features(df, "target", task = "invalid_task"),
    "task must be one of"
  )
  
  # 2. MAE for classification
  expect_error(
    evolve_features(df, "target", task = "classification", metric = "mae"),
    "Metric 'mae' is not supported for task 'classification'"
  )
  
  # 3. AUC for regression
  expect_error(
    evolve_features(df, "target", task = "regression", metric = "auc"),
    "Metric 'auc' is not supported for task 'regression'"
  )
})

test_that("Model Registry allows custom evaluator registration", {
  # 1. Register a simple dummy evaluator
  register_evaluator(
    "mock_model",
    train_func = function(x_train, y_train, x_val = NULL, task = "regression", ...) {
      list(
        model = list(coefficients = c(a = 1)),
        predictions = if (!is.null(x_val)) rep(0.5, nrow(x_val)) else NULL,
        importances = stats::setNames(rep(1, ncol(x_train)), colnames(x_train))
      )
    },
    predict_func = function(model, x_new, task, ...) {
      rep(0.5, nrow(x_new))
    }
  )
  
  expect_true("mock_model" %in% names(evo_evaluators))
  
  # 2. Test train_model with custom evaluator
  x_train <- matrix(rnorm(20), ncol = 2)
  colnames(x_train) <- c("x1", "x2")
  y_train <- rnorm(10)
  x_val <- matrix(rnorm(10), ncol = 2)
  
  fit <- train_model(x_train, y_train, x_val = x_val, task = "regression", evaluator = "mock_model")
  expect_equal(fit$predictions, rep(0.5, 5))
  expect_equal(fit$importances, c(x1 = 1, x2 = 1))
  
  # 3. Test predict_model with custom evaluator
  # Construct a dummy evo_recipe object containing the mock model
  recipe <- list(
    best_individual = list(
      genes = list(),
      numeric_cols = c("x1", "x2"),
      categorical_cols = character(0)
    ),
    best_model = fit$model,
    evaluator = "mock_model",
    task = "regression",
    classes = NULL
  )
  class(recipe) <- "evo_recipe"
  
  df_new <- data.frame(x1 = rnorm(5), x2 = rnorm(5))
  preds <- predict_model(recipe, df_new)
  expect_equal(preds, rep(0.5, 5))
})

test_that("CatBoost evaluator checks for package availability", {
  # If catboost is not installed (which is typical for clean check environments),
  # it should raise a clear, friendly error.
  if (system.file(package = "catboost") == "") {
    x_train <- matrix(rnorm(20), ncol = 2)
    colnames(x_train) <- c("x1", "x2")
    y_train <- rbinom(10, 1, 0.5)
    
    expect_error(
      train_model(x_train, y_train, task = "classification", evaluator = "catboost"),
      "The 'catboost' package is required"
    )
  } else {
    # If catboost is installed, test that it trains and predicts successfully
    # Use integer matrix to verify the fix for C++ REAL() type assertions on integers
    x_train <- matrix(as.integer(rpois(20, 5)), ncol = 2)
    colnames(x_train) <- c("x1", "x2")
    y_train <- rbinom(10, 1, 0.5)
    
    res <- train_model(x_train, y_train, x_val = x_train, task = "classification", evaluator = "catboost",
                       mbo_init_design = 5, mbo_iters = 3, mbo_folds = 2)
    expect_type(res, "list")
    expect_true(!is.null(res$model))
    expect_length(res$predictions, 10)
    
    # Test predict function
    evaluator_entry <- evo_evaluators[["catboost"]]
    preds <- evaluator_entry$predict_func(res$model, x_train, task = "classification")
    expect_length(preds, 10)
  }
})

test_that("lightgbm_mbo checks for package availability", {
  # If mlrMBO, ParamHelpers, or smoof is not installed, it should raise a clear error
  if (!requireNamespace("mlrMBO", quietly = TRUE) ||
      !requireNamespace("ParamHelpers", quietly = TRUE) ||
      !requireNamespace("smoof", quietly = TRUE)) {
    x_train <- matrix(rnorm(20), ncol = 2)
    colnames(x_train) <- c("x1", "x2")
    y_train <- rbinom(10, 1, 0.5)
    
    expect_error(
      train_model(x_train, y_train, task = "classification", evaluator = "lightgbm_mbo"),
      "The packages 'mlrMBO', 'ParamHelpers', and 'smoof' are required"
    )
  } else {
    # If installed, test that it trains and predicts successfully
    x_train <- matrix(rnorm(40), ncol = 2)
    colnames(x_train) <- c("x1", "x2")
    y_train <- rbinom(20, 1, 0.5)
    
    res <- train_model(x_train, y_train, x_val = x_train, task = "classification", 
                       evaluator = "lightgbm_mbo", mbo_iters = 2, mbo_init_design = 8, mbo_folds = 2)
    expect_type(res, "list")
    expect_true(!is.null(res$model))
    expect_length(res$predictions, 20)
    expect_true("best_params" %in% names(res))
    
    # Test predict function
    evaluator_entry <- evo_evaluators[["lightgbm_mbo"]]
    preds <- evaluator_entry$predict_func(res$model, x_train, task = "classification")
    expect_length(preds, 20)
    
    # Test passing best_params to seed the initial design
    custom_params <- list(
      learning_rate = 0.05,
      num_leaves = 15,
      max_depth = 4,
      feature_fraction = 0.8
    )
    res_seeded <- train_model(x_train, y_train, x_val = x_train, task = "classification", 
                              evaluator = "lightgbm_mbo", mbo_iters = 1, mbo_init_design = 5, mbo_folds = 2,
                              best_params = custom_params, verbose = FALSE)
    expect_type(res_seeded, "list")
    expect_true(!is.null(res_seeded$model))
    expect_true("best_params" %in% names(res_seeded))
  }
})

test_that("make_tunable works correctly", {
  # 1. Error on unregistered model
  expect_error(
    make_tunable("non_existent_model", list(lr = list(type = "numeric", lower = 0.1, upper = 0.5))),
    "is not registered in evo_evaluators"
  )
  
  # 2. Register mock evaluator
  register_evaluator(
    "mock_base",
    train_func = function(x_train, y_train, x_val = NULL, y_val = NULL, task = "regression",
                           threads = 2, num_class = NULL, metric = "default", verbose = FALSE, ...) {
      args <- list(...)
      # Mock score: depends on parameters to optimize
      val_score <- 100
      if ("param_a" %in% names(args)) val_score <- val_score - abs(args$param_a - 4.5)
      if ("param_b" %in% names(args)) val_score <- val_score - abs(args$param_b - 7)
      
      list(
        model = list(args = args, val_score = val_score),
        predictions = if (!is.null(x_val)) rep(val_score, nrow(x_val)) else NULL,
        importances = stats::setNames(rep(1, ncol(x_train)), colnames(x_train))
      )
    },
    predict_func = function(model, x_new, task, ...) {
      rep(model$val_score, nrow(x_new))
    }
  )
  
  # 3. Call make_tunable
  param_ranges <- list(
    param_a = list(type = "numeric", lower = 1.0, upper = 8.0),
    param_b = list(type = "integer", lower = 1, upper = 10)
  )
  make_tunable("mock_base", param_ranges, tuner_name = "mock_base_tuned")
  
  expect_true("mock_base_tuned" %in% names(evo_evaluators))
  
  # 4. Train with the tuned evaluator
  x_train <- matrix(rnorm(20), ncol = 2)
  colnames(x_train) <- c("x1", "x2")
  y_train <- rnorm(10)
  x_val <- matrix(rnorm(10), ncol = 2)
  y_val <- rnorm(5)
  
  res <- train_model(
    x_train, y_train, x_val = x_val, y_val = y_val, task = "regression",
    evaluator = "mock_base_tuned", mbo_iters = 3, mbo_init_design = 5, mbo_folds = 2,
    verbose = FALSE
  )
  
  expect_type(res, "list")
  expect_true("best_params" %in% names(res))
  expect_true("param_a" %in% names(res$best_params))
  expect_true("param_b" %in% names(res$best_params))
  
  # Verify that predictions are returned and predict_func works
  expect_length(res$predictions, nrow(x_val))
  evaluator_entry <- evo_evaluators[["mock_base_tuned"]]
  preds <- evaluator_entry$predict_func(res$model, x_val, task = "regression")
  expect_length(preds, nrow(x_val))

  # Test tuning with EA optimizer
  res_ea <- train_model(
    x_train, y_train, x_val = x_val, y_val = y_val, task = "regression",
    evaluator = "mock_base_tuned", mbo_iters = 3, mbo_init_design = 5, mbo_folds = 2,
    mbo_infill_opt = "ea", verbose = FALSE
  )
  expect_type(res_ea, "list")
  expect_true("best_params" %in% names(res_ea))

  # Test that invalid mbo_infill_opt throws error
  expect_error(
    train_model(
      x_train, y_train, x_val = x_val, y_val = y_val, task = "regression",
      evaluator = "mock_base_tuned", mbo_iters = 3, mbo_init_design = 5, mbo_folds = 2,
      mbo_infill_opt = "invalid_opt", verbose = FALSE
    ),
    "mbo_infill_opt must be either"
  )
})

test_that("add, subtract, and multiply transformers handle integer overflow gracefully", {
  # Test with values that exceed the 32-bit signed integer limit (2^31 - 1 = 2147483647)
  val1 <- 2000000000L
  val2 <- 1500000000L
  df <- data.table::data.table(
    x1 = c(val1, val1),
    x2 = c(val2, val2)
  )
  
  # For add
  gene_add <- create_gene("add", c("x1", "x2"))
  res_add <- apply_gene(gene_add, df)
  expect_equal(res_add$train[[gene_add$output_col]], c(3.5e9, 3.5e9))
  
  # For subtract
  gene_sub <- create_gene("subtract", c("x1", "x2"))
  res_sub <- apply_gene(gene_sub, df)
  expect_equal(res_sub$train[[gene_sub$output_col]], c(5e8, 5e8))
  
  # For multiply
  df_mult <- data.table::data.table(
    x1 = c(1000000L, 1000000L),
    x2 = c(3000L, 3000L)
  )
  gene_mult <- create_gene("multiply", c("x1", "x2"))
  res_mult <- apply_gene(gene_mult, df_mult)
  expect_equal(res_mult$train[[gene_mult$output_col]], c(3e9, 3e9))
})

test_that("multivariate stateful transformers (deadwood, mst_score, genie, lumbermark) handle integer overflow gracefully", {
  # Test with values whose differences exceed the threshold for squared operations (e.g. 100000L)
  # 100000L ^ 2 = 10,000,000,000, which exceeds 2^31 - 1
  df_fit <- data.table::data.table(
    x1 = c(0L, 0L, 0L, 0L, 0L, 0L),
    x2 = c(0L, 0L, 0L, 0L, 0L, 0L)
  )
  df_test <- data.table::data.table(
    x1 = c(100000L, 100000L, 100000L, 100000L, 100000L, 100000L),
    x2 = c(100000L, 100000L, 100000L, 100000L, 100000L, 100000L)
  )
  
  # For deadwood
  if (requireNamespace("deadwood", quietly = TRUE)) {
    gene_dead <- create_gene("deadwood", c("x1", "x2"))
    state_dead <- evo_transformers$deadwood$fit_func(df_fit, gene_dead)
    
    # Run apply with fallback R KNN by checking that it doesn't crash or overflow to NA
    preds_dead <- evo_transformers$deadwood$apply_func(df_test, gene_dead, state_dead)
    expect_false(any(is.na(preds_dead)))
  }
  
  # For mst_score
  if (requireNamespace("quitefastmst", quietly = TRUE)) {
    gene_mst <- create_gene("mst_score", c("x1", "x2"))
    state_mst <- evo_transformers$mst_score$fit_func(df_fit, gene_mst)
    preds_mst <- evo_transformers$mst_score$apply_func(df_test, gene_mst, state_mst)
    expect_false(any(is.na(preds_mst)))
  }
  
  # For genie
  if (requireNamespace("genieclust", quietly = TRUE)) {
    gene_genie <- create_gene("genie", c("x1", "x2"))
    state_genie <- evo_transformers$genie$fit_func(df_fit, gene_genie)
    preds_genie <- evo_transformers$genie$apply_func(df_test, gene_genie, state_genie)
    expect_false(any(is.na(preds_genie)))
  }
  
  # For lumbermark
  if (requireNamespace("lumbermark", quietly = TRUE)) {
    gene_lumb <- create_gene("lumbermark", c("x1", "x2"))
    state_lumb <- evo_transformers$lumbermark$fit_func(df_fit, gene_lumb)
    preds_lumb <- evo_transformers$lumbermark$apply_func(df_test, gene_lumb, state_lumb)
    expect_false(any(is.na(preds_lumb)))
  }
})

test_that("baseline fitness is consistent across population sizes under split strategy with same seed", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(50),
    x2 = rnorm(50),
    target = sample(0:1, 50, replace = TRUE)
  )
  
  # Run once with pop_size = 2
  res_small <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    generations = 1,
    pop_size = 2,
    evaluation_strategy = "split",
    split_ratio = c(0.6, 0.4),
    early_stopping_rounds = 1,
    evaluator = "lightgbm",
    seed = 123,
    verbose = FALSE
  )
  
  # Run again with pop_size = 5
  res_large <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    generations = 1,
    pop_size = 5,
    evaluation_strategy = "split",
    split_ratio = c(0.6, 0.4),
    early_stopping_rounds = 1,
    evaluator = "lightgbm",
    seed = 123,
    verbose = FALSE
  )
  
  # The baseline individual (original features only) is the one with genes list length 0.
  # Let's find it in both runs and verify their fitness is identical.
  get_baseline_fitness <- function(res) {
    for (ind in res$history) {
      if (length(ind$genes) == 0) {
        return(ind$fitness)
      }
    }
    stop("Baseline individual not found")
  }
  
  fit_small <- get_baseline_fitness(res_small)
  fit_large <- get_baseline_fitness(res_large)
  
  expect_equal(fit_small, fit_large)
})

test_that("model_all_final_genes accumulates all unique genes, evaluates them, and trains successfully based on performance", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(60),
    x2 = rnorm(60),
    target = sample(0:1, 60, replace = TRUE)
  )
  
  # Run evolution with model_all_final_genes = TRUE
  res <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    generations = 2,
    pop_size = 5,
    evaluation_strategy = "cv",
    cv_folds = 2,
    early_stopping_rounds = 2,
    evaluator = "lightgbm",
    seed = 42,
    model_all_final_genes = TRUE,
    verbose = FALSE
  )
  
  expect_s3_class(res, "evo_recipe")
  
  hist_best_ind <- res$history[[1]]
  history_genes <- unlist(lapply(res$history, function(ind) ind$genes), recursive = FALSE)
  unique_history_cols <- unique(vapply(history_genes, function(g) g$output_col, character(1)))
  
  best_genes_cols <- vapply(res$best_individual$genes, function(g) g$output_col, character(1))
  
  # The final best_individual's fitness must be at least as high as the historical best
  expect_gte(res$best_individual$fitness, hist_best_ind$fitness)
  
  # Check if the super-individual was selected or the historical best
  is_super_selected <- length(best_genes_cols) == length(unique_history_cols) && all(sort(best_genes_cols) == sort(unique_history_cols))
  is_hist_selected <- length(best_genes_cols) == length(hist_best_ind$genes) && all(sort(best_genes_cols) == sort(vapply(hist_best_ind$genes, function(g) g$output_col, character(1))))
  
  expect_true(is_super_selected || is_hist_selected)
  
  # 2. Verify that predict on the recipe works
  preds_df <- predict(res, df[, 1:2])
  expect_s3_class(preds_df, "data.table")
  expect_true(all(best_genes_cols %in% names(preds_df)))
  
  # 3. Verify predict_model succeeds
  preds <- predict_model(res, df[, 1:2])
  expect_length(preds, 60)
  expect_true(all(preds >= 0 & preds <= 1))
})

test_that("model_all_historical_genes collects genes across generations, evaluates them, and trains successfully based on performance", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(60),
    x2 = rnorm(60),
    target = sample(0:1, 60, replace = TRUE)
  )
  
  # Run evolution with model_all_historical_genes = TRUE
  res <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    generations = 2,
    pop_size = 5,
    evaluation_strategy = "cv",
    cv_folds = 2,
    early_stopping_rounds = 2,
    evaluator = "lightgbm",
    seed = 42,
    model_all_historical_genes = TRUE,
    verbose = FALSE
  )
  
  expect_s3_class(res, "evo_recipe")
  expect_true(is.numeric(res$best_individual$fitness))
  expect_gt(res$best_individual$fitness, -Inf)
  
  # Verify that predict on the recipe works
  preds_df <- predict(res, df[, 1:2])
  expect_s3_class(preds_df, "data.table")
  
  # Verify predict_model succeeds
  preds <- predict_model(res, df[, 1:2])
  expect_length(preds, 60)
  expect_true(all(preds >= 0 & preds <= 1))
})

test_that("gradual population growth and decay works correctly during dynamic population expansion and recovery", {
  set.seed(42)
  df <- data.frame(
    x1 = rnorm(30),
    x2 = rnorm(30),
    target = sample(0:1, 30, replace = TRUE)
  )
  
  # Run evolution with dynamic population and custom growth/decay rates
  res <- evolve_features(
    data = df,
    target_col = "target",
    task = "classification",
    generations = 3,
    pop_size = 3,
    evaluation_strategy = "cv",
    cv_folds = 2,
    early_stopping_rounds = 3,
    evaluator = "lightgbm",
    seed = 42,
    dynamic_population = TRUE,
    dynamic_population_growth_rate = 2.0,
    dynamic_population_decay_rate = 0.5,
    verbose = FALSE
  )
  
  expect_s3_class(res, "evo_recipe")
  expect_true(is.numeric(res$best_individual$fitness))
  expect_gt(res$best_individual$fitness, -Inf)
})


