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
  expect_type(res_dead$train[[gene_dead$output_col]], "double")
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
  
  # apply_individual should run without error and return the individual with 0 genes (skipped)
  res <- apply_individual(ind, df, target_col = "target")
  expect_equal(length(res$ind$genes), 0)
  expect_false(gene_log$output_col %in% names(res$train))
  
  # evaluate_fitness should run without error and not set fitness to -Inf
  ind_eval <- evaluate_fitness(ind, df, target_col = "target", cv_folds = 2)
  expect_true(is.numeric(ind_eval$fitness))
  expect_gt(ind_eval$fitness, -Inf)
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

test_that("seed parameter produces reproducible results", {
  run_once <- function(s) {
    evolve_features(
      data = iris[, 1:5],
      target_col = "Petal.Length",
      task = "regression",
      generations = 2,
      pop_size = 3,
      cv_folds = 2,
      early_stopping_rounds = 1,
      evaluator = "xgboost",
      seed = s,
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
  
  while ((!pca_added || !umap_added) && attempts < 100) {
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
  # We can verify this by checking that the state object is identical.
  res2 <- apply_gene(gene_pca2, df, target_col = "target", state_cache = state_cache)
  expect_identical(res1$gene$state, res2$gene$state)
  
  # Verify cache has only one entry
  cache_keys <- ls(envir = state_cache)
  expect_equal(length(cache_keys), 1)
})

test_that("umap, genie, and mst_score respect evoFE.verbose option", {
  df <- data.table::as.data.table(data.frame(
    x1 = rnorm(20),
    x2 = rnorm(20),
    target = sample(0:1, 20, replace = TRUE)
  ))
  
  # Set options(evoFE.verbose = 2)
  options(evoFE.verbose = 2)
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

