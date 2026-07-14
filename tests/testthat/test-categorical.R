test_that("Feature Hashing works correctly", {
  library(data.table)
  
  df <- data.table(
    col1 = c("apple", "banana", "cherry", NA, "apple", "banana"),
    target = c(0, 1, 0, 1, 0, 0)
  )
  
  # Initialize a feature_hash gene
  gene <- create_gene("feature_hash", "col1")
  expect_equal(gene$transformer_name, "feature_hash")
  expect_true("num_bins" %in% names(gene$params))
  expect_true("comp_idx" %in% names(gene$params))
  
  # Override params for deterministic check
  gene$params$num_bins <- 4
  gene$params$comp_idx <- 1
  
  # Run apply
  res_col <- evo_transformers$feature_hash$apply_func(df, gene)
  expect_equal(length(res_col), nrow(df))
  expect_true(is.na(res_col[4])) # NA is preserved
  
  # Run for all bins and check that exactly one bin is active for each row
  all_bins <- sapply(1:4, function(b) {
    g_bin <- gene
    g_bin$params$comp_idx <- b
    evo_transformers$feature_hash$apply_func(df, g_bin)
  })
  
  # Each non-NA row must sum to 1 across all 4 bins (indicator behavior)
  for (i in which(!is.na(df$col1))) {
    expect_equal(sum(all_bins[i, ]), 1)
  }
})

test_that("Multiple Correspondence Analysis (MCA) works correctly", {
  library(data.table)
  
  set.seed(123)
  df_train <- data.table(
    a = c("A1", "A2", "A1", "A3", "A2", "A1"),
    b = c("B1", "B2", "B1", "B1", "B2", "B2")
  )
  
  df_val <- data.table(
    a = c("A1", "A4", NA, "A3"),
    b = c("B2", "B2", "B1", NA)
  )
  
  gene <- create_gene("mca", c("a", "b"))
  expect_equal(gene$transformer_name, "mca")
  
  # Fit MCA
  state <- evo_transformers$mca$fit_func(df_train, gene)
  expect_true(state$valid)
  expect_true(is.matrix(state$v))
  expect_equal(length(state$training_levels$a), 3) # A1, A2, A3
  expect_equal(length(state$training_levels$b), 2) # B1, B2
  
  # Apply MCA to training
  res_train <- evo_transformers$mca$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  expect_true(is.numeric(res_train))
  
  # Apply MCA to validation (unseen and missing values)
  res_val <- evo_transformers$mca$apply_func(df_val, gene, state)
  expect_equal(length(res_val), nrow(df_val))
  expect_true(is.numeric(res_val))
  expect_false(any(is.na(res_val))) # NAs and unseen levels map to 0 (neutral projection)
})

test_that("Factor Analysis of Mixed Data (FAMD) works correctly", {
  library(data.table)
  
  df_train <- data.table(
    num1 = c(1.2, 3.4, 1.1, 5.5, -2.1, 0.0),
    num2 = c(10, 20, 15, 30, 25, 12),
    cat1 = c("A1", "A2", "A1", "A3", "A2", "A1"),
    cat2 = c("B1", "B2", "B1", "B1", "B2", "B2")
  )
  
  df_val <- data.table(
    num1 = c(2.0, NA, 4.0, 1.2),
    num2 = c(11, 22, NA, 15),
    cat1 = c("A1", "A4", NA, "A3"),
    cat2 = c("B2", "B2", "B1", NA)
  )
  
  gene <- create_gene("famd", c("num1", "num2", "cat1", "cat2"))
  expect_equal(gene$transformer_name, "famd")
  
  # Fit FAMD
  state <- evo_transformers$famd$fit_func(df_train, gene)
  expect_true(state$valid)
  expect_true(is.matrix(state$v))
  expect_equal(length(state$num_cols), 2)
  expect_equal(length(state$cat_cols), 2)
  
  # Apply FAMD
  res_train <- evo_transformers$famd$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  
  res_val <- evo_transformers$famd$apply_func(df_val, gene, state)
  expect_equal(length(res_val), nrow(df_val))
  expect_false(any(is.na(res_val))) # missing and unseen categories handled
})

test_that("Between-Class PCA works correctly", {
  library(data.table)
  
  df_train <- data.table(
    cat = c("A", "A", "B", "B", "C", "C"),
    num1 = c(1, 1.2, 10, 10.5, 100, 101),
    num2 = c(-5, -5.2, 50, 52, 500, 502)
  )
  
  df_val <- data.table(
    cat = c("A", "D", NA, "C"),
    num1 = c(2.0, 15.0, NA, 99.0),
    num2 = c(-4.5, 48.0, 150.0, NA)
  )
  
  gene <- create_gene("between_group_pca", c("cat", "num1", "num2"))
  expect_equal(gene$transformer_name, "between_group_pca")
  
  # Fit BCA
  state <- evo_transformers$between_group_pca$fit_func(df_train, gene)
  expect_true(state$valid)
  expect_true(is.matrix(state$v))
  expect_equal(ncol(state$v), 2) # num1, num2
  
  # Apply BCA
  res_train <- evo_transformers$between_group_pca$apply_func(df_train, gene, state)
  expect_equal(length(res_train), nrow(df_train))
  
  res_val <- evo_transformers$between_group_pca$apply_func(df_val, gene, state)
  expect_equal(length(res_val), nrow(df_val))
  expect_false(any(is.na(res_val))) # checks NA imputation
})
