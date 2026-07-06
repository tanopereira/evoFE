test_that("create_individual stores all original columns and defaults them", {
  ind1 <- create_individual(numeric_cols = c("a", "b"), categorical_cols = "c")
  expect_equal(ind1$numeric_cols, c("a", "b"))
  expect_equal(ind1$all_numeric_cols, c("a", "b"))
  expect_equal(ind1$categorical_cols, "c")
  expect_equal(ind1$all_categorical_cols, "c")
  
  ind2 <- create_individual(
    numeric_cols = "a",
    categorical_cols = character(0),
    all_numeric_cols = c("a", "b"),
    all_categorical_cols = "c"
  )
  expect_equal(ind2$numeric_cols, "a")
  expect_equal(ind2$all_numeric_cols, c("a", "b"))
  expect_equal(ind2$all_categorical_cols, "c")
})

test_that("toggle_raw_feature toggles features correctly and respects safety limits", {
  set.seed(42)
  # Case 1: Safety limit. Try to deactivate below min_active = 2 (when total avail = 2).
  ind <- create_individual(numeric_cols = c("a", "b"))
  expect_equal(length(ind$numeric_cols), 2)
  ind_mut <- toggle_raw_feature(ind, verbose = FALSE)
  expect_equal(length(ind_mut$numeric_cols), 2) # should not change because min_active = 2!
  
  # Case 2: Deactivating when we have multiple features active above min_active
  ind2 <- create_individual(numeric_cols = c("a", "b", "c"))
  expect_equal(length(ind2$numeric_cols), 3)
  ind_mut2 <- toggle_raw_feature(ind2, verbose = FALSE)
  expect_equal(length(ind_mut2$numeric_cols), 2)
  expect_true(is.na(ind_mut2$fitness))
  
  # Case 3: Activating when we have inactive features
  ind3 <- create_individual(numeric_cols = c("a", "b"), all_numeric_cols = c("a", "b", "c"))
  # Since total_active is 2 (min_active = 2), we can't deactivate. We must activate "c" first.
  # If k > 1, it might subsequently deactivate one of "a" or "b".
  ind_mut3 <- toggle_raw_feature(ind3, verbose = FALSE)
  expect_true("c" %in% ind_mut3$numeric_cols)
  expect_gte(length(ind_mut3$numeric_cols), 2)
})

test_that("toggle_raw_feature respects temperature-scaled feature importance", {
  # We have three active features. min_active = 2, so we can deactivate one feature.
  # We should be much more likely to deactivate "c" (zero importance).
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b", "c"))
  imps <- c(a = 100.0, b = 100.0, c = 0.0)
  
  deactivated_cols <- character(0)
  for (i in 1:100) {
    res <- toggle_raw_feature(ind, importances = imps, temperature = 1.0)
    deactivated <- setdiff(c("a", "b", "c"), res$numeric_cols)
    deactivated_cols <- c(deactivated_cols, deactivated)
  }
  
  expect_gt(sum(deactivated_cols == "c"), 95)
})

test_that("mutate toggles raw features when raw_toggle_prob = 1.0", {
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b", "c"))
  ind_mut <- mutate(ind, raw_toggle_prob = 1.0)
  expect_equal(length(ind_mut$numeric_cols), 2)
})

test_that("crossover and union_crossover on feature masks work", {
  set.seed(42)
  ind1 <- create_individual(numeric_cols = c("a", "b"), all_numeric_cols = c("a", "b", "c"))
  ind2 <- create_individual(numeric_cols = c("b", "c"), all_numeric_cols = c("a", "b", "c"))
  
  child <- crossover(ind1, ind2)
  expect_equal(child$all_numeric_cols, c("a", "b", "c"))
  expect_gte(length(child$numeric_cols), 2) # enforced by min_active safety checks
  
  child_union <- union_crossover(ind1, ind2)
  expect_equal(sort(child_union$numeric_cols), c("a", "b", "c"))
})

test_that("sample_active_mask works with and without importances", {
  set.seed(42)
  cols <- c("x1", "x2", "x3")
  
  # Case 1: No importances -> fallback uniform rate
  mask_no_imps <- sample_active_mask(cols, importances = NULL, temperature = 0.5)
  expect_gt(length(mask_no_imps), 0)
  
  # Case 2: Under extreme importances
  imps <- c(x1 = 0.99, x2 = 0.01, x3 = 0.0)
  
  keeps_x1 <- 0
  keeps_x3 <- 0
  for (i in 1:100) {
    res <- sample_active_mask(cols, importances = imps, temperature = 0.1)
    if ("x1" %in% res) keeps_x1 <- keeps_x1 + 1
    if ("x3" %in% res) keeps_x3 <- keeps_x3 + 1
  }
  
  expect_gt(keeps_x1, 95)
  expect_lt(keeps_x3, 15)
})

test_that("initialize_population creates baseline and importance-guided populations", {
  set.seed(42)
  cols <- c("x1", "x2", "x3", "x4")
  imps <- c(x1 = 0.5, x2 = 0.5, x3 = 0.0, x4 = 0.0)
  
  # 1. Verify individual 1 (baseline) is indeed all active
  pop <- initialize_population(
    pop_size = 2,
    numeric_cols = cols,
    categorical_cols = character(0),
    initial_genes = 0,
    importances = imps,
    mask_temp_factor = 0.5
  )
  expect_equal(pop[[1]]$numeric_cols, cols)
  
  # 2. Run statistical check on mask initialization over 100 trials
  x1_active_count <- 0
  x3_active_count <- 0
  
  for (i in 1:100) {
    pop_trial <- initialize_population(
      pop_size = 2,
      numeric_cols = cols,
      categorical_cols = character(0),
      initial_genes = 0,
      importances = imps,
      mask_temp_factor = 0.5
    )
    active <- pop_trial[[2]]$numeric_cols
    if ("x1" %in% active) x1_active_count <- x1_active_count + 1
    if ("x3" %in% active) x3_active_count <- x3_active_count + 1
  }
  
  # x1 (high importance): expected ~88% active
  expect_gt(x1_active_count, 70)
  # x3 (zero importance): expected ~12% active (or slightly higher due to safety check fallback, but always < 35%)
  expect_lt(x3_active_count, 35)
})

test_that("sample_gene_inputs works correctly and respects temperature-scaled importances", {
  set.seed(42)
  cols <- c("x1", "x2", "x3")
  imps <- c(x1 = 100.0, x2 = 0.0, x3 = 0.0)
  
  # When sampling 1 input under extreme temperature-scaled weights, x1 should be selected almost always
  sampled <- character(0)
  for (i in 1:100) {
    res <- sample_gene_inputs(cols, n = 1, importances = imps, temperature = 1.0)
    sampled <- c(sampled, res)
  }
  
  expect_gt(sum(sampled == "x1"), 95)
})

test_that("recalculate_mask recalculates active mask and resets fitness if changed", {
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b", "c"))
  ind$fitness <- 0.8
  
  # Case 1: Extreme importances (a = 1.0, b = 1.0, c = 0.0)
  # threshold = 1/3 = 0.33
  # temp = 0.1
  # a and b should remain active, c should become inactive
  imps <- c(a = 1.0, b = 1.0, c = 0.0)
  ind_mut <- recalculate_mask(ind, importances = imps, temperature = 0.1, verbose = FALSE)
  
  expect_equal(sort(ind_mut$numeric_cols), c("a", "b"))
  expect_true(is.na(ind_mut$fitness))
  
  # Case 2: Safety limit. Even if all importances are zero, recalculate_mask must keep min_active active (2 here)
  ind$fitness <- 0.8
  imps_zero <- c(a = 0.0, b = 0.0, c = 0.0)
  ind_mut2 <- recalculate_mask(ind, importances = imps_zero, temperature = 0.1, verbose = FALSE)
  expect_equal(ind_mut2$fitness, 0.8)
})

test_that("mutate triggers recalculate_mask under high probability", {
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b", "c"))
  ind$fitness <- 0.8
  imps <- c(a = 1.0, b = 1.0, c = 0.0)
  
  ind_mut <- mutate(ind, recalculate_mask_prob = 1.0, raw_toggle_prob = 0.0, importances = imps, temperature = 0.1)
  expect_equal(sort(ind_mut$numeric_cols), c("a", "b"))
  expect_true(is.na(ind_mut$fitness))
})

test_that("toggle_raw_feature supports multi-column toggling via geometric distribution", {
  set.seed(42)
  
  cols <- paste0("col", 1:50)
  ind <- create_individual(numeric_cols = cols)
  
  deactivated_counts <- numeric(0)
  for (i in 1:20) {
    ind_mut <- toggle_raw_feature(ind, verbose = FALSE)
    deactivated_counts <- c(deactivated_counts, 50 - length(ind_mut$numeric_cols))
  }
  
  expect_gt(max(deactivated_counts), 1)
  
  # Ensure safety floor of min_active (2) is respected
  ind2 <- create_individual(numeric_cols = c("col1", "col2"), all_numeric_cols = cols)
  for (i in 1:20) {
    ind2_mut <- toggle_raw_feature(ind2, verbose = FALSE)
    expect_gte(length(ind2_mut$numeric_cols), 2)
  }
})

test_that("multivariate sampling uses with-replacement and deduplication, concentrating on active columns", {
  set.seed(42)
  cols <- paste0("col", 1:50)
  ind <- create_individual(numeric_cols = c("col1", "col2"), all_numeric_cols = cols)
  
  imps <- rep(0.0, 50)
  names(imps) <- cols
  imps["col1"] <- 100.0
  imps["col2"] <- 100.0
  
  ind_mut <- mutate(ind, force_add = TRUE, allowed_transformers = "pca", importances = imps, temperature = 0.01)
  
  expect_gt(length(ind_mut$genes), 0)
  gene <- ind_mut$genes[[1]]
  expect_equal(gene$transformer_name, "pca")
  expect_equal(length(gene$input_cols), length(unique(gene$input_cols)))
  expect_equal(sort(gene$input_cols), c("col1", "col2"))
})

test_that("apply_individual restores raw features to safety floor if all genes are pruned", {
  set.seed(42)
  cols <- c("a", "b", "c", "d")
  bad_gene1 <- create_gene("pca", c("nonexistent1", "nonexistent2"))
  bad_gene2 <- create_gene("pca", c("nonexistent3", "nonexistent4"))
  
  ind <- create_individual(
    genes = list(bad_gene1, bad_gene2),
    numeric_cols = character(0),
    all_numeric_cols = cols
  )
  
  df <- data.table(a = 1:5, b = 1:5, c = 1:5, d = 1:5)
  
  res <- apply_individual(ind, df, val_data = NULL, target_col = NULL, allow_prune = TRUE)
  
  expect_equal(length(res$ind$genes), 0)
  expect_equal(length(res$ind$numeric_cols), 2)
})
