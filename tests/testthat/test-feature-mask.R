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
  # Case 1: Safety limit. Try to deactivate the last remaining feature.
  ind <- create_individual(numeric_cols = "a")
  expect_equal(length(ind$numeric_cols), 1)
  ind_mut <- toggle_raw_feature(ind, verbose = FALSE)
  expect_equal(length(ind_mut$numeric_cols), 1) # should not change because of safety check!
  
  # Case 2: Deactivating when we have multiple features active
  ind2 <- create_individual(numeric_cols = c("a", "b"))
  ind_mut2 <- toggle_raw_feature(ind2, verbose = FALSE)
  # One of the features must be deactivated (since no inactive features exist to activate)
  expect_equal(length(ind_mut2$numeric_cols), 1)
  expect_true(is.na(ind_mut2$fitness))
  
  # Case 3: Activating when we have inactive features
  ind3 <- create_individual(numeric_cols = "a", all_numeric_cols = c("a", "b"))
  # Since total_active is 1 (only "a"), we can't deactivate "a", so we must activate "b"
  ind_mut3 <- toggle_raw_feature(ind3, verbose = FALSE)
  expect_equal(length(ind_mut3$numeric_cols), 2)
  expect_true("b" %in% ind_mut3$numeric_cols)
})

test_that("toggle_raw_feature respects temperature-scaled feature importance", {
  # We have two active features: "a" (very high importance) and "b" (very low/zero importance).
  # We should be much more likely to deactivate "b".
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b"))
  imps <- c(a = 100.0, b = 0.0)
  
  deactivated_cols <- character(0)
  for (i in 1:100) {
    # toggle_raw_feature will choose deactivate because no inactive features are available
    res <- toggle_raw_feature(ind, importances = imps, temperature = 1.0)
    deactivated <- setdiff(c("a", "b"), res$numeric_cols)
    deactivated_cols <- c(deactivated_cols, deactivated)
  }
  
  # "b" has 0 importance, so weight is exp(0) = 1
  # "a" has 100 importance, so weight is exp(-100) = 3.7e-44
  # "b" should be deactivated in almost all cases.
  expect_gt(sum(deactivated_cols == "b"), 95)
})

test_that("mutate toggles raw features when raw_toggle_prob = 1.0", {
  set.seed(42)
  ind <- create_individual(numeric_cols = c("a", "b"))
  # Mutate with raw_toggle_prob = 1.0 should toggle a raw feature
  ind_mut <- mutate(ind, raw_toggle_prob = 1.0)
  expect_equal(length(ind_mut$numeric_cols), 1)
})

test_that("crossover and union_crossover on feature masks work", {
  set.seed(42)
  ind1 <- create_individual(numeric_cols = "a", all_numeric_cols = c("a", "b"))
  ind2 <- create_individual(numeric_cols = "b", all_numeric_cols = c("a", "b"))
  
  # Crossover: should sample from both parents' masks.
  child <- crossover(ind1, ind2)
  expect_equal(child$all_numeric_cols, c("a", "b"))
  
  # Union crossover: should take union of active columns
  child_union <- union_crossover(ind1, ind2)
  expect_equal(sort(child_union$numeric_cols), c("a", "b"))
  expect_equal(child_union$all_numeric_cols, c("a", "b"))
})
