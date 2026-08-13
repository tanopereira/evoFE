test_that(".calc_feature_distance: both NULL returns 0 (identical absence)", {
  expect_equal(.calc_feature_distance(NULL, NULL), 0.0)
})

test_that(".calc_feature_distance: one NULL returns 1 (maximum distance)", {
  ind <- list(numeric_cols = "a", categorical_cols = character(0), datetime_cols = character(0), genes = list())
  expect_equal(.calc_feature_distance(ind, NULL), 1.0)
  expect_equal(.calc_feature_distance(NULL, ind), 1.0)
})

test_that(".calc_feature_distance: identical individuals return 0", {
  ind <- list(
    numeric_cols = c("a", "b"),
    categorical_cols = "c",
    datetime_cols = character(0),
    genes = list()
  )
  expect_equal(.calc_feature_distance(ind, ind), 0.0)
})

test_that(".calc_feature_distance: fully disjoint individuals return 1", {
  ind_j <- list(numeric_cols = "a", categorical_cols = character(0), datetime_cols = character(0), genes = list())
  ind_k <- list(numeric_cols = "b", categorical_cols = character(0), datetime_cols = character(0), genes = list())
  expect_equal(.calc_feature_distance(ind_j, ind_k), 1.0)
})

test_that(".calc_feature_distance: partial overlap returns correct Jaccard distance", {
  ind_j <- list(numeric_cols = c("a", "b"), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  ind_k <- list(numeric_cols = c("b", "c"), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  # intersection = {b}, union = {a, b, c} -> Jaccard distance = 1 - 1/3
  expect_equal(.calc_feature_distance(ind_j, ind_k), 1 - 1 / 3)
})

test_that(".calc_feature_distance: two baseline individuals (no features, no genes) return 0", {
  ind_j <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  ind_k <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  expect_equal(.calc_feature_distance(ind_j, ind_k), 0.0)
})

test_that(".calc_feature_distance: NA values in column fields are ignored", {
  ind_j <- list(numeric_cols = c("a", NA), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  ind_k <- list(numeric_cols = c("a", NA), categorical_cols = character(0), datetime_cols = character(0), genes = list())
  # NA dropped from both; signature = {a} for both -> distance = 0
  expect_equal(.calc_feature_distance(ind_j, ind_k), 0.0)
})

test_that(".calc_feature_distance: gene formulas contribute to the signature", {
  # Minimal valid gene structure: transformer_name + input_cols + params (empty -> falls through to sprintf("%s(%s)"))
  gene_a <- list(transformer_name = "log1p", input_cols = "x", params = list())
  gene_b <- list(transformer_name = "sqrt",  input_cols = "x", params = list())

  ind_j <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                genes = list(gene_a))
  ind_k <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                genes = list(gene_b))

  dist <- .calc_feature_distance(ind_j, ind_k)
  # Genes are distinct formulas -> distance should be 1
  expect_equal(dist, 1.0)
})

test_that(".calc_feature_distance: shared gene formula reduces distance", {
  gene_shared <- list(transformer_name = "log1p", input_cols = "x", params = list())
  gene_extra  <- list(transformer_name = "sqrt",  input_cols = "x", params = list())

  ind_j <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                genes = list(gene_shared))
  ind_k <- list(numeric_cols = character(0), categorical_cols = character(0), datetime_cols = character(0),
                genes = list(gene_shared, gene_extra))

  dist <- .calc_feature_distance(ind_j, ind_k)
  # intersection = {shared}, union = {shared, extra} -> distance = 1 - 1/2
  expect_equal(dist, 0.5)
})

test_that(".calc_feature_distance: result is always in [0, 1]", {
  make_ind <- function(cols) {
    list(numeric_cols = cols, categorical_cols = character(0), datetime_cols = character(0), genes = list())
  }
  pairs <- list(
    list(make_ind(character(0)), make_ind(character(0))),
    list(make_ind("a"), make_ind("a")),
    list(make_ind("a"), make_ind("b")),
    list(make_ind(c("a", "b")), make_ind(c("b", "c")))
  )
  for (p in pairs) {
    d <- .calc_feature_distance(p[[1]], p[[2]])
    expect_gte(d, 0.0)
    expect_lte(d, 1.0)
  }
})
